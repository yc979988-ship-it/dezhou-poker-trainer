"""20 手一轮的玩家画像与自适应训练场景调度。

本模块只使用已经完成牌局的聚合统计和牌后复盘 ``reason_code``。它不会
读取屏幕、监听真实牌局，也不会把短期输赢当作决策质量。第一版使用透明、
可测试的启发式阈值，不声称是完整 GTO 求解器。

设计约束
--------
* 仅在第 20、40、60……手生成画像；其他手数返回 ``None``。
* 最多返回三个漏洞，并同时保留严重度、置信度和样本机会数。
* 没有足够相关机会时明确返回“信号不足”，不硬贴玩家标签。
* 有漏洞时下一轮仍保留至少 55% 的完全随机牌，定向场景合计不超过 45%。
* 场景选择只依赖计划、seed 和手数，同一输入可重复回放。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from poker_trainer.analytics.statistics import MetricName, PositionStatistics
from poker_trainer.engine.cards import Card, parse_cards
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import Position, Seat


PROFILE_INTERVAL = 20
MAX_LEAKS = 3
MIN_EVIDENCE = 3
RANDOM_SCENARIO_ID = "random"
RANDOM_WEIGHT_FLOOR = 0.55


@dataclass(frozen=True, slots=True)
class LeakFinding:
    """一个有证据支持、可用于排训练优先级的玩家漏洞。"""

    code: str
    title_zh: str
    severity: float
    confidence: float
    evidence_n: int
    actual: float
    benchmark: float
    position: Position | None
    reason: str

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("漏洞 code 不能为空")
        if not self.title_zh:
            raise ValueError("漏洞中文标题不能为空")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity 必须在 0 到 1 之间")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 0 到 1 之间")
        if self.evidence_n < 0:
            raise ValueError("evidence_n 不能为负数")
        if not math.isfinite(self.actual) or not math.isfinite(self.benchmark):
            raise ValueError("actual 和 benchmark 必须是有限数值")
        if self.position is not None and not isinstance(self.position, Position):
            object.__setattr__(self, "position", Position(self.position))

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title_zh": self.title_zh,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence_n": self.evidence_n,
            "actual": self.actual,
            "benchmark": self.benchmark,
            "position": self.position.value if self.position else None,
            "reason": self.reason,
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """一个 20 手检查点上的玩家画像。"""

    through_hand_no: int
    sample_size: int
    leaks: tuple[LeakFinding, ...] = ()
    message_zh: str = "信号不足"

    def __post_init__(self) -> None:
        if self.through_hand_no <= 0 or self.through_hand_no % PROFILE_INTERVAL:
            raise ValueError("玩家画像只能生成在正的 20 手检查点")
        if self.sample_size < 0:
            raise ValueError("sample_size 不能为负数")
        leaks = tuple(self.leaks)
        if len(leaks) > MAX_LEAKS:
            raise ValueError("玩家画像最多保留 3 个漏洞")
        object.__setattr__(self, "leaks", leaks)

    @property
    def findings(self) -> tuple[LeakFinding, ...]:
        """``leaks`` 的语义别名，方便界面层按 finding 命名。"""

        return self.leaks

    @property
    def has_signal(self) -> bool:
        return self.message_zh != "信号不足"

    def as_dict(self) -> dict[str, Any]:
        return {
            "through_hand_no": self.through_hand_no,
            "sample_size": self.sample_size,
            "message_zh": self.message_zh,
            "leaks": [leak.as_dict() for leak in self.leaks],
        }

    to_dict = as_dict


# 保持数据库/UI 可能采用的命名都可用。
PlayerProfileSnapshot = PlayerProfile
AdaptiveProfile = PlayerProfile


def _position_rows(
    statistics: Mapping[Position | str, PositionStatistics]
    | Iterable[PositionStatistics],
) -> tuple[PositionStatistics, ...]:
    rows = (
        tuple(statistics.values())
        if isinstance(statistics, Mapping)
        else tuple(statistics)
    )
    if any(not isinstance(row, PositionStatistics) for row in rows):
        raise TypeError("position_statistics 必须由 PositionStatistics 组成")
    seen: set[Position] = set()
    for row in rows:
        if row.position in seen:
            raise ValueError(f"位置统计重复: {row.position.value}")
        seen.add(row.position)
    return rows


ReasonCountInput = (
    Mapping[object, object]
    | Iterable[object]
    | str
    | object
    | None
)


def _add_reason_counts(target: Counter[str], value: ReasonCountInput) -> None:
    """把常见的 reason code 序列/计数字典/复盘对象压平成 Counter。"""

    if value is None:
        return
    if isinstance(value, str):
        target[value] += 1
        return
    if isinstance(value, Mapping):
        if "reason_codes" in value:
            _add_reason_counts(target, value["reason_codes"])
            return
        for raw_code, raw_count in value.items():
            code = getattr(raw_code, "value", raw_code)
            if isinstance(code, str) and isinstance(raw_count, (int, float)):
                if isinstance(raw_count, bool) or raw_count < 0 or int(raw_count) != raw_count:
                    raise ValueError("reason code 计数必须是非负整数")
                target[code] += int(raw_count)
                continue
            # 也接受 {code: {hits: x, opportunities: y}} 的持久化形态。
            if isinstance(code, str) and isinstance(raw_count, Mapping) and (
                "hits" in raw_count or "opportunities" in raw_count
            ):
                hits = raw_count.get("hits", 0)
                opportunities = raw_count.get("opportunities", hits)
                if any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or item < 0
                    or int(item) != item
                    for item in (hits, opportunities)
                ):
                    raise ValueError("reason code hits/opportunities 必须是非负整数")
                target[code] += int(hits)
                target[f"{code}_opportunity"] += int(opportunities)
                continue
            _add_reason_counts(target, raw_count)
        return
    reason_codes = getattr(value, "reason_codes", None)
    if reason_codes is not None:
        _add_reason_counts(target, reason_codes)
        return
    if isinstance(value, Iterable):
        for item in value:
            _add_reason_counts(target, item)
        return
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        target[enum_value] += 1
        return
    raise TypeError("reason_code_counts 必须是计数字典、代码序列或复盘对象")


def _reason_counts(value: ReasonCountInput) -> Counter[str]:
    result: Counter[str] = Counter()
    _add_reason_counts(result, value)
    return result


def _confidence(evidence_n: int) -> float:
    """保守的小样本置信度；12 次相关机会后封顶。"""

    return round(min(1.0, evidence_n / 12.0), 6)


def _severity(actual: float, benchmark: float, evidence_n: int) -> float:
    if actual <= benchmark:
        return 0.0
    possible_excess = max(1e-12, 1.0 - benchmark)
    normalized_excess = min(1.0, (actual - benchmark) / possible_excess)
    # 样本较小时降低优先级，但不把已重复出现的明确错误抹掉。
    value = normalized_excess * (0.65 + 0.35 * _confidence(evidence_n))
    return round(min(1.0, value), 6)


def _rate_finding(
    *,
    code: str,
    title_zh: str,
    hits: int,
    opportunities: int,
    benchmark: float,
    position: Position | None,
    reason_prefix: str,
    min_excess: float = 0.15,
) -> LeakFinding | None:
    if hits < 0 or opportunities < 0 or hits > opportunities:
        raise ValueError(f"{code} 的命中数/机会数不合法")
    if opportunities < MIN_EVIDENCE:
        return None
    actual = hits / opportunities
    if actual < benchmark + min_excess:
        return None
    position_text = f"{position.value} " if position else ""
    reason = (
        f"{position_text}{reason_prefix}：{hits}/{opportunities}，"
        f"实际 {actual:.0%}，参考上限 {benchmark:.0%}。"
    )
    return LeakFinding(
        code=code,
        title_zh=title_zh,
        severity=_severity(actual, benchmark, opportunities),
        confidence=_confidence(opportunities),
        evidence_n=opportunities,
        actual=actual,
        benchmark=benchmark,
        position=position,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class _ReasonRule:
    title_zh: str
    benchmark: float
    opportunity_codes: tuple[str, ...]
    position: Position | None
    reason_prefix: str


_REASON_RULES: dict[str, _ReasonRule] = {
    "weak_top_pair_overcall": _ReasonRule(
        "弱顶对过度跟注",
        0.35,
        ("weak_top_pair_overcall_opportunity", "weak_top_pair_decision"),
        None,
        "弱顶对面对压力仍继续过多",
    ),
    "sb_cold_call": _ReasonRule(
        "小盲冷跟过多",
        0.20,
        ("sb_cold_call_opportunity",),
        Position.SB,
        "面对翻前加注时选择冷跟过多",
    ),
    "threebet_too_small": _ReasonRule(
        "3bet 尺度偏小",
        0.15,
        ("threebet_too_small_opportunity", "threebet_sizing_opportunity"),
        None,
        "3bet 尺度不足的复盘错误偏多",
    ),
    "small_pair_overcontinue": _ReasonRule(
        "小对子未中三条仍继续",
        0.20,
        ("small_pair_overcontinue_opportunity", "small_pair_postflop_decision"),
        None,
        "小对子未改善时继续投入过多",
    ),
    "strong_draw_overfold": _ReasonRule(
        "强听牌过度弃牌",
        0.15,
        ("strong_draw_overfold_opportunity", "strong_draw_decision"),
        None,
        "强听牌在赔率允许时弃牌过多",
    ),
}


class LeakDetector:
    """从按位置统计和可选复盘代码生成 20 手画像。"""

    def __init__(
        self,
        *,
        interval: int = PROFILE_INTERVAL,
        max_leaks: int = MAX_LEAKS,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval 必须为正数")
        if not 1 <= max_leaks <= MAX_LEAKS:
            raise ValueError("max_leaks 必须在 1 到 3 之间")
        self.interval = interval
        self.max_leaks = max_leaks

    def analyze(
        self,
        through_hand_no: int,
        position_statistics: Mapping[Position | str, PositionStatistics]
        | Iterable[PositionStatistics],
        reason_code_counts: ReasonCountInput = None,
    ) -> PlayerProfile | None:
        """在检查点生成画像；未到检查点时返回 ``None``。"""

        if isinstance(through_hand_no, bool) or not isinstance(through_hand_no, int):
            raise TypeError("through_hand_no 必须是整数")
        if through_hand_no < 0:
            raise ValueError("through_hand_no 不能为负数")
        if through_hand_no == 0 or through_hand_no % self.interval:
            return None

        rows = _position_rows(position_statistics)
        sample_size = sum(row.hands for row in rows)
        codes = _reason_counts(reason_code_counts)
        candidates: list[LeakFinding] = []
        evaluable_signals = 0

        # 1) 弱顶对打光/过度跟注：使用复盘注入的顶对机会，不用最终输赢判断。
        top_pair_hits = sum(
            row.metric(MetricName.TOP_PAIR_STACKOFF).hits for row in rows
        )
        top_pair_opportunities = sum(
            row.metric(MetricName.TOP_PAIR_STACKOFF).opportunities for row in rows
        )
        if top_pair_opportunities >= MIN_EVIDENCE:
            evaluable_signals += 1
        finding = _rate_finding(
            code="weak_top_pair_overcall",
            title_zh="弱顶对过度跟注",
            hits=top_pair_hits,
            opportunities=top_pair_opportunities,
            benchmark=0.35,
            position=None,
            reason_prefix="顶对打光/重压继续频率偏高",
        )
        if finding:
            candidates.append(finding)

        # 2) 小盲面对加注的 cold call；严格按 SB 位置拆分。
        sb_row = next((row for row in rows if row.position == Position.SB), None)
        sb_hits = (
            sb_row.metric(MetricName.COLD_CALL).hits if sb_row is not None else 0
        )
        sb_opportunities = (
            sb_row.metric(MetricName.COLD_CALL).opportunities
            if sb_row is not None
            else 0
        )
        if sb_opportunities >= MIN_EVIDENCE:
            evaluable_signals += 1
        finding = _rate_finding(
            code="sb_cold_call",
            title_zh="小盲冷跟过多",
            hits=sb_hits,
            opportunities=sb_opportunities,
            benchmark=0.20,
            position=Position.SB,
            reason_prefix="面对翻前加注时选择冷跟过多",
        )
        if finding:
            candidates.append(finding)

        # 3) 听牌赔率错误提供 strong_draw_overfold 的保守候选；若有精确
        # reason code，下面会以同 code 的更强证据替换它。
        draw_hits = sum(row.metric(MetricName.DRAW_ODDS_ERROR).hits for row in rows)
        draw_opportunities = sum(
            row.metric(MetricName.DRAW_ODDS_ERROR).opportunities for row in rows
        )
        if draw_opportunities >= MIN_EVIDENCE:
            evaluable_signals += 1
        finding = _rate_finding(
            code="strong_draw_overfold",
            title_zh="强听牌过度弃牌",
            hits=draw_hits,
            opportunities=draw_opportunities,
            benchmark=0.20,
            position=None,
            reason_prefix="听牌赔率错误偏多，需在定向场景复核是否弃牌过度",
        )
        if finding:
            candidates.append(finding)

        # 精确复盘 reason code 可覆盖所有五类规则。没有显式分母时，每一次
        # 明确错误本身视作一次已审查机会；仍要求至少重复 3 次才进入画像。
        for code, rule in _REASON_RULES.items():
            hits = codes[code]
            explicit_opportunities = sum(codes[item] for item in rule.opportunity_codes)
            opportunities = explicit_opportunities if explicit_opportunities else hits
            if hits > opportunities:
                raise ValueError(f"{code} 的错误次数不能大于机会数")
            if opportunities >= MIN_EVIDENCE:
                evaluable_signals += 1
            finding = _rate_finding(
                code=code,
                title_zh=rule.title_zh,
                hits=hits,
                opportunities=opportunities,
                benchmark=rule.benchmark,
                position=rule.position,
                reason_prefix=rule.reason_prefix,
            )
            if finding:
                candidates.append(finding)

        # 同一漏洞可能同时来自聚合指标和精确 reason code，保留证据更强者。
        best_by_code: dict[str, LeakFinding] = {}
        for finding in candidates:
            current = best_by_code.get(finding.code)
            key = (
                finding.severity,
                finding.confidence,
                finding.evidence_n,
            )
            current_key = (
                current.severity,
                current.confidence,
                current.evidence_n,
            ) if current else (-1.0, -1.0, -1)
            if key > current_key:
                best_by_code[finding.code] = finding

        ranked = sorted(
            best_by_code.values(),
            key=lambda item: (
                -item.severity,
                -item.confidence,
                -item.evidence_n,
                item.code,
            ),
        )[: self.max_leaks]

        if ranked:
            message = f"发现 {len(ranked)} 个优先漏洞"
        elif evaluable_signals:
            message = "暂未发现明显漏洞"
        else:
            message = "信号不足"
        return PlayerProfile(
            through_hand_no=through_hand_no,
            sample_size=sample_size,
            leaks=tuple(ranked),
            message_zh=message,
        )

    generate = analyze
    build_profile = analyze


def generate_player_profile(
    through_hand_no: int,
    position_statistics: Mapping[Position | str, PositionStatistics]
    | Iterable[PositionStatistics],
    reason_code_counts: ReasonCountInput = None,
) -> PlayerProfile | None:
    """使用默认 20 手/最多 3 漏洞配置生成玩家画像。"""

    return LeakDetector().analyze(
        through_hand_no, position_statistics, reason_code_counts
    )


# 更短的服务层别名。
build_player_profile = generate_player_profile
analyze_leaks = generate_player_profile


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """可直接转换为 :class:`HoldemHand` 构造参数的训练场景。"""

    scenario_id: str
    title_zh: str = ""
    preferred_position: Position | None = None
    hole_cards: tuple[Card, ...] | Sequence[Card | str] | str = ()
    board_cards: tuple[Card, ...] | Sequence[Card | str] | str = ()
    forced_open: bool = False
    target_leak_code: str | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id 不能为空")
        position = self.preferred_position
        if position is not None and not isinstance(position, Position):
            position = Position(position)
            object.__setattr__(self, "preferred_position", position)
        holes = tuple(parse_cards(self.hole_cards))
        board = tuple(parse_cards(self.board_cards))
        if len(holes) not in (0, 2):
            raise ValueError("场景底牌必须为空或恰好 2 张")
        if len(board) not in (0, 3, 4, 5):
            raise ValueError("场景公共牌必须为 0、3、4 或 5 张")
        if len(set((*holes, *board))) != len(holes) + len(board):
            raise ValueError("场景预设底牌和公共牌不能重复")
        if not isinstance(self.forced_open, bool):
            raise TypeError("forced_open 必须是布尔值")
        object.__setattr__(self, "hole_cards", holes)
        object.__setattr__(self, "board_cards", board)
        if not self.title_zh:
            object.__setattr__(self, "title_zh", self.scenario_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title_zh": self.title_zh,
            "preferred_position": (
                self.preferred_position.value if self.preferred_position else None
            ),
            "hole_cards": [str(card) for card in self.hole_cards],
            "board_cards": [str(card) for card in self.board_cards],
            "forced_open": self.forced_open,
            "target_leak_code": self.target_leak_code,
        }

    to_dict = as_dict

    def to_hand_kwargs(self, hero_id: str | None = None) -> dict[str, Any]:
        """返回可展开传给 ``HoldemHand(..., **kwargs)`` 的参数。"""

        kwargs: dict[str, Any] = {"scenario_id": self.scenario_id}
        if self.hole_cards:
            resolved_hero = hero_id
            if resolved_hero is None and self.preferred_position is not None:
                # 默认项目座位 id 与位置相同；自定义座位 id 时调用方显式传入。
                resolved_hero = self.preferred_position.value
            if not resolved_hero:
                raise ValueError("带预设底牌的场景必须提供 hero_id 或 preferred_position")
            kwargs["hole_overrides"] = {resolved_hero: self.hole_cards}
        if self.board_cards:
            kwargs["board_override"] = self.board_cards
        return kwargs

    hand_kwargs = to_hand_kwargs

    def create_hand(
        self,
        seats: Sequence[Seat],
        *,
        seed: int,
        hero_id: str | None = None,
        **hand_kwargs: Any,
    ) -> HoldemHand:
        """把场景安全地应用到一手牌，供离线模拟和确定性测试使用。"""

        scenario_kwargs = self.to_hand_kwargs(hero_id)
        if "hole_overrides" in scenario_kwargs and "hole_overrides" in hand_kwargs:
            merged = dict(hand_kwargs.pop("hole_overrides"))
            for player_id, cards in scenario_kwargs.pop("hole_overrides").items():
                if player_id in merged:
                    raise ValueError(f"底牌预设重复指定玩家: {player_id}")
                merged[player_id] = cards
            scenario_kwargs["hole_overrides"] = merged
        for key, value in scenario_kwargs.items():
            if key in hand_kwargs and hand_kwargs[key] != value:
                raise ValueError(f"场景参数与调用参数冲突: {key}")
            hand_kwargs[key] = value
        return HoldemHand(seats, seed=seed, **hand_kwargs)

    to_holdem_hand = create_hand


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    RANDOM_SCENARIO_ID: ScenarioSpec(
        RANDOM_SCENARIO_ID,
        "随机常规牌局",
    ),
    "weak_top_pair_overcall": ScenarioSpec(
        "weak_top_pair_overcall",
        "弱顶对面对持续压力",
        preferred_position=Position.BTN,
        hole_cards="Qh 9d",
        board_cards="Qs 8c 3h",
        forced_open=True,
        target_leak_code="weak_top_pair_overcall",
    ),
    "sb_cold_call": ScenarioSpec(
        "sb_cold_call",
        "小盲面对前位开池",
        preferred_position=Position.SB,
        hole_cards="As Jd",
        forced_open=True,
        target_leak_code="sb_cold_call",
    ),
    "threebet_too_small": ScenarioSpec(
        "threebet_too_small",
        "面对开池选择 3bet 尺度",
        preferred_position=Position.BTN,
        hole_cards="Ac Kd",
        forced_open=True,
        target_leak_code="threebet_too_small",
    ),
    "small_pair_overcontinue": ScenarioSpec(
        "small_pair_overcontinue",
        "小对子未中三条的翻后决策",
        preferred_position=Position.CO,
        hole_cards="7c 7d",
        board_cards="As Jh 3c",
        forced_open=False,
        target_leak_code="small_pair_overcontinue",
    ),
    "strong_draw_overfold": ScenarioSpec(
        "strong_draw_overfold",
        "强听牌按赔率继续",
        preferred_position=Position.BTN,
        hole_cards="Ah Qh",
        board_cards="Jh Th 2c",
        forced_open=True,
        target_leak_code="strong_draw_overfold",
    ),
}

# 简短且向后兼容的公开别名。
SCENARIOS = SCENARIO_SPECS


@dataclass(frozen=True, slots=True)
class ScenarioPlan:
    """下一轮场景概率计划。``weights`` 是归一化概率字典。"""

    weights: dict[str, float]
    effective_from_hand_no: int = 1
    hand_count: int = PROFILE_INTERVAL
    scenarios: dict[str, ScenarioSpec] = field(default_factory=dict)
    source_leaks: tuple[LeakFinding, ...] = ()

    def __post_init__(self) -> None:
        weights = {str(key): float(value) for key, value in self.weights.items()}
        if not weights or any(
            not math.isfinite(value) or value < 0 for value in weights.values()
        ):
            raise ValueError("场景权重必须是非负有限数值且至少有一项")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("场景权重合计必须大于 0")
        # 外部加载的计划也归一化，避免概率字典因持久化/人工编辑而失真。
        weights = {key: value / total for key, value in weights.items() if value > 0}
        if self.effective_from_hand_no <= 0:
            raise ValueError("effective_from_hand_no 必须为正数")
        if self.hand_count <= 0:
            raise ValueError("hand_count 必须为正数")
        scenario_map = dict(self.scenarios)
        for scenario_id in weights:
            if scenario_id not in scenario_map and scenario_id in SCENARIO_SPECS:
                scenario_map[scenario_id] = SCENARIO_SPECS[scenario_id]
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "scenarios", scenario_map)
        object.__setattr__(self, "source_leaks", tuple(self.source_leaks))

    @property
    def effective_through_hand_no(self) -> int:
        return self.effective_from_hand_no + self.hand_count - 1

    def scenario(self, scenario_id: str) -> ScenarioSpec:
        try:
            return self.scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"计划缺少场景定义: {scenario_id}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "effective_from_hand_no": self.effective_from_hand_no,
            "hand_count": self.hand_count,
            "weights": dict(self.weights),
            "source_leak_codes": [leak.code for leak in self.source_leaks],
            "scenarios": {
                scenario_id: spec.as_dict()
                for scenario_id, spec in self.scenarios.items()
                if scenario_id in self.weights
            },
        }

    to_dict = as_dict


class AdaptiveScheduler:
    """把最多三个漏洞转换为下一轮 20 手的混合训练计划。"""

    def __init__(
        self,
        scenarios: Mapping[str, ScenarioSpec] | None = None,
        *,
        random_weight_floor: float = RANDOM_WEIGHT_FLOOR,
    ) -> None:
        if not 0.55 <= random_weight_floor <= 1.0:
            raise ValueError("random_weight_floor 必须在 0.55 到 1 之间")
        self.scenarios = dict(scenarios or SCENARIO_SPECS)
        if RANDOM_SCENARIO_ID not in self.scenarios:
            raise ValueError("场景库必须包含 random")
        self.random_weight_floor = float(random_weight_floor)

    def build_plan(
        self,
        profile_or_leaks: PlayerProfile | Sequence[LeakFinding] | None,
        effective_from_hand_no: int | None = None,
        *,
        hand_count: int = PROFILE_INTERVAL,
    ) -> ScenarioPlan:
        if isinstance(profile_or_leaks, PlayerProfile):
            leaks = list(profile_or_leaks.leaks)
            if effective_from_hand_no is None:
                effective_from_hand_no = profile_or_leaks.through_hand_no + 1
        elif profile_or_leaks is None:
            leaks = []
        else:
            leaks = list(profile_or_leaks)
            if any(not isinstance(item, LeakFinding) for item in leaks):
                raise TypeError("leaks 必须由 LeakFinding 组成")
        if effective_from_hand_no is None:
            effective_from_hand_no = 1

        # 每个 code 只取优先级最高的一项，并且只调度有实现的场景。
        targeted: dict[str, LeakFinding] = {}
        for leak in leaks[:MAX_LEAKS]:
            if leak.code not in self.scenarios or leak.code == RANDOM_SCENARIO_ID:
                continue
            current = targeted.get(leak.code)
            if current is None or (
                leak.severity,
                leak.confidence,
                leak.evidence_n,
            ) > (
                current.severity,
                current.confidence,
                current.evidence_n,
            ):
                targeted[leak.code] = leak

        if not targeted:
            weights = {RANDOM_SCENARIO_ID: 1.0}
        else:
            directed_budget = 1.0 - self.random_weight_floor
            scores = {
                code: max(1e-9, leak.severity * max(leak.confidence, 1e-6))
                for code, leak in targeted.items()
            }
            score_total = sum(scores.values())
            weights = {RANDOM_SCENARIO_ID: self.random_weight_floor}
            ordered_codes = sorted(scores)
            allocated = 0.0
            for index, code in enumerate(ordered_codes):
                if index == len(ordered_codes) - 1:
                    weight = max(0.0, directed_budget - allocated)
                else:
                    weight = directed_budget * scores[code] / score_total
                    allocated += weight
                weights[code] = weight

        selected_scenarios = {
            scenario_id: self.scenarios[scenario_id] for scenario_id in weights
        }
        return ScenarioPlan(
            weights=weights,
            effective_from_hand_no=effective_from_hand_no,
            hand_count=hand_count,
            scenarios=selected_scenarios,
            source_leaks=tuple(targeted.values()),
        )

    def choose_scenario(
        self,
        plan: ScenarioPlan,
        seed: int | str | bytes,
        hand_no: int,
    ) -> ScenarioSpec:
        """按稳定 seed 选择场景；不使用进程相关的 ``hash()``。"""

        if not isinstance(plan, ScenarioPlan):
            raise TypeError("plan 必须是 ScenarioPlan")
        if isinstance(hand_no, bool) or not isinstance(hand_no, int) or hand_no <= 0:
            raise ValueError("hand_no 必须是正整数")
        canonical_weights = sorted(plan.weights.items())
        seed_text = (
            seed.hex() if isinstance(seed, bytes) else str(seed)
        )
        payload = json.dumps(
            {
                "seed": seed_text,
                "hand_no": hand_no,
                "weights": canonical_weights,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        random_unit = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        cumulative = 0.0
        selected_id = canonical_weights[-1][0]
        for scenario_id, weight in canonical_weights:
            cumulative += weight
            if random_unit < cumulative:
                selected_id = scenario_id
                break
        if selected_id in plan.scenarios:
            return plan.scenarios[selected_id]
        try:
            return self.scenarios[selected_id]
        except KeyError as exc:
            raise KeyError(f"未知训练场景: {selected_id}") from exc

    select_scenario = choose_scenario
    choose = choose_scenario


_DEFAULT_SCHEDULER = AdaptiveScheduler()


def build_scenario_plan(
    profile_or_leaks: PlayerProfile | Sequence[LeakFinding] | None,
    effective_from_hand_no: int | None = None,
    *,
    hand_count: int = PROFILE_INTERVAL,
) -> ScenarioPlan:
    """模块级宽兼容入口：生成下一轮场景计划。"""

    return _DEFAULT_SCHEDULER.build_plan(
        profile_or_leaks,
        effective_from_hand_no,
        hand_count=hand_count,
    )


def choose_scenario(
    plan: ScenarioPlan,
    seed: int | str | bytes,
    hand_no: int,
) -> ScenarioSpec:
    return _DEFAULT_SCHEDULER.choose_scenario(plan, seed, hand_no)


__all__ = [
    "AdaptiveProfile",
    "AdaptiveScheduler",
    "LeakDetector",
    "LeakFinding",
    "MAX_LEAKS",
    "MIN_EVIDENCE",
    "PROFILE_INTERVAL",
    "PlayerProfile",
    "PlayerProfileSnapshot",
    "RANDOM_SCENARIO_ID",
    "RANDOM_WEIGHT_FLOOR",
    "SCENARIOS",
    "SCENARIO_SPECS",
    "ScenarioPlan",
    "ScenarioSpec",
    "analyze_leaks",
    "build_player_profile",
    "build_scenario_plan",
    "choose_scenario",
    "generate_player_profile",
]

