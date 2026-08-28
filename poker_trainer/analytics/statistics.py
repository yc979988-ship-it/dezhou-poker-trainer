"""英雄牌局统计投影与按位置聚合。

本模块只从一手已经结束的 :class:`~poker_trainer.engine.hand.HoldemHand`
及其 :class:`~poker_trainer.engine.models.ActionRecord` 历史中读取事实，不修改
牌局状态。

MVP 口径
--------
* VPIP/PFR：每手牌各有一次机会；盲注不算自愿投入。
* open limp：英雄第一次翻前决策时底池无人自愿入池，且英雄不在 BB；
  以跟注（含仅能跟注的短码全下）进入底池记为命中。
* cold call：英雄第一次翻前决策已经面对至少一次加注；跟注记为命中。
* 3bet：英雄第一次翻前决策正好面对一次加注；再次加注记为命中。
* fold to 3bet：英雄做出本手第一次翻前加注，之后面对他人的再次加注并
  获得响应机会；弃牌记为命中。
* c-bet：英雄是最后一名翻前加注者，且翻牌第一次行动时无人领先下注；
  下注记为命中。
* WTSD：以见到翻牌的手数为分母；仍在牌局中进入摊牌为命中。
* W$SD：以进入摊牌的手数为分母；摊牌结算获得任意底池份额为命中。
* AF：分子是翻后 bet/raise 次数，分母是翻后 call 次数；这是比率而非
  百分比，零次 call 时值为 ``None``。
* 顶对打光率和听牌赔率错误依赖复盘语义，分别由本模块定义的稳定
  ``reason_code`` 注入；投影器不凭最终输赢反推决策质量。
* 河牌跟注频率：英雄每次在河牌面对尚需跟注的下注均为一次机会；纯跟注
  （含短码跟注全下）记为命中，加注不算“跟注”。

所有计数都保留分子 ``hits`` 与分母 ``opportunities``。零分母的比率和
百分比均为 ``None``，避免小样本或无样本时硬算。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionRecord, ActionType, Position, Street


class MetricName(str, Enum):
    """持久化使用的稳定指标标识。"""

    VPIP = "vpip"
    PFR = "pfr"
    OPEN_LIMP = "open_limp"
    COLD_CALL = "cold_call"
    THREE_BET = "three_bet"
    FOLD_TO_THREE_BET = "fold_to_three_bet"
    CBET = "cbet"
    WTSD = "wtsd"
    WSD = "wsd"
    AGGRESSION_FACTOR = "aggression_factor"
    TOP_PAIR_STACKOFF = "top_pair_stackoff"
    RIVER_CALL = "river_call"
    DRAW_ODDS_ERROR = "draw_odds_error"


METRIC_ORDER: tuple[MetricName, ...] = tuple(MetricName)

METRIC_LABELS_ZH: dict[MetricName, str] = {
    MetricName.VPIP: "VPIP",
    MetricName.PFR: "PFR",
    MetricName.OPEN_LIMP: "open limp",
    MetricName.COLD_CALL: "cold call",
    MetricName.THREE_BET: "3bet",
    MetricName.FOLD_TO_THREE_BET: "fold to 3bet",
    MetricName.CBET: "c-bet",
    MetricName.WTSD: "WTSD",
    MetricName.WSD: "W$SD",
    MetricName.AGGRESSION_FACTOR: "激进因子 AF",
    MetricName.TOP_PAIR_STACKOFF: "顶对打光率",
    MetricName.RIVER_CALL: "河牌跟注频率",
    MetricName.DRAW_ODDS_ERROR: "听牌赔率错误次数",
}

# 由牌后复盘器写入。命中代码本身蕴含一次机会；为了保持严格分母，若同一
# 批输入显式给出了机会代码，则机会数必须不少于命中数。
TOP_PAIR_STACKOFF_OPPORTUNITY = "top_pair_stackoff_opportunity"
TOP_PAIR_STACKED_OFF = "top_pair_stacked_off"
DRAW_ODDS_OPPORTUNITY = "draw_odds_opportunity"
DRAW_ODDS_ERROR = "draw_odds_error"


@dataclass(frozen=True, slots=True)
class MetricTally:
    """单个指标的分子与分母；可直接转换为 JSON 兼容字典。"""

    hits: int = 0
    opportunities: int = 0

    def __post_init__(self) -> None:
        if self.hits < 0 or self.opportunities < 0:
            raise ValueError("统计计数不能为负数")

    @property
    def value(self) -> float | None:
        """返回未舍入的 hits/opportunities；零分母返回 ``None``。"""

        if self.opportunities == 0:
            return None
        return self.hits / self.opportunities

    @property
    def percentage(self) -> float | None:
        """返回未舍入百分比；AF 展示时应使用 :attr:`value` 而非此属性。"""

        value = self.value
        return None if value is None else value * 100.0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "hits": self.hits,
            "opportunities": self.opportunities,
            "value": self.value,
            "percentage": self.percentage,
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class HandStatistics:
    """一手牌中英雄的统计投影。"""

    hand_id: str
    player_id: str
    position: Position
    metrics: Mapping[MetricName, MetricTally]

    def metric(self, name: MetricName | str) -> MetricTally:
        return self.metrics[MetricName(name)]

    @property
    def draw_odds_error_count(self) -> int:
        """界面主显的听牌赔率错误原始次数。"""

        return self.metric(MetricName.DRAW_ODDS_ERROR).hits

    def as_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "player_id": self.player_id,
            "position": self.position.value,
            "metrics": {
                name.value: self.metric(name).as_dict() for name in METRIC_ORDER
            },
            "draw_odds_error_count": self.draw_odds_error_count,
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class PositionStatistics:
    """某一位置的累计统计；比率由累计分子/累计分母计算。"""

    position: Position
    hands: int
    metrics: Mapping[MetricName, MetricTally]

    def metric(self, name: MetricName | str) -> MetricTally:
        return self.metrics[MetricName(name)]

    @property
    def draw_odds_error_count(self) -> int:
        return self.metric(MetricName.DRAW_ODDS_ERROR).hits

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.value,
            "position_zh": self.position.label_zh,
            "hands": self.hands,
            "metrics": {
                name.value: self.metric(name).as_dict() for name in METRIC_ORDER
            },
            "draw_odds_error_count": self.draw_odds_error_count,
        }

    to_dict = as_dict


ReasonCodeInput = (
    str
    | Iterable[str]
    | Mapping[object, object]
    | object
    | None
)


def _iter_reason_codes(value: ReasonCodeInput) -> Iterable[str]:
    """兼容代码序列、按 action sequence 映射及带 reason_codes 的对象。"""

    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        if "reason_codes" in value:
            yield from _iter_reason_codes(value["reason_codes"])
        else:
            for nested in value.values():
                yield from _iter_reason_codes(nested)
        return
    reason_codes = getattr(value, "reason_codes", None)
    if reason_codes is not None:
        yield from _iter_reason_codes(reason_codes)
        return
    if isinstance(value, Iterable):
        for nested in value:
            yield from _iter_reason_codes(nested)
        return
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        yield enum_value
        return
    raise TypeError("reason_codes 必须是字符串、可迭代对象、映射或带 reason_codes 的对象")


def _is_decision(record: ActionRecord) -> bool:
    return not record.forced and record.action not in {
        ActionType.POST_SB,
        ActionType.POST_BB,
        ActionType.REFUND,
    }


def _is_aggressive(record: ActionRecord) -> bool:
    return (
        _is_decision(record)
        and record.action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
        and record.current_bet_after > record.current_bet_before
    )


def _is_call(record: ActionRecord) -> bool:
    return _is_decision(record) and (
        record.action == ActionType.CALL
        or (
            record.action == ActionType.ALL_IN
            and record.to_call_before > 0
            and record.current_bet_after <= record.current_bet_before
        )
    )


def _voluntarily_invested(record: ActionRecord) -> bool:
    return (
        _is_decision(record)
        and record.paid > 0
        and record.action
        in {ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
    )


def _binary(hit: bool, opportunity: bool = True) -> MetricTally:
    return MetricTally(int(hit), int(opportunity))


def _review_tally(
    counts: Counter[str], opportunity_code: str, hit_code: str, *, label: str
) -> MetricTally:
    hits = counts[hit_code]
    explicit_opportunities = counts[opportunity_code]
    opportunities = explicit_opportunities if explicit_opportunities else hits
    if hits > opportunities:
        raise ValueError(f"{label}命中数不能大于机会数")
    return MetricTally(hits, opportunities)


def project_hand_statistics(
    hand: HoldemHand,
    hero_id: str,
    *,
    reason_codes: ReasonCodeInput = None,
) -> HandStatistics:
    """把一手已结束牌局投影为英雄统计。

    ``reason_codes`` 可以是一个字符串序列、``action_sequence -> codes`` 映射，
    或带 ``reason_codes`` 属性的复盘对象（及其序列）。
    """

    if not hand.is_complete or hand.result is None:
        raise ValueError("只能统计已经结束的牌局")
    hero = hand.player(hero_id)  # 同时验证 player_id
    records = sorted(hand.history, key=lambda item: item.sequence)
    decisions = [record for record in records if _is_decision(record)]
    preflop = [record for record in decisions if record.street == Street.PREFLOP]
    hero_preflop = [record for record in preflop if record.player_id == hero_id]
    first_hero_preflop = hero_preflop[0] if hero_preflop else None

    vpip_hit = any(_voluntarily_invested(record) for record in hero_preflop)
    pfr_hit = any(_is_aggressive(record) for record in hero_preflop)

    earlier_preflop: list[ActionRecord] = []
    if first_hero_preflop is not None:
        earlier_preflop = [
            record for record in preflop if record.sequence < first_hero_preflop.sequence
        ]
    raises_before_first = sum(_is_aggressive(record) for record in earlier_preflop)
    voluntary_entry_before_first = any(
        _voluntarily_invested(record) for record in earlier_preflop
    )

    open_limp_opportunity = bool(
        first_hero_preflop is not None
        and hero.position != Position.BB
        and not voluntary_entry_before_first
    )
    open_limp_hit = bool(
        open_limp_opportunity and first_hero_preflop and _is_call(first_hero_preflop)
    )

    cold_call_opportunity = bool(
        first_hero_preflop is not None and raises_before_first >= 1
    )
    cold_call_hit = bool(
        cold_call_opportunity and first_hero_preflop and _is_call(first_hero_preflop)
    )

    three_bet_opportunity = bool(
        first_hero_preflop is not None and raises_before_first == 1
    )
    three_bet_hit = bool(
        three_bet_opportunity
        and first_hero_preflop
        and _is_aggressive(first_hero_preflop)
    )

    fold_to_three_bet_opportunity = False
    fold_to_three_bet_hit = False
    raise_count = 0
    hero_first_raise: ActionRecord | None = None
    for record in preflop:
        if _is_aggressive(record):
            if record.player_id == hero_id and raise_count == 0:
                hero_first_raise = record
                break
            raise_count += 1
    if hero_first_raise is not None:
        faced_reraise = False
        for record in preflop:
            if record.sequence <= hero_first_raise.sequence:
                continue
            if record.player_id != hero_id and _is_aggressive(record):
                faced_reraise = True
                continue
            if record.player_id == hero_id and faced_reraise:
                if record.to_call_before > 0:
                    fold_to_three_bet_opportunity = True
                    fold_to_three_bet_hit = record.action == ActionType.FOLD
                break

    preflop_aggressors = [record for record in preflop if _is_aggressive(record)]
    last_preflop_aggressor = (
        preflop_aggressors[-1].player_id if preflop_aggressors else None
    )
    hero_flop = [
        record
        for record in decisions
        if record.player_id == hero_id and record.street == Street.FLOP
    ]
    first_hero_flop = hero_flop[0] if hero_flop else None
    cbet_opportunity = bool(
        last_preflop_aggressor == hero_id
        and first_hero_flop is not None
        and first_hero_flop.current_bet_before == 0
    )
    cbet_hit = bool(cbet_opportunity and first_hero_flop and _is_aggressive(first_hero_flop))

    folded_preflop = any(record.action == ActionType.FOLD for record in hero_preflop)
    saw_flop = len(hand.board) >= 3 and not folded_preflop
    went_to_showdown = hand.result.reason == "showdown" and not hero.folded
    won_at_showdown = went_to_showdown and hand.result.payouts.get(hero_id, 0) > 0

    hero_postflop = [
        record
        for record in decisions
        if record.player_id == hero_id
        and record.street in {Street.FLOP, Street.TURN, Street.RIVER}
    ]
    aggressive_actions = sum(_is_aggressive(record) for record in hero_postflop)
    calls = sum(_is_call(record) for record in hero_postflop)

    river_facing_bet = [
        record
        for record in hero_postflop
        if record.street == Street.RIVER and record.to_call_before > 0
    ]
    river_calls = sum(_is_call(record) for record in river_facing_bet)

    code_counts = Counter(_iter_reason_codes(reason_codes))
    top_pair = _review_tally(
        code_counts,
        TOP_PAIR_STACKOFF_OPPORTUNITY,
        TOP_PAIR_STACKED_OFF,
        label="顶对打光",
    )
    draw_odds = _review_tally(
        code_counts,
        DRAW_ODDS_OPPORTUNITY,
        DRAW_ODDS_ERROR,
        label="听牌赔率错误",
    )

    metrics: dict[MetricName, MetricTally] = {
        MetricName.VPIP: _binary(vpip_hit),
        MetricName.PFR: _binary(pfr_hit),
        MetricName.OPEN_LIMP: _binary(open_limp_hit, open_limp_opportunity),
        MetricName.COLD_CALL: _binary(cold_call_hit, cold_call_opportunity),
        MetricName.THREE_BET: _binary(three_bet_hit, three_bet_opportunity),
        MetricName.FOLD_TO_THREE_BET: _binary(
            fold_to_three_bet_hit, fold_to_three_bet_opportunity
        ),
        MetricName.CBET: _binary(cbet_hit, cbet_opportunity),
        MetricName.WTSD: _binary(went_to_showdown, saw_flop),
        MetricName.WSD: _binary(won_at_showdown, went_to_showdown),
        MetricName.AGGRESSION_FACTOR: MetricTally(aggressive_actions, calls),
        MetricName.TOP_PAIR_STACKOFF: top_pair,
        MetricName.RIVER_CALL: MetricTally(river_calls, len(river_facing_bet)),
        MetricName.DRAW_ODDS_ERROR: draw_odds,
    }
    return HandStatistics(hand.hand_id, hero_id, hero.position, metrics)


def aggregate_by_position(
    hands: Iterable[HandStatistics], *, include_empty: bool = True
) -> dict[Position, PositionStatistics]:
    """按位置累计分子/分母，不对单手百分比做算术平均。"""

    positions = list(Position) if include_empty else []
    rows = list(hands)
    if not include_empty:
        positions = list(dict.fromkeys(row.position for row in rows))

    result: dict[Position, PositionStatistics] = {}
    for position in positions:
        position_rows = [row for row in rows if row.position == position]
        totals = {
            name: MetricTally(
                sum(row.metric(name).hits for row in position_rows),
                sum(row.metric(name).opportunities for row in position_rows),
            )
            for name in METRIC_ORDER
        }
        result[position] = PositionStatistics(position, len(position_rows), totals)
    return result


# 语义清晰的短别名，方便服务层调用。
project_hero_hand = project_hand_statistics
aggregate_position_statistics = aggregate_by_position


def serialize_position_statistics(
    statistics: Mapping[Position, PositionStatistics],
) -> dict[str, dict[str, Any]]:
    """把按位置聚合结果转换为 JSON 兼容、字符串键字典。"""

    return {position.value: row.as_dict() for position, row in statistics.items()}


__all__ = [
    "DRAW_ODDS_ERROR",
    "DRAW_ODDS_OPPORTUNITY",
    "HandStatistics",
    "METRIC_LABELS_ZH",
    "METRIC_ORDER",
    "MetricName",
    "MetricTally",
    "PositionStatistics",
    "TOP_PAIR_STACKED_OFF",
    "TOP_PAIR_STACKOFF_OPPORTUNITY",
    "aggregate_by_position",
    "aggregate_position_statistics",
    "project_hand_statistics",
    "project_hero_hand",
    "serialize_position_statistics",
]

