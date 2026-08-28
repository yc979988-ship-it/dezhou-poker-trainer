"""德州扑克自适应训练器的 Streamlit 移动网页。

顶层只导入标准库和本地领域模型。Streamlit 与会话控制器均在
运行界面时延迟导入，因此没有安装网页依赖时仍可以测试格式化、
合法动作与回放辅助函数。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from poker_trainer.analytics.statistics import (
    METRIC_LABELS_ZH,
    METRIC_ORDER,
    MetricName,
)
from poker_trainer.engine.models import ActionType, Position, Street
from poker_trainer.engine.replay import ReplayBundle

from .styles import apply_styles


APP_TITLE = "德州扑克自适应训练器"
PRODUCT_NOTICE = "离线模拟 · 不做真实牌局实时读屏 · MVP 不是完整 GTO 求解器"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = Path(
    os.getenv(
        "POKER_TRAINER_DB_PATH",
        str(_PROJECT_ROOT / "data" / "poker_trainer.sqlite3"),
    )
)
CHIPS_PER_YUAN = 100

POSITION_ORDER: tuple[Position, ...] = (
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
    Position.SB,
    Position.BB,
)

ACTION_LABELS: dict[ActionType, str] = {
    ActionType.FOLD: "弃牌",
    ActionType.CHECK: "过牌",
    ActionType.CALL: "跟注",
    ActionType.BET: "下注",
    ActionType.RAISE: "加注",
    ActionType.ALL_IN: "全下",
    ActionType.POST_SB: "下小盲",
    ActionType.POST_BB: "下大盲",
    ActionType.REFUND: "退回筹码",
}


@dataclass(frozen=True, slots=True)
class ActionControl:
    """界面可显示的单个合法动作。"""

    action: ActionType
    label: str
    needs_amount: bool = False


@dataclass(frozen=True, slots=True)
class BetToRange:
    """Bet/Raise 滑块的合法 bet-to 区间。"""

    minimum: int
    maximum: int
    default: int
    step: int


def _value(value: Any) -> Any:
    """枚举取 value，普通值原样返回。"""

    return getattr(value, "value", value)


def _read(source: Any, key: str, default: Any = None) -> Any:
    """同时读取 dataclass/对象和字典，作为 UI 薄适配层。"""

    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def full_position_name(position: Position | str) -> str:
    """返回不会丢失英文缩写的中文位置全称。"""

    try:
        return Position(_value(position)).display_name
    except (TypeError, ValueError):
        return str(_value(position))


def format_chips(chips: int | float, chips_per_yuan: int = CHIPS_PER_YUAN) -> str:
    """同时显示筹码与仅供换算的人民币金额。"""

    if chips_per_yuan <= 0:
        raise ValueError("chips_per_yuan 必须大于 0")
    amount = float(chips)
    chip_text = f"{amount:,.0f}" if amount.is_integer() else f"{amount:,.2f}"
    return f"{chip_text} 筹码（¥{amount / chips_per_yuan:,.2f}）"


def format_percentage(value: float | None, *, digits: int = 1) -> str:
    """将 0..1 比例转为百分比，空样本明确显示信号不足。"""

    if value is None:
        return "信号不足"
    return f"{float(value) * 100:.{digits}f}%"


_SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}


def format_card(card: Any) -> str:
    """把引擎紧凑牌面 ``As`` 转成手机友好的 ``A♠``。"""

    text = str(card).strip()
    if len(text) < 2:
        return text
    rank, suit = text[:-1], text[-1].lower()
    if rank.upper() == "T":
        rank = "10"
    return f"{rank.upper()}{_SUIT_SYMBOLS.get(suit, suit)}"


def format_cards(
    cards: Iterable[Any] | None,
    *,
    hidden: bool = False,
    hidden_count: int = 2,
    empty: str = "尚未发牌",
) -> str:
    """返回牌面文本；对手未亮牌时只返回牌背占位。"""

    if hidden:
        return " ".join("🂠" for _ in range(max(0, hidden_count)))
    values = tuple(cards or ())
    return " ".join(format_card(card) for card in values) if values else empty


def cards_html(
    cards: Iterable[Any] | None,
    *,
    hidden: bool = False,
    hidden_count: int = 2,
) -> str:
    """生成仅含可控牌面文本的 HTML 牌片。"""

    if hidden:
        return "".join(
            '<span class="playing-card hidden">◆</span>'
            for _ in range(max(0, hidden_count))
        )
    values = tuple(cards or ())
    if not values:
        return '<span style="opacity:.7">—</span>'
    rendered: list[str] = []
    for card in values:
        compact = str(card).strip()
        suit = compact[-1:].lower()
        css = "playing-card red" if suit in {"d", "h"} else "playing-card"
        rendered.append(f'<span class="{css}">{escape(format_card(card))}</span>')
    return "".join(rendered)


def legal_action_controls(legal: Any) -> tuple[ActionControl, ...]:
    """严格按 ``LegalActions`` 布尔字段构建按钮，不显示非法动作。"""

    controls: list[ActionControl] = []
    definitions = (
        ("can_fold", ActionType.FOLD, False),
        ("can_check", ActionType.CHECK, False),
        ("can_call", ActionType.CALL, False),
        ("can_bet", ActionType.BET, True),
        ("can_raise", ActionType.RAISE, True),
        ("can_all_in", ActionType.ALL_IN, False),
    )
    for flag, action, needs_amount in definitions:
        if bool(_read(legal, flag, False)):
            label = ACTION_LABELS[action]
            if action == ActionType.CALL:
                call_amount = int(_read(legal, "call_amount", 0) or 0)
                label = f"{label} {call_amount:,}"
            controls.append(ActionControl(action, label, needs_amount))
    return tuple(controls)


def bet_to_range(legal: Any, *, chip_unit: int = 20) -> BetToRange | None:
    """从合法动作生成最小加注到有效后手的滑块区间。"""

    minimum: int | None = None
    if bool(_read(legal, "can_bet", False)):
        raw = _read(legal, "min_bet_to")
        minimum = None if raw is None else int(raw)
    elif bool(_read(legal, "can_raise", False)):
        raw = _read(legal, "min_raise_to")
        minimum = None if raw is None else int(raw)
    if minimum is None:
        return None

    maximum = int(_read(legal, "max_to", minimum))
    if maximum < minimum:
        return None
    step = max(1, int(chip_unit))
    pot = max(0, int(_read(legal, "pot_before", 0) or 0))
    suggested = minimum + round((pot * 0.5) / step) * step
    default = min(maximum, max(minimum, suggested))
    return BetToRange(minimum, maximum, default, step)


def action_history_rows(history: Iterable[Any]) -> list[dict[str, Any]]:
    """将动作日志转为中文公开表格，不包含对手隐藏参数。"""

    rows: list[dict[str, Any]] = []
    for record in history:
        action = ActionType(_value(_read(record, "action")))
        paid = int(_read(record, "paid", 0) or 0)
        bet_to = int(_read(record, "bet_to", 0) or 0)
        forced = bool(_read(record, "forced", False))
        if action in {ActionType.BET, ActionType.RAISE}:
            detail = f"到 {bet_to:,}"
            summary = f"{ACTION_LABELS[action]}到 {bet_to:,}"
        elif action == ActionType.ALL_IN:
            detail = f"投入 {paid:,}，到 {bet_to:,}"
            summary = f"全下至 {bet_to:,}"
        elif action == ActionType.REFUND:
            detail = f"退回 {abs(paid):,}"
            summary = detail
        elif action in {ActionType.POST_SB, ActionType.POST_BB}:
            detail = f"{paid:,}"
            summary = f"{ACTION_LABELS[action]} {paid:,}"
        elif action == ActionType.CALL:
            detail = f"投入 {paid:,}" if paid else "—"
            summary = f"跟注 {paid:,}" if paid else "跟注"
        elif paid:
            detail = f"投入 {paid:,}"
            summary = f"{ACTION_LABELS[action]} {paid:,}"
        else:
            detail = "—"
            summary = ACTION_LABELS[action]
        position = Position(_value(_read(record, "position")))
        rows.append(
            {
                "#": int(_read(record, "sequence", len(rows))),
                "街道": Street(_value(_read(record, "street"))).label_zh,
                "位置": full_position_name(position),
                "位置简称": position.value,
                "玩家": str(_read(record, "player_id", "")),
                "动作": ACTION_LABELS[action],
                "动作代码": action.value,
                "简述": summary,
                "筹码": detail,
                "底池后": int(_read(record, "pot_after", 0) or 0),
                "强制": forced,
            }
        )
    return rows


def action_timeline_html(
    rows: Iterable[Mapping[str, Any]],
    *,
    hero_id: str | None = None,
    limit: int | None = None,
    active_sequence: int | None = None,
) -> str:
    """把行动记录渲染成适合窄屏的按街时间线。"""

    visible = list(rows)
    if limit is not None and limit >= 0:
        visible = visible[-limit:] if limit else []
    if not visible:
        return '<div class="action-empty">尚无行动</div>'

    parts = ['<div class="action-feed">']
    previous_street: str | None = None
    for index, row in enumerate(visible):
        street = str(row.get("街道", ""))
        if street != previous_street:
            parts.append(f'<div class="action-street">{escape(street)}</div>')
            previous_street = street
        sequence = int(row.get("#", 0) or 0)
        action_code = str(row.get("动作代码", "check")).replace("_", "-")
        classes = ["action-line", f"action-{action_code}"]
        if str(row.get("玩家", "")) == hero_id:
            classes.append("hero-action")
        if sequence == active_sequence or (active_sequence is None and index == len(visible) - 1):
            classes.append("latest")
        actor = (
            "你"
            if str(row.get("玩家", "")) == hero_id
            else str(row.get("位置简称", row.get("位置", "")))
        )
        pot_after = int(row.get("底池后", 0) or 0)
        parts.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="action-actor">{escape(actor)}</span>'
            f'<span class="action-summary">{escape(str(row.get("简述", row.get("动作", ""))))}</span>'
            f'<span class="action-pot">池 {pot_after:,}</span>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def result_pot_rows(hand: Any) -> list[dict[str, Any]]:
    """生成公开的主池/边池结算明细，不暴露任何隐藏画像。"""

    result = _read(hand, "result")
    if result is None:
        return []
    players_source = _read(hand, "players", {}) or {}
    players = (
        players_source
        if isinstance(players_source, Mapping)
        else {_read(player, "player_id"): player for player in players_source}
    )

    def player_label(player_id: str) -> str:
        player = players.get(player_id)
        if player is None:
            return player_id
        return f"{full_position_name(_read(player, 'position'))} · {_read(player, 'name', player_id)}"

    rows: list[dict[str, Any]] = []
    for index, pot in enumerate(tuple(_read(result, "pots", ()) or ())):
        eligible = tuple(_read(pot, "eligible", ()) or ())
        rows.append(
            {
                "底池": "主池" if index == 0 else f"边池 {index}",
                "金额": format_chips(int(_read(pot, "amount", 0) or 0)),
                "可争夺玩家": "、".join(player_label(str(player_id)) for player_id in eligible),
            }
        )
    return rows


def showdown_rank_rows(hand: Any) -> list[dict[str, Any]]:
    """生成摊牌牌型与派奖表；弃牌玩家不会出现在牌型映射中。"""

    result = _read(hand, "result")
    if result is None or _read(result, "reason") != "showdown":
        return []
    players_source = _read(hand, "players", {}) or {}
    players = (
        players_source
        if isinstance(players_source, Mapping)
        else {_read(player, "player_id"): player for player in players_source}
    )
    payouts = _read(result, "payouts", {}) or {}
    rows: list[dict[str, Any]] = []
    for player_id, rank in (_read(result, "hand_ranks", {}) or {}).items():
        player = players.get(player_id)
        rows.append(
            {
                "玩家": (
                    f"{full_position_name(_read(player, 'position'))} · "
                    f"{_read(player, 'name', player_id)}"
                    if player is not None
                    else str(player_id)
                ),
                "牌型": str(rank),
                "获得": format_chips(int(payouts.get(player_id, 0) or 0)),
            }
        )
    return rows


def hand_public_view(
    hand: Any,
    hero_id: str,
    *,
    reveal_showdown: bool = True,
) -> dict[str, Any]:
    """组装安全界面快照：对手底牌只在合法摊牌时显示。"""

    result = _read(hand, "result")
    showdown = bool(
        reveal_showdown
        and result is not None
        and _read(result, "reason") == "showdown"
    )
    players_source = _read(hand, "players", {})
    players = (
        players_source.values()
        if isinstance(players_source, Mapping)
        else players_source
    )
    indexed: dict[Position, Any] = {}
    for player in players:
        indexed[Position(_value(_read(player, "position")))] = player

    seats: list[dict[str, Any]] = []
    for position in POSITION_ORDER:
        player = indexed.get(position)
        if player is None:
            seats.append(
                {
                    "position": position,
                    "position_name": full_position_name(position),
                    "player_id": "",
                    "name": "空位",
                    "stack": 0,
                    "folded": True,
                    "all_in": False,
                    "is_hero": False,
                    "cards": (),
                    "cards_hidden": True,
                }
            )
            continue
        player_id = str(_read(player, "player_id"))
        folded = bool(_read(player, "folded", False))
        can_reveal = player_id == hero_id or (showdown and not folded)
        seats.append(
            {
                "position": position,
                "position_name": full_position_name(position),
                "player_id": player_id,
                "name": str(_read(player, "name", player_id)),
                "stack": int(_read(player, "stack", 0) or 0),
                "folded": folded,
                "all_in": bool(_read(player, "all_in", False)),
                "is_hero": player_id == hero_id,
                "cards": tuple(_read(player, "hole_cards", ()) or ()) if can_reveal else (),
                "cards_hidden": not can_reveal,
            }
        )

    street = Street(_value(_read(hand, "street")))
    current_actor_id = _read(hand, "current_actor_id")
    pot = int(_read(hand, "pot_size", 0) or 0)
    if _read(hand, "is_complete", False) and pot == 0:
        pot = int(_read(hand, "committed_pot", 0) or 0)
    return {
        "hand_id": str(_read(hand, "hand_id", "")),
        "hand_no": int(_read(hand, "hand_no", 0) or 0),
        "seed": int(_read(hand, "seed", 0) or 0),
        "street": street,
        "street_name": street.label_zh,
        "board": tuple(_read(hand, "board", ()) or ()),
        "pot": pot,
        "current_actor_id": current_actor_id,
        "is_complete": bool(_read(hand, "is_complete", False)),
        "seats": seats,
        "history": action_history_rows(_read(hand, "history", ()) or ()),
    }


def _metric_tally(row: Any, name: MetricName) -> tuple[int, int]:
    """从 dataclass 或数据库序列化字典取统计分子/分母。"""

    metric_method = getattr(row, "metric", None)
    if callable(metric_method):
        tally = metric_method(name)
    else:
        metrics = _read(row, "metrics", {}) or {}
        tally = metrics.get(name, metrics.get(name.value, {})) if isinstance(metrics, Mapping) else {}
    return int(_read(tally, "hits", 0) or 0), int(_read(tally, "opportunities", 0) or 0)


def format_metric(name: MetricName | str, hits: int, opportunities: int) -> str:
    """按指标口径格式化：AF 为比值，听牌错误为次数，其余为百分比。"""

    metric = MetricName(_value(name))
    if metric == MetricName.DRAW_ODDS_ERROR:
        return f"{hits}次（{opportunities}次机会）"
    if opportunities <= 0:
        return "信号不足"
    if metric == MetricName.AGGRESSION_FACTOR:
        return f"{hits / opportunities:.2f}（{hits}/{opportunities}）"
    return f"{hits / opportunities * 100:.1f}%（{hits}/{opportunities}）"


def statistics_table_rows(statistics: Mapping[Any, Any] | Iterable[Any] | None) -> list[dict[str, Any]]:
    """生成固定六位置×13指标表，缺样本不硬算。"""

    source: dict[Position, Any] = {}
    if isinstance(statistics, Mapping):
        for key, row in statistics.items():
            raw_position = _read(row, "position", key)
            try:
                source[Position(_value(raw_position))] = row
            except (TypeError, ValueError):
                continue
    elif statistics is not None:
        for row in statistics:
            try:
                source[Position(_value(_read(row, "position")))] = row
            except (TypeError, ValueError):
                continue

    rows: list[dict[str, Any]] = []
    for position in POSITION_ORDER:
        item = source.get(position)
        row: dict[str, Any] = {
            "位置": full_position_name(position),
            "手数": int(_read(item, "hands", 0) or 0),
        }
        for metric in METRIC_ORDER:
            hits, opportunities = _metric_tally(item, metric) if item is not None else (0, 0)
            row[METRIC_LABELS_ZH[metric]] = format_metric(metric, hits, opportunities)
        rows.append(row)
    return rows


def metric_detail_rows(position_row: Any) -> list[dict[str, Any]]:
    """将单位置的 13 个指标转成手机竖向表。"""

    rows: list[dict[str, Any]] = []
    for metric in METRIC_ORDER:
        hits, opportunities = _metric_tally(position_row, metric)
        rows.append(
            {
                "指标": METRIC_LABELS_ZH[metric],
                "当前": format_metric(metric, hits, opportunities),
                "命中": hits,
                "机会": opportunities,
            }
        )
    return rows


def normalize_reviews(value: Any) -> tuple[Any, ...]:
    """将教学单条反馈、测试整手反馈或 None 统一成 tuple。"""

    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def replay_reviews_through_sequence(
    reviews: Iterable[Any],
    active_sequence: int | None,
) -> tuple[Any, ...]:
    """返回回放当前步已经发生的评价，并严格按 ``review.sequence`` 排序。"""

    if active_sequence is None:
        return ()
    reached: list[tuple[int, Any]] = []
    for review in reviews:
        raw_sequence = _read(review, "sequence")
        if raw_sequence is None:
            continue
        try:
            sequence = int(raw_sequence)
        except (TypeError, ValueError):
            continue
        if sequence <= active_sequence:
            reached.append((sequence, review))
    reached.sort(key=lambda item: item[0])
    return tuple(review for _, review in reached)


_RATING_SEVERITY = {
    "推荐": 0,
    "可以接受": 1,
    "偏松/偏紧": 2,
    "明显错误": 3,
}

_RATING_CSS = {
    "推荐": "recommended",
    "可以接受": "acceptable",
    "偏松/偏紧": "marginal",
    "明显错误": "error",
}


def review_cards_html(
    reviews: Sequence[Any],
    *,
    history: Iterable[Mapping[str, Any]] = (),
) -> str:
    """生成带街道、英雄动作和数学依据的移动端复盘卡。"""

    normalized = normalize_reviews(reviews)
    if not normalized:
        return ""
    action_by_sequence = {
        int(row.get("#", -1)): row for row in history
    }
    ratings = [str(_value(_read(review, "rating", _read(review, "grade", "")))) for review in normalized]
    worst = max(ratings, key=lambda item: _RATING_SEVERITY.get(item, -1))
    error_count = sum(rating == "明显错误" for rating in ratings)
    overview = f"{len(normalized)} 次决策 · 最严重：{worst}"
    if error_count:
        overview += f" · {error_count} 个明显错误"

    parts = [f'<div class="review-overview">{escape(overview)}</div>', '<div class="review-list">']
    for review in normalized:
        try:
            sequence = int(_read(review, "sequence", -1))
        except (TypeError, ValueError):
            sequence = -1
        street_raw = _read(review, "street", "")
        try:
            street_name = Street(_value(street_raw)).label_zh
        except (TypeError, ValueError):
            street_name = str(_value(street_raw))
        action_raw = _read(review, "action", "")
        try:
            action_name = ACTION_LABELS[ActionType(_value(action_raw))]
        except (TypeError, ValueError, KeyError):
            action_name = str(_value(action_raw))
        history_row = action_by_sequence.get(sequence)
        if history_row is not None:
            street_name = str(history_row.get("街道", street_name))
            action_name = str(history_row.get("简述", action_name))

        rating = str(_value(_read(review, "rating", _read(review, "grade", ""))))
        reason = str(_read(review, "reason", ""))
        grade_class = _RATING_CSS.get(rating, "acceptable")
        pot_odds = _read(review, "pot_odds")
        equity = _read(review, "equity")
        outs = _read(review, "outs")
        hit_probability = _read(review, "hit_probability")
        numbers: list[str] = []
        if pot_odds is not None:
            numbers.append(f"所需胜率 {float(pot_odds) * 100:.1f}%")
        if equity is not None:
            numbers.append(f"估算权益 {float(equity) * 100:.1f}%")
        if pot_odds is not None and equity is not None:
            edge = (float(equity) - float(pot_odds)) * 100
            numbers.append(f"权益差 {edge:+.1f}pct")
        if outs is not None:
            numbers.append(f"{int(outs)} outs")
        if hit_probability is not None:
            numbers.append(f"命中率 {float(hit_probability) * 100:.1f}%")
        evidence = ""
        if numbers:
            evidence = (
                '<details class="review-evidence"><summary>查看判断依据</summary>'
                f'<div>{escape(" · ".join(numbers))}</div></details>'
            )
        parts.append(
            f'<article class="feedback-card grade-{grade_class}">'
            '<div class="feedback-head">'
            f'<span class="decision">{escape(street_name)} · 你{escape(action_name)}</span>'
            f'<span class="rating">{escape(rating)}</span>'
            "</div>"
            f'<div class="reason">{escape(reason)}</div>'
            f"{evidence}"
            "</article>"
        )
    parts.append("</div>")
    return "".join(parts)


def hero_net_result(hand: Any, hero_id: str) -> int:
    """返回英雄本手结束筹码减去开局筹码。"""

    initial = next(
        (
            int(_read(seat, "stack", 0) or 0)
            for seat in (_read(hand, "initial_seats", ()) or ())
            if str(_read(seat, "player_id", "")) == hero_id
        ),
        0,
    )
    players = _read(hand, "players", {}) or {}
    player = players.get(hero_id) if isinstance(players, Mapping) else next(
        (row for row in players if str(_read(row, "player_id", "")) == hero_id),
        None,
    )
    return int(_read(player, "stack", initial) or 0) - initial


def rebuild_replay(bundle_or_json: ReplayBundle | str, action_count: int) -> Any:
    """按动作步数重建回放牌局，不重新调用对手策略。"""

    bundle = (
        ReplayBundle.from_json(bundle_or_json)
        if isinstance(bundle_or_json, str)
        else bundle_or_json
    )
    return bundle.replay(action_count=int(action_count))


def list_saved_hands(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """从 SQLite 列出可回放手牌，不读取底牌或隐藏对手参数。"""

    from poker_trainer.analytics.database import SQLiteStore

    with SQLiteStore(db_path) as store:
        rows = store.connection.execute(
            """
            SELECT hand_id, session_id, hand_no, seed, mode, final_street,
                   completed, result_reason, saved_at
            FROM hands
            ORDER BY saved_at DESC, hand_no DESC, hand_id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def load_replay_bundle(
    hand_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[ReplayBundle, str]:
    """从 SQLite 读取确定性回放包和本场英雄 ID。"""

    from poker_trainer.analytics.database import SQLiteStore

    with SQLiteStore(db_path) as store:
        bundle = ReplayBundle.from_json(store.load_replay_json(hand_id))
        session = store.get_session(bundle.session_id) if bundle.session_id else None
        hero_id = str(_read(session, "hero_player_id", "hero"))
    return bundle, hero_id


def load_saved_reviews(
    hand_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """读取一手牌按动作序号保存的英雄教练评价。"""

    from poker_trainer.analytics.database import SQLiteStore

    with SQLiteStore(db_path) as store:
        rows = store.load_decision_reviews(hand_id)
    reviews: list[dict[str, Any]] = []
    fallback_fields = (
        "player_id",
        "rating",
        "reason",
        "recommended_action",
        "pot_odds",
        "equity",
        "outs",
        "hit_probability",
    )
    for row in rows:
        payload = dict(row.get("review") or {})
        # decision_reviews.action_sequence 是关联 actions.sequence 的权威字段；
        # 兼容旧 review_json 未保存 sequence 或内部值不一致的记录。
        payload["sequence"] = int(row["action_sequence"])
        for field in fallback_fields:
            if payload.get(field) is None and row.get(field) is not None:
                payload[field] = row[field]
        reviews.append(payload)
    return reviews


def _get_streamlit() -> Any:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - 只在本地启动失败时走到
        raise RuntimeError(
            "未安装 Streamlit。请先执行 `pip install -e .`，"
            "再运行 `streamlit run app.py`。"
        ) from exc
    return st


def _rerun(st: Any) -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
    rerun()


def _mode_key(trainer: Any) -> str:
    return str(_value(_read(trainer, "mode", "test"))).strip().lower()


def _create_training_session(*, mode: str, seed: int, auto_top_up: bool, db_path: Path) -> Any:
    """延迟导入会话层，保持纯 helper 不依赖完整应用环境。"""

    from poker_trainer.training.session import SessionConfig, TrainingSession

    config = SessionConfig(
        mode=mode,
        seed=int(seed),
        auto_top_up=bool(auto_top_up),
        small_blind=20,
        big_blind=40,
        buy_in=4000,
        chips_per_yuan=CHIPS_PER_YUAN,
    )
    trainer = TrainingSession(config=config, db_path=db_path)
    trainer.start_hand()
    return trainer


def _notice_html() -> str:
    return (
        '<details class="notice-strip"><summary>使用说明</summary><div class="notice-body">'
        '<strong>使用边界：</strong>离线模拟和牌后复盘；不连接真实牌局，不做实时读屏提示。'
        '<br><strong>方法边界：</strong>MVP 使用启发式、底池赔率与蒙特卡洛权益，'
        '不是完整 GTO 求解器，也不以最终输赢倒推动作好坏。</div></details>'
    )


def _render_setup(st: Any) -> None:
    st.subheader("训练设置")
    with st.form("session_settings"):
        mode_label = st.selectbox(
            "模式",
            ("测试模式（牌局中不提示）", "教学模式（每次动作后即时反馈）"),
            help="测试模式在一手结束后统一复盘。",
        )
        seed = st.number_input(
            "训练随机种子",
            min_value=0,
            max_value=2_147_483_647,
            value=20260828,
            step=1,
            help="相同种子和动作日志可完整回放。",
        )
        auto_top_up = st.toggle(
            "每手结束后自动补回 100bb",
            value=True,
        )
        st.caption("固定规格：6人桌 · 盲注 20/40 · 默认 4,000 筹码（100bb） · 100筹码=¥1")
        submitted = st.form_submit_button("开始离线训练", type="primary", width="stretch")

    if submitted:
        previous = st.session_state.get("trainer")
        if previous is not None:
            try:
                previous.close()
            except Exception:
                pass
        mode = "test" if mode_label.startswith("测试") else "teaching"
        try:
            trainer = _create_training_session(
                mode=mode,
                seed=int(seed),
                auto_top_up=bool(auto_top_up),
                db_path=Path(st.session_state["db_path"]),
            )
        except Exception as exc:
            st.error(f"创建训练场次失败：{exc}")
            return
        st.session_state["trainer"] = trainer
        st.session_state["latest_reviews"] = ()
        # ``nav`` 已绑定本轮 radio；用下一轮待处理值避免 Streamlit
        # 禁止在控件实例化后直接改写其 session_state。
        st.session_state["_next_nav"] = "训练"
        _rerun(st)

    with st.expander("六个位置说明", expanded=False):
        for position in POSITION_ORDER:
            st.write(f"• {full_position_name(position)}")
    st.info("对手风格会在每个场次小幅漂移。界面只展示历史行动，不展示其隐藏参数或永久标签。")


def _summary_tiles_html(view: Mapping[str, Any], hero_stack: int) -> str:
    return (
        '<div class="table-summary">'
        f'<div class="summary-tile"><div class="label">街道</div><div class="value">{escape(str(view["street_name"]))}</div></div>'
        f'<div class="summary-tile"><div class="label">底池</div><div class="value">{int(view["pot"]):,}</div></div>'
        f'<div class="summary-tile"><div class="label">英雄后手</div><div class="value">{hero_stack:,}</div></div>'
        "</div>"
    )


def seat_grid_html(view: Mapping[str, Any]) -> str:
    """生成座位卡，并保留每位玩家最近一次自愿动作及金额。"""

    latest_by_player = {
        str(row["玩家"]): row
        for row in view.get("history", ())
        if not bool(row.get("强制", False))
    }
    cells: list[str] = []
    for seat in view["seats"]:
        classes = ["seat-card"]
        if seat["is_hero"]:
            classes.append("hero")
        if seat["folded"]:
            classes.append("folded")
        if seat["player_id"] == view["current_actor_id"]:
            classes.append("acting")
        recent = latest_by_player.get(str(seat["player_id"]))
        if not seat["player_id"]:
            state = ""
        elif seat["player_id"] == view["current_actor_id"]:
            state = "轮到你" if seat["is_hero"] else "正在行动"
        elif recent is not None:
            actor = "你" if seat["is_hero"] else str(_value(seat["position"]))
            state = f'{actor}｜{recent["简述"]}'
        elif seat["folded"]:
            state = "已弃牌"
        elif seat["all_in"]:
            state = "已全下"
        else:
            state = ""
        cards = cards_html(
            seat["cards"],
            hidden=seat["cards_hidden"] and bool(seat["player_id"]),
        )
        display_name = str(seat["name"])
        if seat["is_hero"] and display_name.strip() not in {"你", "英雄"}:
            display_name = f"{display_name} · 你"
        cells.append(
            f'<div class="{" ".join(classes)}">'
            f'<div class="position">{escape(seat["position_name"])}</div>'
            f'<div class="name">{escape(display_name)}</div>'
            f'<div class="cards">{cards}</div>'
            f'<div class="stack">{escape(format_chips(seat["stack"]))}</div>'
            f'<div class="state">{escape(state)}</div>'
            "</div>"
        )
    return f'<div class="seat-grid">{"".join(cells)}</div>'


def _render_reviews(
    st: Any,
    reviews: Sequence[Any],
    *,
    heading: str,
    history: Iterable[Mapping[str, Any]] = (),
) -> None:
    if not reviews:
        return
    st.subheader(heading)
    st.markdown(
        review_cards_html(reviews, history=history),
        unsafe_allow_html=True,
    )
    st.caption("只按动作当时可见信息评价，不按本手输赢倒推。")


def _submit_hero_action(st: Any, trainer: Any, action: ActionType, amount: int | None) -> None:
    try:
        feedback = trainer.hero_action(action, amount)
    except Exception as exc:
        st.error(f"动作未执行：{exc}")
        return
    reviews = normalize_reviews(feedback)
    if reviews:
        st.session_state["latest_reviews"] = reviews
    _rerun(st)


def _render_action_dock(st: Any, trainer: Any, hand: Any) -> None:
    try:
        legal = hand.legal_actions(trainer.hero_id)
    except Exception as exc:
        st.warning(f"当前暂无可选动作：{exc}")
        return
    controls = legal_action_controls(legal)
    if not controls:
        st.warning("当前没有合法动作。")
        return

    try:
        dock = st.container(border=True, key="action_dock")
    except TypeError:  # Streamlit 1.40 早期细分版本兼容
        dock = st.container(border=True)
    with dock:
        st.markdown("### 你的动作")
        sizing = bet_to_range(legal, chip_unit=int(_read(hand, "small_blind", 20) or 20))
        target: int | None = None
        if sizing is not None:
            target = st.slider(
                "Bet-to（本街累计到）",
                min_value=sizing.minimum,
                max_value=sizing.maximum,
                value=sizing.default,
                step=sizing.step,
                key=f"bet_to_{_read(hand, 'hand_id', '')}_{_read(hand, 'sequence', 0)}",
                help="滑块下限已按最小合法加注计算，上限为当前有效后手。",
            )
            st.caption(
                f"合法范围 {format_chips(sizing.minimum)} — {format_chips(sizing.maximum)}"
            )

        for start in range(0, len(controls), 2):
            chunk = controls[start : start + 2]
            columns = st.columns(len(chunk))
            for column, control in zip(columns, chunk, strict=True):
                with column:
                    clicked = st.button(
                        control.label,
                        key=f"act_{_read(hand, 'hand_id', '')}_{_read(hand, 'sequence', 0)}_{control.action.value}",
                        width="stretch",
                        type="primary" if control.action in {ActionType.CHECK, ActionType.CALL} else "secondary",
                    )
                    if clicked:
                        amount = target if control.needs_amount else None
                        _submit_hero_action(st, trainer, control.action, amount)


def _render_result(st: Any, trainer: Any, hand: Any) -> None:
    result = _read(hand, "result")
    if result is None:
        return
    reason = _read(result, "reason", "")
    hero_id = str(_read(trainer, "hero_id", "hero"))
    payout = int((_read(result, "payouts", {}) or {}).get(hero_id, 0))
    net = hero_net_result(hand, hero_id)
    big_blind = max(1, int(_read(hand, "big_blind", 40) or 40))
    net_text = f"{net:+,} 筹码（{net / big_blind:+.1f}bb）"
    if net > 0:
        st.success(f"本手净结果 {net_text}")
    elif net < 0:
        st.warning(f"本手净结果 {net_text}")
    else:
        st.info(f"本手净结果 {net_text}")
    ending = "摊牌结算" if reason == "showdown" else "未摊牌结束"
    st.caption(f"{ending} · 结算返还/赢得 {format_chips(payout)}")

    pot_rows = result_pot_rows(hand)
    if pot_rows:
        with st.expander("主池、边池与摊牌明细", expanded=len(pot_rows) > 1):
            st.dataframe(pot_rows, width="stretch", hide_index=True)
            rank_rows = showdown_rank_rows(hand)
            if rank_rows:
                st.dataframe(rank_rows, width="stretch", hide_index=True)

    if _mode_key(trainer) in {"test", "测试", "测试模式"}:
        reviews = normalize_reviews(_read(trainer, "last_hand_reviews", ()))
        if not reviews:
            reviews = normalize_reviews(st.session_state.get("latest_reviews", ()))
        _render_reviews(
            st,
            reviews,
            heading="牌后复盘",
            history=action_history_rows(_read(hand, "history", ()) or ()),
        )
    else:
        # 最后一个英雄动作可能直接结束手牌；仍要展示该次即时反馈。
        reviews = normalize_reviews(st.session_state.get("latest_reviews", ()))
        _render_reviews(
            st,
            reviews[-1:],
            heading="即时反馈",
            history=action_history_rows(_read(hand, "history", ()) or ()),
        )

    if st.button("下一手", type="primary", width="stretch"):
        try:
            trainer.start_hand()
        except Exception as exc:
            st.error(f"无法开始下一手：{exc}")
            return
        st.session_state["latest_reviews"] = ()
        _rerun(st)


def _render_training(st: Any) -> None:
    trainer = st.session_state.get("trainer")
    if trainer is None:
        st.info("请先在「设置」中创建一个离线训练场次。")
        if st.button("去设置", width="stretch"):
            st.session_state["_next_nav"] = "设置"
            _rerun(st)
        return
    hand = _read(trainer, "current_hand")
    if hand is None:
        try:
            hand = trainer.start_hand()
        except Exception as exc:
            st.error(f"无法发牌：{exc}")
            return

    hero_id = str(_read(trainer, "hero_id", "hero"))
    if not _read(hand, "is_complete", False) and _read(hand, "current_actor_id") != hero_id:
        try:
            trainer.advance_bots()
            hand = trainer.current_hand
        except Exception as exc:
            st.error(f"对手行动失败：{exc}")
            return

    view = hand_public_view(hand, hero_id)
    mode_name = "教学模式" if _mode_key(trainer) in {"teaching", "teach", "教学", "教学模式"} else "测试模式"
    st.subheader(f"第 {view['hand_no']} 手 · {mode_name}")
    st.caption(f"手牌 seed：{view['seed']} · 20/40 · 100bb")
    hero_seat = next((seat for seat in view["seats"] if seat["is_hero"]), None)
    hero_stack = int(hero_seat["stack"] if hero_seat else 0)
    st.markdown(_summary_tiles_html(view, hero_stack), unsafe_allow_html=True)
    st.markdown(
        '<div class="board-zone">'
        f'<div class="street">{escape(view["street_name"])}</div>'
        f'<div class="cards">{cards_html(view["board"])}</div>'
        f'<div class="pot">底池 {escape(format_chips(view["pot"]))}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(seat_grid_html(view), unsafe_allow_html=True)

    voluntary_rows = [row for row in view["history"] if not row["强制"]]
    display_street = view["street_name"]
    if view["is_complete"] and voluntary_rows:
        display_street = voluntary_rows[-1]["街道"]
    current_street_rows = [
        row
        for row in voluntary_rows
        if row["街道"] == display_street
    ]
    st.markdown("#### 最后行动" if view["is_complete"] else "#### 本街行动")
    if current_street_rows:
        st.markdown(
            action_timeline_html(current_street_rows, hero_id=hero_id, limit=6),
            unsafe_allow_html=True,
        )
    elif display_street == Street.PREFLOP.label_zh:
        forced_rows = [row for row in view["history"] if row["强制"]]
        st.markdown(
            action_timeline_html(forced_rows, hero_id=hero_id, limit=2),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(action_timeline_html((), hero_id=hero_id), unsafe_allow_html=True)
    if len(view["history"]) > len(current_street_rows):
        with st.expander(f"完整行动（{len(view['history'])}）", expanded=False):
            st.markdown(
                action_timeline_html(view["history"], hero_id=hero_id),
                unsafe_allow_html=True,
            )

    teaching = _mode_key(trainer) in {"teaching", "teach", "教学", "教学模式"}
    if teaching and not view["is_complete"]:
        reviews = normalize_reviews(st.session_state.get("latest_reviews", ()))
        if reviews:
            _render_reviews(
                st,
                reviews[-1:],
                heading="即时反馈",
                history=view["history"],
            )
    elif not teaching and not view["is_complete"]:
        st.caption("测试模式：牌局中不提示动作评价，本手结束后统一复盘。")

    if view["is_complete"]:
        _render_result(st, trainer, hand)
    elif view["current_actor_id"] == hero_id:
        _render_action_dock(st, trainer, hand)
    else:
        st.info("对手行动中……")


def _statistics_mapping(trainer: Any) -> Any:
    value = _read(trainer, "position_statistics", {})
    return value() if callable(value) else value


def _render_profile(st: Any, trainer: Any) -> None:
    profile = _read(trainer, "latest_profile")
    if profile is None:
        completed = int(_read(trainer, "completed_hand_count", 0) or 0)
        st.info(f"已完成 {completed} 手。每 20 手生成一次玩家画像；当前信号不足。")
        return
    st.subheader(f"玩家画像 · 至第 {int(_read(profile, 'through_hand_no', 0))} 手")
    findings = tuple(_read(profile, "findings", _read(profile, "leaks", ())) or ())
    if not findings:
        st.info(str(_read(profile, "message_zh", "信号不足")))
        return
    for index, finding in enumerate(findings[:3], start=1):
        title = str(_read(finding, "title_zh", _read(finding, "code", "待改进项")))
        reason = str(_read(finding, "reason", ""))
        evidence = int(_read(finding, "evidence_n", 0) or 0)
        severity = float(_read(finding, "severity", 0.0) or 0.0)
        st.markdown(f"**{index}. {title}** · 严重度 {severity * 100:.0f}% · 样本 {evidence}")
        st.caption(reason)
    st.caption("下一轮会增加相应场景，同时保留至少 55% 随机牌局。")


def _render_statistics(st: Any) -> None:
    trainer = st.session_state.get("trainer")
    if trainer is None:
        st.info("开始训练后，这里会按位置累计 13 项指标。")
        return
    statistics = _statistics_mapping(trainer)
    rows = statistics_table_rows(statistics)
    st.subheader("按位置统计")
    st.caption("百分比均用累计分子/分母计算；零机会不硬算，显示「信号不足」。")
    st.dataframe(rows, width="stretch", hide_index=True)

    selected = st.selectbox("查看单个位置", [full_position_name(p) for p in POSITION_ORDER])
    position = POSITION_ORDER[[full_position_name(p) for p in POSITION_ORDER].index(selected)]
    selected_row = None
    if isinstance(statistics, Mapping):
        selected_row = statistics.get(position, statistics.get(position.value))
    if selected_row is None:
        st.info("该位置尚无完成手牌，信号不足。")
    else:
        st.dataframe(metric_detail_rows(selected_row), width="stretch", hide_index=True)
    _render_profile(st, trainer)


def _hand_option_label(row: Mapping[str, Any]) -> str:
    mode = "教学" if str(row.get("mode")) == "teaching" else "测试"
    status = "已结束" if bool(row.get("completed")) else "未结束"
    saved_at = str(row.get("saved_at", "")).replace("T", " ")
    timestamp = saved_at[5:19] if len(saved_at) >= 19 else ""
    prefix = f"{timestamp} · " if timestamp else ""
    return f"{prefix}第 {row.get('hand_no', 0)} 手 · {mode} · {status}"


def _render_replay(st: Any) -> None:
    st.subheader("确定性回放")
    st.caption("回放直接使用 SQLite 中的洗牌 seed、牌序和动作日志，不重新采样对手决策。")
    db_path = Path(st.session_state["db_path"])
    try:
        hands = list_saved_hands(db_path)
    except Exception as exc:
        st.error(f"无法读取本地数据库：{exc}")
        return
    if not hands:
        st.info("本地 SQLite 中还没有可回放手牌。")
        return
    options = {str(row["hand_id"]): row for row in hands}
    selected_id = st.selectbox(
        "选择手牌",
        list(options),
        format_func=lambda hand_id: _hand_option_label(options[hand_id]),
    )
    try:
        bundle, hero_id = load_replay_bundle(selected_id, db_path)
    except Exception as exc:
        st.error(f"回放包读取失败：{exc}")
        return
    try:
        saved_reviews = load_saved_reviews(selected_id, db_path)
    except Exception:
        saved_reviews = []

    max_steps = len(bundle.actions)
    step_key = f"replay_step_{selected_id}"
    current_step = max(0, min(max_steps, int(st.session_state.get(step_key, max_steps))))
    st.session_state[step_key] = current_step
    try:
        replay_navigation = st.container(key="replay_nav")
    except TypeError:  # Streamlit 旧版不支持 container key
        replay_navigation = st.container()
    with replay_navigation:
        previous_column, counter_column, next_column = st.columns([1, 0.65, 1])
        with previous_column:
            if st.button(
                "← 上一步",
                key=f"replay_prev_{selected_id}",
                width="stretch",
                disabled=current_step <= 0,
            ):
                st.session_state[step_key] = current_step - 1
                _rerun(st)
        with counter_column:
            st.markdown(
                f'<div class="replay-counter">{current_step}/{max_steps}</div>',
                unsafe_allow_html=True,
            )
        with next_column:
            if st.button(
                "下一步 →",
                key=f"replay_next_{selected_id}",
                width="stretch",
                disabled=current_step >= max_steps,
            ):
                st.session_state[step_key] = current_step + 1
                _rerun(st)
    with st.expander("精确跳转", expanded=False):
        step = st.slider(
            "动作步数",
            min_value=0,
            max_value=max_steps,
            step=1,
            key=step_key,
            help="0 表示只重建发牌与盲注；之后每步增加一个自愿动作。",
        )
    try:
        replayed = rebuild_replay(bundle, step)
    except Exception as exc:
        st.error(f"重建失败：{exc}")
        return
    view = hand_public_view(replayed, hero_id)
    active_sequence = bundle.actions[step - 1].expected_sequence if step else None
    active_row = next(
        (row for row in view["history"] if int(row["#"]) == active_sequence),
        None,
    )
    if active_row is None:
        step_summary = "开局：小盲和大盲已入池"
    else:
        actor = "你" if active_row["玩家"] == hero_id else active_row["位置简称"]
        step_summary = f'{active_row["街道"]} · {actor}｜{active_row["简述"]}'
    st.markdown(
        f'<div class="replay-now"><span>当前动作</span>{escape(step_summary)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"确定性重建 · seed {bundle.seed}")
    hero_seat = next((seat for seat in view["seats"] if seat["is_hero"]), None)
    st.markdown(_summary_tiles_html(view, int(hero_seat["stack"] if hero_seat else 0)), unsafe_allow_html=True)
    st.markdown(
        '<div class="board-zone">'
        f'<div class="street">{escape(view["street_name"])}</div>'
        f'<div class="cards">{cards_html(view["board"])}</div>'
        f'<div class="pot">底池 {escape(format_chips(view["pot"]))}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(seat_grid_html(view), unsafe_allow_html=True)
    st.markdown("#### 按街行动历史")
    st.markdown(
        action_timeline_html(
            view["history"],
            hero_id=hero_id,
            active_sequence=active_sequence,
        ),
        unsafe_allow_html=True,
    )

    if view["is_complete"]:
        net = hero_net_result(replayed, hero_id)
        big_blind = max(1, int(_read(replayed, "big_blind", 40) or 40))
        net_text = f"{net:+,} 筹码（{net / big_blind:+.1f}bb）"
        if net > 0:
            st.success(f"本手净结果 {net_text}")
        elif net < 0:
            st.warning(f"本手净结果 {net_text}")
        else:
            st.info(f"本手净结果 {net_text}")
        pot_rows = result_pot_rows(replayed)
        rank_rows = showdown_rank_rows(replayed)
        if pot_rows or rank_rows:
            with st.expander("结算细节", expanded=False):
                if pot_rows:
                    st.dataframe(pot_rows, width="stretch", hide_index=True)
                if rank_rows:
                    st.dataframe(rank_rows, width="stretch", hide_index=True)

    reached_reviews = replay_reviews_through_sequence(saved_reviews, active_sequence)
    if reached_reviews:
        _render_reviews(
            st,
            reached_reviews,
            heading="你的决策复盘",
            history=view["history"],
        )
    elif step == max_steps:
        st.caption("本手没有可展示的英雄决策评价。")


def main() -> None:
    """Streamlit 入口。"""

    st = _get_streamlit()
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="♠",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    apply_styles(st)
    st.session_state.setdefault("trainer", None)
    st.session_state.setdefault("latest_reviews", ())
    st.session_state.setdefault("db_path", str(DEFAULT_DB_PATH))
    st.session_state.setdefault("nav", "设置")
    next_nav = st.session_state.pop("_next_nav", None)
    if next_nav in {"设置", "训练", "统计", "回放"}:
        st.session_state["nav"] = next_nav

    st.title(APP_TITLE)
    st.markdown(_notice_html(), unsafe_allow_html=True)
    page = st.radio(
        "页面导航",
        ("设置", "训练", "统计", "回放"),
        horizontal=True,
        label_visibility="collapsed",
        key="nav",
    )
    if page == "设置":
        _render_setup(st)
    elif page == "训练":
        _render_training(st)
    elif page == "统计":
        _render_statistics(st)
    else:
        _render_replay(st)

    st.divider()
    st.caption(PRODUCT_NOTICE)


__all__ = [
    "ACTION_LABELS",
    "APP_TITLE",
    "ActionControl",
    "BetToRange",
    "CHIPS_PER_YUAN",
    "DEFAULT_DB_PATH",
    "POSITION_ORDER",
    "PRODUCT_NOTICE",
    "action_timeline_html",
    "action_history_rows",
    "bet_to_range",
    "cards_html",
    "format_card",
    "format_cards",
    "format_chips",
    "format_metric",
    "format_percentage",
    "full_position_name",
    "hand_public_view",
    "hero_net_result",
    "legal_action_controls",
    "list_saved_hands",
    "load_replay_bundle",
    "load_saved_reviews",
    "main",
    "metric_detail_rows",
    "normalize_reviews",
    "replay_reviews_through_sequence",
    "rebuild_replay",
    "review_cards_html",
    "result_pot_rows",
    "showdown_rank_rows",
    "seat_grid_html",
    "statistics_table_rows",
]
