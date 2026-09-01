"""6-max 100bb 翻前启发式策略矩阵。

这是面向新手和偏松朋友局的可解释默认策略，不是完整 GTO 解。模块只处理
动作范围；实际 bet-to 尺度仍由教练上下文按 limper、位置和有效筹码计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poker_trainer.engine.cards import Card
from poker_trainer.engine.models import Position


class StrategicAction(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"

    @property
    def label_zh(self) -> str:
        return {
            StrategicAction.FOLD: "弃牌",
            StrategicAction.CHECK: "过牌",
            StrategicAction.CALL: "跟注",
            StrategicAction.RAISE: "加注",
        }[self]


class ActionRole(str, Enum):
    PRIMARY = "primary"
    ACCEPTABLE = "acceptable"
    MIXED = "mixed"
    DISCOURAGED = "discouraged"
    ERROR = "error"


class PreflopSpot(str, Enum):
    CHECK_OPTION = "check_option"
    UNOPENED = "unopened"
    LIMPED = "limped"
    FACE_OPEN = "face_open"
    SQUEEZE = "squeeze"
    FACE_RERAISE = "face_reraise"

    @property
    def label_zh(self) -> str:
        return {
            PreflopSpot.CHECK_OPTION: "大盲免费看牌",
            PreflopSpot.UNOPENED: "无人入池开池",
            PreflopSpot.LIMPED: "limp 底池",
            PreflopSpot.FACE_OPEN: "面对一次加注",
            PreflopSpot.SQUEEZE: "加注后已有跟注者",
            PreflopSpot.FACE_RERAISE: "面对再加注",
        }[self]


@dataclass(frozen=True, slots=True)
class HandShape:
    key: str
    high: int
    low: int
    pair: bool
    suited: bool


@dataclass(frozen=True, slots=True)
class PreflopSituation:
    hole_cards: tuple[Card, Card]
    position: Position
    raise_count: int
    limpers: int
    callers_after_raise: int = 0
    opener_position: Position | None = None
    effective_stack_bb: float = 100.0
    can_check: bool = False

    @property
    def spot(self) -> PreflopSpot:
        if self.can_check:
            return PreflopSpot.CHECK_OPTION
        if self.raise_count == 0:
            return PreflopSpot.LIMPED if self.limpers else PreflopSpot.UNOPENED
        if self.raise_count == 1:
            return (
                PreflopSpot.SQUEEZE
                if self.callers_after_raise
                else PreflopSpot.FACE_OPEN
            )
        return PreflopSpot.FACE_RERAISE


@dataclass(frozen=True, slots=True)
class ActionOption:
    action: StrategicAction
    role: ActionRole
    explanation: str
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreflopPlan:
    spot: PreflopSpot
    hand_key: str
    position: Position
    options: tuple[ActionOption, ...]

    def __post_init__(self) -> None:
        actions = [option.action for option in self.options]
        if len(actions) != len(set(actions)):
            raise ValueError("同一翻前动作不能在计划中重复")
        if sum(option.role == ActionRole.PRIMARY for option in self.options) != 1:
            raise ValueError("翻前计划必须恰好有一个默认动作")

    @property
    def primary(self) -> ActionOption:
        return next(
            option for option in self.options if option.role == ActionRole.PRIMARY
        )

    def option_for(self, action: StrategicAction) -> ActionOption:
        return next(
            (
                option
                for option in self.options
                if option.action == action
            ),
            ActionOption(
                action,
                ActionRole.ERROR,
                "该动作不符合当前默认范围。",
            ),
        )

    @property
    def recommended_actions(self) -> tuple[ActionOption, ...]:
        return tuple(
            option
            for option in self.options
            if option.role in {ActionRole.PRIMARY, ActionRole.ACCEPTABLE}
        )


_RANK_TEXT = {
    14: "A",
    13: "K",
    12: "Q",
    11: "J",
    10: "T",
    9: "9",
    8: "8",
    7: "7",
    6: "6",
    5: "5",
    4: "4",
    3: "3",
    2: "2",
}


def hand_shape(cards: tuple[Card, Card]) -> HandShape:
    first, second = cards
    high, low = sorted((first.rank, second.rank), reverse=True)
    pair = high == low
    suited = first.suit == second.suit
    key = (
        f"{_RANK_TEXT[high]}{_RANK_TEXT[low]}"
        if pair
        else f"{_RANK_TEXT[high]}{_RANK_TEXT[low]}{'s' if suited else 'o'}"
    )
    return HandShape(key=key, high=high, low=low, pair=pair, suited=suited)


@dataclass(frozen=True, slots=True)
class _OpenProfile:
    pair_min: int
    suited_min: tuple[tuple[int, int], ...]
    offsuit_min: tuple[tuple[int, int], ...]


_OPEN_PROFILES: dict[Position, _OpenProfile] = {
    Position.UTG: _OpenProfile(
        6,
        ((14, 2), (13, 10), (12, 10), (11, 10), (10, 9), (9, 8), (8, 7)),
        ((14, 11), (13, 12)),
    ),
    Position.HJ: _OpenProfile(
        5,
        (
            (14, 2),
            (13, 9),
            (12, 9),
            (11, 9),
            (10, 9),
            (9, 8),
            (8, 7),
            (7, 6),
        ),
        ((14, 10), (13, 11), (12, 11)),
    ),
    Position.CO: _OpenProfile(
        2,
        (
            (14, 2),
            (13, 7),
            (12, 8),
            (11, 8),
            (10, 8),
            (9, 7),
            (8, 6),
            (7, 5),
            (6, 5),
        ),
        ((14, 9), (13, 10), (12, 10), (11, 10)),
    ),
    Position.BTN: _OpenProfile(
        2,
        (
            (14, 2),
            (13, 4),
            (12, 6),
            (11, 7),
            (10, 7),
            (9, 6),
            (8, 6),
            (7, 5),
            (6, 4),
            (5, 4),
        ),
        ((14, 2), (13, 8), (12, 9), (11, 9), (10, 9), (9, 8)),
    ),
    Position.SB: _OpenProfile(
        2,
        (
            (14, 2),
            (13, 5),
            (12, 7),
            (11, 7),
            (10, 7),
            (9, 6),
            (8, 6),
            (7, 5),
            (6, 4),
            (5, 4),
        ),
        ((14, 2), (13, 8), (12, 9), (11, 9), (10, 9), (9, 8)),
    ),
}


def _matches_profile(hand: HandShape, profile: _OpenProfile) -> bool:
    if hand.pair:
        return hand.high >= profile.pair_min
    thresholds = profile.suited_min if hand.suited else profile.offsuit_min
    minimum = next((low for high, low in thresholds if high == hand.high), None)
    return minimum is not None and hand.low >= minimum


def _is_premium(hand: HandShape) -> bool:
    return bool(hand.pair and hand.high >= 12 or hand.high == 14 and hand.low == 13)


def _is_suited_wheel_ace(hand: HandShape) -> bool:
    return bool(hand.suited and hand.high == 14 and 2 <= hand.low <= 5)


def _is_suited_broadway(hand: HandShape) -> bool:
    return bool(hand.suited and hand.high >= 11 and hand.low >= 10)


def _is_speculative(hand: HandShape) -> bool:
    return bool(
        hand.pair
        or hand.suited
        and (
            hand.high == 14
            or hand.low >= 4 and hand.high - hand.low <= 2
            or hand.high >= 11 and hand.low >= 9
        )
    )


def _option(
    action: StrategicAction,
    role: ActionRole,
    explanation: str,
    *conditions: str,
) -> ActionOption:
    return ActionOption(action, role, explanation, tuple(conditions))


def _make_plan(
    situation: PreflopSituation,
    hand: HandShape,
    primary: ActionOption,
    *others: ActionOption,
) -> PreflopPlan:
    return PreflopPlan(
        spot=situation.spot,
        hand_key=hand.key,
        position=situation.position,
        options=(primary, *others),
    )


def _unopened_plan(situation: PreflopSituation, hand: HandShape) -> PreflopPlan:
    profile = _OPEN_PROFILES.get(situation.position)
    in_range = profile is not None and _matches_profile(hand, profile)
    if in_range:
        fold_role = ActionRole.ERROR if _is_premium(hand) else ActionRole.DISCOURAGED
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.RAISE,
                ActionRole.PRIMARY,
                f"{situation.position.value} 的默认开池范围包含 {hand.key}。",
            ),
            _option(
                StrategicAction.FOLD,
                fold_role,
                "弃牌会放弃可盈利的主动开池机会。",
            ),
            _option(
                StrategicAction.CALL,
                ActionRole.DISCOURAGED,
                "开放 limp 容易让范围被动；默认使用统一开池尺度。",
            ),
        )

    raise_role = (
        ActionRole.ERROR
        if situation.position in {Position.UTG, Position.HJ} or hand.key == "96o"
        else ActionRole.DISCOURAGED
    )
    return _make_plan(
        situation,
        hand,
        _option(
            StrategicAction.FOLD,
            ActionRole.PRIMARY,
            f"{hand.key} 不在 {situation.position.value} 的默认开池范围内。",
        ),
        _option(
            StrategicAction.RAISE,
            raise_role,
            "主动开池范围过宽，容易被更好牌跟注或反加。",
        ),
        _option(
            StrategicAction.CALL,
            ActionRole.DISCOURAGED,
            "不建议用开放 limp 代替弃牌。",
        ),
    )


def _isolation_candidate(
    situation: PreflopSituation,
    hand: HandShape,
) -> bool:
    blind = situation.position in {Position.SB, Position.BB}
    late = situation.position in {Position.CO, Position.BTN}
    multiway = situation.limpers >= 2
    if hand.pair:
        threshold = 9 if blind and multiway else 8 if multiway else 7 if blind else 6
        return hand.high >= threshold
    if hand.high == 14:
        if hand.suited:
            threshold = 11 if blind and multiway else 10 if multiway else 9
        else:
            threshold = 12 if blind and multiway else 11 if multiway else 10
        return hand.low >= threshold
    if hand.high == 13:
        return bool(
            hand.low >= (12 if multiway or blind else 10)
            and (hand.suited or hand.low >= 12)
        )
    if late and not multiway and hand.suited:
        return bool(
            hand.high >= 11 and hand.low >= 10
            or hand.high == 10 and hand.low == 9
        )
    if late and multiway and hand.suited:
        return hand.high >= 11 and hand.low >= 10
    return False


def _limped_plan(situation: PreflopSituation, hand: HandShape) -> PreflopPlan:
    if _isolation_candidate(situation, hand):
        passive_role = ActionRole.DISCOURAGED if _is_premium(hand) else ActionRole.ACCEPTABLE
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.RAISE,
                ActionRole.PRIMARY,
                f"{hand.key} 对 {situation.limpers} 名 limper 有足够价值进行隔离。",
            ),
            _option(
                StrategicAction.CALL,
                passive_role,
                "被动入池保留多人底池，但会减少主动取值。",
            ),
            _option(
                StrategicAction.FOLD,
                ActionRole.ERROR if _is_premium(hand) else ActionRole.DISCOURAGED,
                "这手牌强度足以继续。",
            ),
        )

    if _is_speculative(hand) and situation.effective_stack_bb >= 35:
        call_word = "补齐" if situation.position == Position.SB else "跟注"
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.CALL,
                ActionRole.PRIMARY,
                f"{hand.key} 适合用较低价格{call_word}，依靠隐含赔率。",
            ),
            _option(
                StrategicAction.FOLD,
                ActionRole.ACCEPTABLE,
                "不想进入多人底池时弃牌也可接受。",
            ),
            _option(
                StrategicAction.RAISE,
                ActionRole.DISCOURAGED,
                "这类牌更适合便宜看翻牌，不宜默认隔离多人。",
            ),
        )

    return _make_plan(
        situation,
        hand,
        _option(
            StrategicAction.FOLD,
            ActionRole.PRIMARY,
            f"{hand.key} 在该位置缺少隔离价值和翻后可玩性。",
        ),
        _option(
            StrategicAction.CALL,
            ActionRole.DISCOURAGED if situation.position == Position.SB else ActionRole.ERROR,
            "小盲极被动桌可低频补齐，但默认仍应弃牌。",
            "仅在价格很低且后位极少加注时考虑",
        ),
        _option(
            StrategicAction.RAISE,
            ActionRole.ERROR,
            "弱牌隔离容易被多人跟注，并长期处于范围和位置劣势。",
        ),
    )


def _is_value_threebet(
    situation: PreflopSituation,
    hand: HandShape,
) -> bool:
    late_open = situation.opener_position in {Position.CO, Position.BTN, Position.SB}
    if hand.pair:
        return hand.high >= (11 if late_open else 12)
    if hand.high == 14 and hand.low == 13:
        return True
    return bool(late_open and hand.key == "AQs")


def _is_mixed_threebet(
    situation: PreflopSituation,
    hand: HandShape,
) -> bool:
    if situation.callers_after_raise:
        return False
    if _is_suited_wheel_ace(hand) and situation.opener_position != Position.UTG:
        return True
    if hand.pair and hand.high in {10, 11}:
        return True
    if hand.key in {"AQs", "AJs", "KQs"}:
        return True
    return bool(
        situation.position == Position.SB
        and hand.suited
        and hand.high >= 11
        and hand.low >= 10
    )


def _is_call_candidate(
    situation: PreflopSituation,
    hand: HandShape,
) -> bool:
    position = situation.position
    if position == Position.SB:
        return bool(
            hand.pair and hand.high >= 8
            or hand.key in {"AQs", "AJs", "KQs"}
        )
    if hand.pair:
        minimum = 2 if position in {Position.BTN, Position.BB} else 5
        return hand.high >= minimum and situation.effective_stack_bb >= 30
    if hand.key in {
        "AQs",
        "AJs",
        "ATs",
        "KQs",
        "KJs",
        "QJs",
        "JTs",
        "T9s",
        "98s",
        "AQo",
        "AJo",
        "KQo",
    }:
        return True
    return bool(
        position == Position.BB
        and hand.suited
        and (hand.high == 14 or hand.low >= 6)
    )


def _facing_open_plan(situation: PreflopSituation, hand: HandShape) -> PreflopPlan:
    if _is_value_threebet(situation, hand):
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.RAISE,
                ActionRole.PRIMARY,
                f"{hand.key} 属于面对该开池位置的价值 3bet 范围。",
            ),
            _option(
                StrategicAction.CALL,
                ActionRole.DISCOURAGED if hand.high >= 13 else ActionRole.ACCEPTABLE,
                "平跟能隐藏牌力，但默认会损失价值并增加多人底池概率。",
            ),
            _option(
                StrategicAction.FOLD,
                ActionRole.ERROR,
                "强牌弃牌过紧。",
            ),
        )

    if _is_mixed_threebet(situation, hand):
        if hand.pair and hand.high in {10, 11}:
            primary, secondary = StrategicAction.CALL, StrategicAction.RAISE
        else:
            primary, secondary = StrategicAction.FOLD, StrategicAction.RAISE
        return _make_plan(
            situation,
            hand,
            _option(
                primary,
                ActionRole.PRIMARY,
                f"{hand.key} 在这里以{primary.label_zh}作为稳健默认。",
            ),
            _option(
                secondary,
                ActionRole.ACCEPTABLE,
                "可作为条件性 3bet；偏松朋友局应降低纯诈唬频率。",
                "对手开池较宽且弃牌足够时使用",
            ),
            _option(
                StrategicAction.CALL,
                ActionRole.DISCOURAGED
                if primary != StrategicAction.CALL
                else ActionRole.PRIMARY,
                "平跟会保留权益，但可能被后位挤压。",
            )
            if primary != StrategicAction.CALL
            else _option(
                StrategicAction.FOLD,
                ActionRole.ACCEPTABLE,
                "面对偏紧开池者可以收紧。",
            ),
        )

    if _is_call_candidate(situation, hand):
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.CALL,
                ActionRole.PRIMARY,
                f"{hand.key} 有足够可玩性继续，但不必默认扩大底池。",
            ),
            _option(
                StrategicAction.FOLD,
                ActionRole.ACCEPTABLE if hand.pair and hand.high <= 6 else ActionRole.DISCOURAGED,
                "对紧手或大尺度可以收紧。",
            ),
            _option(
                StrategicAction.RAISE,
                ActionRole.DISCOURAGED,
                "加注可以混合，但默认范围不宜过宽。",
            ),
        )

    raise_role = (
        ActionRole.ACCEPTABLE
        if situation.position == Position.SB and _is_suited_broadway(hand)
        else ActionRole.ERROR
    )
    call_role = (
        ActionRole.DISCOURAGED
        if situation.position in {Position.SB, Position.BB}
        else ActionRole.ERROR
    )
    return _make_plan(
        situation,
        hand,
        _option(
            StrategicAction.FOLD,
            ActionRole.PRIMARY,
            f"{hand.key} 面对该开池范围应默认弃牌。",
        ),
        _option(
            StrategicAction.RAISE,
            raise_role,
            "若作为条件性 3bet，必须有阻断、可玩性和足够弃牌率。",
        ),
        _option(
            StrategicAction.CALL,
            call_role,
            "跟注容易被更好牌压制；小盲还要承担位置差和 BB 挤压风险。",
        ),
    )


def _facing_reraise_plan(
    situation: PreflopSituation,
    hand: HandShape,
) -> PreflopPlan:
    if hand.pair and hand.high >= 13 or hand.high == 14 and hand.low == 13:
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.RAISE,
                ActionRole.PRIMARY,
                f"{hand.key} 属于继续做大底池的顶端范围。",
            ),
            _option(
                StrategicAction.CALL,
                ActionRole.ACCEPTABLE,
                "平跟可少量混合，但默认继续取值。",
            ),
            _option(StrategicAction.FOLD, ActionRole.ERROR, "顶端范围不能弃牌。"),
        )
    if hand.key in {"QQ", "JJ", "AKo", "AQs"}:
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.CALL,
                ActionRole.PRIMARY,
                "牌力足以继续，但应控制面对强范围的投入。",
            ),
            _option(
                StrategicAction.RAISE,
                ActionRole.ACCEPTABLE,
                "对宽 3bet 范围可以继续加注。",
            ),
            _option(
                StrategicAction.FOLD,
                ActionRole.DISCOURAGED,
                "除非对手范围极紧，否则弃牌偏紧。",
            ),
        )
    return _make_plan(
        situation,
        hand,
        _option(
            StrategicAction.FOLD,
            ActionRole.PRIMARY,
            f"{hand.key} 面对再加注应默认收紧。",
        ),
        _option(
            StrategicAction.CALL,
            ActionRole.ERROR,
            "3bet 底池的隐含赔率不足，不能只为中牌继续。",
        ),
        _option(
            StrategicAction.RAISE,
            ActionRole.ERROR,
            "继续加注会把过弱范围投入过多。",
        ),
    )


def build_preflop_plan(situation: PreflopSituation) -> PreflopPlan:
    """返回确定性的默认计划；混合动作只标条件，不伪造精确频率。"""

    hand = hand_shape(situation.hole_cards)
    if situation.spot == PreflopSpot.CHECK_OPTION:
        return _make_plan(
            situation,
            hand,
            _option(
                StrategicAction.CHECK,
                ActionRole.PRIMARY,
                "大盲可以免费看翻牌。",
            ),
            _option(
                StrategicAction.FOLD,
                ActionRole.ERROR,
                "可以免费看牌时不应弃牌。",
            ),
            _option(
                StrategicAction.RAISE,
                ActionRole.ACCEPTABLE if _isolation_candidate(situation, hand) else ActionRole.DISCOURAGED,
                "加注只适合有价值优势或明确隔离计划的牌。",
            ),
        )
    if situation.spot == PreflopSpot.UNOPENED:
        return _unopened_plan(situation, hand)
    if situation.spot == PreflopSpot.LIMPED:
        return _limped_plan(situation, hand)
    if situation.spot in {PreflopSpot.FACE_OPEN, PreflopSpot.SQUEEZE}:
        return _facing_open_plan(situation, hand)
    return _facing_reraise_plan(situation, hand)


__all__ = [
    "ActionOption",
    "ActionRole",
    "HandShape",
    "PreflopPlan",
    "PreflopSituation",
    "PreflopSpot",
    "StrategicAction",
    "build_preflop_plan",
    "hand_shape",
]
