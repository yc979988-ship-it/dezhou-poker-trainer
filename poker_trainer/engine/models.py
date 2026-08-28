"""牌局状态机使用的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .cards import Card


class Position(str, Enum):
    UTG = "UTG"
    HJ = "HJ"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"

    @property
    def label_zh(self) -> str:
        return {
            Position.UTG: "前位",
            Position.HJ: "中位",
            Position.CO: "后位，按钮前一位",
            Position.BTN: "按钮位，位置最好",
            Position.SB: "小盲",
            Position.BB: "大盲",
        }[self]

    @property
    def display_name(self) -> str:
        return f"{self.value}（{self.label_zh}）"


TABLE_ORDER = (
    Position.SB,
    Position.BB,
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
)
PREFLOP_ORDER = (
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
    Position.SB,
    Position.BB,
)
POSTFLOP_ORDER = TABLE_ORDER


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETE = "complete"

    @property
    def label_zh(self) -> str:
        return {
            Street.PREFLOP: "翻前",
            Street.FLOP: "翻牌",
            Street.TURN: "转牌",
            Street.RIVER: "河牌",
            Street.SHOWDOWN: "摊牌",
            Street.COMPLETE: "已结束",
        }[self]


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_SB = "post_sb"
    POST_BB = "post_bb"
    REFUND = "refund"

    @property
    def label_zh(self) -> str:
        return {
            ActionType.FOLD: "弃牌",
            ActionType.CHECK: "过牌",
            ActionType.CALL: "跟注",
            ActionType.BET: "下注",
            ActionType.RAISE: "加注",
            ActionType.ALL_IN: "全下",
            ActionType.POST_SB: "下小盲",
            ActionType.POST_BB: "下大盲",
            ActionType.REFUND: "退回未跟注筹码",
        }[self]


@dataclass(frozen=True, slots=True)
class Seat:
    player_id: str
    name: str
    position: Position
    stack: int = 4000

    def __post_init__(self) -> None:
        if not self.player_id:
            raise ValueError("player_id 不能为空")
        if self.stack < 0:
            raise ValueError("筹码不能为负数")


@dataclass(slots=True)
class PlayerState:
    player_id: str
    name: str
    position: Position
    stack: int
    starting_stack: int
    hole_cards: list[Card] = field(default_factory=list)
    street_commitment: int = 0
    total_commitment: int = 0
    folded: bool = False
    all_in: bool = False
    payout: int = 0

    @property
    def live(self) -> bool:
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in and self.stack > 0


@dataclass(frozen=True, slots=True)
class LegalActions:
    player_id: str
    to_call: int
    call_amount: int
    pot_before: int
    min_bet_to: int | None
    min_raise_to: int | None
    max_to: int
    can_fold: bool
    can_check: bool
    can_call: bool
    can_bet: bool
    can_raise: bool
    can_all_in: bool
    raise_reopened: bool

    @property
    def action_types(self) -> tuple[ActionType, ...]:
        result: list[ActionType] = []
        for enabled, action in (
            (self.can_fold, ActionType.FOLD),
            (self.can_check, ActionType.CHECK),
            (self.can_call, ActionType.CALL),
            (self.can_bet, ActionType.BET),
            (self.can_raise, ActionType.RAISE),
            (self.can_all_in, ActionType.ALL_IN),
        ):
            if enabled:
                result.append(action)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    sequence: int
    street: Street
    player_id: str
    position: Position
    stack: int
    street_commitment: int
    total_commitment: int
    pot_before: int
    current_bet: int
    to_call: int
    min_raise_to: int | None
    active_players: int
    board: tuple[Card, ...]
    hole_cards: tuple[Card, ...]
    preflop_raise_count: int
    last_aggressor_id: str | None


@dataclass(frozen=True, slots=True)
class ActionRecord:
    sequence: int
    street: Street
    player_id: str
    position: Position
    action: ActionType
    requested_amount: int | None
    paid: int
    bet_to: int
    pot_before: int
    pot_after: int
    to_call_before: int
    current_bet_before: int
    current_bet_after: int
    min_raise_to_before: int | None
    is_all_in: bool
    is_full_raise: bool
    forced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "street": self.street.value,
            "player_id": self.player_id,
            "position": self.position.value,
            "action": self.action.value,
            "requested_amount": self.requested_amount,
            "paid": self.paid,
            "bet_to": self.bet_to,
            "pot_before": self.pot_before,
            "pot_after": self.pot_after,
            "to_call_before": self.to_call_before,
            "current_bet_before": self.current_bet_before,
            "current_bet_after": self.current_bet_after,
            "min_raise_to_before": self.min_raise_to_before,
            "is_all_in": self.is_all_in,
            "is_full_raise": self.is_full_raise,
            "forced": self.forced,
        }


@dataclass(frozen=True, slots=True)
class Pot:
    amount: int
    cap: int
    contributors: tuple[str, ...]
    eligible: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HandResult:
    reason: str
    board: tuple[Card, ...]
    pots: tuple[Pot, ...]
    payouts: dict[str, int]
    hand_ranks: dict[str, str]


