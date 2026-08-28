"""牌局回放序列化；回放只重放已记录动作，不重新调用对手策略。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .hand import HoldemHand
from .models import ActionType, Seat


@dataclass(frozen=True, slots=True)
class ReplayAction:
    player_id: str
    action: ActionType
    amount: int | None
    expected_sequence: int


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    hand_id: str
    session_id: str | None
    hand_no: int
    seed: int
    small_blind: int
    big_blind: int
    seats: tuple[Seat, ...]
    deck_order: tuple[str, ...]
    hole_overrides: dict[str, tuple[str, ...]]
    board_override: tuple[str, ...]
    scenario_id: str | None
    actions: tuple[ReplayAction, ...]
    engine_version: str
    rules_version: str

    @classmethod
    def from_hand(cls, hand: HoldemHand) -> "ReplayBundle":
        actions = tuple(
            ReplayAction(
                player_id=record.player_id,
                action=record.action,
                amount=record.requested_amount,
                expected_sequence=record.sequence,
            )
            for record in hand.history
            if not record.forced
        )
        return cls(
            hand_id=hand.hand_id,
            session_id=hand.session_id,
            hand_no=hand.hand_no,
            seed=hand.seed,
            small_blind=hand.small_blind,
            big_blind=hand.big_blind,
            seats=hand.initial_seats,
            deck_order=hand.full_deck_order,
            hole_overrides={
                player_id: tuple(str(card) for card in cards)
                for player_id, cards in hand.hole_overrides.items()
            },
            board_override=tuple(str(card) for card in hand.board_override),
            scenario_id=hand.scenario_id,
            actions=actions,
            engine_version=hand.engine_version,
            rules_version=hand.rules_version,
        )

    def replay(self, *, action_count: int | None = None) -> HoldemHand:
        hand = HoldemHand(
            self.seats,
            seed=self.seed,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            hand_id=self.hand_id,
            session_id=self.session_id,
            hand_no=self.hand_no,
            hole_overrides=self.hole_overrides,
            board_override=self.board_override,
            deck_order=self.deck_order,
            scenario_id=self.scenario_id,
        )
        limit = len(self.actions) if action_count is None else action_count
        if not 0 <= limit <= len(self.actions):
            raise ValueError("回放动作数超出范围")
        for command in self.actions[:limit]:
            hand.act(
                command.player_id,
                command.action,
                command.amount,
                expected_sequence=command.expected_sequence,
            )
        return hand

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "session_id": self.session_id,
            "hand_no": self.hand_no,
            "seed": self.seed,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "seats": [
                {
                    "player_id": seat.player_id,
                    "name": seat.name,
                    "position": seat.position.value,
                    "stack": seat.stack,
                }
                for seat in self.seats
            ],
            "deck_order": list(self.deck_order),
            "hole_overrides": {
                player_id: list(cards) for player_id, cards in self.hole_overrides.items()
            },
            "board_override": list(self.board_override),
            "scenario_id": self.scenario_id,
            "actions": [
                {
                    "player_id": action.player_id,
                    "action": action.action.value,
                    "amount": action.amount,
                    "expected_sequence": action.expected_sequence,
                }
                for action in self.actions
            ],
            "engine_version": self.engine_version,
            "rules_version": self.rules_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "ReplayBundle":
        from .models import Position

        data = json.loads(payload)
        return cls(
            hand_id=data["hand_id"],
            session_id=data.get("session_id"),
            hand_no=int(data["hand_no"]),
            seed=int(data["seed"]),
            small_blind=int(data["small_blind"]),
            big_blind=int(data["big_blind"]),
            seats=tuple(
                Seat(
                    player_id=item["player_id"],
                    name=item["name"],
                    position=Position(item["position"]),
                    stack=int(item["stack"]),
                )
                for item in data["seats"]
            ),
            deck_order=tuple(data["deck_order"]),
            hole_overrides={
                player_id: tuple(cards)
                for player_id, cards in data.get("hole_overrides", {}).items()
            },
            board_override=tuple(data.get("board_override", [])),
            scenario_id=data.get("scenario_id"),
            actions=tuple(
                ReplayAction(
                    player_id=item["player_id"],
                    action=ActionType(item["action"]),
                    amount=item.get("amount"),
                    expected_sequence=int(item["expected_sequence"]),
                )
                for item in data["actions"]
            ),
            engine_version=data["engine_version"],
            rules_version=data["rules_version"],
        )


