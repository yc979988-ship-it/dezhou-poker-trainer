"""单手6人桌无限注德州的确定性状态机。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from .cards import Card, Deck, parse_cards
from .evaluator import HandRank, evaluate
from .models import (
    POSTFLOP_ORDER,
    PREFLOP_ORDER,
    TABLE_ORDER,
    ActionRecord,
    ActionType,
    DecisionSnapshot,
    HandResult,
    LegalActions,
    PlayerState,
    Position,
    Seat,
    Street,
)
from .pots import build_pots


ENGINE_VERSION = "0.1.0"
RULES_VERSION = "nlhe-6max-v1"


class InvalidAction(ValueError):
    """动作不符合当前牌局状态。"""


class HoldemHand:
    """一手可回放的无限注德州扑克。

    `amount` 对 Bet/Raise 始终表示“本街累计下注到多少”（bet-to），
    而不是额外增加多少。所有随机性使用本手私有牌堆 RNG。
    """

    def __init__(
        self,
        seats: Sequence[Seat],
        *,
        seed: int,
        small_blind: int = 20,
        big_blind: int = 40,
        hand_id: str | None = None,
        session_id: str | None = None,
        hand_no: int = 1,
        hole_overrides: Mapping[str, Iterable[Card | str]] | None = None,
        board_override: Iterable[Card | str] | None = None,
        deck_order: Iterable[Card | str] | None = None,
        scenario_id: str | None = None,
    ) -> None:
        if not 2 <= len(seats) <= 6:
            raise ValueError("牌桌人数必须为2至6人")
        if small_blind <= 0 or big_blind <= small_blind:
            raise ValueError("盲注必须满足 0 < SB < BB")
        if len({seat.player_id for seat in seats}) != len(seats):
            raise ValueError("player_id 不能重复")
        if len({seat.position for seat in seats}) != len(seats):
            raise ValueError("位置不能重复")
        if Position.SB not in {seat.position for seat in seats}:
            raise ValueError("牌桌必须包含SB")
        if Position.BB not in {seat.position for seat in seats}:
            raise ValueError("牌桌必须包含BB")

        self.seed = int(seed)
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self.hand_id = hand_id or f"hand-{hand_no}-{self.seed}"
        self.session_id = session_id
        self.hand_no = hand_no
        self.scenario_id = scenario_id
        self.engine_version = ENGINE_VERSION
        self.rules_version = RULES_VERSION

        self.players: dict[str, PlayerState] = {
            seat.player_id: PlayerState(
                player_id=seat.player_id,
                name=seat.name,
                position=seat.position,
                stack=seat.stack,
                starting_stack=seat.stack,
                all_in=seat.stack == 0,
            )
            for seat in seats
        }
        self._player_by_position = {
            player.position: player.player_id for player in self.players.values()
        }
        self.initial_seats = tuple(seats)
        self.initial_total_chips = sum(seat.stack for seat in seats)

        self.street = Street.PREFLOP
        self.board: list[Card] = []
        self.burn_cards: list[Card] = []
        self.current_bet = 0
        self.last_full_raise_size = self.big_blind
        self.last_aggressor_id: str | None = None
        self.preflop_raise_count = 0
        self.full_raise_epoch = 0
        self._player_action_epoch: dict[str, int] = {}
        self._last_action_bet_level: dict[str, int] = {}
        self.pending: set[str] = set()
        self.current_actor_id: str | None = None
        self.history: list[ActionRecord] = []
        self.decision_snapshots: dict[int, DecisionSnapshot] = {}
        self.result: HandResult | None = None

        raw_hole_overrides = hole_overrides or {}
        self.hole_overrides: dict[str, tuple[Card, ...]] = {
            player_id: tuple(parse_cards(cards))
            for player_id, cards in raw_hole_overrides.items()
        }
        for player_id, cards in self.hole_overrides.items():
            if player_id not in self.players:
                raise ValueError(f"底牌指定了未知玩家: {player_id}")
            if len(cards) != 2:
                raise ValueError("每名玩家必须恰好指定2张底牌")
        self.board_override = tuple(parse_cards(board_override or []))
        if len(self.board_override) not in (0, 3, 4, 5):
            raise ValueError("公共牌预设必须为0、3、4或5张")

        reserved = [card for cards in self.hole_overrides.values() for card in cards]
        reserved.extend(self.board_override)
        if len(set(reserved)) != len(reserved):
            raise ValueError("预设底牌和公共牌不能重复")

        if deck_order is None:
            self.deck = Deck(self.seed)
        else:
            supplied_deck = parse_cards(deck_order)
            if len(supplied_deck) != 52 or len(set(supplied_deck)) != 52:
                raise ValueError("完整牌序必须包含52张不重复的牌")
            self.deck = Deck(self.seed, shuffle=False)
            self.deck.cards = list(supplied_deck)
        self.full_deck_order = tuple(str(card) for card in self.deck.cards)
        reserved_set = set(reserved)
        self.deck.cards = [card for card in self.deck.cards if card not in reserved_set]
        self._board_override_index = 0

        self._deal_hole_cards()
        self._post_blinds()
        self.pending = {player.player_id for player in self.players.values() if player.can_act}
        self.current_actor_id = self._first_pending(PREFLOP_ORDER)
        self._advance_if_no_decision()

    @property
    def sequence(self) -> int:
        return len(self.history)

    @property
    def is_complete(self) -> bool:
        return self.result is not None

    @property
    def pot_size(self) -> int:
        if self.is_complete:
            return 0
        return sum(player.total_commitment for player in self.players.values())

    @property
    def committed_pot(self) -> int:
        """包括结算后历史投入在内的总投入。"""

        return sum(player.total_commitment for player in self.players.values())

    @property
    def live_players(self) -> list[PlayerState]:
        return [player for player in self.players.values() if player.live]

    def player(self, player_id: str) -> PlayerState:
        try:
            return self.players[player_id]
        except KeyError as exc:
            raise KeyError(f"未知玩家: {player_id}") from exc

    def _ordered_players(self, order: Sequence[Position]) -> list[PlayerState]:
        return [
            self.players[self._player_by_position[position]]
            for position in order
            if position in self._player_by_position
        ]

    def _deal_hole_cards(self) -> None:
        for player_id, cards in self.hole_overrides.items():
            self.players[player_id].hole_cards = list(cards)
        for _ in range(2):
            for player in self._ordered_players(TABLE_ORDER):
                if player.player_id not in self.hole_overrides:
                    player.hole_cards.append(self.deck.deal())

    def _draw_board_card(self) -> Card:
        if self._board_override_index < len(self.board_override):
            card = self.board_override[self._board_override_index]
            self._board_override_index += 1
            return card
        return self.deck.deal()

    def _deal_next_street(self, street: Street) -> None:
        self.burn_cards.append(self.deck.deal())
        count = 3 if street == Street.FLOP else 1
        self.board.extend(self._draw_board_card() for _ in range(count))

    def _pay(self, player: PlayerState, amount: int) -> int:
        if amount < 0:
            raise InvalidAction("投入筹码不能为负数")
        paid = min(player.stack, amount)
        player.stack -= paid
        player.street_commitment += paid
        player.total_commitment += paid
        if player.stack == 0:
            player.all_in = True
        return paid

    def _post_blind(self, position: Position, amount: int, action: ActionType) -> None:
        player = self.players[self._player_by_position[position]]
        pot_before = sum(p.total_commitment for p in self.players.values())
        before = self.current_bet
        paid = self._pay(player, amount)
        self.current_bet = max(self.current_bet, player.street_commitment)
        self.history.append(
            ActionRecord(
                sequence=len(self.history),
                street=Street.PREFLOP,
                player_id=player.player_id,
                position=player.position,
                action=action,
                requested_amount=amount,
                paid=paid,
                bet_to=player.street_commitment,
                pot_before=pot_before,
                pot_after=pot_before + paid,
                to_call_before=0,
                current_bet_before=before,
                current_bet_after=self.current_bet,
                min_raise_to_before=None,
                is_all_in=player.all_in,
                is_full_raise=False,
                forced=True,
            )
        )

    def _post_blinds(self) -> None:
        self._post_blind(Position.SB, self.small_blind, ActionType.POST_SB)
        self._post_blind(Position.BB, self.big_blind, ActionType.POST_BB)
        # 即使大盲短码全下，其他玩家仍面对名义完整大盲。
        self.current_bet = self.big_blind
        self.last_full_raise_size = self.big_blind

    def _first_pending(self, order: Sequence[Position]) -> str | None:
        for player in self._ordered_players(order):
            if player.player_id in self.pending and player.can_act:
                return player.player_id
        return None

    def _next_pending_after(self, player_id: str) -> str | None:
        ordered = self._ordered_players(TABLE_ORDER)
        start = next(index for index, player in enumerate(ordered) if player.player_id == player_id)
        for offset in range(1, len(ordered) + 1):
            candidate = ordered[(start + offset) % len(ordered)]
            if candidate.player_id in self.pending and candidate.can_act:
                return candidate.player_id
        return None

    def _raise_reopened(self, player_id: str) -> bool:
        if player_id not in self._player_action_epoch:
            return True
        if self._player_action_epoch[player_id] < self.full_raise_epoch:
            return True
        previous_level = self._last_action_bet_level[player_id]
        return self.current_bet - previous_level >= self.last_full_raise_size

    def _has_responder(self, player_id: str) -> bool:
        return any(
            other.player_id != player_id and other.can_act
            for other in self.players.values()
        )

    def legal_actions(self, player_id: str | None = None) -> LegalActions:
        if self.is_complete:
            raise InvalidAction("牌局已经结束")
        player_id = player_id or self.current_actor_id
        if player_id is None or player_id != self.current_actor_id:
            raise InvalidAction("当前未轮到该玩家行动")
        player = self.player(player_id)
        to_call = max(0, self.current_bet - player.street_commitment)
        max_to = player.street_commitment + player.stack
        reopened = self._raise_reopened(player_id)
        has_responder = self._has_responder(player_id)
        min_bet_to = self.big_blind if self.current_bet == 0 else None
        min_raise_to = (
            self.current_bet + self.last_full_raise_size if self.current_bet > 0 else None
        )
        can_raise = bool(
            self.current_bet > 0
            and reopened
            and has_responder
            and min_raise_to is not None
            and max_to >= min_raise_to
        )
        can_bet = bool(
            self.current_bet == 0
            and has_responder
            and max_to >= self.big_blind
        )
        all_in_target_is_raise = max_to > self.current_bet
        can_all_in = bool(
            player.stack > 0
            and (not all_in_target_is_raise or (reopened and has_responder))
        )
        return LegalActions(
            player_id=player_id,
            to_call=to_call,
            call_amount=min(to_call, player.stack),
            pot_before=sum(p.total_commitment for p in self.players.values()),
            min_bet_to=min_bet_to,
            min_raise_to=min_raise_to,
            max_to=max_to,
            can_fold=True,
            can_check=to_call == 0,
            can_call=to_call > 0 and player.stack > 0,
            can_bet=can_bet,
            can_raise=can_raise,
            can_all_in=can_all_in,
            raise_reopened=reopened,
        )

    def decision_snapshot(self, player_id: str | None = None) -> DecisionSnapshot:
        player_id = player_id or self.current_actor_id
        legal = self.legal_actions(player_id)
        player = self.player(legal.player_id)
        return DecisionSnapshot(
            sequence=self.sequence,
            street=self.street,
            player_id=player.player_id,
            position=player.position,
            stack=player.stack,
            street_commitment=player.street_commitment,
            total_commitment=player.total_commitment,
            pot_before=legal.pot_before,
            current_bet=self.current_bet,
            to_call=legal.to_call,
            min_raise_to=legal.min_raise_to,
            active_players=len(self.live_players),
            board=tuple(self.board),
            hole_cards=tuple(player.hole_cards),
            preflop_raise_count=self.preflop_raise_count,
            last_aggressor_id=self.last_aggressor_id,
        )

    def act(
        self,
        player_id: str,
        action: ActionType | str,
        amount: int | None = None,
        *,
        expected_sequence: int | None = None,
    ) -> ActionRecord:
        """应用一个原子动作；非法动作不会改变任何状态。"""

        if expected_sequence is not None and expected_sequence != self.sequence:
            raise InvalidAction("动作序号已过期，请刷新牌局状态")
        try:
            action = action if isinstance(action, ActionType) else ActionType(action)
        except ValueError as exc:
            raise InvalidAction(f"未知动作: {action}") from exc
        if action in (ActionType.POST_SB, ActionType.POST_BB, ActionType.REFUND):
            raise InvalidAction("该动作只能由引擎产生")

        legal = self.legal_actions(player_id)
        player = self.player(player_id)
        snapshot = self.decision_snapshot(player_id)
        pot_before = legal.pot_before
        current_bet_before = self.current_bet
        paid = 0
        is_full_raise = False
        increased_bet = False

        if action == ActionType.FOLD:
            player.folded = True
        elif action == ActionType.CHECK:
            if not legal.can_check:
                raise InvalidAction(f"仍需跟注 {legal.to_call}，不能过牌")
        elif action == ActionType.CALL:
            if not legal.can_call:
                raise InvalidAction("当前不能跟注")
            paid = self._pay(player, legal.call_amount)
        elif action == ActionType.BET:
            if not legal.can_bet:
                raise InvalidAction("当前不能下注")
            target = self._validated_target(amount, legal.min_bet_to, legal.max_to, "下注")
            paid = self._pay(player, target - player.street_commitment)
            self.current_bet = target
            self.last_full_raise_size = target
            is_full_raise = True
            increased_bet = True
        elif action == ActionType.RAISE:
            if not legal.can_raise:
                raise InvalidAction("当前不能完整加注")
            target = self._validated_target(amount, legal.min_raise_to, legal.max_to, "加注")
            raise_size = target - current_bet_before
            paid = self._pay(player, target - player.street_commitment)
            self.current_bet = target
            self.last_full_raise_size = raise_size
            is_full_raise = True
            increased_bet = True
        elif action == ActionType.ALL_IN:
            if not legal.can_all_in:
                raise InvalidAction("当前不能全下加注；行动尚未重新开放或无人可响应")
            target = legal.max_to
            paid = self._pay(player, player.stack)
            if target > current_bet_before:
                raise_size = target - current_bet_before
                increased_bet = True
                self.current_bet = target
                if current_bet_before == 0:
                    is_full_raise = target >= self.big_blind
                    if is_full_raise:
                        self.last_full_raise_size = target
                else:
                    is_full_raise = raise_size >= self.last_full_raise_size
                    if is_full_raise:
                        self.last_full_raise_size = raise_size
        else:  # pragma: no cover - enum分支已穷尽
            raise InvalidAction(f"未处理动作: {action}")

        if increased_bet:
            self.last_aggressor_id = player_id
            if self.street == Street.PREFLOP:
                self.preflop_raise_count += 1

        if is_full_raise:
            self.full_raise_epoch += 1
            self.pending = {
                other.player_id
                for other in self.players.values()
                if other.player_id != player_id and other.can_act
            }
        else:
            self.pending.discard(player_id)
            if increased_bet:
                self.pending.update(
                    other.player_id
                    for other in self.players.values()
                    if other.player_id != player_id
                    and other.can_act
                    and other.street_commitment < self.current_bet
                )

        self._player_action_epoch[player_id] = self.full_raise_epoch
        self._last_action_bet_level[player_id] = self.current_bet
        record = ActionRecord(
            sequence=self.sequence,
            street=self.street,
            player_id=player.player_id,
            position=player.position,
            action=action,
            requested_amount=amount,
            paid=paid,
            bet_to=player.street_commitment,
            pot_before=pot_before,
            pot_after=pot_before + paid,
            to_call_before=legal.to_call,
            current_bet_before=current_bet_before,
            current_bet_after=self.current_bet,
            min_raise_to_before=legal.min_raise_to,
            is_all_in=player.all_in,
            is_full_raise=is_full_raise,
        )
        self.history.append(record)
        self.decision_snapshots[record.sequence] = snapshot

        self._after_action(player_id)
        return record

    @staticmethod
    def _validated_target(
        amount: int | None,
        minimum: int | None,
        maximum: int,
        action_name: str,
    ) -> int:
        if amount is None or isinstance(amount, bool) or not isinstance(amount, int):
            raise InvalidAction(f"{action_name}必须提供整数 bet-to 金额")
        if minimum is None or amount < minimum:
            raise InvalidAction(f"{action_name}至少到 {minimum}")
        if amount > maximum:
            raise InvalidAction(f"筹码不足，最多到 {maximum}")
        return amount

    def _after_action(self, actor_id: str) -> None:
        self.pending = {
            player_id
            for player_id in self.pending
            if self.players[player_id].can_act
        }
        if len(self.live_players) == 1:
            self._complete_uncontested()
            return

        actionable = [player for player in self.live_players if player.can_act]
        if len(actionable) <= 1:
            lone = actionable[0] if actionable else None
            if lone is None or lone.street_commitment >= self.current_bet:
                self.pending.clear()

        if not self.pending:
            self._finish_betting_round()
            return
        self.current_actor_id = self._next_pending_after(actor_id)
        if self.current_actor_id is None:
            raise RuntimeError("行动队列不为空但找不到下一位玩家")

    def _advance_if_no_decision(self) -> None:
        if self.is_complete:
            return
        if len(self.live_players) == 1:
            self._complete_uncontested()
            return
        actionable = [player for player in self.live_players if player.can_act]
        if not actionable:
            self.pending.clear()
            self._finish_betting_round()
        elif len(actionable) == 1 and actionable[0].street_commitment >= self.current_bet:
            self.pending.clear()
            self._finish_betting_round()

    def _finish_betting_round(self) -> None:
        self._refund_uncalled()
        if self.street == Street.RIVER:
            self._showdown()
            return

        next_street = {
            Street.PREFLOP: Street.FLOP,
            Street.FLOP: Street.TURN,
            Street.TURN: Street.RIVER,
        }[self.street]
        self.street = next_street
        self._deal_next_street(next_street)
        for player in self.players.values():
            player.street_commitment = 0
        self.current_bet = 0
        self.last_full_raise_size = self.big_blind
        self.last_aggressor_id = None
        self.full_raise_epoch += 1
        self._player_action_epoch.clear()
        self._last_action_bet_level.clear()

        actionable = [player for player in self.live_players if player.can_act]
        if len(actionable) >= 2:
            self.pending = {player.player_id for player in actionable}
            self.current_actor_id = self._first_pending(POSTFLOP_ORDER)
            return

        # 只有一名或没有可行动玩家时不存在可争夺的干边池，直接发完。
        self.pending.clear()
        self.current_actor_id = None
        self._finish_betting_round()

    def _refund_uncalled(self) -> int:
        positive = sorted(
            (
                (player.total_commitment, player)
                for player in self.players.values()
                if player.total_commitment > 0
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not positive:
            return 0
        highest = positive[0][0]
        if sum(1 for amount, _ in positive if amount == highest) != 1:
            return 0
        second = positive[1][0] if len(positive) > 1 else 0
        refund = highest - second
        if refund <= 0:
            return 0
        player = positive[0][1]
        pot_before = sum(p.total_commitment for p in self.players.values())
        player.total_commitment -= refund
        player.street_commitment = max(0, player.street_commitment - refund)
        player.stack += refund
        player.all_in = player.stack == 0
        self.current_bet = max(
            (p.street_commitment for p in self.players.values() if p.live),
            default=0,
        )
        self.history.append(
            ActionRecord(
                sequence=self.sequence,
                street=self.street,
                player_id=player.player_id,
                position=player.position,
                action=ActionType.REFUND,
                requested_amount=None,
                paid=-refund,
                bet_to=player.street_commitment,
                pot_before=pot_before,
                pot_after=pot_before - refund,
                to_call_before=0,
                current_bet_before=self.current_bet + refund,
                current_bet_after=self.current_bet,
                min_raise_to_before=None,
                is_all_in=False,
                is_full_raise=False,
                forced=True,
            )
        )
        return refund

    def _complete_uncontested(self) -> None:
        self._refund_uncalled()
        winner = self.live_players[0]
        pots = build_pots(self.players.values())
        payout = sum(pot.amount for pot in pots)
        winner.stack += payout
        winner.payout += payout
        self.result = HandResult(
            reason="all_others_folded",
            board=tuple(self.board),
            pots=pots,
            payouts={player_id: payout if player_id == winner.player_id else 0 for player_id in self.players},
            hand_ranks={},
        )
        self.street = Street.COMPLETE
        self.pending.clear()
        self.current_actor_id = None

    def _showdown(self) -> None:
        self.street = Street.SHOWDOWN
        self._refund_uncalled()
        pots = build_pots(self.players.values())
        ranks: dict[str, HandRank] = {
            player.player_id: evaluate([*player.hole_cards, *self.board])
            for player in self.live_players
        }
        payouts = {player_id: 0 for player_id in self.players}
        odd_chip_order = [player.player_id for player in self._ordered_players(TABLE_ORDER)]
        for pot in pots:
            eligible = [player_id for player_id in pot.eligible if player_id in ranks]
            if not eligible:
                continue
            best = max(ranks[player_id] for player_id in eligible)
            winners = [player_id for player_id in eligible if ranks[player_id] == best]
            share, remainder = divmod(pot.amount, len(winners))
            for player_id in winners:
                payouts[player_id] += share
            for player_id in odd_chip_order:
                if remainder == 0:
                    break
                if player_id in winners:
                    payouts[player_id] += 1
                    remainder -= 1
        for player_id, payout in payouts.items():
            self.players[player_id].stack += payout
            self.players[player_id].payout += payout
        self.result = HandResult(
            reason="showdown",
            board=tuple(self.board),
            pots=pots,
            payouts=payouts,
            hand_ranks={player_id: rank.name_zh for player_id, rank in ranks.items()},
        )
        self.street = Street.COMPLETE
        self.pending.clear()
        self.current_actor_id = None

    def effective_stack_by_opponent(self, player_id: str) -> dict[str, int]:
        player = self.player(player_id)
        return {
            other.player_id: min(player.stack, other.stack)
            for other in self.live_players
            if other.player_id != player_id
        }

    def assert_chip_conservation(self) -> None:
        if self.is_complete:
            observed = sum(player.stack for player in self.players.values())
        else:
            observed = sum(player.stack for player in self.players.values()) + self.pot_size
        if observed != self.initial_total_chips:
            raise AssertionError(
                f"筹码不守恒: initial={self.initial_total_chips}, observed={observed}"
            )
        if any(player.stack < 0 for player in self.players.values()):
            raise AssertionError("出现负筹码")

    def public_view(self, viewer_id: str | None = None, *, reveal_all: bool = False) -> dict[str, Any]:
        players: list[dict[str, Any]] = []
        for player in self._ordered_players(TABLE_ORDER):
            show_cards = reveal_all or player.player_id == viewer_id or (
                self.result is not None and self.result.reason == "showdown" and not player.folded
            )
            players.append(
                {
                    "player_id": player.player_id,
                    "name": player.name,
                    "position": player.position.value,
                    "position_zh": player.position.label_zh,
                    "stack": player.stack,
                    "street_commitment": player.street_commitment,
                    "total_commitment": player.total_commitment,
                    "folded": player.folded,
                    "all_in": player.all_in,
                    "hole_cards": [str(card) for card in player.hole_cards] if show_cards else [],
                }
            )
        return {
            "hand_id": self.hand_id,
            "seed": self.seed,
            "street": self.street.value,
            "board": [str(card) for card in self.board],
            "pot": self.pot_size,
            "committed_pot": self.committed_pot,
            "current_bet": self.current_bet,
            "current_actor_id": self.current_actor_id,
            "players": players,
            "history": [record.as_dict() for record in self.history],
            "result": asdict(self.result) if self.result else None,
        }

