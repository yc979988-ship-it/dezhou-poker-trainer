"""Deterministic, private-information-safe policy for simulated opponents.

The policy deliberately is *not* a GTO solver.  It turns a continuously
parameterised :class:`OpponentProfile` into plausible legal actions while
keeping every random stream separate from the hand's deck RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import random
from typing import TYPE_CHECKING, Protocol

from poker_trainer.engine.cards import Card, SUIT_CHARS
from poker_trainer.engine.evaluator import evaluate
from poker_trainer.engine.models import (
    ActionRecord,
    ActionType,
    LegalActions,
    Position,
    Street,
)

if TYPE_CHECKING:
    from poker_trainer.engine.hand import HoldemHand
    from poker_trainer.opponents.profiles import OpponentProfile


POLICY_VERSION = "opponent-policy-v2"
_MONTE_CARLO_TRIALS = 56


class _ProfileLike(Protocol):
    """Structural type kept private so importing this module has no cycle."""

    vpip: float
    pfr: float
    three_bet: float
    aggression_factor: float
    fold_tendency: float
    limp_tendency: float
    mistake_rate: float


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Immutable copy of exactly what a bot may use at one decision point.

    ``from_hand`` copies cards and history into tuples and never retains a
    reference to the mutable hand or its deck.  Opponents' hole cards and
    hidden profile parameters are intentionally absent.
    """

    hand_id: str
    hand_seed: int
    sequence: int
    player_id: str
    position: Position
    street: Street
    hole_cards: tuple[Card, ...]
    board: tuple[Card, ...]
    stack: int
    starting_stack: int
    street_commitment: int
    total_commitment: int
    pot_before: int
    current_bet: int
    to_call: int
    call_amount: int
    min_bet_to: int | None
    min_raise_to: int | None
    max_to: int
    legal: LegalActions
    active_players: int
    effective_stack: int
    preflop_raise_count: int
    limper_count: int
    cold_caller_count: int
    last_aggressor_id: str | None
    was_preflop_aggressor: bool
    history: tuple[ActionRecord, ...]
    small_blind: int
    big_blind: int

    @classmethod
    def from_hand(cls, hand: "HoldemHand", player_id: str) -> "PolicyContext":
        """Build a decision snapshot without consuming or exposing the deck."""

        legal = hand.legal_actions(player_id)
        snapshot = hand.decision_snapshot(player_id)
        player = hand.player(player_id)
        history = tuple(hand.history)
        preflop_actions = tuple(
            record
            for record in history
            if record.street == Street.PREFLOP and not record.forced
        )
        limper_count = sum(
            record.action == ActionType.CALL
            and record.current_bet_before <= hand.big_blind
            for record in preflop_actions
        )
        cold_caller_count = sum(
            record.action == ActionType.CALL
            and record.current_bet_before > hand.big_blind
            for record in preflop_actions
        )
        preflop_aggressors = [
            record.player_id
            for record in preflop_actions
            if record.action in (ActionType.RAISE, ActionType.ALL_IN)
            and record.current_bet_after > record.current_bet_before
        ]
        effective = hand.effective_stack_by_opponent(player_id)
        return cls(
            hand_id=hand.hand_id,
            hand_seed=hand.seed,
            sequence=hand.sequence,
            player_id=player_id,
            position=player.position,
            street=hand.street,
            hole_cards=tuple(player.hole_cards),
            board=tuple(hand.board),
            stack=player.stack,
            starting_stack=player.starting_stack,
            street_commitment=player.street_commitment,
            total_commitment=player.total_commitment,
            pot_before=legal.pot_before,
            current_bet=hand.current_bet,
            to_call=legal.to_call,
            call_amount=legal.call_amount,
            min_bet_to=legal.min_bet_to,
            min_raise_to=legal.min_raise_to,
            max_to=legal.max_to,
            legal=legal,
            active_players=snapshot.active_players,
            effective_stack=max(effective.values(), default=0),
            preflop_raise_count=snapshot.preflop_raise_count,
            limper_count=limper_count,
            cold_caller_count=cold_caller_count,
            last_aggressor_id=snapshot.last_aggressor_id,
            was_preflop_aggressor=bool(
                preflop_aggressors and preflop_aggressors[-1] == player_id
            ),
            history=history,
            small_blind=hand.small_blind,
            big_blind=hand.big_blind,
        )

    @property
    def pot_size(self) -> int:
        """Alias used by UI and analysis code."""

        return self.pot_before

    @property
    def legal_actions(self) -> tuple[ActionType, ...]:
        return self.legal.action_types

    @property
    def facing_raise(self) -> bool:
        return self.street == Street.PREFLOP and self.preflop_raise_count > 0

    @property
    def limped_before_raise(self) -> bool:
        """当前玩家是否已在本手翻前 limp，供 limp-call 混合策略使用。"""

        return any(
            not record.forced
            and record.street == Street.PREFLOP
            and record.player_id == self.player_id
            and record.action == ActionType.CALL
            and record.current_bet_before <= self.big_blind
            for record in self.history
        )

    @property
    def pot_odds(self) -> float:
        if self.call_amount <= 0:
            return 0.0
        return self.call_amount / (self.pot_before + self.call_amount)


@dataclass(frozen=True, slots=True)
class BotDecision:
    """An engine-ready action selected by :class:`OpponentPolicy`."""

    action: ActionType
    amount: int | None = None
    reason: str = ""
    estimated_equity: float | None = None
    is_mistake: bool = False
    policy_version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionType):
            object.__setattr__(self, "action", ActionType(self.action))
        if self.action in (ActionType.BET, ActionType.RAISE):
            if self.amount is None or self.amount < 0:
                raise ValueError("下注或加注必须带有非负 bet-to 金额")
        elif self.amount is not None:
            raise ValueError("只有下注或加注动作可以携带金额")

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "amount": self.amount,
            "reason": self.reason,
            "estimated_equity": self.estimated_equity,
            "is_mistake": self.is_mistake,
            "policy_version": self.policy_version,
        }


class OpponentPolicy:
    """Choose plausible actions from a profile and an immutable context."""

    version = POLICY_VERSION

    @staticmethod
    def choose(
        context: PolicyContext,
        profile: "OpponentProfile | _ProfileLike",
        policy_seed: int | str | bytes,
    ) -> BotDecision:
        _validate_profile(profile)
        if context.player_id != context.legal.player_id:
            raise ValueError("策略上下文与合法动作玩家不一致")

        if _uniform(policy_seed, context, "mistake-trigger") < profile.mistake_rate:
            decision = _mistake_decision(context, policy_seed)
        elif context.street == Street.PREFLOP:
            decision = _choose_preflop(context, profile, policy_seed)
        else:
            decision = _choose_postflop(context, profile, policy_seed)
        return _ensure_legal(context, decision)


def _validate_profile(profile: _ProfileLike) -> None:
    required = (
        "vpip",
        "pfr",
        "three_bet",
        "aggression_factor",
        "fold_tendency",
        "limp_tendency",
        "mistake_rate",
    )
    if any(not hasattr(profile, name) for name in required):
        raise TypeError("profile 缺少策略所需参数")


def _seed_part(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _named_rng(
    policy_seed: int | str | bytes,
    context: PolicyContext,
    namespace: str,
) -> random.Random:
    """Return an isolated RNG derived from a stable, named BLAKE2 stream."""

    digest = blake2b(digest_size=32, person=b"poker-policy-v1")
    for value in (
        POLICY_VERSION,
        policy_seed,
        context.hand_id,
        context.hand_seed,
        context.sequence,
        context.player_id,
        context.street.value,
        ",".join(map(str, context.hole_cards)),
        ",".join(map(str, context.board)),
        namespace,
    ):
        part = _seed_part(value)
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return random.Random(int.from_bytes(digest.digest(), "big"))


def _uniform(
    policy_seed: int | str | bytes,
    context: PolicyContext,
    namespace: str,
) -> float:
    return _named_rng(policy_seed, context, namespace).random()


_POSITION_ADJUSTMENT = {
    Position.UTG: -0.065,
    Position.HJ: -0.025,
    Position.CO: 0.030,
    Position.BTN: 0.075,
    Position.SB: -0.020,
    Position.BB: 0.030,
}


def _preflop_strength(hole_cards: tuple[Card, ...]) -> float:
    """Compact 0..1 starting-hand score; ordering matters more than precision."""

    if len(hole_cards) != 2:
        raise ValueError("翻前策略需要恰好两张底牌")
    first, second = hole_cards
    high, low = sorted((first.rank, second.rank), reverse=True)
    if high == low:
        return 0.54 + (high - 2) / 12 * 0.44

    score = (high - 2) / 12 * 0.45 + (low - 2) / 12 * 0.25
    if first.suit == second.suit:
        score += 0.08
    gap = high - low
    if gap <= 1:
        score += 0.08
    elif gap == 2:
        score += 0.05
    elif gap == 3:
        score += 0.02
    if low >= 10:
        score += 0.08
    if high == 14:
        score += 0.04
    return max(0.0, min(1.0, score))


def _choose_preflop(
    context: PolicyContext,
    profile: _ProfileLike,
    policy_seed: int | str | bytes,
) -> BotDecision:
    strength = _preflop_strength(context.hole_cards)
    position_score = strength + _POSITION_ADJUSTMENT[context.position]
    selection_noise = (_uniform(policy_seed, context, "preflop-selection") - 0.5) * 0.12
    selection_score = position_score + selection_noise

    # A free option is never surrendered by the normal policy.
    if context.legal.can_check and strength < 0.40:
        return BotDecision(ActionType.CHECK, reason="免费过牌保留权益", estimated_equity=strength)

    if context.facing_raise:
        return _choose_facing_preflop_raise(
            context, profile, policy_seed, strength, selection_score
        )

    play_threshold = 0.73 - 0.68 * profile.vpip
    raise_threshold = 0.78 - 0.65 * profile.pfr
    if context.limper_count:
        play_threshold -= min(0.06, context.limper_count * 0.02)
        raise_threshold += min(0.04, context.limper_count * 0.01)

    premium = strength >= 0.90
    wants_raise = premium or selection_score >= raise_threshold
    wants_play = wants_raise or selection_score >= play_threshold

    if wants_raise and context.legal.can_raise:
        target = _preflop_raise_target(context, policy_seed, three_bet=False)
        return BotDecision(
            ActionType.RAISE,
            target,
            "强度足够，主动加注",
            strength,
        )
    if premium and context.legal.can_all_in and not context.legal.can_raise:
        return BotDecision(ActionType.ALL_IN, reason="短码强牌全下", estimated_equity=strength)

    if wants_play and context.legal.can_call:
        # Limp tendency is most visible when opening; over-limping remains
        # plausible after prior callers.  The action is still simply CALL.
        limp_roll = _uniform(policy_seed, context, "preflop-limp")
        if context.limper_count or limp_roll < profile.limp_tendency or not context.legal.can_raise:
            return BotDecision(ActionType.CALL, reason="进入底池观察后续", estimated_equity=strength)
        if selection_score + 0.025 >= raise_threshold:
            target = _preflop_raise_target(context, policy_seed, three_bet=False)
            return BotDecision(ActionType.RAISE, target, "边缘牌主动加注", strength)
        return BotDecision(ActionType.CALL, reason="边缘牌平跟", estimated_equity=strength)

    if context.legal.can_check:
        return BotDecision(ActionType.CHECK, reason="免费过牌", estimated_equity=strength)
    return BotDecision(ActionType.FOLD, reason="起手牌不足以继续", estimated_equity=strength)


def _choose_facing_preflop_raise(
    context: PolicyContext,
    profile: _ProfileLike,
    policy_seed: int | str | bytes,
    strength: float,
    selection_score: float,
) -> BotDecision:
    three_bet_noise = (_uniform(policy_seed, context, "preflop-three-bet") - 0.5) * 0.08
    three_bet_threshold = 0.88 - 1.50 * profile.three_bet
    if context.preflop_raise_count >= 2:
        three_bet_threshold += 0.075 * (context.preflop_raise_count - 1)

    premium = strength >= 0.90
    wants_reraise = premium or selection_score + three_bet_noise >= three_bet_threshold
    if wants_reraise and context.legal.can_raise:
        target = _preflop_raise_target(context, policy_seed, three_bet=True)
        return BotDecision(ActionType.RAISE, target, "强牌再加注", strength)
    if premium and context.legal.can_all_in:
        return BotDecision(ActionType.ALL_IN, reason="强牌短码全下", estimated_equity=strength)

    # 3bb 是朋友局的普通开池尺度，应比 4bb 及以上更容易获得跟注；
    # 已经 limp 的玩家面对首次小加注也会以混合频率补齐，但面对 3bet
    # 不继承该优惠。已有冷跟者时再给少量多人池倾向。
    open_size_bb = context.current_bet / context.big_blind
    sizing_pressure = max(-0.04, min(0.16, (open_size_bb - 4.0) * 0.04))
    limp_call_discount = (
        0.075
        if context.limped_before_raise and context.preflop_raise_count == 1
        else 0.0
    )
    cold_call_discount = min(0.06, context.cold_caller_count * 0.02)
    continue_threshold = (
        0.76
        - 0.50 * profile.vpip
        + 0.12 * profile.fold_tendency
        + max(0, context.preflop_raise_count - 1) * 0.09
        + sizing_pressure
        - limp_call_discount
        - cold_call_discount
    )
    continue_noise = (_uniform(policy_seed, context, "preflop-continue") - 0.5) * 0.09
    if context.legal.can_call and selection_score + continue_noise >= continue_threshold:
        return BotDecision(ActionType.CALL, reason="牌力足够承受当前加注", estimated_equity=strength)
    if context.legal.can_check:
        return BotDecision(ActionType.CHECK, reason="无须补筹码", estimated_equity=strength)
    return BotDecision(ActionType.FOLD, reason="面对加注牌力不足", estimated_equity=strength)


def _preflop_raise_target(
    context: PolicyContext,
    policy_seed: int | str | bytes,
    *,
    three_bet: bool,
) -> int:
    if context.min_raise_to is None:
        raise ValueError("当前没有合法加注目标")
    variation = _uniform(policy_seed, context, "preflop-sizing")
    if three_bet:
        if context.preflop_raise_count >= 2:
            multiplier = 2.15 + variation * 0.35
        else:
            multiplier = (
                3.35 if context.position in (Position.SB, Position.BB) else 2.90
            ) + variation * 0.35
        raw_target = round(context.current_bet * multiplier)
        raw_target += context.cold_caller_count * context.big_blind
    else:
        multiplier = 2.25 + variation * 0.55
        if context.position in (Position.SB, Position.BB):
            multiplier += 0.35
        raw_target = round(context.big_blind * (multiplier + context.limper_count))
    return _legal_target(context, raw_target, context.min_raise_to)


def _choose_postflop(
    context: PolicyContext,
    profile: _ProfileLike,
    policy_seed: int | str | bytes,
) -> BotDecision:
    equity = _estimate_equity(context, policy_seed)
    aggression = profile.aggression_factor / (profile.aggression_factor + 2.0)

    if context.to_call == 0:
        bet_roll = _uniform(policy_seed, context, "postflop-bet")
        strong_threshold = 0.59 if context.active_players <= 2 else 0.52
        cbet_bonus = 0.15 if context.street == Street.FLOP and context.was_preflop_aggressor else 0.0
        if equity >= strong_threshold:
            wants_bet = bet_roll < 0.48 + 0.47 * aggression
        elif equity >= 0.34:
            wants_bet = bet_roll < 0.08 + 0.38 * aggression + cbet_bonus
        else:
            wants_bet = bet_roll < 0.02 + 0.11 * aggression + cbet_bonus * 0.55

        if wants_bet and context.legal.can_bet:
            target = _postflop_bet_target(context, policy_seed, equity)
            return BotDecision(ActionType.BET, target, "牌力与激进度支持下注", equity)
        if context.legal.can_check:
            return BotDecision(ActionType.CHECK, reason="控制底池或保留过牌范围", estimated_equity=equity)
        return _fallback(context, equity)

    required_equity = context.pot_odds + 0.025 + (profile.fold_tendency - 0.5) * 0.16
    continue_roll = (_uniform(policy_seed, context, "postflop-continue") - 0.5) * 0.055
    should_continue = equity + continue_roll >= required_equity
    if not should_continue:
        return BotDecision(ActionType.FOLD, reason="权益低于继续所需门槛", estimated_equity=equity)

    raise_roll = _uniform(policy_seed, context, "postflop-raise")
    strong_raise = equity >= (0.69 if context.active_players <= 2 else 0.62)
    semi_bluff_raise = equity >= 0.39 and raise_roll < 0.10 * aggression
    if (
        (strong_raise and raise_roll < 0.30 + 0.60 * aggression) or semi_bluff_raise
    ) and context.legal.can_raise:
        target = _postflop_raise_target(context, policy_seed, equity)
        return BotDecision(ActionType.RAISE, target, "价值牌或强听牌加注", equity)
    if context.legal.can_call:
        return BotDecision(ActionType.CALL, reason="权益覆盖底池赔率", estimated_equity=equity)
    if context.legal.can_all_in:
        return BotDecision(ActionType.ALL_IN, reason="短码继续只能全下", estimated_equity=equity)
    return _fallback(context, equity)


def _estimate_equity(
    context: PolicyContext,
    policy_seed: int | str | bytes,
    *,
    trials: int = _MONTE_CARLO_TRIALS,
) -> float:
    """Estimate equity versus uniformly sampled unknown hands.

    This intentionally builds a fresh card universe.  It never reads, shuffles,
    or otherwise advances ``HoldemHand.deck``.
    """

    known = set((*context.hole_cards, *context.board))
    unseen = [
        Card(rank, suit)
        for suit in SUIT_CHARS
        for rank in range(2, 15)
        if Card(rank, suit) not in known
    ]
    opponent_count = max(1, context.active_players - 1)
    board_needed = 5 - len(context.board)
    needed = opponent_count * 2 + board_needed
    if needed > len(unseen):
        raise ValueError("已知牌与活跃玩家数量不一致")

    rng = _named_rng(policy_seed, context, "postflop-equity-monte-carlo")
    equity_total = 0.0
    for _ in range(max(1, trials)):
        sampled = rng.sample(unseen, needed)
        completed_board = [*context.board, *sampled[:board_needed]]
        hero_rank = evaluate([*context.hole_cards, *completed_board])
        cursor = board_needed
        opponent_ranks = []
        for _opponent in range(opponent_count):
            opponent_hole = sampled[cursor : cursor + 2]
            cursor += 2
            opponent_ranks.append(evaluate([*opponent_hole, *completed_board]))
        best = max([hero_rank, *opponent_ranks])
        if hero_rank == best:
            tied_winners = 1 + sum(rank == hero_rank for rank in opponent_ranks)
            equity_total += 1.0 / tied_winners
    return equity_total / max(1, trials)


def _postflop_bet_target(
    context: PolicyContext,
    policy_seed: int | str | bytes,
    equity: float,
) -> int:
    if context.min_bet_to is None:
        raise ValueError("当前没有合法下注目标")
    size_roll = _uniform(policy_seed, context, "postflop-bet-sizing")
    if equity >= 0.72:
        fraction = 0.66 + size_roll * 0.24
    elif equity >= 0.45:
        fraction = 0.45 + size_roll * 0.22
    else:
        fraction = 0.30 + size_roll * 0.18
    raw_target = context.street_commitment + round(context.pot_before * fraction)
    return _legal_target(context, raw_target, context.min_bet_to)


def _postflop_raise_target(
    context: PolicyContext,
    policy_seed: int | str | bytes,
    equity: float,
) -> int:
    if context.min_raise_to is None:
        raise ValueError("当前没有合法加注目标")
    size_roll = _uniform(policy_seed, context, "postflop-raise-sizing")
    fraction = (0.62 if equity >= 0.65 else 0.48) + size_roll * 0.28
    raw_target = context.current_bet + round(
        (context.pot_before + context.call_amount) * fraction
    )
    return _legal_target(context, raw_target, context.min_raise_to)


def _legal_target(context: PolicyContext, raw_target: int, minimum: int) -> int:
    unit = max(1, context.small_blind)
    rounded = int(round(raw_target / unit) * unit)
    return max(minimum, min(context.max_to, rounded))


def _mistake_decision(
    context: PolicyContext,
    policy_seed: int | str | bytes,
) -> BotDecision:
    """Pick a legal but deliberately non-standard action."""

    candidates: list[BotDecision] = []
    if context.legal.can_fold and context.to_call > 0:
        candidates.append(BotDecision(ActionType.FOLD, reason="偶发过度弃牌", is_mistake=True))
    if context.legal.can_check:
        candidates.append(BotDecision(ActionType.CHECK, reason="偶发被动过牌", is_mistake=True))
    if context.legal.can_call:
        candidates.append(BotDecision(ActionType.CALL, reason="偶发宽松跟注", is_mistake=True))
    if context.legal.can_bet and context.min_bet_to is not None:
        candidates.append(
            BotDecision(
                ActionType.BET,
                context.min_bet_to,
                "偶发最小下注",
                is_mistake=True,
            )
        )
    if context.legal.can_raise and context.min_raise_to is not None:
        candidates.append(
            BotDecision(
                ActionType.RAISE,
                context.min_raise_to,
                "偶发最小加注",
                is_mistake=True,
            )
        )
    # 普通100bb朋友局不把“偶发错误”实现成无脑深码推入；仅在短码或
    # 已形成低SPR底池时保留非常规全下。
    if context.legal.can_all_in and (
        context.stack <= 20 * context.big_blind
        or (context.pot_before > 0 and context.effective_stack <= context.pot_before)
    ):
        candidates.append(BotDecision(ActionType.ALL_IN, reason="偶发非常规全下", is_mistake=True))
    if not candidates:
        return _fallback(context, None, is_mistake=True)
    rng = _named_rng(policy_seed, context, "mistake-action")
    return candidates[rng.randrange(len(candidates))]


def _ensure_legal(context: PolicyContext, decision: BotDecision) -> BotDecision:
    if decision.action not in context.legal_actions:
        return _fallback(context, decision.estimated_equity, is_mistake=decision.is_mistake)
    if decision.action == ActionType.BET:
        minimum = context.min_bet_to
        if minimum is None or decision.amount is None:
            return _fallback(context, decision.estimated_equity, is_mistake=decision.is_mistake)
        target = _legal_target(context, decision.amount, minimum)
        if target != decision.amount:
            return BotDecision(
                decision.action,
                target,
                decision.reason,
                decision.estimated_equity,
                decision.is_mistake,
            )
    elif decision.action == ActionType.RAISE:
        minimum = context.min_raise_to
        if minimum is None or decision.amount is None:
            return _fallback(context, decision.estimated_equity, is_mistake=decision.is_mistake)
        target = _legal_target(context, decision.amount, minimum)
        if target != decision.amount:
            return BotDecision(
                decision.action,
                target,
                decision.reason,
                decision.estimated_equity,
                decision.is_mistake,
            )
    return decision


def _fallback(
    context: PolicyContext,
    equity: float | None,
    *,
    is_mistake: bool = False,
) -> BotDecision:
    if context.legal.can_check:
        return BotDecision(
            ActionType.CHECK,
            reason="合法性兜底过牌",
            estimated_equity=equity,
            is_mistake=is_mistake,
        )
    if context.legal.can_call:
        return BotDecision(
            ActionType.CALL,
            reason="合法性兜底跟注",
            estimated_equity=equity,
            is_mistake=is_mistake,
        )
    if context.legal.can_fold:
        return BotDecision(
            ActionType.FOLD,
            reason="合法性兜底弃牌",
            estimated_equity=equity,
            is_mistake=is_mistake,
        )
    if context.legal.can_all_in:
        return BotDecision(
            ActionType.ALL_IN,
            reason="合法性兜底全下",
            estimated_equity=equity,
            is_mistake=is_mistake,
        )
    raise RuntimeError("上下文不存在任何合法动作")


__all__ = ["BotDecision", "OpponentPolicy", "POLICY_VERSION", "PolicyContext"]

