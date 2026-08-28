from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import random

import pytest

from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position, Seat, Street
from poker_trainer.opponents.policy import OpponentPolicy, PolicyContext
from poker_trainer.opponents.profiles import OpponentProfile, generate_base_profile


def profile(
    opponent_id: str = "bot",
    *,
    vpip: float = 0.31,
    pfr: float = 0.19,
    three_bet: float = 0.08,
    aggression_factor: float = 2.1,
    fold_tendency: float = 0.49,
    limp_tendency: float = 0.12,
    mistake_rate: float = 0.0,
) -> OpponentProfile:
    return OpponentProfile(
        opponent_id=opponent_id,
        vpip=vpip,
        pfr=pfr,
        three_bet=three_bet,
        aggression_factor=aggression_factor,
        fold_tendency=fold_tendency,
        limp_tendency=limp_tendency,
        mistake_rate=mistake_rate,
    )


def test_aa_open_raises_and_72o_folds(six_seats) -> None:
    aa_hand = HoldemHand(
        six_seats(),
        seed=10,
        hole_overrides={"UTG": ["As", "Ah"]},
    )
    aa_context = PolicyContext.from_hand(aa_hand, "UTG")
    aa_decision = OpponentPolicy.choose(aa_context, profile(), policy_seed=99)

    assert aa_decision.action == ActionType.RAISE
    assert aa_context.min_raise_to <= aa_decision.amount <= aa_context.max_to
    aa_hand.act("UTG", aa_decision.action, aa_decision.amount)

    trash_hand = HoldemHand(
        six_seats(),
        seed=11,
        hole_overrides={"UTG": ["7c", "2d"]},
    )
    trash_context = PolicyContext.from_hand(trash_hand, "UTG")
    trash_decision = OpponentPolicy.choose(trash_context, profile(), policy_seed=99)
    assert trash_decision.action == ActionType.FOLD


def test_big_blind_uses_free_check_with_72o() -> None:
    seats = [
        Seat("SB", "SB", Position.SB, 4000),
        Seat("BB", "BB", Position.BB, 4000),
    ]
    hand = HoldemHand(
        seats,
        seed=12,
        hole_overrides={"BB": ["7c", "2d"]},
    )
    hand.act("SB", ActionType.CALL)
    context = PolicyContext.from_hand(hand, "BB")
    decision = OpponentPolicy.choose(context, profile("BB"), policy_seed=1)

    assert context.to_call == 0
    assert decision.action == ActionType.CHECK


def test_policy_is_deterministic_and_does_not_touch_any_rng_or_deck(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=20260827,
        hole_overrides={"UTG": ["Qs", "Js"]},
    )
    context = PolicyContext.from_hand(hand, "UTG")
    deck_before = tuple(hand.deck.cards)
    random.seed(314159)
    global_state_before = random.getstate()

    first = OpponentPolicy.choose(context, profile(), policy_seed="session-7")
    second = OpponentPolicy.choose(context, profile(), policy_seed="session-7")

    assert first == second
    assert tuple(hand.deck.cards) == deck_before
    assert random.getstate() == global_state_before


def test_high_three_bet_profile_reraises_where_low_profile_does_not(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=30,
        hole_overrides={"HJ": ["8c", "8d"]},
    )
    hand.act("UTG", ActionType.RAISE, 120)
    context = PolicyContext.from_hand(hand, "HJ")
    passive = profile(
        vpip=0.22,
        pfr=0.10,
        three_bet=0.025,
        fold_tendency=0.70,
    )
    active = profile(
        vpip=0.50,
        pfr=0.40,
        three_bet=0.14,
        fold_tendency=0.28,
    )

    passive_action = OpponentPolicy.choose(context, passive, policy_seed=4).action
    active_action = OpponentPolicy.choose(context, active, policy_seed=4).action

    assert active_action == ActionType.RAISE
    assert passive_action != ActionType.RAISE


def test_medium_suited_hand_can_flat_three_bb_but_folds_to_larger_open(
    six_seats,
) -> None:
    def decide(open_to: int) -> ActionType:
        hand = HoldemHand(
            six_seats(),
            seed=99,
            hand_id="three-bb-call-test",
            hole_overrides={"HJ": ["Jc", "9c"]},
        )
        hand.act("UTG", ActionType.RAISE, open_to)
        context = PolicyContext.from_hand(hand, "HJ")
        decision = OpponentPolicy.choose(
            context,
            profile("HJ", vpip=0.50, pfr=0.16, fold_tendency=0.48),
            policy_seed=0,
        )
        return decision.action

    assert decide(120) == ActionType.CALL
    assert decide(140) == ActionType.CALL
    assert decide(240) == ActionType.FOLD


def test_seeded_population_produces_multiway_limp_calls_without_calling_everything(
    six_seats,
) -> None:
    def sample() -> tuple[int, int, int, int, int]:
        profiles = {
            position.value: generate_base_profile(position.value, master_seed=777)
            for position in (
                Position.UTG,
                Position.HJ,
                Position.CO,
                Position.BTN,
                Position.SB,
                Position.BB,
            )
        }
        multiway_hands = cold_calls = cold_call_spots = 0
        limp_calls = limp_call_spots = 0
        for hand_seed in range(320):
            hand = HoldemHand(
                six_seats(),
                seed=hand_seed,
                hand_id=f"friend-game-calibration-{hand_seed}",
            )
            voluntary_players: set[str] = set()
            while hand.street == Street.PREFLOP and not hand.is_complete:
                player_id = hand.current_actor_id
                assert player_id is not None
                context = PolicyContext.from_hand(hand, player_id)
                decision = OpponentPolicy.choose(
                    context,
                    profiles[player_id],
                    policy_seed=f"friend-game-policy-{hand_seed}",
                )
                if decision.action in (
                    ActionType.CALL,
                    ActionType.RAISE,
                    ActionType.ALL_IN,
                ):
                    voluntary_players.add(player_id)
                if context.preflop_raise_count > 0 and context.legal.can_call:
                    if context.limped_before_raise:
                        limp_call_spots += 1
                        limp_calls += decision.action == ActionType.CALL
                    else:
                        cold_call_spots += 1
                        cold_calls += decision.action == ActionType.CALL
                hand.act(player_id, decision.action, decision.amount)
            multiway_hands += len(voluntary_players) >= 3
        return (
            multiway_hands,
            cold_calls,
            cold_call_spots,
            limp_calls,
            limp_call_spots,
        )

    first = sample()
    assert sample() == first
    multiway_hands, cold_calls, cold_call_spots, limp_calls, limp_call_spots = first

    assert 220 <= multiway_hands <= 290
    assert 0.22 <= cold_calls / cold_call_spots <= 0.42
    assert 0.45 <= limp_calls / limp_call_spots <= 0.75


def test_limper_mixes_call_and_fold_against_three_bb_raise(six_seats) -> None:
    def decide(cards: list[str], open_to: int) -> tuple[PolicyContext, ActionType]:
        hand = HoldemHand(
            six_seats(),
            seed=901,
            hand_id="limp-call-test",
            hole_overrides={"UTG": cards},
        )
        hand.act("UTG", ActionType.CALL)
        hand.act("HJ", ActionType.RAISE, open_to)
        for player_id in ("CO", "BTN", "SB", "BB"):
            hand.act(player_id, ActionType.FOLD)
        context = PolicyContext.from_hand(hand, "UTG")
        decision = OpponentPolicy.choose(
            context,
            profile(
                "UTG",
                vpip=0.37,
                pfr=0.24,
                fold_tendency=0.48,
                limp_tendency=0.18,
            ),
            policy_seed=0,
        )
        return context, decision.action

    small_context, medium_vs_small = decide(["9s", "8s"], 120)
    _, medium_vs_large = decide(["9s", "8s"], 240)
    _, weak_vs_small = decide(["7s", "2d"], 120)

    assert small_context.limped_before_raise
    assert medium_vs_small == ActionType.CALL
    assert medium_vs_large == ActionType.FOLD
    assert weak_vs_small == ActionType.FOLD


def test_context_is_a_read_only_copy_and_contains_no_other_hole_cards(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=41,
        hole_overrides={"UTG": ["As", "Kd"], "HJ": ["2c", "3c"]},
    )
    deck_before = tuple(hand.deck.cards)
    context = PolicyContext.from_hand(hand, "UTG")

    assert context.hole_cards == tuple(hand.player("UTG").hole_cards)
    assert "2c" not in repr(context)
    assert "3c" not in repr(context)
    assert tuple(hand.deck.cards) == deck_before
    with pytest.raises(FrozenInstanceError):
        context.stack = 1  # type: ignore[misc]


def test_hidden_profile_fields_never_enter_hand_public_view(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=55)
    context = PolicyContext.from_hand(hand, "UTG")
    hidden = generate_base_profile("UTG", master_seed=777)
    OpponentPolicy.choose(context, hidden, policy_seed=888)
    public_json = json.dumps(hand.public_view(viewer_id="BTN"), sort_keys=True)

    for hidden_name in (
        "vpip",
        "pfr",
        "three_bet",
        "aggression_factor",
        "fold_tendency",
        "limp_tendency",
        "mistake_rate",
    ):
        assert hidden_name not in public_json


@pytest.mark.parametrize("hand_seed", range(8))
def test_seeded_bot_play_only_produces_engine_legal_actions(
    six_seats, hand_seed: int
) -> None:
    hand = HoldemHand(six_seats(), seed=hand_seed)
    profiles = {
        player_id: generate_base_profile(player_id, master_seed=9000)
        for player_id in hand.players
    }
    decisions = 0

    while not hand.is_complete:
        assert decisions < 100
        player_id = hand.current_actor_id
        assert player_id is not None
        context = PolicyContext.from_hand(hand, player_id)
        deck_before = tuple(hand.deck.cards)
        decision = OpponentPolicy.choose(
            context,
            profiles[player_id],
            policy_seed=f"run:{hand_seed}",
        )
        assert tuple(hand.deck.cards) == deck_before
        assert decision.action in context.legal_actions
        if decision.action == ActionType.BET:
            assert context.min_bet_to <= decision.amount <= context.max_to
        elif decision.action == ActionType.RAISE:
            assert context.min_raise_to <= decision.amount <= context.max_to
        else:
            assert decision.amount is None
        hand.act(player_id, decision.action, decision.amount)
        decisions += 1

    hand.assert_chip_conservation()


def test_forced_mistakes_are_still_legal(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=72,
        hole_overrides={"UTG": ["7c", "2d"]},
    )
    context = PolicyContext.from_hand(hand, "UTG")
    always_errs = profile(mistake_rate=1.0)

    for policy_seed in range(40):
        decision = OpponentPolicy.choose(context, always_errs, policy_seed)
        assert decision.is_mistake
        assert decision.action in context.legal_actions
        if decision.action == ActionType.RAISE:
            assert context.min_raise_to <= decision.amount <= context.max_to
        else:
            assert decision.amount is None
