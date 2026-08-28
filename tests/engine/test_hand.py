from __future__ import annotations

import pytest

from poker_trainer.engine.hand import HoldemHand, InvalidAction
from poker_trainer.engine.models import ActionType, Position, Street


def passive_action(hand: HoldemHand) -> None:
    actor = hand.current_actor_id
    legal = hand.legal_actions(actor)
    hand.act(actor, ActionType.CALL if legal.to_call else ActionType.CHECK)


def test_blinds_and_correct_preflop_then_postflop_action_order(six_seats):
    hand = HoldemHand(six_seats(), seed=11)
    assert hand.pot_size == 60
    assert hand.current_actor_id == Position.UTG.value

    order = []
    while hand.street == Street.PREFLOP:
        order.append(hand.current_actor_id)
        passive_action(hand)

    assert order == ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    assert hand.committed_pot == 240
    assert hand.street == Street.FLOP
    assert len(hand.board) == 3
    assert hand.current_actor_id == "SB"
    hand.assert_chip_conservation()


def test_folded_players_are_removed_but_their_chips_stay_in_pot(six_seats):
    hand = HoldemHand(six_seats(), seed=12)
    for player_id in ("UTG", "HJ", "CO", "BTN"):
        assert hand.current_actor_id == player_id
        hand.act(player_id, ActionType.FOLD)
    assert hand.current_actor_id == "SB"
    hand.act("SB", ActionType.FOLD)

    assert hand.is_complete
    assert hand.result.reason == "all_others_folded"
    # BB未被跟注的20退回，只赢SB的20。
    assert hand.result.payouts["BB"] == 40
    assert hand.player("BB").stack == 4020
    assert hand.player("SB").stack == 3980
    assert hand.player("SB").total_commitment == 20
    assert not any(
        "SB" in pot.eligible for pot in hand.result.pots
    )
    hand.assert_chip_conservation()


def test_minimum_raise_uses_last_full_raise_increment_and_is_atomic(six_seats):
    hand = HoldemHand(six_seats(), seed=13)
    legal = hand.legal_actions()
    assert legal.min_raise_to == 80

    hand.act("UTG", ActionType.RAISE, 120)
    assert hand.last_full_raise_size == 80
    assert hand.legal_actions().min_raise_to == 200

    stack_before = hand.player("HJ").stack
    sequence_before = hand.sequence
    with pytest.raises(InvalidAction, match="至少到"):
        hand.act("HJ", ActionType.RAISE, 199)
    assert hand.player("HJ").stack == stack_before
    assert hand.sequence == sequence_before
    assert hand.current_actor_id == "HJ"

    hand.act("HJ", ActionType.RAISE, 250)
    assert hand.last_full_raise_size == 130
    assert hand.legal_actions().min_raise_to == 380


def test_short_all_in_does_not_reopen_action_but_can_be_called(six_seats):
    seats = six_seats({Position.HJ: 150})
    hand = HoldemHand(seats, seed=14)
    hand.act("UTG", ActionType.RAISE, 100)  # 完整加注增量60
    hand.act("HJ", ActionType.ALL_IN)  # 到150，仅增加50
    assert hand.current_bet == 150
    assert hand.last_full_raise_size == 60

    hand.act("CO", ActionType.CALL)
    hand.act("BTN", ActionType.FOLD)
    hand.act("SB", ActionType.FOLD)
    hand.act("BB", ActionType.FOLD)

    assert hand.current_actor_id == "UTG"
    legal = hand.legal_actions()
    assert legal.to_call == 50
    assert not legal.raise_reopened
    assert not legal.can_raise
    hand.act("UTG", ActionType.CALL)
    assert hand.street == Street.FLOP


def test_cumulative_short_all_ins_can_reopen_action(six_seats):
    seats = six_seats({Position.HJ: 150, Position.CO: 200})
    hand = HoldemHand(seats, seed=15)
    hand.act("UTG", ActionType.RAISE, 100)  # last full raise = 60
    hand.act("HJ", ActionType.ALL_IN)  # +50
    hand.act("CO", ActionType.ALL_IN)  # 再+50，累计面对100
    hand.act("BTN", ActionType.CALL)
    hand.act("SB", ActionType.FOLD)
    hand.act("BB", ActionType.FOLD)

    assert hand.current_actor_id == "UTG"
    legal = hand.legal_actions()
    assert legal.raise_reopened
    assert legal.can_raise
    assert legal.min_raise_to == 260


def test_postflop_short_open_all_in_has_no_nlhe_completion(six_seats):
    """无限注中20短开全下可跟20，但未行动者最小加注是再加40到60。"""

    hand = HoldemHand(six_seats({Position.BB: 60}), seed=151)
    for player_id, action in (
        ("UTG", ActionType.CALL),
        ("HJ", ActionType.FOLD),
        ("CO", ActionType.FOLD),
        ("BTN", ActionType.FOLD),
        ("SB", ActionType.CALL),
        ("BB", ActionType.CHECK),
        ("SB", ActionType.CHECK),
        ("BB", ActionType.ALL_IN),
    ):
        hand.act(player_id, action)

    unacted = hand.legal_actions("UTG")
    assert unacted.to_call == 20
    assert unacted.min_raise_to == 60
    assert unacted.can_raise
    hand.act("UTG", ActionType.CALL)

    checked = hand.legal_actions("SB")
    assert checked.to_call == 20
    assert not checked.raise_reopened
    assert not checked.can_raise


def test_all_in_call_never_reduces_current_bet(six_seats):
    seats = six_seats({Position.HJ: 70})
    hand = HoldemHand(seats, seed=16)
    hand.act("UTG", ActionType.RAISE, 120)
    hand.act("HJ", ActionType.ALL_IN)
    assert hand.player("HJ").stack == 0
    assert hand.player("HJ").street_commitment == 70
    assert hand.current_bet == 120
    assert hand.current_actor_id == "CO"


def test_no_dry_side_pot_betting_and_board_runs_out(six_seats):
    seats = six_seats({Position.UTG: 100, Position.HJ: 100})
    hand = HoldemHand(seats, seed=17)
    hand.act("UTG", ActionType.ALL_IN)
    hand.act("HJ", ActionType.ALL_IN)
    for player_id in ("CO", "BTN", "SB", "BB"):
        hand.act(player_id, ActionType.FOLD)
    assert hand.is_complete
    assert len(hand.board) == 5
    assert hand.result.reason == "showdown"
    hand.assert_chip_conservation()


def test_showdown_compares_best_hand_and_folded_hand_cannot_win(six_seats):
    overrides = {
        "UTG": ["As", "Ad"],
        "HJ": ["Js", "Jd"],
        "BB": ["Ks", "Kd"],
    }
    board = ["2h", "3d", "7c", "9s", "Th"]
    hand = HoldemHand(
        six_seats(), seed=18, hole_overrides=overrides, board_override=board
    )
    hand.act("UTG", ActionType.CALL)
    hand.act("HJ", ActionType.FOLD)
    hand.act("CO", ActionType.FOLD)
    hand.act("BTN", ActionType.FOLD)
    hand.act("SB", ActionType.FOLD)
    hand.act("BB", ActionType.CHECK)
    while not hand.is_complete:
        passive_action(hand)

    assert hand.result.reason == "showdown"
    assert hand.result.payouts["UTG"] == 100
    assert hand.result.payouts["BB"] == 0
    assert hand.result.payouts["HJ"] == 0
    assert hand.player("UTG").stack == 4060
    assert hand.player("BB").stack == 3960
    hand.assert_chip_conservation()


def test_end_to_end_multiway_main_and_side_pots_pay_different_winners(six_seats):
    """主池、两个边池必须分别只在仍有资格的玩家之间结算。"""

    hand = HoldemHand(
        six_seats(
            {
                Position.UTG: 100,
                Position.HJ: 300,
                Position.CO: 500,
                Position.BTN: 500,
            }
        ),
        seed=19,
        hole_overrides={
            "UTG": ["As", "Ad"],
            "HJ": ["Ks", "Kd"],
            "CO": ["Qs", "Qd"],
            "BTN": ["Js", "Jd"],
        },
        board_override=["2c", "3d", "7h", "8s", "9c"],
    )

    for player_id, action in (
        ("UTG", ActionType.ALL_IN),
        ("HJ", ActionType.ALL_IN),
        ("CO", ActionType.ALL_IN),
        ("BTN", ActionType.ALL_IN),
        ("SB", ActionType.FOLD),
        ("BB", ActionType.FOLD),
    ):
        hand.act(player_id, action)

    assert hand.is_complete
    assert hand.result.payouts == {
        "UTG": 460,
        "HJ": 600,
        "CO": 400,
        "BTN": 0,
        "SB": 0,
        "BB": 0,
    }

    # 盲注弃牌会把同一主池拆成相邻层；按获奖资格聚合后应为
    # 四人主池460、三人边池600、两人边池400。
    by_eligible: dict[frozenset[str], int] = {}
    for pot in hand.result.pots:
        key = frozenset(pot.eligible)
        by_eligible[key] = by_eligible.get(key, 0) + pot.amount
    assert by_eligible == {
        frozenset({"UTG", "HJ", "CO", "BTN"}): 460,
        frozenset({"HJ", "CO", "BTN"}): 600,
        frozenset({"CO", "BTN"}): 400,
    }
    hand.assert_chip_conservation()

