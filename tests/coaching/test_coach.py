from __future__ import annotations

import json

from poker_trainer.coaching.coach import (
    ActionRating,
    DRAW_ODDS_ERROR,
    DRAW_ODDS_OPPORTUNITY,
    NOT_FULL_GTO_NOTICE,
    STRONG_DRAW_OVERFOLD,
    THREEBET_TOO_SMALL,
    TOP_PAIR_STACKED_OFF,
    TOP_PAIR_STACKOFF_OPPORTUNITY,
    WEAK_TOP_PAIR_OVERCALL,
    capture_context,
    review_decision,
)
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType


def _act(
    hand: HoldemHand,
    player_id: str,
    action: ActionType,
    amount: int | None = None,
):
    assert hand.current_actor_id == player_id
    return hand.act(player_id, action, amount)


def _button_raises_and_bb_calls(hand: HoldemHand) -> None:
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, 120)
    _act(hand, "SB", ActionType.FOLD)
    _act(hand, "BB", ActionType.CALL)
    assert hand.current_actor_id == "BB"


def _draw_decision(six_seats, *, seed: int) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"BTN": "As Qs"},
        board_override="Js 8d 2s 4c 6h",
    )
    _button_raises_and_bb_calls(hand)
    _act(hand, "BB", ActionType.BET, 120)
    return hand


def test_negative_ev_river_call_is_clear_error(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=201,
        hole_overrides={"BTN": "7c 2d"},
        board_override="As Kd Qh 9s 4c",
    )
    _button_raises_and_bb_calls(hand)
    for _street in range(2):
        _act(hand, "BB", ActionType.CHECK)
        _act(hand, "BTN", ActionType.CHECK)
    _act(hand, "BB", ActionType.BET, 800)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.CALL)
    review = review_decision(context, record, trials=600)

    assert review.rating == ActionRating.CLEAR_ERROR
    assert review.equity is not None and review.pot_odds is not None
    assert review.equity + 0.10 < review.pot_odds
    assert "负期望" in review.reason
    json.dumps(review.as_dict(), ensure_ascii=False)


def test_strong_draw_continue_is_recommended_but_fold_is_odds_error(six_seats) -> None:
    continuing = _draw_decision(six_seats, seed=202)
    continue_context = capture_context(continuing, "BTN")
    continue_record = _act(continuing, "BTN", ActionType.CALL)
    continue_review = review_decision(continue_context, continue_record, trials=600)

    assert continue_review.outs == 9
    assert continue_review.rating == ActionRating.RECOMMENDED
    assert DRAW_ODDS_OPPORTUNITY in continue_review.reason_codes
    assert DRAW_ODDS_ERROR not in continue_review.reason_codes
    assert STRONG_DRAW_OVERFOLD not in continue_review.reason_codes

    folding = _draw_decision(six_seats, seed=203)
    fold_context = capture_context(folding, "BTN")
    fold_record = _act(folding, "BTN", ActionType.FOLD)
    fold_review = review_decision(fold_context, fold_record, trials=600)

    assert fold_review.outs == 9
    assert fold_review.rating == ActionRating.CLEAR_ERROR
    assert {
        DRAW_ODDS_OPPORTUNITY,
        DRAW_ODDS_ERROR,
        STRONG_DRAW_OVERFOLD,
    }.issubset(fold_review.reason_codes)


def test_threebet_below_eighty_five_percent_of_reference_is_flagged(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=204,
        hole_overrides={"CO": "Qs Qd"},
    )
    _act(hand, "UTG", ActionType.FOLD)
    _act(hand, "HJ", ActionType.RAISE, 120)

    context = capture_context(hand, "CO")
    record = _act(hand, "CO", ActionType.RAISE, 240)
    review = review_decision(context, record, trials=10)

    assert THREEBET_TOO_SMALL in review.reason_codes
    assert review.rating == ActionRating.LOOSE_OR_TIGHT
    assert "参考约 360" in review.reason
    assert review.equity is None


def test_weak_top_pair_allin_call_emits_stackoff_and_leak_codes(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=205,
        hole_overrides={"BTN": "Ks 4s"},
        board_override="Kh 9d 2c 7s Qc",
    )
    _button_raises_and_bb_calls(hand)
    _act(hand, "BB", ActionType.ALL_IN)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.ALL_IN)
    review = review_decision(context, record, trials=500)

    assert hand.is_complete
    assert review.rating == ActionRating.CLEAR_ERROR
    assert {
        TOP_PAIR_STACKOFF_OPPORTUNITY,
        TOP_PAIR_STACKED_OFF,
        WEAK_TOP_PAIR_OVERCALL,
    }.issubset(review.reason_codes)
    assert "过度跟注" in review.reason


def test_context_is_frozen_before_action_and_independent_of_final_result(six_seats) -> None:
    hand = _draw_decision(six_seats, seed=206)
    context = capture_context(hand, "BTN")
    payload_before = context.as_dict()

    assert len(context.snapshot.board) == 3
    assert len(context.history) == context.snapshot.sequence
    assert "result" not in payload_before
    assert all("hole_cards" not in row for row in payload_before["commitments"])

    record = _act(hand, "BTN", ActionType.FOLD)
    assert hand.is_complete and hand.result is not None
    assert context.as_dict() == payload_before
    assert len(context.snapshot.board) == 3

    first_review = review_decision(context, record, trials=400)
    # 更改结算对象所属牌局的状态不会改变相同上下文的评级。
    hand.result.payouts["BB"] = 0
    second_review = review_decision(context, record, trials=400)
    assert first_review == second_review


def test_rating_has_four_values_and_review_carries_mvp_notice() -> None:
    assert len(ActionRating) == 4
    assert "不是完整 GTO" in NOT_FULL_GTO_NOTICE


def test_pot_odds_exclude_side_pot_hero_cannot_win(six_seats) -> None:
    from poker_trainer.engine.models import Position

    hand = HoldemHand(
        six_seats({Position.UTG: 500, Position.HJ: 100}),
        seed=207,
        hole_overrides={"HJ": "As Kd"},
    )
    _act(hand, "UTG", ActionType.ALL_IN)
    context = capture_context(hand, "HJ")
    record = _act(hand, "HJ", ActionType.ALL_IN)
    review = review_decision(context, record, trials=10)

    # UTG仅有100、SB 20、BB 40和英雄100能进入英雄可争夺的主池。
    assert review.pot_odds == 100 / 260

