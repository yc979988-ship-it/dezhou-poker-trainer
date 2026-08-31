from __future__ import annotations

import json

from poker_trainer.coaching.coach import (
    ActionRating,
    DRAW_ODDS_ERROR,
    DRAW_ODDS_OPPORTUNITY,
    NOT_FULL_GTO_NOTICE,
    PREFLOP_RAISE_TOO_LARGE,
    RANDOM_EQUITY_BASIS,
    RIVER_SHOWDOWN_VALUE_OVERPLAY,
    SB_COLD_CALL,
    SB_COLD_CALL_OPPORTUNITY,
    SMALL_PAIR_POSTFLOP_DECISION,
    STRONG_DRAW_DECISION,
    STRONG_DRAW_OVERFOLD,
    THREEBET_SIZING_OPPORTUNITY,
    THREEBET_TOO_LARGE,
    THREEBET_TOO_SMALL,
    TOP_PAIR_STACKED_OFF,
    TOP_PAIR_STACKOFF_OPPORTUNITY,
    WEAK_TOP_PAIR_DECISION,
    WEAK_TOP_PAIR_OVERCALL,
    capture_context,
    review_decision,
)
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position


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
    assert STRONG_DRAW_DECISION in continue_review.reason_codes
    assert continue_review.draw_names == ("同花听牌",)
    assert continue_review.equity_basis == RANDOM_EQUITY_BASIS

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
        STRONG_DRAW_DECISION,
    }.issubset(fold_review.reason_codes)
    assert "纯强听牌" in fold_review.reason


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
    assert THREEBET_SIZING_OPPORTUNITY in review.reason_codes
    assert review.rating == ActionRating.LOOSE_OR_TIGHT
    assert "参考约 360" in review.reason
    assert review.equity is None

    correct = HoldemHand(
        six_seats(),
        seed=214,
        hole_overrides={"CO": "Qs Qd"},
    )
    _act(correct, "UTG", ActionType.FOLD)
    _act(correct, "HJ", ActionType.RAISE, 120)
    correct_context = capture_context(correct, "CO")
    correct_record = _act(correct, "CO", ActionType.RAISE, 360)
    correct_review = review_decision(correct_context, correct_record, trials=10)

    assert THREEBET_SIZING_OPPORTUNITY in correct_review.reason_codes
    assert THREEBET_TOO_SMALL not in correct_review.reason_codes


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
        WEAK_TOP_PAIR_DECISION,
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


def test_preflop_open_and_threebet_sizes_have_upper_bounds(six_seats) -> None:
    opening = HoldemHand(
        six_seats(),
        seed=208,
        hole_overrides={"UTG": "6s 6d"},
    )
    opening_context = capture_context(opening, "UTG")
    opening_record = _act(opening, "UTG", ActionType.RAISE, 400)
    opening_review = review_decision(opening_context, opening_record, trials=10)

    assert opening_review.rating == ActionRating.CLEAR_ERROR
    assert PREFLOP_RAISE_TOO_LARGE in opening_review.reason_codes
    assert opening_review.recommended_action == "加注到约 120"
    assert "投入与可赢底池不成比例" in opening_review.reason

    threebet = HoldemHand(
        six_seats(),
        seed=209,
        hole_overrides={"CO": "Ks Kd"},
    )
    _act(threebet, "UTG", ActionType.FOLD)
    _act(threebet, "HJ", ActionType.RAISE, 120)
    threebet_context = capture_context(threebet, "CO")
    threebet_record = _act(threebet, "CO", ActionType.RAISE, 960)
    threebet_review = review_decision(threebet_context, threebet_record, trials=10)

    assert threebet_review.rating == ActionRating.CLEAR_ERROR
    assert {
        PREFLOP_RAISE_TOO_LARGE,
        THREEBET_TOO_LARGE,
        THREEBET_SIZING_OPPORTUNITY,
    }.issubset(threebet_review.reason_codes)
    assert threebet_review.recommended_action == "加注到约 360"


def _sb_faces_utg_open_and_caller(six_seats, *, seed: int) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"SB": "Qh Th"},
    )
    _act(hand, "UTG", ActionType.RAISE, 160)
    _act(hand, "HJ", ActionType.FOLD)
    _act(hand, "CO", ActionType.CALL)
    _act(hand, "BTN", ActionType.FOLD)
    assert hand.current_actor_id == "SB"
    return hand


def test_sb_cold_call_is_position_aware_and_has_a_real_opportunity_denominator(
    six_seats,
) -> None:
    calling = _sb_faces_utg_open_and_caller(six_seats, seed=210)
    call_context = capture_context(calling, "SB")
    call_record = _act(calling, "SB", ActionType.CALL)
    call_review = review_decision(call_context, call_record, trials=10)

    assert call_review.rating == ActionRating.LOOSE_OR_TIGHT
    assert {SB_COLD_CALL_OPPORTUNITY, SB_COLD_CALL}.issubset(
        call_review.reason_codes
    )
    assert call_review.recommended_action == "弃牌或加注"
    assert "位置差" in call_review.reason

    folding = _sb_faces_utg_open_and_caller(six_seats, seed=211)
    fold_context = capture_context(folding, "SB")
    fold_record = _act(folding, "SB", ActionType.FOLD)
    fold_review = review_decision(fold_context, fold_record, trials=10)

    assert fold_review.rating == ActionRating.RECOMMENDED
    assert SB_COLD_CALL_OPPORTUNITY in fold_review.reason_codes
    assert SB_COLD_CALL not in fold_review.reason_codes


def test_sb_limp_call_is_not_mislabeled_as_a_cold_call(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=212,
        hole_overrides={"SB": "7h 6h"},
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.CALL)
    _act(hand, "SB", ActionType.CALL)
    _act(hand, "BB", ActionType.RAISE, 160)
    _act(hand, "BTN", ActionType.CALL)

    context = capture_context(hand, "SB")
    record = _act(hand, "SB", ActionType.CALL)
    review = review_decision(context, record, trials=10)

    assert SB_COLD_CALL_OPPORTUNITY not in review.reason_codes
    assert SB_COLD_CALL not in review.reason_codes


def test_made_hand_plus_draw_is_not_described_or_scored_as_a_pure_draw(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=213,
        hole_overrides={"BTN": "Kh Qh"},
        board_override="Kd Th 4h 2c 6s",
    )
    _button_raises_and_bb_calls(hand)
    _act(hand, "BB", ActionType.BET, 120)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.FOLD)
    review = review_decision(context, record, trials=600)

    assert review.rating == ActionRating.LOOSE_OR_TIGHT
    assert {
        DRAW_ODDS_OPPORTUNITY,
        STRONG_DRAW_DECISION,
        STRONG_DRAW_OVERFOLD,
    }.issubset(review.reason_codes)
    assert DRAW_ODDS_ERROR not in review.reason_codes
    assert "已有一对" in review.reason
    assert "范围" in review.reason
    assert any("一对+听牌" in line for line in review.detail_lines)
    payload = review.as_dict()
    assert payload["recommended_action"] == review.recommended_action
    assert payload["draw_names"] == ["同花听牌"]
    assert payload["equity_basis"] == "random_unknown_hands"


def _heads_up_to_river(hand: HoldemHand, *, open_to: int = 120) -> None:
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, open_to)
    _act(hand, "SB", ActionType.FOLD)
    _act(hand, "BB", ActionType.CALL)
    for _street in range(2):
        _act(hand, "BB", ActionType.CHECK)
        _act(hand, "BTN", ActionType.CHECK)


def test_river_one_pair_overraise_keeps_bluffs_in_with_a_call(six_seats) -> None:
    hand = HoldemHand(
        six_seats({Position.BTN: 8_000, Position.BB: 1_000}),
        seed=215,
        hole_overrides={"BTN": "Ah Qs"},
        board_override="Qd Jh 7c 2s 4d",
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 300)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.ALL_IN)
    review = review_decision(context, record, trials=500)

    assert record.bet_to > 880  # raw all-in 大于对手能够跟注的金额
    assert RIVER_SHOWDOWN_VALUE_OVERPLAY in review.reason_codes
    assert review.rating == ActionRating.LOOSE_OR_TIGHT
    assert review.recommended_action == "跟注"
    assert "诈唬" in review.reason and "更强牌" in review.reason
    assert any("实际最多到 880" in line for line in review.detail_lines)


def test_tiny_river_block_bet_is_not_blanket_flagged_as_one_pair_overplay(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=216,
        hole_overrides={"BTN": "Ah Qs"},
        board_override="Qd Jh 7c 2s 4d",
    )
    _heads_up_to_river(hand, open_to=160)
    _act(hand, "BB", ActionType.BET, 40)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.RAISE, 120)
    review = review_decision(context, record, trials=300)

    assert RIVER_SHOWDOWN_VALUE_OVERPLAY not in review.reason_codes


def test_missed_small_pair_fold_records_the_training_opportunity(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=217,
        hole_overrides={"BTN": "5s 5d"},
        board_override="Ah 9c 4c 2d 6h",
    )
    _button_raises_and_bb_calls(hand)
    _act(hand, "BB", ActionType.BET, 120)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.FOLD)
    review = review_decision(context, record, trials=400)

    assert SMALL_PAIR_POSTFLOP_DECISION in review.reason_codes
