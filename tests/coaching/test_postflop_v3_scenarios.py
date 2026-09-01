from __future__ import annotations

import pytest

from poker_trainer.coaching.coach import (
    ActionRating,
    DRAW_ODDS_ERROR,
    DRAW_ODDS_OPPORTUNITY,
    SMALL_PAIR_OVERCONTINUE,
    STRONG_DRAW_DECISION,
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


def _heads_up_to_flop(hand: HoldemHand) -> None:
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, 120)
    _act(hand, "SB", ActionType.FOLD)
    _act(hand, "BB", ActionType.CALL)
    assert hand.current_actor_id == "BB"


def _three_way_to_flop(hand: HoldemHand) -> None:
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, 120)
    _act(hand, "SB", ActionType.CALL)
    _act(hand, "BB", ActionType.CALL)
    assert hand.current_actor_id == "SB"


def _heads_up_to_river(hand: HoldemHand) -> None:
    _heads_up_to_flop(hand)
    for _street in range(2):
        _act(hand, "BB", ActionType.CHECK)
        _act(hand, "BTN", ActionType.CHECK)
    assert hand.current_actor_id == "BB"


def _three_way_to_river(hand: HoldemHand) -> None:
    _three_way_to_flop(hand)
    for _street in range(2):
        _act(hand, "SB", ActionType.CHECK)
        _act(hand, "BB", ActionType.CHECK)
        _act(hand, "BTN", ActionType.CHECK)
    assert hand.current_actor_id == "SB"


def _review_btn_response(
    hand: HoldemHand,
    action: ActionType,
    *,
    trials: int = 1_200,
):
    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", action)
    return review_decision(context, record, trials=trials)


def test_folding_when_check_is_available_is_always_a_clear_error(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=301,
        hole_overrides={"BTN": "7c 2d"},
        board_override="As Kd Qh 9s 4c",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.CHECK)

    review = _review_btn_response(hand, ActionType.FOLD, trials=200)

    assert review.rating == ActionRating.CLEAR_ERROR
    assert review.recommended_action == "过牌"
    assert "无需弃牌" in review.reason


def test_river_pure_air_folds_to_a_normal_bet_by_pot_odds(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=302,
        hole_overrides={"BTN": "7c 2d"},
        board_override="As Kd Qh 9s 4c",
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 160)

    review = _review_btn_response(hand, ActionType.FOLD)

    assert review.rating == ActionRating.RECOMMENDED
    assert review.recommended_action == "弃牌"
    assert review.equity is not None and review.pot_odds is not None
    assert review.equity < review.pot_odds
    assert review.outs == 0
    assert review.hit_probability == 0.0
    assert review.draw_names == ()
    assert "河牌" in review.reason and "空气" in review.reason


def test_weak_top_pair_does_not_call_a_large_bet_on_random_equity_alone(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=303,
        hole_overrides={"BTN": "Ks 4s"},
        board_override="Kh 9d 2c 7s Qc",
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 600)

    review = _review_btn_response(hand, ActionType.CALL)

    assert review.rating in {
        ActionRating.LOOSE_OR_TIGHT,
        ActionRating.CLEAR_ERROR,
    }
    assert review.recommended_action == "弃牌"
    assert {WEAK_TOP_PAIR_DECISION, WEAK_TOP_PAIR_OVERCALL}.issubset(
        review.reason_codes
    )
    assert "随机" not in review.reason
    assert "弱顶对" in review.reason and "大" in review.reason


@pytest.mark.parametrize(
    ("hole_cards", "board"),
    (
        ("Ks 9s", "Kh 9d 2c 7h Qc"),
        ("9s 9c", "Kh 9d 2c 7h Qc"),
    ),
    ids=("two-pair", "set"),
)
def test_two_pair_and_set_continue_against_a_normal_bet(
    six_seats,
    hole_cards: str,
    board: str,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=304,
        hole_overrides={"BTN": hole_cards},
        board_override=board,
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.BET, 160)

    review = _review_btn_response(hand, ActionType.CALL)

    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert review.recommended_action in {"跟注", "跟注或加注", "加注"}
    assert "弃牌" not in (review.recommended_action or "")


def _strong_flush_draw_faces_bet(six_seats, *, seed: int, amount: int) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"BTN": "As Qs"},
        board_override="Js 8d 2s 4c 6h",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.BET, amount)
    return hand


def test_strong_draw_continues_against_a_small_bet(six_seats) -> None:
    hand = _strong_flush_draw_faces_bet(six_seats, seed=305, amount=80)

    review = _review_btn_response(hand, ActionType.CALL)

    assert review.rating == ActionRating.RECOMMENDED
    assert review.outs == 9
    assert review.recommended_action in {"跟注", "跟注或加注", "加注"}
    assert {DRAW_ODDS_OPPORTUNITY, STRONG_DRAW_DECISION}.issubset(
        review.reason_codes
    )
    assert DRAW_ODDS_ERROR not in review.reason_codes


def test_strong_draw_may_fold_to_a_massive_overbet(six_seats) -> None:
    hand = _strong_flush_draw_faces_bet(six_seats, seed=306, amount=1_200)

    review = _review_btn_response(hand, ActionType.FOLD)

    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert review.recommended_action == "弃牌"
    assert review.outs == 9
    assert review.equity is not None and review.pot_odds is not None
    # review.equity 是随机未知手牌基准，不能冒充对手超池下注范围；
    # 此处真正约束决策的是听牌命中率与价格。
    assert review.hit_probability is not None
    assert review.hit_probability < review.pot_odds
    assert STRONG_DRAW_DECISION in review.reason_codes
    assert DRAW_ODDS_ERROR not in review.reason_codes
    assert "赔率" in review.reason or "超池" in review.reason


def _middle_pair_probe_bet_review(six_seats, *, multiway: bool, seed: int):
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"BTN": "Qs 9s"},
        board_override="Kh 9d 2c 7h 4c",
    )
    if multiway:
        _three_way_to_flop(hand)
        _act(hand, "SB", ActionType.CHECK)
        _act(hand, "BB", ActionType.CHECK)
        bet_to = 240
    else:
        _heads_up_to_flop(hand)
        _act(hand, "BB", ActionType.CHECK)
        bet_to = 160
    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.BET, bet_to)
    return review_decision(context, record, trials=1_600)


def test_same_middle_pair_is_coached_more_conservatively_multiway(six_seats) -> None:
    heads_up = _middle_pair_probe_bet_review(six_seats, multiway=False, seed=307)
    multiway = _middle_pair_probe_bet_review(six_seats, multiway=True, seed=308)

    rating_order = {
        ActionRating.RECOMMENDED: 0,
        ActionRating.ACCEPTABLE: 1,
        ActionRating.LOOSE_OR_TIGHT: 2,
        ActionRating.CLEAR_ERROR: 3,
    }
    assert heads_up.equity is not None and multiway.equity is not None
    assert multiway.equity < heads_up.equity
    assert rating_order[multiway.rating] > rating_order[heads_up.rating]
    assert multiway.recommended_action in {"过牌", "下注或过牌"}
    assert "多人" in multiway.reason or any(
        "多人" in line for line in multiway.detail_lines
    )


@pytest.mark.parametrize(
    ("hole_cards", "board"),
    (
        ("As Qs", "Js 8d 2s 4c 6h"),
        ("Th 9h", "8c 7d 2s Ks 3c"),
    ),
    ids=("four-flush-visible", "straight-shape-visible"),
)
def test_river_never_reports_future_outs(
    six_seats,
    hole_cards: str,
    board: str,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=309,
        hole_overrides={"BTN": hole_cards},
        board_override=board,
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 120)

    review = _review_btn_response(hand, ActionType.FOLD, trials=400)

    assert review.outs == 0
    assert review.hit_probability == 0.0
    assert review.draw_names == ()
    assert DRAW_ODDS_OPPORTUNITY not in review.reason_codes
    assert DRAW_ODDS_ERROR not in review.reason_codes
    assert all("命中率" not in line for line in review.detail_lines)


def test_flopped_top_set_check_is_not_recommended_as_the_default_line(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=310,
        hole_overrides={"BTN": "Qs Qd"},
        board_override="Qh 7c 2s 4d 6c",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.CHECK)

    review = _review_btn_response(hand, ActionType.CHECK, trials=600)

    assert review.rating in {
        ActionRating.ACCEPTABLE,
        ActionRating.LOOSE_OR_TIGHT,
    }
    assert review.recommended_action == "下注"
    assert "价值" in review.reason


def test_strong_flush_draw_jam_over_one_third_pot_is_not_recommended(
    six_seats,
) -> None:
    hand = _strong_flush_draw_faces_bet(six_seats, seed=311, amount=80)

    review = _review_btn_response(hand, ActionType.ALL_IN, trials=1_200)

    assert review.outs == 9
    assert STRONG_DRAW_DECISION in review.reason_codes
    assert review.rating != ActionRating.RECOMMENDED
    assert review.recommended_action in {"跟注", "跟注或加注", "小尺度加注"}
    assert "全下" in review.reason or "尺度" in review.reason


def test_small_pair_can_call_a_four_percent_pot_probe_on_the_turn(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=312,
        hole_overrides={"BTN": "9s 9d"},
        board_override="Th 7d 2c 2s 4c",
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, 500)
    _act(hand, "SB", ActionType.FOLD)
    _act(hand, "BB", ActionType.CALL)
    _act(hand, "BB", ActionType.CHECK)
    _act(hand, "BTN", ActionType.CHECK)
    assert hand.pot_size == 1_020
    _act(hand, "BB", ActionType.BET, 40)

    review = _review_btn_response(hand, ActionType.CALL, trials=1_000)

    assert review.pot_odds is not None and review.pot_odds < 0.04
    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert review.recommended_action == "跟注"
    assert SMALL_PAIR_OVERCONTINUE not in review.reason_codes
    assert "继续过多" not in review.reason


def _weak_top_pair_call_review(
    six_seats,
    *,
    multiway: bool,
    seed: int,
):
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"BTN": "Kd 4c"},
        board_override=("Ks Qs Js 7h 2c" if multiway else "Kh 7d 2c 8s Qc"),
    )
    if multiway:
        _three_way_to_flop(hand)
        _act(hand, "SB", ActionType.CHECK)
        _act(hand, "BB", ActionType.BET, 270)
    else:
        _heads_up_to_flop(hand)
        _act(hand, "BB", ActionType.BET, 180)
    return _review_btn_response(hand, ActionType.CALL, trials=1_600)


def test_weak_top_pair_is_more_conservative_multiway_on_a_wet_board(
    six_seats,
) -> None:
    heads_up = _weak_top_pair_call_review(
        six_seats,
        multiway=False,
        seed=313,
    )
    multiway = _weak_top_pair_call_review(
        six_seats,
        multiway=True,
        seed=314,
    )
    rating_order = {
        ActionRating.RECOMMENDED: 0,
        ActionRating.ACCEPTABLE: 1,
        ActionRating.LOOSE_OR_TIGHT: 2,
        ActionRating.CLEAR_ERROR: 3,
    }

    assert rating_order[multiway.rating] > rating_order[heads_up.rating]
    assert multiway.recommended_action == "弃牌"
    multiway_text = " ".join((multiway.reason, *multiway.detail_lines))
    assert "多人" in multiway_text
    assert "湿润" in multiway_text or "协同" in multiway_text or "听牌" in multiway_text


def test_draw_details_call_out_raw_outs_instead_of_unadjusted_outs(
    six_seats,
) -> None:
    hand = _strong_flush_draw_faces_bet(six_seats, seed=315, amount=80)

    review = _review_btn_response(hand, ActionType.CALL, trials=500)

    outs_lines = [line for line in review.detail_lines if "outs" in line]
    assert outs_lines
    assert all("未折损" not in line for line in outs_lines)
    assert any(
        "原始" in line or "未扣脏 outs" in line
        for line in outs_lines
    )


def test_public_royal_flush_is_a_profitable_chop_call_not_river_air(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=316,
        hole_overrides={"BTN": "7c 2d"},
        board_override="As Ks Qs Js Ts",
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 200)

    review = _review_btn_response(hand, ActionType.CALL, trials=300)

    assert review.equity == pytest.approx(0.5)
    assert review.pot_odds is not None and review.equity > review.pot_odds
    assert review.rating == ActionRating.RECOMMENDED
    assert review.recommended_action == "跟注"
    review_text = " ".join((review.reason, *review.detail_lines))
    assert "空气" not in review_text
    assert "公牌" in review_text and ("平分" in review_text or "同牌" in review_text)


def test_ace_king_kickers_on_a_paired_board_are_not_pure_air(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=317,
        hole_overrides={"BTN": "Ah Kd"},
        board_override="7s 7d 2c 3h 4s",
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 40)

    review = _review_btn_response(hand, ActionType.CALL, trials=800)

    assert review.rating != ActionRating.CLEAR_ERROR
    assert review.recommended_action in {"跟注", "跟注或弃牌"}
    review_text = " ".join((review.reason, *review.detail_lines))
    assert "空气" not in review_text
    assert "踢脚" in review_text


def test_set_without_a_spade_does_not_auto_call_multiway_four_flush_overbet(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=318,
        hole_overrides={"BTN": "9c 9d"},
        board_override="9s As Ks 2s 4d",
    )
    _three_way_to_river(hand)
    _act(hand, "SB", ActionType.CHECK)
    _act(hand, "BB", ActionType.BET, 1_200)

    review = _review_btn_response(hand, ActionType.CALL, trials=1_200)

    assert review.rating in {
        ActionRating.LOOSE_OR_TIGHT,
        ActionRating.CLEAR_ERROR,
    }
    assert review.recommended_action == "弃牌"
    review_text = " ".join((review.reason, *review.detail_lines))
    assert "多人" in review_text
    assert "四同花" in review_text
    assert "超池" in review_text


def test_weak_top_pair_plus_flush_draw_does_not_fold_to_a_normal_bet(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=319,
        hole_overrides={"BTN": "Kh 4h"},
        board_override="Kd 9h 2h 7c Qs",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.BET, 120)

    review = _review_btn_response(hand, ActionType.CALL, trials=900)

    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert "弃牌" not in (review.recommended_action or "")
    assert review.outs >= 8
    assert DRAW_ODDS_ERROR not in review.reason_codes
    review_text = " ".join((review.reason, *review.detail_lines))
    assert "弱顶对" in review_text
    assert "听牌" in review_text


def test_short_stack_flop_call_all_in_uses_probability_to_the_river(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats({Position.BTN: 320}),
        seed=320,
        hole_overrides={"BTN": "As Qs"},
        board_override="Js 8d 2s 4c 6h",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.BET, 200)
    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.CALL)
    assert record.is_all_in

    review = review_decision(context, record, trials=900)

    assert review.hit_probability is not None and review.hit_probability > 0.30
    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert review.recommended_action == "跟注"
    assert DRAW_ODDS_ERROR not in review.reason_codes
    assert any("到河牌" in line for line in review.detail_lines)


def test_fold_with_only_implied_odds_support_is_not_a_clear_error(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=321,
        hole_overrides={"BTN": "8s 7d"},
        board_override="Ks 6c 4h Qd 2c",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.BET, 40)

    review = _review_btn_response(hand, ActionType.FOLD, trials=700)

    assert review.outs == 4
    assert review.pot_odds is not None
    assert review.hit_probability is not None
    next_card_probability = review.outs / 47
    assert next_card_probability < review.pot_odds
    assert next_card_probability + 0.06 >= review.pot_odds
    assert review.rating != ActionRating.CLEAR_ERROR
    assert review.rating in {
        ActionRating.ACCEPTABLE,
        ActionRating.LOOSE_OR_TIGHT,
    }
    assert "隐含赔率" in " ".join((review.reason, *review.detail_lines))


def test_semibluff_raise_does_not_emit_a_calling_odds_error(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=322,
        hole_overrides={"BTN": "8s 7d"},
        board_override="Ks 6c 4h Qd 2c",
    )
    _heads_up_to_flop(hand)
    _act(hand, "BB", ActionType.BET, 240)
    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.RAISE, 720)
    review = review_decision(context, record, trials=800)

    assert review.outs == 4
    assert DRAW_ODDS_OPPORTUNITY in review.reason_codes
    assert DRAW_ODDS_ERROR not in review.reason_codes
    assert review.rating in {
        ActionRating.LOOSE_OR_TIGHT,
        ActionRating.CLEAR_ERROR,
    }
    assert "半诈唬" in " ".join((review.reason, *review.detail_lines))


def test_river_one_pair_overjam_cannot_downgrade_an_existing_clear_error(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=323,
        hole_overrides={"BTN": "Ah Qc"},
        board_override="Qd Jh 7c 2s 4d",
    )
    _heads_up_to_river(hand)
    _act(hand, "BB", ActionType.BET, 300)

    review = _review_btn_response(hand, ActionType.ALL_IN, trials=900)

    assert review.rating == ActionRating.CLEAR_ERROR
    assert review.recommended_action == "跟注"
    assert "全下" in " ".join((review.reason, *review.detail_lines))
