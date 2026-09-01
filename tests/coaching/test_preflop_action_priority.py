from __future__ import annotations

import pytest

from poker_trainer.coaching.coach import (
    ActionRating,
    PREFLOP_RAISE_TOO_SMALL,
    THREEBET_SIZING_OPPORTUNITY,
    THREEBET_TOO_SMALL,
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


def _sb_facing_three_limpers(six_seats, *, hole_cards: str, seed: int) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"SB": hole_cards},
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.CALL)
    _act(hand, "BTN", ActionType.FOLD)
    assert hand.current_actor_id == "SB"
    return hand


@pytest.mark.parametrize("raise_to", (160, 280, 400))
def test_sb_96o_bad_isolation_is_an_action_error_not_a_sizing_lesson(
    six_seats,
    raise_to: int,
) -> None:
    hand = _sb_facing_three_limpers(six_seats, hole_cards="9s 6h", seed=301)

    context = capture_context(hand, "SB")
    record = _act(hand, "SB", ActionType.RAISE, raise_to)
    review = review_decision(context, record, trials=10)

    assert review.rating == ActionRating.CLEAR_ERROR
    assert review.recommended_action == "弃牌"
    assert "弱牌隔离 3 名 limp 玩家过松" in review.reason
    assert "位置较差" in review.reason
    assert "偏小" not in review.reason
    assert "280" not in review.reason
    assert review.recommended_action != "加注到约 280"
    assert PREFLOP_RAISE_TOO_SMALL not in review.reason_codes
    assert THREEBET_SIZING_OPPORTUNITY not in review.reason_codes
    assert any("主动加注本身不合适" in line for line in review.detail_lines)
    assert all("280" not in line for line in review.detail_lines)


def test_sb_96o_complete_never_recommends_the_bad_isolation(six_seats) -> None:
    hand = _sb_facing_three_limpers(six_seats, hole_cards="9s 6h", seed=305)

    context = capture_context(hand, "SB")
    record = _act(hand, "SB", ActionType.CALL)
    review = review_decision(context, record, trials=10)

    assert review.rating == ActionRating.LOOSE_OR_TIGHT
    assert review.recommended_action == "弃牌；桌面极被动时可补齐"
    assert "加注" not in review.recommended_action


@pytest.mark.parametrize(
    ("hole_cards", "seed"),
    (("As Qs", 302), ("9s 9h", 303)),
    ids=("AQs", "99"),
)
def test_sb_valid_isolation_hand_still_gets_the_280_sizing_lesson(
    six_seats,
    hole_cards: str,
    seed: int,
) -> None:
    hand = _sb_facing_three_limpers(
        six_seats,
        hole_cards=hole_cards,
        seed=seed,
    )

    context = capture_context(hand, "SB")
    record = _act(hand, "SB", ActionType.RAISE, 160)
    review = review_decision(context, record, trials=10)

    assert review.rating == ActionRating.CLEAR_ERROR
    assert review.recommended_action == "加注到约 280"
    assert PREFLOP_RAISE_TOO_SMALL in review.reason_codes
    assert "隔离加注到 160 偏小" in review.reason
    assert "参考约 280" in review.reason


def test_weak_threebet_does_not_create_a_sizing_training_opportunity(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=304,
        hole_overrides={"CO": "9s 6h"},
    )
    _act(hand, "UTG", ActionType.FOLD)
    _act(hand, "HJ", ActionType.RAISE, 120)

    context = capture_context(hand, "CO")
    record = _act(hand, "CO", ActionType.RAISE, 240)
    review = review_decision(context, record, trials=10)

    assert review.rating == ActionRating.CLEAR_ERROR
    assert review.recommended_action == "弃牌"
    assert "弱牌面对既有加注再次加注过松" in review.reason
    assert "调整加注尺度" in review.reason
    assert PREFLOP_RAISE_TOO_SMALL not in review.reason_codes
    assert THREEBET_TOO_SMALL not in review.reason_codes
    assert THREEBET_SIZING_OPPORTUNITY not in review.reason_codes


def test_k9o_button_open_is_in_the_v4_range_and_receives_size_context(
    six_seats,
) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=306,
        hole_overrides={"BTN": "Ks 9h"},
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)

    context = capture_context(hand, "BTN")
    record = _act(hand, "BTN", ActionType.RAISE, 120)
    review = review_decision(context, record, trials=10)

    assert review.rating == ActionRating.RECOMMENDED
    assert review.recommended_action == "加注到约 120"
    assert any("牌型 K9o" in line for line in review.detail_lines)
    assert any("开池参考中点约 120" in line for line in review.detail_lines)
    assert PREFLOP_RAISE_TOO_SMALL not in review.reason_codes
