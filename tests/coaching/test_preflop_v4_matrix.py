from __future__ import annotations

import pytest

from poker_trainer.coaching.coach import (
    ActionRating,
    PREFLOP_HEURISTIC_VERSION,
    capture_context,
    review_decision,
)
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType


V4_VERSION = "6max-100bb-preflop-v4"


def _act(
    hand: HoldemHand,
    player_id: str,
    action: ActionType,
    amount: int | None = None,
):
    assert hand.current_actor_id == player_id
    return hand.act(player_id, action, amount)


def _review(
    hand: HoldemHand,
    player_id: str,
    action: ActionType,
    amount: int | None = None,
):
    context = capture_context(hand, player_id)
    record = _act(hand, player_id, action, amount)
    return review_decision(context, record, trials=10)


def _unopened_button(six_seats, *, hole_cards: str, seed: int) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"BTN": hole_cards},
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    assert hand.current_actor_id == "BTN"
    return hand


def _unopened_cutoff(six_seats, *, hole_cards: str, seed: int) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"CO": hole_cards},
    )
    for player_id in ("UTG", "HJ"):
        _act(hand, player_id, ActionType.FOLD)
    assert hand.current_actor_id == "CO"
    return hand


def _sb_facing_three_limpers(
    six_seats,
    *,
    hole_cards: str,
    seed: int,
) -> HoldemHand:
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


def _button_facing_three_limpers(
    six_seats,
    *,
    hole_cards: str,
    seed: int,
) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"BTN": hole_cards},
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.CALL)
    assert hand.current_actor_id == "BTN"
    return hand


def _cutoff_facing_hj_open(
    six_seats,
    *,
    hole_cards: str,
    seed: int,
) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=seed,
        hole_overrides={"CO": hole_cards},
    )
    _act(hand, "UTG", ActionType.FOLD)
    _act(hand, "HJ", ActionType.RAISE, 120)
    assert hand.current_actor_id == "CO"
    return hand


def test_preflop_matrix_uses_v4_version() -> None:
    assert PREFLOP_HEURISTIC_VERSION == V4_VERSION


def test_k9o_button_unopened_normal_open_is_at_least_acceptable(six_seats) -> None:
    hand = _unopened_button(six_seats, hole_cards="Ks 9h", seed=401)

    review = _review(hand, "BTN", ActionType.RAISE, 120)

    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert review.recommended_action is not None
    assert "加注" in review.recommended_action
    assert "弃牌" not in review.recommended_action


def test_96o_button_unopened_is_a_fold_not_a_loose_open(six_seats) -> None:
    folding = _unopened_button(six_seats, hole_cards="9s 6h", seed=402)
    fold_review = _review(folding, "BTN", ActionType.FOLD)

    assert fold_review.rating == ActionRating.RECOMMENDED
    assert fold_review.recommended_action == "弃牌"

    opening = _unopened_button(six_seats, hole_cards="9s 6h", seed=403)
    open_review = _review(opening, "BTN", ActionType.RAISE, 120)

    assert open_review.rating == ActionRating.CLEAR_ERROR
    assert open_review.recommended_action == "弃牌"


def test_q9s_cutoff_is_at_least_a_marginal_open(six_seats) -> None:
    hand = _unopened_cutoff(six_seats, hole_cards="Qs 9s", seed=404)

    review = _review(hand, "CO", ActionType.RAISE, 120)

    assert review.rating in {
        ActionRating.RECOMMENDED,
        ActionRating.ACCEPTABLE,
        ActionRating.LOOSE_OR_TIGHT,
    }
    assert review.recommended_action is not None
    assert "加注" in review.recommended_action
    assert "弃牌" not in review.recommended_action


def test_q9s_utg_defaults_to_folding(six_seats) -> None:
    hand = HoldemHand(
        six_seats(),
        seed=405,
        hole_overrides={"UTG": "Qs 9s"},
    )

    review = _review(hand, "UTG", ActionType.FOLD)

    assert review.rating == ActionRating.RECOMMENDED
    assert review.recommended_action == "弃牌"


@pytest.mark.parametrize(
    ("hole_cards", "seed"),
    (("2s 2h", 406), ("5s 5h", 407)),
    ids=("22", "55"),
)
def test_sb_small_pair_facing_multiple_limpers_recommends_completing(
    six_seats,
    hole_cards: str,
    seed: int,
) -> None:
    hand = _sb_facing_three_limpers(
        six_seats,
        hole_cards=hole_cards,
        seed=seed,
    )

    review = _review(hand, "SB", ActionType.CALL)

    assert review.rating == ActionRating.RECOMMENDED
    assert review.recommended_action is not None
    assert "补齐" in review.recommended_action
    assert "加注" not in review.recommended_action


@pytest.mark.parametrize(
    ("hole_cards", "seed"),
    (("2s 2h", 408), ("5s 5h", 409)),
    ids=("22", "55"),
)
def test_sb_small_pair_facing_multiple_limpers_does_not_recommend_isolation(
    six_seats,
    hole_cards: str,
    seed: int,
) -> None:
    hand = _sb_facing_three_limpers(
        six_seats,
        hole_cards=hole_cards,
        seed=seed,
    )

    review = _review(hand, "SB", ActionType.RAISE, 280)

    assert review.rating in {ActionRating.LOOSE_OR_TIGHT, ActionRating.CLEAR_ERROR}
    assert review.recommended_action is not None
    assert "补齐" in review.recommended_action
    assert "加注" not in review.recommended_action


@pytest.mark.parametrize(
    ("hole_cards", "seed"),
    (("2s 2h", 410), ("5s 5h", 411)),
    ids=("22", "55"),
)
def test_button_small_pair_can_overlimp_after_multiple_limpers(
    six_seats,
    hole_cards: str,
    seed: int,
) -> None:
    hand = _button_facing_three_limpers(
        six_seats,
        hole_cards=hole_cards,
        seed=seed,
    )

    review = _review(hand, "BTN", ActionType.CALL)

    assert review.rating in {ActionRating.RECOMMENDED, ActionRating.ACCEPTABLE}
    assert review.recommended_action is not None
    assert any(word in review.recommended_action for word in ("跟注", "补入"))


@pytest.mark.parametrize(
    ("action", "amount", "seed"),
    (
        (ActionType.RAISE, 360, 412),
        (ActionType.CALL, None, 413),
        (ActionType.FOLD, None, 414),
    ),
    ids=("threebet", "call", "fold"),
)
def test_a5s_facing_hj_open_allows_a_mixed_response_without_blanket_errors(
    six_seats,
    action: ActionType,
    amount: int | None,
    seed: int,
) -> None:
    hand = _cutoff_facing_hj_open(
        six_seats,
        hole_cards="As 5s",
        seed=seed,
    )

    review = _review(hand, "CO", action, amount)

    assert review.rating != ActionRating.CLEAR_ERROR
    if action == ActionType.RAISE:
        assert review.recommended_action is not None
        assert "加注" in review.recommended_action


@pytest.mark.parametrize(
    ("action", "amount", "expected_rating", "seed"),
    (
        (ActionType.FOLD, None, ActionRating.RECOMMENDED, 415),
        (ActionType.CALL, None, ActionRating.CLEAR_ERROR, 416),
        (ActionType.RAISE, 360, ActionRating.CLEAR_ERROR, 417),
    ),
    ids=("fold", "call", "threebet"),
)
def test_96o_facing_an_open_should_fold(
    six_seats,
    action: ActionType,
    amount: int | None,
    expected_rating: ActionRating,
    seed: int,
) -> None:
    hand = _cutoff_facing_hj_open(
        six_seats,
        hole_cards="9s 6h",
        seed=seed,
    )

    review = _review(hand, "CO", action, amount)

    assert review.rating == expected_rating
    assert review.recommended_action == "弃牌"


def _premium_spot(
    six_seats,
    *,
    hole_cards: str,
    scenario: str,
    seed: int,
) -> tuple[HoldemHand, str, int]:
    if scenario == "unopened":
        return (
            HoldemHand(
                six_seats(),
                seed=seed,
                hole_overrides={"UTG": hole_cards},
            ),
            "UTG",
            120,
        )
    if scenario == "multiple_limpers":
        return (
            _sb_facing_three_limpers(
                six_seats,
                hole_cards=hole_cards,
                seed=seed,
            ),
            "SB",
            280,
        )
    if scenario == "facing_open":
        return (
            _cutoff_facing_hj_open(
                six_seats,
                hole_cards=hole_cards,
                seed=seed,
            ),
            "CO",
            360,
        )
    raise AssertionError(f"unknown premium scenario: {scenario}")


@pytest.mark.parametrize("hole_cards", ("As Ah", "Ks Kh"), ids=("AA", "KK"))
@pytest.mark.parametrize(
    ("scenario", "seed"),
    (("unopened", 418), ("multiple_limpers", 419), ("facing_open", 420)),
)
def test_aa_kk_take_the_aggressive_line_in_every_preflop_scenario(
    six_seats,
    hole_cards: str,
    scenario: str,
    seed: int,
) -> None:
    hand, player_id, raise_to = _premium_spot(
        six_seats,
        hole_cards=hole_cards,
        scenario=scenario,
        seed=seed,
    )

    review = _review(hand, player_id, ActionType.RAISE, raise_to)

    assert review.rating == ActionRating.RECOMMENDED
    assert review.recommended_action is not None
    assert "加注" in review.recommended_action


@pytest.mark.parametrize("hole_cards", ("As Ah", "Ks Kh"), ids=("AA", "KK"))
@pytest.mark.parametrize(
    ("scenario", "passive_action", "seed"),
    (
        ("unopened", ActionType.FOLD, 421),
        ("multiple_limpers", ActionType.CALL, 422),
        ("facing_open", ActionType.CALL, 423),
    ),
)
def test_aa_kk_passive_lines_still_recommend_the_aggressive_action(
    six_seats,
    hole_cards: str,
    scenario: str,
    passive_action: ActionType,
    seed: int,
) -> None:
    hand, player_id, _raise_to = _premium_spot(
        six_seats,
        hole_cards=hole_cards,
        scenario=scenario,
        seed=seed,
    )

    review = _review(hand, player_id, passive_action)

    assert review.rating in {ActionRating.LOOSE_OR_TIGHT, ActionRating.CLEAR_ERROR}
    assert review.recommended_action is not None
    assert "加注" in review.recommended_action
