import random

import pytest

from poker_trainer.coaching.equity import (
    analyze_common_draws,
    calculate_pot_odds,
    estimate_equity,
)


def test_calculate_pot_odds_uses_final_pot_after_call() -> None:
    assert calculate_pot_odds(300, 100) == pytest.approx(0.25)
    assert calculate_pot_odds(300, 0) == 0.0


@pytest.mark.parametrize("pot, call", [(-1, 10), (100, -1)])
def test_calculate_pot_odds_rejects_negative_chips(pot: int, call: int) -> None:
    with pytest.raises(ValueError):
        calculate_pot_odds(pot, call)


def test_nine_out_flush_draw_and_exact_flop_probabilities() -> None:
    result = analyze_common_draws("Ah Kh", "2h 7h Qc")

    assert result.names == ("同花听牌",)
    assert result.outs == 9
    assert len(set(result.out_cards)) == 9
    assert {card.suit for card in result.out_cards} == {"h"}
    assert result.hit_next == pytest.approx(9 / 47)
    assert result.hit_by_river == pytest.approx(1 - (38 / 47) * (37 / 46))


def test_eight_out_open_ended_straight_draw() -> None:
    result = analyze_common_draws("8s 9d", "6h 7c Ks")

    assert result.names == ("两头顺听牌",)
    assert result.outs == 8
    assert {card.rank for card in result.out_cards} == {5, 10}


def test_four_out_gutshot() -> None:
    result = analyze_common_draws("8s 9d", "5h 7c Ks")

    assert result.names == ("卡顺听牌",)
    assert result.outs == 4
    assert {card.rank for card in result.out_cards} == {6}


def test_combo_draw_de_duplicates_shared_straight_flush_outs() -> None:
    result = analyze_common_draws("8h 9h", "6h 7c Kh")

    assert result.names == ("同花听牌", "两头顺听牌")
    assert result.outs == 15  # 9 flush + 8 straight - 2 shared hearts
    assert len(result.out_cards) == len(set(result.out_cards))


def test_made_hand_is_not_reported_as_same_category_draw() -> None:
    made_straight = analyze_common_draws("8s 9d", "5h 6c 7s")
    made_flush = analyze_common_draws("Ah Kh", "2h 7h Qh")

    assert not any("顺" in name for name in made_straight.names)
    assert "同花听牌" not in made_flush.names


def test_board_only_four_flush_or_straight_is_not_a_personal_draw() -> None:
    board_flush = analyze_common_draws("As Kd", "2h 7h Qh Jh")
    board_straight = analyze_common_draws("As Ad", "5h 6c 7s 8d")

    assert "同花听牌" not in board_flush.names
    assert not any("顺" in name for name in board_straight.names)


def test_equity_is_deterministic_for_same_seed() -> None:
    kwargs = {
        "hero_hole": "As Qs",
        "board": "Js 8d 2s",
        "opponents": 3,
        "trials": 300,
        "seed": 20260827,
    }
    assert estimate_equity(**kwargs) == estimate_equity(**kwargs)


def test_equity_does_not_advance_global_random_state() -> None:
    random.seed(73)
    expected = random.random()
    random.seed(73)

    estimate_equity("As Qs", "Js 8d 2s", opponents=2, trials=20, seed=9)

    assert random.random() == expected


def test_locked_nut_hand_has_full_equity() -> None:
    # Hero already holds a royal flush; no runout or opponent can tie it.
    equity = estimate_equity(
        "As Ks", "Qs Js Ts", opponents=5, trials=50, seed=17
    )

    assert equity == 1.0


def test_board_lock_tie_splits_equity_between_every_player() -> None:
    # The royal flush is entirely on the board: hero plus two opponents tie.
    equity = estimate_equity(
        "2c 3d", "Ah Kh Qh Jh Th", opponents=2, trials=40, seed=19
    )

    assert equity == pytest.approx(1 / 3)


def test_known_opponent_hole_cards_are_respected() -> None:
    equity = estimate_equity(
        "As Ad",
        "Ac Ah 2c 3d 4s",
        opponents=1,
        trials=5,
        seed=1,
        known_opponent_holes=["Kc Kd"],
    )

    assert equity == 1.0

