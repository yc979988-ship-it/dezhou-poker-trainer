from __future__ import annotations

import pytest

from poker_trainer.coaching.postflop_strategy import (
    BoardWetness,
    MadeStrength,
    analyze_postflop_profile,
)
from poker_trainer.engine.cards import parse_cards


def _profile(
    hole: str,
    board: str,
    *,
    players: int = 2,
):
    return analyze_postflop_profile(
        parse_cards(hole),
        parse_cards(board),
        active_players=players,
    )


@pytest.mark.parametrize(
    ("hole", "board", "expected"),
    (
        ("As Kd", "Ah 7d 2c", MadeStrength.TOP_PAIR_STRONG),
        ("As 4d", "Ah 7d 2c", MadeStrength.TOP_PAIR_WEAK),
        ("Qs Qd", "9h 7d 2c", MadeStrength.OVERPAIR),
        ("Qs 9d", "Kh 9c 2s", MadeStrength.MEDIUM_PAIR),
        ("5s 5d", "Kh 9c 2s", MadeStrength.WEAK_PAIR),
        ("As Kd", "7h 7d 2c", MadeStrength.BOARD_ONLY),
        ("Qs Jd", "Qh Jc 2s", MadeStrength.STRONG_MADE),
    ),
)
def test_made_strength_categories(
    hole: str,
    board: str,
    expected: MadeStrength,
) -> None:
    assert _profile(hole, board).strength == expected


def test_paired_board_plus_pocket_pair_is_not_mislabeled_as_strong_two_pair() -> None:
    profile = _profile("9s 9d", "Th 7d 2c 2s")

    assert profile.rank.name_zh == "两对"
    assert profile.strength == MadeStrength.MEDIUM_PAIR
    assert profile.texture.paired is True


def test_paired_board_plus_one_matched_hole_card_stays_top_pair() -> None:
    assert _profile("As Kd", "Ah 7d 7c 2s").strength == MadeStrength.TOP_PAIR_STRONG
    assert _profile("As 4d", "Ah 7d 7c 2s").strength == MadeStrength.TOP_PAIR_WEAK


def test_two_personal_pairs_remain_a_strong_made_hand_on_paired_board() -> None:
    profile = _profile("Qs Jd", "Qh Jc 2s 2d")

    assert profile.strength == MadeStrength.STRONG_MADE


def test_board_texture_and_player_count_are_explicit_features() -> None:
    dry = _profile("As Kd", "Kh 7d 2c")
    wet_multiway = _profile("Kd 4c", "Ks Qs Js", players=3)

    assert dry.texture.wetness == BoardWetness.DRY
    assert dry.multiway is False
    assert wet_multiway.texture.wetness == BoardWetness.WET
    assert wet_multiway.texture.three_flush is True
    assert wet_multiway.texture.connected is True
    assert wet_multiway.multiway is True
    assert wet_multiway.players_label_zh == "3 人底池"


def test_paired_broadway_flop_is_not_mislabeled_as_connected_and_wet() -> None:
    profile = _profile("2s 3d", "Ah Ad Kc")

    assert profile.texture.paired is True
    assert profile.texture.connected is False
    assert profile.texture.four_straight is False
    assert profile.texture.wetness != BoardWetness.WET


def test_river_kickers_are_separate_from_pure_board_play() -> None:
    kicker = _profile("Ah Kd", "7s 7d 2c 3h 4s")
    board_only = _profile("2d 3c", "As Ks Qs Js Ts")

    assert kicker.strength == MadeStrength.WEAK_SHOWDOWN
    assert kicker.hole_cards_play is True
    assert kicker.board_plays is False
    assert board_only.strength == MadeStrength.BOARD_ONLY
    assert board_only.hole_cards_play is False
    assert board_only.board_plays is True
    assert board_only.board_locked is True


def test_four_card_straight_and_flush_hazards_are_exposed() -> None:
    profile = _profile("9c 9d", "9s As Ks 2s 4d")
    straight_board = _profile("As 2d", "9h 8d 7c 6s Kc")

    assert profile.texture.four_flush is True
    assert straight_board.texture.four_straight is True


@pytest.mark.parametrize(
    ("hole", "board", "players", "message"),
    (
        ("As", "Kh 7d 2c", 2, "恰好两张底牌"),
        ("As Kd", "Kh 7d", 2, "三至五张公共牌"),
        ("As Kd", "Kh 7d 2c", 1, "至少需要两名"),
    ),
)
def test_profile_rejects_impossible_inputs(
    hole: str,
    board: str,
    players: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _profile(hole, board, players=players)
