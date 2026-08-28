from __future__ import annotations

import json

import pytest

from poker_trainer.analytics.statistics import (
    DRAW_ODDS_ERROR,
    DRAW_ODDS_OPPORTUNITY,
    TOP_PAIR_STACKED_OFF,
    TOP_PAIR_STACKOFF_OPPORTUNITY,
    MetricName,
    aggregate_by_position,
    project_hand_statistics,
    serialize_position_statistics,
)
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position


def _act(hand: HoldemHand, player_id: str, action: ActionType, amount: int | None = None) -> None:
    assert hand.current_actor_id == player_id
    hand.act(player_id, action, amount)


def _button_cbet_river_stackoff(six_seats) -> HoldemHand:
    hand = HoldemHand(
        six_seats(),
        seed=101,
        hand_id="btn-cbet-river-call",
        hole_overrides={"BTN": "As 9s", "BB": "Kc Qd"},
        board_override="9h 7d 2c 3s 4c",
    )
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, 120)
    _act(hand, "SB", ActionType.FOLD)
    _act(hand, "BB", ActionType.CALL)

    _act(hand, "BB", ActionType.CHECK)
    _act(hand, "BTN", ActionType.BET, 160)
    _act(hand, "BB", ActionType.CALL)
    _act(hand, "BB", ActionType.CHECK)
    _act(hand, "BTN", ActionType.CHECK)
    _act(hand, "BB", ActionType.ALL_IN)
    _act(hand, "BTN", ActionType.ALL_IN)
    assert hand.is_complete
    return hand


def _small_blind_cold_call_fold_flop(six_seats) -> HoldemHand:
    hand = HoldemHand(six_seats(), seed=102, hand_id="sb-cold-call")
    _act(hand, "UTG", ActionType.RAISE, 120)
    for player_id in ("HJ", "CO", "BTN"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "SB", ActionType.CALL)
    _act(hand, "BB", ActionType.FOLD)
    _act(hand, "SB", ActionType.CHECK)
    _act(hand, "UTG", ActionType.BET, 120)
    _act(hand, "SB", ActionType.FOLD)
    assert hand.is_complete
    return hand


def _utg_open_limp(six_seats) -> HoldemHand:
    hand = HoldemHand(six_seats(), seed=103, hand_id="utg-open-limp")
    _act(hand, "UTG", ActionType.CALL)
    for player_id in ("HJ", "CO", "BTN", "SB"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BB", ActionType.CHECK)
    _act(hand, "BB", ActionType.BET, 40)
    _act(hand, "UTG", ActionType.FOLD)
    assert hand.is_complete
    return hand


def _cutoff_three_bet(six_seats) -> HoldemHand:
    hand = HoldemHand(six_seats(), seed=104, hand_id="co-three-bet")
    _act(hand, "UTG", ActionType.FOLD)
    _act(hand, "HJ", ActionType.RAISE, 120)
    _act(hand, "CO", ActionType.RAISE, 360)
    for player_id in ("BTN", "SB", "BB"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "HJ", ActionType.FOLD)
    assert hand.is_complete
    return hand


def _button_raise_fold_to_three_bet(six_seats) -> HoldemHand:
    hand = HoldemHand(six_seats(), seed=105, hand_id="btn-fold-to-three-bet")
    for player_id in ("UTG", "HJ", "CO"):
        _act(hand, player_id, ActionType.FOLD)
    _act(hand, "BTN", ActionType.RAISE, 120)
    _act(hand, "SB", ActionType.RAISE, 360)
    _act(hand, "BB", ActionType.FOLD)
    _act(hand, "BTN", ActionType.FOLD)
    assert hand.is_complete
    return hand


def test_gold_projection_for_button_cbet_showdown_and_review_signals(six_seats) -> None:
    stats = project_hand_statistics(
        _button_cbet_river_stackoff(six_seats),
        "BTN",
        reason_codes={
            10: {"reason_codes": [TOP_PAIR_STACKOFF_OPPORTUNITY, TOP_PAIR_STACKED_OFF]},
        },
    )

    expected = {
        MetricName.VPIP: (1, 1),
        MetricName.PFR: (1, 1),
        MetricName.OPEN_LIMP: (0, 1),
        MetricName.COLD_CALL: (0, 0),
        MetricName.THREE_BET: (0, 0),
        MetricName.FOLD_TO_THREE_BET: (0, 0),
        MetricName.CBET: (1, 1),
        MetricName.WTSD: (1, 1),
        MetricName.WSD: (1, 1),
        MetricName.AGGRESSION_FACTOR: (1, 1),
        MetricName.TOP_PAIR_STACKOFF: (1, 1),
        MetricName.RIVER_CALL: (1, 1),
        MetricName.DRAW_ODDS_ERROR: (0, 0),
    }
    assert {
        name: (stats.metric(name).hits, stats.metric(name).opportunities)
        for name in MetricName
    } == expected
    assert stats.metric(MetricName.AGGRESSION_FACTOR).value == 1.0
    assert stats.metric(MetricName.WSD).percentage == 100.0
    assert stats.draw_odds_error_count == 0


def test_gold_projection_distinguishes_cold_call_three_bet_and_open_limp(six_seats) -> None:
    sb = project_hand_statistics(
        _small_blind_cold_call_fold_flop(six_seats),
        "SB",
        reason_codes=[DRAW_ODDS_OPPORTUNITY, DRAW_ODDS_OPPORTUNITY, DRAW_ODDS_ERROR],
    )
    utg = project_hand_statistics(_utg_open_limp(six_seats), "UTG")
    co = project_hand_statistics(_cutoff_three_bet(six_seats), "CO")

    assert (sb.metric(MetricName.COLD_CALL).hits, sb.metric(MetricName.COLD_CALL).opportunities) == (1, 1)
    assert (sb.metric(MetricName.THREE_BET).hits, sb.metric(MetricName.THREE_BET).opportunities) == (0, 1)
    assert (sb.metric(MetricName.OPEN_LIMP).hits, sb.metric(MetricName.OPEN_LIMP).opportunities) == (0, 0)
    assert (sb.metric(MetricName.WTSD).hits, sb.metric(MetricName.WTSD).opportunities) == (0, 1)
    assert (sb.metric(MetricName.DRAW_ODDS_ERROR).hits, sb.metric(MetricName.DRAW_ODDS_ERROR).opportunities) == (1, 2)
    assert sb.metric(MetricName.DRAW_ODDS_ERROR).percentage == 50.0
    assert sb.draw_odds_error_count == 1

    assert (utg.metric(MetricName.OPEN_LIMP).hits, utg.metric(MetricName.OPEN_LIMP).opportunities) == (1, 1)
    assert (utg.metric(MetricName.PFR).hits, utg.metric(MetricName.PFR).opportunities) == (0, 1)

    assert (co.metric(MetricName.COLD_CALL).hits, co.metric(MetricName.COLD_CALL).opportunities) == (0, 1)
    assert (co.metric(MetricName.THREE_BET).hits, co.metric(MetricName.THREE_BET).opportunities) == (1, 1)
    assert co.metric(MetricName.THREE_BET).percentage == 100.0


def test_fold_to_three_bet_has_its_own_opportunity(six_seats) -> None:
    stats = project_hand_statistics(_button_raise_fold_to_three_bet(six_seats), "BTN")

    assert (stats.metric(MetricName.FOLD_TO_THREE_BET).hits, stats.metric(MetricName.FOLD_TO_THREE_BET).opportunities) == (1, 1)
    assert stats.metric(MetricName.CBET).opportunities == 0
    assert stats.metric(MetricName.WTSD).opportunities == 0


def test_position_aggregation_sums_counts_before_calculating_percentages(six_seats) -> None:
    rows = [
        project_hand_statistics(
            _button_cbet_river_stackoff(six_seats),
            "BTN",
            reason_codes=[TOP_PAIR_STACKOFF_OPPORTUNITY, TOP_PAIR_STACKED_OFF],
        ),
        project_hand_statistics(_button_raise_fold_to_three_bet(six_seats), "BTN"),
        project_hand_statistics(
            _small_blind_cold_call_fold_flop(six_seats),
            "SB",
            reason_codes=[DRAW_ODDS_OPPORTUNITY, DRAW_ODDS_OPPORTUNITY, DRAW_ODDS_ERROR],
        ),
        project_hand_statistics(_utg_open_limp(six_seats), "UTG"),
        project_hand_statistics(_cutoff_three_bet(six_seats), "CO"),
    ]
    by_position = aggregate_by_position(rows)

    button = by_position[Position.BTN]
    assert button.hands == 2
    assert (button.metric(MetricName.VPIP).hits, button.metric(MetricName.VPIP).opportunities) == (2, 2)
    assert button.metric(MetricName.VPIP).percentage == 100.0
    assert (button.metric(MetricName.CBET).hits, button.metric(MetricName.CBET).opportunities) == (1, 1)
    assert (button.metric(MetricName.FOLD_TO_THREE_BET).hits, button.metric(MetricName.FOLD_TO_THREE_BET).opportunities) == (1, 1)

    small_blind = by_position[Position.SB]
    assert (small_blind.metric(MetricName.THREE_BET).hits, small_blind.metric(MetricName.THREE_BET).opportunities) == (0, 1)
    assert small_blind.metric(MetricName.THREE_BET).percentage == 0.0
    assert small_blind.draw_odds_error_count == 1

    assert by_position[Position.HJ].hands == 0
    assert by_position[Position.HJ].metric(MetricName.VPIP).value is None
    assert by_position[Position.HJ].metric(MetricName.VPIP).percentage is None

    payload = serialize_position_statistics(by_position)
    assert payload["BTN"]["metrics"]["vpip"]["opportunities"] == 2
    assert payload["HJ"]["metrics"]["vpip"]["percentage"] is None
    json.dumps(payload, ensure_ascii=False)


def test_review_hit_implies_opportunity_but_inconsistent_explicit_counts_fail(six_seats) -> None:
    hand = _button_cbet_river_stackoff(six_seats)
    inferred = project_hand_statistics(hand, "BTN", reason_codes=[TOP_PAIR_STACKED_OFF])
    assert (inferred.metric(MetricName.TOP_PAIR_STACKOFF).hits, inferred.metric(MetricName.TOP_PAIR_STACKOFF).opportunities) == (1, 1)

    with pytest.raises(ValueError, match="命中数不能大于机会数"):
        project_hand_statistics(
            hand,
            "BTN",
            reason_codes=[
                TOP_PAIR_STACKOFF_OPPORTUNITY,
                TOP_PAIR_STACKED_OFF,
                TOP_PAIR_STACKED_OFF,
            ],
        )


def test_incomplete_hand_is_not_projected(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=999)
    with pytest.raises(ValueError, match="已经结束"):
        project_hand_statistics(hand, "BTN")

