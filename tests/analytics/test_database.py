from __future__ import annotations

import json
import sqlite3

import pytest

from poker_trainer.analytics.database import CURRENT_SCHEMA_VERSION, SQLiteStore
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position
from poker_trainer.engine.replay import ReplayBundle


EXPECTED_TABLES = {
    "schema_version",
    "sessions",
    "hands",
    "hand_players",
    "actions",
    "decision_reviews",
    "metric_snapshots",
    "leak_snapshots",
    "scenario_plans",
}


def _fold_to_completion(hand: HoldemHand) -> None:
    for player_id in ("UTG", "HJ", "CO", "BTN", "SB"):
        assert hand.current_actor_id == player_id
        hand.act(player_id, ActionType.FOLD)
    assert hand.is_complete


def _saved_hand(store: SQLiteStore, six_seats) -> tuple[HoldemHand, str]:
    session_id = store.create_session(
        "session-1",
        mode="测试模式",
        hero_player_id="BTN",
        seed=101,
    )
    hand = HoldemHand(
        six_seats(),
        seed=20260827,
        hand_id="hand-1",
        session_id=session_id,
        hand_no=1,
    )
    _fold_to_completion(hand)
    replay_json = ReplayBundle.from_hand(hand).to_json()
    store.save_hand(hand, replay_json, "test", policy_seed=9988)
    return hand, replay_json


def test_schema_migration_is_complete_and_idempotent(tmp_path) -> None:
    database_path = tmp_path / "trainer.sqlite3"

    with SQLiteStore(database_path) as store:
        assert store.schema_version == CURRENT_SCHEMA_VERSION
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert EXPECTED_TABLES <= tables
        versions = store.connection.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [CURRENT_SCHEMA_VERSION]

    # Reopening applies no duplicate migration and preserves the marker.
    with SQLiteStore(database_path) as reopened:
        assert reopened.schema_version == CURRENT_SCHEMA_VERSION
        assert reopened.connection.execute(
            "SELECT COUNT(*) FROM schema_version"
        ).fetchone()[0] == 1


def test_complete_hand_round_trip_actions_positions_and_hidden_cards(
    tmp_path, six_seats
) -> None:
    with SQLiteStore(tmp_path / "round-trip.sqlite3") as store:
        original, replay_json = _saved_hand(store, six_seats)

        assert store.load_replay_json(original.hand_id) == replay_json
        replayed = ReplayBundle.from_json(store.load_replay_json(original.hand_id)).replay()
        assert replayed.public_view(reveal_all=True) == original.public_view(reveal_all=True)

        summaries = store.list_hands("session-1")
        assert len(summaries) == 1
        assert summaries[0]["hand_id"] == "hand-1"
        assert summaries[0]["policy_seed"] == 9988
        assert "replay_json" not in summaries[0]

        persisted = store.load_hand("hand-1")
        assert persisted is not None
        # 2 blind posts + 5 folds + the unmatched half-BB refund are all
        # audit events and must survive persistence.
        assert len(persisted["actions"]) == len(original.history) == 8
        assert persisted["actions"][-1]["action_type"] == "refund"
        assert [player["position"] for player in persisted["players"]] == [
            "UTG",
            "HJ",
            "CO",
            "BTN",
            "SB",
            "BB",
        ]
        assert all(player["hole_cards_hidden"] for player in persisted["players"])
        assert all(player["hole_cards"] is None for player in persisted["players"])

        trusted = store.load_hand("hand-1", reveal_hole_cards=True)
        assert trusted is not None
        assert all(len(player["hole_cards"]) == 2 for player in trusted["players"])
        raw_cards = store.connection.execute(
            "SELECT hole_cards_json FROM hand_players WHERE hand_id = ?",
            ("hand-1",),
        ).fetchall()
        assert all(len(json.loads(row[0])) == 2 for row in raw_cards)


def test_reviews_and_adaptive_snapshots_round_trip(tmp_path, six_seats) -> None:
    with SQLiteStore(tmp_path / "analytics.sqlite3") as store:
        hand, _ = _saved_hand(store, six_seats)

        review_id = store.save_decision_review(
            hand.hand_id,
            2,
            {
                "rating": "可以接受",
                "reason": "前位弱牌弃牌合理",
                "recommended_action": "fold",
                "pot_odds": 0.02,
                "equity": 0.18,
            },
        )
        reviews = store.load_decision_reviews(hand.hand_id)
        assert reviews[0]["review_id"] == review_id
        assert reviews[0]["player_id"] == "UTG"
        assert reviews[0]["review"]["rating"] == "可以接受"

        metric_id = store.save_metric_snapshot(
            "session-1",
            20,
            {"vpip": 0.31, "pfr": 0.18},
            position=Position.SB,
            sample_size=4,
        )
        leak_id = store.save_leak_snapshot(
            "session-1",
            20,
            [
                {"code": "sb_cold_call", "severity": 0.9},
                {"code": "river_overcall", "severity": 0.7},
            ],
        )
        plan_id = store.save_scenario_plan(
            "session-1",
            21,
            {"weights": {"sb_3bet_or_fold": 2.0}},
            source_leak_snapshot_id=leak_id,
        )

        metrics = store.load_metric_snapshots("session-1", position=Position.SB)
        assert metrics == [
            {
                "snapshot_id": metric_id,
                "session_id": "session-1",
                "through_hand_no": 20,
                "sample_size": 4,
                "position": "SB",
                "created_at": metrics[0]["created_at"],
                "metrics": {"pfr": 0.18, "vpip": 0.31},
            }
        ]
        leaks = store.load_leak_snapshots("session-1")
        assert leaks[0]["snapshot_id"] == leak_id
        assert leaks[0]["leaks"][0]["code"] == "sb_cold_call"
        plans = store.load_scenario_plans("session-1")
        assert plans[0]["plan_id"] == plan_id
        assert plans[0]["source_leak_snapshot_id"] == leak_id
        assert plans[0]["plan"]["weights"]["sb_3bet_or_fold"] == 2.0


def test_foreign_keys_are_enforced_and_failed_write_rolls_back(tmp_path) -> None:
    with SQLiteStore(tmp_path / "foreign-key.sqlite3") as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.save_metric_snapshot("missing-session", 20, {"vpip": 0.2})
        assert store.connection.execute(
            "SELECT COUNT(*) FROM metric_snapshots"
        ).fetchone()[0] == 0


def test_invalid_replay_does_not_write_partial_hand(tmp_path, six_seats) -> None:
    with SQLiteStore(tmp_path / "atomic.sqlite3") as store:
        session_id = store.create_session("session-atomic")
        hand = HoldemHand(
            six_seats(), seed=44, hand_id="hand-atomic", session_id=session_id
        )
        with pytest.raises(ValueError, match="hand_id 不一致"):
            store.save_hand(hand, '{"hand_id":"wrong"}', "test")
        assert store.connection.execute("SELECT COUNT(*) FROM hands").fetchone()[0] == 0

