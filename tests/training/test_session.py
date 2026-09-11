from __future__ import annotations

import pytest

from poker_trainer.analytics.database import SQLiteStore
from poker_trainer.coaching.coach import DecisionReview
from poker_trainer.engine.models import PREFLOP_ORDER, ActionType, Position, positions_for_table_size
from poker_trainer.engine.replay import ReplayBundle
from poker_trainer.opponents.profiles import OpponentHabits, profile_from_habits
from poker_trainer.training.adaptive import (
    AdaptiveScheduler,
    ScenarioPlan,
    ScenarioSpec,
)
from poker_trainer.training.session import (
    SessionConfig,
    TrainingMode,
    TrainingSession,
)


def _fold_hero_and_finish(session: TrainingSession) -> None:
    hand = session.current_hand
    assert hand is not None
    assert hand.current_actor_id == session.hero_id
    session.hero_action(ActionType.FOLD)
    assert hand.is_complete


def test_session_config_uses_mvp_stakes_and_display_conversion() -> None:
    config = SessionConfig()

    assert config.mode is TrainingMode.TEST
    assert (config.small_blind, config.big_blind, config.buy_in) == (20, 40, 4_000)
    assert config.buy_in_big_blinds == 100
    assert config.chips_to_yuan(100) == 1
    assert config.buy_in_yuan == 40
    assert config.auto_top_up is True


def test_custom_friends_change_bots_without_persisting_identity() -> None:
    friends = (
        OpponentHabits(
            "friend-xiao-wang",
            "小王",
            entry_frequency=5,
            limp_frequency=5,
            calling_tendency=5,
        ),
        OpponentHabits(
            "friend-lao-li",
            "老李",
            preflop_aggression=5,
            postflop_aggression=5,
        ),
    )
    store = SQLiteStore(":memory:")
    session = TrainingSession(
        SessionConfig(seed=20260907, coach_trials=5, opponent_habits=friends),
        store=store,
    )
    hand = session.start_hand()

    assert len(hand.players) == 6
    assert hand.player("bot-1").name == "对手1"
    assert hand.player("bot-2").name == "对手2"
    assert session.opponent_display_names["bot-1"] == "小王"
    assert session.opponent_display_names["bot-2"] == "老李"
    assert set(session.player_stacks) == {
        "hero",
        "bot-1",
        "bot-2",
        "bot-3",
        "bot-4",
        "bot-5",
    }
    expected = profile_from_habits(friends[0])
    actual = session._bot_profiles["bot-1"]
    assert actual.opponent_id == "bot-1"
    assert abs(actual.vpip - expected.vpip) < 0.04

    public_text = repr(session.public_state())
    forbidden = (
        "小王",
        "老李",
        "friend-xiao-wang",
        "friend-lao-li",
        "entry_frequency",
        "limp_frequency",
        "fold_tendency",
    )
    assert not any(term in public_text for term in forbidden)

    _fold_hero_and_finish(session)
    replay_text = ReplayBundle.from_hand(hand).to_json()
    session.close()
    database_text = "\n".join(store.connection.iterdump())
    assert not any(term in replay_text for term in forbidden)
    assert "vpip" not in replay_text and "pfr" not in replay_text
    assert not any(term in database_text for term in forbidden)
    stored_players = store.connection.execute(
        "SELECT player_id, name FROM hand_players ORDER BY player_id"
    ).fetchall()
    assert {(row[0], row[1]) for row in stored_players} == {
        ("hero", "你"),
        ("bot-1", "对手1"),
        ("bot-2", "对手2"),
        ("bot-3", "对手3"),
        ("bot-4", "对手4"),
        ("bot-5", "对手5"),
    }
    store.close()


def test_session_rejects_more_than_table_capacity_or_duplicate_custom_friends() -> None:
    too_many = tuple(
        OpponentHabits(f"friend-{index}", f"牌友{index}")
        for index in range(8)
    )
    with pytest.raises(ValueError, match="最多"):
        SessionConfig(opponent_habits=too_many)

    repeated = OpponentHabits("same-id", "牌友")
    with pytest.raises(ValueError, match="不能重复"):
        SessionConfig(opponent_habits=(repeated, repeated))

    same_name = (
        OpponentHabits("friend-a", " 阿杰 "),
        OpponentHabits("friend-b", "阿杰"),
    )
    with pytest.raises(ValueError, match="昵称不能重复"):
        SessionConfig(opponent_habits=same_name)


@pytest.mark.parametrize("table_size", [5, 6, 7, 8])
def test_session_supports_variable_table_sizes(table_size: int) -> None:
    session = TrainingSession(SessionConfig(table_size=table_size, seed=100 + table_size, coach_trials=5))
    hand = session.start_hand()
    assert len(hand.players) == table_size
    assert {player.position for player in hand.players.values()} == set(positions_for_table_size(table_size))
    assert Position.SB in {player.position for player in hand.players.values()}
    assert Position.BB in {player.position for player in hand.players.values()}
    session.close()


def test_same_seed_habits_and_hero_action_recreate_the_same_hand() -> None:
    habits = (
        OpponentHabits(
            "friend-a",
            "阿杰",
            entry_frequency=5,
            limp_frequency=4,
            calling_tendency=5,
        ),
    )
    config = SessionConfig(seed=20260907, coach_trials=25, opponent_habits=habits)
    first = TrainingSession(config)
    second = TrainingSession(config)

    first_hand = first.start_hand()
    second_hand = second.start_hand()
    assert first_hand.hand_id != second_hand.hand_id
    first.hero_action(ActionType.FOLD)
    second.hero_action(ActionType.FOLD)

    assert [row.as_dict() for row in first_hand.history] == [
        row.as_dict() for row in second_hand.history
    ]
    assert tuple(map(str, first_hand.board)) == tuple(map(str, second_hand.board))
    assert first_hand.result == second_hand.result
    assert [row.as_dict() for row in first.last_hand_reviews] == [
        row.as_dict() for row in second.last_hand_reviews
    ]
    first.close()
    second.close()


def test_teaching_returns_immediate_review_but_test_hides_it_until_complete() -> None:
    teaching = TrainingSession(
        SessionConfig(mode=TrainingMode.TEACHING, seed=11, coach_trials=5)
    )
    teaching.start_hand()
    review = teaching.hero_action(ActionType.FOLD)
    assert isinstance(review, DecisionReview)
    assert teaching.current_reviews == (review,)
    teaching.close()

    testing = TrainingSession(
        SessionConfig(mode=TrainingMode.TEST, seed=3, coach_trials=5)
    )
    hand = testing.start_hand()
    assert testing.hero_action(ActionType.CALL) is None
    assert not hand.is_complete
    assert hand.current_actor_id == testing.hero_id
    assert testing.current_reviews == ()

    completed_reviews = testing.hero_action(ActionType.FOLD)
    assert hand.is_complete
    assert isinstance(completed_reviews, tuple)
    assert len(completed_reviews) == 2
    assert testing.current_reviews == completed_reviews
    testing.close()


class _ForcedOpenScheduler(AdaptiveScheduler):
    def __init__(self) -> None:
        scenario = ScenarioSpec(
            "forced-test",
            preferred_position=Position.BTN,
            forced_open=True,
        )
        super().__init__(
            scenarios={
                "random": ScenarioSpec("random"),
                scenario.scenario_id: scenario,
            }
        )
        self.scenario = scenario
        self.plan = ScenarioPlan(
            weights={scenario.scenario_id: 1.0},
            scenarios={scenario.scenario_id: scenario},
        )

    def build_plan(self, profile_or_leaks, effective_from_hand_no=None, *, hand_count=20):
        return self.plan

    def choose_scenario(self, plan, seed, hand_no):
        return self.scenario


def test_bots_only_apply_legal_actions_and_forced_open_occurs_once() -> None:
    session = TrainingSession(
        SessionConfig(seed=17, coach_trials=5),
        scheduler=_ForcedOpenScheduler(),
    )
    hand = session.start_hand()
    voluntary = [record for record in hand.history if not record.forced]

    assert hand.player(session.hero_id).position is Position.BTN
    assert voluntary
    opening_raises = [
        record
        for record in voluntary
        if record.street.value == "preflop"
        and record.current_bet_before == session.config.big_blind
        and record.current_bet_after > record.current_bet_before
    ]
    assert len(opening_raises) == 1
    assert opening_raises[0].action is ActionType.RAISE
    assert opening_raises[0].bet_to >= opening_raises[0].min_raise_to_before

    # 若任何 bot 动作不合法，按动作重放会在同一序号抛 InvalidAction。
    replayed = ReplayBundle.from_hand(hand).replay()
    assert [row.as_dict() for row in replayed.history] == [
        row.as_dict() for row in hand.history
    ]
    session.close()


def test_auto_top_up_restores_every_short_stack_before_next_hand() -> None:
    session = TrainingSession(SessionConfig(seed=23, coach_trials=5))
    session.start_hand()
    _fold_hero_and_finish(session)
    ending_stacks = dict(session.player_stacks)
    short_players = {
        player_id: stack
        for player_id, stack in ending_stacks.items()
        if stack < session.config.buy_in
    }
    assert short_players

    next_hand = session.start_hand()
    for player_id, prior_stack in short_players.items():
        assert next_hand.player(player_id).starting_stack == session.config.buy_in
        assert session.last_top_ups[player_id] == session.config.buy_in - prior_stack
    _fold_hero_and_finish(session)
    session.close()


def test_six_physical_players_rotate_through_all_six_positions() -> None:
    session = TrainingSession(SessionConfig(seed=29, coach_trials=5))
    hero_positions: list[Position] = []
    physical_ids: set[str] | None = None

    for _ in range(6):
        hand = session.start_hand()
        hero_positions.append(hand.player(session.hero_id).position)
        ids = set(hand.players)
        physical_ids = ids if physical_ids is None else physical_ids
        assert ids == physical_ids
        assert {player.position for player in hand.players.values()} == set(positions_for_table_size(6))
        _fold_hero_and_finish(session)

    assert hero_positions == list(positions_for_table_size(6))
    session.close()


def test_completed_hand_is_saved_as_a_replayable_sqlite_bundle(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "training.sqlite3")
    session = TrainingSession(
        SessionConfig(mode="teaching", seed=31, coach_trials=5),
        store=store,
    )
    original = session.start_hand()
    _fold_hero_and_finish(session)

    rows = session.stored_hands
    assert len(rows) == 1
    assert rows[0]["completed"] is True
    bundle = ReplayBundle.from_json(store.load_replay_json(original.hand_id))
    replayed = bundle.replay()
    assert replayed.is_complete
    assert replayed.result == original.result
    assert [row.as_dict() for row in replayed.history] == [
        row.as_dict() for row in original.history
    ]
    assert len(store.load_decision_reviews(original.hand_id)) == 1

    session.close()
    store.close()


def test_twentieth_hand_persists_profile_metrics_leaks_and_next_plan(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "adaptive.sqlite3")
    scheduler = AdaptiveScheduler()
    config = SessionConfig(seed=37, coach_trials=5)
    session = TrainingSession(config, store=store, scheduler=scheduler)

    for _ in range(20):
        session.start_hand()
        _fold_hero_and_finish(session)

    assert session.completed_hand_count == 20
    assert len(session.hand_statistics) == 20
    assert sum(row.hands for row in session.position_statistics.values()) == 20
    assert session.latest_profile is not None
    assert session.latest_profile.through_hand_no == 20
    assert session.latest_profile.sample_size == 20
    assert session.current_plan.effective_from_hand_no == 21
    assert len(store.load_metric_snapshots(session.session_id)) == 6
    assert len(store.load_leak_snapshots(session.session_id)) == 1
    assert len(store.load_scenario_plans(session.session_id)) == 1

    expected = scheduler.choose_scenario(session.current_plan, config.seed, 21)
    hand_21 = session.start_hand()
    assert hand_21.hand_no == 21
    assert session.current_scenario is not None
    assert session.current_scenario.scenario_id == expected.scenario_id
    _fold_hero_and_finish(session)

    # UI 快照不暴露任何对手画像对象或画像字段集合。
    state = session.public_state()
    assert "opponent_profiles" not in state
    assert "bot_profiles" not in state

    session.close()
    store.close()
