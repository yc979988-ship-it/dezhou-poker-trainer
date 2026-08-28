from __future__ import annotations

from poker_trainer.analytics.database import SQLiteStore
from poker_trainer.coaching.coach import DecisionReview
from poker_trainer.engine.models import PREFLOP_ORDER, ActionType, Position
from poker_trainer.engine.replay import ReplayBundle
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
        assert {player.position for player in hand.players.values()} == set(Position)
        _fold_hero_and_finish(session)

    assert hero_positions == list(PREFLOP_ORDER)
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

