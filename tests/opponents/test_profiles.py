from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import random
from statistics import mean

import pytest

from poker_trainer.opponents.profiles import (
    OpponentHabits,
    OpponentProfile,
    drift_for_session,
    generate_base_profile,
    profile_from_habits,
)


PROBABILITY_FIELDS = (
    "vpip",
    "pfr",
    "three_bet",
    "fold_tendency",
    "limp_tendency",
    "mistake_rate",
)


def test_base_profiles_have_realistic_ranges_and_pfr_below_vpip() -> None:
    for master_seed in range(100):
        profile = generate_base_profile(f"villain-{master_seed % 5}", master_seed)

        assert 0.34 <= profile.vpip <= 0.70
        assert 0.14 <= profile.pfr / profile.vpip <= 0.42
        assert 0.02 <= profile.three_bet <= 0.12
        assert 0.80 <= profile.aggression_factor <= 4.00
        assert 0.26 <= profile.fold_tendency <= 0.70
        assert 0.28 <= profile.limp_tendency <= 0.72
        assert 0.01 <= profile.mistake_rate <= 0.07
        assert 0.0 <= profile.pfr < profile.vpip <= 1.0


def test_population_is_calibrated_as_loose_passive_friend_game() -> None:
    profiles = [
        generate_base_profile(f"villain-{seed}", seed)
        for seed in range(400)
    ]

    assert 0.50 <= mean(profile.vpip for profile in profiles) <= 0.54
    assert 0.13 <= mean(profile.pfr for profile in profiles) <= 0.16
    assert 0.47 <= mean(profile.limp_tendency for profile in profiles) <= 0.51
    assert 0.46 <= mean(profile.fold_tendency for profile in profiles) <= 0.50


def test_base_derivation_is_deterministic_and_seed_types_are_separated() -> None:
    first = generate_base_profile("opponent-1", 20260827)
    second = generate_base_profile("opponent-1", 20260827)

    assert first == second
    assert generate_base_profile("opponent-2", 20260827) != first
    assert generate_base_profile("opponent-1", 20260828) != first
    assert generate_base_profile("opponent-1", "20260827") != first
    assert generate_base_profile("opponent-1", b"20260827") != first


def test_profile_generation_does_not_touch_global_random_state() -> None:
    random.seed(918273)
    before = random.getstate()

    base = generate_base_profile("opponent-1", 42)
    drift_for_session(base, "session-a")

    assert random.getstate() == before


def test_session_drift_is_deterministic_small_and_keeps_all_invariants() -> None:
    base = generate_base_profile("opponent-1", 42)
    same_a = drift_for_session(base, "session-a")
    same_b = drift_for_session(base, "session-a")
    other = drift_for_session(base, "session-b")

    assert same_a == same_b
    assert other != same_a
    assert same_a is not base
    assert base == generate_base_profile("opponent-1", 42)
    assert same_a.opponent_id == base.opponent_id
    assert 0.0 <= same_a.pfr < same_a.vpip <= 1.0
    for field_name in PROBABILITY_FIELDS:
        assert 0.0 <= getattr(same_a, field_name) <= 1.0
        assert abs(getattr(same_a, field_name) - getattr(base, field_name)) < 0.04
    assert same_a.aggression_factor > 0.0
    assert abs(same_a.aggression_factor - base.aggression_factor) < 0.20


def test_drift_bound_holds_across_many_profiles_and_sessions() -> None:
    for master_seed in range(30):
        base = generate_base_profile(f"villain-{master_seed}", master_seed)
        for session_seed in range(20):
            session = drift_for_session(base, session_seed)
            assert session.pfr < session.vpip
            for field_name in PROBABILITY_FIELDS:
                assert abs(
                    getattr(session, field_name) - getattr(base, field_name)
                ) < 0.04
            assert (
                abs(session.aggression_factor - base.aggression_factor) < 0.20
            )


def test_profile_has_only_hidden_continuous_parameters_without_style_label() -> None:
    assert {field.name for field in fields(OpponentProfile)} == {
        "opponent_id",
        "vpip",
        "pfr",
        "three_bet",
        "aggression_factor",
        "fold_tendency",
        "limp_tendency",
        "mistake_rate",
    }

    profile = generate_base_profile("opponent-1", 1)
    with pytest.raises(FrozenInstanceError):
        profile.vpip = 0.99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vpip": -0.1}, "vpip"),
        ({"three_bet": 1.1}, "three_bet"),
        ({"aggression_factor": 0.0}, "aggression_factor"),
        ({"pfr": 0.31}, "pfr 必须严格小于 vpip"),
        ({"mistake_rate": float("nan")}, "有限数值"),
    ],
)
def test_profile_rejects_invalid_values(
    overrides: dict[str, float], message: str
) -> None:
    values = {
        "opponent_id": "opponent-1",
        "vpip": 0.30,
        "pfr": 0.20,
        "three_bet": 0.08,
        "aggression_factor": 2.0,
        "fold_tendency": 0.50,
        "limp_tendency": 0.10,
        "mistake_rate": 0.03,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        OpponentProfile(**values)


@pytest.mark.parametrize("bad_seed", [None, 1.5, True, object()])
def test_seed_type_must_be_stably_encodable(bad_seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        generate_base_profile("opponent-1", bad_seed)  # type: ignore[arg-type]


def test_observed_habits_map_to_hidden_parameters_monotonically() -> None:
    quiet = OpponentHabits(
        "friend-1",
        "小王",
        entry_frequency=1,
        limp_frequency=1,
        preflop_aggression=1,
        calling_tendency=1,
        postflop_aggression=1,
        mistake_frequency=1,
    )
    active = OpponentHabits(
        "friend-2",
        "老李",
        entry_frequency=5,
        limp_frequency=5,
        preflop_aggression=5,
        calling_tendency=5,
        postflop_aggression=5,
        mistake_frequency=5,
    )

    quiet_profile = profile_from_habits(quiet)
    active_profile = profile_from_habits(active)

    assert active_profile.vpip > quiet_profile.vpip
    assert active_profile.pfr > quiet_profile.pfr
    assert active_profile.three_bet > quiet_profile.three_bet
    assert active_profile.limp_tendency > quiet_profile.limp_tendency
    assert active_profile.aggression_factor > quiet_profile.aggression_factor
    assert active_profile.mistake_rate > quiet_profile.mistake_rate
    assert active_profile.fold_tendency < quiet_profile.fold_tendency
    assert active_profile.pfr < active_profile.vpip


def test_observed_habits_validate_levels_and_round_trip_mapping() -> None:
    habits = OpponentHabits.from_mapping(
        {
            "opponent_id": " friend-7 ",
            "nickname": " 阿七 ",
            "entry_frequency": 4,
            "limp_frequency": 3,
            "preflop_aggression": 3,
            "calling_tendency": 5,
            "postflop_aggression": 3,
            "mistake_frequency": 2,
        }
    )
    assert habits.opponent_id == "friend-7"
    assert habits.nickname == "阿七"
    assert OpponentHabits.from_mapping(habits.as_dict()) == habits

    with pytest.raises(ValueError, match="1 到 5"):
        OpponentHabits("bad", "牌友", entry_frequency=0)
    with pytest.raises(TypeError, match="整数"):
        OpponentHabits("bad", "牌友", limp_frequency=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nickname"):
        OpponentHabits("bad", " ")
