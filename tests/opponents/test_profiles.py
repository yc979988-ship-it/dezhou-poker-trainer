from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import random
from statistics import mean

import pytest

from poker_trainer.opponents.profiles import (
    OpponentProfile,
    drift_for_session,
    generate_base_profile,
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

        assert 0.21 <= profile.vpip <= 0.53
        assert 0.45 <= profile.pfr / profile.vpip <= 0.88
        assert 0.025 <= profile.three_bet <= 0.14
        assert 0.80 <= profile.aggression_factor <= 4.00
        assert 0.26 <= profile.fold_tendency <= 0.70
        assert 0.05 <= profile.limp_tendency <= 0.31
        assert 0.01 <= profile.mistake_rate <= 0.10
        assert 0.0 <= profile.pfr < profile.vpip <= 1.0


def test_population_is_calibrated_as_slightly_loose_friend_game() -> None:
    profiles = [
        generate_base_profile(f"villain-{seed}", seed)
        for seed in range(400)
    ]

    assert 0.35 <= mean(profile.vpip for profile in profiles) <= 0.39
    assert 0.16 <= mean(profile.limp_tendency for profile in profiles) <= 0.20
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

