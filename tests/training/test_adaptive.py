from __future__ import annotations

import math

import pytest

from poker_trainer.analytics.statistics import (
    METRIC_ORDER,
    MetricName,
    MetricTally,
    PositionStatistics,
)
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import Position
from poker_trainer.training.adaptive import (
    SCENARIO_SPECS,
    AdaptiveScheduler,
    LeakFinding,
    PlayerProfile,
    ScenarioSpec,
    build_scenario_plan,
    generate_player_profile,
)


def _position_stat(
    position: Position,
    *,
    hands: int = 0,
    metrics: dict[MetricName, tuple[int, int]] | None = None,
) -> PositionStatistics:
    supplied = metrics or {}
    return PositionStatistics(
        position=position,
        hands=hands,
        metrics={
            name: MetricTally(*supplied.get(name, (0, 0))) for name in METRIC_ORDER
        },
    )


def _empty_rows(*, hands: int = 20) -> dict[Position, PositionStatistics]:
    return {
        position: _position_stat(
            position,
            hands=hands if position == Position.UTG else 0,
        )
        for position in Position
    }


def _finding(code: str, severity: float, confidence: float = 0.8) -> LeakFinding:
    return LeakFinding(
        code=code,
        title_zh=code,
        severity=severity,
        confidence=confidence,
        evidence_n=10,
        actual=0.8,
        benchmark=0.2,
        position=None,
        reason="测试证据",
    )


def test_profile_does_not_trigger_at_19_but_triggers_at_20() -> None:
    rows = _empty_rows()

    assert generate_player_profile(19, rows) is None
    profile = generate_player_profile(20, rows)

    assert profile is not None
    assert profile.through_hand_no == 20
    assert profile.sample_size == 20


def test_20_hand_checkpoint_with_too_few_relevant_opportunities_says_insufficient() -> None:
    profile = generate_player_profile(20, _empty_rows())

    assert profile is not None
    assert profile.leaks == ()
    assert profile.message_zh == "信号不足"
    assert profile.has_signal is False


@pytest.mark.parametrize(
    "reason_code",
    [
        "weak_top_pair_overcall",
        "sb_cold_call",
        "threebet_too_small",
        "small_pair_overcontinue",
        "strong_draw_overfold",
    ],
)
def test_all_required_reason_code_leaks_are_recognized(reason_code: str) -> None:
    profile = generate_player_profile(
        20,
        _empty_rows(),
        reason_code_counts={reason_code: 3},
    )

    assert profile is not None
    assert [finding.code for finding in profile.leaks] == [reason_code]
    assert profile.leaks[0].evidence_n == 3
    assert profile.leaks[0].actual == 1.0


def test_position_metrics_and_reason_counts_are_ranked_and_trimmed_to_top_three() -> None:
    rows = _empty_rows()
    rows[Position.UTG] = _position_stat(
        Position.UTG,
        hands=12,
        metrics={MetricName.TOP_PAIR_STACKOFF: (7, 8)},
    )
    rows[Position.SB] = _position_stat(
        Position.SB,
        hands=8,
        metrics={MetricName.COLD_CALL: (8, 8)},
    )
    profile = generate_player_profile(
        20,
        rows,
        reason_code_counts={
            "threebet_too_small": {"hits": 9, "opportunities": 10},
            "small_pair_overcontinue": {"hits": 6, "opportunities": 10},
            "strong_draw_overfold": {"hits": 5, "opportunities": 10},
        },
    )

    assert profile is not None
    assert len(profile.leaks) == 3
    assert [item.severity for item in profile.leaks] == sorted(
        (item.severity for item in profile.leaks), reverse=True
    )
    assert {item.code for item in profile.leaks} == {
        "sb_cold_call",
        "threebet_too_small",
        "weak_top_pair_overcall",
    }
    sb = next(item for item in profile.leaks if item.code == "sb_cold_call")
    assert sb.position == Position.SB
    assert sb.actual == 1.0
    assert sb.benchmark == 0.20
    assert 0 < sb.confidence <= 1


def test_enough_opportunities_without_threshold_breach_is_not_called_a_leak() -> None:
    rows = _empty_rows()
    rows[Position.SB] = _position_stat(
        Position.SB,
        hands=20,
        metrics={MetricName.COLD_CALL: (1, 10)},
    )

    profile = generate_player_profile(20, rows)

    assert profile is not None
    assert profile.leaks == ()
    assert profile.message_zh == "暂未发现明显漏洞"


def test_scenario_plan_keeps_random_floor_and_caps_total_directed_weight() -> None:
    leaks = (
        _finding("sb_cold_call", 0.9),
        _finding("threebet_too_small", 0.7),
        _finding("strong_draw_overfold", 0.5),
    )
    profile = PlayerProfile(20, 20, leaks, "发现 3 个优先漏洞")

    plan = build_scenario_plan(profile)

    assert plan.effective_from_hand_no == 21
    assert plan.hand_count == 20
    assert plan.weights["random"] >= 0.55
    assert math.isclose(sum(plan.weights.values()), 1.0)
    assert sum(
        weight for scenario_id, weight in plan.weights.items() if scenario_id != "random"
    ) <= 0.45 + 1e-12
    assert all(
        weight <= 0.45
        for scenario_id, weight in plan.weights.items()
        if scenario_id != "random"
    )

    random_only = build_scenario_plan(None, effective_from_hand_no=1)
    assert random_only.weights == {"random": 1.0}


def test_scenario_choice_is_stable_for_same_seed_and_hand_number() -> None:
    scheduler_a = AdaptiveScheduler()
    scheduler_b = AdaptiveScheduler()
    plan = scheduler_a.build_plan(
        [
            _finding("sb_cold_call", 0.9),
            _finding("small_pair_overcontinue", 0.6),
        ],
        21,
    )

    first = scheduler_a.choose_scenario(plan, "session-seed", 23)
    repeated = scheduler_a.choose_scenario(plan, "session-seed", 23)
    separate_instance = scheduler_b.choose_scenario(plan, "session-seed", 23)

    assert first == repeated == separate_instance
    assert first.scenario_id in plan.weights


def test_predefined_cards_are_unique_and_can_be_passed_to_holdem_hand(six_seats) -> None:
    for index, scenario in enumerate(SCENARIO_SPECS.values(), start=1):
        preset_cards = (*scenario.hole_cards, *scenario.board_cards)
        assert len(preset_cards) == len(set(preset_cards))

        kwargs = scenario.to_hand_kwargs()
        hand = HoldemHand(six_seats(), seed=1000 + index, **kwargs)
        assert hand.scenario_id == scenario.scenario_id
        assert hand.hole_overrides == kwargs.get("hole_overrides", {})
        assert hand.board_override == tuple(scenario.board_cards)


def test_scenario_rejects_duplicate_or_invalid_presets() -> None:
    with pytest.raises(ValueError, match="不能重复"):
        ScenarioSpec(
            "duplicate",
            preferred_position=Position.BTN,
            hole_cards="As Kd",
            board_cards="As 7c 2h",
        )

    with pytest.raises(ValueError, match="恰好 2 张"):
        ScenarioSpec("one-card", hole_cards="As")


