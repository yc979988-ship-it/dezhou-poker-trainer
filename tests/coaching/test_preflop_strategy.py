"""翻前纯策略矩阵的不变量与关键范围边界。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from poker_trainer.coaching.preflop_strategy import (
    ActionRole,
    PreflopSituation,
    StrategicAction,
    build_preflop_plan,
    hand_shape,
)
from poker_trainer.engine.cards import Card
from poker_trainer.engine.models import Position


RANKS = "AKQJT98765432"


def _canonical_hands() -> Iterator[tuple[str, tuple[Card, Card]]]:
    """生成 13 对子 + 78 同花 + 78 非同花的 169 种规范起手牌。"""

    for high_index, high in enumerate(RANKS):
        yield f"{high}{high}", (Card(high, "s"), Card(high, "h"))
        for low in RANKS[high_index + 1 :]:
            yield f"{high}{low}s", (Card(high, "s"), Card(low, "s"))
            yield f"{high}{low}o", (Card(high, "s"), Card(low, "h"))


CANONICAL_HANDS = tuple(_canonical_hands())


def _cards(first: str, second: str) -> tuple[Card, Card]:
    return Card.from_str(first), Card.from_str(second)


def _situation(
    cards: tuple[Card, Card],
    position: Position,
    *,
    raise_count: int = 0,
    limpers: int = 0,
    callers_after_raise: int = 0,
    opener_position: Position | None = None,
    can_check: bool = False,
) -> PreflopSituation:
    return PreflopSituation(
        hole_cards=cards,
        position=position,
        raise_count=raise_count,
        limpers=limpers,
        callers_after_raise=callers_after_raise,
        opener_position=opener_position,
        effective_stack_bb=100.0,
        can_check=can_check,
    )


def _representative_situations(
    cards: tuple[Card, Card],
    position: Position,
) -> tuple[PreflopSituation, ...]:
    return (
        _situation(cards, position, can_check=True),
        _situation(cards, position),
        _situation(cards, position, limpers=3),
        _situation(
            cards,
            position,
            raise_count=1,
            opener_position=Position.HJ,
        ),
        _situation(
            cards,
            position,
            raise_count=1,
            callers_after_raise=1,
            opener_position=Position.HJ,
        ),
        _situation(
            cards,
            position,
            raise_count=2,
            opener_position=Position.HJ,
        ),
    )


def test_all_169_canonical_hand_keys_are_unique_and_order_independent() -> None:
    assert len(CANONICAL_HANDS) == 169
    assert len({expected for expected, _ in CANONICAL_HANDS}) == 169

    actual = {hand_shape(cards).key for _, cards in CANONICAL_HANDS}
    assert actual == {expected for expected, _ in CANONICAL_HANDS}

    for expected, cards in CANONICAL_HANDS:
        assert hand_shape(cards).key == expected
        assert hand_shape(tuple(reversed(cards))).key == expected


def test_every_hand_position_and_representative_spot_builds_a_valid_plan() -> None:
    checked = 0
    for expected_key, cards in CANONICAL_HANDS:
        for position in Position:
            for situation in _representative_situations(cards, position):
                plan = build_preflop_plan(situation)
                actions = [option.action for option in plan.options]
                primary = [
                    option
                    for option in plan.options
                    if option.role == ActionRole.PRIMARY
                ]

                assert plan.hand_key == expected_key
                assert plan.position == position
                assert plan.spot == situation.spot
                assert len(primary) == 1
                assert plan.primary == primary[0]
                assert len(actions) == len(set(actions))
                checked += 1

    assert checked == 169 * len(Position) * 6


@pytest.mark.parametrize(
    ("cards", "position", "expected_action"),
    [
        (_cards("Ks", "9h"), Position.BTN, StrategicAction.RAISE),
        (_cards("9s", "6h"), Position.BTN, StrategicAction.FOLD),
        (_cards("Qs", "9s"), Position.UTG, StrategicAction.FOLD),
        (_cards("Qs", "9s"), Position.CO, StrategicAction.RAISE),
        (_cards("6s", "5s"), Position.BTN, StrategicAction.RAISE),
    ],
)
def test_unopened_range_boundaries(
    cards: tuple[Card, Card],
    position: Position,
    expected_action: StrategicAction,
) -> None:
    plan = build_preflop_plan(_situation(cards, position))

    assert plan.primary.action == expected_action


def test_small_pair_in_small_blind_prefers_completing_multi_limp_pot() -> None:
    plan = build_preflop_plan(
        _situation(_cards("2s", "2h"), Position.SB, limpers=3)
    )

    assert plan.primary.action == StrategicAction.CALL
    assert plan.option_for(StrategicAction.RAISE).role == ActionRole.DISCOURAGED


def test_a5s_cutoff_facing_hijack_open_is_fold_threebet_mix() -> None:
    plan = build_preflop_plan(
        _situation(
            _cards("As", "5s"),
            Position.CO,
            raise_count=1,
            opener_position=Position.HJ,
        )
    )

    assert plan.primary.action == StrategicAction.FOLD
    assert plan.option_for(StrategicAction.RAISE).role == ActionRole.ACCEPTABLE
    assert plan.option_for(StrategicAction.CALL).role == ActionRole.DISCOURAGED
