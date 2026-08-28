from __future__ import annotations

import pytest

from poker_trainer.engine.models import Position, Seat


@pytest.fixture
def six_seats():
    def factory(stacks: dict[Position, int] | None = None) -> list[Seat]:
        stacks = stacks or {}
        return [
            Seat(position.value, position.value, position, stacks.get(position, 4000))
            for position in (
                Position.UTG,
                Position.HJ,
                Position.CO,
                Position.BTN,
                Position.SB,
                Position.BB,
            )
        ]

    return factory


