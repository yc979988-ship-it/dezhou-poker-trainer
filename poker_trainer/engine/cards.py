"""Playing-card primitives and a deterministic Texas Hold'em deck."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Iterator


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
RANK_TO_VALUE = {rank: value for value, rank in enumerate(RANK_CHARS, start=2)}
VALUE_TO_RANK = {value: rank for rank, value in RANK_TO_VALUE.items()}


@dataclass(frozen=True, slots=True)
class Card:
    """A standard playing card.

    Ranks are stored as integers from 2 through 14 (ace high). Suits use the
    compact poker notation ``c``, ``d``, ``h`` and ``s``.
    """

    rank: int
    suit: str

    def __post_init__(self) -> None:
        rank = self.rank
        if isinstance(rank, str):
            rank_text = rank.strip().upper()
            if rank_text == "10":
                rank_text = "T"
            try:
                rank = RANK_TO_VALUE[rank_text]
            except KeyError as exc:
                raise ValueError(f"无效点数: {self.rank!r}") from exc

        suit = self.suit.strip().lower() if isinstance(self.suit, str) else self.suit
        if rank not in VALUE_TO_RANK:
            raise ValueError(f"无效点数: {self.rank!r}")
        if suit not in SUIT_CHARS:
            raise ValueError(f"无效花色: {self.suit!r}")

        object.__setattr__(self, "rank", int(rank))
        object.__setattr__(self, "suit", suit)

    @classmethod
    def from_str(cls, value: str) -> "Card":
        """Parse compact notation such as ``As``, ``Td`` or ``10h``."""

        if not isinstance(value, str):
            raise TypeError("牌面必须是字符串")
        text = value.strip()
        if len(text) == 3 and text[:2] == "10":
            text = f"T{text[2]}"
        if len(text) != 2:
            raise ValueError(f"无效牌面: {value!r}")
        return cls(text[0], text[1])

    parse = from_str

    def __str__(self) -> str:
        return f"{VALUE_TO_RANK[self.rank]}{self.suit}"

    def __repr__(self) -> str:
        return f"Card('{self}')"


def parse_cards(values: str | Iterable[str | Card]) -> list[Card]:
    """Parse cards from whitespace-separated text or an iterable.

    ``parse_cards("As Td 7c")`` and ``parse_cards(["As", "Td", "7c"])`` are
    equivalent. Existing :class:`Card` instances pass through unchanged.
    """

    items: Iterable[str | Card]
    if isinstance(values, str):
        items = values.replace(",", " ").split()
    else:
        items = values
    return [item if isinstance(item, Card) else Card.from_str(item) for item in items]


class Deck:
    """A 52-card deck using a private RNG for reproducible replay."""

    def __init__(self, seed: int | str | bytes | None = None, *, shuffle: bool = True):
        self.seed = seed
        self._rng = random.Random(seed)
        self.cards: list[Card] = self._fresh_cards()
        if shuffle:
            self.shuffle()

    @staticmethod
    def _fresh_cards() -> list[Card]:
        return [Card(rank, suit) for suit in SUIT_CHARS for rank in range(2, 15)]

    @property
    def remaining(self) -> int:
        return len(self.cards)

    def __len__(self) -> int:
        return self.remaining

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def shuffle(self, seed: int | str | bytes | None = None) -> None:
        """Shuffle the cards in place, optionally restarting from a new seed."""

        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self._rng.shuffle(self.cards)

    def reset(self, seed: int | str | bytes | None = None, *, shuffle: bool = True) -> None:
        """Restore all 52 cards and optionally shuffle them reproducibly."""

        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self.cards = self._fresh_cards()
        if shuffle:
            self.shuffle()

    def deal(self, count: int = 1) -> Card | list[Card]:
        """Deal from the top; one card returns a card, larger counts a list."""

        if not isinstance(count, int) or count < 1:
            raise ValueError("发牌张数必须是正整数")
        if count > self.remaining:
            raise ValueError("牌堆剩余牌数不足")
        dealt = [self.cards.pop() for _ in range(count)]
        return dealt[0] if count == 1 else dealt

    draw = deal


