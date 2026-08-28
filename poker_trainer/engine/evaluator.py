"""Complete five-to-seven card poker hand evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering
from itertools import combinations
from typing import Iterable

from .cards import Card, parse_cards


class HandCategory(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


_CATEGORY_NAMES_ZH = {
    HandCategory.HIGH_CARD: "高牌",
    HandCategory.ONE_PAIR: "一对",
    HandCategory.TWO_PAIR: "两对",
    HandCategory.THREE_OF_A_KIND: "三条",
    HandCategory.STRAIGHT: "顺子",
    HandCategory.FLUSH: "同花",
    HandCategory.FULL_HOUSE: "葫芦",
    HandCategory.FOUR_OF_A_KIND: "四条",
    HandCategory.STRAIGHT_FLUSH: "同花顺",
}


@total_ordering
@dataclass(frozen=True, eq=False, slots=True)
class HandRank:
    """Comparable result; suits never break a poker tie."""

    category: HandCategory
    kickers: tuple[int, ...]
    best_five: tuple[Card, ...]

    @property
    def score(self) -> tuple[int, ...]:
        return (int(self.category), *self.kickers)

    @property
    def name_zh(self) -> str:
        if self.category == HandCategory.STRAIGHT_FLUSH and self.kickers == (14,):
            return "皇家同花顺"
        return _CATEGORY_NAMES_ZH[self.category]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.score == other.score

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.score < other.score

    def __hash__(self) -> int:
        return hash(self.score)


def _normalise_cards(cards: Iterable[Card | str]) -> list[Card]:
    parsed = parse_cards(cards)
    if len(set(parsed)) != len(parsed):
        raise ValueError("同一张牌不能重复出现")
    return parsed


def _straight_high(ranks: Iterable[int]) -> int | None:
    unique = set(ranks)
    if {14, 2, 3, 4, 5}.issubset(unique):
        unique.add(1)
    ordered = sorted(unique)
    for high_index in range(len(ordered) - 1, 3, -1):
        window = ordered[high_index - 4 : high_index + 1]
        if window[-1] - window[0] == 4:
            return 5 if window[-1] == 5 and 1 in window else window[-1]
    return None


def _ordered_five(cards: list[Card], category: HandCategory, kickers: tuple[int, ...]) -> tuple[Card, ...]:
    """Give deterministic display order without affecting hand comparison."""

    if category in (HandCategory.STRAIGHT, HandCategory.STRAIGHT_FLUSH) and kickers == (5,):
        rank_order = {5: 5, 4: 4, 3: 3, 2: 2, 14: 1}
        return tuple(sorted(cards, key=lambda card: (rank_order[card.rank], card.suit), reverse=True))
    counts = Counter(card.rank for card in cards)
    return tuple(
        sorted(cards, key=lambda card: (counts[card.rank], card.rank, card.suit), reverse=True)
    )


def evaluate_five(cards: Iterable[Card | str]) -> HandRank:
    """Evaluate exactly five cards."""

    hand = _normalise_cards(cards)
    if len(hand) != 5:
        raise ValueError("五张牌评估必须恰好传入5张牌")

    ranks = [card.rank for card in hand]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({card.suit for card in hand}) == 1
    straight_high = _straight_high(ranks)

    if flush and straight_high is not None:
        category = HandCategory.STRAIGHT_FLUSH
        kickers = (straight_high,)
    elif groups[0][0] == 4:
        quad_rank = groups[0][1]
        category = HandCategory.FOUR_OF_A_KIND
        kickers = (quad_rank, max(rank for rank in ranks if rank != quad_rank))
    elif sorted(counts.values()) == [2, 3]:
        trip_rank = max(rank for rank, count in counts.items() if count == 3)
        pair_rank = max(rank for rank, count in counts.items() if count == 2)
        category = HandCategory.FULL_HOUSE
        kickers = (trip_rank, pair_rank)
    elif flush:
        category = HandCategory.FLUSH
        kickers = tuple(sorted(ranks, reverse=True))
    elif straight_high is not None:
        category = HandCategory.STRAIGHT
        kickers = (straight_high,)
    elif groups[0][0] == 3:
        trip_rank = groups[0][1]
        category = HandCategory.THREE_OF_A_KIND
        kickers = (trip_rank, *sorted((rank for rank in ranks if rank != trip_rank), reverse=True))
    elif sorted(counts.values()) == [1, 2, 2]:
        pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
        lone = next(rank for rank, count in counts.items() if count == 1)
        category = HandCategory.TWO_PAIR
        kickers = (pairs[0], pairs[1], lone)
    elif groups[0][0] == 2:
        pair_rank = groups[0][1]
        category = HandCategory.ONE_PAIR
        kickers = (pair_rank, *sorted((rank for rank in ranks if rank != pair_rank), reverse=True))
    else:
        category = HandCategory.HIGH_CARD
        kickers = tuple(sorted(ranks, reverse=True))

    return HandRank(category, kickers, _ordered_five(hand, category, kickers))


def evaluate(cards: Iterable[Card | str]) -> HandRank:
    """Return the strongest five-card hand from five, six or seven cards."""

    hand = _normalise_cards(cards)
    if not 5 <= len(hand) <= 7:
        raise ValueError("牌型评估只接受5至7张牌")
    return max(evaluate_five(combo) for combo in combinations(hand, 5))


best_hand = evaluate


def compare_hands(first: Iterable[Card | str], second: Iterable[Card | str]) -> int:
    """Return 1 when first wins, -1 when second wins, and 0 for a tie."""

    first_rank = evaluate(first)
    second_rank = evaluate(second)
    return (first_rank > second_rank) - (first_rank < second_rank)

