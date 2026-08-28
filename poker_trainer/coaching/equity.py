"""Pure poker maths used by the offline coaching layer.

The functions in this module deliberately have no dependency on a running
hand or its deck.  Equity simulations build their own card universe and use a
private random-number generator, so asking the coach for feedback cannot alter
gameplay or replay state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import comb, isfinite
import random
from typing import Iterable, Sequence

from poker_trainer.engine.cards import Card, SUIT_CHARS, parse_cards
from poker_trainer.engine.evaluator import evaluate


CardInput = str | Card
CardsInput = str | Iterable[CardInput]

_FULL_DECK = tuple(
    Card(rank, suit)
    for suit in SUIT_CHARS
    for rank in range(2, 15)
)
_STRAIGHT_WINDOWS = (
    frozenset((14, 2, 3, 4, 5)),
    *(frozenset(range(low, low + 5)) for low in range(2, 11)),
)


@dataclass(frozen=True, slots=True)
class DrawAnalysis:
    """Aggregate common-draw information with overlapping outs removed.

    ``hit_next`` and ``hit_by_river`` are probabilities in the closed interval
    0..1.  ``out_cards`` is kept in a canonical deck order, making results easy
    to test, display, and replay deterministically.
    """

    names: tuple[str, ...]
    out_cards: tuple[Card, ...]
    outs: int
    hit_next: float
    hit_by_river: float

    @property
    def name(self) -> str:
        """A compact display label; ``names`` remains the lossless value."""

        return " + ".join(self.names) if self.names else "无常见听牌"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly values for Streamlit or persistence."""

        return {
            "names": list(self.names),
            "out_cards": [str(card) for card in self.out_cards],
            "outs": self.outs,
            "hit_next": self.hit_next,
            "hit_by_river": self.hit_by_river,
        }


def _as_nonnegative_number(value: int | float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}必须是数字")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name}必须是有限数字")
    if number < 0:
        raise ValueError(f"{field_name}不能为负数")
    return number


def calculate_pot_odds(pot_before: int | float, call_amount: int | float) -> float:
    """Return the break-even equity for a call.

    ``pot_before`` is the pot visible before the hero calls.  Consequently the
    final contestable pot is ``pot_before + call_amount`` and pot odds are
    ``call_amount / final_pot``.  Checking costs nothing and therefore returns
    zero.
    """

    pot = _as_nonnegative_number(pot_before, "跟注前底池")
    call = _as_nonnegative_number(call_amount, "跟注额")
    if call == 0:
        return 0.0
    return call / (pot + call)


def _normalise_holdem_cards(
    hole_cards: CardsInput,
    board: CardsInput,
) -> tuple[tuple[Card, Card], tuple[Card, ...]]:
    hole = tuple(parse_cards(hole_cards))
    community = tuple(parse_cards(board))
    if len(hole) != 2:
        raise ValueError("德州扑克手牌必须恰好为2张")
    if len(community) > 5:
        raise ValueError("公共牌不能超过5张")
    all_known = (*hole, *community)
    if len(set(all_known)) != len(all_known):
        raise ValueError("已知牌中不能出现重复牌")
    return (hole[0], hole[1]), community


def _has_made_straight(ranks: set[int]) -> bool:
    return any(window.issubset(ranks) for window in _STRAIGHT_WINDOWS)


def _draw_probabilities(outs: int, unseen: int, cards_to_come: int) -> tuple[float, float]:
    if outs <= 0 or unseen <= 0 or cards_to_come <= 0:
        return 0.0, 0.0

    next_card = outs / unseen
    draws = min(cards_to_come, unseen)
    misses = unseen - outs
    if misses < draws:
        by_river = 1.0
    else:
        by_river = 1.0 - comb(misses, draws) / comb(unseen, draws)
    return next_card, by_river


def analyze_common_draws(hole_cards: CardsInput, board: CardsInput) -> DrawAnalysis:
    """Find common raw draws and return their de-duplicated aggregate outs.

    The MVP recognises four-card flush draws, open-ended/double-ended straight
    draws, and gutshots.  A made flush is not also labelled a flush draw, and a
    made straight is not also labelled a straight draw.  Outs shared by a
    straight and flush draw appear only once in the aggregate result.
    """

    hole, community = _normalise_holdem_cards(hole_cards, board)
    known = set((*hole, *community))
    unseen_count = 52 - len(known)
    cards_to_come = 5 - len(community)

    # There is no future card on the river, so an incomplete hand is no longer
    # a draw even if four cards of a suit or sequence happen to be visible.
    if cards_to_come <= 0:
        return DrawAnalysis((), (), 0, 0.0, 0.0)

    names: list[str] = []
    out_set: set[Card] = set()

    suit_counts = Counter(card.suit for card in known)
    made_flush = any(count >= 5 for count in suit_counts.values())
    if not made_flush:
        flush_suits = tuple(
            suit
            for suit in SUIT_CHARS
            if suit_counts.get(suit, 0) == 4
            and any(card.suit == suit for card in hole)
        )
        if flush_suits:
            names.append("同花听牌")
            out_set.update(
                card
                for card in _FULL_DECK
                if card.suit in flush_suits and card not in known
            )

    ranks = {card.rank for card in known}
    if not _has_made_straight(ranks):
        board_ranks = {card.rank for card in community}
        missing_ranks = {
            next(iter(window - ranks))
            for window in _STRAIGHT_WINDOWS
            if len(window & ranks) == 4
            and any(
                card.rank in window and card.rank not in board_ranks
                for card in hole
            )
        }
        if missing_ranks:
            names.append("两头顺听牌" if len(missing_ranks) >= 2 else "卡顺听牌")
            out_set.update(
                card
                for card in _FULL_DECK
                if card.rank in missing_ranks and card not in known
            )

    out_cards = tuple(card for card in _FULL_DECK if card in out_set)
    hit_next, hit_by_river = _draw_probabilities(
        len(out_cards), unseen_count, cards_to_come
    )
    return DrawAnalysis(
        names=tuple(names),
        out_cards=out_cards,
        outs=len(out_cards),
        hit_next=hit_next,
        hit_by_river=hit_by_river,
    )


def _normalise_opponent_holes(
    known_opponent_holes: Sequence[CardsInput] | None,
    opponents: int,
) -> tuple[tuple[Card, Card], ...]:
    if known_opponent_holes is None:
        return ()
    hands: list[tuple[Card, Card]] = []
    for cards in known_opponent_holes:
        parsed = tuple(parse_cards(cards))
        if len(parsed) != 2:
            raise ValueError("每名已知对手必须恰好有2张手牌")
        hands.append((parsed[0], parsed[1]))
    if len(hands) > opponents:
        raise ValueError("已知对手手牌数量不能超过对手人数")
    return tuple(hands)


def estimate_equity(
    hero_hole: CardsInput,
    board: CardsInput,
    opponents: int,
    trials: int = 2_000,
    seed: int | str | bytes | None = None,
    known_opponent_holes: Sequence[CardsInput] | None = None,
) -> float:
    """Estimate showdown equity against uniformly sampled unknown hands.

    Ties split the pot equally between all tied winners.  Known opponent hands
    may be supplied for review scenarios; remaining opponents are sampled.  A
    fresh local ``random.Random`` instance and an internal 52-card universe are
    used on every call, so neither global randomness nor a live hand's deck is
    read or advanced.
    """

    if isinstance(opponents, bool) or not isinstance(opponents, int) or opponents < 1:
        raise ValueError("对手人数必须是正整数")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("模拟次数必须是正整数")

    hero, community = _normalise_holdem_cards(hero_hole, board)
    fixed_opponents = _normalise_opponent_holes(known_opponent_holes, opponents)
    all_known = [*hero, *community]
    for hand in fixed_opponents:
        all_known.extend(hand)
    if len(set(all_known)) != len(all_known):
        raise ValueError("英雄、公共牌和已知对手手牌不能重复")

    known_set = set(all_known)
    unseen = tuple(card for card in _FULL_DECK if card not in known_set)
    board_needed = 5 - len(community)
    unknown_opponents = opponents - len(fixed_opponents)
    cards_needed = board_needed + unknown_opponents * 2
    if cards_needed > len(unseen):
        raise ValueError("对手人数与剩余可用牌数不一致")

    rng = random.Random(seed)
    equity_total = 0.0
    for _ in range(trials):
        sampled = rng.sample(unseen, cards_needed)
        completed_board = (*community, *sampled[:board_needed])
        cursor = board_needed
        opponent_hands: list[tuple[Card, Card]] = list(fixed_opponents)
        for _opponent_index in range(unknown_opponents):
            opponent_hands.append((sampled[cursor], sampled[cursor + 1]))
            cursor += 2

        hero_rank = evaluate((*hero, *completed_board))
        opponent_ranks = [
            evaluate((*opponent_hole, *completed_board))
            for opponent_hole in opponent_hands
        ]
        best_rank = max((hero_rank, *opponent_ranks))
        if hero_rank == best_rank:
            tied_opponents = sum(rank == hero_rank for rank in opponent_ranks)
            equity_total += 1.0 / (1 + tied_opponents)

    return equity_total / trials


__all__ = [
    "DrawAnalysis",
    "analyze_common_draws",
    "calculate_pot_odds",
    "estimate_equity",
]

