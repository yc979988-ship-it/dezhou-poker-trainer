import pytest

from poker_trainer.engine.cards import Card, Deck, parse_cards
from poker_trainer.engine.evaluator import HandCategory, compare_hands, evaluate


def test_card_parsing_and_seeded_deck_are_deterministic():
    assert Card.from_str("As") == Card(14, "s")
    assert Card.parse("Td") == Card(10, "d")
    assert str(Card.from_str("10H")) == "Th"
    assert len(set(Deck(seed=20260827).cards)) == 52
    assert Deck(seed=17).deal(7) == Deck(seed=17).deal(7)
    assert Deck(seed=17).deal(7) != Deck(seed=18).deal(7)


@pytest.mark.parametrize(
    ("cards", "category"),
    [
        ("As Jd 9c 5h 2s", HandCategory.HIGH_CARD),
        ("As Ad 9c 5h 2s", HandCategory.ONE_PAIR),
        ("As Ad 9c 9h 2s", HandCategory.TWO_PAIR),
        ("As Ad Ac 5h 2s", HandCategory.THREE_OF_A_KIND),
        ("9s 8d 7c 6h 5s", HandCategory.STRAIGHT),
        ("As Js 9s 5s 2s", HandCategory.FLUSH),
        ("As Ad Ac 5h 5s", HandCategory.FULL_HOUSE),
        ("As Ad Ac Ah 2s", HandCategory.FOUR_OF_A_KIND),
        ("9s 8s 7s 6s 5s", HandCategory.STRAIGHT_FLUSH),
    ],
)
def test_every_hand_category(cards, category):
    assert evaluate(parse_cards(cards)).category == category


def test_hand_categories_sort_from_high_card_to_straight_flush():
    hands = [
        "As Jd 9c 5h 2s",
        "As Ad 9c 5h 2s",
        "As Ad 9c 9h 2s",
        "As Ad Ac 5h 2s",
        "9s 8d 7c 6h 5s",
        "As Js 9s 5s 2s",
        "As Ad Ac 5h 5s",
        "As Ad Ac Ah 2s",
        "9s 8s 7s 6s 5s",
    ]
    ranks = [evaluate(parse_cards(cards)) for cards in hands]
    assert ranks == sorted(ranks)


def test_kickers_break_ties_in_correct_order():
    ace_king = evaluate(parse_cards("As Ad Kh Qc Js"))
    ace_queen = evaluate(parse_cards("Ah Ac Qh Jc Ts"))
    assert ace_king > ace_queen

    queens_over_twos = evaluate(parse_cards("Qs Qd 2h 2c As"))
    jacks_over_tens = evaluate(parse_cards("Jh Jc Ts Td As"))
    assert queens_over_twos > jacks_over_tens


def test_suits_do_not_break_an_exact_poker_tie():
    first = parse_cards("As Kd Qc Jh 9s")
    second = parse_cards("Ah Ks Qd Jc 9h")
    assert evaluate(first) == evaluate(second)
    assert compare_hands(first, second) == 0


def test_seven_cards_choose_the_best_five():
    rank = evaluate(parse_cards("As Ks Qs Js Ts 2d 2c"))
    assert rank.category == HandCategory.STRAIGHT_FLUSH
    assert rank.kickers == (14,)
    assert rank.name_zh == "皇家同花顺"
    assert set(rank.best_five) == set(parse_cards("As Ks Qs Js Ts"))


def test_wheel_is_five_high_and_loses_to_six_high_straight():
    wheel = evaluate(parse_cards("As 2d 3c 4h 5s Kd Qc"))
    six_high = evaluate(parse_cards("2s 3d 4c 5h 6s Kd Qc"))
    assert wheel.category == HandCategory.STRAIGHT
    assert wheel.kickers == (5,)
    assert six_high > wheel


def test_duplicate_cards_and_invalid_card_count_are_rejected():
    with pytest.raises(ValueError, match="重复"):
        evaluate(parse_cards("As As Qc Jh 9s"))
    with pytest.raises(ValueError, match="5至7"):
        evaluate(parse_cards("As Kd Qc Jh"))

