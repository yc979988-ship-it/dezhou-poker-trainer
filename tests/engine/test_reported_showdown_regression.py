"""用户报告的摊牌场景回归测试。"""

from poker_trainer.engine.cards import parse_cards
from poker_trainer.engine.evaluator import HandCategory, evaluate
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position, Seat


BOARD = tuple(parse_cards("Qd Ts 9d 6d 6h"))


def test_reported_two_pair_hands_compare_pairs_before_kicker():
    """TT99 > TT66 > 9966；A 踢脚不能越过更高的对子。"""

    hero = evaluate([*parse_cards("Td 4s"), *BOARD])
    utg = evaluate([*parse_cards("As 9h"), *BOARD])
    sb = evaluate([*parse_cards("9c Tc"), *BOARD])

    assert hero.category == utg.category == sb.category == HandCategory.TWO_PAIR
    assert hero.kickers == (10, 6, 12)  # TT66Q
    assert utg.kickers == (9, 6, 14)  # 9966A
    assert sb.kickers == (10, 9, 12)  # TT99Q
    assert sb > hero > utg


def test_reported_showdown_awards_the_1120_pot_to_sb():
    """复现截图的公共牌、三家底牌、英雄投入和总底池。"""

    seats = [
        Seat(position.value, position.value, position, 4_000)
        for position in (
            Position.UTG,
            Position.HJ,
            Position.CO,
            Position.BTN,
            Position.SB,
            Position.BB,
        )
    ]
    hand = HoldemHand(
        seats,
        seed=20260907,
        hole_overrides={
            "BB": ["Td", "4s"],
            "UTG": ["As", "9h"],
            "SB": ["9c", "Tc"],
        },
        board_override=BOARD,
    )

    # 三家各投入360，已弃牌的HJ留下40，共形成1,120的单一逻辑主池。
    actions = (
        ("UTG", ActionType.CALL, None),
        ("HJ", ActionType.CALL, None),
        ("CO", ActionType.FOLD, None),
        ("BTN", ActionType.FOLD, None),
        ("SB", ActionType.CALL, None),
        ("BB", ActionType.RAISE, 120),
        ("UTG", ActionType.CALL, None),
        ("HJ", ActionType.FOLD, None),
        ("SB", ActionType.CALL, None),
        ("SB", ActionType.CHECK, None),
        ("BB", ActionType.BET, 120),
        ("UTG", ActionType.CALL, None),
        ("SB", ActionType.CALL, None),
        ("SB", ActionType.CHECK, None),
        ("BB", ActionType.BET, 120),
        ("UTG", ActionType.CALL, None),
        ("SB", ActionType.CALL, None),
        ("SB", ActionType.CHECK, None),
        ("BB", ActionType.CHECK, None),
        ("UTG", ActionType.CHECK, None),
    )
    for player_id, action, amount in actions:
        hand.act(player_id, action, amount)

    assert hand.is_complete
    assert tuple(hand.board) == BOARD
    assert hand.committed_pot == 1_120
    assert [(pot.amount, pot.eligible) for pot in hand.result.pots] == [
        (1_120, ("UTG", "SB", "BB"))
    ]
    assert hand.result.payouts == {
        "UTG": 0,
        "HJ": 0,
        "CO": 0,
        "BTN": 0,
        "SB": 1_120,
        "BB": 0,
    }
    assert hand.player("BB").stack - hand.player("BB").starting_stack == -360
    assert hand.player("SB").stack - hand.player("SB").starting_stack == 760
    hand.assert_chip_conservation()
