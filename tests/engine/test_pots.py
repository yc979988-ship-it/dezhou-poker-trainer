from poker_trainer.engine.models import PlayerState, Position, Pot
from poker_trainer.engine.pots import build_pots


def state(player_id, position, contribution, *, folded=False):
    return PlayerState(
        player_id=player_id,
        name=player_id,
        position=position,
        stack=0,
        starting_stack=contribution,
        total_commitment=contribution,
        folded=folded,
        all_in=True,
    )


def test_multiplayer_main_and_side_pots_include_folded_contributions():
    players = [
        state("A", Position.UTG, 100),
        state("B", Position.HJ, 300),
        state("C", Position.CO, 500),
        state("D", Position.BTN, 500, folded=True),
    ]
    pots = build_pots(players)
    assert [pot.amount for pot in pots] == [400, 600, 400]
    assert pots[0].eligible == ("A", "B", "C")
    assert pots[1].eligible == ("B", "C")
    assert pots[2].eligible == ("C",)
    assert sum(pot.amount for pot in pots) == 1400


def test_folded_player_is_never_eligible_for_any_layer():
    players = [
        state("A", Position.UTG, 100),
        state("B", Position.HJ, 300, folded=True),
        state("C", Position.CO, 300),
    ]
    assert all("B" not in pot.eligible for pot in build_pots(players))


def test_folded_commitment_levels_do_not_create_duplicate_logical_side_pots():
    """同一获奖人集合必须合并，否则逐层平分会重复分配奇数筹码。"""

    players = [
        state("A", Position.UTG, 5),
        state("B", Position.HJ, 5),
        state("C", Position.CO, 1, folded=True),
        state("D", Position.BTN, 2, folded=True),
        state("E", Position.SB, 3, folded=True),
        state("F", Position.BB, 4, folded=True),
    ]

    assert build_pots(players) == (
        # 总池20应一次性平分；若按1/2/3/4/5五层分别结算，两个奇数层
        # 会把两个余数都给A，错误地产生11:9。
        Pot(
            amount=20,
            cap=5,
            contributors=("A", "B", "C", "D", "E", "F"),
            eligible=("A", "B"),
        ),
    )

