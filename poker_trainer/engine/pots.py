"""从玩家累计投入分层生成主池和边池。"""

from __future__ import annotations

from collections.abc import Iterable

from .models import PlayerState, Pot


def build_pots(players: Iterable[PlayerState]) -> tuple[Pot, ...]:
    """按累计投入分层并合并同资格层；弃牌者贡献但不能获奖。"""

    states = list(players)
    levels = sorted({player.total_commitment for player in states if player.total_commitment > 0})
    previous = 0
    pots: list[Pot] = []
    for level in levels:
        contributors = tuple(
            player.player_id for player in states if player.total_commitment >= level
        )
        amount = (level - previous) * len(contributors)
        eligible = tuple(
            player.player_id
            for player in states
            if player.total_commitment >= level and not player.folded
        )
        if amount:
            # 只有获奖资格发生变化时才形成新的逻辑边池。弃牌玩家在较低
            # 档位停止贡献，不应凭空切出一个获奖人完全相同的“边池”。
            # 除了展示语义，这也关系到平分时的奇数筹码：若把同一池拆成
            # 多层并逐层取整，可能把多个余数错误地都发给同一位玩家。
            if pots and pots[-1].eligible == eligible:
                previous_pot = pots[-1]
                pots[-1] = Pot(
                    amount=previous_pot.amount + amount,
                    cap=level,
                    contributors=previous_pot.contributors,
                    eligible=eligible,
                )
            else:
                pots.append(Pot(amount, level, contributors, eligible))
        previous = level
    return tuple(pots)

