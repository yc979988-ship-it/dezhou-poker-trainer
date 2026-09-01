"""6-max 100bb 翻后场景特征。

本模块只把公开牌面转成可解释的牌力、牌面与人数特征。评分规则仍由
``coach.py`` 结合实际动作和下注价格完成；随机未知手牌权益只作参考，
不在这里冒充对手下注范围。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from poker_trainer.engine.cards import Card
from poker_trainer.engine.evaluator import HandCategory, HandRank, evaluate


class MadeStrength(str, Enum):
    AIR = "air"
    BOARD_ONLY = "board_only"
    WEAK_SHOWDOWN = "weak_showdown"
    WEAK_PAIR = "weak_pair"
    MEDIUM_PAIR = "medium_pair"
    TOP_PAIR_WEAK = "top_pair_weak"
    TOP_PAIR_STRONG = "top_pair_strong"
    OVERPAIR = "overpair"
    STRONG_MADE = "strong_made"

    @property
    def label_zh(self) -> str:
        return {
            MadeStrength.AIR: "空气牌",
            MadeStrength.BOARD_ONLY: "公牌成牌",
            MadeStrength.WEAK_SHOWDOWN: "底牌踢脚摊牌值",
            MadeStrength.WEAK_PAIR: "弱对子",
            MadeStrength.MEDIUM_PAIR: "中等对子",
            MadeStrength.TOP_PAIR_WEAK: "弱顶对",
            MadeStrength.TOP_PAIR_STRONG: "强顶对",
            MadeStrength.OVERPAIR: "超对",
            MadeStrength.STRONG_MADE: "两对及以上",
        }[self]


class BoardWetness(str, Enum):
    DRY = "dry"
    COORDINATED = "coordinated"
    WET = "wet"

    @property
    def label_zh(self) -> str:
        return {
            BoardWetness.DRY: "干燥牌面",
            BoardWetness.COORDINATED: "一般牌面",
            BoardWetness.WET: "湿润牌面",
        }[self]


@dataclass(frozen=True, slots=True)
class BoardTexture:
    wetness: BoardWetness
    paired: bool
    two_tone: bool
    three_flush: bool
    four_flush: bool
    connected: bool
    four_straight: bool

    @property
    def label_zh(self) -> str:
        features: list[str] = []
        if self.paired:
            features.append("成对")
        if self.four_flush:
            features.append("四同花")
        elif self.three_flush:
            features.append("三同花")
        elif self.two_tone:
            features.append("两同花")
        if self.connected:
            features.append("四连顺" if self.four_straight else "连张")
        suffix = f"（{'、'.join(features)}）" if features else ""
        return f"{self.wetness.label_zh}{suffix}"


@dataclass(frozen=True, slots=True)
class PostflopProfile:
    rank: HandRank
    strength: MadeStrength
    personal_made: bool
    texture: BoardTexture
    active_players: int
    board_rank: HandRank | None
    board_plays: bool
    hole_cards_play: bool

    @property
    def multiway(self) -> bool:
        return self.active_players >= 3

    @property
    def players_label_zh(self) -> str:
        return f"{self.active_players} 人底池" if self.multiway else "单挑底池"

    @property
    def board_locked(self) -> bool:
        """公共牌是否已锁死所有玩家的最终五张牌。"""

        if not self.board_plays or self.board_rank is None:
            return False
        if (
            self.board_rank.category == HandCategory.STRAIGHT_FLUSH
            and self.board_rank.kickers == (14,)
        ):
            return True
        if self.board_rank.category == HandCategory.FOUR_OF_A_KIND:
            quad_rank, kicker = self.board_rank.kickers
            highest_available_kicker = 13 if quad_rank == 14 else 14
            return kicker == highest_available_kicker
        return False


def _personal_made(rank: HandRank, hole_cards: tuple[Card, Card]) -> bool:
    hole_set = set(hole_cards)
    if rank.category == HandCategory.HIGH_CARD:
        return False
    if rank.category == HandCategory.ONE_PAIR:
        return any(card.rank == rank.kickers[0] for card in hole_set)
    if rank.category == HandCategory.TWO_PAIR:
        return any(card.rank in rank.kickers[:2] for card in hole_set)
    if rank.category == HandCategory.THREE_OF_A_KIND:
        return any(card.rank == rank.kickers[0] for card in hole_set)
    return any(card in rank.best_five for card in hole_set)


def _board_texture(board: tuple[Card, ...]) -> BoardTexture:
    rank_counts = Counter(card.rank for card in board)
    suit_counts = Counter(card.suit for card in board)
    max_suit = max(suit_counts.values(), default=0)
    paired = any(count >= 2 for count in rank_counts.values())

    ranks = set(rank_counts)
    if 14 in ranks:
        ranks.add(1)
    window_hits = max(
        (sum(low <= rank <= low + 4 for rank in ranks) for low in range(1, 11)),
        default=0,
    )
    # “连张”至少需要三个不同点数；AAK、887 不能因为只有两个不同点数
    # 就被误标成典型湿润连张面。
    connected = window_hits >= 3
    four_straight = window_hits >= 4

    wet_score = 0
    if max_suit >= 4:
        wet_score += 3
    elif max_suit == 3:
        wet_score += 2
    elif max_suit == 2:
        wet_score += 1
    if four_straight:
        wet_score += 3
    elif connected:
        wet_score += 2
    if paired:
        wet_score += 1
    wetness = (
        BoardWetness.WET
        if wet_score >= 3
        else BoardWetness.COORDINATED
        if wet_score >= 2
        else BoardWetness.DRY
    )
    return BoardTexture(
        wetness=wetness,
        paired=paired,
        two_tone=max_suit == 2,
        three_flush=max_suit == 3,
        four_flush=max_suit >= 4,
        connected=connected,
        four_straight=four_straight,
    )


def _pair_rank_strength(
    pair_rank: int,
    hole_cards: tuple[Card, Card],
    board: tuple[Card, ...],
) -> MadeStrength:
    hole_ranks = (hole_cards[0].rank, hole_cards[1].rank)
    board_ranks = {card.rank for card in board}
    if hole_ranks[0] == hole_ranks[1] == pair_rank:
        if pair_rank > max(board_ranks):
            return MadeStrength.OVERPAIR
        higher = sum(rank_value > pair_rank for rank_value in board_ranks)
        return MadeStrength.MEDIUM_PAIR if higher <= 1 else MadeStrength.WEAK_PAIR

    matching_indexes = [
        index for index, card in enumerate(hole_cards) if card.rank == pair_rank
    ]
    if not matching_indexes:
        return MadeStrength.BOARD_ONLY
    if pair_rank == max(board_ranks):
        kicker = hole_cards[1 - matching_indexes[0]].rank
        return (
            MadeStrength.TOP_PAIR_STRONG
            if kicker >= 11
            else MadeStrength.TOP_PAIR_WEAK
        )
    higher = sum(rank_value > pair_rank for rank_value in board_ranks)
    return MadeStrength.MEDIUM_PAIR if higher <= 1 else MadeStrength.WEAK_PAIR


def _pair_strength(
    rank: HandRank,
    hole_cards: tuple[Card, Card],
    board: tuple[Card, ...],
) -> MadeStrength:
    return _pair_rank_strength(rank.kickers[0], hole_cards, board)


def _two_pair_strength(
    rank: HandRank,
    hole_cards: tuple[Card, Card],
    board: tuple[Card, ...],
) -> MadeStrength:
    """区分真正的个人两对与“公牌对子 + 自己一对”。

    例如 99 在 T-7-2-2 上，评估器的五张牌类别是“两对”，但策略上仍是
    一个中等口袋对子；不能因此建议把它当强成牌加注。
    """

    board_counts = Counter(card.rank for card in board)
    board_pair_ranks = {
        rank_value for rank_value, count in board_counts.items() if count >= 2
    }
    selected_pair_ranks = set(rank.kickers[:2])
    hole_ranks = (hole_cards[0].rank, hole_cards[1].rank)
    novel_personal_pairs: list[int] = []
    for pair_rank in selected_pair_ranks - board_pair_ranks:
        pocket_pair = hole_ranks[0] == hole_ranks[1] == pair_rank
        board_match = pair_rank in board_counts and pair_rank in hole_ranks
        if pocket_pair or board_match:
            novel_personal_pairs.append(pair_rank)

    if len(novel_personal_pairs) >= 2:
        return MadeStrength.STRONG_MADE
    if len(novel_personal_pairs) == 1:
        return _pair_rank_strength(novel_personal_pairs[0], hole_cards, board)
    if not board_pair_ranks:
        return MadeStrength.STRONG_MADE
    return MadeStrength.BOARD_ONLY


def analyze_postflop_profile(
    hole_cards: Iterable[Card],
    board: Iterable[Card],
    *,
    active_players: int,
) -> PostflopProfile:
    hole = tuple(hole_cards)
    community = tuple(board)
    if len(hole) != 2:
        raise ValueError("翻后画像需要恰好两张底牌")
    if not 3 <= len(community) <= 5:
        raise ValueError("翻后画像需要三至五张公共牌")
    if active_players < 2:
        raise ValueError("翻后至少需要两名仍在牌局中的玩家")

    typed_hole = (hole[0], hole[1])
    rank = evaluate((*typed_hole, *community))
    board_rank = evaluate(community) if len(community) == 5 else None
    board_plays = bool(board_rank is not None and rank == board_rank)
    hole_cards_play = bool(board_rank is None or rank > board_rank)
    personal_made = _personal_made(rank, typed_hole)
    if rank.category == HandCategory.HIGH_CARD:
        strength = MadeStrength.AIR
    elif rank.category == HandCategory.ONE_PAIR:
        strength = _pair_strength(rank, typed_hole, community)
    elif rank.category == HandCategory.TWO_PAIR:
        strength = _two_pair_strength(rank, typed_hole, community)
    elif personal_made:
        strength = MadeStrength.STRONG_MADE
    else:
        strength = MadeStrength.BOARD_ONLY
    if (
        len(community) == 5
        and strength == MadeStrength.BOARD_ONLY
        and hole_cards_play
    ):
        # 公牌提供成牌类别，但底牌踢脚仍决定胜负；不能按纯空气处理。
        strength = MadeStrength.WEAK_SHOWDOWN
    return PostflopProfile(
        rank=rank,
        strength=strength,
        personal_made=personal_made,
        texture=_board_texture(community),
        active_players=active_players,
        board_rank=board_rank,
        board_plays=board_plays,
        hole_cards_play=hole_cards_play,
    )


__all__ = [
    "BoardTexture",
    "BoardWetness",
    "MadeStrength",
    "PostflopProfile",
    "analyze_postflop_profile",
]
