"""离线牌后教练：只使用动作发生前可见的信息评价决策。

本模块不是完整 GTO 求解器。翻前使用带版本号的 6-max 100bb 启发式，
翻后用底池赔率、常见听牌与对未知随机手牌的蒙特卡洛权益给出短反馈。
评价不读取结算结果、对手底牌或尚未发出的公共牌。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any

from poker_trainer.engine.evaluator import HandCategory, evaluate
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import (
    ActionRecord,
    ActionType,
    DecisionSnapshot,
    LegalActions,
    Position,
    Street,
)

from .equity import analyze_common_draws, calculate_pot_odds, estimate_equity


PREFLOP_HEURISTIC_VERSION = "6max-100bb-preflop-v1"
NOT_FULL_GTO_NOTICE = "MVP 为启发式离线教练，不是完整 GTO 求解器。"

# 与统计层保持稳定兼容的 reason_code。
TOP_PAIR_STACKOFF_OPPORTUNITY = "top_pair_stackoff_opportunity"
TOP_PAIR_STACKED_OFF = "top_pair_stacked_off"
DRAW_ODDS_OPPORTUNITY = "draw_odds_opportunity"
DRAW_ODDS_ERROR = "draw_odds_error"

# 自适应训练使用的新 reason_code。
WEAK_TOP_PAIR_OVERCALL = "weak_top_pair_overcall"
SB_COLD_CALL = "sb_cold_call"
THREEBET_TOO_SMALL = "threebet_too_small"
SMALL_PAIR_OVERCONTINUE = "small_pair_overcontinue"
STRONG_DRAW_OVERFOLD = "strong_draw_overfold"


class ActionRating(str, Enum):
    """MVP 的四档动作评级。"""

    RECOMMENDED = "推荐"
    ACCEPTABLE = "可以接受"
    LOOSE_OR_TIGHT = "偏松/偏紧"
    CLEAR_ERROR = "明显错误"

    # 语义别名不增加枚举值，便于服务层使用更短的名字。
    MARGINAL = "偏松/偏紧"
    OBVIOUS_ERROR = "明显错误"

    def __str__(self) -> str:
        """数据库直接接收 dataclass 时仍写入中文值。"""

        return self.value


@dataclass(frozen=True, slots=True)
class PlayerCommitment:
    """动作前单个座位的公开筹码状态；故意不包含底牌。"""

    player_id: str
    position: Position
    stack: int
    street_commitment: int
    total_commitment: int
    folded: bool
    all_in: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "position": self.position.value,
            "stack": self.stack,
            "street_commitment": self.street_commitment,
            "total_commitment": self.total_commitment,
            "folded": self.folded,
            "all_in": self.all_in,
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class CoachContext:
    """一个不可变、无事后信息的动作前复盘快照。"""

    hand_id: str
    small_blind: int
    big_blind: int
    snapshot: DecisionSnapshot
    legal_actions: LegalActions
    commitments: tuple[PlayerCommitment, ...]
    history: tuple[ActionRecord, ...]

    @property
    def legal(self) -> LegalActions:
        """兼容简写。"""

        return self.legal_actions

    def commitment(self, player_id: str) -> PlayerCommitment:
        for row in self.commitments:
            if row.player_id == player_id:
                return row
        raise KeyError(f"未知玩家: {player_id}")

    def as_dict(self) -> dict[str, Any]:
        snapshot = self.snapshot
        legal = self.legal_actions
        return {
            "hand_id": self.hand_id,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "snapshot": {
                "sequence": snapshot.sequence,
                "street": snapshot.street.value,
                "player_id": snapshot.player_id,
                "position": snapshot.position.value,
                "stack": snapshot.stack,
                "street_commitment": snapshot.street_commitment,
                "total_commitment": snapshot.total_commitment,
                "pot_before": snapshot.pot_before,
                "current_bet": snapshot.current_bet,
                "to_call": snapshot.to_call,
                "min_raise_to": snapshot.min_raise_to,
                "active_players": snapshot.active_players,
                "board": [str(card) for card in snapshot.board],
                "hole_cards": [str(card) for card in snapshot.hole_cards],
                "preflop_raise_count": snapshot.preflop_raise_count,
                "last_aggressor_id": snapshot.last_aggressor_id,
            },
            "legal_actions": {
                "player_id": legal.player_id,
                "to_call": legal.to_call,
                "call_amount": legal.call_amount,
                "pot_before": legal.pot_before,
                "min_bet_to": legal.min_bet_to,
                "min_raise_to": legal.min_raise_to,
                "max_to": legal.max_to,
                "can_fold": legal.can_fold,
                "can_check": legal.can_check,
                "can_call": legal.can_call,
                "can_bet": legal.can_bet,
                "can_raise": legal.can_raise,
                "can_all_in": legal.can_all_in,
                "raise_reopened": legal.raise_reopened,
                "action_types": [action.value for action in legal.action_types],
            },
            "commitments": [row.as_dict() for row in self.commitments],
            "history": [record.as_dict() for record in self.history],
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class DecisionReview:
    """可直接持久化为 JSON 的单次动作复盘。"""

    sequence: int
    player_id: str
    street: Street
    action: ActionType
    rating: ActionRating
    reason: str
    reason_codes: tuple[str, ...]
    pot_odds: float | None
    equity: float | None
    outs: int | None
    hit_probability: float | None
    heuristic_version: str
    disclaimer: str = NOT_FULL_GTO_NOTICE

    @property
    def grade(self) -> ActionRating:
        """兼容把评级字段称为 ``grade`` 的展示层。"""

        return self.rating

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "player_id": self.player_id,
            "street": self.street.value,
            "action": self.action.value,
            "rating": self.rating.value,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "pot_odds": self.pot_odds,
            "equity": self.equity,
            "outs": self.outs,
            "hit_probability": self.hit_probability,
            "heuristic_version": self.heuristic_version,
            "disclaimer": self.disclaimer,
        }

    to_dict = as_dict


def capture_context(hand: HoldemHand, player_id: str | None = None) -> CoachContext:
    """在玩家行动前复制教练需要的公开状态。

    返回值没有对运行中 ``hand`` 的引用，因此之后发出的牌、对手底牌和
    ``HandResult`` 都不会进入或改变这个上下文。
    """

    snapshot = hand.decision_snapshot(player_id)
    legal = hand.legal_actions(snapshot.player_id)
    position_order = {position: index for index, position in enumerate(Position)}
    commitments = tuple(
        PlayerCommitment(
            player_id=player.player_id,
            position=player.position,
            stack=player.stack,
            street_commitment=player.street_commitment,
            total_commitment=player.total_commitment,
            folded=player.folded,
            all_in=player.all_in,
        )
        for player in sorted(
            hand.players.values(), key=lambda row: position_order[row.position]
        )
    )
    return CoachContext(
        hand_id=hand.hand_id,
        small_blind=hand.small_blind,
        big_blind=hand.big_blind,
        snapshot=snapshot,
        legal_actions=legal,
        commitments=commitments,
        history=tuple(hand.history),
    )


def _validate_record(context: CoachContext, record: ActionRecord) -> None:
    snapshot = context.snapshot
    if (
        record.sequence != snapshot.sequence
        or record.player_id != snapshot.player_id
        or record.street != snapshot.street
        or record.pot_before != snapshot.pot_before
        or record.to_call_before != snapshot.to_call
        or record.current_bet_before != snapshot.current_bet
    ):
        raise ValueError("ActionRecord 与动作前 CoachContext 不匹配")


def _is_aggressive(record: ActionRecord) -> bool:
    return record.action in {ActionType.BET, ActionType.RAISE} or (
        record.action == ActionType.ALL_IN
        and record.current_bet_after > record.current_bet_before
    )


def _is_call(record: ActionRecord) -> bool:
    return record.action == ActionType.CALL or (
        record.action == ActionType.ALL_IN
        and record.current_bet_after <= record.current_bet_before
    )


def _worse(first: ActionRating, second: ActionRating) -> ActionRating:
    severity = {
        ActionRating.RECOMMENDED: 0,
        ActionRating.ACCEPTABLE: 1,
        ActionRating.LOOSE_OR_TIGHT: 2,
        ActionRating.CLEAR_ERROR: 3,
    }
    return first if severity[first] >= severity[second] else second


def _equity_seed(context: CoachContext) -> int:
    snapshot = context.snapshot
    public_key = "|".join(
        (
            context.hand_id,
            str(snapshot.sequence),
            snapshot.player_id,
            snapshot.street.value,
            *(str(card) for card in snapshot.hole_cards),
            *(str(card) for card in snapshot.board),
        )
    )
    return int.from_bytes(hashlib.sha256(public_key.encode("utf-8")).digest()[:8], "big")


def _contestable_pot_odds(context: CoachContext) -> float:
    """按英雄能够赢取的投入上限计算底池赔率，排除无权争夺的边池。"""

    call_amount = context.legal_actions.call_amount
    if call_amount <= 0:
        return 0.0
    hero_cap = context.snapshot.total_commitment + call_amount
    contestable_before = sum(
        min(row.total_commitment, hero_cap) for row in context.commitments
    )
    return calculate_pot_odds(contestable_before, call_amount)


def _preflop_bucket(snapshot: DecisionSnapshot) -> str:
    first, second = snapshot.hole_cards
    high, low = sorted((first.rank, second.rank), reverse=True)
    pair = high == low
    suited = first.suit == second.suit
    if pair and high >= 11 or {high, low} == {14, 13}:
        return "premium"
    if pair and high >= 8 or high == 14 and low >= 11 or suited and high >= 13 and low >= 11:
        return "strong"
    if pair or high == 14 and suited or high >= 11 and low >= 10 or suited and high - low <= 2 and low >= 6:
        return "playable"
    return "weak"


def _threebet_reference_to(context: CoachContext) -> int | None:
    snapshot = context.snapshot
    if snapshot.street != Street.PREFLOP or snapshot.preflop_raise_count != 1:
        return None
    open_to = snapshot.current_bet
    if open_to <= context.big_blind:
        return None
    multiplier = 4 if snapshot.position in {Position.SB, Position.BB} else 3
    callers = sum(
        1
        for row in context.history
        if row.street == Street.PREFLOP
        and not row.forced
        and row.action == ActionType.CALL
        and row.current_bet_before == open_to
    )
    return open_to * (multiplier + callers)


def _review_preflop(context: CoachContext, record: ActionRecord) -> DecisionReview:
    snapshot = context.snapshot
    legal = context.legal_actions
    bucket = _preflop_bucket(snapshot)
    pot_odds = _contestable_pot_odds(context)
    codes: list[str] = []

    if record.action == ActionType.FOLD:
        if legal.can_check:
            rating, reason = ActionRating.CLEAR_ERROR, "可以免费看牌，不应弃牌。"
        elif bucket == "premium":
            rating, reason = ActionRating.CLEAR_ERROR, "强起手牌弃得过紧。"
        elif bucket == "strong":
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "这类强牌通常可继续。"
        else:
            rating, reason = ActionRating.RECOMMENDED, "牌力与位置不足，弃牌合理。"
    elif record.action == ActionType.CHECK:
        rating, reason = ActionRating.RECOMMENDED, "大盲可免费看翻牌。"
    elif _is_call(record):
        if snapshot.position == Position.SB and snapshot.preflop_raise_count >= 1:
            codes.append(SB_COLD_CALL)
            rating = ActionRating.LOOSE_OR_TIGHT
            reason = "小盲冷跟位置差，优先加注或弃牌。"
        elif bucket in {"premium", "strong", "playable"}:
            rating, reason = ActionRating.ACCEPTABLE, "跟注可接受，但仍要结合位置与范围。"
        elif snapshot.preflop_raise_count == 0:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "弱牌 limp 容易被动，范围偏松。"
        else:
            rating, reason = ActionRating.CLEAR_ERROR, "弱牌面对加注继续过松。"
    elif _is_aggressive(record):
        rating = (
            ActionRating.RECOMMENDED
            if bucket in {"premium", "strong"}
            else ActionRating.ACCEPTABLE
            if bucket == "playable"
            else ActionRating.LOOSE_OR_TIGHT
        )
        reason = "主动加注与当前起手牌强度基本匹配。"
        reference = _threebet_reference_to(context)
        if (
            reference is not None
            and context.legal_actions.max_to >= reference * 0.85
            and record.bet_to < reference * 0.85
        ):
            codes.append(THREEBET_TOO_SMALL)
            ratio = record.bet_to / reference
            rating = _worse(
                rating,
                ActionRating.CLEAR_ERROR if ratio < 0.65 else ActionRating.LOOSE_OR_TIGHT,
            )
            reason = f"3bet 到 {record.bet_to} 偏小，参考约 {reference}。"
    else:
        rating, reason = ActionRating.ACCEPTABLE, "动作可接受。"

    return DecisionReview(
        sequence=record.sequence,
        player_id=record.player_id,
        street=record.street,
        action=record.action,
        rating=rating,
        reason=reason,
        reason_codes=tuple(codes),
        pot_odds=pot_odds,
        equity=None,
        outs=None,
        hit_probability=None,
        heuristic_version=PREFLOP_HEURISTIC_VERSION,
    )


def _top_pair_state(snapshot: DecisionSnapshot) -> tuple[bool, bool]:
    if len(snapshot.board) < 3:
        return False, False
    rank = evaluate((*snapshot.hole_cards, *snapshot.board))
    top_board_rank = max(card.rank for card in snapshot.board)
    matching = [card for card in snapshot.hole_cards if card.rank == top_board_rank]
    if rank.category != HandCategory.ONE_PAIR or len(matching) != 1:
        return False, False
    kicker = next(card.rank for card in snapshot.hole_cards if card != matching[0])
    return True, kicker <= 10


def _small_pair_missed(snapshot: DecisionSnapshot) -> bool:
    first, second = snapshot.hole_cards
    return bool(
        first.rank == second.rank <= 9
        and all(card.rank != first.rank for card in snapshot.board)
        and any(card.rank > first.rank for card in snapshot.board)
    )


def _review_postflop(
    context: CoachContext, record: ActionRecord, *, trials: int
) -> DecisionReview:
    snapshot = context.snapshot
    legal = context.legal_actions
    pot_odds = _contestable_pot_odds(context)
    opponents = max(1, snapshot.active_players - 1)
    equity = estimate_equity(
        snapshot.hole_cards,
        snapshot.board,
        opponents=opponents,
        trials=trials,
        seed=_equity_seed(context),
    )
    draw = analyze_common_draws(snapshot.hole_cards, snapshot.board)
    facing_bet = legal.to_call > 0
    strong_draw = draw.outs >= 8
    codes: list[str] = []

    if facing_bet and record.action == ActionType.FOLD:
        if equity >= pot_odds + 0.12:
            rating, reason = ActionRating.CLEAR_ERROR, "权益明显高于底池赔率，弃牌过紧。"
        elif equity >= pot_odds + 0.04:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "权益高于底池赔率，弃牌略紧。"
        else:
            rating, reason = ActionRating.RECOMMENDED, "权益不足以覆盖底池赔率，弃牌合理。"
    elif facing_bet and _is_call(record):
        if equity + 0.10 < pot_odds:
            rating, reason = ActionRating.CLEAR_ERROR, "权益明显低于底池赔率，跟注为负期望。"
        elif equity + 0.03 < pot_odds:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "权益略低于底池赔率，跟注偏松。"
        elif equity >= pot_odds + 0.10:
            rating, reason = ActionRating.RECOMMENDED, "权益充分覆盖底池赔率，继续合理。"
        else:
            rating, reason = ActionRating.ACCEPTABLE, "权益接近或高于底池赔率，跟注可接受。"
    elif facing_bet and _is_aggressive(record):
        if equity >= max(0.50, pot_odds + 0.10) or strong_draw and equity >= pot_odds:
            rating, reason = ActionRating.RECOMMENDED, "价值或强听牌权益足够，主动继续合理。"
        elif equity + 0.03 >= pot_odds:
            rating, reason = ActionRating.ACCEPTABLE, "加注可接受，但需控制尺度。"
        elif equity + 0.12 < pot_odds:
            rating, reason = ActionRating.CLEAR_ERROR, "权益不足，激进投入过多。"
        else:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "权益偏薄，激进动作略松。"
    elif record.action == ActionType.FOLD:
        rating, reason = ActionRating.CLEAR_ERROR, "可以过牌保留权益，无需弃牌。"
    elif record.action == ActionType.CHECK:
        if equity >= 0.75:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "牌力较强，过牌可能损失价值。"
        else:
            rating, reason = ActionRating.ACCEPTABLE, "过牌控制底池可以接受。"
    elif _is_aggressive(record):
        if equity >= 0.58 or strong_draw:
            rating, reason = ActionRating.RECOMMENDED, "牌力或听牌权益支持主动下注。"
        elif equity >= 0.42:
            rating, reason = ActionRating.ACCEPTABLE, "下注可接受，但范围不宜过宽。"
        else:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "摊牌权益偏低，下注略松。"
    else:
        rating, reason = ActionRating.ACCEPTABLE, "动作可接受。"

    if facing_bet and draw.outs > 0:
        codes.append(DRAW_ODDS_OPPORTUNITY)
        draw_error = (
            record.action == ActionType.FOLD and equity >= pot_odds + 0.03
        ) or (
            (_is_call(record) or _is_aggressive(record))
            and equity + 0.03 < pot_odds
        )
        if draw_error:
            codes.append(DRAW_ODDS_ERROR)

    if strong_draw and facing_bet and record.action == ActionType.FOLD and equity >= pot_odds:
        codes.append(STRONG_DRAW_OVERFOLD)
        rating = _worse(
            rating,
            ActionRating.CLEAR_ERROR
            if equity >= pot_odds + 0.10
            else ActionRating.LOOSE_OR_TIGHT,
        )
        reason = "强听牌权益覆盖底池赔率，弃牌过紧。"
    elif strong_draw and facing_bet and record.action != ActionType.FOLD and equity >= pot_odds:
        if rating in {ActionRating.ACCEPTABLE, ActionRating.RECOMMENDED}:
            rating = ActionRating.RECOMMENDED
            reason = "强听牌权益覆盖底池赔率，继续合理。"

    top_pair, weak_top_pair = _top_pair_state(snapshot)
    stackoff_decision = record.is_all_in or legal.call_amount >= snapshot.stack
    if top_pair and stackoff_decision:
        codes.append(TOP_PAIR_STACKOFF_OPPORTUNITY)
        if record.action != ActionType.FOLD and record.is_all_in:
            codes.append(TOP_PAIR_STACKED_OFF)
    if (
        weak_top_pair
        and facing_bet
        and _is_call(record)
        and (record.is_all_in or legal.call_amount >= legal.pot_before * 0.5)
    ):
        codes.append(WEAK_TOP_PAIR_OVERCALL)
        deep_commit = snapshot.stack >= 40 * context.big_blind and record.paid >= 25 * context.big_blind
        rating = _worse(
            rating,
            ActionRating.CLEAR_ERROR if deep_commit else ActionRating.LOOSE_OR_TIGHT,
        )
        reason = "弱顶对面对大额投入仍跟到底，过度跟注。"

    if (
        _small_pair_missed(snapshot)
        and facing_bet
        and record.action != ActionType.FOLD
        and (snapshot.street in {Street.TURN, Street.RIVER} or equity < pot_odds + 0.08)
    ):
        codes.append(SMALL_PAIR_OVERCONTINUE)
        rating = _worse(
            rating,
            ActionRating.CLEAR_ERROR if record.is_all_in else ActionRating.LOOSE_OR_TIGHT,
        )
        reason = "小口袋对子未中三条，面对压力继续过多。"

    return DecisionReview(
        sequence=record.sequence,
        player_id=record.player_id,
        street=record.street,
        action=record.action,
        rating=rating,
        reason=reason,
        reason_codes=tuple(dict.fromkeys(codes)),
        pot_odds=pot_odds,
        equity=equity,
        outs=draw.outs,
        hit_probability=(draw.hit_by_river if snapshot.street == Street.FLOP else draw.hit_next),
        heuristic_version=PREFLOP_HEURISTIC_VERSION,
    )


def review_decision(
    context: CoachContext,
    record: ActionRecord,
    trials: int = 2_000,
) -> DecisionReview:
    """评价一个已发生动作，但只读取其动作前 ``context``。"""

    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("模拟次数必须是正整数")
    _validate_record(context, record)
    if context.snapshot.street == Street.PREFLOP:
        return _review_preflop(context, record)
    if context.snapshot.street not in {Street.FLOP, Street.TURN, Street.RIVER}:
        raise ValueError("只能复盘翻前至河牌的玩家决策")
    return _review_postflop(context, record, trials=trials)


__all__ = [
    "ActionRating",
    "CoachContext",
    "DecisionReview",
    "DRAW_ODDS_ERROR",
    "DRAW_ODDS_OPPORTUNITY",
    "NOT_FULL_GTO_NOTICE",
    "PREFLOP_HEURISTIC_VERSION",
    "PlayerCommitment",
    "SB_COLD_CALL",
    "SMALL_PAIR_OVERCONTINUE",
    "STRONG_DRAW_OVERFOLD",
    "THREEBET_TOO_SMALL",
    "TOP_PAIR_STACKED_OFF",
    "TOP_PAIR_STACKOFF_OPPORTUNITY",
    "WEAK_TOP_PAIR_OVERCALL",
    "capture_context",
    "review_decision",
]

