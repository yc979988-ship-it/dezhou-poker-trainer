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
from .preflop_strategy import (
    ActionRole,
    PreflopPlan,
    PreflopSituation,
    PreflopSpot,
    StrategicAction,
    build_preflop_plan,
)


PREFLOP_HEURISTIC_VERSION = "6max-100bb-preflop-v4"
POSTFLOP_HEURISTIC_VERSION = "postflop-random-equity-v2"
NOT_FULL_GTO_NOTICE = "MVP 为启发式离线教练，不是完整 GTO 求解器。"
RANDOM_EQUITY_BASIS = "random_unknown_hands"

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

# 结构性尺寸与河牌过度加注；用于复盘解释，不假装求解对手真实范围。
PREFLOP_RAISE_TOO_SMALL = "preflop_raise_too_small"
PREFLOP_RAISE_TOO_LARGE = "preflop_raise_too_large"
THREEBET_TOO_LARGE = "threebet_too_large"
RIVER_SHOWDOWN_VALUE_OVERPLAY = "river_showdown_value_overplay"

# 自适应画像分母：正确与错误决策都必须记录机会，不能只统计犯错次数。
WEAK_TOP_PAIR_DECISION = "weak_top_pair_decision"
SB_COLD_CALL_OPPORTUNITY = "sb_cold_call_opportunity"
THREEBET_SIZING_OPPORTUNITY = "threebet_sizing_opportunity"
SMALL_PAIR_POSTFLOP_DECISION = "small_pair_postflop_decision"
STRONG_DRAW_DECISION = "strong_draw_decision"


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
    recommended_action: str | None = None
    detail_lines: tuple[str, ...] = ()
    draw_names: tuple[str, ...] = ()
    equity_basis: str | None = None

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
            "recommended_action": self.recommended_action,
            "detail_lines": list(self.detail_lines),
            "draw_names": list(self.draw_names),
            "equity_basis": self.equity_basis,
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


def _effective_max_to(context: CoachContext) -> int:
    """返回仍在牌局中的对手实际能够匹配的最大本街 bet-to。"""

    opponents = (
        row.street_commitment + row.stack
        for row in context.commitments
        if row.player_id != context.snapshot.player_id and not row.folded
    )
    return min(context.legal_actions.max_to, max(opponents, default=context.legal_actions.max_to))


def _effective_bet_to(context: CoachContext, record: ActionRecord) -> int:
    """忽略最终会退回的超额全下，只评价对手能够跟注的有效金额。"""

    return min(record.bet_to, _effective_max_to(context))


def _preflop_limp_count(context: CoachContext) -> int:
    return sum(
        1
        for row in context.history
        if row.street == Street.PREFLOP
        and not row.forced
        and row.action == ActionType.CALL
        and row.current_bet_before == context.big_blind
        and row.bet_to == context.big_blind
    )


def _opening_raise_reference_to(context: CoachContext) -> int | None:
    snapshot = context.snapshot
    if snapshot.street != Street.PREFLOP or snapshot.preflop_raise_count != 0:
        return None
    limpers = _preflop_limp_count(context)
    # 无 limper 以 3bb 为中点；隔离时使用 4bb + 每名 limper 1bb。
    return context.big_blind * (3 if limpers == 0 else 4 + limpers)


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
    return open_to * (multiplier + callers) + _preflop_limp_count(context) * context.big_blind


def _preflop_raise_reference(context: CoachContext) -> tuple[str, int] | None:
    opening = _opening_raise_reference_to(context)
    if opening is not None:
        label = "开池" if _preflop_limp_count(context) == 0 else "隔离加注"
        return label, opening
    threebet = _threebet_reference_to(context)
    if threebet is not None:
        return "3bet", threebet
    return None


def _is_initial_sb_raise_response(context: CoachContext) -> bool:
    """SB 首次面对加注才是 cold-call 机会；此前 limp 后再跟不算。"""

    snapshot = context.snapshot
    prior_voluntary_action = any(
        row.street == Street.PREFLOP
        and row.player_id == snapshot.player_id
        and not row.forced
        for row in context.history
    )
    return bool(
        snapshot.position == Position.SB
        and snapshot.preflop_raise_count >= 1
        and not prior_voluntary_action
    )


def _preflop_opener_position(context: CoachContext) -> Position | None:
    """返回第一名非强制翻前进攻者的位置。"""

    return next(
        (
            row.position
            for row in context.history
            if row.street == Street.PREFLOP
            and not row.forced
            and _is_aggressive(row)
        ),
        None,
    )


def _preflop_callers_after_raise(context: CoachContext) -> int:
    """计算第一次加注后的跟注者，用于区分普通 3bet 与 squeeze。"""

    opener = next(
        (
            row
            for row in context.history
            if row.street == Street.PREFLOP
            and not row.forced
            and _is_aggressive(row)
        ),
        None,
    )
    if opener is None:
        return 0
    return sum(
        1
        for row in context.history
        if row.street == Street.PREFLOP
        and row.sequence > opener.sequence
        and not row.forced
        and _is_call(row)
        and row.current_bet_before == opener.bet_to
    )


def _preflop_situation(context: CoachContext) -> PreflopSituation:
    snapshot = context.snapshot
    return PreflopSituation(
        hole_cards=(snapshot.hole_cards[0], snapshot.hole_cards[1]),
        position=snapshot.position,
        raise_count=snapshot.preflop_raise_count,
        limpers=_preflop_limp_count(context),
        callers_after_raise=_preflop_callers_after_raise(context),
        opener_position=_preflop_opener_position(context),
        effective_stack_bb=_effective_max_to(context) / context.big_blind,
        can_check=context.legal_actions.can_check,
    )


def _strategic_action(record: ActionRecord) -> StrategicAction:
    if record.action == ActionType.FOLD:
        return StrategicAction.FOLD
    if record.action == ActionType.CHECK:
        return StrategicAction.CHECK
    if _is_call(record):
        return StrategicAction.CALL
    if _is_aggressive(record):
        return StrategicAction.RAISE
    raise ValueError(f"无法评价翻前动作: {record.action.value}")


def _role_rating(role: ActionRole) -> ActionRating:
    return {
        ActionRole.PRIMARY: ActionRating.RECOMMENDED,
        ActionRole.ACCEPTABLE: ActionRating.ACCEPTABLE,
        ActionRole.MIXED: ActionRating.ACCEPTABLE,
        ActionRole.DISCOURAGED: ActionRating.LOOSE_OR_TIGHT,
        ActionRole.ERROR: ActionRating.CLEAR_ERROR,
    }[role]


def _preflop_action_label(plan: PreflopPlan, action: StrategicAction) -> str:
    if action == StrategicAction.CALL and plan.spot == PreflopSpot.LIMPED:
        return "补齐" if plan.position == Position.SB else "跟注"
    return action.label_zh


def _strategic_action_is_legal(
    context: CoachContext,
    action: StrategicAction,
) -> bool:
    legal = context.legal_actions
    if action == StrategicAction.FOLD:
        return legal.can_fold
    if action == StrategicAction.CHECK:
        return legal.can_check
    if action == StrategicAction.CALL:
        return legal.can_call or (
            legal.can_all_in and legal.max_to <= context.snapshot.current_bet
        )
    return legal.can_bet or legal.can_raise or (
        legal.can_all_in and legal.max_to > context.snapshot.current_bet
    )


def _recommended_preflop_actions(
    context: CoachContext,
    plan: PreflopPlan,
    actual_action: StrategicAction,
    actual_role: ActionRole,
) -> tuple[StrategicAction, ...]:
    """选择可执行建议；正确动作只显示自身，错误动作显示最佳替代。"""

    if actual_role in {ActionRole.PRIMARY, ActionRole.ACCEPTABLE, ActionRole.MIXED}:
        preferred = (actual_action,)
    else:
        preferred = tuple(
            option.action
            for option in plan.options
            if option.role
            in {ActionRole.PRIMARY, ActionRole.ACCEPTABLE, ActionRole.MIXED}
        )
    legal_preferred = tuple(
        action
        for action in preferred
        if _strategic_action_is_legal(context, action)
    )
    if legal_preferred:
        return legal_preferred

    role_order = {
        ActionRole.PRIMARY: 0,
        ActionRole.ACCEPTABLE: 1,
        ActionRole.MIXED: 1,
        ActionRole.DISCOURAGED: 2,
        ActionRole.ERROR: 3,
    }
    legal_options = sorted(
        (
            option
            for option in plan.options
            if _strategic_action_is_legal(context, option.action)
        ),
        key=lambda option: role_order[option.role],
    )
    return (legal_options[0].action,) if legal_options else ()


def _preflop_recommendation_text(
    context: CoachContext,
    plan: PreflopPlan,
    actual_action: StrategicAction,
    actual_role: ActionRole,
) -> str | None:
    if (
        plan.spot == PreflopSpot.LIMPED
        and plan.position == Position.SB
        and plan.primary.action == StrategicAction.FOLD
        and actual_action == StrategicAction.CALL
        and actual_role == ActionRole.DISCOURAGED
    ):
        return "弃牌；桌面极被动时可补齐"
    actions = _recommended_preflop_actions(
        context,
        plan,
        actual_action,
        actual_role,
    )
    return "或".join(_preflop_action_label(plan, action) for action in actions) or None


def _preflop_detail_lines(
    plan: PreflopPlan,
    actual_action: StrategicAction,
) -> list[str]:
    details = [
        f"牌型 {plan.hand_key}｜位置 {plan.position.value}｜场景 {plan.spot.label_zh}。",
        (
            f"默认：{_preflop_action_label(plan, plan.primary.action)}。"
            f"{plan.primary.explanation}"
        ),
    ]
    alternatives = tuple(
        option
        for option in plan.options
        if option.role in {ActionRole.ACCEPTABLE, ActionRole.MIXED}
    )
    for option in alternatives:
        prefix = "混合" if option.role == ActionRole.MIXED else "可接受"
        details.append(
            f"{prefix}：{_preflop_action_label(plan, option.action)}。"
            f"{option.explanation}"
        )
    selected = plan.option_for(actual_action)
    if selected.role not in {ActionRole.PRIMARY, ActionRole.ACCEPTABLE, ActionRole.MIXED}:
        details.append(
            f"本次动作：{_preflop_action_label(plan, actual_action)}。"
            f"{selected.explanation}"
        )
    if selected.conditions:
        details.append(f"适用条件：{'；'.join(selected.conditions)}。")
    return details


def _review_preflop(context: CoachContext, record: ActionRecord) -> DecisionReview:
    snapshot = context.snapshot
    plan = build_preflop_plan(_preflop_situation(context))
    actual_action = _strategic_action(record)
    selected = plan.option_for(actual_action)
    rating = _role_rating(selected.role)
    reason = selected.explanation
    codes: list[str] = []
    details = _preflop_detail_lines(plan, actual_action)
    recommended_action = _preflop_recommendation_text(
        context,
        plan,
        actual_action,
        selected.role,
    )

    sb_cold_call_spot = _is_initial_sb_raise_response(context)
    if sb_cold_call_spot:
        codes.append(SB_COLD_CALL_OPPORTUNITY)
        if actual_action == StrategicAction.CALL:
            codes.append(SB_COLD_CALL)
            if selected.role in {ActionRole.DISCOURAGED, ActionRole.ERROR}:
                reason = (
                    "小盲冷跟不能只看即时赔率，还要计入位置差和 BB 挤压风险。"
                )

    aggression_allowed = selected.role in {
        ActionRole.PRIMARY,
        ActionRole.ACCEPTABLE,
        ActionRole.MIXED,
    }
    if actual_action == StrategicAction.RAISE and not aggression_allowed:
        limpers = _preflop_limp_count(context)
        if selected.role == ActionRole.ERROR and snapshot.preflop_raise_count >= 1:
            reason = (
                "弱牌面对既有加注再次加注过松；应先收紧范围，"
                "而不是调整加注尺度。"
            )
        elif selected.role == ActionRole.ERROR and limpers:
            position_note = (
                "位置较差、" if snapshot.position in {Position.SB, Position.BB} else ""
            )
            reason = (
                f"弱牌隔离 {limpers} 名 limp 玩家过松："
                f"{position_note}又很难让偏松对手弃牌。"
            )
        details.append(
            "本次先评价具体牌型、位置与前序动作；主动加注本身不合适，"
            "因此不提供加注尺度建议。"
        )

    if actual_action == StrategicAction.RAISE and aggression_allowed:
        reference_data = _preflop_raise_reference(context)
        if reference_data is not None:
            label, reference = reference_data
            effective_max = _effective_max_to(context)
            effective_to = _effective_bet_to(context, record)
            can_use_normal_size = effective_max >= reference * 0.85
            if label == "3bet" and can_use_normal_size:
                codes.append(THREEBET_SIZING_OPPORTUNITY)
            if can_use_normal_size:
                lower_bound = (
                    0.85
                    if label == "3bet"
                    else 0.65
                    if label == "开池"
                    else 0.75
                )
                upper_bound = 1.75 if label != "隔离加注" else 1.60
                ratio = effective_to / reference
                details.append(
                    f"尺度：{label}参考中点约 {reference}，实际有效到 {effective_to}。"
                )
                recommended_action = f"加注到约 {reference}"
                if ratio < lower_bound:
                    codes.append(PREFLOP_RAISE_TOO_SMALL)
                    if label == "3bet":
                        codes.append(THREEBET_TOO_SMALL)
                    rating = _worse(
                        rating,
                        ActionRating.CLEAR_ERROR
                        if ratio < 0.60
                        else ActionRating.LOOSE_OR_TIGHT,
                    )
                    reason = (
                        f"{label}到 {effective_to} 偏小，参考约 {reference}；"
                        "过小会给多人便宜跟注。"
                    )
                elif ratio > upper_bound:
                    codes.append(PREFLOP_RAISE_TOO_LARGE)
                    if label == "3bet":
                        codes.append(THREEBET_TOO_LARGE)
                    rating = _worse(
                        rating,
                        ActionRating.CLEAR_ERROR
                        if ratio >= 2.50
                        else ActionRating.LOOSE_OR_TIGHT,
                    )
                    reason = (
                        f"{label}到 {effective_to} 偏大，参考约 {reference}；"
                        "投入与可赢底池不成比例。"
                    )
            else:
                details.append("有效后手不足以使用常规尺度，本次按短码全下处理。")

    return DecisionReview(
        sequence=record.sequence,
        player_id=record.player_id,
        street=record.street,
        action=record.action,
        rating=rating,
        reason=reason,
        reason_codes=tuple(dict.fromkeys(codes)),
        pot_odds=_contestable_pot_odds(context),
        equity=None,
        outs=None,
        hit_probability=None,
        heuristic_version=PREFLOP_HEURISTIC_VERSION,
        recommended_action=recommended_action,
        detail_lines=tuple(details),
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


def _personal_made_hand(snapshot: DecisionSnapshot) -> tuple[bool, str, HandCategory]:
    """区分英雄自己的成牌与仅由公共牌构成的牌型。"""

    rank = evaluate((*snapshot.hole_cards, *snapshot.board))
    hole_cards = set(snapshot.hole_cards)
    if rank.category == HandCategory.HIGH_CARD:
        contributes = False
    elif rank.category == HandCategory.ONE_PAIR:
        contributes = any(card.rank == rank.kickers[0] for card in hole_cards)
    elif rank.category == HandCategory.TWO_PAIR:
        contributes = any(card.rank in rank.kickers[:2] for card in hole_cards)
    elif rank.category == HandCategory.THREE_OF_A_KIND:
        contributes = any(card.rank == rank.kickers[0] for card in hole_cards)
    else:
        contributes = any(card in rank.best_five for card in hole_cards)
    return contributes, rank.name_zh, rank.category


def _last_aggressive_record(context: CoachContext) -> ActionRecord | None:
    return next(
        (
            row
            for row in reversed(context.history)
            if row.street == context.snapshot.street and _is_aggressive(row)
        ),
        None,
    )


def _river_one_pair_overraise(context: CoachContext, record: ActionRecord) -> bool:
    """识别有摊牌价值的一对牌面对非阻断下注再次加注。"""

    snapshot = context.snapshot
    if (
        snapshot.street != Street.RIVER
        or context.legal_actions.to_call <= 0
        or not _is_aggressive(record)
    ):
        return False
    personal_made, _name, category = _personal_made_hand(snapshot)
    if not personal_made or category != HandCategory.ONE_PAIR:
        return False
    bettor = _last_aggressive_record(context)
    if bettor is None or bettor.pot_before <= 0:
        return True
    # 面对 <=15% 底池的阻断下注，薄价值加注可能合理，不一刀切。
    return bettor.paid / bettor.pot_before > 0.15


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
    personal_made, made_hand_name, _made_category = _personal_made_hand(snapshot)
    made_plus_draw = personal_made and draw.outs > 0
    hit_probability = draw.hit_by_river if snapshot.street == Street.FLOP else draw.hit_next
    codes: list[str] = []
    details: list[str] = []
    recommended_action: str | None = None

    if facing_bet:
        details.append(f"跟注盈亏平衡点为 {pot_odds:.1%}。")
    details.append(
        f"随机未知手牌基准权益约 {equity:.1%}；它不是对手当前下注范围的真实权益。"
    )
    if draw.outs > 0:
        draw_window = "到河牌" if snapshot.street == Street.FLOP else "下一张"
        details.append(
            f"{draw.name}共 {draw.outs} 个未折损 outs，{draw_window}原始命中率约 {hit_probability:.1%}。"
        )
        if made_plus_draw:
            details.append(
                f"这是{made_hand_name}+听牌；听牌未命中时，现有{made_hand_name}仍可能赢。"
            )
        else:
            details.append("这是纯听牌；未命中时通常不能依赖现有成牌摊牌。")

    if strong_draw and facing_bet:
        codes.append(STRONG_DRAW_DECISION)

    if facing_bet and record.action == ActionType.FOLD:
        if equity >= pot_odds + 0.12:
            rating = ActionRating.CLEAR_ERROR
            reason = "随机范围基准权益明显高于底池赔率，弃牌过紧。"
            recommended_action = "跟注或加注"
        elif equity >= pot_odds + 0.04:
            rating = ActionRating.LOOSE_OR_TIGHT
            reason = "随机范围基准权益高于底池赔率，弃牌略紧。"
            recommended_action = "跟注"
        else:
            rating, reason = ActionRating.RECOMMENDED, "权益不足以覆盖底池赔率，弃牌合理。"
            recommended_action = "弃牌"
    elif facing_bet and _is_call(record):
        if equity + 0.10 < pot_odds:
            rating = ActionRating.CLEAR_ERROR
            reason = "随机范围基准权益明显低于底池赔率，跟注为负期望。"
            recommended_action = "弃牌"
        elif equity + 0.03 < pot_odds:
            rating = ActionRating.LOOSE_OR_TIGHT
            reason = "随机范围基准权益略低于底池赔率，跟注偏松。"
            recommended_action = "弃牌"
        elif equity >= pot_odds + 0.10:
            rating = ActionRating.RECOMMENDED
            reason = "随机范围基准权益充分覆盖底池赔率，继续合理。"
            recommended_action = "跟注"
        else:
            rating, reason = ActionRating.ACCEPTABLE, "权益接近或高于底池赔率，跟注可接受。"
            recommended_action = "跟注"
    elif facing_bet and _is_aggressive(record):
        if equity >= max(0.50, pot_odds + 0.10) or strong_draw and equity >= pot_odds:
            rating, reason = ActionRating.RECOMMENDED, "价值或强听牌权益足够，主动继续合理。"
            recommended_action = "加注"
        elif equity + 0.03 >= pot_odds:
            rating, reason = ActionRating.ACCEPTABLE, "加注可接受，但需控制尺度。"
            recommended_action = "跟注或加注"
        elif equity + 0.12 < pot_odds:
            rating, reason = ActionRating.CLEAR_ERROR, "权益不足，激进投入过多。"
            recommended_action = "弃牌"
        else:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "权益偏薄，激进动作略松。"
            recommended_action = "跟注或弃牌"
    elif record.action == ActionType.FOLD:
        rating, reason = ActionRating.CLEAR_ERROR, "可以过牌保留权益，无需弃牌。"
        recommended_action = "过牌"
    elif record.action == ActionType.CHECK:
        if equity >= 0.75:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "牌力较强，过牌可能损失价值。"
            recommended_action = "下注"
        else:
            rating, reason = ActionRating.ACCEPTABLE, "过牌控制底池可以接受。"
            recommended_action = "过牌"
    elif _is_aggressive(record):
        if equity >= 0.58 or strong_draw:
            rating, reason = ActionRating.RECOMMENDED, "牌力或听牌权益支持主动下注。"
            recommended_action = "下注"
        elif equity >= 0.42:
            rating, reason = ActionRating.ACCEPTABLE, "下注可接受，但范围不宜过宽。"
            recommended_action = "下注或过牌"
        else:
            rating, reason = ActionRating.LOOSE_OR_TIGHT, "摊牌权益偏低，下注略松。"
            recommended_action = "过牌"
    else:
        rating, reason = ActionRating.ACCEPTABLE, "动作可接受。"

    if facing_bet and draw.outs > 0:
        codes.append(DRAW_ODDS_OPPORTUNITY)
        # 成牌+听牌的未命中分支仍可能摊牌获胜，且高度依赖下注范围；
        # 不把随机手牌基准直接记成“听牌赔率错误”。
        draw_error = not made_plus_draw and (
            (record.action == ActionType.FOLD and equity >= pot_odds + 0.03)
            or (
                (_is_call(record) or _is_aggressive(record))
                and equity + 0.03 < pot_odds
            )
        )
        if draw_error:
            codes.append(DRAW_ODDS_ERROR)

    if strong_draw and facing_bet and record.action == ActionType.FOLD and equity >= pot_odds:
        codes.append(STRONG_DRAW_OVERFOLD)
        if made_plus_draw:
            # 对手若极度缺少半诈唬，成牌+听牌仍可能弃牌；不以随机范围
            # 蒙特卡洛直接判为明显错误。
            rating = ActionRating.LOOSE_OR_TIGHT
            recommended_action = "通常跟注；对手范围极强时可弃牌"
            reason = (
                f"已有{made_hand_name}且兼有{draw.name}，通常应继续；"
                "但结论对对手范围中的价值牌与半诈唬比例敏感。"
            )
        else:
            rating = _worse(
                rating,
                ActionRating.CLEAR_ERROR
                if equity >= pot_odds + 0.10
                else ActionRating.LOOSE_OR_TIGHT,
            )
            recommended_action = "跟注或加注"
            reason = "纯强听牌的基准权益覆盖底池赔率，弃牌过紧。"
    elif strong_draw and facing_bet and record.action != ActionType.FOLD and equity >= pot_odds:
        if rating in {ActionRating.ACCEPTABLE, ActionRating.RECOMMENDED}:
            rating = ActionRating.RECOMMENDED
            if made_plus_draw:
                reason = (
                    f"已有{made_hand_name}且兼有{draw.name}；未中听牌时也可能领先，继续合理。"
                )
            else:
                reason = "纯强听牌的基准权益覆盖底池赔率，继续合理。"

    top_pair, weak_top_pair = _top_pair_state(snapshot)
    stackoff_decision = record.is_all_in or legal.call_amount >= snapshot.stack
    if top_pair and stackoff_decision:
        codes.append(TOP_PAIR_STACKOFF_OPPORTUNITY)
        if record.action != ActionType.FOLD and record.is_all_in:
            codes.append(TOP_PAIR_STACKED_OFF)
    weak_top_pair_spot = (
        weak_top_pair
        and facing_bet
        and (record.is_all_in or legal.call_amount >= legal.pot_before * 0.5)
    )
    if weak_top_pair_spot:
        codes.append(WEAK_TOP_PAIR_DECISION)
    if weak_top_pair_spot and _is_call(record):
        codes.append(WEAK_TOP_PAIR_OVERCALL)
        effective_paid = max(
            0,
            _effective_bet_to(context, record) - snapshot.street_commitment,
        )
        deep_commit = (
            snapshot.stack >= 40 * context.big_blind
            and effective_paid >= 25 * context.big_blind
        )
        rating = _worse(
            rating,
            ActionRating.CLEAR_ERROR if deep_commit else ActionRating.LOOSE_OR_TIGHT,
        )
        reason = "弱顶对面对大额投入仍跟到底，过度跟注。"
        recommended_action = "弃牌"

    small_pair_spot = _small_pair_missed(snapshot) and facing_bet
    if small_pair_spot:
        codes.append(SMALL_PAIR_POSTFLOP_DECISION)
    if (
        small_pair_spot
        and record.action != ActionType.FOLD
        and (snapshot.street in {Street.TURN, Street.RIVER} or equity < pot_odds + 0.08)
    ):
        codes.append(SMALL_PAIR_OVERCONTINUE)
        rating = _worse(
            rating,
            ActionRating.CLEAR_ERROR if record.is_all_in else ActionRating.LOOSE_OR_TIGHT,
        )
        reason = "小口袋对子未中三条，面对压力继续过多。"
        recommended_action = "弃牌"

    if _river_one_pair_overraise(context, record):
        codes.append(RIVER_SHOWDOWN_VALUE_OVERPLAY)
        effective_to = _effective_bet_to(context, record)
        rating = _worse(rating, ActionRating.LOOSE_OR_TIGHT)
        recommended_action = "跟注"
        reason = (
            "河牌一对牌已有抓诈唬价值；再加注会赶走诈唬和多数较弱牌，"
            "主要被更强牌跟注。"
        )
        details.append(f"按对手有效筹码，本次加注实际最多到 {effective_to}。")

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
        hit_probability=hit_probability,
        heuristic_version=POSTFLOP_HEURISTIC_VERSION,
        recommended_action=recommended_action,
        detail_lines=tuple(details),
        draw_names=draw.names,
        equity_basis=RANDOM_EQUITY_BASIS,
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
    "POSTFLOP_HEURISTIC_VERSION",
    "PREFLOP_RAISE_TOO_LARGE",
    "PREFLOP_RAISE_TOO_SMALL",
    "PREFLOP_HEURISTIC_VERSION",
    "PlayerCommitment",
    "RANDOM_EQUITY_BASIS",
    "RIVER_SHOWDOWN_VALUE_OVERPLAY",
    "SB_COLD_CALL",
    "SB_COLD_CALL_OPPORTUNITY",
    "SMALL_PAIR_OVERCONTINUE",
    "SMALL_PAIR_POSTFLOP_DECISION",
    "STRONG_DRAW_DECISION",
    "STRONG_DRAW_OVERFOLD",
    "THREEBET_SIZING_OPPORTUNITY",
    "THREEBET_TOO_LARGE",
    "THREEBET_TOO_SMALL",
    "TOP_PAIR_STACKED_OFF",
    "TOP_PAIR_STACKOFF_OPPORTUNITY",
    "WEAK_TOP_PAIR_DECISION",
    "WEAK_TOP_PAIR_OVERCALL",
    "capture_context",
    "review_decision",
]
