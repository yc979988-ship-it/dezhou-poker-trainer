"""离线牌后教练：只使用动作发生前可见的信息评价决策。

本模块不是完整 GTO 求解器。翻前使用带版本号的 6-max 100bb 启发式，
翻后按成牌强度、牌面结构、人数、压力、底池赔率与听牌质量给出反馈；
未知随机手牌的蒙特卡洛权益只作参考。
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
from .postflop_strategy import (
    BoardWetness,
    MadeStrength,
    PostflopProfile,
    analyze_postflop_profile,
)


PREFLOP_HEURISTIC_VERSION = "6max-100bb-preflop-v4"
POSTFLOP_HEURISTIC_VERSION = "postflop-structured-v3"
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


class _PressureLevel(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    OVERBET = "overbet"


@dataclass(frozen=True, slots=True)
class _PressureState:
    level: _PressureLevel
    fraction: float
    label_zh: str


@dataclass(frozen=True, slots=True)
class _DrawPriceState:
    raw_probability: float
    conservative_probability: float
    implied_buffer: float
    direct_supported: bool
    supported: bool
    sees_both_cards: bool

    @property
    def only_implied(self) -> bool:
        return self.supported and not self.direct_supported

    def direct_margin(self, pot_odds: float) -> float:
        return self.conservative_probability - pot_odds


def _postflop_pressure(context: CoachContext) -> _PressureState:
    bettor = _last_aggressive_record(context)
    fraction = (
        bettor.paid / bettor.pot_before
        if bettor is not None and bettor.pot_before > 0
        else 0.0
    )
    if fraction <= 0.20:
        level, label = _PressureLevel.TINY, "微小下注"
    elif fraction <= 0.50:
        level, label = _PressureLevel.SMALL, "小注"
    elif fraction <= 0.80:
        level, label = _PressureLevel.MEDIUM, "中等下注"
    elif fraction <= 1.25:
        level, label = _PressureLevel.LARGE, "大注"
    else:
        level, label = _PressureLevel.OVERBET, "超池下注"
    if bettor is not None and bettor.is_all_in:
        label += "（全下）"
    return _PressureState(level=level, fraction=fraction, label_zh=label)


def _pressure_at_least(
    pressure: _PressureState,
    threshold: _PressureLevel,
) -> bool:
    order = {
        _PressureLevel.TINY: 0,
        _PressureLevel.SMALL: 1,
        _PressureLevel.MEDIUM: 2,
        _PressureLevel.LARGE: 3,
        _PressureLevel.OVERBET: 4,
    }
    return order[pressure.level] >= order[threshold]


def _aggressive_risk_ratio(context: CoachContext, record: ActionRecord) -> float:
    effective_paid = max(
        0,
        _effective_bet_to(context, record) - context.snapshot.street_commitment,
    )
    return effective_paid / max(1, context.snapshot.pot_before)


def _draw_price_state(
    context: CoachContext,
    profile: PostflopProfile,
    draw: Any,
    pot_odds: float,
) -> _DrawPriceState:
    """返回可解释的直接赔率与隐含赔率状态。

    普通翻牌跟注先按下一张牌计算；英雄跟注后自己全下，或所有仍在局
    对手都已全下时，才视为已经买到转河两张牌。成对牌面和多人底池会
    折损原始 outs；只有深度足够且牌面较干净时才给隐含赔率缓冲。
    """

    snapshot = context.snapshot
    hero_all_in_after_call = context.legal_actions.call_amount >= snapshot.stack
    live_opponents = tuple(
        row
        for row in context.commitments
        if row.player_id != snapshot.player_id and not row.folded
    )
    all_opponents_all_in = bool(live_opponents) and all(
        row.all_in for row in live_opponents
    )
    sees_both_cards = bool(
        snapshot.street == Street.FLOP
        and (hero_all_in_after_call or all_opponents_all_in)
    )
    raw_probability = draw.hit_by_river if sees_both_cards else draw.hit_next
    discount = 1.0
    if profile.texture.paired:
        discount -= 0.08
    if profile.multiway:
        discount -= 0.07
    conservative_probability = raw_probability * max(0.75, discount)
    hero_after_call = max(0, snapshot.stack - context.legal_actions.call_amount)
    opponent_after_call = max(
        (
            max(0, row.stack - max(0, snapshot.current_bet - row.street_commitment))
            for row in live_opponents
            if not row.all_in
        ),
        default=0,
    )
    effective_after_call = min(hero_after_call, opponent_after_call)
    implied_buffer = 0.0
    if (
        not sees_both_cards
        and effective_after_call >= 10 * context.big_blind
        and not profile.texture.paired
        and not profile.texture.four_flush
    ):
        implied_buffer = 0.06 if snapshot.street == Street.FLOP else 0.03
        if profile.multiway:
            implied_buffer = min(implied_buffer, 0.02)
    direct_supported = conservative_probability >= pot_odds
    supported = conservative_probability + implied_buffer >= pot_odds
    return _DrawPriceState(
        raw_probability=raw_probability,
        conservative_probability=conservative_probability,
        implied_buffer=implied_buffer,
        direct_supported=direct_supported,
        supported=supported,
        sees_both_cards=sees_both_cards,
    )


def _vulnerable_strong_made(profile: PostflopProfile) -> bool:
    category = profile.rank.category
    return bool(
        profile.texture.four_flush and category < HandCategory.FLUSH
        or profile.texture.four_straight and category < HandCategory.STRAIGHT
    )


def _near_nut_made(profile: PostflopProfile) -> bool:
    """只给真正很强的牌豁免超大取值尺度检查。"""

    return profile.rank.category >= HandCategory.FULL_HOUSE


def _facing_bet_decision(
    context: CoachContext,
    record: ActionRecord,
    profile: PostflopProfile,
    *,
    draw_outs: int,
    draw_price: _DrawPriceState,
    pot_odds: float,
    pressure: _PressureState,
) -> tuple[ActionRating, str, str]:
    fold = record.action == ActionType.FOLD
    call = _is_call(record)
    aggressive = _is_aggressive(record)
    strength = profile.strength
    wet = profile.texture.wetness == BoardWetness.WET
    large = _pressure_at_least(pressure, _PressureLevel.LARGE)
    overbet = pressure.level == _PressureLevel.OVERBET
    strong_draw = draw_outs >= 8
    clean_draw = not profile.multiway and not profile.texture.paired

    if profile.board_locked:
        if fold:
            return ActionRating.CLEAR_ERROR, "公共牌已锁定同牌平分，弃牌会放弃应得底池份额。", "跟注"
        if call:
            return ActionRating.RECOMMENDED, "公共牌已锁定同牌平分，跟注可拿回自己的投入并分享原底池。", "跟注"
        if aggressive:
            return ActionRating.LOOSE_OR_TIGHT, "公共牌已锁定平分；加注不能提高牌力，默认跟注即可。", "跟注"

    made_plus_strong_draw = bool(
        strong_draw
        and strength
        in {
            MadeStrength.WEAK_PAIR,
            MadeStrength.MEDIUM_PAIR,
            MadeStrength.TOP_PAIR_WEAK,
            MadeStrength.TOP_PAIR_STRONG,
            MadeStrength.OVERPAIR,
        }
    )
    if made_plus_strong_draw and (draw_price.supported or not large):
        if fold:
            return (
                ActionRating.LOOSE_OR_TIGHT,
                f"{strength.label_zh}兼有强听牌，普通压力下直接弃牌偏紧。",
                "跟注",
            )
        if call:
            rating = (
                ActionRating.RECOMMENDED
                if draw_price.direct_supported and clean_draw and not wet
                else ActionRating.ACCEPTABLE
            )
            return (
                rating,
                f"{strength.label_zh}兼有强听牌，成牌摊牌值与听牌权益共同支持继续。",
                "跟注",
            )
        if aggressive:
            rating = ActionRating.LOOSE_OR_TIGHT if large else ActionRating.ACCEPTABLE
            return (
                rating,
                f"{strength.label_zh}兼有强听牌可以半诈唬，但仍应控制加注尺度。",
                "跟注或小尺度加注",
            )

    if strength == MadeStrength.STRONG_MADE:
        vulnerable = _vulnerable_strong_made(profile)
        if vulnerable:
            hazard = "四同花" if profile.texture.four_flush else "四连顺"
            severe = overbet or (profile.multiway and large)
            if fold:
                rating = ActionRating.RECOMMENDED if severe else ActionRating.ACCEPTABLE
                recommendation = "弃牌" if severe else "跟注或弃牌"
                return rating, f"{hazard}牌面显著压低当前成牌；高压下可以弃牌。", recommendation
            if call:
                rating = ActionRating.LOOSE_OR_TIGHT if severe else ActionRating.ACCEPTABLE
                recommendation = "弃牌" if severe else "跟注或弃牌"
                return (
                    rating,
                    f"多人{hazard}牌面面对{pressure.label_zh}，两对或三条不能自动当坚果跟注。",
                    recommendation,
                )
            if aggressive:
                rating = ActionRating.CLEAR_ERROR if severe else ActionRating.LOOSE_OR_TIGHT
                return rating, f"{hazard}牌面使当前成牌相对脆弱，不宜继续做大底池。", "跟注或弃牌"
        cautious = profile.multiway or wet or large
        if fold:
            rating = ActionRating.LOOSE_OR_TIGHT if cautious and large else ActionRating.CLEAR_ERROR
            return rating, "两对及以上通常应继续，直接弃牌过紧。", "跟注"
        if call:
            recommendation = "跟注" if cautious else "跟注或加注"
            rating = ActionRating.ACCEPTABLE if cautious else ActionRating.RECOMMENDED
            return rating, "两对及以上足以继续；复杂牌面先跟注可保留较弱范围。", recommendation
        if aggressive:
            rating = ActionRating.ACCEPTABLE if cautious else ActionRating.RECOMMENDED
            recommendation = "跟注或加注" if cautious else "加注"
            return rating, "强成牌可以主动取值，但湿润或多人底池要控制加注范围。", recommendation

    if strength in {
        MadeStrength.TOP_PAIR_WEAK,
        MadeStrength.TOP_PAIR_STRONG,
        MadeStrength.OVERPAIR,
    }:
        weak = strength == MadeStrength.TOP_PAIR_WEAK
        caution = int(weak) + int(profile.multiway) + int(wet) + int(large)
        if fold:
            if weak and caution >= 2:
                return ActionRating.RECOMMENDED, "弱顶对面对收紧后的大额范围，弃牌合理。", "弃牌"
            rating = ActionRating.ACCEPTABLE if caution >= 2 else ActionRating.LOOSE_OR_TIGHT
            return rating, "一对牌有摊牌价值；是否弃牌取决于人数、牌面和下注压力。", "跟注"
        if call:
            if weak and caution >= 3:
                return (
                    ActionRating.LOOSE_OR_TIGHT,
                    "弱顶对在多人湿润牌面面对较大压力，跟注范围应明显收紧。",
                    "弃牌",
                )
            if weak and caution >= 2 or not weak and caution >= 3:
                return ActionRating.LOOSE_OR_TIGHT, "一对牌面对多重风险继续偏松。", "弃牌"
            if caution:
                return ActionRating.ACCEPTABLE, "一对牌可以跟注，但不宜把随机权益当成对下注范围的保证。", "跟注"
            return ActionRating.RECOMMENDED, "单挑干燥牌面面对正常下注，一对牌继续合理。", "跟注"
        if aggressive:
            if caution >= 3:
                return ActionRating.CLEAR_ERROR, "脆弱的一对牌不适合在高压场景继续做大底池。", "跟注或弃牌"
            return ActionRating.LOOSE_OR_TIGHT, "一对牌加注会赶走诈唬并更多被强牌继续。", "跟注"

    if strength in {MadeStrength.WEAK_SHOWDOWN, MadeStrength.BOARD_ONLY}:
        kicker_plays = strength == MadeStrength.WEAK_SHOWDOWN
        cheap = pot_odds <= (0.15 if kicker_plays else 0.08)
        description = "底牌踢脚仍参与最终五张牌" if kicker_plays else "当前主要依靠公牌成牌"
        if fold:
            if cheap:
                return ActionRating.LOOSE_OR_TIGHT, f"{description}，面对微小下注直接弃牌略紧。", "跟注或弃牌"
            return ActionRating.RECOMMENDED, f"{description}，但面对正常压力通常应弃牌。", "弃牌"
        if call:
            if cheap:
                return ActionRating.ACCEPTABLE, f"{description}；价格很低时可以抓诈唬。", "跟注或弃牌"
            rating = ActionRating.CLEAR_ERROR if pot_odds >= 0.30 else ActionRating.LOOSE_OR_TIGHT
            return rating, f"{description}，不足以支撑高价跟注。", "弃牌"
        if aggressive:
            rating = ActionRating.CLEAR_ERROR if large else ActionRating.LOOSE_OR_TIGHT
            return rating, f"{description}；加注需要明确弃牌率，不能按强成牌取值。", "弃牌或跟注"

    if strength in {MadeStrength.WEAK_PAIR, MadeStrength.MEDIUM_PAIR}:
        cheap = pot_odds <= 0.12
        tolerable = pot_odds <= 0.22 and not profile.multiway and not wet
        if fold:
            if cheap:
                return ActionRating.LOOSE_OR_TIGHT, "价格很低，带摊牌价值的对子弃牌略紧。", "跟注"
            return ActionRating.RECOMMENDED, "中弱对子面对正常压力应以弃牌控制损失。", "弃牌"
        if call:
            if cheap:
                return ActionRating.RECOMMENDED, "下注很小，当前对子按价格跟注合理。", "跟注"
            if tolerable:
                return ActionRating.ACCEPTABLE, "单挑正常价格下可继续一次，但后续压力需收紧。", "跟注"
            rating = ActionRating.CLEAR_ERROR if pot_odds >= 0.35 else ActionRating.LOOSE_OR_TIGHT
            return rating, "中弱对子在多人、湿润或高价场景继续偏多。", "弃牌"
        if aggressive:
            rating = ActionRating.CLEAR_ERROR if large or profile.multiway or wet else ActionRating.LOOSE_OR_TIGHT
            return rating, "中弱对子缺少足够取值目标，加注会把底池做得过大。", "跟注或弃牌"

    pure_draw = draw_outs > 0 and strength in {MadeStrength.AIR, MadeStrength.BOARD_ONLY}
    if pure_draw:
        if fold:
            if draw_price.direct_supported:
                clear_miss = bool(
                    strong_draw
                    and clean_draw
                    and draw_price.direct_margin(pot_odds) >= 0.05
                )
                rating = ActionRating.CLEAR_ERROR if clear_miss else ActionRating.LOOSE_OR_TIGHT
                return rating, "听牌的折损后直接命中率覆盖赔率，弃牌过紧。", "跟注或加注"
            if draw_price.only_implied:
                return (
                    ActionRating.ACCEPTABLE,
                    "直接赔率略不足，但深筹码隐含赔率可补足；弃牌可以接受，跟注也有条件。",
                    "弃牌或跟注",
                )
            return ActionRating.RECOMMENDED, "超池压力下听牌命中率不足以覆盖赔率，弃牌合理。", "弃牌"
        if call:
            if draw_price.supported:
                rating = (
                    ActionRating.RECOMMENDED
                    if strong_draw and clean_draw and draw_price.direct_supported
                    else ActionRating.ACCEPTABLE
                )
                reason = (
                    "听牌的折损后直接命中率覆盖价格，继续合理。"
                    if draw_price.direct_supported
                    else "直接赔率略不足，但深筹码隐含赔率使跟注可以接受。"
                )
                return rating, reason, "跟注"
            gap = pot_odds - draw_price.conservative_probability
            rating = ActionRating.CLEAR_ERROR if gap >= 0.10 else ActionRating.LOOSE_OR_TIGHT
            return rating, "听牌价格过高，原始 outs 不能覆盖本次跟注成本。", "弃牌"
        if aggressive:
            if not draw_price.supported:
                return ActionRating.LOOSE_OR_TIGHT, "平跟赔率不足；半诈唬还需要足够弃牌率并控制尺度。", "弃牌或小尺度加注"
            rating = (
                ActionRating.ACCEPTABLE
                if profile.multiway or wet
                else ActionRating.RECOMMENDED
                if strong_draw
                else ActionRating.LOOSE_OR_TIGHT
            )
            return rating, "强听牌可半诈唬，但仍需单独控制加注尺度。", "跟注或加注"

    if context.snapshot.street == Street.RIVER:
        if fold:
            return ActionRating.RECOMMENDED, "河牌已经没有未来权益，空气牌面对下注应弃牌。", "弃牌"
        if call:
            rating = ActionRating.LOOSE_OR_TIGHT if pot_odds <= 0.08 else ActionRating.CLEAR_ERROR
            return rating, "河牌空气牌没有未来权益，跟注为负期望。", "弃牌"
        if aggressive:
            return ActionRating.CLEAR_ERROR, "河牌空气牌加注需要明确阻断与弃牌率，不能靠随机权益支撑。", "弃牌"

    if fold:
        return ActionRating.RECOMMENDED, "没有成牌或足够听牌，面对下注弃牌合理。", "弃牌"
    if call:
        rating = ActionRating.CLEAR_ERROR if pot_odds >= 0.25 else ActionRating.LOOSE_OR_TIGHT
        return rating, "缺少成牌和足够 outs，跟注偏松。", "弃牌"
    if aggressive:
        return ActionRating.LOOSE_OR_TIGHT, "纯诈唬需要牌面和弃牌率支持，随机权益不能证明加注合理。", "弃牌或小尺度加注"
    return ActionRating.ACCEPTABLE, "动作可以接受。", "跟注或弃牌"


def _unopened_postflop_decision(
    context: CoachContext,
    record: ActionRecord,
    profile: PostflopProfile,
    *,
    draw_outs: int,
) -> tuple[ActionRating, str, str]:
    strength = profile.strength
    wet = profile.texture.wetness == BoardWetness.WET
    aggressive = _is_aggressive(record)
    if record.action == ActionType.FOLD:
        return ActionRating.CLEAR_ERROR, "可以过牌保留权益，无需弃牌。", "过牌"
    if record.action == ActionType.CHECK:
        if strength == MadeStrength.STRONG_MADE:
            return ActionRating.ACCEPTABLE, "强牌过牌可以诱导，但通常仍应考虑价值下注。", "下注"
        if strength in {
            MadeStrength.TOP_PAIR_STRONG,
            MadeStrength.TOP_PAIR_WEAK,
            MadeStrength.OVERPAIR,
        }:
            return ActionRating.ACCEPTABLE, "一对牌过牌控池可以接受，湿润牌面尤其需要保护范围。", "过牌或下注"
        return ActionRating.RECOMMENDED, "当前牌力适合过牌保留权益。", "过牌"
    if aggressive:
        if strength == MadeStrength.STRONG_MADE:
            return ActionRating.RECOMMENDED, "两对及以上主动价值下注合理。", "下注"
        if strength in {
            MadeStrength.TOP_PAIR_STRONG,
            MadeStrength.TOP_PAIR_WEAK,
            MadeStrength.OVERPAIR,
        }:
            rating = ActionRating.ACCEPTABLE if profile.multiway or wet else ActionRating.RECOMMENDED
            return rating, "一对牌可以价值下注；多人或湿润牌面应减少薄价值范围。", "下注或过牌"
        if strength in {MadeStrength.MEDIUM_PAIR, MadeStrength.WEAK_PAIR}:
            if profile.multiway or wet:
                return ActionRating.LOOSE_OR_TIGHT, "多人或湿润牌面用中弱对子下注偏薄。", "过牌"
            return ActionRating.ACCEPTABLE, "单挑干燥牌面可小注保护，也可过牌控池。", "下注或过牌"
        if draw_outs >= 8:
            rating = ActionRating.ACCEPTABLE if profile.multiway or wet else ActionRating.RECOMMENDED
            recommendation = "过牌或下注" if profile.multiway else "下注或过牌"
            return rating, "强听牌可以半诈唬；多人底池应降低纯进攻频率。", recommendation
        risk_ratio = _aggressive_risk_ratio(context, record)
        if not profile.multiway and not wet and risk_ratio <= 0.60:
            return ActionRating.ACCEPTABLE, "单挑合适牌面的小尺度诈唬可以接受。", "小尺度下注或过牌"
        rating = ActionRating.CLEAR_ERROR if profile.multiway and risk_ratio > 0.80 else ActionRating.LOOSE_OR_TIGHT
        return rating, "缺少摊牌价值的下注需要弃牌率；多人底池不宜频繁诈唬。", "过牌"
    return ActionRating.ACCEPTABLE, "动作可以接受。", "过牌"


def _review_postflop_random_v2(
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


def _review_postflop(
    context: CoachContext, record: ActionRecord, *, trials: int
) -> DecisionReview:
    """按牌力、牌面、人数、价格和尺度评价翻后动作。

    随机未知手牌 Monte Carlo 仍作为可复核的基准展示，但不再直接充当
    对手下注范围；动作规则先于权益基准，进攻尺度再单独校验。
    """

    snapshot = context.snapshot
    pot_odds = _contestable_pot_odds(context)
    profile = analyze_postflop_profile(
        snapshot.hole_cards,
        snapshot.board,
        active_players=snapshot.active_players,
    )
    equity = estimate_equity(
        snapshot.hole_cards,
        snapshot.board,
        opponents=max(1, snapshot.active_players - 1),
        trials=trials,
        seed=_equity_seed(context),
    )
    draw = analyze_common_draws(snapshot.hole_cards, snapshot.board)
    facing_bet = context.legal_actions.to_call > 0
    pressure = _postflop_pressure(context) if facing_bet else None
    strong_draw = draw.outs >= 8
    personal_made, made_hand_name, _made_category = _personal_made_hand(snapshot)
    made_plus_draw = personal_made and draw.outs > 0
    hit_probability = (
        draw.hit_by_river if snapshot.street == Street.FLOP else draw.hit_next
    )
    draw_price = _DrawPriceState(
        raw_probability=0.0,
        conservative_probability=0.0,
        implied_buffer=0.0,
        direct_supported=False,
        supported=False,
        sees_both_cards=False,
    )
    if facing_bet and draw.outs > 0:
        draw_price = _draw_price_state(context, profile, draw, pot_odds)

    pressure_label = f"面对{pressure.label_zh}" if pressure is not None else "无人下注"
    details = [
        (
            f"{profile.strength.label_zh}｜{profile.texture.label_zh}｜"
            f"{profile.players_label_zh}｜{pressure_label}。"
        )
    ]
    if facing_bet:
        details.append(f"跟注盈亏平衡点为 {pot_odds:.1%}。")
    details.append(
        f"随机未知手牌基准权益约 {equity:.1%}；它不代表对手当前下注范围。"
    )
    if draw.outs > 0:
        draw_window = "到河牌" if snapshot.street == Street.FLOP else "下一张"
        details.append(
            f"{draw.name}共 {draw.outs} 个原始 outs（未扣除脏 outs），"
            f"{draw_window}原始命中率约 {hit_probability:.1%}。"
        )
        if facing_bet:
            if draw_price.sees_both_cards:
                details.append(
                    f"跟注后已确保看到转河两张牌，本次按到河牌保守命中率约 "
                    f"{draw_price.conservative_probability:.1%} 判断。"
                )
            else:
                details.append(
                    f"本次先按下一张保守命中率约 "
                    f"{draw_price.conservative_probability:.1%} 判断；"
                    "成对牌面和多人底池会折损。"
                )
            if draw_price.only_implied:
                details.append(
                    f"直接赔率尚差约 "
                    f"{max(0.0, pot_odds - draw_price.conservative_probability):.1%}；"
                    f"仅靠约 {draw_price.implied_buffer:.1%} 的深筹码隐含赔率缓冲才覆盖。"
                )
        if made_plus_draw:
            details.append(
                f"这是{made_hand_name}+听牌；未中听牌时，现有{made_hand_name}仍可能赢。"
            )
        else:
            details.append("这是纯听牌；未命中时通常缺少现成摊牌价值。")
    elif snapshot.street == Street.RIVER:
        details.append("河牌不会再发公共牌，因此不计算未来 outs。")

    if facing_bet and pressure is not None:
        rating, reason, recommended_action = _facing_bet_decision(
            context,
            record,
            profile,
            draw_outs=draw.outs,
            draw_price=draw_price,
            pot_odds=pot_odds,
            pressure=pressure,
        )
    else:
        rating, reason, recommended_action = _unopened_postflop_decision(
            context,
            record,
            profile,
            draw_outs=draw.outs,
        )

    if _is_aggressive(record):
        risk_ratio = _aggressive_risk_ratio(context, record)
        details.append(f"进攻尺度约为动作前底池的 {risk_ratio:.0%}。")
        if risk_ratio > 1.50 and not _near_nut_made(profile):
            severe_size = bool(
                risk_ratio >= 3.0 or record.is_all_in and risk_ratio >= 2.0
            )
            rating = _worse(
                rating,
                ActionRating.CLEAR_ERROR if severe_size else ActionRating.LOOSE_OR_TIGHT,
            )
            if draw.outs > 0:
                reason = (
                    "听牌可以继续，但本次全下/超大加注尺度投入过多；"
                    "半诈唬动作合理不等于任意尺度合理。"
                )
                recommended_action = "跟注或加注"
            elif facing_bet:
                reason = "动作可能有一定权益，但本次加注尺度使风险明显过大。"
                recommended_action = "跟注或弃牌"
            else:
                reason = "诈唬需要弃牌率支持，本次下注尺度投入过大。"
                recommended_action = "过牌或小尺度下注"

    codes: list[str] = []
    if strong_draw and facing_bet:
        codes.append(STRONG_DRAW_DECISION)
    if facing_bet and draw.outs > 0:
        codes.append(DRAW_ODDS_OPPORTUNITY)
        if not made_plus_draw:
            clean_direct_fold = bool(
                record.action == ActionType.FOLD
                and draw_price.direct_supported
                and not profile.multiway
                and not profile.texture.paired
                and draw_price.direct_margin(pot_odds) >= 0.03
            )
            unsupported_call = _is_call(record) and not draw_price.supported
            draw_error = clean_direct_fold or unsupported_call
            if draw_error:
                codes.append(DRAW_ODDS_ERROR)

    if strong_draw and facing_bet and record.action == ActionType.FOLD and draw_price.supported:
        codes.append(STRONG_DRAW_OVERFOLD)
        if made_plus_draw:
            if draw_price.direct_supported:
                rating = _worse(rating, ActionRating.LOOSE_OR_TIGHT)
            recommended_action = "通常跟注；对手范围极强时可弃牌"
            reason = (
                f"已有{made_hand_name}且兼有{draw.name}，通常应继续；"
                "但结论仍取决于对手范围中的价值牌与半诈唬比例。"
            )
        else:
            if draw_price.direct_supported:
                clear_miss = bool(
                    not profile.multiway
                    and not profile.texture.paired
                    and draw_price.direct_margin(pot_odds) >= 0.05
                )
                rating = _worse(
                    rating,
                    ActionRating.CLEAR_ERROR
                    if clear_miss
                    else ActionRating.LOOSE_OR_TIGHT,
                )
                recommended_action = "跟注或加注"
                reason = "纯强听牌的折损后直接命中率覆盖底池赔率，弃牌过紧。"
            else:
                recommended_action = "弃牌或跟注"
                reason = "直接赔率略不足；深筹码隐含赔率可支持跟注，但弃牌并非明显错误。"
    elif (
        strong_draw
        and facing_bet
        and record.action != ActionType.FOLD
        and draw_price.supported
        and rating in {ActionRating.ACCEPTABLE, ActionRating.RECOMMENDED}
    ):
        reason = (
            f"已有{made_hand_name}且兼有{draw.name}，继续合理。"
            if made_plus_draw
            else "纯强听牌的价格条件支持继续。"
        )

    top_pair = profile.strength in {
        MadeStrength.TOP_PAIR_WEAK,
        MadeStrength.TOP_PAIR_STRONG,
    }
    weak_top_pair = profile.strength == MadeStrength.TOP_PAIR_WEAK
    stackoff_decision = record.is_all_in or context.legal_actions.call_amount >= snapshot.stack
    if top_pair and stackoff_decision:
        codes.append(TOP_PAIR_STACKOFF_OPPORTUNITY)
        if record.action != ActionType.FOLD and record.is_all_in:
            codes.append(TOP_PAIR_STACKED_OFF)
    wet_multiway_pressure = bool(
        weak_top_pair
        and facing_bet
        and profile.multiway
        and profile.texture.wetness == BoardWetness.WET
        and pressure is not None
        and _pressure_at_least(pressure, _PressureLevel.MEDIUM)
    )
    weak_top_pair_spot = bool(
        weak_top_pair
        and facing_bet
        and (
            stackoff_decision
            or wet_multiway_pressure
            or context.legal_actions.call_amount
            >= context.legal_actions.pot_before * 0.5
        )
    )
    if weak_top_pair_spot:
        codes.append(WEAK_TOP_PAIR_DECISION)
    protected_by_strong_draw = bool(
        made_plus_draw
        and strong_draw
        and pressure is not None
        and not _pressure_at_least(pressure, _PressureLevel.LARGE)
    )
    if weak_top_pair_spot and _is_call(record) and not protected_by_strong_draw:
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
        reason = (
            "弱顶对在多人湿润牌面面对大额下注仍跟注，范围过宽。"
            if wet_multiway_pressure
            else "弱顶对面对大额投入仍跟到底，过度跟注。"
        )
        recommended_action = "弃牌"

    small_pair_spot = _small_pair_missed(snapshot) and facing_bet
    if small_pair_spot:
        codes.append(SMALL_PAIR_POSTFLOP_DECISION)
    if (
        small_pair_spot
        and record.action != ActionType.FOLD
        and pot_odds > 0.10
        and (
            snapshot.street in {Street.TURN, Street.RIVER}
            or equity < pot_odds + 0.08
        )
    ):
        codes.append(SMALL_PAIR_OVERCONTINUE)
        rating = _worse(
            rating,
            ActionRating.CLEAR_ERROR if record.is_all_in else ActionRating.LOOSE_OR_TIGHT,
        )
        reason = "小口袋对子未中三条，面对正常以上压力继续过多。"
        recommended_action = "弃牌"

    if _river_one_pair_overraise(context, record):
        codes.append(RIVER_SHOWDOWN_VALUE_OVERPLAY)
        effective_to = _effective_bet_to(context, record)
        rating = _worse(rating, ActionRating.LOOSE_OR_TIGHT)
        recommended_action = "跟注"
        overplay_reason = (
            "河牌一对牌已有抓诈唬价值；再加注会赶走诈唬和多数较弱牌，"
            "主要被更强牌跟注。"
        )
        reason = (
            f"本次全下尺度风险过大；{overplay_reason}"
            if record.is_all_in and rating == ActionRating.CLEAR_ERROR
            else overplay_reason
        )
        details.append(f"按对手有效筹码，本次加注实际最多到 {effective_to}。")

    details.insert(1, f"结构化建议：{recommended_action}。{reason}")
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
