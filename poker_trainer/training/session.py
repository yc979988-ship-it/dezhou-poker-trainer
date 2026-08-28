"""训练场次编排：连接牌局引擎、动态对手、教练、统计和 SQLite。

本模块只模拟离线牌局。对手的连续画像只保存在 ``TrainingSession`` 内部，
任何公开状态都不会返回这些隐藏参数。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from types import MappingProxyType
from typing import Any

from poker_trainer.analytics.database import SQLiteStore
from poker_trainer.analytics.statistics import (
    HandStatistics,
    PositionStatistics,
    aggregate_by_position,
    project_hand_statistics,
    serialize_position_statistics,
)
from poker_trainer.coaching.coach import (
    DecisionReview,
    capture_context,
    review_decision,
)
from poker_trainer.engine.hand import HoldemHand, InvalidAction
from poker_trainer.engine.models import (
    PREFLOP_ORDER,
    ActionRecord,
    ActionType,
    Position,
    Seat,
    Street,
)
from poker_trainer.engine.replay import ReplayBundle
from poker_trainer.opponents.policy import BotDecision, OpponentPolicy, PolicyContext
from poker_trainer.opponents.profiles import (
    OpponentProfile,
    drift_for_session,
    generate_base_profile,
)

from .adaptive import (
    PROFILE_INTERVAL,
    AdaptiveScheduler,
    PlayerProfile,
    ScenarioPlan,
    ScenarioSpec,
    generate_player_profile,
)


class TrainingMode(str, Enum):
    """训练中教练反馈的展示时机。"""

    TEST = "test"
    TEACHING = "teaching"

    @property
    def label_zh(self) -> str:
        return "测试模式" if self is TrainingMode.TEST else "教学模式"

    @classmethod
    def parse(cls, value: "TrainingMode | str") -> "TrainingMode":
        if isinstance(value, cls):
            return value
        aliases = {
            "test": cls.TEST,
            "测试": cls.TEST,
            "测试模式": cls.TEST,
            "teaching": cls.TEACHING,
            "teach": cls.TEACHING,
            "教学": cls.TEACHING,
            "教学模式": cls.TEACHING,
        }
        try:
            return aliases[str(value).strip().lower()]
        except KeyError as exc:
            raise ValueError("mode 必须是 test/测试模式 或 teaching/教学模式") from exc


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """一场 6 人桌训练的稳定配置。"""

    mode: TrainingMode | str = TrainingMode.TEST
    small_blind: int = 20
    big_blind: int = 40
    buy_in: int = 4_000
    chips_per_yuan: int = 100
    auto_top_up: bool = True
    hero_id: str = "hero"
    seed: int = 0
    coach_trials: int = 2_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", TrainingMode.parse(self.mode))
        for field_name in (
            "small_blind",
            "big_blind",
            "buy_in",
            "chips_per_yuan",
            "seed",
            "coach_trials",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} 必须是整数")
        if self.small_blind <= 0 or self.big_blind <= self.small_blind:
            raise ValueError("盲注必须满足 0 < small_blind < big_blind")
        if self.buy_in <= 0:
            raise ValueError("buy_in 必须大于 0")
        if self.chips_per_yuan <= 0:
            raise ValueError("chips_per_yuan 必须大于 0")
        if self.coach_trials <= 0:
            raise ValueError("coach_trials 必须大于 0")
        if not isinstance(self.auto_top_up, bool):
            raise TypeError("auto_top_up 必须是布尔值")
        if not isinstance(self.hero_id, str) or not self.hero_id.strip():
            raise ValueError("hero_id 不能为空")

    @property
    def buy_in_big_blinds(self) -> float:
        return self.buy_in / self.big_blind

    @property
    def buy_in_yuan(self) -> float:
        return self.buy_in / self.chips_per_yuan

    def chips_to_yuan(self, chips: int | float) -> float:
        return float(chips) / self.chips_per_yuan


def _derived_seed(master_seed: int, namespace: str, number: int = 0) -> int:
    """生成 SQLite 可安全保存的稳定正 63 位种子。"""

    digest = blake2b(digest_size=8, person=b"pkr-session-v1")
    for part in (str(master_seed), namespace, str(number)):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big") & ((1 << 63) - 1)


class TrainingSession:
    """一个可复现的离线训练场次。

    ``start_hand`` 会创建下一手并自动让对手行动，直至轮到英雄或牌局结束。
    教学模式下 ``hero_action`` 立即返回该动作的 ``DecisionReview``；测试模式
    在牌局结束前返回 ``None``，结束时一次返回整手英雄复盘元组。
    """

    _BOT_COUNT = 5
    _MAX_BOT_ACTIONS_PER_ADVANCE = 1_000

    def __init__(
        self,
        config: SessionConfig | None = None,
        *,
        db_path: str | Path | None = None,
        store: SQLiteStore | None = None,
        session_id: str | None = None,
        scheduler: AdaptiveScheduler | None = None,
    ) -> None:
        self.config = config or SessionConfig()
        if not isinstance(self.config, SessionConfig):
            raise TypeError("config 必须是 SessionConfig")
        if db_path is not None and store is not None:
            raise ValueError("db_path 与 store 只能提供一个")
        if store is not None and not isinstance(store, SQLiteStore):
            raise TypeError("store 必须是 SQLiteStore")

        self._owns_store = store is None
        self._store = store or SQLiteStore(db_path or ":memory:")
        self._scheduler = scheduler or AdaptiveScheduler()
        self.session_id = self._store.create_session(
            session_id,
            mode=self.config.mode,
            hero_player_id=self.config.hero_id,
            seed=self.config.seed,
            effective_stack=self.config.buy_in,
            auto_top_up=self.config.auto_top_up,
            metadata={
                "small_blind": self.config.small_blind,
                "big_blind": self.config.big_blind,
                "chips_per_yuan": self.config.chips_per_yuan,
                "buy_in_yuan": self.config.buy_in_yuan,
                "offline_only": True,
            },
        )

        self._physical_player_ids = self._build_player_ids()
        self._player_names = {
            self.config.hero_id: "你",
            **{
                player_id: f"对手{index}"
                for index, player_id in enumerate(
                    self._physical_player_ids[1:], start=1
                )
            },
        }
        self._stacks = {
            player_id: self.config.buy_in for player_id in self._physical_player_ids
        }
        self._bot_profiles: dict[str, OpponentProfile] = {
            player_id: drift_for_session(
                generate_base_profile(player_id, self.config.seed),
                _derived_seed(self.config.seed, "profile-drift"),
            )
            for player_id in self._physical_player_ids
            if player_id != self.config.hero_id
        }

        self.current_hand: HoldemHand | None = None
        self._current_scenario: ScenarioSpec | None = None
        self._current_policy_seed: int | None = None
        self._forced_open_pending = False
        self._current_reviews_internal: list[DecisionReview] = []
        self._last_hand_reviews: tuple[DecisionReview, ...] = ()
        self._all_reviews: list[DecisionReview] = []
        self._hand_statistics: list[HandStatistics] = []
        self._profile_history: list[PlayerProfile] = []
        self._current_plan = self._scheduler.build_plan(None, 1)
        self._completed_hand_count = 0
        self._finalized_hand_ids: set[str] = set()
        self._last_top_ups: dict[str, int] = {}
        self._closed = False

    def _build_player_ids(self) -> tuple[str, ...]:
        ids = [self.config.hero_id]
        suffix = 1
        while len(ids) <= self._BOT_COUNT:
            candidate = f"bot-{suffix}"
            suffix += 1
            if candidate not in ids:
                ids.append(candidate)
        return tuple(ids)

    @property
    def hero_id(self) -> str:
        return self.config.hero_id

    @property
    def mode(self) -> TrainingMode:
        return self.config.mode

    @property
    def database_path(self) -> Path:
        return self._store.path

    @property
    def completed_hand_count(self) -> int:
        return self._completed_hand_count

    @property
    def current_scenario(self) -> ScenarioSpec | None:
        return self._current_scenario

    @property
    def current_plan(self) -> ScenarioPlan:
        return self._current_plan

    @property
    def latest_profile(self) -> PlayerProfile | None:
        return self._profile_history[-1] if self._profile_history else None

    @property
    def profile_history(self) -> tuple[PlayerProfile, ...]:
        """只返回英雄的训练画像；不包含任何对手隐藏参数。"""

        return tuple(self._profile_history)

    @property
    def current_reviews(self) -> tuple[DecisionReview, ...]:
        if (
            self.mode is TrainingMode.TEST
            and self.current_hand is not None
            and not self.current_hand.is_complete
        ):
            return ()
        return tuple(self._current_reviews_internal)

    @property
    def last_hand_reviews(self) -> tuple[DecisionReview, ...]:
        return self._last_hand_reviews

    @property
    def hand_statistics(self) -> tuple[HandStatistics, ...]:
        return tuple(self._hand_statistics)

    @property
    def position_statistics(self) -> dict[Position, PositionStatistics]:
        return aggregate_by_position(self._hand_statistics)

    @property
    def player_stacks(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._stacks))

    @property
    def last_top_ups(self) -> Mapping[str, int]:
        """最近一次开新手前的补码额，值为增加的筹码数。"""

        return MappingProxyType(dict(self._last_top_ups))

    @property
    def stored_hands(self) -> list[dict[str, Any]]:
        return self._store.list_hands(self.session_id)

    def start_hand(self) -> HoldemHand:
        """开始下一手，并把机器人推进到英雄的决策点。"""

        self._ensure_open()
        if self.current_hand is not None and not self.current_hand.is_complete:
            raise RuntimeError("当前手牌尚未结束")
        if self.current_hand is not None:
            self._finalize_if_complete()

        self._apply_top_ups()
        hand_no = self._completed_hand_count + 1
        scenario = self._scheduler.choose_scenario(
            self._current_plan, self.config.seed, hand_no
        )
        positions = self._positions_for_hand(hand_no, scenario.preferred_position)
        seats = tuple(
            Seat(
                player_id=player_id,
                name=self._player_names[player_id],
                position=positions[player_id],
                stack=self._stacks[player_id],
            )
            for player_id in self._physical_player_ids
        )
        hand_seed = _derived_seed(self.config.seed, "hand", hand_no)
        self._current_policy_seed = _derived_seed(
            self.config.seed, "policy", hand_no
        )
        self.current_hand = scenario.create_hand(
            seats,
            seed=hand_seed,
            hero_id=self.hero_id,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            hand_id=f"{self.session_id}-hand-{hand_no}",
            session_id=self.session_id,
            hand_no=hand_no,
        )
        self._current_scenario = scenario
        self._forced_open_pending = scenario.forced_open
        self._current_reviews_internal = []
        self.advance_bots()
        return self.current_hand

    def _positions_for_hand(
        self, hand_no: int, preferred_hero_position: Position | None
    ) -> dict[str, Position]:
        # PREFLOP_ORDER 让英雄前六手依次遍历 UTG/HJ/CO/BTN/SB/BB。
        positions = {
            player_id: PREFLOP_ORDER[(index + hand_no - 1) % len(PREFLOP_ORDER)]
            for index, player_id in enumerate(self._physical_player_ids)
        }
        if preferred_hero_position is not None:
            current = positions[self.hero_id]
            if current != preferred_hero_position:
                # 整桌统一旋转，而非把英雄与某名 bot 对调；这样六名物理
                # 玩家在定向场景里仍保持原有相对座次。
                shift = (
                    PREFLOP_ORDER.index(preferred_hero_position)
                    - PREFLOP_ORDER.index(current)
                ) % len(PREFLOP_ORDER)
                positions = {
                    player_id: PREFLOP_ORDER[
                        (PREFLOP_ORDER.index(position) + shift)
                        % len(PREFLOP_ORDER)
                    ]
                    for player_id, position in positions.items()
                }
        return positions

    def _apply_top_ups(self) -> None:
        self._last_top_ups = {}
        if not self.config.auto_top_up:
            return
        for player_id, stack in tuple(self._stacks.items()):
            if stack < self.config.buy_in:
                self._last_top_ups[player_id] = self.config.buy_in - stack
                self._stacks[player_id] = self.config.buy_in

    def advance_bots(self) -> tuple[ActionRecord, ...]:
        """让机器人连续行动，直到轮到英雄或该手结束。"""

        self._ensure_open()
        hand = self._require_hand()
        records: list[ActionRecord] = []
        steps = 0
        while (
            not hand.is_complete
            and hand.current_actor_id is not None
            and hand.current_actor_id != self.hero_id
        ):
            steps += 1
            if steps > self._MAX_BOT_ACTIONS_PER_ADVANCE:
                raise RuntimeError("机器人行动超过安全上限")
            player_id = hand.current_actor_id
            context = PolicyContext.from_hand(hand, player_id)
            decision = self._forced_open_decision(context)
            if decision is None:
                decision = OpponentPolicy.choose(
                    context,
                    self._bot_profiles[player_id],
                    self._current_policy_seed if self._current_policy_seed is not None else 0,
                )
            record = hand.act(player_id, decision.action, decision.amount)
            records.append(record)
            if hand.preflop_raise_count > 0:
                self._forced_open_pending = False

        self._finalize_if_complete()
        return tuple(records)

    def _forced_open_decision(self, context: PolicyContext) -> BotDecision | None:
        """定向场景只强制一次合法开池，其余动作仍交给动态策略。"""

        if not (
            self._forced_open_pending
            and context.street is Street.PREFLOP
            and context.preflop_raise_count == 0
            and context.legal.can_raise
            and context.min_raise_to is not None
        ):
            return None
        target = min(
            context.max_to,
            max(context.min_raise_to, context.big_blind * 3),
        )
        return BotDecision(ActionType.RAISE, target, "训练场景的合法开池")

    def hero_action(
        self, action: ActionType | str, amount: int | None = None
    ) -> DecisionReview | tuple[DecisionReview, ...] | None:
        """应用英雄动作，并按模式控制反馈时机。"""

        self._ensure_open()
        hand = self._require_hand()
        if hand.is_complete:
            raise InvalidAction("牌局已经结束")
        if hand.current_actor_id != self.hero_id:
            raise InvalidAction("当前未轮到英雄行动")

        context = capture_context(hand, self.hero_id)
        record = hand.act(self.hero_id, action, amount)
        review = review_decision(
            context,
            record,
            trials=self.config.coach_trials,
        )
        self._current_reviews_internal.append(review)
        if hand.preflop_raise_count > 0:
            self._forced_open_pending = False
        self.advance_bots()

        if self.mode is TrainingMode.TEACHING:
            return review
        if hand.is_complete:
            return self._last_hand_reviews
        return None

    def _finalize_if_complete(self) -> None:
        hand = self.current_hand
        if (
            hand is None
            or not hand.is_complete
            or hand.hand_id in self._finalized_hand_ids
        ):
            return
        hand.assert_chip_conservation()
        replay = ReplayBundle.from_hand(hand)
        self._store.save_hand(
            hand,
            replay.to_json(),
            self.mode,
            policy_seed=self._current_policy_seed,
        )
        for review in self._current_reviews_internal:
            self._store.save_decision_review(hand.hand_id, review.sequence, review.as_dict())

        statistics = project_hand_statistics(
            hand,
            self.hero_id,
            reason_codes=self._current_reviews_internal,
        )
        self._hand_statistics.append(statistics)
        self._last_hand_reviews = tuple(self._current_reviews_internal)
        self._all_reviews.extend(self._current_reviews_internal)
        self._stacks.update(
            {player_id: player.stack for player_id, player in hand.players.items()}
        )
        self._completed_hand_count += 1
        self._finalized_hand_ids.add(hand.hand_id)
        if self._completed_hand_count % PROFILE_INTERVAL == 0:
            self._create_checkpoint()

    def _create_checkpoint(self) -> None:
        through_hand_no = self._completed_hand_count
        position_rows = self.position_statistics
        for position, row in position_rows.items():
            self._store.save_metric_snapshot(
                self.session_id,
                through_hand_no,
                row.as_dict(),
                position=position,
                sample_size=row.hands,
            )
        profile = generate_player_profile(
            through_hand_no,
            position_rows,
            self._all_reviews,
        )
        if profile is None:  # pragma: no cover - 仅防御被替换的 interval 实现
            return
        self._profile_history.append(profile)
        leak_snapshot_id = self._store.save_leak_snapshot(
            self.session_id,
            through_hand_no,
            profile.as_dict(),
        )
        plan = self._scheduler.build_plan(profile)
        self._store.save_scenario_plan(
            self.session_id,
            plan.effective_from_hand_no,
            plan.as_dict(),
            hand_count=plan.hand_count,
            source_leak_snapshot_id=leak_snapshot_id,
        )
        self._current_plan = plan

    def public_state(self) -> dict[str, Any]:
        """返回 UI 安全快照；绝不包含动态对手的隐藏参数或底牌。"""

        hand = self.current_hand
        feedback = self.current_reviews
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "mode_zh": self.mode.label_zh,
            "hero_id": self.hero_id,
            "stakes": {
                "small_blind": self.config.small_blind,
                "big_blind": self.config.big_blind,
                "buy_in": self.config.buy_in,
                "buy_in_bb": self.config.buy_in_big_blinds,
                "chips_per_yuan": self.config.chips_per_yuan,
            },
            "completed_hand_count": self.completed_hand_count,
            "current_hand": (
                hand.public_view(self.hero_id) if hand is not None else None
            ),
            # 定向场景名称可能暗示英雄当前牌型；行动中不向测试界面泄露，
            # 仅在该手结束后供牌后复盘使用。
            "current_scenario": (
                self._current_scenario.as_dict()
                if self._current_scenario is not None
                and hand is not None
                and hand.is_complete
                else None
            ),
            "feedback": [review.as_dict() for review in feedback],
            "position_statistics": serialize_position_statistics(
                self.position_statistics
            ),
            "latest_profile": (
                self.latest_profile.as_dict() if self.latest_profile else None
            ),
            "scenario_plan": self.current_plan.as_dict(),
            "stacks": dict(self._stacks),
            "last_top_ups": dict(self._last_top_ups),
        }

    def _require_hand(self) -> HoldemHand:
        if self.current_hand is None:
            raise RuntimeError("请先开始一手牌")
        return self.current_hand

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("训练场次已经关闭")

    def close(self) -> None:
        if self._closed:
            return
        if self.current_hand is not None:
            self._finalize_if_complete()
        self._store.end_session(self.session_id)
        if self._owns_store:
            self._store.close()
        self._closed = True

    def __enter__(self) -> "TrainingSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = ["SessionConfig", "TrainingMode", "TrainingSession"]

