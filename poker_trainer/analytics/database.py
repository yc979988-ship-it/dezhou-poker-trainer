"""SQLite persistence for sessions, deterministic hand replays and analytics.

The store deliberately keeps persistence independent from Streamlit.  A single
``SQLiteStore`` owns one connection, enables foreign keys, and wraps every
multi-row write in an explicit transaction.  Schema changes are applied in
order and recorded in both ``schema_version`` and SQLite's ``user_version``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from poker_trainer.engine.hand import HoldemHand


CURRENT_SCHEMA_VERSION = 1


_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY CHECK (version > 0),
        applied_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL CHECK (mode IN ('test', 'teaching')),
        hero_player_id TEXT NOT NULL,
        session_seed INTEGER,
        effective_stack INTEGER NOT NULL CHECK (effective_stack >= 0),
        auto_top_up INTEGER NOT NULL CHECK (auto_top_up IN (0, 1)),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS hands (
        hand_id TEXT PRIMARY KEY,
        session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
        hand_no INTEGER NOT NULL CHECK (hand_no > 0),
        seed INTEGER NOT NULL,
        policy_seed INTEGER,
        mode TEXT NOT NULL CHECK (mode IN ('test', 'teaching')),
        scenario_id TEXT,
        small_blind INTEGER NOT NULL CHECK (small_blind > 0),
        big_blind INTEGER NOT NULL CHECK (big_blind > small_blind),
        engine_version TEXT NOT NULL,
        rules_version TEXT NOT NULL,
        final_street TEXT NOT NULL,
        completed INTEGER NOT NULL CHECK (completed IN (0, 1)),
        result_reason TEXT,
        board_json TEXT NOT NULL,
        result_json TEXT,
        replay_json TEXT NOT NULL,
        saved_at TEXT NOT NULL,
        UNIQUE (session_id, hand_no)
    );

    CREATE TABLE IF NOT EXISTS hand_players (
        hand_id TEXT NOT NULL REFERENCES hands(hand_id) ON DELETE CASCADE,
        player_id TEXT NOT NULL,
        name TEXT NOT NULL,
        position TEXT NOT NULL CHECK (position IN ('UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB')),
        starting_stack INTEGER NOT NULL CHECK (starting_stack >= 0),
        ending_stack INTEGER NOT NULL CHECK (ending_stack >= 0),
        total_commitment INTEGER NOT NULL CHECK (total_commitment >= 0),
        payout INTEGER NOT NULL CHECK (payout >= 0),
        folded INTEGER NOT NULL CHECK (folded IN (0, 1)),
        all_in INTEGER NOT NULL CHECK (all_in IN (0, 1)),
        hole_cards_json TEXT NOT NULL,
        hole_cards_hidden INTEGER NOT NULL CHECK (hole_cards_hidden IN (0, 1)),
        PRIMARY KEY (hand_id, player_id)
    );

    CREATE TABLE IF NOT EXISTS actions (
        hand_id TEXT NOT NULL REFERENCES hands(hand_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL CHECK (sequence >= 0),
        street TEXT NOT NULL,
        player_id TEXT NOT NULL,
        position TEXT NOT NULL CHECK (position IN ('UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB')),
        action_type TEXT NOT NULL,
        requested_amount INTEGER,
        paid INTEGER NOT NULL,
        bet_to INTEGER NOT NULL,
        pot_before INTEGER NOT NULL,
        pot_after INTEGER NOT NULL,
        to_call_before INTEGER NOT NULL,
        current_bet_before INTEGER NOT NULL,
        current_bet_after INTEGER NOT NULL,
        min_raise_to_before INTEGER,
        is_all_in INTEGER NOT NULL CHECK (is_all_in IN (0, 1)),
        is_full_raise INTEGER NOT NULL CHECK (is_full_raise IN (0, 1)),
        forced INTEGER NOT NULL CHECK (forced IN (0, 1)),
        decision_snapshot_json TEXT,
        PRIMARY KEY (hand_id, sequence),
        FOREIGN KEY (hand_id, player_id)
            REFERENCES hand_players(hand_id, player_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS decision_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hand_id TEXT NOT NULL,
        action_sequence INTEGER NOT NULL,
        player_id TEXT NOT NULL,
        rating TEXT NOT NULL,
        reason TEXT NOT NULL,
        recommended_action TEXT,
        pot_odds REAL,
        equity REAL,
        outs INTEGER,
        hit_probability REAL,
        review_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (hand_id, action_sequence),
        FOREIGN KEY (hand_id, action_sequence)
            REFERENCES actions(hand_id, sequence) ON DELETE CASCADE,
        FOREIGN KEY (hand_id, player_id)
            REFERENCES hand_players(hand_id, player_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS metric_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        through_hand_no INTEGER NOT NULL CHECK (through_hand_no >= 0),
        sample_size INTEGER NOT NULL CHECK (sample_size >= 0),
        position TEXT CHECK (position IS NULL OR position IN ('UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB')),
        metrics_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS leak_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        through_hand_no INTEGER NOT NULL CHECK (through_hand_no >= 0),
        leaks_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS scenario_plans (
        plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        source_leak_snapshot_id INTEGER REFERENCES leak_snapshots(snapshot_id) ON DELETE SET NULL,
        effective_from_hand_no INTEGER NOT NULL CHECK (effective_from_hand_no > 0),
        hand_count INTEGER NOT NULL CHECK (hand_count > 0),
        plan_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_hands_session_hand_no
        ON hands(session_id, hand_no);
    CREATE INDEX IF NOT EXISTS idx_actions_hand_player
        ON actions(hand_id, player_id, sequence);
    CREATE INDEX IF NOT EXISTS idx_reviews_hand
        ON decision_reviews(hand_id, action_sequence);
    CREATE INDEX IF NOT EXISTS idx_metrics_session_hand
        ON metric_snapshots(session_id, through_hand_no, snapshot_id);
    CREATE INDEX IF NOT EXISTS idx_leaks_session_hand
        ON leak_snapshots(session_id, through_hand_no, snapshot_id);
    CREATE INDEX IF NOT EXISTS idx_plans_session_hand
        ON scenario_plans(session_id, effective_from_hand_no, plan_id);
    """,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "__str__") and value.__class__.__name__ == "Card":
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    )


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是映射或 dataclass")
    return dict(value)


def _normalise_mode(mode: Any) -> str:
    raw = getattr(mode, "value", mode)
    aliases = {
        "test": "test",
        "测试": "test",
        "测试模式": "test",
        "teaching": "teaching",
        "teach": "teaching",
        "教学": "teaching",
        "教学模式": "teaching",
    }
    try:
        return aliases[str(raw).strip().lower()]
    except KeyError as exc:
        raise ValueError("mode 必须是 test/测试模式 或 teaching/教学模式") from exc


class SQLiteStore:
    """Transactional SQLite repository used by the offline trainer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if str(self.path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the configured connection for read-only diagnostics/tests."""

        return self._connection

    @property
    def schema_version(self) -> int:
        row = self._connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a write transaction, rolling the whole operation back on error."""

        with self._lock:
            if self._connection.in_transaction:
                raise RuntimeError("不支持嵌套事务")
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _migrate(self) -> None:
        current = self.schema_version
        if current > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"数据库版本 {current} 高于程序支持版本 {CURRENT_SCHEMA_VERSION}"
            )
        for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            migration = _MIGRATIONS.get(version)
            if migration is None:
                raise RuntimeError(f"缺少数据库迁移脚本: {version}")
            # executescript manages its own transaction, so transaction control is
            # included in the script to keep DDL + version markers atomic.
            escaped_now = _utc_now().replace("'", "''")
            script = (
                "BEGIN IMMEDIATE;\n"
                f"{migration}\n"
                "INSERT INTO schema_version(version, applied_at) "
                f"VALUES ({version}, '{escaped_now}');\n"
                f"PRAGMA user_version = {version};\n"
                "COMMIT;"
            )
            try:
                with self._lock:
                    self._connection.executescript(script)
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def create_session(
        self,
        session_id: str | None = None,
        *,
        mode: Any = "test",
        hero_player_id: str = "hero",
        seed: int | None = None,
        effective_stack: int = 4000,
        auto_top_up: bool = True,
        started_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Create a training session and return its stable identifier."""

        session_id = session_id or f"session-{uuid4().hex}"
        if not session_id.strip():
            raise ValueError("session_id 不能为空")
        if not hero_player_id.strip():
            raise ValueError("hero_player_id 不能为空")
        if isinstance(effective_stack, bool) or effective_stack < 0:
            raise ValueError("effective_stack 必须是非负整数")
        mode_value = _normalise_mode(mode)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, mode, hero_player_id, session_seed,
                    effective_stack, auto_top_up, started_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    mode_value,
                    hero_player_id,
                    seed,
                    int(effective_stack),
                    int(bool(auto_top_up)),
                    started_at or _utc_now(),
                    _json_dumps(dict(metadata or {})),
                ),
            )
        return session_id

    def end_session(self, session_id: str, *, ended_at: str | None = None) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (ended_at or _utc_now(), session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"训练场次不存在: {session_id}")

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["auto_top_up"] = bool(result["auto_top_up"])
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def save_hand(
        self,
        hand: HoldemHand,
        replay_json: str,
        mode: Any,
        policy_seed: int | None = None,
    ) -> str:
        """Atomically save the hand header, players and every action.

        Saving an existing ``hand_id`` replaces its prior snapshot.  This makes
        it safe for the UI to persist an in-progress hand and save it again once
        complete without ever leaving mixed old/new child rows.
        """

        if not isinstance(hand, HoldemHand):
            raise TypeError("hand 必须是 HoldemHand")
        try:
            replay_data = json.loads(replay_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("replay_json 不是有效 JSON") from exc
        if not isinstance(replay_data, dict):
            raise ValueError("replay_json 顶层必须是对象")
        if replay_data.get("hand_id") != hand.hand_id:
            raise ValueError("replay_json 与 hand 的 hand_id 不一致")

        mode_value = _normalise_mode(mode)
        result_json = self._serialise_result(hand)
        saved_at = _utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO hands (
                    hand_id, session_id, hand_no, seed, policy_seed, mode,
                    scenario_id, small_blind, big_blind, engine_version,
                    rules_version, final_street, completed, result_reason,
                    board_json, result_json, replay_json, saved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hand_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    hand_no = excluded.hand_no,
                    seed = excluded.seed,
                    policy_seed = excluded.policy_seed,
                    mode = excluded.mode,
                    scenario_id = excluded.scenario_id,
                    small_blind = excluded.small_blind,
                    big_blind = excluded.big_blind,
                    engine_version = excluded.engine_version,
                    rules_version = excluded.rules_version,
                    final_street = excluded.final_street,
                    completed = excluded.completed,
                    result_reason = excluded.result_reason,
                    board_json = excluded.board_json,
                    result_json = excluded.result_json,
                    replay_json = excluded.replay_json,
                    saved_at = excluded.saved_at
                """,
                (
                    hand.hand_id,
                    hand.session_id,
                    hand.hand_no,
                    hand.seed,
                    policy_seed,
                    mode_value,
                    hand.scenario_id,
                    hand.small_blind,
                    hand.big_blind,
                    hand.engine_version,
                    hand.rules_version,
                    hand.street.value,
                    int(hand.is_complete),
                    hand.result.reason if hand.result else None,
                    _json_dumps([str(card) for card in hand.board]),
                    result_json,
                    replay_json,
                    saved_at,
                ),
            )
            # Child data is a point-in-time snapshot.  Reviews are intentionally
            # removed by the cascading delete if a hand itself is resaved.
            connection.execute("DELETE FROM actions WHERE hand_id = ?", (hand.hand_id,))
            connection.execute("DELETE FROM hand_players WHERE hand_id = ?", (hand.hand_id,))

            showdown = bool(hand.result and hand.result.reason == "showdown")
            for player in hand.players.values():
                cards_hidden = not (showdown and not player.folded)
                connection.execute(
                    """
                    INSERT INTO hand_players (
                        hand_id, player_id, name, position, starting_stack,
                        ending_stack, total_commitment, payout, folded, all_in,
                        hole_cards_json, hole_cards_hidden
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hand.hand_id,
                        player.player_id,
                        player.name,
                        player.position.value,
                        player.starting_stack,
                        player.stack,
                        player.total_commitment,
                        player.payout,
                        int(player.folded),
                        int(player.all_in),
                        _json_dumps([str(card) for card in player.hole_cards]),
                        int(cards_hidden),
                    ),
                )

            for action in hand.history:
                snapshot = hand.decision_snapshots.get(action.sequence)
                connection.execute(
                    """
                    INSERT INTO actions (
                        hand_id, sequence, street, player_id, position,
                        action_type, requested_amount, paid, bet_to, pot_before,
                        pot_after, to_call_before, current_bet_before,
                        current_bet_after, min_raise_to_before, is_all_in,
                        is_full_raise, forced, decision_snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hand.hand_id,
                        action.sequence,
                        action.street.value,
                        action.player_id,
                        action.position.value,
                        action.action.value,
                        action.requested_amount,
                        action.paid,
                        action.bet_to,
                        action.pot_before,
                        action.pot_after,
                        action.to_call_before,
                        action.current_bet_before,
                        action.current_bet_after,
                        action.min_raise_to_before,
                        int(action.is_all_in),
                        int(action.is_full_raise),
                        int(action.forced),
                        _json_dumps(snapshot) if snapshot is not None else None,
                    ),
                )
        return hand.hand_id

    @staticmethod
    def _serialise_result(hand: HoldemHand) -> str | None:
        if hand.result is None:
            return None
        return _json_dumps(
            {
                "reason": hand.result.reason,
                "board": [str(card) for card in hand.result.board],
                "pots": [
                    {
                        "amount": pot.amount,
                        "cap": pot.cap,
                        "contributors": list(pot.contributors),
                        "eligible": list(pot.eligible),
                    }
                    for pot in hand.result.pots
                ],
                "payouts": hand.result.payouts,
                "hand_ranks": hand.result.hand_ranks,
            }
        )

    def load_replay_json(self, hand_id: str) -> str:
        row = self._connection.execute(
            "SELECT replay_json FROM hands WHERE hand_id = ?", (hand_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"手牌不存在: {hand_id}")
        return str(row["replay_json"])

    def list_hands(self, session_id: str) -> list[dict[str, Any]]:
        """List safe hand summaries; replay text and hole cards are excluded."""

        rows = self._connection.execute(
            """
            SELECT hand_id, session_id, hand_no, seed, policy_seed, mode,
                   scenario_id, small_blind, big_blind, engine_version,
                   rules_version, final_street, completed, result_reason,
                   board_json, result_json, saved_at
            FROM hands
            WHERE session_id = ?
            ORDER BY hand_no, hand_id
            """,
            (session_id,),
        ).fetchall()
        return [self._decode_hand_row(row) for row in rows]

    def load_hand(
        self, hand_id: str, *, reveal_hole_cards: bool = False
    ) -> dict[str, Any] | None:
        """Load a persisted hand with players and actions.

        Hidden cards are returned as ``None`` unless ``reveal_hole_cards`` is
        explicitly true.  The deterministic replay remains available through
        ``load_replay_json`` for trusted replay code.
        """

        row = self._connection.execute(
            """
            SELECT hand_id, session_id, hand_no, seed, policy_seed, mode,
                   scenario_id, small_blind, big_blind, engine_version,
                   rules_version, final_street, completed, result_reason,
                   board_json, result_json, saved_at
            FROM hands WHERE hand_id = ?
            """,
            (hand_id,),
        ).fetchone()
        if row is None:
            return None
        hand = self._decode_hand_row(row)
        player_rows = self._connection.execute(
            """
            SELECT * FROM hand_players
            WHERE hand_id = ?
            ORDER BY CASE position
                WHEN 'UTG' THEN 1 WHEN 'HJ' THEN 2 WHEN 'CO' THEN 3
                WHEN 'BTN' THEN 4 WHEN 'SB' THEN 5 WHEN 'BB' THEN 6 END
            """,
            (hand_id,),
        ).fetchall()
        players: list[dict[str, Any]] = []
        for player_row in player_rows:
            player = dict(player_row)
            hidden = bool(player["hole_cards_hidden"])
            cards = json.loads(player.pop("hole_cards_json"))
            player["hole_cards_hidden"] = hidden
            player["hole_cards"] = cards if reveal_hole_cards or not hidden else None
            for key in ("folded", "all_in"):
                player[key] = bool(player[key])
            players.append(player)
        action_rows = self._connection.execute(
            "SELECT * FROM actions WHERE hand_id = ? ORDER BY sequence",
            (hand_id,),
        ).fetchall()
        actions: list[dict[str, Any]] = []
        for action_row in action_rows:
            action = dict(action_row)
            snapshot_json = action.pop("decision_snapshot_json")
            action["decision_snapshot"] = (
                json.loads(snapshot_json) if snapshot_json is not None else None
            )
            for key in ("is_all_in", "is_full_raise", "forced"):
                action[key] = bool(action[key])
            actions.append(action)
        hand["players"] = players
        hand["actions"] = actions
        return hand

    @staticmethod
    def _decode_hand_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["completed"] = bool(result["completed"])
        result["board"] = json.loads(result.pop("board_json"))
        result_json = result.pop("result_json")
        result["result"] = json.loads(result_json) if result_json is not None else None
        return result

    def save_decision_review(
        self,
        hand_id: str,
        action_sequence: int,
        review: Mapping[str, Any] | Any,
    ) -> int:
        """Insert or replace the coach review for one player decision."""

        data = _as_mapping(review, label="review")
        rating = data.get("rating") or data.get("grade")
        reason = data.get("reason")
        if not rating or not reason:
            raise ValueError("review 必须包含 rating 和 reason")
        action = self._connection.execute(
            """
            SELECT player_id FROM actions
            WHERE hand_id = ? AND sequence = ?
            """,
            (hand_id, action_sequence),
        ).fetchone()
        if action is None:
            raise KeyError(f"动作不存在: {hand_id}#{action_sequence}")
        player_id = str(data.get("player_id") or action["player_id"])
        if player_id != action["player_id"]:
            raise ValueError("review.player_id 与动作玩家不一致")
        created_at = str(data.get("created_at") or _utc_now())
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO decision_reviews (
                    hand_id, action_sequence, player_id, rating, reason,
                    recommended_action, pot_odds, equity, outs,
                    hit_probability, review_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hand_id, action_sequence) DO UPDATE SET
                    player_id = excluded.player_id,
                    rating = excluded.rating,
                    reason = excluded.reason,
                    recommended_action = excluded.recommended_action,
                    pot_odds = excluded.pot_odds,
                    equity = excluded.equity,
                    outs = excluded.outs,
                    hit_probability = excluded.hit_probability,
                    review_json = excluded.review_json,
                    created_at = excluded.created_at
                """,
                (
                    hand_id,
                    action_sequence,
                    player_id,
                    str(rating),
                    str(reason),
                    data.get("recommended_action"),
                    data.get("pot_odds"),
                    data.get("equity"),
                    data.get("outs"),
                    data.get("hit_probability"),
                    _json_dumps(data),
                    created_at,
                ),
            )
            row = connection.execute(
                """
                SELECT review_id FROM decision_reviews
                WHERE hand_id = ? AND action_sequence = ?
                """,
                (hand_id, action_sequence),
            ).fetchone()
        return int(row["review_id"])

    def load_decision_reviews(self, hand_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT * FROM decision_reviews
            WHERE hand_id = ? ORDER BY action_sequence
            """,
            (hand_id,),
        ).fetchall()
        return [self._decode_payload_row(row, "review_json", "review") for row in rows]

    def save_metric_snapshot(
        self,
        session_id: str,
        through_hand_no: int,
        metrics: Mapping[str, Any] | Any,
        *,
        position: Any | None = None,
        sample_size: int | None = None,
    ) -> int:
        data = _as_mapping(metrics, label="metrics")
        position_value = getattr(position, "value", position)
        if position_value is not None:
            position_value = str(position_value)
        sample = through_hand_no if sample_size is None else sample_size
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO metric_snapshots (
                    session_id, through_hand_no, sample_size, position,
                    metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    through_hand_no,
                    sample,
                    position_value,
                    _json_dumps(data),
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def load_metric_snapshots(
        self, session_id: str, *, position: Any | None = None
    ) -> list[dict[str, Any]]:
        position_value = getattr(position, "value", position)
        if position is None:
            rows = self._connection.execute(
                """
                SELECT * FROM metric_snapshots
                WHERE session_id = ?
                ORDER BY through_hand_no, snapshot_id
                """,
                (session_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT * FROM metric_snapshots
                WHERE session_id = ? AND position = ?
                ORDER BY through_hand_no, snapshot_id
                """,
                (session_id, str(position_value)),
            ).fetchall()
        return [self._decode_payload_row(row, "metrics_json", "metrics") for row in rows]

    def save_leak_snapshot(
        self,
        session_id: str,
        through_hand_no: int,
        leaks: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    ) -> int:
        payload: Any = dict(leaks) if isinstance(leaks, Mapping) else list(leaks)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO leak_snapshots (
                    session_id, through_hand_no, leaks_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (session_id, through_hand_no, _json_dumps(payload), _utc_now()),
            )
            return int(cursor.lastrowid)

    def load_leak_snapshots(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT * FROM leak_snapshots
            WHERE session_id = ? ORDER BY through_hand_no, snapshot_id
            """,
            (session_id,),
        ).fetchall()
        return [self._decode_payload_row(row, "leaks_json", "leaks") for row in rows]

    def save_scenario_plan(
        self,
        session_id: str,
        effective_from_hand_no: int,
        plan: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        hand_count: int = 20,
        source_leak_snapshot_id: int | None = None,
    ) -> int:
        payload: Any = dict(plan) if isinstance(plan, Mapping) else list(plan)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scenario_plans (
                    session_id, source_leak_snapshot_id,
                    effective_from_hand_no, hand_count, plan_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    source_leak_snapshot_id,
                    effective_from_hand_no,
                    hand_count,
                    _json_dumps(payload),
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def load_scenario_plans(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT * FROM scenario_plans
            WHERE session_id = ? ORDER BY effective_from_hand_no, plan_id
            """,
            (session_id,),
        ).fetchall()
        return [self._decode_payload_row(row, "plan_json", "plan") for row in rows]

    @staticmethod
    def _decode_payload_row(
        row: sqlite3.Row, json_column: str, output_key: str
    ) -> dict[str, Any]:
        result = dict(row)
        result[output_key] = json.loads(result.pop(json_column))
        return result


__all__ = ["CURRENT_SCHEMA_VERSION", "SQLiteStore"]

