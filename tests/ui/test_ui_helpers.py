from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest

import poker_trainer.ui.app as ui_app
from poker_trainer.analytics.database import SQLiteStore
from poker_trainer.analytics.statistics import (
    METRIC_ORDER,
    MetricName,
    MetricTally,
    PositionStatistics,
    aggregate_by_position,
)
from poker_trainer.engine.cards import parse_cards
from poker_trainer.engine.evaluator import evaluate
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import (
    ActionRecord,
    ActionType,
    HandResult,
    LegalActions,
    PlayerState,
    Position,
    Pot,
    Street,
)
from poker_trainer.engine.replay import ReplayBundle
from poker_trainer.opponents.profiles import OpponentHabits
from poker_trainer.ui.app import (
    HABIT_LABELS_ZH,
    OPPONENT_HABITS_FORMAT,
    POSITION_ORDER,
    action_history_rows,
    action_timeline_html,
    bet_to_range,
    cards_html,
    format_card,
    format_cards,
    format_chips,
    format_hand_rank,
    format_metric,
    full_position_name,
    habit_level_label,
    hand_public_view,
    hero_net_result,
    legal_action_controls,
    list_saved_hands,
    load_replay_bundle,
    load_saved_reviews,
    metric_detail_rows,
    normalize_reviews,
    opponent_habits_from_json,
    opponent_habits_to_json,
    opponent_library_from_json,
    opponent_library_to_json,
    rebuild_replay,
    replay_reviews_through_sequence,
    review_cards_html,
    result_pot_rows,
    seat_grid_html,
    settlement_summary,
    showdown_rank_rows,
    statistics_table_rows,
)
from poker_trainer.ui.styles import MOBILE_CSS


def _play_passively(hand: HoldemHand) -> None:
    while not hand.is_complete:
        actor = hand.current_actor_id
        assert actor is not None
        legal = hand.legal_actions(actor)
        if legal.can_check:
            hand.act(actor, ActionType.CHECK)
        elif legal.can_call:
            hand.act(actor, ActionType.CALL)
        else:
            hand.act(actor, ActionType.FOLD)


def _legal(**updates: object) -> LegalActions:
    values: dict[str, object] = {
        "player_id": "hero",
        "to_call": 40,
        "call_amount": 40,
        "pot_before": 100,
        "min_bet_to": None,
        "min_raise_to": 120,
        "max_to": 4_000,
        "can_fold": True,
        "can_check": False,
        "can_call": True,
        "can_bet": False,
        "can_raise": True,
        "can_all_in": True,
        "raise_reopened": True,
    }
    values.update(updates)
    return LegalActions(**values)  # type: ignore[arg-type]


def _friend_habits(
    number: int,
    *,
    opponent_id: str | None = None,
    nickname: str | None = None,
    **levels: int,
) -> OpponentHabits:
    values: dict[str, object] = {
        "opponent_id": opponent_id or f"friend-{number}",
        "nickname": nickname or f"牌友{number}",
        "entry_frequency": 3,
        "limp_frequency": 3,
        "preflop_aggression": 3,
        "calling_tendency": 3,
        "postflop_aggression": 3,
        "mistake_frequency": 2,
    }
    values.update(levels)
    return OpponentHabits(**values)  # type: ignore[arg-type]


def _habit_json_payload(
    items: list[dict[str, object]],
    *,
    version: object = 1,
    format_name: object = OPPONENT_HABITS_FORMAT,
) -> str:
    return json.dumps(
        {"format": format_name, "version": version, "opponents": items},
        ensure_ascii=False,
    )


def _reported_showdown_hand() -> object:
    """还原截图：SB 的 TT99Q 应击败英雄的 TT66Q。"""

    board = tuple(parse_cards("Qd Ts 9d 6d 6h"))
    players = {
        "UTG": PlayerState(
            "UTG",
            "对手1",
            Position.UTG,
            stack=4_970,
            starting_stack=4_000,
            hole_cards=parse_cards("As 9h"),
        ),
        "SB": PlayerState(
            "SB",
            "对手5",
            Position.SB,
            stack=18_730,
            starting_stack=18_170,
            hole_cards=parse_cards("9c Tc"),
            payout=1_120,
        ),
        "BB": PlayerState(
            "BB",
            "你",
            Position.BB,
            stack=3_640,
            starting_stack=4_000,
            hole_cards=parse_cards("Td 4s"),
        ),
    }
    ranks = {
        player_id: evaluate((*player.hole_cards, *board))
        for player_id, player in players.items()
    }
    result = HandResult(
        reason="showdown",
        board=board,
        pots=(
            Pot(
                amount=1_120,
                cap=360,
                contributors=("UTG", "SB", "BB"),
                eligible=("UTG", "SB", "BB"),
            ),
        ),
        payouts={"UTG": 0, "SB": 1_120, "BB": 0},
        # 持久化结果只保存牌型名称；展示层需用公开牌面重建完整比较信息。
        hand_ranks={player_id: rank.name_zh for player_id, rank in ranks.items()},
    )
    return type(
        "ReportedShowdown",
        (),
        {"players": players, "board": list(board), "result": result},
    )()


def _split_pot_winners_hand() -> object:
    """三人争主池、两人争边池，且两个池由不同玩家获胜。"""

    board = tuple(parse_cards("2c 3d 7h 8s Kc"))
    players = {
        "UTG": PlayerState(
            "UTG",
            "短码玩家",
            Position.UTG,
            stack=900,
            starting_stack=900,
            hole_cards=parse_cards("Kh Kd"),
            all_in=True,
            payout=900,
        ),
        "SB": PlayerState(
            "SB",
            "中码玩家",
            Position.SB,
            stack=800,
            starting_stack=1_700,
            hole_cards=parse_cards("8c 8d"),
            all_in=True,
            payout=800,
        ),
        "BB": PlayerState(
            "BB",
            "你",
            Position.BB,
            stack=0,
            starting_stack=1_700,
            hole_cards=parse_cards("As Qs"),
            all_in=True,
        ),
    }
    ranks = {
        player_id: evaluate((*player.hole_cards, *board))
        for player_id, player in players.items()
    }
    result = HandResult(
        reason="showdown",
        board=board,
        pots=(
            Pot(
                amount=900,
                cap=300,
                contributors=("UTG", "SB", "BB"),
                eligible=("UTG", "SB", "BB"),
            ),
            Pot(
                amount=800,
                cap=700,
                contributors=("SB", "BB"),
                eligible=("SB", "BB"),
            ),
        ),
        payouts={"UTG": 900, "SB": 800, "BB": 0},
        hand_ranks={player_id: rank.name_zh for player_id, rank in ranks.items()},
    )
    return type(
        "SplitPotWinners",
        (),
        {"players": players, "board": list(board), "result": result},
    )()


def test_ui_module_import_does_not_require_streamlit() -> None:
    sys.modules.pop("streamlit", None)
    module = importlib.reload(sys.modules["poker_trainer.ui.app"])
    assert module.format_chips(4_000).startswith("4,000")
    assert "streamlit" not in sys.modules


def test_position_names_keep_all_six_abbreviations_and_chinese_meanings() -> None:
    assert POSITION_ORDER == (
        Position.UTG,
        Position.HJ,
        Position.CO,
        Position.BTN,
        Position.SB,
        Position.BB,
    )
    assert [full_position_name(position) for position in POSITION_ORDER] == [
        "UTG（前位）",
        "HJ（中位）",
        "CO（后位，按钮前一位）",
        "BTN（按钮位，位置最好）",
        "SB（小盲）",
        "BB（大盲）",
    ]


def test_habit_controls_use_plain_chinese_without_hidden_model_terms() -> None:
    fields = (
        "entry_frequency",
        "limp_frequency",
        "preflop_aggression",
        "calling_tendency",
        "postflop_aggression",
        "mistake_frequency",
    )
    assert set(HABIT_LABELS_ZH) == set(fields)

    visible_text = " ".join(str(HABIT_LABELS_ZH[field]) for field in fields)
    for field in fields:
        labels = [habit_level_label(field, level) for level in range(1, 6)]
        assert len(set(labels)) == 5
        assert all(any("\u4e00" <= char <= "\u9fff" for char in label) for label in labels)
        visible_text += " " + " ".join(labels)

    forbidden = (
        "vpip",
        "pfr",
        "three_bet",
        "fold_tendency",
        "aggression_factor",
    )
    assert not any(term in visible_text.lower() for term in forbidden)
    with pytest.raises((KeyError, ValueError)):
        habit_level_label("unknown_habit", 3)
    with pytest.raises((TypeError, ValueError)):
        habit_level_label("entry_frequency", 0)


def test_opponent_habit_json_is_versioned_and_round_trips_all_six_levels() -> None:
    friends = (
        _friend_habits(
            1,
            nickname="阿杰",
            entry_frequency=5,
            limp_frequency=4,
            preflop_aggression=2,
            calling_tendency=5,
            postflop_aggression=1,
            mistake_frequency=3,
        ),
        _friend_habits(
            2,
            nickname="老周",
            entry_frequency=2,
            limp_frequency=1,
            preflop_aggression=5,
            calling_tendency=2,
            postflop_aggression=4,
            mistake_frequency=1,
        ),
    )

    encoded = opponent_habits_to_json(friends)
    payload = json.loads(encoded)
    assert payload["format"] == OPPONENT_HABITS_FORMAT
    assert payload["version"] == 1
    assert len(payload["opponents"]) == 2
    assert set(payload["opponents"][0]) == {
        "opponent_id",
        "nickname",
        "entry_frequency",
        "limp_frequency",
        "preflop_aggression",
        "calling_tendency",
        "postflop_aggression",
        "mistake_frequency",
    }
    assert tuple(opponent_habits_from_json(encoded)) == friends

    hidden_terms = (
        "vpip",
        "pfr",
        "three_bet",
        "fold_tendency",
        "aggression_factor",
    )
    assert not any(term in encoded.lower() for term in hidden_terms)


def test_opponent_habit_json_limits_roster_to_five_on_export_and_import() -> None:
    six_friends = tuple(_friend_habits(number) for number in range(1, 7))
    with pytest.raises(ValueError, match="5|五|最多"):
        opponent_habits_to_json(six_friends)

    payload = _habit_json_payload([friend.as_dict() for friend in six_friends])
    with pytest.raises(ValueError, match="5|五|最多"):
        opponent_habits_from_json(payload)


def test_opponent_library_json_allows_twelve_and_keeps_v1_compatibility() -> None:
    friends = tuple(_friend_habits(number) for number in range(1, 8))
    encoded = opponent_library_to_json(friends)
    payload = json.loads(encoded)
    assert payload["version"] == 2
    assert len(opponent_library_from_json(encoded)) == 7
    with pytest.raises(ValueError):
        opponent_habits_from_json(encoded)

    too_many = tuple(_friend_habits(number) for number in range(1, 14))
    with pytest.raises(ValueError, match="12|十二|最多"):
        opponent_library_to_json(too_many)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("opponent_id", ""),
        ("nickname", "  "),
        ("entry_frequency", 0),
        ("limp_frequency", 6),
        ("preflop_aggression", 2.5),
        ("calling_tendency", True),
        ("postflop_aggression", "3"),
        ("mistake_frequency", -1),
    ),
)
def test_opponent_habit_json_rejects_empty_identity_and_bad_levels(
    field: str,
    value: object,
) -> None:
    item = _friend_habits(1).as_dict()
    item[field] = value
    with pytest.raises((TypeError, ValueError)):
        opponent_habits_from_json(_habit_json_payload([item]))


def test_opponent_habit_json_rejects_duplicate_ids_and_trimmed_nicknames() -> None:
    first = _friend_habits(1, opponent_id="friend-a", nickname="阿杰")
    duplicate_id = _friend_habits(2, opponent_id="friend-a", nickname="老周")
    duplicate_name = _friend_habits(3, opponent_id="friend-c", nickname=" 阿杰 ")

    with pytest.raises(ValueError, match="重复|不能相同"):
        opponent_habits_to_json((first, duplicate_id))
    with pytest.raises(ValueError, match="重复|不能相同"):
        opponent_habits_from_json(
            _habit_json_payload([first.as_dict(), duplicate_name.as_dict()])
        )


def test_opponent_habit_json_rejects_unknown_version() -> None:
    payload = _habit_json_payload([_friend_habits(1).as_dict()], version=2)
    with pytest.raises(ValueError, match="版本|version|1"):
        opponent_habits_from_json(payload)


@pytest.mark.parametrize("bad_version", (True, 1.0, "1", None))
def test_opponent_habit_json_rejects_non_integer_version(
    bad_version: object,
) -> None:
    payload = _habit_json_payload(
        [_friend_habits(1).as_dict()], version=bad_version
    )
    with pytest.raises(ValueError, match="版本"):
        opponent_habits_from_json(payload)


def test_opponent_habit_json_rejects_wrong_format_and_oversized_input() -> None:
    wrong_format = _habit_json_payload(
        [_friend_habits(1).as_dict()], format_name="another-app"
    )
    with pytest.raises(ValueError, match="德州训练器"):
        opponent_habits_from_json(wrong_format)
    with pytest.raises(ValueError, match="64KB"):
        opponent_habits_from_json(" " * (64 * 1024 + 1))


def test_opponent_habit_json_accepts_utf8_bom_and_rejects_bad_schema() -> None:
    friend = _friend_habits(1)
    assert opponent_habits_from_json(
        "\ufeff" + _habit_json_payload([friend.as_dict()])
    ) == (friend,)

    missing = friend.as_dict()
    missing.pop("calling_tendency")
    unknown = friend.as_dict() | {"vpip": 0.9}
    for row in (missing, unknown):
        with pytest.raises(ValueError, match="字段|无效"):
            opponent_habits_from_json(_habit_json_payload([row]))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("opponent_id", None),
        ("nickname", None),
        ("nickname", "牌友\n甲"),
        ("nickname", "这是一位昵称特别特别特别特别长的牌友"),
    ),
)
def test_opponent_habit_json_rejects_unsafe_identity_values(
    field: str, value: object
) -> None:
    item = _friend_habits(1).as_dict()
    item[field] = value
    with pytest.raises(ValueError):
        opponent_habits_from_json(_habit_json_payload([item]))


def test_web_sessions_get_distinct_default_databases(monkeypatch, tmp_path) -> None:
    base_path = tmp_path / "poker_trainer.sqlite3"
    monkeypatch.setattr(ui_app, "_CONFIGURED_DB_PATH", None)
    monkeypatch.setattr(ui_app, "DEFAULT_DB_PATH", base_path)

    first = ui_app._new_web_session_db_path()
    second = ui_app._new_web_session_db_path()

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert "-session-" in first.name and first.suffix == ".sqlite3"

    monkeypatch.setattr(ui_app, "_CONFIGURED_DB_PATH", str(base_path))
    assert ui_app._new_web_session_db_path() == base_path


def test_import_reset_clears_only_stale_opponent_form_widgets() -> None:
    state = {
        "opponent_habits": ("keep",),
        "opponent_habits_upload": "keep-upload",
        "opponent_nickname_friend-a": "旧昵称",
        "opponent_entry_frequency_friend-a": "偏少",
        "opponent_calling_tendency_friend-a": "偏爱跟",
        "nav": "设置",
    }
    fake_streamlit = SimpleNamespace(session_state=state)

    ui_app._clear_opponent_form_widgets(fake_streamlit)

    assert state == {
        "opponent_habits": ("keep",),
        "opponent_habits_upload": "keep-upload",
        "nav": "设置",
    }


def test_seat_grid_escapes_untrusted_friend_nickname() -> None:
    malicious_name = '<script>alert("x")</script><b>牌友</b>'
    view = {
        "current_actor_id": None,
        "street_name": "翻前",
        "history": [],
        "seats": [
            {
                "player_id": "friend-unsafe",
                "position": Position.HJ,
                "position_name": "HJ（中位）",
                "name": malicious_name,
                "stack": 4_000,
                "folded": False,
                "all_in": False,
                "is_hero": False,
                "cards": (),
                "cards_hidden": True,
            }
        ],
    }

    html = seat_grid_html(view)
    assert malicious_name not in html
    assert "<script>" not in html and "<b>" not in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html
    assert "&lt;b&gt;牌友&lt;/b&gt;" in html


def test_chip_and_card_formatting_is_mobile_friendly() -> None:
    assert format_chips(4_000) == "4,000 筹码（¥40.00）"
    assert format_chips(60) == "60 筹码（¥0.60）"
    assert format_card("Ts") == "10♠"
    assert format_cards(["As", "Kh"]) == "A♠ K♥"
    assert format_cards([], hidden=True) == "🂠 🂠"
    assert "&lt;" in cards_html(["<s"])
    with pytest.raises(ValueError):
        format_chips(100, chips_per_yuan=0)


def test_hand_rank_formatting_explains_the_reported_two_pair_tiebreakers() -> None:
    board = parse_cards("Qd Ts 9d 6d 6h")
    sb_rank = evaluate((*parse_cards("9c Tc"), *board))
    hero_rank = evaluate((*parse_cards("Td 4s"), *board))
    utg_rank = evaluate((*parse_cards("As 9h"), *board))

    assert sb_rank > hero_rank > utg_rank
    assert format_hand_rank(sb_rank) == "两对 10 和 9（Q 踢脚）"
    assert format_hand_rank(hero_rank) == "两对 10 和 6（Q 踢脚）"
    assert format_hand_rank(utg_rank) == "两对 9 和 6（A 踢脚）"
    assert "HandRank" not in format_hand_rank(sb_rank)


def test_only_legal_action_buttons_are_built_and_call_shows_amount() -> None:
    controls = legal_action_controls(_legal())
    assert [control.action for control in controls] == [
        ActionType.FOLD,
        ActionType.CALL,
        ActionType.RAISE,
        ActionType.ALL_IN,
    ]
    assert controls[1].label == "跟注 40"
    assert controls[2].needs_amount is True
    assert ActionType.CHECK not in {control.action for control in controls}
    assert ActionType.BET not in {control.action for control in controls}


def test_bet_to_slider_uses_engine_minimum_and_effective_stack_maximum() -> None:
    raise_range = bet_to_range(_legal(pot_before=500, min_raise_to=220, max_to=1_000))
    assert raise_range is not None
    assert (raise_range.minimum, raise_range.maximum) == (220, 1_000)
    assert 220 <= raise_range.default <= 1_000
    assert raise_range.step == 20

    bet_range = bet_to_range(
        _legal(
            to_call=0,
            call_amount=0,
            min_bet_to=40,
            min_raise_to=None,
            can_call=False,
            can_bet=True,
            can_raise=False,
            max_to=800,
        )
    )
    assert bet_range is not None and bet_range.minimum == 40
    assert bet_to_range(_legal(can_raise=False, min_raise_to=None)) is None


def test_public_hand_view_hides_running_opponent_cards_and_no_profile_fields(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=20260828)
    view = hand_public_view(hand, "UTG")
    hero = next(row for row in view["seats"] if row["player_id"] == "UTG")
    opponents = [row for row in view["seats"] if row["player_id"] not in {"", "UTG"}]
    assert len(hero["cards"]) == 2 and hero["cards_hidden"] is False
    assert all(row["cards"] == () and row["cards_hidden"] for row in opponents)
    assert view["pot"] == 60
    public_keys = set(view)
    public_keys.update(key for seat in view["seats"] for key in seat)
    assert not {
        "vpip",
        "pfr",
        "three_bet",
        "aggression_factor",
        "fold_tendency",
        "limp_tendency",
        "mistake_rate",
    } & public_keys


def test_public_hand_view_applies_names_only_as_a_display_overlay(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=20260907)
    original_name = hand.player("HJ").name

    view = hand_public_view(
        hand,
        "UTG",
        display_names={"HJ": "阿杰"},
    )

    hj = next(row for row in view["seats"] if row["player_id"] == "HJ")
    assert hj["name"] == "阿杰"
    assert hand.player("HJ").name == original_name
    assert "阿杰" not in repr(hand.public_view("UTG"))


def test_public_hand_view_reveals_live_hands_only_after_showdown(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=77)
    _play_passively(hand)
    view = hand_public_view(hand, "UTG")
    assert hand.result is not None and hand.result.reason == "showdown"
    assert all(len(row["cards"]) == 2 for row in view["seats"] if row["player_id"])

    pots = result_pot_rows(hand)
    ranks = showdown_rank_rows(hand)
    assert pots and pots[0]["底池"] == "主池"
    assert sum("筹码" in row["金额"] for row in pots) == len(pots)
    assert all(row["赢家"] != "待确认" for row in pots)
    assert ranks and {row["玩家"].split(" · ")[0] for row in ranks} <= {
        full_position_name(position) for position in POSITION_ORDER
    }
    assert ranks[0]["结果"] == "赢家"
    assert all(row["最佳五张"] != "—" for row in ranks)
    assert all("HandRank" not in row["牌型"] for row in ranks)


def test_reported_showdown_rows_and_summary_name_the_actual_winner() -> None:
    hand = _reported_showdown_hand()

    pots = result_pot_rows(hand)
    assert len(pots) == 1
    assert pots[0]["底池"] == "主池"
    assert pots[0]["金额"] == "1,120 筹码（¥11.20）"
    assert "SB（小盲） · 对手5" in pots[0]["赢家"]
    assert "BB（大盲）" not in pots[0]["赢家"]

    rows = showdown_rank_rows(hand)
    assert [row["玩家"].split(" · ")[0] for row in rows] == [
        "SB（小盲）",
        "BB（大盲）",
        "UTG（前位）",
    ]
    assert rows[0]["结果"] == "赢家"
    assert rows[0]["牌型"] == "两对 10 和 9（Q 踢脚）"
    assert rows[0]["获得"] == "1,120 筹码（¥11.20）"
    assert rows[1]["结果"] == "未获底池"
    assert rows[1]["牌型"] == "两对 10 和 6（Q 踢脚）"
    assert rows[2]["牌型"] == "两对 9 和 6（A 踢脚）"
    assert all(row["最佳五张"] != "—" for row in rows)
    assert all("HandRank" not in row["牌型"] for row in rows)

    summary = settlement_summary(hand, "BB")
    assert "赢家：SB（小盲） · 对手5" in summary
    assert "两对 10 和 9（Q 踢脚）" in summary
    assert "获得 1,120 筹码" in summary
    assert "你的牌型是两对 10 和 6（Q 踢脚）" in summary
    assert "本手未获得底池" in summary


def test_showdown_helpers_apply_the_same_temporary_name_overlay() -> None:
    hand = _reported_showdown_hand()
    names = {"SB": "阿杰"}

    pots = result_pot_rows(hand, display_names=names)
    rows = showdown_rank_rows(hand, display_names=names)
    summary = settlement_summary(hand, "BB", display_names=names)

    assert "SB（小盲） · 阿杰" in pots[0]["赢家"]
    assert any(row["玩家"] == "SB（小盲） · 阿杰" for row in rows)
    assert "SB（小盲） · 阿杰" in summary
    assert getattr(hand, "players")["SB"].name == "对手5"


def test_main_and_side_pot_rows_name_each_winner_and_summary_lists_both() -> None:
    hand = _split_pot_winners_hand()

    pots = result_pot_rows(hand)
    assert [(row["底池"], row["金额"]) for row in pots] == [
        ("主池", "900 筹码（¥9.00）"),
        ("边池 1", "800 筹码（¥8.00）"),
    ]
    assert "UTG（前位） · 短码玩家" in pots[0]["赢家"]
    assert "SB（小盲）" not in pots[0]["赢家"]
    assert "SB（小盲） · 中码玩家" in pots[1]["赢家"]
    assert "UTG（前位）" not in pots[1]["赢家"]

    summary = settlement_summary(hand, "BB")
    assert "UTG（前位） · 短码玩家以三条 K" in summary
    assert "获得 900 筹码" in summary
    assert "SB（小盲） · 中码玩家以三条 8" in summary
    assert "获得 800 筹码" in summary
    assert summary.index("短码玩家") < summary.index("中码玩家")
    assert "你的牌型是A 高牌" in summary
    assert "本手未获得底池" in summary


def test_history_rows_are_chinese_and_include_forced_blinds(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=91)
    rows = action_history_rows(hand.history)
    assert [row["动作"] for row in rows] == ["下小盲", "下大盲"]
    assert rows[0]["位置"] == "SB（小盲）"
    assert rows[0]["简述"] == "下小盲 20"
    assert rows[0]["强制"] is True
    assert rows[1]["底池后"] == 60
    assert rows[1]["底池前"] == 20
    assert rows[1]["投入"] == 40
    assert rows[1]["需跟"] == 0


def test_action_rows_use_compact_amount_words_and_positive_refund() -> None:
    common = {
        "street": Street.FLOP,
        "player_id": "CO",
        "position": Position.CO,
        "requested_amount": None,
        "pot_before": 200,
        "to_call_before": 0,
        "current_bet_before": 0,
        "current_bet_after": 0,
        "min_raise_to_before": 80,
        "is_all_in": False,
        "is_full_raise": False,
        "forced": False,
    }
    bet = ActionRecord(
        sequence=4,
        action=ActionType.BET,
        paid=120,
        bet_to=120,
        pot_after=320,
        **common,
    )
    refund = ActionRecord(
        sequence=5,
        action=ActionType.REFUND,
        paid=-40,
        bet_to=80,
        pot_after=280,
        **common,
    )
    rows = action_history_rows([bet, refund])
    assert rows[0]["简述"] == "下注到 120"
    assert rows[1]["简述"] == "退回 40"
    assert "-" not in rows[1]["筹码"]


def test_action_timeline_is_grouped_compact_and_highlights_hero() -> None:
    rows = [
        {"#": 2, "街道": "翻前", "位置简称": "CO", "玩家": "villain", "动作代码": "raise", "简述": "加注到 120", "底池后": 180},
        {"#": 3, "街道": "翻前", "位置简称": "BTN", "玩家": "hero", "动作代码": "call", "简述": "跟注 120", "底池后": 300},
        {"#": 7, "街道": "翻牌", "位置简称": "CO", "玩家": "villain", "动作代码": "bet", "简述": "下注到 200", "底池后": 500},
    ]
    html = action_timeline_html(rows, hero_id="hero", active_sequence=7)
    assert html.count("action-street") == 2
    assert "你" in html and "跟注 120" in html
    assert "hero-action" in html
    assert "action-bet latest" in html


def test_seat_cards_keep_latest_action_amount_and_highlight_hero() -> None:
    view = {
        "current_actor_id": None,
        "street_name": "翻牌",
        "history": [
            {"玩家": "hero", "街道": "翻牌", "简述": "跟注 120", "强制": False},
            {"玩家": "villain", "街道": "翻牌", "简述": "加注到 350", "强制": False},
        ],
        "seats": [
            {
                "player_id": "hero",
                "position": Position.UTG,
                "position_name": "UTG（前位）",
                "name": "英雄",
                "stack": 3_880,
                "folded": False,
                "all_in": False,
                "is_hero": True,
                "cards": ("As", "Kh"),
                "cards_hidden": False,
            },
            {
                "player_id": "villain",
                "position": Position.HJ,
                "position_name": "HJ（中位）",
                "name": "对手1",
                "stack": 3_650,
                "folded": False,
                "all_in": False,
                "is_hero": False,
                "cards": (),
                "cards_hidden": True,
            },
        ],
    }
    html = seat_grid_html(view)
    assert 'class="seat-card hero"' in html
    assert "你｜跟注 120" in html
    assert "HJ｜加注到 350" in html


def test_seat_cards_hide_previous_street_actions_and_keep_terminal_state() -> None:
    view = {
        "current_actor_id": None,
        "street_name": "转牌",
        "history": [
            {"玩家": "live", "街道": "翻前", "简述": "跟注 120", "强制": False},
            {"玩家": "folded", "街道": "翻牌", "简述": "弃牌", "强制": False},
            {"玩家": "allin", "街道": "翻牌", "简述": "全下至 900", "强制": False},
        ],
        "seats": [
            {
                "player_id": player_id,
                "position": position,
                "position_name": full_position_name(position),
                "name": player_id,
                "stack": 3_000,
                "folded": player_id == "folded",
                "all_in": player_id == "allin",
                "is_hero": False,
                "cards": (),
                "cards_hidden": True,
            }
            for player_id, position in (
                ("live", Position.UTG),
                ("folded", Position.HJ),
                ("allin", Position.CO),
            )
        ],
    }

    html = seat_grid_html(view)
    assert "跟注 120" not in html
    assert "已弃牌" in html
    assert "已全下" in html


def test_statistics_table_always_has_six_positions_and_all_13_metrics() -> None:
    empty = aggregate_by_position([])
    rows = statistics_table_rows(empty)
    assert len(rows) == 6
    assert len(METRIC_ORDER) == 13
    assert all(len(row) == 15 for row in rows)  # 位置 + 手数 + 13 指标
    assert rows[0]["VPIP"] == "信号不足"
    assert rows[-1]["听牌赔率错误次数"] == "0次（0次机会）"


def test_metric_formats_keep_af_ratio_and_draw_error_count() -> None:
    assert format_metric(MetricName.VPIP, 1, 4) == "25.0%（1/4）"
    assert format_metric(MetricName.AGGRESSION_FACTOR, 3, 2) == "1.50（3/2）"
    assert format_metric(MetricName.DRAW_ODDS_ERROR, 2, 5) == "2次（5次机会）"
    assert format_metric(MetricName.WSD, 0, 0) == "信号不足"


def test_metric_detail_has_exactly_13_rows() -> None:
    tallies = {name: MetricTally(1, 2) for name in METRIC_ORDER}
    row = PositionStatistics(Position.BTN, 2, tallies)
    details = metric_detail_rows(row)
    assert len(details) == 13
    assert {item["指标"] for item in details} >= {"VPIP", "W$SD", "顶对打光率"}


def test_review_return_shapes_are_normalized_without_losing_order() -> None:
    first, second = object(), object()
    assert normalize_reviews(None) == ()
    assert normalize_reviews(first) == (first,)
    assert normalize_reviews([first, second]) == (first, second)


def test_replay_reviews_use_review_sequence_not_list_position() -> None:
    reviews = [
        {"sequence": 11, "reason": "later"},
        {"sequence": 4, "reason": "first"},
        {"sequence": "7", "reason": "current"},
        {"sequence": "bad", "reason": "invalid"},
    ]
    reached = replay_reviews_through_sequence(reviews, 7)
    assert [review["reason"] for review in reached] == ["first", "current"]
    assert replay_reviews_through_sequence(reviews, None) == ()


def test_review_cards_link_exact_action_sequence_and_show_equity_gap() -> None:
    reviews = [
        {
            "sequence": 7,
            "street": "flop",
            "action": "call",
            "rating": "明显错误",
            "reason": "权益不足，跟注过松。",
            "pot_odds": 0.30,
            "equity": 0.18,
            "outs": 4,
            "hit_probability": 0.16,
            "draw_names": ["卡顺听牌"],
            "equity_basis": "random_unknown_hands",
            "recommended_action": "fold",
            "detail_lines": ["多人底池应收紧继续范围。"],
        }
    ]
    history = [
        {"#": 6, "简述": "过牌"},
        {
            "#": 7,
            "简述": "跟注 300",
            "动作代码": "call",
            "投入": 300,
            "需跟": 300,
            "底池前": 700,
        },
    ]
    html = review_cards_html(reviews, history=history)
    assert "翻牌 · 你跟注 300" in html
    assert "明显错误" in html and "grade-error" in html
    assert "所需胜率 30.0%" in html
    assert "随机未知手牌基准权益 18.0%" in html
    assert "基准权益差 -12.0pct" in html
    assert "常见听牌 4 outs" in html
    assert "听牌类型 卡顺听牌" in html
    assert "到河牌命中 16.0%" in html
    assert "更好的选择：弃牌" in html
    assert "300 ÷（700 + 300）= 30.0%" in html
    assert "不是对手动作加权后的真实范围" in html
    assert "多人底池应收紧继续范围。" in html
    assert "赔率与计算" in html


def test_review_cards_keep_core_strategy_visible_and_split_extra_analysis() -> None:
    html = review_cards_html(
        [
            {
                "sequence": 3,
                "street": "preflop",
                "action": "raise",
                "rating": "推荐",
                "reason": "按钮位可以主动开池。",
                "recommended_action": "加注到约 120",
                "detail_lines": [
                    "牌型 K9o｜位置 BTN｜场景 无人入池开池。",
                    "默认：加注。",
                    "可接受：桌面异常激进时收紧。",
                ],
                "equity": 0.54,
            }
        ]
    )

    assert '<div class="review-strategy">' in html
    assert "牌型 K9o｜位置 BTN｜场景 无人入池开池。" in html
    assert "默认：加注。" in html
    assert "更多策略说明" in html
    assert "可接受：桌面异常激进时收紧。" in html
    assert "赔率与计算" in html
    assert "展开完整分析" not in html
    assert "建议打法：加注到约 120" in html
    assert "推荐替代" not in html
    assert "更好的选择" not in html


def test_review_cards_do_not_invent_missing_math_or_alternative() -> None:
    html = review_cards_html(
        [
            {
                "sequence": 2,
                "street": "preflop",
                "action": "check",
                "rating": "推荐",
                "reason": "大盲可免费看牌。",
            }
        ]
    )

    assert "大盲可免费看牌。" in html
    assert "所需胜率" not in html
    assert "基准权益" not in html
    assert "outs" not in html
    assert "建议打法" not in html
    assert "更好的选择" not in html


def test_review_cards_use_contestable_pot_formula_when_side_pot_is_not_known() -> None:
    html = review_cards_html(
        [
            {
                "sequence": 9,
                "street": "turn",
                "action": "call",
                "rating": "可以接受",
                "reason": "权益接近底池赔率。",
                "pot_odds": 0.25,
            }
        ],
        history=[
            {
                "#": 9,
                "简述": "跟注 100",
                "动作代码": "call",
                "投入": 100,
                "需跟": 100,
                "底池前": 900,
            }
        ],
    )

    assert "跟注额 ÷（可争夺底池 + 跟注额）= 25.0%" in html
    assert "100 ÷（900 + 100）" not in html


def test_review_cards_do_not_show_call_odds_for_a_raise() -> None:
    html = review_cards_html(
        [
            {
                "sequence": 5,
                "street": "preflop",
                "action": "raise",
                "rating": "明显错误",
                "reason": "弱牌不适合隔离多人 limp。",
                "pot_odds": 0.10,
                "recommended_action": "弃牌",
            }
        ],
        history=[
            {
                "#": 5,
                "街道": "翻前",
                "简述": "加注到 160",
                "动作代码": "raise",
                "投入": 140,
                "需跟": 20,
                "底池前": 180,
                "当前下注前": 40,
                "当前下注后": 160,
            }
        ],
    )

    assert "更好的选择：弃牌" in html
    assert "所需胜率" not in html
    assert "赔率公式" not in html


def test_review_cards_only_show_call_odds_for_a_calling_allin() -> None:
    review = {
        "sequence": 8,
        "street": "turn",
        "action": "all_in",
        "rating": "可以接受",
        "reason": "短码全下跟注。",
        "pot_odds": 0.25,
    }
    calling_html = review_cards_html(
        [review],
        history=[
            {
                "#": 8,
                "街道": "转牌",
                "简述": "全下至 300",
                "动作代码": "all_in",
                "投入": 300,
                "需跟": 400,
                "底池前": 900,
                "当前下注前": 400,
                "当前下注后": 400,
            }
        ],
    )
    raising_html = review_cards_html(
        [review],
        history=[
            {
                "#": 8,
                "街道": "转牌",
                "简述": "全下至 1,200",
                "动作代码": "all_in",
                "投入": 1_080,
                "需跟": 200,
                "底池前": 900,
                "当前下注前": 400,
                "当前下注后": 1_200,
            }
        ],
    )

    assert "所需胜率 25.0%" in calling_html
    assert "300 ÷（900 + 300）= 25.0%" in calling_html
    assert "所需胜率" not in raising_html
    assert "赔率公式" not in raising_html


def test_review_cards_hide_allin_call_math_when_old_history_cannot_classify_it() -> None:
    html = review_cards_html(
        [
            {
                "sequence": 8,
                "street": "turn",
                "action": "all_in",
                "rating": "偏松/偏紧",
                "reason": "旧记录缺少全下分类字段。",
                "pot_odds": 0.25,
            }
        ],
        history=[
            {
                "#": 8,
                "街道": "转牌",
                "简述": "全下至 300",
                "动作代码": "all_in",
                "投入": 300,
                "需跟": 400,
                "底池前": 900,
            }
        ],
    )

    assert "所需胜率" not in html
    assert "赔率公式" not in html


def test_hero_net_result_uses_ending_stack_minus_initial_stack(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=7788)
    initial = hand.players["UTG"].stack
    _play_passively(hand)
    assert hero_net_result(hand, "UTG") == hand.players["UTG"].stack - initial


def test_replay_helper_rebuilds_same_state_for_same_action_step(six_seats) -> None:
    hand = HoldemHand(six_seats(), seed=314159)
    _play_passively(hand)
    bundle = ReplayBundle.from_hand(hand)
    action_count = len(bundle.actions) // 2
    first = rebuild_replay(bundle, action_count)
    second = rebuild_replay(bundle.to_json(), action_count)
    assert hand_public_view(first, "UTG") == hand_public_view(second, "UTG")
    assert [record.as_dict() for record in first.history] == [
        record.as_dict() for record in second.history
    ]


def test_sqlite_replay_helpers_list_and_load_bundle(tmp_path, six_seats) -> None:
    db_path = tmp_path / "trainer.sqlite3"
    with SQLiteStore(db_path) as store:
        store.create_session("ui-session", hero_player_id="UTG", seed=12)
        hand = HoldemHand(
            six_seats(),
            seed=12,
            session_id="ui-session",
            hand_id="ui-hand",
        )
        _play_passively(hand)
        bundle = ReplayBundle.from_hand(hand)
        store.save_hand(hand, bundle.to_json(), "test")
        hero_record = next(record for record in hand.history if record.player_id == "UTG")
        store.save_decision_review(
            hand.hand_id,
            hero_record.sequence,
            {
                "player_id": "UTG",
                "street": hero_record.street.value,
                "action": hero_record.action.value,
                "rating": "可以接受",
                "reason": "测试复盘",
            },
        )

    rows = list_saved_hands(db_path)
    assert rows[0]["hand_id"] == "ui-hand"
    loaded, hero_id = load_replay_bundle("ui-hand", db_path)
    assert loaded.to_json() == bundle.to_json()
    assert hero_id == "UTG"
    reviews = load_saved_reviews("ui-hand", db_path)
    assert [row["sequence"] for row in reviews] == [hero_record.sequence]
    assert reviews[0]["reason"] == "测试复盘"


def test_mobile_css_has_touch_target_and_phone_breakpoint() -> None:
    assert "min-height: 46px" in MOBILE_CSS
    assert "@media (max-width: 480px)" in MOBILE_CSS
    assert "@media (max-width: 380px)" in MOBILE_CSS
    assert ".seat-grid" in MOBILE_CSS
    assert ".action-feed" in MOBILE_CSS
    assert ".review-overview" in MOBILE_CSS
    assert ".review-metrics" in MOBILE_CSS
    assert ".review-alternative" in MOBILE_CSS
    assert ".review-strategy" in MOBILE_CSS
    assert ".review-calculation" in MOBILE_CSS
    assert ".review-formula" in MOBILE_CSS
    assert ".replay-now" in MOBILE_CSS
    assert 'button[kind="secondary"]' in MOBILE_CSS
    assert "background: #fff" in MOBILE_CSS
    assert ".st-key-action_dock" in MOBILE_CSS
    assert ".st-key-replay_nav" in MOBILE_CSS
    assert "grid-template-columns: repeat(2" in MOBILE_CSS
    assert 'header[data-testid="stHeader"]' in MOBILE_CSS
    assert "display: none !important" in MOBILE_CSS
