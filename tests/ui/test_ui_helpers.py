from __future__ import annotations

import importlib
import sys

import pytest

from poker_trainer.analytics.database import SQLiteStore
from poker_trainer.analytics.statistics import (
    METRIC_ORDER,
    MetricName,
    MetricTally,
    PositionStatistics,
    aggregate_by_position,
)
from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionRecord, ActionType, LegalActions, Position, Street
from poker_trainer.engine.replay import ReplayBundle
from poker_trainer.ui.app import (
    POSITION_ORDER,
    action_history_rows,
    action_timeline_html,
    bet_to_range,
    cards_html,
    format_card,
    format_cards,
    format_chips,
    format_metric,
    full_position_name,
    hand_public_view,
    hero_net_result,
    legal_action_controls,
    list_saved_hands,
    load_replay_bundle,
    load_saved_reviews,
    metric_detail_rows,
    normalize_reviews,
    rebuild_replay,
    replay_reviews_through_sequence,
    review_cards_html,
    result_pot_rows,
    seat_grid_html,
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


def test_chip_and_card_formatting_is_mobile_friendly() -> None:
    assert format_chips(4_000) == "4,000 筹码（¥40.00）"
    assert format_chips(60) == "60 筹码（¥0.60）"
    assert format_card("Ts") == "10♠"
    assert format_cards(["As", "Kh"]) == "A♠ K♥"
    assert format_cards([], hidden=True) == "🂠 🂠"
    assert "&lt;" in cards_html(["<s"])
    with pytest.raises(ValueError):
        format_chips(100, chips_per_yuan=0)


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
    assert ranks and {row["玩家"].split(" · ")[0] for row in ranks} <= {
        full_position_name(position) for position in POSITION_ORDER
    }


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
