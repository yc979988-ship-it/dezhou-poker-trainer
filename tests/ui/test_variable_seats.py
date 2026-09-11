from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position, Seat, positions_for_table_size
from poker_trainer.engine.replay import ReplayBundle
from poker_trainer.ui.app import hand_public_view, rebuild_replay, seat_grid_html, statistics_table_rows


@pytest.mark.parametrize('size,hero_position', [
    (size, position) for size in (5, 6, 7, 8) for position in positions_for_table_size(size)
])
def test_every_hero_position_survives_display_and_replay(size, hero_position):
    seats = [Seat(p.value, p.value, p) for p in positions_for_table_size(size)]
    hand = HoldemHand(seats, seed=13823307061057370)
    for state in (hand, rebuild_replay(ReplayBundle.from_hand(hand), 0)):
        view = hand_public_view(state, hero_position.value)
        assert len(view['seats']) == size
        assert {s['player_id'] for s in view['seats']} == set(hand.players)
        hero, = [s for s in view['seats'] if s['is_hero']]
        assert hero['cards'] == tuple(hand.player(hero_position.value).hole_cards)
        assert hero['stack'] == hand.player(hero_position.value).stack
        assert all(s['cards_hidden'] for s in view['seats'] if not s['is_hero'])
        assert seat_grid_html(view).count('class="seat-card') == size


def test_eight_seat_hero_utg2_visible_with_real_stack_and_action_controls():
    seats = [Seat(p.value, '你' if p == Position.UTG2 else p.value, p)
             for p in positions_for_table_size(8)]
    hand = HoldemHand(seats, seed=13823307061057370)
    hand.act('UTG', ActionType.CALL)
    hand.act('UTG+1', ActionType.FOLD)
    trainer = SimpleNamespace(current_hand=hand, hero_id='UTG+2', mode='test',
                              opponent_display_names={}, advance_bots=lambda: None)
    page = AppTest.from_file(Path(__file__).resolve().parents[2] / 'app.py')
    page.session_state['trainer'] = trainer
    page.session_state['nav'] = '训练'
    page.run()
    assert not page.exception
    html = '\n'.join(m.value for m in page.markdown)
    assert '英雄后手</div><div class="value">4,000' in html
    assert 'UTG+2（前位2）' in html
    assert 'seat-card hero acting' in html
    assert any(b.label == '弃牌' for b in page.button)
    hand.act('UTG+2', ActionType.FOLD)
    page.run()
    assert not page.exception
    html = '\n'.join(m.value for m in page.markdown)
    assert 'seat-card hero folded' in html
    assert '英雄后手</div><div class="value">4,000' in html


@pytest.mark.parametrize('size', [5, 6, 7, 8])
def test_position_statistics_follow_table_size(size):
    positions = positions_for_table_size(size)
    rows = statistics_table_rows({}, positions=positions)
    assert [r['位置'] for r in rows] == [p.display_name for p in positions]

