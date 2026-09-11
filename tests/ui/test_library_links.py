import base64
import zlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from poker_trainer.opponents.profiles import OpponentHabits
from poker_trainer.ui.app import opponent_library_from_token, opponent_library_link


def friends():
    return tuple(OpponentHabits(f"friend-{i}", f"测试牌友{i}", calling_tendency=4) for i in range(8))


def test_link_preserves_all_habits_in_two_independent_web_sessions():
    roster = friends()
    token = opponent_library_link(roster).split("friends=", 1)[1]
    assert opponent_library_from_token(token) == roster
    for _ in range(2):
        page = AppTest.from_file(Path(__file__).resolve().parents[2] / "app.py")
        page.query_params["friends"] = token
        page.run()
        assert not page.exception
        assert page.session_state["opponent_habits"] == roster
        assert len(page.checkbox) == 8
        # 普通重绘不会把当前会话的编辑重新覆盖为链接旧快照。
        page.session_state["opponent_habits"] = roster[:7]
        page.run()
        assert len(page.session_state["opponent_habits"]) == 7


@pytest.mark.parametrize("token", ["", "invalid!", "x" * 12001,
    base64.urlsafe_b64encode(zlib.compress(b"x" * 100000)).decode(),
    base64.urlsafe_b64encode(zlib.compress(b"{}") + b"extra").decode()])
def test_rejects_corrupt_or_oversized_link(token):
    with pytest.raises(ValueError):
        opponent_library_from_token(token)


def test_bad_link_preserves_existing_library():
    page = AppTest.from_file(Path(__file__).resolve().parents[2] / "app.py")
    page.session_state["opponent_habits"] = friends()
    page.query_params["friends"] = "broken!"
    page.run()
    assert not page.exception
    assert page.session_state["opponent_habits"] == friends()
    assert page.warning

