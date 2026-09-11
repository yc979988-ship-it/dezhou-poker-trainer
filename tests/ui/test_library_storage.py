import json
import shutil
import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from poker_trainer.opponents.profiles import OpponentHabits
from poker_trainer.ui.library_storage import (
    STORAGE_JS, decode_record, encode_record, restore_record,
)


def saved_state():
    return {"opponent_habits": tuple(OpponentHabits(f"f{i}", f"测试{i}", calling_tendency=5)
                                    for i in range(8)),
            "active_opponent_ids": ("f0", "f2", "f6"),
            "_browser_library_source": "original-link"}


def test_fresh_browser_session_restores_complete_library_and_choices():
    old = saved_state()
    new = {}
    restore_record(new, encode_record(old))
    assert new["opponent_habits"] == old["opponent_habits"]
    assert new["active_opponent_ids"] == old["active_opponent_ids"]
    assert new["_browser_library_ready"]


def test_old_link_does_not_undo_edits_or_resurrect_deleted_friends():
    old = saved_state()
    old["opponent_habits"] = ()
    new = {"_loaded_friends_token": "original-link", **saved_state()}
    restore_record(new, encode_record(old))
    assert new["opponent_habits"] == ()


def test_new_explicit_link_takes_priority():
    new = {"_loaded_friends_token": "new-link", "opponent_habits": (),
           "active_opponent_ids": ()}
    restore_record(new, encode_record(saved_state()))
    assert new["opponent_habits"] == ()
    assert new["_browser_library_source"] == "new-link"


def test_empty_browser_does_not_erase_link_import():
    state = saved_state()
    restore_record(state, None)
    assert len(state["opponent_habits"]) == 8


@pytest.mark.parametrize("raw", ["broken", "[]", "{}", "x" * 100000,
                                '{"version":1,"library":null}'],
                         ids=["bad-json", "list", "no-version", "oversize", "null-library"])
def test_corrupt_storage_never_changes_session_library(raw):
    state = saved_state()
    before = state.copy()
    with pytest.raises((TypeError, ValueError, AttributeError)):
        restore_record(state, raw)
    assert state == before


def test_invalid_active_ids_are_rejected():
    raw = json.loads(encode_record(saved_state()))
    raw["active"] = ["stranger"]
    with pytest.raises(ValueError):
        decode_record(json.dumps(raw))


def test_initial_ui_waits_for_browser_before_allowing_library_edits():
    page = AppTest.from_file(Path(__file__).resolve().parents[2] / "app.py").run()
    assert not page.exception
    assert not any(b.label == "＋ 新增一位牌友" for b in page.button)
    assert any("正在读取本机" in c.value for c in page.caption)
    page.button[0].click().run()
    assert not page.exception
    assert page.session_state["_browser_library_disabled"]
    assert any(b.label == "＋ 新增一位牌友" for b in page.button)


def test_browser_js_read_write_reload_conflict_and_storage_denial():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the browser storage bridge")
    js = STORAGE_JS.replace("export default function", "function bridge", 1)
    script = js + r'''
    const assert = require('node:assert/strict');
    let stored = null, result, writes = 0;
    global.window = {localStorage: {
        getItem: () => stored,
        setItem: (_, value) => {stored = value; writes++;}
    }};
    const label = {};
    function run(data) {
        bridge({data, parentElement: {querySelector: () => label},
                setStateValue: (_, value) => result = value});
    }
    run({mode:'read'});
    assert.deepEqual(result, {kind:'loaded', raw:null});
    assert.equal(writes, 0); // Never write an empty library before hydration.
    run({mode:'write', expected:null, record:'eight-friends', count:8});
    assert.equal(stored, 'eight-friends');
    run({mode:'read'}); // Fresh page / cloud restart reads from the browser.
    assert.equal(result.raw, 'eight-friends');
    run({mode:'write', expected:'eight-friends', record:'edited', count:7});
    assert.equal(stored, 'edited');
    run({mode:'write', expected:'eight-friends', record:'stale-tab', count:8});
    assert.equal(result.kind, 'conflict');
    assert.equal(stored, 'edited');
    run({mode:'write', expected:'edited', record:'empty-library', count:0});
    assert.equal(stored, 'empty-library');
    run({mode:'read'});
    assert.equal(result.raw, 'empty-library');
    window.localStorage.getItem = () => {throw new Error('blocked');};
    run({mode:'read'});
    assert.equal(result.kind, 'unavailable');
    '''
    result = subprocess.run([node, "-"], input=script, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
