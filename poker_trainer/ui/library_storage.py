"""Browser-local roster persistence; no shared server file or public roster."""

from __future__ import annotations

import json
from typing import Any


STORAGE_JS = r'''
export default function({data, setStateValue, parentElement}) {
    const key = "poker-trainer.friends.v1";
    const label = parentElement.querySelector("p");
    try {
        const current = window.localStorage.getItem(key);
        if (data.mode === "read") {
            label.textContent = "正在读取本机牌友库…";
            setStateValue("result", {kind: "loaded", raw: current});
            return;
        }
        if (data.mode === "disabled") {
            label.textContent = "本机自动保存未启用，请保留备份链接。";
            return;
        }
        // Do not silently overwrite a newer library saved by another tab.
        if (current !== data.record && current !== data.expected) {
            label.textContent = "其他标签页已修改牌友库；本页未覆盖它，请刷新后继续。";
            setStateValue("result", {kind: "conflict"});
            return;
        }
        if (current !== data.record) window.localStorage.setItem(key, data.record);
        label.textContent = `已自动保存 ${data.count} 位牌友到本机浏览器`;
        setStateValue("result", {kind: "saved", raw: data.record});
    } catch (_) {
        label.textContent = "浏览器不允许本机保存，请保留备份链接。";
        setStateValue("result", {kind: "unavailable"});
    }
}
'''


def decode_record(raw: str) -> dict[str, Any]:
    from .app import opponent_library_from_json

    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 96 * 1024:
        raise ValueError("本机档案过大或格式错误")
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError("本机档案版本不受支持")
    habits = opponent_library_from_json(record.get("library"))
    active = record.get("active")
    source = record.get("source", "")
    if (not isinstance(active, list) or len(active) > 12
            or any(not isinstance(x, str) for x in active)
            or not isinstance(source, str) or len(source) > 12000):
        raise ValueError("本机档案字段无效")
    ids = {h.opponent_id for h in habits}
    if len(set(active)) != len(active) or not set(active) <= ids:
        raise ValueError("本机档案包含无效座位选择")
    return {"habits": habits, "active": tuple(active), "source": source}


def encode_record(state: Any) -> str:
    from .app import opponent_library_to_json

    habits = state.get("opponent_habits", ())
    ids = {h.opponent_id for h in habits}
    active = list(dict.fromkeys(x for x in state.get("active_opponent_ids", ()) if x in ids))
    return json.dumps({
        "version": 1, "library": opponent_library_to_json(habits),
        "active": active, "source": state.get("_browser_library_source", ""),
    }, ensure_ascii=False, sort_keys=True)


def restore_record(state: Any, raw: str | None) -> None:
    """A new explicit link overrides storage; reopening the old link keeps edits."""
    saved = decode_record(raw) if raw is not None else None
    link = state.get("_loaded_friends_token", "")
    if saved is not None and (not link or link == saved["source"]):
        state["opponent_habits"] = saved["habits"]
        state["active_opponent_ids"] = saved["active"]
        state["_browser_library_source"] = saved["source"]
        state["_reset_opponent_form_widgets"] = True
        state["_opponent_flash"] = f"已恢复本机保存的 {len(saved['habits'])} 位牌友及习惯"
    else:
        state["_browser_library_source"] = link
    state["_browser_library_expected"] = raw
    state["_browser_library_ready"] = True


def sync_browser_library(st: Any) -> None:
    """Read before allowing edits, write after rendering so selection edits persist."""
    from streamlit.components.v2 import component

    state = st.session_state
    ready = state.get("_browser_library_ready", False)
    disabled = state.get("_browser_library_disabled", False)
    record = encode_record(state)
    mode = "disabled" if disabled else ("write" if ready else "read")
    mount = component(
        "poker_trainer_browser_library", js=STORAGE_JS,
        html='<p role="status"></p>',
        css="p {font-size: 13px; color: #53736a; margin: 0;}",
    )
    result = mount(
        key="browser_library", data={"mode": mode, "record": record,
            "expected": state.get("_browser_library_expected"),
            "count": len(state.get("opponent_habits", ()))},
        on_result_change=lambda: None,
    ).result
    if not isinstance(result, dict):
        return
    if result.get("kind") == "loaded" and not ready:
        try:
            restore_record(state, result.get("raw"))
        except (TypeError, ValueError, AttributeError):
            state["_browser_library_ready"] = True
            state["_browser_library_disabled"] = True
            state["_browser_library_error"] = "本机存档损坏，未覆盖原档案。请先下载备份，再清理本站浏览器数据并重新导入。"
        st.rerun()
    elif result.get("kind") == "saved" and result.get("raw") == record:
        state["_browser_library_expected"] = record
        st.caption(f"已固定保存 {len(state.get('opponent_habits', ()))} 位牌友到本机浏览器")
    elif result.get("kind") == "unavailable" and not disabled:
        state["_browser_library_ready"] = True
        state["_browser_library_disabled"] = True
        state["_browser_library_error"] = "浏览器禁止本机保存；当前仅保留在本次会话，请使用备份链接或下载档案。"
        st.rerun()
