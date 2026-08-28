"""Streamlit 移动端优先样式。

模块顶层不导入 Streamlit，便于核心测试在未安装网页依赖时运行。
"""

from __future__ import annotations

from typing import Any


MOBILE_CSS = r"""
<style>
:root {
  --felt: #0d5b43;
  --felt-dark: #073d2e;
  --ink: #10231d;
  --muted: #5f716b;
  --paper: #f7faf8;
  --line: #d9e5df;
  --gold: #d6a94d;
  --danger: #b42318;
}

/* 竖屏手机先紧凑，大屏也不让牌桌过宽。 */
.stApp {
  background: linear-gradient(180deg, #edf5f1 0%, #ffffff 38%);
  color: var(--ink);
}

/* 训练器不需要 Streamlit 的 Deploy/菜单工具栏，移除顶部黑条。 */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {
  display: none !important;
}
.block-container {
  max-width: 680px;
  padding-top: .3rem;
  padding-bottom: 6.5rem;
}
h1 { font-size: clamp(1.4rem, 6.5vw, 2.15rem) !important; line-height: 1.12 !important; }
h2 { font-size: clamp(1.25rem, 5vw, 1.65rem) !important; }
h3 { font-size: clamp(1.05rem, 4.5vw, 1.3rem) !important; }

/* 所有操作按钮满足触控尺寸。 */
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button,
button[kind] {
  min-height: 46px;
  border-radius: 12px;
  font-size: .92rem;
  font-weight: 700;
  touch-action: manipulation;
}
div[role="radiogroup"] label {
  min-height: 48px;
  display: inline-flex;
  align-items: center;
}
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: var(--felt);
  border-color: var(--felt);
  color: #fff;
}
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stFormSubmitButton"] > button[kind="secondary"] {
  background: #fff;
  border-color: #b8cec4;
  color: var(--felt-dark);
}
div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stFormSubmitButton"] > button[kind="secondary"]:hover {
  background: #edf5f1;
  border-color: var(--felt);
  color: var(--felt-dark);
}

.notice-strip {
  border: 1px solid #b9d6ca;
  border-left: 5px solid var(--felt);
  border-radius: 12px;
  background: rgba(255,255,255,.88);
  padding: .7rem .85rem;
  margin: .2rem 0 .85rem;
  font-size: .88rem;
  line-height: 1.5;
}
.notice-strip strong { color: var(--felt-dark); }
.notice-strip summary { cursor: pointer; font-weight: 800; color: var(--felt-dark); }
.notice-strip .notice-body { margin-top: .45rem; }

.table-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .5rem;
  margin: .55rem 0 .75rem;
}
.summary-tile {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 13px;
  padding: .62rem .45rem;
  text-align: center;
  box-shadow: 0 3px 10px rgba(9, 57, 42, .05);
}
.summary-tile .label { color: var(--muted); font-size: .75rem; }
.summary-tile .value { color: var(--ink); font-weight: 800; font-size: 1rem; margin-top: .15rem; }

.board-zone {
  background: radial-gradient(circle at 50% 30%, #15785a, var(--felt-dark));
  border: 3px solid #bd9550;
  border-radius: 999px;
  padding: 1.05rem .8rem;
  margin: .4rem 0 .8rem;
  text-align: center;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.14), 0 5px 18px rgba(7,61,46,.18);
  color: white;
}
.board-zone .street { opacity: .8; font-size: .76rem; letter-spacing: .08em; }
.board-zone .cards { font-size: clamp(1.3rem, 6vw, 2rem); font-weight: 850; margin: .32rem 0; }
.board-zone .pot { font-size: .88rem; }

.seat-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .55rem;
  margin: .45rem 0 .8rem;
}
.seat-card {
  min-width: 0;
  background: rgba(255,255,255,.95);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .62rem .68rem;
  box-shadow: 0 3px 10px rgba(9,57,42,.05);
}
.seat-card.hero { border: 2px solid var(--gold); background: #fffaf0; }
.seat-card.acting {
  border-color: var(--felt);
  box-shadow: 0 0 0 2px rgba(13,91,67,.16), 0 3px 10px rgba(9,57,42,.08);
}
.seat-card.folded { opacity: .58; filter: grayscale(.35); }
.seat-card .position { font-size: .82rem; font-weight: 800; color: var(--felt-dark); }
.seat-card .name { font-size: .75rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.seat-card .cards { font-size: 1.12rem; font-weight: 850; margin: .25rem 0; min-height: 1.45rem; }
.seat-card .stack { font-size: .78rem; }
.seat-card .state {
  color: var(--felt-dark);
  font-size: .72rem;
  font-weight: 800;
  min-height: 1.05rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.seat-card.folded .state { color: var(--muted); }
.seat-card.acting .state { color: var(--felt); }

.action-feed {
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255,255,255,.94);
  margin: .25rem 0 .6rem;
}
.action-street {
  padding: .28rem .58rem;
  background: #e9f2ee;
  color: var(--felt-dark);
  font-size: .7rem;
  font-weight: 850;
  letter-spacing: .05em;
}
.action-line {
  display: grid;
  grid-template-columns: 2.8rem minmax(0, 1fr) auto;
  align-items: center;
  gap: .35rem;
  min-height: 38px;
  padding: .38rem .55rem;
  border-top: 1px solid #edf2ef;
  font-size: .82rem;
}
.action-line.latest { background: #fff7e6; box-shadow: inset 3px 0 0 var(--gold); }
.action-line.hero-action .action-actor { color: #9a5b00; }
.action-actor { color: var(--felt-dark); font-weight: 900; }
.action-summary { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-weight: 750; }
.action-pot { color: var(--muted); font-size: .7rem; white-space: nowrap; }
.action-fold .action-summary { color: var(--muted); }
.action-bet .action-summary,
.action-raise .action-summary { color: #a15c00; }
.action-all-in .action-summary { color: var(--danger); }
.action-empty { color: var(--muted); font-size: .82rem; padding: .5rem 0; }

.playing-card {
  display: inline-block;
  min-width: 1.55em;
  margin: 0 .08em;
  padding: .08em .15em;
  border-radius: .28em;
  background: #fff;
  color: #131a17;
  border: 1px solid rgba(0,0,0,.15);
  box-shadow: 0 1px 2px rgba(0,0,0,.16);
}
.playing-card.red { color: #c62828; }
.playing-card.hidden { color: #eef6f2; background: repeating-linear-gradient(45deg,#164f3e,#164f3e 4px,#276c56 4px,#276c56 8px); }

.feedback-card {
  border-radius: 14px;
  border: 1px solid var(--line);
  border-left: 4px solid #438b68;
  background: #fff;
  padding: .68rem .72rem;
  margin: .45rem 0;
}
.feedback-head { display: flex; align-items: center; justify-content: space-between; gap: .45rem; }
.feedback-card .decision { min-width: 0; font-weight: 850; font-size: .86rem; }
.feedback-card .rating {
  flex: 0 0 auto;
  border-radius: 999px;
  padding: .16rem .42rem;
  font-size: .7rem;
  font-weight: 900;
  color: var(--felt-dark);
  background: #e8f3ed;
}
.feedback-card .reason { margin-top: .3rem; line-height: 1.48; }
.feedback-card .numbers { margin-top: .35rem; color: var(--muted); font-size: .78rem; }
.feedback-card.grade-acceptable { border-left-color: #3976b8; }
.feedback-card.grade-acceptable .rating { color: #21598f; background: #e9f2fb; }
.feedback-card.grade-marginal { border-left-color: #d58a18; }
.feedback-card.grade-marginal .rating { color: #8b5200; background: #fff1d9; }
.feedback-card.grade-error { border-left-color: var(--danger); }
.feedback-card.grade-error .rating { color: #8d1b12; background: #fde9e7; }
.review-overview {
  border-radius: 10px;
  background: #eaf3ef;
  color: var(--felt-dark);
  padding: .48rem .62rem;
  font-size: .78rem;
  font-weight: 800;
}
.review-evidence { margin-top: .42rem; color: var(--muted); font-size: .75rem; }
.review-evidence summary { cursor: pointer; font-weight: 750; }
.review-evidence div { padding-top: .28rem; }

.replay-counter {
  display: flex;
  min-height: 46px;
  align-items: center;
  justify-content: center;
  color: var(--felt-dark);
  font-weight: 900;
  font-size: .82rem;
}
.replay-now {
  display: flex;
  align-items: center;
  gap: .45rem;
  padding: .55rem .65rem;
  margin: .32rem 0 .45rem;
  border: 1px solid #e0c889;
  border-radius: 11px;
  background: #fff8e8;
  font-size: .82rem;
  font-weight: 800;
}
.replay-now span {
  flex: 0 0 auto;
  color: #8b5c00;
  font-size: .68rem;
  text-transform: uppercase;
}

/* Streamlit 1.40+ 带 key 容器时将动作区固定在手机底部；旧版仍正常顺序显示。 */
.st-key-action_dock {
  position: sticky;
  bottom: .35rem;
  z-index: 99;
  padding: .55rem;
  border-radius: 16px;
  background: rgba(250,253,251,.96);
  border: 1px solid #c9dbd3;
  box-shadow: 0 -5px 18px rgba(7,61,46,.14);
  backdrop-filter: blur(8px);
}

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
div[data-testid="stSlider"] { padding-top: .15rem; }
[data-testid="stMetricValue"] { font-size: clamp(1.25rem, 6vw, 1.8rem); }

@media (max-width: 480px) {
  .block-container { padding-left: .6rem; padding-right: .6rem; padding-top: .25rem; }
  .seat-grid { gap: .42rem; }
  .seat-card { padding: .52rem; }
  .seat-card .position { font-size: .76rem; }
  .table-summary { gap: .35rem; }
  .summary-tile { padding: .55rem .3rem; }
  div[data-testid="stHorizontalBlock"] { gap: .35rem; }
  .st-key-action_dock div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: .35rem !important;
  }
  .st-key-replay_nav div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 3.4rem minmax(0, 1fr) !important;
    gap: .35rem !important;
    align-items: center;
  }
  .st-key-action_dock div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  .st-key-replay_nav div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
  }
}

@media (max-width: 380px) {
  .block-container { padding-left: .45rem; padding-right: .45rem; }
  .notice-strip { padding: .5rem .6rem; margin-bottom: .55rem; }
  .table-summary { gap: .25rem; margin: .35rem 0 .5rem; }
  .summary-tile { padding: .42rem .2rem; border-radius: 10px; }
  .summary-tile .label { font-size: .66rem; }
  .summary-tile .value { font-size: .86rem; }
  .board-zone { padding: .7rem .45rem; margin-bottom: .55rem; }
  .seat-grid { gap: .28rem; margin-bottom: .55rem; }
  .seat-card { padding: .4rem .42rem; border-radius: 11px; }
  .seat-card .position { font-size: .68rem; }
  .seat-card .name, .seat-card .stack { font-size: .68rem; }
  .seat-card .cards { font-size: .96rem; margin: .14rem 0; }
  .action-line { grid-template-columns: 2.25rem minmax(0, 1fr) auto; padding: .34rem .42rem; }
  .st-key-action_dock { padding: .4rem; bottom: .2rem; }
}

@media (min-width: 700px) {
  .seat-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
</style>
"""


def apply_styles(st_module: Any | None = None) -> None:
    """注入样式；默认延迟导入 Streamlit。"""

    if st_module is None:
        import streamlit as st_module  # type: ignore[no-redef]
    st_module.markdown(MOBILE_CSS, unsafe_allow_html=True)


__all__ = ["MOBILE_CSS", "apply_styles"]
