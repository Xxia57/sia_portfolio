# =============================================================================
# 내 집 마련 — 감당할 수 있는 집부터 찾기 (Streamlit 초안, 앱 시안 기반)
#   실행: ..\.venv\Scripts\streamlit.exe run app.py   (home_loan_mvp 폴더에서)
#   메뉴 (views/ 폴더)
#     찾기     : 내 구매력 / 지역·단지 찾기 (+ 숨김 페이지: 단지 상세) / 계산 근거
#     내 정보  : 내 재무 정보 / 계산 가정 / 마이데이터 연결 관리
#     시뮬레이션: 직접 입력 (마이데이터 없이 값을 넣어보는 화면)
#   ※ 먼저 02_train.py를 실행해 models/home_loan_models.joblib이 있어야 함
#   ※ 마이데이터 파일: output/accounts.csv, customer_debt_summary.csv, customer_financial_profiles.csv
# =============================================================================
import importlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# 공통 모듈이 바뀌었으면 다시 읽기 — 페이지 파일은 매번 새로 실행되지만 import한 모듈은 메모리에 남아,
# 앱을 켜 둔 채 코드를 고치면 옛 버전이 쓰일 수 있음. 하나가 바뀌면 그 뒤(그 모듈을 import하는 것)까지 다시 읽음
_HERE = Path(__file__).resolve().parent
_LOCAL = ["prep", "recommender", "ui", "app_state"]   # 의존 순서
_changed = False
for _name in _LOCAL:
    _mod = sys.modules.get(_name)
    if _mod is None:
        continue
    _mtime = (_HERE / f"{_name}.py").stat().st_mtime
    if _changed or getattr(_mod, "_loaded_mtime", None) != _mtime:
        importlib.reload(_mod)
        _changed = True
for _name in _LOCAL:
    __import__(_name)
    sys.modules[_name]._loaded_mtime = (_HERE / f"{_name}.py").stat().st_mtime

from ui import get_mydata, get_recommender  # noqa: E402

st.set_page_config(page_title="내 집 마련", page_icon="🏠", layout="wide")

MYDATA_PAGES = [
    st.Page("views/power.py", title="내 구매력", icon="🏠", default=True),
    st.Page("views/region.py", title="지역·단지 찾기", icon="📍"),
    st.Page("views/detail.py", title="단지 상세", icon="🏢", visibility="hidden"),
    st.Page("views/basis.py", title="계산 근거", icon="📄"),
    st.Page("views/finance.py", title="내 재무 정보", icon="🗂️"),
    st.Page("views/assumptions.py", title="계산 가정", icon="⚙️"),
    st.Page("views/mydata_link.py", title="마이데이터 연결 관리", icon="🔗"),
]
pg = st.navigation({
    "찾기": MYDATA_PAGES[:4],
    "내 정보": MYDATA_PAGES[4:],
    "시뮬레이션": [st.Page("views/direct_input.py", title="직접 입력", icon="✏️")],
})

# -----------------------------------------------------------------------------
# 사이드바: 서비스 이름 + 고객 선택(가상 로그인) — 마이데이터 페이지들이 같은 고객을 공유
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 내 집 마련\n<span style='opacity:.7'>감당할 수 있는 집부터 찾기</span>", unsafe_allow_html=True)
    if pg.title in [p.title for p in MYDATA_PAGES]:
        C, _, missing = get_mydata()
        with st.container(border=True):
            if C is None or C.empty:
                st.error("마이데이터 파일이 없습니다.\n\n" + "\n".join(f"- `{m}`" for m in missing))
            else:
                ids = C.index.to_numpy()
                # 선택값은 'cid'에 보관 (페이지를 옮겨 위젯이 사라져도 유지되도록 위젯 값과 분리)
                st.session_state.setdefault("cid", int(ids[0]))
                st.session_state.setdefault("_cid_widget", st.session_state.cid)

                def _sync():
                    st.session_state.cid = st.session_state._cid_widget

                def _random():
                    st.session_state.cid = st.session_state._cid_widget = int(np.random.default_rng().choice(ids))

                st.number_input("고객 ID", int(ids.min()), int(ids.max()), step=1, key="_cid_widget", on_change=_sync)
                st.button("🎲 사용자 바꾸기 (무작위)", on_click=_random, width="stretch")
                st.markdown(":gray-badge[무주택] :blue-badge[● 마이데이터 연결됨]")
                if st.session_state.cid not in C.index:
                    st.error(f"고객 ID {st.session_state.cid}는 무주택 고객 목록에 없습니다.")

    st.divider()
    st.markdown(f":green[●] **합성 데이터 기반 서비스**  \n"
                f"<span style='font-size:12px;opacity:.75'>실제 매물이나 대출 승인을 보장하지 않아요<br>"
                f"데이터 기준 {get_recommender().data_date()}</span>", unsafe_allow_html=True)

pg.run()
