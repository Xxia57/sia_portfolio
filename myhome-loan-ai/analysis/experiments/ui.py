# =============================================================================
# 공통 화면 모듈 — views/의 모든 페이지가 같이 사용
#   get_recommender() / get_mydata() / current_customer(): 데이터·모델·선택 고객 (캐시)
#   render_results(): 1. 비슷한 구매자 → 2. 내 조건 예측 → 3. 지역 시세·대안
#   render_model_trace(): 모델이 결과를 만드는 과정을 단계별로 펼쳐 보여줌
# =============================================================================
import pandas as pd
import streamlit as st

from prep import (BASE_DIR, COMBO_NAMES, DATA_DIR, DSR_OK, DSR_WARN, LTV_CAP, MIN_GROUP, MODEL_DIR, MYDATA_DIR,
                  MYDATA_FILES, NUM_FEATURES, load_mydata, user_frame)
from recommender import (CL_MAX_RATIO, ERR_DSR, ERR_LOAN_PCT, KNN_FEATURES, PYEONG,
                         Recommender, loan_payment, pct, split_loan, won)

DISCLAIMER = ("⚠️ **가상 데이터 기반** 결과입니다. 실제 금융기관의 대출 심사·한도·금리와 다를 수 있습니다. "
              f"다른 사람의 개별 기록은 표시하지 않으며, {MIN_GROUP}명 이상 집단의 통계와 모델 예측값만 보여줍니다.")


@st.cache_resource(show_spinner="데이터와 모델을 불러오는 중...", max_entries=1)   # 옛 버전은 메모리에서 비움
def _load_recommender(version):
    # 데이터·모델은 한 번만 읽고 재사용 (입력을 바꿀 때마다 다시 읽지 않도록, 모든 페이지가 공유)
    return Recommender()


def _version(*paths):
    """파일 수정 시각 묶음 — 코드·모델·데이터가 바뀌면 값이 달라져 캐시를 새로 만든다
    (앱을 켜 둔 채 재학습·코드 수정을 해도 옛 계산 객체가 남지 않도록)"""
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


def get_recommender():
    return _load_recommender(_version(MODEL_DIR / "home_loan_models.joblib", BASE_DIR / "prep.py",
                                      BASE_DIR / "recommender.py", DATA_DIR / "home_purchases.csv"))


@st.cache_data(show_spinner="마이데이터(계좌·부채·프로필)를 불러오는 중...", max_entries=1)
def _load_mydata(version):
    # (고객 표, 계좌 원본, 없는 파일 목록) — 무주택 고객만
    return load_mydata()


def get_mydata():
    return _load_mydata(_version(BASE_DIR / "prep.py", *(MYDATA_DIR / f for f in MYDATA_FILES.values())))


def current_customer():
    """사이드바에서 고른 마이데이터 고객 (customer_id, 고객 1행, 그 고객의 계좌들). 없으면 None"""
    C, ACC, _ = get_mydata()
    cid = st.session_state.get("cid")
    if C is None or cid not in C.index:
        return None
    return cid, C.loc[cid], ACC[ACC.customer_id == cid]


VERDICT_TEXT = {"여유": "✅ 감당 가능 (여유)", "주의": "⚠️ 감당 가능하지만 주의", "어려움": "❌ 감당 어려움"}
VERDICT_BOX = {"여유": st.success, "주의": st.warning, "어려움": st.error}


def verdict_reasons(p, existing):
    """판정 사유 목록 (요약과 상세 섹션이 같이 사용)"""
    dsr_label = "DSR (기존 대출 포함)" if existing > 0 else "DSR"
    reasons = []
    new_loan = p.loan > 0      # DSR 기준은 새 대출이 있을 때만 적용
    if new_loan and p.dsr > DSR_WARN:
        reasons.append(f"{dsr_label} {pct(p.dsr)} → 40% 초과")
    elif new_loan and p.dsr > DSR_OK:
        reasons.append(f"{dsr_label} {pct(p.dsr)} → 30~40% 구간")
    elif not new_loan and p.dsr > DSR_OK:
        reasons.append(f"새 대출 없이 사는 경우라 DSR 기준은 적용하지 않아요 (기존 대출만으로 {dsr_label} {pct(p.dsr)})")
    if p.surplus < 0:
        reasons.append(f"매달 {won(-p.surplus)} 적자 (소득 − 소비 − 상환)")
    if p.shortfall > 0:
        reasons.append(f"주담대(LTV {LTV_CAP:.0%})와 마통 한도를 다 써도 {won(p.shortfall)} 부족")
    if new_loan and (abs(p.dsr - DSR_WARN) <= ERR_DSR or abs(p.dsr - DSR_OK) <= ERR_DSR):
        reasons.append(f"DSR이 판정 경계와 가까워 예측 오차(±{ERR_DSR * 100:.1f}%p)에 따라 판정이 바뀔 수 있음")
    return reasons


def compute(R, inp):
    """화면에 필요한 계산을 한 번에 수행 (판정 사유·감당 가능 최대 가격 포함)"""
    u0 = user_frame(inp["income"], inp["assets"], inp["consumption"], inp["price"],
                    inp["area"], inp["age"], inp["sido"], R.cats, inp.get("existing_pay", 0.0))
    g = R.similar_group(u0)
    u, proba, use_cl, rates, cash, p = R.predict(inp, g)
    # 감당 가능 최대 가격은 '여유'가 아닐 때만 계산 (여유면 희망가 그대로 가능)
    afford = R.max_affordable(inp, rates, use_cl, cash) if p.verdict != "여유" else (inp["price"], inp["price"])
    return dict(g=g, u=u, proba=proba, use_cl=use_cl, rates=rates, cash=cash, p=p,
                reasons=verdict_reasons(p, inp.get("existing_pay", 0.0)), afford=afford)


def render_summary(inp, res):
    """한눈에 보기 — 판정과 핵심 숫자를 페이지 맨 위에 크게 표시"""
    p, reasons, (ok_price, comfy_price) = res["p"], res["reasons"], res["afford"]
    existing = inp.get("existing_pay", 0.0)
    with st.container(border=True):
        st.subheader(f"한눈에 보기 — 희망가 {won(inp['price'])} · {inp['sigungu']}")
        VERDICT_BOX[p.verdict](f"### 판정: {VERDICT_TEXT[p.verdict]}"
                               + ("\n\n- " + "\n- ".join(reasons) if reasons else ""))
        m = st.columns(4)
        m[0].metric("예상 대출 금액", won(p.loan), help=f"주담대 {won(p.mort)} + 마통 {won(p.cl)} · 검증 오차 약 ±{ERR_LOAN_PCT:.0%}")
        m[1].metric("월 상환액 (새 대출)", won(p.pay))
        m[2].metric("DSR (기존 대출 포함)" if existing > 0 else "DSR", pct(p.dsr),
                    help=f"(새 대출 + 기존 대출 연 상환액) ÷ 연 소득 {won(inp['income'])}")
        m[3].metric("월 잉여현금", won(p.surplus), help="연 소득 ÷ 12 − 월 소비 − 월 상환")
        m = st.columns(4)
        m[0].metric("자기자금", won(p.price - p.loan - p.shortfall), help=f"보유 자금 합계 {won(res['cash'])}")
        m[1].metric("자금 부족액", won(p.shortfall) if p.shortfall > 0 else "없음")
        if p.verdict == "여유":
            m[2].metric("감당 가능한 최대 가격", f"{won(inp['price'])} 이상", help="희망가가 이미 '여유' 판정")
        else:
            m[2].metric("감당 가능한 최대 가격", won(ok_price) if ok_price else "없음",
                        help="DSR 40% 이내 · 월 적자 없음 · 자금 부족 없음"
                             + ("" if ok_price else " — 3,000만 원짜리 집도 이 조건을 못 맞춤"))
            m[3].metric("여유 있는 최대 가격", won(comfy_price) if comfy_price else "-", help="DSR 30% 이내")
        st.caption("자세한 근거는 아래 1~3번 섹션과 🔍 모델 작동 과정에서 볼 수 있어요.")


# 대출 구성 막대 색 — dataviz 기본 팔레트 1~3번(모든 쌍 검증 통과)과 상태색 critical
#   부족액은 상태(문제)라 빗금 + ⚠️ 아이콘 + 글자로 함께 표시 (색만으로 구분하지 않음)
FUND_PARTS = [  # (키, 이름, 색)
    ("mort", "주택담보대출", "#2a78d6"),
    ("cl", "마이너스통장", "#eb6834"),
    ("own", "자기자금", "#1baf7a"),
    ("short", "⚠️ 자금 부족", "#d03b3b"),
]


def render_loan_structure(inp, res):
    """집값을 자기자금·주담대·마통·부족액으로 나눈 막대 + 주담대/마통 상세 카드"""
    p, rates = res["p"], res["rates"]
    price = p.price
    own = price - p.loan - p.shortfall
    amounts = {"mort": p.mort, "cl": p.cl, "own": own, "short": p.shortfall}

    st.subheader("💳 이렇게 빌려요 — 대출 구성")
    parts = [(k, n, c, amounts[k]) for k, n, c in FUND_PARTS if amounts[k] > 0]

    # ---- 집값 구성 막대 (한 줄 누적 막대, 칸 사이 2px 간격, 양 끝 4px 둥글게) ----
    segs = []
    for i, (k, n, c, v) in enumerate(parts):
        share = v / price
        hatch = ("background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.35) 0 4px,"
                 "transparent 4px 9px);") if k == "short" else ""
        label = f"{n.replace('⚠️ ', '')} {share:.0%}" if share >= 0.14 else ""   # 좁은 칸은 범례로
        radius = ("4px 0 0 4px" if i == 0 else "0") if len(parts) > 1 else "4px"
        if i == len(parts) - 1 and len(parts) > 1:
            radius = "0 4px 4px 0"
        segs.append(
            f'<div title="{n}: {won(v)} ({share:.1%})" style="flex:{share:.4f} 1 0;min-width:6px;'
            f'background:{c};{hatch}border-radius:{radius};color:#fff;font-size:13px;font-weight:600;'
            f'display:flex;align-items:center;justify-content:center;white-space:nowrap;'
            f'overflow:visible">{label}</div>')
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};display:inline-block"></span>'
        f'<span>{n} <b>{won(v)}</b> <span style="opacity:.7">({v / price:.0%})</span></span></span>'
        for k, n, c, v in parts)
    st.markdown(
        f'<div style="font-size:14px;margin:4px 0 6px">집값 <b>{won(price)}</b>을 이렇게 마련해요</div>'
        f'<div style="display:flex;gap:2px;height:38px;width:100%">{"".join(segs)}</div>'
        f'<div style="font-size:13px;margin-top:8px;line-height:1.9">{legend}</div>',
        unsafe_allow_html=True)

    # ---- 대출별 DSR 기여분: 각 대출의 연 상환액 ÷ 연 소득 (DSR 합계 = 아래 셋의 합) ----
    income, existing = inp["income"], inp.get("existing_pay", 0.0)
    mp, cp = loan_payment(p.mort, 0, rates), loan_payment(0, p.cl, rates)
    dsr_m, dsr_c, dsr_e = mp * 12 / income, cp * 12 / income, existing * 12 / income

    # ---- 빌리는 순서 (마통은 '집값 몇 %'가 아니라 DSR을 얼마나 쓰는지로 안내) ----
    if p.loan <= 0:
        steps = [f"① 계좌 잔액에서 **자기자금 {won(own)}**으로 집값을 모두 채워요 — 새 대출 없음"]
    else:
        steps = [f"① 계좌 잔액에서 **자기자금 {won(own)}**을 넣고",
                 f"② **주택담보대출 {won(p.mort)}**을 받고 (집값의 {p.mort / price:.0%}) → 월 {won(mp)}, **DSR {dsr_m * 100:.1f}%p**"]
    if p.cl > 0:
        steps.append(f"③ 모자라는 **{won(p.cl)}**은 마이너스통장으로 채워요 → 월 {won(cp)}, **DSR +{dsr_c * 100:.1f}%p**")
    if p.shortfall > 0:
        steps.append(f"⚠️ 그래도 **{won(p.shortfall)}**이 모자라요 — 대출로 채울 수 없는 금액")
    st.markdown("  \n".join(steps))

    # DSR 합계와 기준까지 남은 여유
    parts_txt = f"주담대 {dsr_m * 100:.1f}%p" + (f" + 마통 {dsr_c * 100:.1f}%p" if p.cl > 0 else "") \
        + (f" + 기존 대출 {dsr_e * 100:.1f}%p" if existing > 0 else "")
    room = DSR_WARN - p.dsr
    room_txt = ("새 대출이 없어 DSR 기준은 적용하지 않아요" if p.loan <= 0 else
                f"40% 기준까지 **{room * 100:.1f}%p 여유**" if room >= 0
                else f"40% 기준을 **{-room * 100:.1f}%p 초과**")
    st.markdown(f"**DSR {pct(p.dsr)}** = {parts_txt} · {room_txt}")

    # ---- 대출별 상세 카드 ----
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        n = rates["term"] * 12
        st.markdown(f"**🏦 주택담보대출** · {won(p.mort)}")
        m = st.columns(3)
        m[0].metric("금리", f"{rates['mort']:.2f}%")
        m[1].metric("월 상환", won(mp))
        m[2].metric("DSR 차지", f"{dsr_m * 100:.1f}%p", help="주담대 연 상환액 ÷ 연 소득")
        st.caption(f"원리금균등 상환 — {rates['term']:.0f}년({n:.0f}개월) 동안 매달 같은 금액을 갚아요. "
                   f"총 이자 약 **{won(mp * n - p.mort)}** (금리가 그대로일 때)")
    with c2.container(border=True):
        st.markdown(f"**💳 마이너스통장** · {won(p.cl) if p.cl > 0 else '사용 안 함'}")
        if p.cl > 0:
            m = st.columns(3)
            m[0].metric("금리", f"{rates['cl']:.2f}%")
            m[1].metric("월 상환", won(cp))
            m[2].metric("DSR 차지", f"{dsr_c * 100:.1f}%p", help="마통 연 상환액 ÷ 연 소득")
            st.caption(f"월 상환 = 잔액 × (연 금리 ÷ 12 + 원금 0.3%). 이자만 월 약 {won(p.cl * rates['cl'] / 1200)}이고, "
                       "보통 1년마다 한도를 갱신해요. 주담대보다 금리가 높아 먼저 갚는 게 유리해요.")
        else:
            st.caption("주담대만으로 필요한 금액을 채울 수 있어 마이너스통장은 쓰지 않아요."
                       if p.need_loan <= p.mort else "비슷한 구매자 패턴상 주담대만 쓰는 쪽으로 계산했어요.")
    st.caption(f"새 대출 월 상환 합계 **{won(p.pay)}** = 주담대 {won(mp)}" + (f" + 마통 {won(cp)}" if p.cl > 0 else "")
               + (f" · 기존 대출 월 상환 {won(existing)}은 DSR에 함께 포함" if existing > 0 else ""))


def render_tiers(R, inp, res):
    """DSR 구간(보수·권장·최대)별로 빌릴 수 있는 금액과 부담 + 비슷한 구매자 패턴의 위치"""
    p, rates, cash = res["p"], res["rates"], res["cash"]
    existing = inp.get("existing_pay", 0.0)
    st.subheader("📊 대출 구간별 부담 — DSR 기준")
    st.caption(f"희망가 {won(inp['price'])}·연소득 {won(inp['income'])} 기준으로, 각 구간의 DSR 상한까지 빌린다고 보고 계산했습니다. "
               f"주담대(LTV {LTV_CAP:.0%})를 먼저 쓰고 나머지는 마통으로 채우며, 필요한 금액보다 더 빌리지는 않습니다. "
               "금리 상승은 주담대·마통 모두 변동금리라고 가정했습니다."
               + (" 기존 대출 상환액은 DSR 한도에서 먼저 뺐습니다." if existing > 0 else ""))
    LIMIT_TEXT = {"필요 금액": "필요한 금액을 모두 빌릴 수 있음", "DSR": "DSR 상한에 걸림",
                  "대출 한도": "주담대·마통 한도에 걸림"}
    tiers = R.loan_tiers(inp, rates, cash)
    for col, t in zip(st.columns(len(tiers)), tiers):
        with col.container(border=True):
            star = " ⭐" if t["tier"] == "권장" else ""
            st.markdown(f"**{t['tier']}{star}** · DSR {t['label']}  \n<small>{t['desc']}</small>", unsafe_allow_html=True)
            st.metric("대출액", won(t["loan"]), help=LIMIT_TEXT[t["limited_by"]])
            st.caption(f"주담대 {won(t['mort'])} + 마통 {won(t['cl'])} · {LIMIT_TEXT[t['limited_by']]}")
            st.metric("월 상환액", won(t["pay"]), delta=f"DSR {pct(t['dsr'])}", delta_color="off", delta_arrow="off")
            lines = [f"월 잉여현금 **{won(t['surplus'])}**" + (" ⚠️ 적자" if t["surplus"] < 0 else "")]
            lines.append(f"자금 부족 **{won(t['shortfall'])}**" if t["shortfall"] > 0 else "자금 부족 없음")
            for s, pay_s, dsr_s in t["shocks"]:
                warn = " ⚠️ 40% 초과" if t["loan"] > 0 and dsr_s > DSR_WARN else ""   # 새 대출이 있을 때만
                lines.append(f"금리 +{s:.0f}%p → 월 {won(pay_s)} (DSR {pct(dsr_s)}){warn}")
            st.markdown("  \n".join(lines))

    # 비슷한 구매자 패턴(GBM)이 어느 구간에 해당하는지
    pm, pc = split_loan(p.pattern_loan, inp["price"])
    pattern_dsr = (loan_payment(pm, pc, rates) + existing) * 12 / inp["income"]
    # 구간 라벨(35~40% 등)과 계산 상한(25/30/40%)이 달라, 상한선과 비교한 문장으로 표시
    caps = [(t["tier"], t["dsr_cap"]) for t in tiers]
    if pattern_dsr <= caps[0][1]:
        where = f"{caps[0][0]} 구간(DSR {caps[0][1]:.0%} 이하)"
    elif pattern_dsr <= caps[1][1]:
        where = f"{caps[1][0]} 구간(DSR {caps[1][1]:.0%} 이하)"
    elif pattern_dsr <= caps[2][1]:
        where = f"{caps[1][0]}({caps[1][1]:.0%})보다 높고 {caps[2][0]}({caps[2][1]:.0%}) 이하"
    else:
        where = f"{caps[2][0]}({caps[2][1]:.0%})를 넘는 수준"
    st.caption(f"참고: 비슷한 구매자 패턴대로 빌리면 {won(pm + pc)}, DSR {pct(pattern_dsr)} → {where}입니다.")


def render_results(R, inp, res, compact=False):
    """compact=True: 위에 요약·대출 구성·대출 구간을 이미 보여준 페이지(②)용 — '내 조건 예측'과 대출 구간을 생략"""
    g, proba = res["g"], res["proba"]
    desc, share, tbl = R.group_summary(g)
    sec = iter(range(1, 10))   # 생략되는 섹션이 있어도 번호가 이어지도록

    # ---- 비슷한 구매자 --------------------------------------------------------
    st.header(f"{next(sec)}. 나와 비슷한 구매자들은 이렇게 샀어요")
    st.caption("최근 5년(2022~) 매입 건 중 매입 당시 나이·가구소득·금융자산·월소비·집값·면적이 가장 가까운 구매자 집단"
               + (f" — 대출 이용자가 {MIN_GROUP}명이 되도록 {len(g):,}명으로 넓혔습니다." if len(g) > 100 else ""))
    # 긴 범위 문자열이 잘리지 않도록 metric 대신 한 줄 텍스트로 표시 (나이·소득·매입가는 10~90% 범위)
    st.markdown(" · ".join(f"**{k}** {v}" for k, v in desc.items()) + "  *(나이·소득·매입가는 10~90% 범위)*")

    st.subheader("대출 조합 비중")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**비슷한 구매자 집단의 실제 비중**")
        for name, v in share.items():
            st.progress(float(v), text=f"{name}: {v:.0%}")
    with c2:
        st.markdown("**모델이 예측한 내 조건의 조합 확률**")
        for name, v in proba.items():
            st.progress(float(v), text=f"{name}: {v:.0%}")

    st.subheader("대출 조건 범위 — 중앙값 (25% ~ 75% 구간)")
    st.dataframe(tbl.T, width="stretch")
    st.caption("구매자들의 DSR (전체 대출) = (집을 살 때 받은 주담대·마통 + 그때 이미 갚고 있던 신용대출 등) 연 상환액 ÷ "
               "**가구** 연소득, 매입 당시 기준. 기존 대출까지 넣은 내 DSR과 같은 기준입니다 (소득은 구매자 = 가구소득).")

    if not compact:
        _render_my_prediction(R, inp, res, next(sec))
        render_tiers(R, inp, res)

    # ---- 희망 지역 시세와 대안 -----------------------------------------------------
    _render_region_alternatives(R, inp, res, next(sec))


def _render_my_prediction(R, inp, res, num):
    """내 조건 예측 (직접 입력 페이지용 — ②는 위의 '한눈에 보기'가 대신함)"""
    g, use_cl, rates, cash, p = (res[k] for k in ["g", "use_cl", "rates", "cash", "p"])
    existing = inp.get("existing_pay", 0.0)
    st.header(f"{num}. 내 조건으로 희망 주택을 산다면")
    combo_txt = COMBO_NAMES[2] if (use_cl or p.cl > 0) else COMBO_NAMES[1]
    st.caption(f"가정: {combo_txt} · 주담대 금리 {rates['mort']:.2f}% · 만기 {rates['term']:.0f}년 원리금균등"
               + (f" · 마통 금리 {rates['cl']:.2f}%" if p.cl > 0 else "")
               + " (금리는 입력값, 만기는 비슷한 구매자 집단의 최빈값)")
    dsr_label = "DSR (기존 대출 포함)" if existing > 0 else "DSR"

    m = st.columns(4)
    m[0].metric("예상 대출 금액", won(p.loan), help=f"검증 오차 약 ±{ERR_LOAN_PCT:.0%}")
    m[1].metric("월 상환액 (새 대출)", won(p.pay))
    m[2].metric(dsr_label, pct(p.dsr), help=f"검증 오차 약 ±{ERR_DSR * 100:.1f}%p")
    m[3].metric("월 잉여현금", won(p.surplus),
                help="연 소득 ÷ 12 − 월 소비액 − 새 대출 월상환" + (" − 기존 대출 월상환" if existing > 0 else ""))

    m = st.columns(4)
    m[0].metric(f"주택담보대출 (LTV {pct(p.mort / p.price)})", won(p.mort))
    m[1].metric("마이너스통장", won(p.cl))
    m[2].metric("자기자금", won(p.price - p.loan - p.shortfall), help=f"보유 자금 합계 {won(cash)}")
    m[3].metric("자금 부족액", won(p.shortfall) if p.shortfall > 0 else "없음")
    if existing > 0:
        st.caption(f"ℹ️ 기존 대출 월 상환액 {won(existing)}을 DSR과 월 잉여현금에 포함했습니다.")

    if p.need_loan > p.pattern_loan:
        st.caption(f"ℹ️ 비슷한 구매자 패턴대로면 대출은 약 {won(p.pattern_loan)}이지만, 보유 자금({won(cash)})으로 "
                   f"모자라는 {won(p.need_loan)}을 빌리는 것으로 계산했습니다. (보유 자금을 전부 쓰는 가정)")

    # 판정 (사유는 compute()에서 계산)
    reasons = res["reasons"]
    VERDICT_BOX[p.verdict](f"**판정: {VERDICT_TEXT[p.verdict]}**" + ("\n\n- " + "\n- ".join(reasons) if reasons else ""))


def _render_region_alternatives(R, inp, res, num):
    """희망 지역 시세와 대안 (판정이 '여유'가 아닐 때 대안 추천)"""
    u, p = res["u"], res["p"]
    sido, sigungu, area = inp["sido"], inp["sigungu"], inp["area"]
    basis, ctx = R.region_context(sido, sigungu)
    st.header(f"{num}. 희망 지역 시세와 대안")
    st.markdown(f"**{basis}** — 최근 5년 매입 {ctx['거래 건수']:,}건")
    c = st.columns(3)
    c[0].metric("매입가 중앙값", ctx["매입가"])
    c[0].caption(f"25~75% 구간: {ctx['매입가 구간']}")
    c[1].metric("전용면적 중앙값", ctx["전용면적"])
    c[1].caption(f"25~75% 구간: {ctx['전용면적 구간']}")
    est = ctx["평당가 중앙값"] * area / PYEONG
    c[2].metric(f"희망 면적 {area}㎡ 시세 추정", won(est), help="지역 평당가 중앙값 × 희망 평수")

    if p.verdict == "여유":
        st.success("희망 조건으로 감당 가능해 대안을 따로 추천하지 않습니다.")
    else:
        ok_price, comfy_price = res["afford"]
        if ok_price is None:
            st.error("입력한 소득·자산으로는 3,000만 원 이상 주택도 감당이 어렵다고 계산됩니다. "
                     "자금을 더 모으거나 소득 조건을 다시 확인해 주세요.")
        else:
            target = comfy_price or ok_price
            c = st.columns(3)
            c[0].metric("감당 가능한 최대 가격 (DSR 40% 이내)", won(ok_price))
            c[1].metric("여유 있는 최대 가격 (DSR 30% 이내)", won(comfy_price) if comfy_price else "-")
            c[2].metric(f"{won(target)}으로 가능한 면적",
                        f"{target / ctx['평당가 중앙값'] * PYEONG:.0f}㎡ ({target / ctx['평당가 중앙값']:.0f}평)",
                        help="희망 지역 평당가 중앙값 기준 추정")
            c[2].caption(f"{ctx['기준']} 평당가 기준")
            st.caption(f"아래 대안은 {won(target)} 이하 기준입니다.")

            alt = R.alternatives(u, target, sido)
            st.markdown(f"나와 **나이·소득·자산·소비가 비슷한 구매자 {alt['pool_n']:,}명**이 {won(target)} 이하로 산 집 기준 "
                        f"({MIN_GROUP}명 미만 구간은 숨김)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**많이 산 가격대**")
                st.dataframe(alt["price_bands"], width="stretch")
            with c2:
                st.markdown("**많이 산 면적대**")
                st.dataframe(alt["area_bands"], width="stretch")
            st.markdown(f"**추천 지역** — {alt['region_basis']}")
            if len(alt["regions"]):
                st.dataframe(alt["regions"], width="stretch")
            else:
                st.write(f"{MIN_GROUP}건 이상 모이는 지역이 없습니다.")


def render_model_trace(R, inp, res):
    """모델이 결과를 만드는 과정을 단계별로 보여줌 (사용자 본인 값과 집단 통계만 표시)"""
    g, u, proba, use_cl, rates, cash, p = (res[k] for k in ["g", "u", "proba", "use_cl", "rates", "cash", "p"])
    existing = inp.get("existing_pay", 0.0)
    st.header("🔍 모델 작동 과정")

    with st.expander("① 입력값을 모델 입력 변수로 변환", expanded=True):
        st.markdown("금액은 분포가 오른쪽으로 길어서 **log 변환**해서 넣고, 매입연도는 지금(2026)으로 둡니다.")
        x = u[NUM_FEATURES + ["sido"]].T.rename(columns={0: "모델 입력값"})
        x["모델 입력값"] = x["모델 입력값"].astype(str)
        st.dataframe(x, width="stretch")

    with st.expander("② kNN — 나와 비슷한 구매자 찾기"):
        z = R.models["scaler"].transform(u[KNN_FEATURES])[0]
        st.markdown(f"6개 변수({', '.join(KNN_FEATURES)})를 **표준화**(평균 0, 표준편차 1)한 뒤 거리가 가까운 순으로 고릅니다. "
                    f"100명에서 시작해 대출 이용자가 {MIN_GROUP}명 이상이 될 때까지 2배씩 늘리고, "
                    f"이번에는 **{len(g):,}명**에서 멈췄습니다.")
        st.dataframe(pd.DataFrame({"표준화 값": z.round(2)}, index=KNN_FEATURES).T, width="stretch")
        st.markdown(f"집단 안 대출 이용자 **{(g.combo > 0).sum()}명** → 금리 중앙값 {g[g.combo > 0].mortgage_rate.median():.2f}%, "
                    f"만기 최빈값 {rates['term']:.0f}년 (만기는 이 값을 그대로 사용)")

    with st.expander("③ GBM — 대출 조합·LTV·마통 비율 예측"):
        c = st.columns(3)
        c[0].markdown("**조합 확률 (분류 모델)**")
        c[0].dataframe(proba.map(lambda v: f"{v:.1%}").rename("확률"), width="stretch")
        c[0].caption(f"주담대만 vs 주담대+마통 중 확률이 큰 쪽 → **{'마통 포함' if use_cl else '주담대만'}**으로 계산")
        c[1].markdown("**주담대 LTV (회귀 모델)**")
        c[1].metric("예측 LTV", pct(p.ltv_pred), help=f"0~{LTV_CAP:.0%} 사이로 자름")
        c[2].markdown("**마통 / 매입가 (회귀 모델)**")
        c[2].metric("예측 마통 비율", pct(p.cl_ratio_pred) if use_cl else "사용 안 함",
                    help=f"0~{CL_MAX_RATIO:.0%} 사이로 자름")

    with st.expander("④ 대출 금액 결정 (규칙)"):
        rows = [
            ("비슷한 구매자 패턴 대출", f"희망가 {won(p.price)} × (LTV {pct(p.ltv_pred)} + 마통 {pct(p.cl_ratio_pred)})",
             won(p.pattern_loan)),
            ("보유 자금으로 부족한 금액", f"희망가 {won(p.price)} − 보유 자금 {won(cash)}", won(p.need_loan)),
            (("빌릴 금액", "필요한 만큼만 (비슷한 사람 패턴은 비교용)", won(p.need_loan))
             if inp.get("basis") == "need" else
             ("빌릴 금액", "위 두 값 중 큰 값", won(max(p.pattern_loan, p.need_loan)))),
            ("→ 주담대", f"LTV {LTV_CAP:.0%} 한도({won(p.price * LTV_CAP)})까지", won(p.mort)),
            ("→ 마이너스통장", f"나머지, 매입가 {CL_MAX_RATIO:.0%} 한도({won(p.price * CL_MAX_RATIO)})까지", won(p.cl)),
            ("→ 자금 부족", "대출 한도로도 못 채운 금액", won(p.shortfall)),
        ]
        st.table(pd.DataFrame(rows, columns=["단계", "계산", "금액"]).set_index("단계"))

    with st.expander("⑤ 월 상환·DSR·판정 (공식과 규칙)"):
        mp = loan_payment(p.mort, 0, rates)
        cp = loan_payment(0, p.cl, rates)
        rows = [
            ("주담대 월상환", f"원리금균등: {won(p.mort)}, 연 {rates['mort']:.2f}%, {rates['term']:.0f}년", won(mp)),
            ("마통 월상환", f"잔액 × (월금리 {rates['cl']:.2f}%/12 + 0.3%)", won(cp)),
            ("기존 대출 월상환", "마이데이터 부채 요약" if existing > 0 else "없음", won(existing)),
            ("DSR", f"({won(p.pay)} + {won(existing)}) × 12 ÷ 가구 연소득 {won(inp['income'])}", pct(p.dsr)),
            ("월 잉여현금", f"월소득 {won(inp['income'] / 12)} − 소비 {won(inp['consumption'])} − 상환 {won(p.pay + existing)}",
             won(p.surplus)),
        ]
        st.table(pd.DataFrame(rows, columns=["항목", "계산", "값"]).set_index("항목"))
        checks = [
            (f"DSR > {DSR_WARN:.0%} (새 대출이 있을 때)", p.loan > 0 and p.dsr > DSR_WARN, "어려움"),
            ("월 잉여현금 < 0", p.surplus < 0, "어려움"),
            ("자금 부족 > 0", p.shortfall > 0, "어려움"),
            (f"DSR {DSR_OK:.0%}~{DSR_WARN:.0%} (새 대출이 있을 때)", p.loan > 0 and DSR_OK < p.dsr <= DSR_WARN, "주의"),
        ]
        st.table(pd.DataFrame([(n, "✅ 해당" if hit else "—", v) for n, hit, v in checks],
                              columns=["판정 규칙", "결과", "해당 시 판정"]).set_index("판정 규칙"))
        st.markdown(f"→ 최종 판정: **{p.verdict}** (어려움 규칙이 하나라도 해당하면 어려움, 아니면 주의 규칙, 둘 다 아니면 여유)")
