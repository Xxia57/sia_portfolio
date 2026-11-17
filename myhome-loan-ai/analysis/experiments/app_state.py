# =============================================================================
# 화면 공통 상태 — 모든 페이지가 같이 쓰는 '내 정보'와 '계산 가정'
#   profile()      : 선택한 고객의 마이데이터 + '내 재무 정보'에서 직접 고친 값(덮어쓰기)
#   assumptions()  : 금리·만기·목표 여유자금·예비비·기준 면적·금리+1%p
#   build_inp()    : 위 둘로 계산 입력(dict)을 만듦
#   zone_bar()     : 안정권·가능권·한계권·범위 밖 막대 (HTML)
#   na()           : '정보 없음, ○○ 필요' 표시
#   chat_stub()    : 하단 '조건을 말로 바꿔 보세요' — LLM 연동 전이라 안내만
# =============================================================================
import streamlit as st

from prep import DSR_WARN, LTV_CAP, user_frame
from recommender import (FEATURES, TERMS, ZONE_DESC, ZONES, dsr_limit_loan, existing_dsr, loan_payment, pct,
                         recommend_loan, won)
from ui import current_customer, get_recommender

# 구간 색 — 안정권(진한 파랑)·가능권(연한 파랑)·한계권(주황)·범위 밖(회색). 색과 함께 항상 글자로 구간명을 표시
ZONE_COLOR = {"안정권": "#2a78d6", "가능권": "#86b6ef", "한계권": "#eb6834", "범위 밖": "#c9ced6"}
EDITABLE = {  # 내 재무 정보에서 고칠 수 있는 값: (이름, 단위 설명)
    "income": "연 소득 (적용 기준)", "consumption": "월 평균 지출", "assets": "금융자산",
    "acct_total": "계좌 잔액 합계", "other_funds": "기타 동원 자금", "existing_pay": "기존 대출 월 상환",
}


def assumptions():
    """계산 가정 (세션 동안 유지). 기본값: 금리 = 데이터 중앙값, 만기 = 데이터 최빈값"""
    if "assume" not in st.session_state:
        base = get_recommender().default_rates()
        st.session_state.assume = dict(mort_rate=round(base["mort"], 2), cl_rate=round(base["cl"], 2),
                                       term=int(base["term"]), goal=50e4, reserve=1000e4,
                                       base_area=59, shock=False)
    return st.session_state.assume


def profile():
    """선택한 고객의 정보. 직접 고친 값이 있으면 그 값을 쓰고, 출처(sources)를 함께 돌려줌"""
    cur = current_customer()
    if cur is None or "annual_income" not in cur[1].index:
        return None
    cid, me, accts = cur
    g = me.get
    base = dict(
        income=float(me.annual_income), consumption=float(me.consumption),
        assets=float(me.financial_assets_estimated), acct_total=float(me.acct_total),
        other_funds=0.0, existing_pay=float(me.existing_pay),
    )
    ov = st.session_state.setdefault("overrides", {}).get(cid, {})
    prof = {**base, **ov}
    prof.update(
        cid=cid, name=g("name"), age=int(me.age), household_size=int(me.household_size),
        residence=str(me.residence_region), residence_sido=str(me.residence_sido),
        income_basis=str(me.income_basis), primary=float(me.primary_annual_income),
        household=float(me.household_annual_income), card_spend=g("monthly_card_budget"),
        edu_cost=g("monthly_private_education_cost"), checking=float(g("bal_CHECKING", 0)),
        savings=float(g("bal_SAVINGS", 0)), debt_total=float(me.total_financial_debt),
        cl_balance=float(me.credit_line_balance), pl_balance=float(me.personal_loan_balance),
        accounts=accts, base=base,
        sources={k: ("직접 입력" if k == "other_funds" else "직접 수정") if k in ov else
                 ("직접 입력" if k == "other_funds" else "마이데이터") for k in base},
    )
    return prof


def assume_input(fn, label, field, scale=1.0, cast=float, **kw):
    """계산 가정을 바꾸는 입력 칸. 값이 바뀌면 콜백으로 먼저 저장 → 다시 그릴 때 바로 새 값으로 계산됨
    (위젯을 계산보다 아래에 둬도 한 박자 늦지 않음). 다른 페이지와 같은 assumptions()를 공유"""
    a = assumptions()
    key = f"as_{field}"
    # 그릴 때마다 저장값으로 맞춤 — 페이지를 옮기면 Streamlit이 같은 키라도 새 위젯으로 보고 값을 버리기 때문
    # (값을 바꾸면 콜백이 먼저 저장하므로 방금 바꾼 값이 유지됨)
    st.session_state[key] = cast(a[field] / scale)

    def _save():
        a[field] = st.session_state[key] * scale
    return fn(label, key=key, on_change=_save, **kw)


def profile_input(prof, label, field, help_note=""):
    """내 재무 값(만원)을 바꾸는 입력 칸. 마이데이터 값과 다르면 '직접 수정'으로 저장 (내 재무 정보 페이지와 공유)"""
    cid = prof["cid"]
    key = f"pf_{cid}_{field}"
    st.session_state[key] = int(round(prof[field] / 1e4))   # 그릴 때마다 현재 값으로 맞춤 (위와 같은 이유)
    base = prof["base"][field]

    def _save():
        v = st.session_state[key] * 1e4
        ov = st.session_state.setdefault("overrides", {}).setdefault(cid, {})
        if abs(v - base) < 1e4:
            ov.pop(field, None)
        else:
            ov[field] = v
    return st.number_input(label, 0, 10_000_000, step=10, key=key, on_change=_save,
                           help=f"마이데이터 값 {won(base)}" + (f" · {help_note}" if help_note else ""))


def reset_inputs(cid=None):
    """입력 칸 상태를 지워 기본값으로 되돌림 (cid 있으면 그 고객의 재무 값, 없으면 계산 가정)"""
    if cid is not None:
        st.session_state.setdefault("overrides", {}).pop(cid, None)
        prefix = f"pf_{cid}_"
    else:
        st.session_state.pop("assume", None)
        prefix = "as_"
    for k in [k for k in st.session_state if str(k).startswith(prefix)]:
        del st.session_state[k]


def rates(a, shock=None):
    """계산 금리 (금리 +1%p 토글이 켜져 있으면 +1%p)"""
    s = 1.0 if (a["shock"] if shock is None else shock) else 0.0
    return {"mort": a["mort_rate"] + s, "cl": a["cl_rate"] + s, "term": a["term"]}


def cash_of(prof, a):
    """집 사는 데 쓸 수 있는 돈 = 계좌 잔액 − 예비비 (기타 동원 자금은 계산 때 따로 더함)"""
    return max(prof["acct_total"] - a["reserve"], 0.0)


def build_inp(prof, a, price=5e8, area=None, sido=None, sigungu=None, shock=None):
    r = rates(a, shock)
    return dict(age=prof["age"], income=max(prof["income"], 1e6), assets=prof["assets"],
                cash=cash_of(prof, a), other_funds=prof["other_funds"], consumption=prof["consumption"],
                existing_pay=prof["existing_pay"], price=price, area=area or a["base_area"],
                sido=sido or prof["residence_sido"], sigungu=sigungu or "", goal=a["goal"],
                mort_rate=r["mort"], cl_rate=r["cl"], term=r["term"],
                basis="need")   # 마이데이터 화면: 필요한 만큼만 빌리는 기준 (비슷한 사람 패턴은 비교용)


def zone_bar(bounds, marker=None, height=16, show_ticks=True):
    """구간 막대 HTML. bounds = {'안정권': 상단, '가능권': 상단, '한계권': 상단(=계산상 최대)}"""
    tops = [bounds.get(z) or 0 for z in ZONES[:3]]
    tops = [max(tops[0], 0), max(tops[1], tops[0]), max(tops[2], tops[1], tops[0])]
    axis = max(tops[2] * 1.3, (marker or 0) * 1.1, 1e8)
    edges = [0] + tops + [axis]
    segs, ticks = [], []
    for i, z in enumerate(ZONES):
        w = (edges[i + 1] - edges[i]) / axis * 100
        if w <= 0:
            continue
        segs.append(f'<div title="{z}" style="width:{w:.2f}%;background:{ZONE_COLOR[z]};height:{height}px"></div>')
    if show_ticks:
        for t in tops:
            if t > 0:
                ticks.append(f'<span style="position:absolute;left:{t / axis * 100:.2f}%;transform:translateX(-50%);'
                             f'font-size:11px;opacity:.75">{won(t)}</span>')
    mark = ""
    if marker:
        mark = (f'<div style="position:absolute;left:{min(marker / axis, 1) * 100:.2f}%;top:-16px;transform:translateX(-50%);'
                f'font-size:11px;font-weight:700">이 집</div>'
                f'<div style="position:absolute;left:{min(marker / axis, 1) * 100:.2f}%;top:-3px;width:2px;'
                f'height:{height + 6}px;background:currentColor"></div>')
    return (f'<div style="position:relative;margin:{18 if marker else 4}px 0 {18 if show_ticks else 4}px">'
            f'<div style="display:flex;gap:2px;border-radius:8px;overflow:hidden">{"".join(segs)}</div>{mark}'
            f'<div style="position:relative;height:14px;margin-top:4px">{"".join(ticks)}</div></div>')


def zone_legend(bounds):
    """구간 설명 (색 + 이름 + 가격 범위 + 의미)"""
    s, p, c = bounds["안정권"], bounds["가능권"], bounds["한계권"]
    ranges = {
        "안정권": f"{won(s)} 이하" if s else "해당 가격 없음",
        "가능권": f"{won(s) if s else '0'} ~ {won(p)}" if p and p != s else "해당 가격 없음",
        "한계권": f"{won(p) if p else '0'} ~ {won(c)}" if c and c != p else "해당 가격 없음",
        "범위 밖": f"{won(c)} 초과" if c else "모든 가격",
    }
    cols = st.columns(4)
    for col, z in zip(cols, ZONES):
        col.markdown(f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                     f'background:{ZONE_COLOR[z]}"></span> **{z}**  \n{ranges[z]}  \n'
                     f'<span style="font-size:12px;opacity:.7">{ZONE_DESC[z]}</span>', unsafe_allow_html=True)


def zone_badge(z):
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;'
            f'border:1px solid {ZONE_COLOR[z]};color:{"#1f2328" if z in ("가능권", "범위 밖") else ZONE_COLOR[z]};'
            f'background:{ZONE_COLOR[z]}22">● {z}</span>')


def benchmark(row):
    """GBM 패턴(비슷한 사람들이 보통 빌린 비율)과 실제로 빌려야 하는 비율 비교 문구
    row: plan() 결과 한 줄 (price, pattern_loan, loan, shortfall)"""
    usual, need = row.pattern_loan / row.price, (row.loan + row.shortfall) / row.price
    if need > usual + 0.005:
        msg = (f"👥 비슷한 사람들은 집값의 **{usual:.0%}\u200b**를 빌렸는데, 이 집은 **{need:.0%}\u200b**를 빌려야 해요 "
               f"(**{won(row.loan + row.shortfall - row.pattern_loan)}** 더). 또래보다 자기자금이 그만큼 부족하다는 뜻이에요.")
        if row.shortfall > 0:
            msg += f" 대출 한도(집값의 82%)를 넘어서 {won(row.shortfall)}은 빌릴 수도 없어요."
        return msg
    if need < usual - 0.005:
        return (f"👥 비슷한 사람들은 집값의 **{usual:.0%}**({won(row.pattern_loan)})를 빌렸지만, 보유 자금이 충분해서 "
                f"**필요한 {need:.0%}\u200b**만 빌리는 것으로 계산했어요. 덜 빌린 만큼 이자가 줄어요.")
    return f"👥 비슷한 사람들도 집값의 **{usual:.0%}**쯤 빌렸어요. 필요한 금액과 거의 같아요."


def dsr_block_notice(prof, cash):
    """기존 대출만으로 DSR 40%를 넘으면 '새 대출 어려움' 안내를 그리고 True 반환 (안 넘으면 아무것도 안 그림)"""
    income = max(prof["income"], 1e6)
    d0 = existing_dsr(income, prof["existing_pay"])
    if d0 <= DSR_WARN:
        return False
    limit = DSR_WARN * income / 12                     # DSR 40%가 되는 월 상환액
    st.warning(f"🔒 기존 대출 상환만으로 DSR이 **{d0:.1%}​**예요. 지금은 **새 대출이 어려워요** (DSR 40% 초과). "
               f"그래서 보유 자금 {won(cash)}으로 **대출 없이 살 수 있는 집**까지만 계산했어요.  \n"
               f"기존 대출 월 상환을 **{won(prof['existing_pay'] - limit)}** 이상 줄이면(월 {won(limit)} 이하) "
               "DSR이 40% 아래로 내려가 새 대출을 계산할 수 있어요.")
    return True


# 계산에 쓰는 값의 출처 — 어떤 값이 데이터에서 왔고, 어떤 값이 개발 중 임의로 정한 가정인지 밝힘
SOURCE_TAG = {"data": "🧪 데이터", "guess": "✋ 임의 가정", "rule": "📜 규제 단순화",
              "design": "🛠 설계 선택", "missing": "⛔ 미반영"}


def assumption_sources(a):
    """(항목, 현재 값, 출처, 실제와 다른 점) 목록"""
    return [
        ("주담대 금리", f"{a['mort_rate']:.2f}%", "data", "데이터 중앙값. 실제 금리는 신용점수·은행·우대조건에 따라 달라요"),
        ("마이너스통장 금리", f"{a['cl_rate']:.2f}%", "data", "데이터 중앙값. 실제 금리는 사람마다 달라요"),
        ("주담대 만기", f"{a['term']}년", "data", "데이터 최빈값. 추천 카드는 안정권을 지키는 가장 짧은 만기를 따로 골라요"),
        ("목표 여유자금 (안정권 기준)", f"월 {won(a['goal'])}", "guess",
         "근거 없이 정한 값이에요. 이 값에 따라 '안정권 상단' 금액이 크게 바뀌어요"),
        ("예비비", won(a["reserve"]), "guess", "비상금으로 얼마를 남길지는 사람마다 달라요"),
        ("기준 전용면적", f"{a['base_area']}㎡", "guess", "내 구매력 화면에서 '비슷한 사람' 모델 예측에만 쓰여요"),
        ("DSR 주의 기준", "30%", "guess", "규제가 아니라 '여유'와 '주의'를 나누려고 정한 값이에요"),
        ("금리 +1%p 시나리오", "주담대·마통 모두 오름", "guess", "모두 변동금리라고 가정했어요. 고정금리면 부담이 과장돼요"),
        ("DSR 상한", "40% (기존 대출 포함)", "rule",
         "실제로는 금융권·대출 규모에 따라 적용 여부와 한도가 달라요. 스트레스 DSR 가산금리는 반영하지 않았어요"),
        ("주담대 LTV 한도", f"{LTV_CAP:.0%}", "rule",
         "데이터에서 관찰된 최대치예요. 실제 한도는 지역·주택 수·생애최초 여부 등에 따라 달라요"),
        ("마이너스통장 한도", "집값의 12%", "rule",
         "데이터에서 관찰된 최대치예요. 실제 마통 한도는 집값이 아니라 소득·신용으로 정해져요"),
        ("보유 자금", "계좌 잔액 − 예비비", "design", "계좌 밖 투자자산은 빠져요. '기타 동원 자금'에 직접 넣을 수 있어요"),
        ("추천 방식", "필요한 만큼만 빌림", "design", "현금을 최대한 쓰고 덜 빌리는 보수적인 방식이에요"),
        ("DSR 적용", "새 대출이 있을 때만", "design", "대출 없이 현금으로 사는 집은 DSR로 막지 않아요"),
        ("부대비용", "미반영", "missing", "취득세·중개보수·등기비가 빠져 있어 실제로 필요한 돈은 더 많아요"),
    ]


def assumption_note(a):
    """구매력·추천 화면 아래에 붙이는 짧은 안내 — 임의 가정과 단순화한 규제를 밝힘"""
    st.caption(f"✋ 안정권 기준(매달 {won(a['goal'])} 남기기)과 예비비 {won(a['reserve'])}은 **임의로 정한 가정**이에요. "
               f"📜 DSR 40%·LTV {LTV_CAP:.0%}·마통 12%는 실제 규제를 **단순화**한 값이고, ⛔ 부대비용(취득세 등)은 빠져 있어요.")
    st.page_link("views/assumptions.py", label="값의 출처와 실제와 다른 점 보기", icon="📌")


def na(what, why):
    """만들 수 없는 정보 표시: '정보 없음 — ○○ 필요 / ○○에 걸림'"""
    st.markdown(f'<div style="padding:8px 12px;border:1px dashed #9aa4b2;border-radius:8px;font-size:13px;opacity:.85">'
                f'<b>{what}</b> · 정보 없음 — {why}</div>', unsafe_allow_html=True)


def data_chip():
    st.caption(f"🧪 합성 데이터 기준 {get_recommender().data_date()}")


def chat_stub():
    q = st.chat_input("조건을 말로 바꿔 보세요")
    if q:
        st.info(f"💬 \"{q}\"\n\n정보 없음 — 말로 조건을 바꾸려면 **LLM(예: Claude API) 연동이 필요**해요. "
                "지금은 '계산 가정'과 '내 재무 정보'에서 직접 바꿔 주세요.")


def render_recommendation(R, prof, a, price, area, sido, title=None):
    """💡 추천 대출 (필요한 만큼만 · 주담대 먼저 · 안정권을 지키는 가장 짧은 만기) + 👥 비슷한 사람은 어떻게?(클릭)"""
    r = rates(a)
    cash = cash_of(prof, a) + prof["other_funds"]
    income = max(prof["income"], 1e6)
    rec, T = recommend_loan(price, cash, income, prof["consumption"], prof["existing_pay"], r, a["goal"])
    with st.container(border=True):
        st.subheader(title or f"💡 추천 대출 — {won(price)} 집")
        if rec["need"] <= 0:
            st.success(f"대출 없이 살 수 있어요. 보유 자금 {won(cash)}으로 집값 {won(price)}을 채워요.")
            return
        combo = f"주담대 **{won(rec['mort'])}** ({rec['term']}년)" + (f" + 마통 **{won(rec['cl'])}**" if rec["cl"] > 0 else "")
        st.markdown(f"#### {combo}")
        st.markdown(f"{'주담대 + 마이너스통장' if rec['cl'] > 0 else '주담대만'} · 대출 합계 {won(rec['mort'] + rec['cl'])} "
                    f"(집값의 {(rec['mort'] + rec['cl']) / price:.0%}) &nbsp; {zone_badge(rec['zone'])}", unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("월 상환", won(rec["pay"]), help=f"주담대 {won(rec['mort_pay'])} + 마통 {won(rec['cl_pay'])}")
        m[1].metric("DSR", pct(rec["dsr"]), help="기존 대출 포함")
        m[2].metric("매달 남는 돈", won(rec["surplus"]), help=f"목표 여유자금 {won(a['goal'])}")
        m[3].metric("주담대 총이자", won(rec["interest"]), help="금리가 만기까지 그대로일 때")
        if rec["shortfall"] > 0:
            st.error(f"대출 한도(LTV 70% + 마통 12%)를 다 써도 {won(rec['shortfall'])}이 부족해요.")
        if existing_dsr(income, prof["existing_pay"]) > DSR_WARN:
            st.error("기존 대출만으로 DSR이 40%를 넘어서, 이 대출은 실제로 받기 어려워요. 기존 대출을 먼저 줄여야 해요.")
        elif rec["dsr"] > DSR_WARN:
            st.error(f"새 대출까지 더하면 DSR이 {pct(rec['dsr'])}로 40%를 넘어서, 이 금액은 실제로 받기 어려워요.")
        why = [f"**필요한 만큼만**: 집값 {won(price)} − 보유 자금 {won(cash)} = {won(rec['need'])}",
               f"**싼 대출부터**: 주담대(연 {r['mort']:.2f}%)를 LTV 70%까지 먼저"
               + (f", 넘는 {won(rec['cl'])}만 마통(연 {r['cl']:.2f}%)" if rec["cl"] > 0 else " — 마통 필요 없음")]
        shorter = [t for t in TERMS if t < rec["term"]]
        if rec["all_terms_safe"]:
            why.append(f"**만기 {rec['term']}년**: 매달 {won(a['goal'])} 이상 남기는(안정권) 가장 짧은 만기라 총이자가 가장 적어요"
                       + (f" — {shorter[-1]}년이면 남는 돈 {won(T.set_index('term').surplus[shorter[-1]])}" if shorter else ""))
        else:
            why.append(f"**만기 {rec['term']}년**: 어떤 만기로도 안정권이 안 돼서, 남는 돈이 가장 큰 만기를 골랐어요")
        if rec["cl"] > 0:
            why.append(f"**마통 먼저 갚기**: 남는 돈 중 {won(rec['extra'])}씩 더 갚으면 약 **{rec['cl_months']}개월** 뒤 마통 상환 완료 "
                       f"(그동안 마통 이자 약 {won(rec['cl_interest'])})" if rec["cl_months"] else
                       "**마통 상환**: 목표 여유자금보다 남는 돈이 적어 추가 상환 여력이 없어요 — 금리가 높아 부담이 커요")
        st.markdown("\n".join(f"- {w}" for w in why))
        with st.expander("만기별 비교"):
            t = T.assign(**{"만기": T.term.map(lambda v: f"{v}년" + (" ⭐" if v == rec["term"] else "")),
                            "월 상환": T.pay.map(won), "남는 돈": T.surplus.map(won), "DSR": T.dsr.map(pct),
                            "주담대 총이자": T.interest.map(won), "구간": T.zone})
            st.dataframe(t[["만기", "월 상환", "남는 돈", "DSR", "주담대 총이자", "구간"]], hide_index=True, width="stretch")
        na("정책대출(디딤돌·보금자리론 등)", "정책대출 자격(소득·주택가격·무주택 기간)과 한도·금리 규정이 필요해요.")
        st.caption("📜 한도(LTV 70%·마통 12%·DSR 40%)는 실제 규제를 단순화한 값이에요. "
                   "⛔ 부대비용(취득세·중개보수·등기비)이 빠져 있어 실제로 필요한 대출은 이보다 많을 수 있어요.")

    # ---- 👥 비슷한 사람은 어떻게? (클릭) ------------------------------------------------
    with st.expander("👥 비슷한 사람은 어떻게 빌렸을까?"):
        inp = build_inp(prof, a, price=price, area=area, sido=sido)
        u = user_frame(inp["income"], inp["assets"], inp["consumption"], price, area, inp["age"], sido, R.cats,
                       inp["existing_pay"])
        proba = R.models["combo"].predict_proba(u[FEATURES])[0]
        use_cl = proba[2] >= proba[1]
        # 비슷한 사람들이 '실제로 빌린 비율 그대로' (GBM 예측) — 필요액과 상관없이
        sim = R.plan_many({**inp, "basis": "pattern"}, [price], [area], [sido], r, use_cl, cash).iloc[0]
        s_m, s_c = price * sim.ltv_pred, price * sim.cl_ratio_pred
        mx = dsr_limit_loan(price, income, prof["existing_pay"], r)
        mx_m = min(mx, price * LTV_CAP)
        mx_c = max(mx - mx_m, 0)

        def row(name, m, c, term):
            rr = {**r, "term": term}
            pay = loan_payment(m, c, rr)
            return {"방식": name, "주담대": won(m), "마통": won(c), "만기": f"{term}년",
                    "필요 자기자금": won(max(price - m - c, 0)), "월 상환": won(pay),
                    "남는 돈": won(income / 12 - prof["consumption"] - pay - prof["existing_pay"]),
                    "DSR": pct((pay + prof["existing_pay"]) * 12 / income)}
        st.dataframe([row("💡 추천 (필요한 만큼만)", rec["mort"], rec["cl"], rec["term"]),
                      row("👥 비슷한 사람들처럼", s_m, s_c, r["term"]),
                      row("⬆️ 최대로 (DSR 40%)", mx_m, mx_c, r["term"])], hide_index=True, width="stretch")
        st.markdown(f"**비슷한 사람들**은 이 조건에서 보통 집값의 **{sim.ltv_pred:.0%}​**를 주담대로"
                    + (f", **{sim.cl_ratio_pred:.0%}​**를 마통으로" if use_cl else "(마통 없이)")
                    + f" 빌렸어요 (GBM 예측, 마통 병행 확률 {proba[2]:.0%}).")
        gap = (s_m + s_c) - rec["need"]
        if gap < -1e5:
            st.markdown(f"그렇게만 빌리면 자기자금이 **{won(price - s_m - s_c)}** 필요한데 보유 자금은 {won(cash)}이라 "
                        f"**{won(-gap)}이 모자라요**. 그래서 추천은 그만큼 더 빌려요.")
        elif gap > 1e5:
            st.markdown(f"필요한 금액보다 **{won(gap)} 더 빌리고** 그만큼 현금을 남기는 방식이에요. 이자는 늘지만 손에 쥔 돈이 많아져요.")
        else:
            st.markdown("필요한 금액과 거의 같아요.")
        st.caption("'최대로'는 DSR 40% 안에서 빌릴 수 있는 최대치(LTV 70% + 마통 12% 한도 안)예요. 필요액보다 많으면 남는 돈은 현금으로 들고 있게 돼요.")
