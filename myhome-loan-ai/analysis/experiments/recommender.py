# =============================================================================
# 추천 계산 로직 (화면과 분리 — app.py가 이 모듈을 호출)
#   1. 비슷한 구매자 집단(kNN) 통계: 대출 조합 비중, 대출금액·LTV·금리·월상환·DSR·자기자금비율 범위
#   2. 내 조건 예측: GBM(조합 확률, LTV, 마통 비율) + 집단 중앙값(금리, 만기) → 월상환·DSR·판정
#   3. 대안: 감당 가능한 최대 가격 역산 → 비슷한 사람들이 그 가격 이하에서 많이 산 가격대·면적·지역
#   ※ 개인정보 보호: 개별 행은 절대 반환하지 않고, MIN_GROUP(30)명 이상 집단의 통계만 반환
# =============================================================================
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from prep import (CL_EXTRA, COMBO_NAMES, DSR_OK, DSR_WARN, KNN_FEATURES, LTV_CAP, MIN_GROUP, MODEL_DIR,
                  NUM_FEATURES, load_purchases, monthly_credit_line, monthly_mortgage, user_frame)

FEATURES = NUM_FEATURES + ["sido"]
PERSON_FEATURES = ["age_at_purchase", "log_income", "log_assets", "log_consumption"]  # 대안 추천용 (집 조건 제외)
CL_MAX_RATIO = 0.12       # 마통 한도: 데이터 99% 분위수 (매입가의 12%)
PYEONG = 3.3058           # 1평 = 3.3058㎡
# 3단계 검증 오차 (GBM, 검증 A) — 화면에 '예측 오차 범위'로 표시
ERR_DSR = 0.028
ERR_LOAN_PCT = 0.12
# 대출 구간 (이름, DSR 상한, 표시 범위, 설명) — 각 구간은 DSR 상한까지 빌리는 금액으로 계산
LOAN_TIERS = [
    ("보수", 0.25, "20~25%", "생활비·저축 여유 충분"),
    ("권장", 0.30, "약 30%", "흔히 말하는 \"적정선\""),
    ("최대", 0.40, "35~40%", "규제상 가능하지만 빠듯함"),
]
RATE_SHOCKS = [1.0, 2.0]  # 금리 상승 부담 (%p) — 주담대·마통 모두 변동금리로 가정


def loan_dsr(dsr, loan=None):
    """판정에 쓰는 DSR. DSR은 '새 대출을 받을 때' 적용되는 규제라, 새 대출이 0원(현금 구입)이면 0으로 봄
    (기존 대출만으로 DSR 40%를 넘어도 대출 없이 사는 집은 DSR로 막지 않음). loan=None이면 그대로"""
    dsr = np.asarray(dsr, dtype=float)
    return dsr if loan is None else np.where(np.asarray(loan) > 0, dsr, 0.0)


def existing_dsr(income, existing_pay):
    """기존 대출만의 DSR — 40%를 넘으면 새 대출이 어려움"""
    return existing_pay * 12 / max(income, 1.0)


def judge(dsr, surplus, shortfall, loan=None):
    """감당 가능 판정: 어려움(DSR>40% 또는 월 적자 또는 자금 부족) / 주의(DSR 30~40%) / 여유
    loan: 새 대출 금액 — 0이면 DSR 기준을 적용하지 않음"""
    dsr = loan_dsr(dsr, loan)
    surplus, shortfall = map(np.asarray, (surplus, shortfall))
    return np.where((dsr > DSR_WARN) | (surplus < 0) | (shortfall > 0), "어려움",
                    np.where(dsr > DSR_OK, "주의", "여유"))


# 구매력 구간 (내 구매력·지역 찾기 화면) — 월 잉여현금과 목표 여유자금 기준
#   범위 밖: 새 대출의 DSR 40% 초과 또는 대출 한도(LTV 70% + 마통 12%)로도 자금 부족 → 계산상 살 수 없음
#            (대출 없이 현금으로 사는 집은 DSR로 막지 않음)
#   한계권: 한도 안이지만 지금 지출을 유지하면 매달 적자
#   가능권: 남는 돈이 0 이상이지만 목표 여유자금보다 적음
#   안정권: 남는 돈이 목표 여유자금 이상
ZONES = ["안정권", "가능권", "한계권", "범위 밖"]
ZONE_DESC = {"안정권": "지금 소비 유지 가능", "가능권": "목표보다 적게 남음",
             "한계권": "지금 지출이면 적자", "범위 밖": "계산상 최대 초과"}


def zone_of(dsr, surplus, shortfall, goal, loan=None):
    dsr = loan_dsr(dsr, loan)
    surplus, shortfall = map(np.asarray, (surplus, shortfall))
    return np.select([(dsr > DSR_WARN) | (shortfall > 0), surplus < 0, surplus < goal],
                     ["범위 밖", "한계권", "가능권"], "안정권")


def dsr_limit_loan(price, income, existing_pay, rates):
    """DSR 40% 안에서 빌릴 수 있는 최대 대출 (주담대 LTV 70% 먼저 → 마통 12%)"""
    a = float(monthly_mortgage(1.0, rates["mort"], rates["term"]))
    b = float(monthly_credit_line(1.0, rates["cl"]))
    budget = max(DSR_WARN * income / 12 - existing_pay, 0.0)
    mort_cap, cl_cap = price * LTV_CAP, price * CL_MAX_RATIO
    if budget <= a * mort_cap:
        return budget / a
    return mort_cap + min((budget - a * mort_cap) / b, cl_cap)


TERMS = [20, 25, 30, 35, 40]


def recommend_loan(price, cash, income, consumption, existing_pay, rates, goal):
    """추천 대출: ① 필요한 만큼만(집값 − 보유 자금) ② 주담대 먼저, LTV 70% 넘는 부분만 마통(12%까지)
    ③ 만기: 남는 돈 ≥ 목표 여유자금(안정권)을 지키는 가장 짧은 만기 — 없으면 남는 돈이 가장 큰 만기
    반환: 추천 1건(dict) + 만기별 비교표(DataFrame)"""
    need = max(price - cash, 0.0)
    mort = min(need, price * LTV_CAP)
    cl = min(need - mort, price * CL_MAX_RATIO)
    short = need - mort - cl
    cl_pay = float(monthly_credit_line(cl, rates["cl"]))
    rows = []
    for t in TERMS:
        mp = float(monthly_mortgage(mort, rates["mort"], t)) if mort > 0 else 0.0
        pay = mp + cl_pay
        surplus = income / 12 - consumption - pay - existing_pay
        dsr = (pay + existing_pay) * 12 / income
        rows.append(dict(term=t, mort_pay=mp, pay=pay, surplus=surplus, dsr=dsr,
                         interest=mp * t * 12 - mort, zone=str(zone_of(dsr, surplus, short, goal, mort + cl))))
    T = pd.DataFrame(rows)
    ok = T[T.zone == "안정권"]
    pick = ok.iloc[0] if len(ok) else T.loc[T.surplus.idxmax()]
    # 마통은 금리가 높아 먼저 갚기: 매달 '남는 돈 − 목표 여유자금'을 더 넣으면 몇 개월 걸리나
    months, cl_interest = None, 0.0
    extra = max(pick.surplus - goal, 0.0)
    if cl > 0 and extra > 0:
        bal, r = cl, rates["cl"] / 1200
        for m in range(1, 601):              # 매달: 이자 + 원금 0.3%(기본 상환) + 여유분 추가 상환
            cl_interest += bal * r
            bal -= bal * CL_EXTRA + extra
            if bal <= 0:
                months = m
                break
    rec = dict(price=price, need=need, mort=mort, cl=cl, shortfall=short, term=int(pick.term),
               pay=pick.pay, mort_pay=pick.mort_pay, cl_pay=cl_pay, surplus=pick.surplus, dsr=pick.dsr,
               zone=pick.zone, interest=pick.interest, cl_months=months, cl_interest=cl_interest,
               extra=extra, all_terms_safe=len(ok) > 0)
    return rec, T


def split_loan(loan, price):
    """대출액을 주담대(LTV 70%까지) → 마통(매입가 12%까지) 순으로 나눔. 한도를 넘는 부분은 버림"""
    mort = min(loan, price * LTV_CAP)
    return mort, min(loan - mort, price * CL_MAX_RATIO)


def loan_payment(mort, cl, rates, shock=0.0):
    """주담대(원리금균등)+마통 월 상환액. shock: 금리 상승 폭(%p)"""
    return float(monthly_mortgage(mort, rates["mort"] + shock, rates["term"])
                 + monthly_credit_line(cl, rates["cl"] + shock))


def iqr_text(s, fmt):
    """'중앙값 (25%~75%)' 문자열"""
    q = s.quantile([0.5, 0.25, 0.75])
    return f"{fmt(q[0.5])} ({fmt(q[0.25])} ~ {fmt(q[0.75])})"


def won(x):
    """원 → '3.25억' / '4,500만' 형식"""
    if pd.isna(x):
        return "-"
    return f"{x / 1e8:,.2f}억" if abs(x) >= 1e8 else f"{x / 1e4:,.0f}만"


def pct(x):
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


class Recommender:
    def __init__(self):
        self.d = load_purchases()
        m = joblib.load(MODEL_DIR / "home_loan_models.joblib")
        self.models, self.cats = m, m["sido_categories"]
        # 비슷한 구매자 kNN (사람 + 희망 집 조건)
        self.nn = NearestNeighbors().fit(m["scaler"].transform(self.d[KNN_FEATURES]))
        # 대안 추천용 kNN (사람 조건만 — 집 가격·면적은 대안이므로 제외)
        self.p_scaler = StandardScaler().fit(self.d[PERSON_FEATURES])
        self.nn_person = NearestNeighbors().fit(self.p_scaler.transform(self.d[PERSON_FEATURES]))

    # -------------------------------------------------------------------------
    def default_rates(self):
        """금리 입력 기본값: 데이터 중앙값. 이 데이터의 금리는 신용점수·소득·연도와 상관이 없어(|r| < 0.02)
        개인별로 예측할 수 없으므로, 사용자가 안내받은 금리를 입력하도록 하고 없으면 중앙값을 쓴다"""
        return {"mort": float(self.d[self.d.combo > 0].mortgage_rate.median()),
                "cl": float(self.d[self.d.combo == 2].cl_rate.median()),
                "term": float(self.d[self.d.combo > 0].mortgage_term_years.mode()[0])}

    def data_date(self):
        """합성 데이터 기준일 (데이터의 created_at)"""
        return str(self.d.created_at.iloc[0]).replace("-", ".")

    # -------------------------------------------------------------------------
    def regions(self):
        """화면 선택 목록: {시도: [시군구...]} — 거래가 많은 시군구가 먼저 오도록 정렬"""
        return {s: list(g.sigungu.value_counts().index) for s, g in self.d.groupby("sido", observed=True)}

    # -------------------------------------------------------------------------
    # 1. 비슷한 구매자 집단
    # -------------------------------------------------------------------------
    def similar_group(self, u):
        """가까운 100명부터 시작, 대출 이용자가 30명 미만이면 인원을 2배씩 늘림"""
        x = self.models["scaler"].transform(u[KNN_FEATURES])
        for k in [100, 200, 400, 800, 1600, 3200, 6400]:
            idx = self.nn.kneighbors(x, n_neighbors=k, return_distance=False)[0]
            g = self.d.iloc[idx]
            if (g.combo > 0).sum() >= MIN_GROUP:
                break
        return g

    def group_summary(self, g):
        """집단 통계표 (개별 값 없이 분위수만). 30명 미만 조합은 숨김"""
        desc = {
            "인원": f"{len(g):,}명",
            "매입 당시 나이": f"{g.age_at_purchase.quantile(.1):.0f}~{g.age_at_purchase.quantile(.9):.0f}세",
            "가구 연소득": f"{won(g.household_annual_income.quantile(.1))} ~ {won(g.household_annual_income.quantile(.9))}",
            "매입가": f"{won(g.purchase_price_est.quantile(.1))} ~ {won(g.purchase_price_est.quantile(.9))}",
        }
        share = (g.combo.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
                 .rename(index=COMBO_NAMES))

        rows = []
        loaners = g[g.combo > 0]
        for name, sub in [("대출 이용자 전체", loaners), (COMBO_NAMES[1], g[g.combo == 1]),
                          (COMBO_NAMES[2], g[g.combo == 2])]:
            if len(sub) < MIN_GROUP:
                rows.append({"구분": name, "인원": f"{len(sub)}명 (30명 미만 — 표시 안 함)"})
                continue
            cl = sub[sub.combo == 2]
            rows.append({
                "구분": name, "인원": f"{len(sub)}명",
                "총 대출금액": iqr_text(sub.total_loan_principal, won),
                "주담대 LTV": iqr_text(sub.mort_ltv, pct),
                "주담대 금리": iqr_text(sub.mortgage_rate, lambda v: f"{v:.2f}%"),
                "마통 금리": iqr_text(cl.cl_rate, lambda v: f"{v:.2f}%") if len(cl) >= MIN_GROUP else "-",
                "만기": f"{sub.mortgage_term_years.mode()[0]:.0f}년 (최빈)",
                "월 상환액": iqr_text(sub.pay_orig, won),
                "DSR (전체 대출)": iqr_text(sub.dsr_total, pct),
                "자기자금 비율": iqr_text(sub.own_funds_ratio, pct),
            })
        return desc, share, pd.DataFrame(rows).set_index("구분").fillna("-")

    # -------------------------------------------------------------------------
    # 2. 내 조건 예측
    # -------------------------------------------------------------------------
    def plan(self, U, cash, consumption, rates, use_cl, existing_pay=0.0, basis="pattern"):
        """여러 가격(U의 각 행)에 대해 대출 계획·월상환·DSR·판정을 한 번에 계산
        existing_pay: 기존 대출 월 상환액 — DSR과 월 잉여현금에 함께 포함 (주택 보유자 화면은 0)
        basis: 'pattern' = max(비슷한 사람 패턴, 필요액) / 'need' = 필요한 만큼만 (마이데이터 화면 기본)"""
        price = U.purchase_price_est.to_numpy()
        income = U.household_annual_income.to_numpy()
        ltv = np.clip(self.models["mort_ltv"].predict(U[FEATURES]), 0, LTV_CAP)
        cl_r = np.clip(self.models["cl_ratio"].predict(U[FEATURES]), 0, CL_MAX_RATIO) if use_cl else 0.0

        pattern_loan = price * (ltv + cl_r)                 # 비슷한 사람들 패턴대로 빌릴 때
        need_loan = np.maximum(price - cash, 0)              # 보유 자금으로 모자라는 금액
        blocked = existing_pay * 12 / income > DSR_WARN       # 기존 대출만으로 DSR 40% 초과 → 필요 이상은 빌리지 않음
        loan = need_loan if basis == "need" else np.where(blocked, need_loan, np.maximum(pattern_loan, need_loan))
        mort = np.minimum(loan, price * LTV_CAP)             # 주담대는 LTV 70%까지
        cl = np.minimum(loan - mort, price * CL_MAX_RATIO)   # 나머지는 마통 (매입가 12%까지)
        shortfall = loan - mort - cl                         # 대출로도 못 채우는 금액 = 자금 부족

        pay = monthly_mortgage(mort, rates["mort"], rates["term"]) + monthly_credit_line(cl, rates["cl"])
        dsr = (pay + existing_pay) * 12 / income             # DSR은 기존 대출 상환액까지 포함
        surplus = income / 12 - consumption - pay - existing_pay   # 월 잉여현금
        return pd.DataFrame({
            "price": price, "mort": mort, "cl": cl, "loan": mort + cl, "pattern_loan": pattern_loan,
            "need_loan": need_loan, "shortfall": shortfall, "pay": pay, "dsr": dsr,
            "ltv_pred": ltv, "cl_ratio_pred": cl_r,          # 모델 원 예측값 (작동 과정 표시용)
            "surplus": surplus, "verdict": judge(dsr, surplus, shortfall, mort + cl),
        })

    def predict(self, inp, g):
        """inp: 사용자 입력 dict (원 단위), g: 비슷한 구매자 집단"""
        u = user_frame(inp["income"], inp["assets"], inp["consumption"], inp["price"],
                       inp["area"], inp["age"], inp["sido"], self.cats, inp.get("existing_pay", 0.0))
        proba = pd.Series(self.models["combo"].predict_proba(u[FEATURES])[0],
                          index=[COMBO_NAMES[i] for i in range(3)])
        use_cl = proba.iloc[2] >= proba.iloc[1]              # 대출을 쓴다면 어느 조합이 더 흔한지
        loaners = g[g.combo > 0]
        cl_grp = g[g.combo == 2]
        rates = {  # 금리·만기는 입력으로 설명되지 않아(3단계) 집단 중앙값/최빈값 사용
            "mort": loaners.mortgage_rate.median(),
            "cl": cl_grp.cl_rate.median() if len(cl_grp) >= MIN_GROUP else self.d[self.d.combo == 2].cl_rate.median(),
            "term": loaners.mortgage_term_years.mode()[0],
        }
        # 사용자가 금리·만기를 정했으면 그 값을 우선 사용 (없으면 금리는 집단 중앙값, 만기는 집단 최빈값)
        if inp.get("mort_rate") is not None:
            rates["mort"] = inp["mort_rate"]
        if inp.get("cl_rate") is not None:
            rates["cl"] = inp["cl_rate"]
        if inp.get("term") is not None:
            rates["term"] = inp["term"]
        # 보유 자금: 마이데이터 화면은 계좌 총잔액(inp["cash"]), 직접 입력 화면은 금융자산
        cash = inp.get("cash", inp["assets"]) + inp["other_funds"]
        p = self.plan(u, cash, inp["consumption"], rates, use_cl, inp.get("existing_pay", 0.0),
                      inp.get("basis", "pattern")).iloc[0]
        return u, proba, use_cl, rates, cash, p

    def loan_tiers(self, inp, rates, cash):
        """희망 가격 기준으로 DSR 구간(보수·권장·최대)마다 빌릴 수 있는 금액과 부담을 계산.

        - 월 상환 한도 = DSR 상한 × 연소득 ÷ 12. 이 한도 안에서 주담대(LTV 70%)를 먼저, 나머지를 마통(12%)으로
        - 필요한 금액(희망가 − 보유 자금)보다 더 빌리지는 않음. 모자라면 '자금 부족'으로 표시
        - 금리가 RATE_SHOCKS만큼 오를 때의 월 상환·DSR도 함께 계산
        """
        price, income = inp["price"], inp["income"]
        existing = inp.get("existing_pay", 0.0)               # 기존 대출 월 상환액 (DSR 한도에서 먼저 차감)
        need = max(price - cash, 0.0)
        mort_cap, cl_cap = price * LTV_CAP, price * CL_MAX_RATIO
        a = float(monthly_mortgage(1.0, rates["mort"], rates["term"]))  # 주담대 1원당 월 상환
        b = float(monthly_credit_line(1.0, rates["cl"]))                 # 마통 1원당 월 상환

        rows = []
        for name, dsr_cap, label, desc in LOAN_TIERS:
            budget = max(dsr_cap * income / 12 - existing, 0.0)
            if budget <= a * mort_cap:                       # 주담대만으로 한도에 닿음
                dsr_loan, limit = budget / a, "DSR"
            else:                                              # 주담대를 다 쓰고 남는 여력을 마통으로
                extra = (budget - a * mort_cap) / b
                dsr_loan = mort_cap + min(extra, cl_cap)
                limit = "DSR" if extra <= cl_cap else "대출 한도"
            loan = min(dsr_loan, need)
            mort, cl = split_loan(loan, price)
            pay = loan_payment(mort, cl, rates)
            rows.append({
                "tier": name, "dsr_cap": dsr_cap, "label": label, "desc": desc,
                "loan": loan, "mort": mort, "cl": cl, "pay": pay, "dsr": (pay + existing) * 12 / income,
                "surplus": income / 12 - inp["consumption"] - pay - existing,
                "shortfall": max(need - loan, 0.0),
                "limited_by": "필요 금액" if dsr_loan >= need else limit,
                "shocks": [(s, loan_payment(mort, cl, rates, s),
                            (loan_payment(mort, cl, rates, s) + existing) * 12 / income) for s in RATE_SHOCKS],
            })
        return rows

    # -------------------------------------------------------------------------
    # 3. 대안 추천
    # -------------------------------------------------------------------------
    def max_affordable(self, inp, rates, use_cl, cash):
        """가격을 바꿔가며 판정을 계산해, '여유'와 '어려움 아님'의 최대 가격을 찾음"""
        prices = np.linspace(3e7, max(inp["price"], 3e7), 150)
        U = pd.concat([user_frame(inp["income"], inp["assets"], inp["consumption"], p,
                                  inp["area"], inp["age"], inp["sido"], self.cats,
                                  inp.get("existing_pay", 0.0)) for p in prices],
                      ignore_index=True)
        P = self.plan(U, cash, inp["consumption"], rates, use_cl, inp.get("existing_pay", 0.0), inp.get("basis", "pattern"))
        ok = P[P.verdict != "어려움"].price.max()
        comfy = P[P.verdict == "여유"].price.max()
        return (None if pd.isna(ok) else ok), (None if pd.isna(comfy) else comfy)

    def plan_many(self, inp, prices, areas, sidos, rates, use_cl, cash):
        """여러 집(가격·면적·시도)에 대해 대출 계획과 구매력 구간을 한 번에 계산"""
        U = pd.concat([user_frame(inp["income"], inp["assets"], inp["consumption"], p, a, inp["age"], s,
                                  self.cats, inp.get("existing_pay", 0.0))
                       for p, a, s in zip(prices, areas, sidos)], ignore_index=True)
        P = self.plan(U, cash, inp["consumption"], rates, use_cl, inp.get("existing_pay", 0.0), inp.get("basis", "pattern"))
        P["zone"] = zone_of(P.dsr, P.surplus, P.shortfall, inp["goal"], P.loan)
        return P

    def power(self, inp, rates, cash):
        """내 구매력: 가격을 3천만~30억으로 바꿔가며 구간을 계산해 각 구간의 상단 가격을 찾음"""
        u = user_frame(inp["income"], inp["assets"], inp["consumption"], 5e8, inp["area"], inp["age"],
                       inp["sido"], self.cats, inp.get("existing_pay", 0.0))
        proba = self.models["combo"].predict_proba(u[FEATURES])[0]
        use_cl = proba[2] >= proba[1]
        prices = np.unique(np.round(np.geomspace(3e7, 3e9, 240), -6))
        n = len(prices)
        P = self.plan_many(inp, prices, [inp["area"]] * n, [inp["sido"]] * n, rates, use_cl, cash)

        def top(mask):
            return float(P.price[mask].max()) if mask.any() else None
        bounds = {"안정권": top(P.zone == "안정권"),
                  "가능권": top(P.zone.isin(["안정권", "가능권"])),
                  "한계권": top(P.zone != "범위 밖")}   # = 계산상 최대 가격
        return P, bounds, use_cl

    def region_context(self, sido, sigungu):
        """희망 지역 시세: 시군구 30건 이상이면 시군구, 아니면 시도로 넓힘"""
        sub = self.d[(self.d.sido == sido) & (self.d.sigungu == sigungu)]
        basis, short = f"{sido} {sigungu}", f"{sigungu}에서"
        if len(sub) < MIN_GROUP:
            sub, basis, short = (self.d[self.d.sido == sido],
                                 f"{sido} 전체 ({sigungu} 거래 30건 미만 → 시도로 확대)", f"{sido} 시세로")
        return basis, {
            "기준": short,
            "거래 건수": len(sub),
            "매입가": won(sub.purchase_price_est.median()),
            "매입가 구간": f"{won(sub.purchase_price_est.quantile(.25))} ~ {won(sub.purchase_price_est.quantile(.75))}",
            "전용면적": f"{sub.exclusive_area_m2.median():.0f}㎡",
            "전용면적 구간": f"{sub.exclusive_area_m2.quantile(.25):.0f}㎡ ~ {sub.exclusive_area_m2.quantile(.75):.0f}㎡",
            "평당가 중앙값": sub.price_per_pyeong.median(),
        }

    def alternatives(self, u, max_price, sido):
        """비슷한 사람들(사람 조건 기준)이 max_price 이하에서 많이 산 가격대·면적·지역"""
        x = self.p_scaler.transform(u[PERSON_FEATURES])
        pool = None
        for k in [3000, 6000, 12000, len(self.d)]:
            idx = self.nn_person.kneighbors(x, n_neighbors=k, return_distance=False)[0]
            pool = self.d.iloc[idx]
            pool = pool[pool.purchase_price_est <= max_price]
            if len(pool) >= MIN_GROUP * 5:
                break
        out = {"pool_n": len(pool)}

        # 가격대 (5천만 원 단위)
        edges = np.arange(0, max_price + 5e7, 5e7)
        band = pd.cut(pool.purchase_price_est, edges, right=True)
        t = pool.groupby(band, observed=True).agg(인원=("combo", "size"),
                                                  면적중앙=("exclusive_area_m2", "median"))
        t = t[t.인원 >= MIN_GROUP]
        t.index = [f"{won(iv.left)} ~ {won(iv.right)}" for iv in t.index]
        t["비중"] = (t.인원 / len(pool)).map(pct)
        t["면적중앙"] = t.면적중앙.map(lambda v: f"{v:.0f}㎡ ({v / PYEONG:.0f}평)")
        out["price_bands"] = t.sort_values("인원", ascending=False).head(6).rename_axis("가격대")

        # 면적대
        a_edges = [0, 40, 60, 85, 102, 135, 1000]
        a_labels = ["40㎡ 미만", "40~60㎡", "60~85㎡", "85~102㎡", "102~135㎡", "135㎡ 이상"]
        t = pool.groupby(pd.cut(pool.exclusive_area_m2, a_edges, labels=a_labels), observed=True).agg(
            인원=("combo", "size"), 매입가중앙=("purchase_price_est", "median"))
        t = t[t.인원 >= MIN_GROUP]
        t["비중"] = (t.인원 / len(pool)).map(pct)
        t["매입가중앙"] = t.매입가중앙.map(won)
        out["area_bands"] = t.rename_axis("면적대")

        # 지역: ① 비슷한 사람들 중 희망 시도 안의 시군구 → ② 희망 시도 전체 거래(가격 이하) → ③ 전국 시도
        def region_table(df, key):
            r = df.groupby(key, observed=True).agg(건수=("combo", "size"),
                                                    매입가중앙=("purchase_price_est", "median"),
                                                    면적중앙=("exclusive_area_m2", "median"))
            r = r[r.건수 >= MIN_GROUP].sort_values("건수", ascending=False).head(8)
            r["매입가중앙"] = r.매입가중앙.map(won)
            r["면적중앙"] = r.면적중앙.map(lambda v: f"{v:.0f}㎡")
            return r.rename_axis("지역")

        r = region_table(pool[pool.sido == sido], "sigungu")
        basis = f"나와 비슷한 구매자들이 {sido}에서 {won(max_price)} 이하로 산 시군구"
        if len(r) < 3:
            market = self.d[(self.d.sido == sido) & (self.d.purchase_price_est <= max_price)]
            r = region_table(market, "sigungu")
            basis = f"{sido} 전체 거래 중 {won(max_price)} 이하 거래가 많은 시군구 (비슷한 구매자만으로는 30명 미만)"
        if len(r) == 0:
            r = region_table(pool, "sido")
            basis = f"나와 비슷한 구매자들이 {won(max_price)} 이하로 산 시도 (전국으로 확대)"
        out["regions"], out["region_basis"] = r, basis
        return out
