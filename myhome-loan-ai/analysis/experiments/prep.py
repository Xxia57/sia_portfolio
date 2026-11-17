# =============================================================================
# 공통 데이터 준비 모듈 — 학습(02_train.py)과 웹 화면(app.py)이 같이 사용
#   - 분석 대상 필터, 광주·전남 시도명 매핑, 파생 변수, 월상환 공식
# =============================================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "output"
MODEL_DIR = BASE_DIR / "models"

MIN_GROUP = 30          # 화면에 보여줄 집단의 최소 인원
LTV_CAP = 0.70          # 주담대 LTV 상한 (데이터 최대 0.699)
DSR_OK, DSR_WARN = 0.30, 0.40   # DSR 판정: 30% 미만 여유 / 30~40% 주의 / 40% 초과 어려움
CL_EXTRA = 0.003        # 마이너스통장 월상환 = 잔액 × (월금리 + 0.3%)  ← 데이터에서 역산한 공식
CURRENT_YEAR = 2026

# 대출 조합 코드
COMBO_NAMES = {0: "대출 없음", 1: "주담대만", 2: "주담대 + 마이너스통장"}

# 모델·kNN에 쓰는 입력 변수 (사용자가 화면에서 입력하는 항목으로만 구성)
#   금액은 분포가 오른쪽으로 길어서 log 변환
#   pre_dsr: 집을 사기 전부터 갚던 대출(신용대출 등)의 DSR — GBM에만 사용
#            (구매자의 94%가 0이라 kNN 거리에 넣으면 기존 대출이 있는 사용자가 소수 집단에만 묶임)
NUM_FEATURES = ["age_at_purchase", "log_income", "log_assets", "log_consumption",
                "log_price", "exclusive_area_m2", "purchase_year", "pre_dsr"]
KNN_FEATURES = ["age_at_purchase", "log_income", "log_assets", "log_consumption",
                "log_price", "exclusive_area_m2"]

GWANGJU_GU = {"광산구", "동구", "서구", "남구", "북구"}


def monthly_mortgage(principal, annual_rate_pct, term_years):
    """주담대 월상환액 — 원리금균등 (데이터의 monthly_payment와 오차 0 확인)"""
    r = np.asarray(annual_rate_pct) / 100 / 12
    n = np.asarray(term_years) * 12
    return np.asarray(principal) * r / (1 - (1 + r) ** -n)


def monthly_credit_line(principal, annual_rate_pct):
    """마이너스통장 월상환액 — 잔액 × (월금리 + 0.3%) (데이터에서 역산, 오차 0 확인)"""
    return np.asarray(principal) * (np.asarray(annual_rate_pct) / 100 / 12 + CL_EXTRA)


def add_features(df):
    """입력 변수 파생 (학습 데이터와 사용자 입력에 똑같이 적용)"""
    df = df.copy()
    df["log_income"] = np.log(df["household_annual_income"])
    df["log_assets"] = np.log(df["financial_assets_estimated"].clip(lower=1e5))
    df["log_consumption"] = np.log(df["avg_monthly_consumption"].clip(lower=1e4))
    df["log_price"] = np.log(df["purchase_price_est"])
    # 기존 대출 DSR = 기존 대출 월 상환 × 12 ÷ 연 소득 (없으면 0)
    existing = df["existing_pay"] if "existing_pay" in df else 0.0
    df["pre_dsr"] = existing * 12 / df["household_annual_income"]
    df["sido"] = df["sido"].astype("category")
    return df


def load_purchases():
    """home_purchases + 대출별 금리를 합쳐 분석용 테이블을 만든다"""
    hp = pd.read_csv(DATA_DIR / "home_purchases.csv", encoding="utf-8-sig")
    loans = pd.read_csv(DATA_DIR / "loans_raw.csv", encoding="utf-8-sig")

    # 1) 분석 대상: 최근 5년(2022~) 매입, 매입 당시 20세 이상, 자기자금 > 0 (대출>매입가 1건 제외)
    hp = hp[(hp.purchase_year >= 2022) & (hp.under20_at_purchase == 0) & (hp.own_funds_est > 0)].copy()

    # 2) '전남광주통합특별시' → 구(區)는 광주광역시, 시·군은 전라남도로 되돌림 (사용자 결정 A)
    m = hp.sido == "전남광주통합특별시"
    hp.loc[m, "sido"] = np.where(hp.loc[m, "sigungu"].isin(GWANGJU_GU), "광주광역시", "전라남도")
    hp["region_key"] = hp.sido + " " + hp.sigungu

    # 3) 마이너스통장 금리는 home_purchases에 없어서 loans_raw에서 가져옴 (구매연결 대출만)
    cl = (loans[(loans.loan_type == "CREDIT_LINE") & (loans.housing_purchase_linked == 1)]
          .groupby("customer_id").interest_rate.mean().rename("cl_rate"))
    hp = hp.merge(cl, on="customer_id", how="left")

    # 4) 목표 변수(모델이 맞출 값)
    hp["combo"] = np.select([hp.loan_types == "CREDIT_LINE+MORTGAGE", hp.loan_types == "MORTGAGE"],
                            [2, 1], default=0)
    hp["mort_ltv"] = hp.mortgage_principal / hp.purchase_price_est          # 주담대 LTV
    hp["cl_ratio"] = hp.other_loan_principal / hp.purchase_price_est         # 마통 / 매입가
    hp["loan_ratio"] = hp.total_loan_principal.fillna(0) / hp.purchase_price_est

    # 5) 매입 당시 기준 월상환·DSR 재계산 (마통은 최초 한도 전액 사용 가정 = 새 사용자와 같은 조건)
    pay = np.where(hp.combo > 0, monthly_mortgage(hp.mortgage_principal, hp.mortgage_rate,
                                                  hp.mortgage_term_years), 0.0)
    pay = pay + np.where(hp.combo == 2, monthly_credit_line(hp.other_loan_principal, hp.cl_rate), 0.0)
    hp["pay_orig"] = np.nan_to_num(pay)
    hp["dsr_orig"] = hp.pay_orig * 12 / hp.household_annual_income      # 구매 대출(주담대·마통)만의 DSR

    # 6) 매입 당시 이미 갚고 있던 대출 (구매와 무관한 신용대출 등: 실행일 ≤ 매입일 < 만기일)
    #    → 기존 대출 월상환(existing_pay)과 전체 대출 기준 DSR(dsr_total) — 사용자 DSR과 같은 기준
    other = loans[loans.housing_purchase_linked == 0].merge(hp[["customer_id", "purchase_date"]], on="customer_id")
    live = other[(other.opened_at <= other.purchase_date) & (other.maturity_date > other.purchase_date)]
    hp["existing_pay"] = hp.customer_id.map(live.groupby("customer_id").monthly_payment_estimated.sum()).fillna(0.0)
    hp["dsr_total"] = (hp.pay_orig + hp.existing_pay) * 12 / hp.household_annual_income

    return add_features(hp)


# -----------------------------------------------------------------------------
# 마이데이터(무주택 고객) — extract_customer_debt_summary.py로 추출한 파일 3개
#   accounts.csv                     : 계좌 (고객당 입출금 CHECKING + 저축 SAVINGS)
#   customer_debt_summary.csv        : 부채 요약 + 무주택 여부(is_home_owner)
#   customer_financial_profiles.csv  : 나이·가구원수·본인/가구 소득·금융자산·소비 예산 (전체 고객)
#     (또는 extract 스크립트 --mydata-profile 결과 customer_profiles_non_owner.csv)
#   환경변수 MYDATA_DIR로 다른 폴더를 지정할 수 있음 (테스트용)
# -----------------------------------------------------------------------------
MYDATA_DIR = Path(os.environ.get("MYDATA_DIR", DATA_DIR))
MYDATA_FILES = {"accounts": "accounts.csv", "debt": "customer_debt_summary.csv",
                "profile": "customer_financial_profiles.csv"}
PROFILE_ALT = "customer_profiles_non_owner.csv"   # 프로필 파일 대안 (12개월 현금흐름 소비 평균 포함)
PII_COLS = ["birth_date"]                          # 생년월일은 읽자마자 버림
# name(이름)은 '선택한 본인'의 ① 화면 인사말에만 사용 — 계산·비교 집단에는 쓰지 않음
SIDO_RENAME = {"전라북도": "전북특별자치도", "강원도": "강원특별자치도", "제주도": "제주특별자치도"}
ACCOUNT_NAMES = {"CHECKING": "입출금 계좌", "SAVINGS": "저축 계좌"}


def load_mydata():
    """무주택 고객 1명 = 1행 표(customers)와 계좌 원본(accounts)을 돌려줌. 없는 파일 목록도 함께 반환.

    소득 기준 (사용자 결정): 1인 가구는 본인 소득(primary), 그 외는 가구 소득(household)
    """
    path = {k: MYDATA_DIR / f for k, f in MYDATA_FILES.items()}
    if not path["profile"].exists() and (MYDATA_DIR / PROFILE_ALT).exists():
        path["profile"] = MYDATA_DIR / PROFILE_ALT
    missing = [str(p) for p in path.values() if not p.exists()]
    if not path["debt"].exists() or not path["accounts"].exists():
        return None, None, missing

    debt = pd.read_csv(path["debt"], encoding="utf-8-sig")
    debt = debt[debt.is_home_owner == 0].set_index("customer_id")
    acc = pd.read_csv(path["accounts"], encoding="utf-8-sig")
    acc = acc[acc.customer_id.isin(debt.index)]

    # 계좌 잔액: 종류별 + 총합 (고객 1명 = 1행)
    bal = acc.pivot_table(index="customer_id", columns="account_type", values="balance", aggfunc="sum").fillna(0)
    c = debt.join(bal.add_prefix("bal_"))
    c["acct_total"] = bal.sum(axis=1).reindex(c.index).fillna(0)
    c["existing_pay"] = c.monthly_debt_service.fillna(0.0)

    if path["profile"].exists():
        prof = pd.read_csv(path["profile"], encoding="utf-8-sig")
        prof = (prof.drop(columns=[x for x in PII_COLS if x in prof.columns])   # 이름·생년월일 제외
                .rename(columns={"region": "residence_region"})               # 거주지 ≠ 주택 소재지
                .set_index("customer_id"))
        c = c.join(prof[[x for x in prof.columns if x not in c.columns]], how="inner")
        one = c.household_size == 1
        # 1인 가구 → 본인 소득, 그 외 → 가구 소득
        c["annual_income"] = np.where(one, c.primary_annual_income, c.household_annual_income)
        c["income_basis"] = np.where(one, "본인 소득 (1인 가구)", "가구 소득")
        c["monthly_income"] = c.annual_income / 12
        # 월 소비: 12개월 현금흐름 평균 → 없으면 프로필의 소비 예산
        #   (주택 보유자 데이터에서 두 값은 비율 1.00·상관 1.00으로 사실상 같음 → 학습 데이터와 같은 척도)
        if "avg_monthly_consumption" in c:
            c["consumption"] = c.avg_monthly_consumption.fillna(c.monthly_consumption_budget)
        else:
            c["consumption"] = c.monthly_consumption_budget
        c["residence_sido"] = (c.residence_region.replace(SIDO_RENAME)
                               .replace({"전남광주통합특별시": "광주광역시"}))  # 통합특별시는 시군구가 없어 광주로
    return c, acc, missing


def user_frame(income, assets, consumption, price, area, age, sido, categories, existing_pay=0.0):
    """사용자 입력 1건을 모델 입력 형태(DataFrame)로 변환 (existing_pay: 기존 대출 월 상환액)"""
    df = pd.DataFrame([{
        "household_annual_income": income, "financial_assets_estimated": assets,
        "avg_monthly_consumption": consumption, "purchase_price_est": price,
        "exclusive_area_m2": area, "age_at_purchase": age,
        "purchase_year": CURRENT_YEAR, "sido": sido, "existing_pay": existing_pay,
    }])
    df = add_features(df)
    df["sido"] = pd.Categorical(df["sido"], categories=categories)
    return df
