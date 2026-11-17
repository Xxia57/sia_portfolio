# =============================================================================
# 1단계: 데이터 탐색 (EDA) — 재추출 CSV(주택 보유 고객 전체) 기준
#   - 결측치, 이상치, 주요 컬럼 분포, 대출 조합별 비중, price_source별 건수
#   - 분석 대상: 최근 5년(2022~) 매입 + 매입 당시 20세 이상
#   - 실행: .venv\Scripts\python.exe home_loan_mvp\01_eda.py   (02_MYDATA 폴더에서)
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# 이 파일 기준으로 ../output 폴더의 CSV를 읽음
DATA_DIR = Path(__file__).resolve().parent.parent / "output"
hp_all = pd.read_csv(DATA_DIR / "home_purchases.csv", encoding="utf-8-sig")
loans = pd.read_csv(DATA_DIR / "loans_raw.csv", encoding="utf-8-sig")
cust = pd.read_csv(DATA_DIR / "customers.csv", encoding="utf-8-sig")

# 분석 대상 필터 (사용자 확정 사항)
#   - 최근 5년(2022~) 매입 건만: 현재 소득·자산과 매입 당시 값의 차이를 줄이기 위해
#   - under20_at_purchase=1 제외: 매입 당시 20세 미만은 데이터 생성 과정의 흔적
hp = hp_all[(hp_all.purchase_year >= 2022) & (hp_all.under20_at_purchase == 0)].copy()


def title(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# -----------------------------------------------------------------------------
title("1. 기본 정보")
print(f"home_purchases 전체 {hp_all.shape}, customers {cust.shape}, loans_raw {loans.shape}")
print("고객 ID 일치(home_purchases vs customers):", set(hp_all.customer_id) == set(cust.customer_id))
print(f"분석 대상(2022~ & 매입시 20세 이상): {len(hp):,}건 "
      f"(2022~ {(hp_all.purchase_year >= 2022).sum():,}건 중 20세 미만 "
      f"{((hp_all.purchase_year >= 2022) & (hp_all.under20_at_purchase == 1)).sum()}건 제외)")

# -----------------------------------------------------------------------------
title("2. 결측치 (분석 대상, 결측 있는 컬럼만)")
na = hp.isna().sum()
print(na[na > 0].to_string())
print("대출 없이 산 건(bought_with_loan=0):", (hp.bought_with_loan == 0).sum())

# -----------------------------------------------------------------------------
title("3. price_source별 건수")
print(hp.price_source.value_counts().to_string())

# -----------------------------------------------------------------------------
title("4. 대출 조합별 비중과 대표값(중앙값)")
hp["loan_types_f"] = hp.loan_types.fillna("대출없음")
agg = dict(
    건수=("customer_id", "size"),
    매입가=("purchase_price_est", "median"),
    총대출=("total_loan_principal", "median"),
    LTV=("ltv_at_origination", "median"),
    가중금리=("weighted_rate", "median"),
    월상환=("monthly_payment_total", "median"),
    DSR=("payment_to_income", "median"),
    자기자금비율=("own_funds_ratio", "median"),
    가구연소득=("household_annual_income", "median"),
    매입나이=("age_at_purchase", "median"),
)
g = hp.groupby("loan_types_f").agg(**agg)
g.insert(1, "비중", (g.건수 / len(hp)).round(3))
print(g.sort_values("건수", ascending=False).to_string(float_format=lambda x: f"{x:,.3f}"))

# 매입 당시 나이대별 대출 조합 비중 (청년일수록 마통 병행이 많은지)
hp["buy_age_band"] = pd.cut(hp.age_at_purchase, [19, 24, 29, 34, 39, 49, 59, 120],
                            labels=["20-24", "25-29", "30-34", "35-39", "40대", "50대", "60+"])
print("\n매입 당시 나이대 × 대출 조합 (행 비율)")
print(pd.crosstab(hp.buy_age_band, hp.loan_types_f, normalize="index", margins=True)
      .round(3).to_string())

# -----------------------------------------------------------------------------
title("5. 주요 수치 컬럼 분포 (1%/25%/50%/75%/99%)")
num_cols = [
    "purchase_price_est", "exclusive_area_m2", "price_per_pyeong",
    "total_loan_principal", "ltv_at_origination", "mortgage_rate", "weighted_rate",
    "mortgage_term_years", "monthly_payment_total", "payment_to_income", "own_funds_ratio",
    "household_annual_income", "financial_assets_estimated", "avg_monthly_consumption",
    "credit_score", "age_at_purchase",
]
desc = hp[num_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
print(desc[["count", "min", "1%", "25%", "50%", "75%", "99%", "max"]].to_string(float_format=lambda x: f"{x:,.3f}"))
print("\n만기(년) 분포:", hp.mortgage_term_years.value_counts().sort_index().to_dict())

# -----------------------------------------------------------------------------
title("6. 이상치 점검")
# (a) IQR 기준 바깥 건수 — 금액 분포가 오른쪽으로 긴 정상적인 형태인지 확인용
for c in ["purchase_price_est", "exclusive_area_m2", "household_annual_income",
          "financial_assets_estimated", "avg_monthly_consumption", "weighted_rate"]:
    x = hp[c].dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    out = ((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum()
    print(f"  {c:28s} IQR 밖 {out:5d}건 ({out / len(x):.1%})  min {x.min():,.0f} / max {x.max():,.0f}")

# (b) 논리적으로 이상한 값
print("  LTV > 70%:", (hp.ltv_at_origination > 0.7).sum(), "건")
print("  DSR(payment_to_income) > 40%:", (hp.payment_to_income > 0.4).sum(), "건")
print("  월소비 > 가구월소득:", f"{(hp.avg_monthly_consumption > hp.household_monthly_income).mean():.1%}",
      "/ 월소비 > 본인월소득:", f"{(hp.avg_monthly_consumption > hp.avg_monthly_primary_income).mean():.1%}")
print("  건축연도 > 매입연도:", (hp.build_year > hp.purchase_year).sum(), "건")

# -----------------------------------------------------------------------------
title("7. 정합성 점검 (재추출 반영 확인)")
L = hp[hp.bought_with_loan == 1]
gap = hp.purchase_price_est - hp.total_loan_principal.fillna(0) - hp.own_funds_est
print("  매입가 = 총대출 + 자기자금 불일치:", (gap.abs() > 1).sum(), "건",
      "(대출 > 매입가라 자기자금이 0으로 잘린 건:",
      ((gap.abs() > 1) & (hp.total_loan_principal > hp.purchase_price_est)).sum(), "건)")
print("  총대출 = 주담대 + 기타(마통):",
      np.allclose(L.total_loan_principal, L.mortgage_principal + L.other_loan_principal))
print("  LTV = 주담대/매입가:",
      np.allclose(L.mortgage_principal / L.purchase_price_est, L.ltv_at_origination, atol=1e-4))
print("  DSR(payment_to_income) = 연원리금/가구연소득:",
      np.allclose(L.payment_to_income, L.annual_debt_service_total / L.household_annual_income))
print("  가구월소득 × 12 = 가구연소득:",
      np.allclose(hp.household_monthly_income * 12, hp.household_annual_income, rtol=1e-3))
print("  매입 당시 나이 = 현재 나이 − (2026 − 매입연도) 차이 최대:",
      (hp.age_at_purchase - (hp.age - (2026 - hp.purchase_year))).abs().max())

# 구매용 대출만 집계됐는지: loans_raw의 구매연결 대출 원금 합과 비교
linked = (loans[loans.housing_purchase_linked == 1]
          .groupby("customer_id").original_principal.sum())
chk = L.set_index("customer_id").total_loan_principal
print("  총대출 = loans_raw 구매연결 대출(original_principal) 합:",
      np.allclose(chk, linked.reindex(chk.index)))
print("  loans_raw 대출종류 × 구매연결:")
print(pd.crosstab(loans.loan_type, loans.housing_purchase_linked).to_string())
print("  [한계2] loans_raw outstanding == principal 비율:", (loans.outstanding_balance == loans.principal).mean())

# -----------------------------------------------------------------------------
title("8. 매입연도 / 매입 당시 나이 / 지역")
print(hp.purchase_year.value_counts().sort_index().to_string())
print("\n매입 당시 나이대"); print(hp.buy_age_band.value_counts().sort_index().to_string())
print("\n시도별 건수"); print(hp.sido.value_counts().to_string())
# 광주·전남은 2024년 중반부터 '전남광주통합특별시'로 표기가 바뀜 → 같은 지역이 두 이름으로 나뉨
gj = hp_all[hp_all.sido.isin(["광주광역시", "전라남도", "전남광주통합특별시"])]
print("\n광주·전남 매입연도 × 시도명 (전체 기간)")
print(pd.crosstab(gj.purchase_year, gj.sido).to_string())


# 시군구별 건수: 이전 3,000명 샘플 → 전체 → 분석 대상 비교
def sgg_stats(df):
    s = df.groupby(["sido", "sigungu"]).size()
    return {"시군구 수": len(s), "중앙 건수": s.median(), "30건 이상": (s >= 30).sum(),
            "30건 이상 비율": round((s >= 30).mean(), 3),
            "30건 이상 시군구에 속한 행 비율": round(s[s >= 30].sum() / s.sum(), 3)}


prev = {"시군구 수": 348, "중앙 건수": 4.0, "30건 이상": 19, "30건 이상 비율": round(19 / 348, 3),
        "30건 이상 시군구에 속한 행 비율": None}  # 이전 샘플 탐색 결과
print("\n시군구별 건수 비교")
print(pd.DataFrame({"이전 샘플(3,000)": prev, "전체(42,225)": sgg_stats(hp_all),
                    "분석 대상(2022~)": sgg_stats(hp)}).to_string())
