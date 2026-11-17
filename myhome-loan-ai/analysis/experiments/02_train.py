# =============================================================================
# 3단계: 모델 학습 + 검증
#   - 학습 80% / 검증 20% 로 나누고, 4가지 방법의 예측 오차를 같은 검증 데이터로 비교
#       B0 전체 중앙값      : 아무 조건 없이 학습 데이터 중앙값 (최소 기준선)
#       B1 조건완화 필터    : 나이/소득/가격 범위를 넓혀가며 30명 이상 집단의 중앙값 (설계안 1)
#       B2 kNN(100명)      : 가장 비슷한 100명의 중앙값 (설계안 2)
#       M  GBM 모델        : HistGradientBoosting 회귀/분류 (설계안 3)
#   - 추가 검증: 2022~2025 학습 → 2026 검증 (앞으로 올 사용자 = 최신 매입에 가까움)
#   - 실행: ..\.venv\Scripts\python.exe 02_train.py   (home_loan_mvp 폴더에서)
# =============================================================================
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from prep import (KNN_FEATURES, LTV_CAP, MIN_GROUP, MODEL_DIR, NUM_FEATURES, load_purchases,
                  monthly_credit_line, monthly_mortgage)

pd.set_option("display.width", 200)
SEED = 42
K = 100
FEATURES = NUM_FEATURES + ["sido"]

# 대출 구조 목표 변수: (컬럼, 한글명, 대상 조합)
#   combo_min=1 → 주담대가 있는 사람(조합 1, 2), combo_min=2 → 마통까지 쓴 사람(조합 2)
TARGETS = [
    ("mort_ltv", "주담대 LTV", 1),
    ("cl_ratio", "마통/매입가", 2),
    ("mortgage_rate", "주담대 금리(%)", 1),
    ("cl_rate", "마통 금리(%)", 2),
    ("mortgage_term_years", "만기(년)", 1),
]


# -----------------------------------------------------------------------------
# 예측 방법별 함수
# -----------------------------------------------------------------------------
def make_gbm(kind):
    """GBM 모델 생성 — sido는 범주형으로 자동 처리, 과적합 방지를 위해 조기 종료 사용"""
    common = dict(categorical_features="from_dtype", learning_rate=0.05, max_iter=400,
                  early_stopping=True, validation_fraction=0.15, random_state=SEED)
    return HistGradientBoostingRegressor(**common) if kind == "reg" else HistGradientBoostingClassifier(**common)


def filter_groups(train, test):
    """설계안 1: 조건을 단계적으로 넓혀 30명 이상 모이는 첫 집단의 인덱스 목록을 돌려줌"""
    tiers = [(2, 0.3, 0.2), (5, 0.5, 0.3), (8, 0.7, 0.5), (15, 1.0, 1.0)]  # (나이±, 소득±%, 가격±%)
    a, inc, pr = (train[c].to_numpy() for c in ["age_at_purchase", "household_annual_income", "purchase_price_est"])
    groups = []
    for _, u in test.iterrows():
        idx = np.arange(len(train))  # 끝까지 부족하면 전체
        for da, di, dp in tiers:
            m = ((np.abs(a - u.age_at_purchase) <= da)
                 & (np.abs(inc / u.household_annual_income - 1) <= di)
                 & (np.abs(pr / u.purchase_price_est - 1) <= dp))
            if m.sum() >= MIN_GROUP:
                idx = np.flatnonzero(m)
                break
        groups.append(idx)
    return groups


def knn_groups(train, test, scaler):
    """설계안 2: 표준화한 입력 변수 거리로 가장 가까운 K명의 인덱스"""
    nn = NearestNeighbors(n_neighbors=min(K, len(train))).fit(scaler.transform(train[KNN_FEATURES]))
    return nn.kneighbors(scaler.transform(test[KNN_FEATURES]), return_distance=False)


def group_stat(values, groups, q):
    """각 집단의 분위수 (q=0.5 → 중앙값)"""
    return np.array([np.quantile(values[g], q) for g in groups])


# -----------------------------------------------------------------------------
# 한 번의 학습/검증 실행
# -----------------------------------------------------------------------------
def evaluate(train, test, label):
    print("\n" + "=" * 78 + f"\n[{label}] 학습 {len(train):,}건 / 검증 {len(test):,}건\n" + "=" * 78)
    scaler = StandardScaler().fit(train[KNN_FEATURES])
    rows, preds = [], {}

    # ---- (1) 대출 조합 분류: 대출없음 / 주담대만 / 주담대+마통 -------------------------
    y_tr, y_te = train.combo.to_numpy(), test.combo.to_numpy()
    p = {}
    p["B0 전체 비율"] = np.tile(np.bincount(y_tr, minlength=3) / len(y_tr), (len(test), 1))
    for name, groups in [("B1 조건완화 필터", filter_groups(train, test)),
                         ("B2 kNN(100명)", knn_groups(train, test, scaler))]:
        p[name] = np.array([np.bincount(y_tr[g], minlength=3) / len(g) for g in groups])
    clf = make_gbm("clf").fit(train[FEATURES], y_tr)
    p["M GBM"] = clf.predict_proba(test[FEATURES])
    print("\n(1) 대출 조합 분류 — 정확도(높을수록 좋음), log loss(낮을수록 좋음)")
    for name, pr in p.items():
        pr = np.clip(pr, 1e-3, 1)
        pr = pr / pr.sum(1, keepdims=True)
        print(f"   {name:16s} 정확도 {accuracy_score(y_te, pr.argmax(1)):.3f}   log loss {log_loss(y_te, pr, labels=[0, 1, 2]):.3f}")

    # ---- (2) 대출 구조 회귀: 조합이 정해졌다는 조건에서 각 값 예측 ------------------------
    print("\n(2) 대출 구조 — 평균절대오차(MAE, 낮을수록 좋음) / kNN 25~75% 구간에 실제값이 들어간 비율(50%에 가까울수록 좋음)")
    for col, kor, cmin in TARGETS:
        tr, te = train[train.combo >= cmin], test[test.combo >= cmin]
        y, yt = tr[col].to_numpy(), te[col].to_numpy()
        kg = knn_groups(tr, te, scaler)
        pr = {
            "B0 전체 중앙값": np.full(len(te), np.median(y)),
            "B1 조건완화 필터": group_stat(y, filter_groups(tr, te), 0.5),
            "B2 kNN(100명)": group_stat(y, kg, 0.5),
            "M GBM": make_gbm("reg").fit(tr[FEATURES], y).predict(te[FEATURES]),
        }
        lo, hi = group_stat(y, kg, 0.25), group_stat(y, kg, 0.75)
        cover = ((yt >= lo) & (yt <= hi)).mean()
        r = {"목표": kor, "검증건수": len(te)}
        r.update({k: mean_absolute_error(yt, v) for k, v in pr.items()})
        r["kNN구간포함"] = cover
        rows.append(r)
        preds[col] = (te.index, pr)
    tbl = pd.DataFrame(rows).set_index("목표")
    print(tbl.to_string(float_format=lambda x: f"{x:,.4f}"))

    # ---- (3) 최종 결과값: 조합은 실제값으로 두고, 예측한 LTV·마통·금리·만기로 대출액/월상환/DSR 계산 ----
    print("\n(3) 최종 결과값 오차 (주담대 있는 검증 건, 조합은 실제 기준)")
    te = test[test.combo >= 1]
    out = []
    for name in ["B0 전체 중앙값", "B1 조건완화 필터", "B2 kNN(100명)", "M GBM"]:
        def get(col):
            idx, pr = preds[col]
            return pd.Series(pr[name], index=idx).reindex(te.index)
        ltv = get("mort_ltv").clip(0, LTV_CAP)
        is_cl = te.combo == 2
        clr = get("cl_ratio").where(is_cl, 0).clip(lower=0)
        mort_amt = te.purchase_price_est * ltv
        cl_amt = te.purchase_price_est * clr
        pay = monthly_mortgage(mort_amt, get("mortgage_rate"), get("mortgage_term_years").round())
        pay = pay + np.where(is_cl, monthly_credit_line(cl_amt, get("cl_rate").fillna(0)), 0)
        # DSR은 앱과 같은 기준: 구매 대출 + 매입 당시 이미 갚던 대출(existing_pay)
        dsr = (pay + te.existing_pay) * 12 / te.household_annual_income
        out.append({"방법": name,
                    "총대출 MAE(만원)": mean_absolute_error(te.total_loan_principal, mort_amt + cl_amt) / 1e4,
                    "월상환 MAE(만원)": mean_absolute_error(te.pay_orig, pay) / 1e4,
                    "DSR MAE(%p)": mean_absolute_error(te.dsr_total, dsr) * 100,
                    "DSR 판정 일치율": (np.digitize(te.dsr_total, [0.3, 0.4]) == np.digitize(dsr, [0.3, 0.4])).mean()})
    print(pd.DataFrame(out).set_index("방법").to_string(float_format=lambda x: f"{x:,.3f}"))
    print(f"   (참고) 검증 건 실제값 중앙값: 총대출 {te.total_loan_principal.median() / 1e4:,.0f}만원, "
          f"월상환 {te.pay_orig.median() / 1e4:,.1f}만원, 전체 대출 DSR {te.dsr_total.median() * 100:.1f}%")
    return tbl


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    d = load_purchases()
    print(f"분석 대상 {len(d):,}건, 조합별 {d.combo.value_counts().sort_index().to_dict()}")

    # 검증 A: 무작위 80/20 (조합 비율을 학습/검증에 똑같이 맞춤)
    train, test = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d.combo)
    evaluate(train, test, "검증 A: 무작위 80/20")

    # 검증 B: 시간 순서 — 2022~2025 학습 → 2026 검증
    evaluate(d[d.purchase_year <= 2025], d[d.purchase_year == 2026], "검증 B: 2022~2025 학습 → 2026 검증")

    # 최종 모델: 검증이 끝났으니 전체 데이터로 다시 학습해 저장 (웹 화면에서 사용)
    MODEL_DIR.mkdir(exist_ok=True)
    final = {
        "combo": make_gbm("clf").fit(d[FEATURES], d.combo),
        "mort_ltv": make_gbm("reg").fit(d[d.combo >= 1][FEATURES], d[d.combo >= 1].mort_ltv),
        "cl_ratio": make_gbm("reg").fit(d[d.combo == 2][FEATURES], d[d.combo == 2].cl_ratio),
    }
    final["scaler"] = StandardScaler().fit(d[KNN_FEATURES])
    final["sido_categories"] = list(d.sido.cat.categories)
    joblib.dump(final, MODEL_DIR / "home_loan_models.joblib")
    print(f"\n최종 모델 저장: {MODEL_DIR / 'home_loan_models.joblib'}")
