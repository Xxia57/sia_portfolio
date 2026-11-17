# =============================================================================
# 실험: 나이의 영향을 단계적으로 낮추면 어떻게 되나
#   (A) GBM — 변수별 가중치를 직접 줄 수 없어서, '나이 포함 모델'과 '나이 제외 모델'의 예측을 섞음
#       예측 = w × (나이 포함) + (1 − w) × (나이 제외),  w = 나이 반영도 (1 = 지금, 0 = 나이 없음)
#   (B) kNN — 표준화한 거리에서 나이 가중치는 낮추고 소득·자산 가중치는 높임
#   검증: 02_train.py와 같은 무작위 80/20 분할
#   실행: ..\.venv\Scripts\python.exe 04_experiment_age_weight.py   (home_loan_mvp 폴더에서)
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from prep import KNN_FEATURES, LTV_CAP, NUM_FEATURES, load_purchases

pd.set_option("display.width", 200)
SEED = 42


def gbm(kind):
    kw = dict(categorical_features="from_dtype", learning_rate=0.05, max_iter=400,
              early_stopping=True, validation_fraction=0.15, random_state=SEED)
    return HistGradientBoostingRegressor(**kw) if kind == "reg" else HistGradientBoostingClassifier(**kw)


d = load_purchases()
tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d.combo)
F_AGE = NUM_FEATURES + ["sido"]
F_NO = [f for f in F_AGE if f != "age_at_purchase"]

# ---------------------------------------------------------------------------
# (A) GBM 예측 섞기
# ---------------------------------------------------------------------------
L_tr, L_te = tr[tr.combo > 0], te[te.combo > 0]
C_tr, C_te = tr[tr.combo == 2], te[te.combo == 2]
pred = {}
for name, F in [("age", F_AGE), ("no", F_NO)]:
    pred[name] = {
        "ltv": np.clip(gbm("reg").fit(L_tr[F], L_tr.mort_ltv).predict(L_te[F]), 0, LTV_CAP),
        "cl": gbm("reg").fit(C_tr[F], C_tr.cl_ratio).predict(C_te[F]),
        "combo": gbm("clf").fit(tr[F], tr.combo).predict_proba(te[F]),
    }
cl_full = pd.Series(0.0, index=L_te.index)
rows = []
for w in [1.0, 0.75, 0.5, 0.25, 0.0]:
    mix = {k: w * pred["age"][k] + (1 - w) * pred["no"][k] for k in pred["age"]}
    cl = cl_full.copy()
    cl.loc[C_te.index] = mix["cl"]
    loan = L_te.purchase_price_est * (mix["ltv"] + np.where(L_te.combo == 2, cl.clip(lower=0), 0))
    # 젊은층(34세 이하)과 고령층(55세 이상)의 LTV 예측 평균 — 나이 영향이 줄면 두 값이 가까워짐
    young, old = (L_te.age_at_purchase <= 34).to_numpy(), (L_te.age_at_purchase >= 55).to_numpy()
    rows.append({
        "나이 반영도 w": w,
        "조합 정확도": accuracy_score(te.combo, mix["combo"].argmax(1)),
        "LTV MAE(%p)": mean_absolute_error(L_te.mort_ltv, mix["ltv"]) * 100,
        "총대출 MAE(만원)": mean_absolute_error(L_te.total_loan_principal, loan) / 1e4,
        "예측 LTV 34세↓": mix["ltv"][young].mean() * 100,
        "예측 LTV 55세↑": mix["ltv"][old].mean() * 100,
    })
print("(A) GBM — 나이 반영도별 정확도 (실제 LTV 평균: 34세 이하 "
      f"{L_te.mort_ltv[L_te.age_at_purchase <= 34].mean() * 100:.1f}%, 55세 이상 "
      f"{L_te.mort_ltv[L_te.age_at_purchase >= 55].mean() * 100:.1f}%)")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

# ---------------------------------------------------------------------------
# (B) kNN 가중치 — 나이 ↓, 소득·자산 ↑
# ---------------------------------------------------------------------------
sc = StandardScaler().fit(tr[KNN_FEATURES])
Xtr, Xte = sc.transform(tr[KNN_FEATURES]), sc.transform(te[KNN_FEATURES])
i_age, i_inc, i_ast = (KNN_FEATURES.index(c) for c in ["age_at_purchase", "log_income", "log_assets"])
sample = te.sample(1500, random_state=SEED)
pos = te.index.get_indexer(sample.index)
rows = []
for age_w, fin_w in [(1, 1), (0.5, 1.5), (0.25, 2), (0, 2)]:
    wv = np.ones(len(KNN_FEATURES))
    wv[i_age], wv[i_inc], wv[i_ast] = age_w, fin_w, fin_w
    nn = NearestNeighbors(n_neighbors=100).fit(Xtr * wv)
    idx = nn.kneighbors(Xte[pos] * wv, return_distance=False)
    g_age = np.array([np.abs(tr.age_at_purchase.to_numpy()[i] - a).mean() for i, a in zip(idx, sample.age_at_purchase)])
    g_inc = np.array([np.abs(tr.log_income.to_numpy()[i] - v).mean() for i, v in zip(idx, sample.log_income)])
    g_ast = np.array([np.abs(tr.log_assets.to_numpy()[i] - v).mean() for i, v in zip(idx, sample.log_assets)])
    ltv_tr = tr.mort_ltv.to_numpy()
    med = np.array([np.nanmedian(ltv_tr[i][tr.combo.to_numpy()[i] > 0]) for i in idx])
    has = sample.combo.to_numpy() > 0
    rows.append({
        "나이 가중치": age_w, "소득·자산 가중치": fin_w,
        "집단 나이 차(세)": g_age.mean(),
        "집단 소득 차(%)": (np.exp(g_inc.mean()) - 1) * 100,
        "집단 자산 차(%)": (np.exp(g_ast.mean()) - 1) * 100,
        "집단 중앙값 LTV MAE(%p)": np.abs(med[has] - sample.mort_ltv.to_numpy()[has]).mean() * 100,
    })
print("\n(B) kNN — 가중치별 비교 집단의 '닮은 정도'와 LTV 오차 (검증 1,500명 표본)")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
