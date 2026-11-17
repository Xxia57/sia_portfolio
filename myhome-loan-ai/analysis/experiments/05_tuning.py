# =============================================================================
# 하이퍼파라미터 튜닝 — 지금 설정값이 최선인지 확인
#   공정한 비교를 위해 데이터를 3개로 나눔
#     학습(64%) → 후보 설정으로 학습 / 튜닝용 검증(16%) → 설정 고르기 / 최종 검증(20%) → 마지막 1번만 확인
#   (최종 검증 20%는 02_train.py의 검증 A와 같은 행)
#   (A) kNN: 집단 크기 K
#   (B) GBM: learning_rate, max_leaf_nodes(나무 복잡도), min_samples_leaf(잎당 최소 표본), l2 규제
#   실행: ..\.venv\Scripts\python.exe 05_tuning.py   (home_loan_mvp 폴더에서)
# =============================================================================
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from prep import KNN_FEATURES, LTV_CAP, NUM_FEATURES, load_purchases

pd.set_option("display.width", 220)
SEED = 42
FEATURES = NUM_FEATURES + ["sido"]
CURRENT = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0)

d = load_purchases()
trainval, test = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d.combo)
train, val = train_test_split(trainval, test_size=0.2, random_state=SEED, stratify=trainval.combo)
print(f"학습 {len(train):,} / 튜닝용 검증 {len(val):,} / 최종 검증 {len(test):,}")

# ---------------------------------------------------------------------------
# (A) kNN 집단 크기 K — 목표: 대출 이용자의 LTV를 집단 중앙값으로 맞힐 때 오차, 25~75% 구간 포함률(50%가 이상적)
# ---------------------------------------------------------------------------
def knn_eval(tr, te, k):
    tr_l, te_l = tr[tr.combo > 0], te[te.combo > 0]
    sc = StandardScaler().fit(tr_l[KNN_FEATURES])
    idx = NearestNeighbors(n_neighbors=k).fit(sc.transform(tr_l[KNN_FEATURES])).kneighbors(
        sc.transform(te_l[KNN_FEATURES]), return_distance=False)
    y, yt = tr_l.mort_ltv.to_numpy(), te_l.mort_ltv.to_numpy()
    med = np.median(y[idx], axis=1)
    lo, hi = np.quantile(y[idx], 0.25, axis=1), np.quantile(y[idx], 0.75, axis=1)
    return mean_absolute_error(yt, med) * 100, ((yt >= lo) & (yt <= hi)).mean() * 100, (hi - lo).mean() * 100


rows = []
for k in [30, 50, 100, 200, 400]:
    mae, cover, width = knn_eval(train, val, k)
    rows.append({"K": k, "LTV MAE(%p)": mae, "25~75% 포함률(%)": cover, "구간 폭(%p)": width})
print("\n(A) kNN 집단 크기 K — 튜닝용 검증")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ---------------------------------------------------------------------------
# (B) GBM 격자 탐색 — 모델 3개(조합 분류, LTV, 마통 비율) 각각
# ---------------------------------------------------------------------------
def make(kind, params):
    kw = dict(categorical_features="from_dtype", max_iter=1000, early_stopping=True,
              validation_fraction=0.15, n_iter_no_change=10, random_state=SEED, **params)
    return HistGradientBoostingRegressor(**kw) if kind == "reg" else HistGradientBoostingClassifier(**kw)


TASKS = {  # 이름: (종류, 대상 행 조건, 목표 컬럼)
    "조합(분류)": ("clf", lambda x: x, "combo"),
    "주담대 LTV": ("reg", lambda x: x[x.combo > 0], "mort_ltv"),
    "마통 비율": ("reg", lambda x: x[x.combo == 2], "cl_ratio"),
}


def score(kind, model, X, y):
    # 분류는 log loss(확률 품질), 회귀는 MAE — 둘 다 낮을수록 좋음
    if kind == "clf":
        return log_loss(y, model.predict_proba(X), labels=[0, 1, 2])
    return mean_absolute_error(y, model.predict(X))


grid = [dict(learning_rate=lr, max_leaf_nodes=ml, min_samples_leaf=ms, l2_regularization=l2)
        for lr, ml, ms, l2 in itertools.product([0.03, 0.05, 0.1], [15, 31, 63], [20, 50, 100], [0.0, 1.0])]
best = {}
t0 = time.time()
for name, (kind, sel, col) in TASKS.items():
    a, b = sel(train), sel(val)
    res = []
    for prm in grid:
        m = make(kind, prm).fit(a[FEATURES], a[col])
        res.append({**prm, "점수": score(kind, m, b[FEATURES], b[col]), "반복수": m.n_iter_})
    r = pd.DataFrame(res).sort_values("점수")
    cur = r[(r.learning_rate == CURRENT["learning_rate"]) & (r.max_leaf_nodes == CURRENT["max_leaf_nodes"])
            & (r.min_samples_leaf == CURRENT["min_samples_leaf"]) & (r.l2_regularization == CURRENT["l2_regularization"])]
    best[name] = r.iloc[0][list(CURRENT)].to_dict()
    unit = "log loss" if kind == "clf" else "MAE"
    rank = list(r.index).index(cur.index[0]) + 1
    print(f"\n(B) {name} — 튜닝용 검증 {unit} (낮을수록 좋음), {len(grid)}개 조합 중 상위 5개 + 현재 설정 ({rank}위)")
    show = pd.concat([r.head(5), cur]).drop_duplicates()
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.5f}"))
print(f"\n(격자 탐색 {time.time() - t0:.0f}초)")

# ---------------------------------------------------------------------------
# (C) 최종 검증 20% — 현재 설정 vs 튜닝 설정 (학습+튜닝용 검증 80%로 다시 학습)
# ---------------------------------------------------------------------------
rows = []
for label, pick in [("현재 설정", lambda n: CURRENT), ("튜닝 설정", lambda n: best[n])]:
    row = {"설정": label}
    for name, (kind, sel, col) in TASKS.items():
        prm = {k: (int(v) if k in ("max_leaf_nodes", "min_samples_leaf") else float(v)) for k, v in pick(name).items()}
        a, b = sel(trainval), sel(test)
        m = make(kind, prm).fit(a[FEATURES], a[col])
        if kind == "clf":
            row["조합 정확도"] = accuracy_score(b[col], m.predict(b[FEATURES]))
            row["조합 log loss"] = score(kind, m, b[FEATURES], b[col])
        else:
            pr = m.predict(b[FEATURES])
            pr = np.clip(pr, 0, LTV_CAP) if col == "mort_ltv" else pr
            row[f"{name} MAE(%p)"] = mean_absolute_error(b[col], pr) * 100
    rows.append(row)
print("\n(C) 최종 검증 (한 번만 확인)")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print("\n튜닝으로 고른 설정:", {k: v for k, v in best.items()})
