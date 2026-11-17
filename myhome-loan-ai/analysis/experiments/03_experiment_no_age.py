# =============================================================================
# 실험: 1인 가구 소득 보정 + 나이 제거 모델
#   - 1인 가구인데 가구소득이 개인소득의 3배 이상 → 가구소득을 개인소득으로 통일
#   - 나이(age_at_purchase)를 입력 변수에서 제거한 모델을 따로 학습
#   - 효과를 나눠 보기 위해 2×2 비교: 소득(원본/보정) × 나이(포함/제거)
#   - 기존 모델(models/home_loan_models.joblib)은 그대로 두고 별도 파일로 저장
#   - 실행: ..\.venv\Scripts\python.exe 03_experiment_no_age.py   (home_loan_mvp 폴더에서)
# =============================================================================
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split

from prep import (LTV_CAP, MODEL_DIR, NUM_FEATURES, load_purchases, monthly_credit_line,
                  monthly_mortgage)

pd.set_option("display.width", 200)
SEED = 42
RATIO_LIMIT = 3.0


def make_gbm(kind):
    # 02_train.py와 같은 설정 (비교를 공정하게)
    common = dict(categorical_features="from_dtype", learning_rate=0.05, max_iter=400,
                  early_stopping=True, validation_fraction=0.15, random_state=SEED)
    return HistGradientBoostingRegressor(**common) if kind == "reg" else HistGradientBoostingClassifier(**common)


def fix_income(df):
    """1인 가구 & 가구소득 ≥ 개인소득×3 → 가구소득을 개인소득으로, 소득 관련 파생값도 다시 계산"""
    df = df.copy()
    m = (df.household_size == 1) & (df.household_annual_income >= df.primary_annual_income * RATIO_LIMIT)
    df["income_fixed"] = m
    df.loc[m, "household_annual_income"] = df.loc[m, "primary_annual_income"]
    df["log_income"] = np.log(df["household_annual_income"])
    df["dsr_orig"] = df.pay_orig * 12 / df.household_annual_income   # DSR 정답값도 보정 소득 기준으로
    return df


def run_variant(train, test, features, rates):
    """한 가지 설정(데이터 + 입력 변수)으로 모델 3개 학습 → 검증 지표 반환"""
    out, models = {}, {}

    # (1) 대출 조합 분류
    clf = make_gbm("clf").fit(train[features], train.combo)
    pr = clf.predict_proba(test[features])
    out["조합 정확도"] = accuracy_score(test.combo, pr.argmax(1))
    out["조합 log loss"] = log_loss(test.combo, pr, labels=[0, 1, 2])
    models["combo"] = clf

    # (2) 주담대 LTV, 마통 비율 (해당 조합인 사람만)
    tr1, te1 = train[train.combo >= 1], test[test.combo >= 1]
    tr2, te2 = train[train.combo == 2], test[test.combo == 2]
    models["mort_ltv"] = make_gbm("reg").fit(tr1[features], tr1.mort_ltv)
    models["cl_ratio"] = make_gbm("reg").fit(tr2[features], tr2.cl_ratio)
    ltv = pd.Series(np.clip(models["mort_ltv"].predict(te1[features]), 0, LTV_CAP), index=te1.index)
    clr = pd.Series(models["cl_ratio"].predict(te2[features]), index=te2.index).reindex(te1.index).fillna(0).clip(lower=0)
    out["LTV MAE(%p)"] = mean_absolute_error(te1.mort_ltv, ltv) * 100
    out["마통비율 MAE(%p)"] = mean_absolute_error(te2.cl_ratio, clr.loc[te2.index]) * 100

    # (3) 최종값: 조합은 실제 기준, 금리·만기는 모든 설정에 같은 값(학습 데이터 중앙값/최빈값) 사용
    mort_amt, cl_amt = te1.purchase_price_est * ltv, te1.purchase_price_est * clr
    pay = (monthly_mortgage(mort_amt, rates["mort"], rates["term"])
           + np.where(te1.combo == 2, monthly_credit_line(cl_amt, rates["cl"]), 0))
    dsr = pay * 12 / te1.household_annual_income
    out["총대출 MAE(만원)"] = mean_absolute_error(te1.total_loan_principal, mort_amt + cl_amt) / 1e4
    out["월상환 MAE(만원)"] = mean_absolute_error(te1.pay_orig, pay) / 1e4
    out["DSR MAE(%p)"] = mean_absolute_error(te1.dsr_orig, dsr) * 100
    out["DSR 판정 일치율"] = (np.digitize(te1.dsr_orig, [0.3, 0.4]) == np.digitize(dsr, [0.3, 0.4])).mean()

    # 보정 대상(1인 가구 3배 이상) 검증 건만 따로
    f = te1.income_fixed.to_numpy() if "income_fixed" in te1 else np.zeros(len(te1), bool)
    if f.any():
        out["[보정건] LTV MAE(%p)"] = mean_absolute_error(te1.mort_ltv[f], ltv[f]) * 100
        out["[보정건] DSR MAE(%p)"] = mean_absolute_error(te1.dsr_orig[f], dsr[f]) * 100
    return out, models


if __name__ == "__main__":
    base = load_purchases()
    fixed = fix_income(base)
    base["income_fixed"] = fixed.income_fixed  # 원본 데이터에도 '보정 대상' 표시만 붙여 같은 건끼리 비교

    print(f"분석 대상 {len(base):,}건 중 소득 보정 대상 {fixed.income_fixed.sum():,}건 "
          f"(1인 가구 {(base.household_size == 1).sum():,}건의 {fixed.income_fixed.sum() / (base.household_size == 1).sum():.1%})")
    t = fixed[fixed.income_fixed & (fixed.combo > 0)]
    print(f"보정 대상 중 대출자 {len(t)}명: DSR 중앙값 {base.loc[t.index].dsr_orig.median():.1%} → {t.dsr_orig.median():.1%}, "
          f"DSR 40% 초과 {(base.loc[t.index].dsr_orig > 0.4).sum()}명 → {(t.dsr_orig > 0.4).sum()}명")

    # 같은 행을 같은 학습/검증으로 나눔 (4가지 설정 모두 동일한 분할)
    tr_idx, te_idx = train_test_split(base.index, test_size=0.2, random_state=SEED, stratify=base.combo)
    tr = base.loc[tr_idx]
    rates = {"mort": tr[tr.combo >= 1].mortgage_rate.median(), "cl": tr[tr.combo == 2].cl_rate.median(),
             "term": tr[tr.combo >= 1].mortgage_term_years.mode()[0]}

    feats_age = NUM_FEATURES + ["sido"]
    feats_no_age = [c for c in feats_age if c != "age_at_purchase"]
    variants = {
        "① 원본 소득 + 나이 포함 (기존)": (base, feats_age),
        "② 원본 소득 + 나이 제거": (base, feats_no_age),
        "③ 보정 소득 + 나이 포함": (fixed, feats_age),
        "④ 보정 소득 + 나이 제거 (요청)": (fixed, feats_no_age),
    }
    results, saved = {}, None
    for name, (df, feats) in variants.items():
        res, models = run_variant(df.loc[tr_idx], df.loc[te_idx], feats, rates)
        results[name] = res
        if name.startswith("④"):
            saved = (models, df, feats)
    print("\n검증 결과 (검증 데이터 20%, 같은 행끼리 비교 / DSR은 각 설정의 소득 기준 정답과 비교)")
    print(pd.DataFrame(results).to_string(float_format=lambda x: f"{x:,.3f}"))

    # 나이를 빼면 모델이 무엇에 기대는지: 순열 중요도 (값을 섞었을 때 오차가 얼마나 커지나)
    models, df, feats = saved
    te1 = df.loc[te_idx][df.loc[te_idx].combo >= 1]
    for key, label in [("mort_ltv", "④ LTV 모델"), ("combo", "④ 조합 모델")]:
        X = te1[feats] if key == "mort_ltv" else df.loc[te_idx][feats]
        y = te1.mort_ltv if key == "mort_ltv" else df.loc[te_idx].combo
        imp = permutation_importance(models[key], X, y, n_repeats=5, random_state=SEED)
        print(f"\n{label} 변수 중요도 (클수록 중요)")
        print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(4).to_string())

    # 나이 대신 무엇이 LTV를 설명하는지 참고: 나이와 다른 입력 변수의 상관
    print("\n(참고) 매입 당시 나이와 다른 입력 변수의 상관 (Spearman, 보정 데이터)")
    print(fixed[NUM_FEATURES].corr(method="spearman")["age_at_purchase"].drop("age_at_purchase").round(2).to_string())

    # 요청한 ④ 모델을 전체 데이터로 다시 학습해 별도 파일로 저장 (기존 모델은 건드리지 않음)
    final = {"combo": make_gbm("clf").fit(df[feats], df.combo),
             "mort_ltv": make_gbm("reg").fit(df[df.combo >= 1][feats], df[df.combo >= 1].mort_ltv),
             "cl_ratio": make_gbm("reg").fit(df[df.combo == 2][feats], df[df.combo == 2].cl_ratio),
             "features": feats, "income_fix_ratio": RATIO_LIMIT}
    path = MODEL_DIR / "home_loan_models_no_age_income_fix.joblib"
    joblib.dump(final, path)
    print(f"\n④ 모델 저장: {path}")
