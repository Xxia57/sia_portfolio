# 원본과 팀원 전달 데이터 비교

검증일: 2026-10-06. 현재 full data 폴더의 CSV 10개와 csv/raw 데이터의 CSV 6개를 비교했다. 원본은 수정하지 않았다. SQL 또는 최초 결합 코드가 없으므로 실행 이력 자체는 확정할 수 없다. 아래의 확인은 ID 집합과 전체 행의 값 대조로 확인한 데이터 관계다. 숫자는 상대오차 1e-10, 절대오차 1e-6 이내를 동일하게 취급했고 날짜의 자정 표시는 정규화했다.

## 판단

주택 매입자 42,225명의 구매가격·대출·소득 관계를 분석한다면 팀원 결합 파일을 기반으로 전처리하는 것이 합리적이다. 원본과 대조한 주요 원본 컬럼 및 최근 12개월 집계가 일치했다. 원본을 보관하고 파생컬럼 및 대상 제한을 문서화해야 한다.

전체 8만 명의 재무건강, 무주택자 비교, 월별 변화, 향후 위험 예측이 목적이라면 full data에서 다시 구성하는 편이 적합하다. 팀원 고객 파일은 부동산 보유자만 남았고, 월별 기록도 12개월 평균으로 축약됐다. 이 선택은 전체 고객 분석에 사용할 경우 표본을 제한한다.

full data에도 summary·profiles 등 이미 집계된 테이블이 포함되어 있다. 따라서 이 폴더를 모든 거래가 들어 있는 완전한 최초 원자료라고 확정할 수는 없다. 현재 파일은 10개이며, 말씀하신 약 30개 전체 테이블은 아직 여기서 확인되지 않는다.

## 6개 파일의 구성

| 팀원 파일 | 확인한 구성 | 행·컬럼 |
|---|---|---|
| customers.csv | financial_profiles + debt_summary + 월별 cashflow의 최근 12개월 집계 + 계산컬럼. 부동산 보유자만 포함 | 42,225 × 72 |
| home_purchases_loanremove.csv | real_assets + customer_real_estate 일부 + 주택 구매에 연결된 대출 집계 + 팀원 customers 전체 + 구매 관련 계산컬럼 | 42,225 × 116 |
| loans_raw.csv | 원본 loans에서 위 고객 42,225명의 대출만 선택. 컬럼 삭제 없음 | 52,846 × 22 |
| real_estate_raw.csv | 원본 real_assets와 전체 행·컬럼 값 동일. 날짜 표시 형식 차이만 정규화 | 42,225 × 9 |
| real_estate_detail_raw.csv | 원본 customer_real_estate와 전체 행·컬럼 값 동일 | 42,225 × 17 |
| customer_financial_health_with_ai.csv | 전체 8만 명의 프로필·부채·자산 관련 값과 점수·AI 결과. 점수/AI의 생성 원본은 현재 폴더에 없음 | 80,000 × 19 |

## customers.csv에서 달라진 내용

- 대상 ID는 원본 real_assets의 customer_id 집합과 정확히 같다. 원본 80,000명 중 37,775명은 포함되지 않았다. 결합 시 inner join인지 이후 필터인지 실행 순서는 알 수 없다.
- financial_profiles에서 name, birth_date를 제외하고 region을 residence_region으로 바꿨다. 나머지 값은 일치한다.
- debt_summary의 모든 정보를 유지했다. 중복 이름인 is_home_owner는 debt_is_home_owner로, monthly_private_education_cost는 debt_monthly_private_education_cost로 구분했다.
- household_monthly_income = household_annual_income / 12.
- 원본 cashflow는 2025-04~2026-09의 18개월, 1,440,000행이다. 팀원 평균 10개는 2025-10~2026-09의 최근 12개월 평균과 전체 행에서 일치한다. 전체 18개월 또는 최초 12개월 평균은 일치하지 않는다.
- cf_months는 12. avg_monthly_consumption = avg_monthly_card_spend + avg_monthly_non_card_spend. savings_rate = avg_monthly_savings / avg_monthly_primary_income. 현재 파일의 전체 행에서 계산이 일치한다.
- 월별 시계열 자체와 교육비 세부 항목, 사기 이체 손실 등은 이 요약파일에 유지되지 않았다.
- DB customers 테이블의 name, birth_date, city, occupation, kyc_level, signup_channel, created_at, updated_at은 팀원 고객 파일에 없다. 공통 컬럼이 다른 프로필에도 있으므로 DB customers가 직접 결합됐다는 증거는 아니다.

## 주택 매입 결합과 컬럼 제외

- real_assets 9개 컬럼은 모두 유지된다.
- customer_real_estate에서는 기존 키 customer_id·real_asset_id와 중복 estimated_value 외에 10개 상세컬럼이 추가된다. reference_trade_id, apt_seq, jibun, reference_only는 매입 결합 파일에 없다. 이 4개는 별도 real_estate_detail_raw에는 남아 있다.
- 팀원 customers의 customer_id를 제외한 71개 컬럼이 모두 붙고 값도 일치한다.
- 대출은 housing_purchase_linked = 1인 행을 reference_real_asset_id로 집계한 결과와 일치한다. 전체 참조 대출을 집계하면 맞지 않는다. 부동산 ID가 연결된 개인대출이라고 해서 모두 매입 대출로 들어가는 것은 아니다.
- n_loans, total_loan_principal, mortgage_principal, other_loan_principal, monthly_payment_total, annual_debt_service_total의 집계가 전체 행에서 일치한다.
- 매입 대출 금액은 현재 principal 또는 outstanding_balance가 아닌 original_principal이다. 고객 부채 요약의 mortgage_balance 등은 현재 잔액이므로 구분해야 한다.
- mortgage_rate, mortgage_term_years, ltv_at_origination은 연결된 주택담보대출의 값과 일치한다. weighted_rate는 original_principal을 가중치로 계산한 금리와 일치한다.
- loan_types, loan_opened_at은 대출 유형 및 개설일을 나타낸다. 문자열 결합의 정확한 생성 순서는 최초 코드 없이 확정하지 않았다.

### 확인한 구매 계산

| 컬럼 | 확인한 계산 |
|---|---|
| bought_with_loan | n_loans > 0 |
| purchase_price_est | purchase_reference_price. 현재 모든 행에서 동일 |
| price_per_pyeong | purchase_price_est / (exclusive_area_m2 / 3.3058) |
| own_funds_est | max(purchase_price_est - total_loan_principal, 0) |
| own_funds_ratio | own_funds_est / purchase_price_est |
| purchase_year | purchase_date의 연도 |
| age_at_purchase | floor((purchase_date - birth_date)의 일수 / 365.25) |
| under20_at_purchase | age_at_purchase < 20 |
| price_to_income | purchase_price_est / household_annual_income |
| loan_to_income | 원래 total_loan_principal / household_annual_income. 무대출은 원래 결측 유지 |
| payment_to_income | annual_debt_service_total / household_annual_income |

위 식은 전체 행에서 일치했다. age_at_purchase는 정확한 생일 기준 만 나이와 52건 다르다. sido와 sigungu는 region 분리 결과로 보이지만 모든 행의 구체적 분리 규칙은 아직 검증하지 않았다. price_source의 의미는 최초 생성 코드 또는 자료 설명이 필요하다. reference_only가 전부 1이므로 purchase_reference_price를 고객의 증빙된 실제 구매가격으로 단정하지 않는다.

자기자금은 매입 대출이 구매가격을 초과한 2건(customer_id 2058, 52735)에서 음수를 0으로 잘라 놓았다. 이 처리는 초과대출을 숨길 수 있으므로 원래 차액을 별도로 보존하는 것이 좋다.

### loanremove가 바꾼 것

csv/home_purchases.csv와 대조하면 행 수·컬럼 수·ID가 동일하다. 변경된 컬럼은 total_loan_principal, mortgage_principal, other_loan_principal 3개이며 각각 11,920건의 결측값이 0으로 변경됐다. 다른 112개 비키 컬럼은 동일하다. 따라서 이름과 달리 대출 컬럼이나 고객 행을 삭제한 파일이 아니다.

무대출 11,920건의 금리·상환기간·대출 관련 비율은 여전히 결측이다. 대출금액의 0과 금리의 비해당 결측을 구분해서 처리해야 한다.

## loans_raw.csv에서 제외된 행

원본 loans는 60,434행. 팀원 loans_raw는 52,846행이다. 원본에서 팀원 customers의 customer_id만 선택한 loan_id 집합과 정확히 일치한다. 제외된 7,588행은 모두 PERSONAL 대출이다. PERSONAL 전체를 삭제한 것은 아니며 부동산 보유자의 PERSONAL 대출 6,481행은 남아 있다. 포함된 대출은 22개 컬럼이 그대로이고 정규화 후 값 차이가 없다.

## 재무건강·AI 파일의 확인 범위

- age, employment_type은 financial_profiles와 일치한다.
- dsr_income, credit_line_balance, housing_education_burden_ratio는 debt_summary와 일치한다.
- net_worth_observed는 asset_summary의 net_worth, stock_current_value는 stock_value와 전체 행에서 일치한다.
- savings_rate_12m은 최근 12개월 평균 저축 / 평균 개인소득과 일치한다.
- stock_concentration은 stock_hhi와 다른 지표다. 23,133행에서 값이 다르며 같은 이름으로 치환하면 안 된다.
- financial_health_score, financial_health_grade, liquidity_months, stock_concentration, diagnosis_summary, next_month_stress_probability, next_month_stress_alert, ai_stress_level, ai_recommendation의 정확한 생성식 또는 모델은 현재 원본 10개로 재현하지 못했다.
- 기준월은 2026-09-01. 이 파일은 전체 80,000명이므로 팀원 customers와 inner join하면 42,225명만 남는다.

## 전처리 시 우선 정할 기준

1. 분석 대상을 전체 고객으로 할지, 주택 매입자로 할지 정한다.
2. 현재 소득과 과거 매입시점의 대출·가격을 비교한다는 시점 차이를 명시한다. 과거 소득 자료는 현재 파일에서 확인되지 않는다.
3. 구매 당시 original_principal과 현재 outstanding_balance를 구분한다.
4. 정확한 만 나이가 필요하면 birth_date로 다시 계산한다. 현재 20세 미만 매입 플래그 558건은 삭제 전에 별도 검토한다.
5. 무대출의 0과 비해당 결측을 구분하고, 이미 0으로 잘린 자기자금 차액도 기록한다.
6. 위험 예측 모델을 새로 만들 경우 기존 AI 예측값·점수·등급을 입력으로 넣을지는 예측 시점과 목표에 따라 결정한다. 목표에서 파생한 값을 입력에 넣는 누수가 있는지 점검한다.
7. 결합에 필요한 테이블만 선택한다. 계좌는 accounts를 고객별로 집계하고, 월별 현금흐름은 customer_id + month의 단위를 유지한다. 고객별 1행 데이터에 그대로 합치면 행이 늘어나 중복 집계가 발생할 수 있다.

자세한 전체 행 비교 건수는 같은 폴더의 컬럼_값_대조.csv에 있다. 후보 계산식이 불일치하는 행도 검증 흔적으로 포함되어 있으므로 보고서에서 확정한 식과 구분해서 읽어야 한다.
