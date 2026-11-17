**당신이 잠든 사이에 · 보유 종목 맞춤형 장전 브리핑**

[데모](https://4digit-demo.vercel.app/) · [WI26 검증](analysis/wi26-validation/review.md) · [야간선물 데이터 검증](analysis/kis-data-review/validation-summary.md) · [팀 원본](https://github.com/minseokgeum/ai_campus_main_proj)
2026.10.12~2026.11.27

한국 투자자에게 **밤사이 미국 뉴스 중 보유 종목과 관련된 것**을 골라 보여 주는 서비스. DART 공시와 미국 뉴스로 국내 기업–해외 기업·제품을 잇는 그래프 DB를 만들고, 그래프에서 종목마다 구글 뉴스 검색어를 만든다. 결정 사항과 근거는 [`graph_db/graphDB.md`](graph_db/graphDB.md)(요약: [`graphDB_summary.md`](graph_db/graphDB_summary.md))에 있다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `graph_db/` | 그래프 DB 만들기: DART·뉴스 수집과 LLM 추출, 이름 통일, 제품 사전, Neo4j 적재, 검색어 만들기. 노트북과 `graphDB.md`, 지시문(`prompts/`) |
| `news_search/` | 밤사이 기사 모으기: 검색어로 구글 뉴스 RSS 검색, 거르기, 같은 소식 묶기, 대표 기사 본문 받기, 프롬프트 입력. 24시간 수집 `news_collect.ipynb`, 수집 마감 뒤 처리 `news_process.ipynb`(사용자가 고른 표시 시각 30분 전에 한 번). 결정은 [`news_search.md`](news_search/news_search.md) |
| `news_search_tests/` | 구글 뉴스 검색 방식 시험(`search_tests.ipynb`: 매체, OR 묶음, 기간 조건, 거르기, 검색어 모양). 서비스 실행에는 필요 없음 |
| `news_source_tests/` | 매체별 RSS·Atom 피드 점검 및 본문 추출·정제 시험. RSS 요약과 정제 본문을 구분해 저장하고, 영상·라이브 콘텐츠, 구독 제한, 본문 추출 실패를 상태별로 기록. [RSS 통합 수집·정제 안내](news_source_tests/team_rss_news/docs/RSS_HANDOFF.md) |
| `graph_news_overview.md` | **처음 보는 팀원용 안내**: 그래프 DB, 검색어, 구글 뉴스 수집·처리가 무엇을 만들고 어떻게 도는지, 결과 위치와 첫 실전 결과 |
| `nightly_prompt_requirements.md` | 밤사이 뉴스 프롬프트를 만드는 팀원에게 전하는 요구사항(입력·출력, 유형 분류, 그래프용 추출 형식, 실행 조건, 시험 방법) |
| `scripts/` | 보조 스크립트: 데이터 묶음 만들기·풀기(`data_bundle.py`) |
| `market_data/` | 기준 자료: 상장사·WI26 업종(`sector.json`), 토스 테마(`toss_theme.json`), 지수 구성 종목. [README](market_data/README.txt) |
| `backtest/`, `data_research/`, `stockdata_collection/` | 다른 작업(시세 수집, 백테스트 등) |

## 준비

1. **Python 3.11**과 패키지: `python -m pip install -r requirements.txt`
2. **비밀값**: `.env.example`을 프로젝트 루트에 `.env`로 복사해 값을 채운다. `.env`는 git에 올리지 않는다.
3. **Codex CLI**(일부 LLM 단계를 구독으로 돌린다): 설치 후 `codex login`. 노트북은 모델을 직접 지정한다(`gpt-6.1-terra`, 일부 `gpt-6.1-sol`).
4. **Neo4j**: 로컬 Docker(`neo4j-graph`, 7474·7687 포트). 주소·계정은 `.env`.
5. **데이터**: 작업별 입력 범위는 아래 ‘데이터 범위’를 참고하세요.

## 데이터 범위

이 저장소에서 공개 코드와 문서, 기준 자료를 확인할 수 있습니다. 원천 기사와 대용량 시세 데이터가 필요한 작업은 별도의 입력 데이터가 있어야 실행할 수 있습니다.

## 실행 순서 (graph_db)

노트북은 `graph_db/` 폴더에서 연다. 비용이 드는 단계는 스위치가 꺼져 있어(`RUN = False` 등) 켜야 요청을 보낸다. 같은 입력으로 끝난 요청은 건너뛰므로 다시 실행해도 된다. 자세한 흐름과 갱신 방식은 `graphDB.md` 1.1·1.2.

| 순서 | 노트북 | 하는 일 | 필요한 것 | 스위치 | 외부 요청·비용 |
|---|---|---|---|---|---|
| 1 | `dart_targets` | 서비스 대상 후보(`targets.csv`) | `market_data/`, `KRX_API_KEY` | | KRX API(무료) |
| 2 | `dart_collector` | DART 공시 받기 | `OPENDART_API_KEY` | | OpenDART(무료, 처음 약 18,000건·2~3시간) |
| 3 | `dart_parser` | 지분·계열 표 | | | 없음 |
| 4 | `dart_business_parser` | 사업의 내용 → LLM 입력 | | | 없음 |
| 5 | `dart_business_extractor` | 사업의 내용 LLM 추출 | `OPENAI_API_KEY` 또는 Codex | `RUN`, `RUNNER` | 배치 API 약 $15~20 |
| 6 | `dart_business_cleaner` | DART 최종본 | | | 없음 |
| 7 | `news_collector` | 미국 경제 매체와 업종 전문지 RSS를 주기적으로 조회하고, 정규화한 URL로 신규 기사와 본문 변경을 식별해 제목·발행 시각·RSS 요약·본문을 저장 | `NEWS_DB_URL`, 알림 키 | `REQUIRE_MYSQL`, `RUN_FOREVER` | 언론사 RSS(무료) |
| 8 | `news_filter` | URL·분류 태그·제목으로 광고, 영상·라이브 페이지, 중복 등을 판별하고, 원본을 보존한 채 본문 사용·RSS 요약 사용·제외로 표시해 관계 추출용 입력을 구성 | `NEWS_DB_READ_URL`(안 되면 `source/news/snapshot_*`) | | 없음 |
| 9 | `news_extractor` | 뉴스 LLM 추출 | `OPENAI_API_KEY` 또는 Codex | `RUN`, `RUNNER` | 배치 API 기사당 약 $0.006 |
| 10 | `news_cleaner` | 뉴스 최종본 | | | 없음 |
| 11 | `name_unifier` | 기업 이름 통일 | | | 없음 |
| 12 | `product_dictionary` | 제품 사전(규칙) | | | 처음 한 번 WordNet 받기 |
| 13 | `product_llm` | 제품 사전(LLM): 제품군 → 노드 분류 → 계층 → 브랜드. **10절에서 검토표(`graph_db/review/family_review.csv`) 알림이 나오면 LLM 결과를 직접 검토해 표를 고친 뒤 다음으로** | Codex, `OPENAI_API_KEY` | `RUN`, `BATCH_SUBMIT`, `TREE_BATCH_SUBMIT`, `RETRY_CHOICE` | Codex 구독 + 배치 API 약 $6 |
| 14 | `product_apply` | LLM 제품 사전 적용 | | | 없음 |
| 15 | `graph_db` | 그래프 표 만들기와 Neo4j 적재 | Neo4j, `NEO4J_*` | `LOAD` | 없음(로컬) |
| 16 | `search_terms` | 종목 유형, 검색어 후보, 업종·테마·회사 기본 검색어 | Codex | `RUN` | Codex 구독(요청 4건) |
| 17 | `../news_search_tests/search_tests` | 구글 뉴스 검색 방식 시험(매체, OR 묶음, 거르기, 검색어 시험. 서비스 실행에는 필요 없음) | | `RUN` | 구글 뉴스 RSS(무료, 한 IP에서 시간당 수백 건이면 막힘) |
| 18 | `../news_search/news_collect` | 24시간 수집(11절): 수집 마감(표시 시각 `SUMMARY_TIME` 30분 전, 기본 07:30)에 맞춘 일정(밤 22시~마감 1시간, 저녁 16~22시 2시간, 낮 마감~16시 4시간 간격), 수집 마감 표시 파일 | 알림 키 | `RUN_COLLECT`(시험 칸은 `RUN`) | 구글 뉴스 RSS(무료, 시험 종목 17개·검색어 169개 기준 하루 약 7,100건) |
| 19 | `../news_search/news_process` | 수집 마감 뒤 처리(10절 대기하며 돌리기): 취침 시각(`BEDTIME`) 전 기사는 비교용으로 등록, 뒤 기사는 묶기 → 이전 소식 확인 → 종목별 10개 묶음 → 본문 받기 → 프롬프트 입력. 수집과 함께 켜 둔다 | `OPENAI_API_KEY`(임베딩), 알림 키 | `WATCH_RUN`(골라 돌리기는 `PROCESS_RUN`) | OpenAI 임베딩(하룻밤 $0.01~0.03), 원문 주소 복원·원문 페이지(무료) |

- `db_test`는 처리 단계가 아니라 수집 DB를 조회하는 노트북이다.
- 비용과 시간은 2026년 11월 한 번 전체를 만들 때 기준이다.

## 관리 규칙

- 결정은 그래프 DB·검색어 만들기는 `graph_db/graphDB.md`, 뉴스 검색·밤사이 처리는 `news_search/news_search.md`에 적는다(정한 것은 본문, 정할 것은 "결정할 것").
- 외부 요청(웹, LLM API)을 보내는 실행은 예상 건수·시간·비용을 먼저 알리고 승인받는다.
- 새 패키지를 쓰면 `requirements.txt`, 새 키를 쓰면 `.env.example`에 더한다.
