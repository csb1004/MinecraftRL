# P3: LOGO 판별 함수와 구체적 정책 문구 — 실행 전 고정

2026-10-31 사용자 지시 “실험해봐”에 따른 새 실제 게임 비교다.
기존 P1/P2를 재실행하는 일회용 배치가 아니며 원본을 보존한다.
근거와 구현은 [LOGO adapter 명세](2026-10-31-logo-validator-adapter.md)를 따른다.

## 조건과 해석

| 조건 | 모드 | 목표 문구/전환 |
|---|---|---|
| A | `purpose-fixed` | 기존 `collect wood. Purpose: collect at least one wood log. Expected result: wood is added to the inventory.` 고정 |
| B | `logo-fixed` | 새 `chop down the tree, gather wood, pick up wood` 고정 |
| C | `logo-staged` | 탐색에서는 `go explore to find a tree, chop down the tree, gather wood`; 그 외 비종료 단계에서는 B 문구 |

A는 원래 P2 어댑터·은행을 사용하고, B/C는 검토된 LOGO 판별 함수·동일 은행을 사용한다.
세 조건의 최종 성공 의미는 동일한 서버 인벤토리 원목 증가 ≥1이다. 원래 판정과 생성 판정은
P2 9,327행동 replay에서 일치했다. A/B는 고정 문구 비교로 해석하되 판정 구현·부하 차이도 공개한다.
B/C는 같은 새 판정/보상/가중치/문구 은행에서 목표 조건 전환 사용 여부를 비교한다.
C의 각 predicate와 전환 규칙 개별 효과를 분리하거나 특정 단어의 인과 효과를 주장하지 않는다.
공식 전체 wood 문구 4/5였던 P2 결과는 역사적 참고이며 이번 동시 대조군이 아니다.

## 고정 변수

- 개발 시드 81031001, 81031003, 81031007, 81031009, 81031013. 총 15회, 최종 평가 아님.
- 시드별 순서 CAB / ABC / BCA / CAB / ABC. 첫 C가 실제 연결 진단도 겸하며 기술 오류면 배치 중단.
- 각 조건은 해당 시드로 새 월드 생성, 고정 60 neutral tick 준비. 초기 pose/PNG 비교를 보고한다.
  시드가 같아도 동적 화면까지 같다고 가정하지 않는다.
- 제어 wall-clock 최대 180초, startup 최대 600초, 행동당 서버 1 tick. 종료 정리는 제어 시간과 분리.
- 공식 STEVE-1 2x 가중치 동결, guidance 6, prior seed 31031, 정책 sampling seed=world seed.
- B/C는 이미 준비한 13/16 token 벡터를 그대로 복사·검증. A는 P2 검증 은행을 복사해 사용.
- 이미지+기존 목표 임베딩+RNN 입력 유지. 새로운 센서/JAR/actor 입력 없음. privileged server RM 조건.
- 비종료 단계 3연속 확인, 최종 완료는 즉시. 조건 전환 시 RNN 유지, 실제 dispatch만 memory commit.
- 최초 원목 획득만 +1, shaping/시간 페널티 없음. optimizer update·외부 LLM API·클라우드 모두 0.
- 시작 전에 소스/프로토콜/은행/실험 설정·관측기·공식 가중치 receipt 해시를 고정하고 매 실행 전 검사.
  모델 실물 해시는 기존 실행 검증에서도 확인한다. 수행 중 정책 코드를 바꾸지 않는다.

## 판정·분석·중단

성공/획득량은 실제 서버 inventory로 독립 재검사한다. 생성 predicate, raw/확인 stage, prompt/vector,
행동·reward·tick·terminal의 연결도 원본 이벤트로 검사한다. STOP/완료/terminal 후 새 dispatch 금지.
입력 해제/held key 0, 소유 PID 종료, 추론 중앙값/p95, 전체 제어 주기, GPU 메모리를 기록한다.
terminal 게임이 중립 step을 거부하면 중립/held key는 미측정으로 남기고 실제 terminal·PID 종료를 확인한다.

실패는 미발견 / 발견했으나 도달 못함 / 도달했으나 실제 채굴 없음 / 채굴했으나 획득 없음으로
증거를 구분한다. 이것은 관측된 실패 단계이며 인과적 진단이 전부 증명됐다는 뜻이 아니다.
낙사 등 종료 원인과 inventory GUI 체류, 채굴 지속성도 따로 기록한다.

기술 오류·사용자 중단은 성능 표본에서 제외하고 미실행은 분모에 넣지 않는다. 오류 시 남은 배치를
중단하고 원본을 보존한다. 성능 실패 재시도 없음. 기술적 재시도는 원인/변경/명시적 amendment 기록 후만.
15회가 끝나면 별도 seed 81031999, C 모드에서 실제 transition 10회 후 STOP을 요청하고 실제 입력 해제와
자식 프로세스 종료를 검증한다. 이 진단은 성능 표본에서 제외하며 끝난 뒤 STOP을 유지한다.

이전 STOP은 이번 사용자 실행 지시를 근거로 원문·해시와 재가동 기록을 보존한 뒤 한 번만 해제한다.
이후 새 STOP을 배치가 지우지 않는다. 기술 오류 시에도 STOP을 남긴다.
실행/원본: `results/rm-logo-conditioning-20261031`. 작은 개발 비교로 RL 학습·일반화 개선을 주장하지 않는다.
