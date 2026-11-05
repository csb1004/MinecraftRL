# P2 결과: 대상·목적을 보존하는 RM 목표 조건

2026-10-31, Windows 로컬 PC / WSL Ubuntu-24.04에서 실제 MineRL 게임으로 실행했다.
**조건 전달과 전환은 정상 동작했지만, 새 규칙으로 성공률이 높아졌다는 결과는 얻지 못했다.**
공식 문구 4/5, 계약 수집 문구 고정 3/5, 목적 있는 탐색→수집 3/5 성공이다.
이는 개발 파일럿이며 RL 학습·새 스킬 습득·일반화 성능 검증이 아니다.

## 고정한 비교

[사전 프로토콜](specs/2026-10-31-purpose-conditioning-p2.md)과
[공통 작성 계약](specs/2026-10-31-goal-condition-authoring-rules.md)을 사용했다.

| 조건 | 실제 목표 문구 | 선택 방식 |
|---|---|---|
| A `official` | `chop down the tree, gather wood, pick up wood, chop it down, break tree` | 공식 전체 문구 고정 |
| B `purpose-fixed` | `collect wood. Purpose: collect at least one wood log. Expected result: wood is added to the inventory.` | 계약 문구 고정 |
| C `purpose-switch` | `find a tree. Purpose: collect at least one wood log. Expected result: a tree is located.` → B 문구 | 대상 확인 후 최대 한 번 전환 |

B/C는 객체 이름을 특별 취급하지 않는 공통 compiler로 생성했다. 이번 실행 어댑터와
등록 predicate만 기존 wood 과업에 한정했다. 미래 LLM도 같은 schema를 사용할 수 있지만
이번에는 사람이 proposal을 작성했으며 API 호출은 0이다.

공식 STEVE-1 가중치, guidance 6, prior RNG 31031, 정책 RNG=월드 시드를 고정했다.
MineCLIP 실제 tokenizer로 A/탐색/B를 검사한 결과 각각 20/23/24token으로 잘림이 없었다.
임베딩은 게임과 분리해 생성했으며 RNN은 조건 전환 시 유지하고 실제 dispatch에만 commit했다.
compiler 후보의 승인/능력/tokenizer 플래그는 원본 그대로 보존하고, 별도 P2 bank의
토큰 영수증과 잠근 context/predicate/모델 검사를 통해 이 실험에 한정해 실행했다.
후보의 구조 검증을 정책 능력 검증으로 승격하지 않았다.

성공은 실제 서버 인벤토리 원목 증가 ≥1이다. 보상은 최초 성공 +1이며 shaping과 optimizer update는 0이다.
180초는 실제 제어 wall-clock 상한이며, 모델/게임 시작 시간은 제외한다.
한 행동은 서버 1tick이다. 별도 STOP 진단은 성능 표본에서 제외했다.

## 실제 결과

각 셀은 **획득 원목 / 제어 초 / 정책 행동 수**이다. 초기 인벤토리 원목은 전부 0이다.

| 개발 시드 | A 공식 문구 | B 수집 고정 | C 탐색→수집 |
|---|---:|---:|---:|
| 81031001 | 1 / 46.0 / 281 | 0 / 180.1 / 1095 | 0 / 180.1 / 1253 |
| 81031003 | 1 / 18.9 / 95 | 1 / 77.0 / 459 | 1 / 41.4 / 237 |
| 81031007 | 1 / 42.4 / 246 | 0 / 170.7 / 1102 — 낙사 | 0 / 180.1 / 1368 |
| 81031009 | 0 / 180.1 / 1337 | 2 / 69.0 / 444 | 1 / 72.9 / 470 |
| 81031013 | 1 / 50.4 / 346 | 1 / 48.5 / 312 | 1 / 43.3 / 282 |
| 성공 | **4/5** | **3/5** | **3/5** |

15회 모두 감사상 유효하며 기술 오류로 제외하거나 재실행한 성능 표본은 0이다.
총 9,327행동, 제어 1,400.71초다. 실제 누적 서버 tick은 명목상 게임 시간 466.35초에 해당한다.
B의 시드 81031009에서는 성공한 한 transition에서 원목 2개가 확인됐다.
성공 이후 정책 행동을 더 실행해 얻은 수량이 아니다.

### 실제 전환과 비교의 한계

C의 첫 gather 행동은 시드 순서대로 **149 / 32 / 46 / 157 / 1**번째였다.
마지막 시드는 초기 관측부터 대상이 잡혀 **seek 문구를 사용하지 않았다**.
B와 C가 같은 gather 조건으로 시작한 이 시드의 시간 차이는 탐색 효과가 아니다.
특히 해당 B/C의 초기 PNG도 달랐다.

실제 탐색 문구를 사용한 네 시드에서는 B/C 모두 2/4 성공했다.
성공한 두 경우 중 시드 81031003은 C가 빨랐고, 81031009는 C가 조금 느렸다.
단계 전환이 항상 이득이라는 근거는 없다.

모든 시드의 초기 위치/시점 자세는 세 조건에서 같았다. 초기 PNG는 시드 81031001/81031003만
세 조건 모두 동일했고 나머지 세 시드에서는 달랐다. 같은 시드가 동적 환경과 렌더링의
완전한 일치를 보장하지 않는다. 기존 P1과 같은 개발 시드이며 최종 평가나 독립적인 재현 실험이 아니다.
A↔B는 여러 문구 요소를 함께 바꿨으므로 특정 단어나 문장 길이의 인과 효과를 분리하지 못한다.

## 관측한 실패와 해석

| 실패 | 실제 관측 | 해석 범위 |
|---|---|---|
| 81031001 B | mine 라벨 692tick, 서버가 받아들인 채굴 287tick, collect 센서 6tick, 최종 원목 0 | 접근 이후 채굴·수집 연결 실패 |
| 81031001 C | mine 라벨 936tick, 받아들인 채굴 554tick, 같은 대상 채굴의 최장 연속 구간 58tick, collect 0 | 채굴 시도가 실제 획득으로 이어지지 않음 |
| 81031007 B | 받아들인 채굴 423tick, 같은 대상 최장 연속 18tick, 이후 y=40으로 추락, 서버 health=0/alive=0 | 채굴 지속성 문제와 낙사 |
| 81031007 C | approach 1323tick, 채굴 0, y=63→39, 마지막 화면은 물이 있는 돌 지형 | 이동·접근 실패 |
| 81031009 A | search 1337tick, 채굴 0, xz 경로 약 164.2블록, 끝까지 센서 대상 없음 | 이동은 했으나 과업에 필요한 대상을 확보하지 못함 |

mine은 기존 센서의 준비/진행 라벨이며 블록 파괴 증명이 아니다. 같은 대상 채굴 연속 구간도
서버 텔레메트리에서 계산한 진단값으로, 실제 블록 파괴 진행률 자체와 동일하지 않다.
wood_v3는 이전에 포착한 ray 기반 대상을 기억한다. 따라서 approach가 계속되는 동안
현재 화면에 나무가 보인다거나 접근 가능한 경로가 있다는 뜻은 아니다.

첫 실패 화면에서 빈 인벤토리를 연 모습이 보여 **사후 탐색 분석**을 추가했다.
인벤토리 화면이 열린 transition의 조건별 비율은 A 0.43%, B 6.36%, C 11.41%였다.
각 조건의 실제 행동 수로 가중한 기술 통계이며 사전에 정한 인과 검정이 아니다.
성공한 C/81031003에서도 인벤토리 화면이 28/237tick 나타나므로 UI 열기를 실패의
유일한 원인으로 볼 수 없다. `Expected result: ... inventory` 표현의 영향도 아직 가설이다.
추가 진단은 실행 코드·문구·시드·재시도 여부를 바꾸지 않았다.

## 지연·자원

환경은 GTX 1060 3GB, WSL Python 3.10.14, torch 2.0.1/CUDA 11.7,
numpy 1.23.5, gym 0.19.0, MineRL 1.0.0/Minecraft 1.16.5, Zulu Java 8u472이다.
정책은 WSL에서 실행했고 결과 감사는 Windows Python 3.11.9/numpy 1.26.4로 수행했다.

| 측정 | 15개 실행에서의 범위 |
|---|---:|
| 실행별 추론 지연 중앙값 | 46.12–50.97ms |
| 실행별 추론 지연 p95 | 54.63–128.72ms |
| 실행별 전체 제어 주기 중앙값 | 124.60–161.62ms |
| 실행별 전체 제어 주기 p95 | 157.24–250.52ms |
| PyTorch 최대 할당 | 모든 실행 1915.00MiB |
| 전체 GPU 사용량의 표본 최대치 | 2754–2863MiB |

위 지연 범위는 실행별 통계의 최소/최대이지 전체 행동을 합친 백분위가 아니다.
GPU 표본에는 다른 프로세스도 포함되며 짧은 최대치를 놓칠 수 있다. OOM은 관측되지 않았다.
실행별 정확한 값은 `run-metrics.csv`와 `audit.json`에 보존했다.
게임·모델 시작 비용이 커서 총 작업 시간은 제어 시간보다 길었다. 이를 성공 시간에 섞지 않았다.

## 종료·STOP 검증

- 성능 15회 모두 worker exit 0, STOP acknowledgment 이후 새 dispatch 0, 잔존 소유 PID 0.
- 살아 있는 환경에서 끝난 14회는 neutral 명령과 held key 0을 실제 서버에서 확인했다.
- 낙사 1회는 MineRL terminal 상태가 추가 step을 거부하므로 `environment_done_no_neutral_step`이다.
  **중립 입력/held-key 값은 미측정**이며, 추가 정책 행동 없음과 게임 프로세스 종료를 확인했다.
- 최초 감사 코드는 이 terminal 예외에서 없는 입력 해제 파일을 읽으려다 중단됐다.
  감사기만 수정하고 3개 회귀 테스트를 통과한 뒤 원본 15회를 다시 감사했다.
  낙사를 실패 표본에서 빼거나 게임을 재실행하지 않았다. 이전 감사 코드와 오류 로그도 보존했다.
- 별도 시드 81031999: 실제 10 transition 뒤 STOP 요청. 요청 가시화 이후 dispatch 0,
  ack 이후 0, worker/supervisor exit 2(정상 취소), helper exit 0, neutral 검증 true,
  held key 0, 잔존 소유 PID 0. ack 지연 22.01ms, 요청 쓰기 시작 기준 상한 67.71ms.
- `results/STOP`을 남겼으며 게임과 정책 실행은 종료했다.

## 구현·근거 파일

- 공통 compiler: `src/minecraftrl/goal_contract.py`.
- 잠근 실험 context/조건 선택/검증: `src/minecraftrl/rm_purpose_conditioning.py`.
- 기존 P1 실행기의 목표 선택 hook을 확장했으며 P1 문구 상수와 기존 결과는 보존했다.
- 준비/실행: `scripts/prepare_rm_purpose_bank.py`, `scripts/Invoke-RMPurposePilot.ps1`.
- 감사: `scripts/analyze_rm_purpose.py`, STOP 진단: `scripts/probe_rm_conditioning_stop.py`.
- 실행 전 WSL 관련 테스트 51개, 감사 수정 테스트 3개 통과.

증거 루트: `results/rm-purpose-conditioning-20261031`.
`pilot-plan.json`, `goal-bank/manifest.json`, `source-at-game`, `work.json`, 실행별 command/summary/events/PNG,
`audit.json`, `run-metrics.csv`, `stop-probe-audit.json`, `reproducibility-check.json`을 보존했다.
감사 원본/수정본은 `analysis-source.py`, `analysis-source-r1.py`와 각 method JSON에 남겼다.
실행 중 정책 Python 소스 114개와 사전 프로토콜의 해시 변경은 없었다.
연구 명세 원문 SHA256은 `4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad`로 유지됐다.
공통 실행 JAR SHA256은 `dac9a1c9cb6b0a7d848559cd7f55d88f7d5bd89db4bad4d5ed35cf8fe9a530a2`이다.
Git HEAD만으로는 수정된 작업 트리를 재현할 수 없으므로 실행 당시 source snapshot을 기준으로 삼는다.
대용량 결과/임베딩은 Git에서 제외돼 있다. 이번 결과는 로컬 기록이며 GitHub 추가 게시나 유료 API/클라우드 사용은 없었다.

## 후속 판단

대상·목적·관측 가능한 완료 조건을 보존하는 공통 계약은 유지할 가치가 있다.
그러나 구조적으로 올바른 문장이 사전학습 정책에 효과적인 목표 표현이라는 보장은 없다.
현재 문구와 전환 방식을 성능이 개선된 기본 설정으로 승격하지 않는다.

다음 비교는 계약 메타데이터를 보존하면서 정책에 전달하는 표현만 공식 문구와 가까운
짧은 행동 문장으로 바꿔, **문구 렌더링과 RM 전환 효과를 분리**하는 것이 적절하다.
예를 들어 탐색 지시는 찾을 대상과 그 목적을 계속 포함해야 한다. 이 제안은 아직 실행하지 않았다.
이후 동일한 초기 상태/추가 RNG 반복과 별도 평가 시드로 확인하고,
이동·지속 채굴·수집 실패가 남으면 별도의 RL 학습 실험으로 다뤄야 한다.
이번 작은 개발 진단으로 RL의 학습 가능성, 새로운 RM 일반화, 인간다움이나 문화 이해를 결론 내리지 않는다.
