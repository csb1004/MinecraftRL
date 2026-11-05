# LOGO 원리를 적용한 RM 판별 함수와 행동 문구 — 2026-10-31

사용자 요청: LOGO로 RM 성공 판별 함수를 작성하고, 정책 문구를 STEVE-1 논문과 유사한
구체적 행동 표현으로 바꾼다. 기존 P1/P2 결과와 원문 연구 명세는 소급 변경하지 않는다.

## 적용 범위와 근거

[LOGO: A framework for language-based goal decomposition and skill recomposition in reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0925231226016899)
(Yin et al., Neurocomputing 698, 134291, DOI 10.1016/j.neucom.2026.134291)의 공개 초록/서론은
LLM으로 하위 목표와 실행 가능한 검증 함수를 생성·검토하고, 이를 목표 조건부 정책의 보상과
스킬 재조합에 활용하는 구조를 제시한다. 논문의 평가 환경은 MiniGrid/MPE다.

이번 구현은 **Minecraft용 LOGO-inspired adapter**다. 출판사 본문 직접 열람은 403이어서
세부 알고리즘 전체나 공식 구현을 재현했다고 주장하지 않는다. 현재 Codex가 제안식을 작성했고,
런타임 소유 검사 사례와 기존 실제 로그로 검증했다. 외부 LLM API 호출, 독립 다중 LLM 심사,
자동 반복 질의, 스킬 라이브러리 재조합, LOGO 방식의 새 정책 학습은 실행하지 않았다.

## 생성 → 검토 → 실행

| 단계 | 소유자와 구현 | 실패 처리 |
|---|---|---|
| 관측·목표 계약 | 호출자가 `logo_wood.context()`로 고정 관측 사실, 필요한 판별 함수, 최종 성공 조건 제공 | 없는 관측을 LLM이 추가할 수 없음 |
| LLM 작성 요청 | `logo_validators.request()`가 관측 의미·한계와 JSON Schema 생성 | 근거가 부족하면 `unsupported` |
| 제안 | `configs/logo-wood-validators-v1.json`; 현재 Codex가 작성한 식 | context hash 불일치, 미등록 사실/연산, 자유 실행 코드 거부 |
| 컴파일 | `ValidatorSet.functions`의 callable로 변환 | 제한된 `all/any/eq/ge/le`, 현재 값 또는 시작 대비 변화량만 허용 |
| 검토·수정 피드백 | 호출자가 소유한 양성·음성·결측 사례, 함수별 양성/음성 coverage 검사 | `review.json`에 반례를 남기고 실행 거부; 수정본을 다시 제출 |
| 실제 실행 연결 | `--rm-conditioning logo-staged`; `LogoEnvironment`가 판별값으로 RM 단계·희소 보상·종료 결정 | 모르는 관측/오래된 tick이면 종료 경로로 이행 |

최종 성공 조건은 호출자가 소유한다. LLM이 “나무를 공격했다”처럼 더 쉬운 조건으로 바꿔
스스로 성공을 정의할 수 없다. 하위 함수의 의미 검토는 필드 형식 검사만으로 끝나지 않는다.
현재 계약 사례 통과도 모든 상황의 논리적 정당성을 증명하지는 않는다.

생성 Python을 `exec/eval`로 실행하지 않는다. 이 제한은 LLM이 새 센서나 행동 API를 만들어
관측/행동 경계를 넓히는 것을 막는다. 필요한 새 관측이 있다면 별도 연구 조건으로 검토한다.

## 나무 수집에 적용한 판정

| 함수 | 관측과 판정 | 뜻하지 않는 것 |
|---|---|---|
| `target_known` | `wood_v3_kind == 1` | 지금 화면에 보인다는 판정 아님. 기존 서버 추적기의 기억된 원목 위치 |
| `target_reachable` | 알려진 원목 + `wood_v3_reachable == 1` + 게임 화면 + 생존 | 경로 계획 완료/앞으로도 접근 가능하다는 보장 아님 |
| `log_broken` | 초기값 대비 `server_log_harvests >= 1` 증가 | 공격 버튼/채굴 접수 횟수 아님. 특정 현재 목표와의 인과 연결까지 보장하지 않음 |
| `item_acquired` | 초기값 대비 서버 인벤토리의 6종 원목 합계가 1 이상 증가 | 화면 예측·클라이언트 인벤토리·드롭 생성 아님 |

`log_broken`은 이 에피소드에서 실제 원목을 하나 이상 캔 이력이다. 인벤토리 획득과 별개다.
이미 있던 원목/과거 채굴 카운터는 새 성과로 세지 않는다. 원목 드롭을 먼저 주웠다면 채굴
단계를 강제로 거치지 않고 최종 완료할 수 있다. 원목 이외 아이템/수량을 쓰는 컴파일러 검사는
있지만 그 과업의 실제 정책 능력을 주장하지 않는다.

최종 성공이 최우선이다. 성공 증거가 있으면 하위 목표 힌트 일부가 없어도 즉시 완료한다.
그 외 결측/비유한 값/이전 tick은 성공으로 바꾸거나 탐색으로 조용히 대체하지 않는다.
MineRL의 `1.0/0.0` 숫자형 플래그를 허용하고 bool·문자열·비유한 값은 거부한다.

스테이지 선택은 완료 → 도달 가능 → 알려진 대상 → 실제 채굴 이력과 잠시 유지되는
대상 소실 힌트가 함께 있을 때 줍기 시도 → 탐색 순서다. 소실 힌트는 획득 판정이 아니다.
비종료 단계는 기존처럼 3회 연속 관측으로 안정화한다. 종료는 즉시 반영한다.
대상을 잃고 탐색으로 돌아올 수 있으며, P2의 되돌릴 수 없는 gather latch는 이 새 모드에 없다.
**아직 알려져 있지만 접근 불가능한 대상에서의 정체/우회는 별도 미해결 과제**다.

## 정책 문구 v2

기존 `goal_contract.py`의 v1 및 P1/P2 문구·임베딩은 그대로 둔다. 새 `goal_actions.py`는
같은 계약을 검증한 뒤 행동·대상 절을 조합한다. 수량, 기대 관측, predicate, 현재/상위 목표의
원문은 계약 메타데이터에 유지한다. 정책 문구에는 구체적인 행동·대상·목적과 적용 제약을 남긴다.
호출자는 purpose 절이 현재/상위 목표를 제대로 표현하는지 확인해야 한다. 문자열 검사만으로
자연어 의미 동등성을 증명하지 않는다.

| 실행 조건 | 실제 새 MineCLIP 입력 | 출처와 경계 |
|---|---|---|
| 탐색 | `go explore to find a tree, chop down the tree, gather wood` | 공식 행동 표현에서 파생한 **새 후보**. 논문에 그대로 있는 문장 아님 |
| 접근·채굴·줍기 | `chop down the tree, gather wood, pick up wood` | 공식 `paper_prompts.py` wood 문구의 앞 세 절. 전체 공식 문구와 동일한 비교 조건은 아님 |

공식 문구 근거는 `external/steve1/steve1/run_agent/paper_prompts.py`다. 새 문구에는
`Purpose:`/`Expected result:` 같은 필드 설명을 섞지 않는다. 일반 규칙은 **모든 조건에 행동의
대상과 목적 절을 보존**하는 것이다. 컴파일러에 나무 전용 분기는 없으며 광물/목적지 등으로
바꿔 검사한다. 승인된 action/target 참조가 아닌 자유 문구로 덮어쓸 수 없다.

탐색은 `seek`, 다른 세 상태는 하나의 `root` 벡터를 공유한다. RM 상태가 바뀔 때마다
임베딩을 갈아 끼우지 않는다. 목표 변경 시 recurrent memory를 보존하며 실제 dispatch된
행동만 메모리와 로그에 commit한다. STOP/완료 이후 새로운 정책 추론·행동은 금지한다.

## 검증 결과와 한계

자료: `results/logo-validators-20261031` 및 그 안의 `goal-bank`.

- caller 소유 검사 19개에서 제안식 통과. 잘못된 채굴 식을 넣는 별도 음성 대조 테스트에서 거부됨.
- WSL 관련 테스트 66개 통과: 생성식/문구, wrapper 보상·종료 연결, 기존 조건 회귀,
  STOP 경합과 종료 supervisor 검사. 합성 환경 검사이며 새 모드의 실게임 성공은 아니다.
- 기존 P2 15회·9,327행동 로그를 재판정해 기존 최종 결과와 모두 일치: 성공 10/15, 실패 5/15.
  이것은 **기존 자료 재검사**이며 새 정책 실행 결과나 독립 최종 평가가 아니다.
- `81031001-purpose-fixed`는 557번째 행동에서 실제 원목 채굴을 확인했으나 이후 539개 관측까지
  획득이 없었다. `81031001-purpose-switch`와 `81031007-purpose-fixed`는 채굴 접수 이력과
  별개로 실제 원목 채굴 카운터 증가가 없었다. 따라서 실패를 모두 “못 찾음”으로 묶지 않는다.
- 새 문구의 실제 MineCLIP 토큰은 root 13 / seek 16개(BOS/EOS 포함), 잘림 없음.
  공식 MineCLIP·prior, seed 31031로 (1,512) float32 벡터를 생성하고 해시/로드를 확인했다.
  생성 중 PyTorch 최대 할당 약 703.4 MiB. 게임 정책 추론 메모리 측정이 아니다.
- 처음 재검사에서 MineRL 숫자형 플래그가 float로 저장된 것을 확인해 검사기를 수정했다.
  초기 준비 provenance와 후속 코드 해시는 구분한다. 최종 검증 기록을 함께 보존한다.
- 기존 root 성공 판정도 서버 인벤토리에 기반했다. 이번 결과는 과거 성공 오판이 실패의
  원인이었다는 증거가 아니다. 생성·검토 경로와 하위 단계의 실패 구분을 추가한 것이다.
- 새 관측/JAR 수정, 게임 실행, RL 가중치 업데이트, 유료 API, GitHub 게시 모두 0. STOP 유지.
  새 조건의 실게임 STOP/입력 해제/PID 정리와 수행 성능은 아직 재측정하지 않았다.

최종 테스트 수/소스·은행 해시: `final-verification.json`. 초기 replay의 시작 상태에서 이미
참인 predicate 최초 시각 표기를 1이 아닌 0으로 바로잡은 `replay-r1.json`이 최종 재검사본이다.
기존 최초 replay와 준비 파일은 보존한다. 실제 성공/실패 수에는 변동이 없다.

## 실행 방법

프로젝트 루트에서 기존 Python으로 작성·검토만 실행할 수 있다. 출력은 새 경로여야 한다.

```powershell
$env:PYTHONPATH = 'src'
python scripts/prepare_logo_rm.py --root D:/DATA/Desktop/MinecraftRL `
  --proposal configs/logo-wood-validators-v1.json --output results/logo-review-next
```

`request.json → proposal.json → review.json`을 보존한다. 거부되면 반례를 작성자/LLM에
돌려주고 새 revision/출력 경로로 다시 검토한다. 요청 생성 자체는 API 호출을 하지 않는다.
`--encode`는 기존 WSL STEVE-1 환경에서만 사용한다. STOP을 해제하지 않고 게임도 시작하지 않는다.
실행기는 `--rm-conditioning logo-staged --rm-goal-bank <검토된 goal-bank>`를 명시해야 사용한다.
실게임 후속 비교는 prompt 변경과 validator/전환 변경의 효과를 혼동하지 않도록 분리해서 고정한다.

4단계 연구 중 이번 범위는 S1 판정·조건 연결과 S2 작성 인터페이스 일부다. 완성된 자동
LLM→RM→RL 학습 시스템, SQL 지식 학습, 일반화된 복구 정책으로 보고하지 않는다.
