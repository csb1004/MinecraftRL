# 공통 목표 조건부 S2 실행/학습

S2는 기존 S1/P4와 분리된 opt-in 경로다. 명세는
[S2 실행 전 계약](specs/2026-11-01-general-rm-s2.md)을 따른다.
`src/minecraftrl/semantic_goal.py`는 목표의 대상·결과·수량·선행 조건·진행을 보존하고,
`conditioned_head.py`는 frozen STEVE-1 특징에 작은 조건부 계층을 붙인다.

## 네 조건

| 옵션 | 학습 구조 | 추가 RM context |
|---|---|---|
| linear-off | 선형 actor/value | 제거 |
| linear-on | 선형 actor/value | 사용 |
| nonlinear-off | residual actor / 비선형 value | 제거 |
| nonlinear-on | residual actor / 비선형 value | 사용 |

모든 조건에서 기존 목표 임베딩, 상위 목표 큐, 실제 인벤토리 성공 기준은 유지한다.
off는 목표 문장 자체를 없애는 조건이 아니다. 원본 모델 파일은 수정하지 않는다.
초기 추가 계층 출력과 value는 0이며, variant의 실제 수집 확률 재현을 trainer가 검증한다.

## 실행과 데이터

기존 `episode --policy steve1 --condition wood --rm-conditioning goal-queue`에
`--rm-goal-bank <bank> --rm-program <program>`을 지정한 뒤 다음을 사용한다.

- `--s2-variant nonlinear-on`: S2 경로 활성화. S1 옵션과 동시에 쓸 수 없다.
- `--s2-checkpoint <project/results/.../checkpoint.pt>`: 같은 variant의 후보만 허용.
- `--s2-split train|development|evaluation`: 데이터 용도. 평가 자료는 trainer에서 거부한다.
- `--s2-action-budget 1200`: 최대 실제 행동 수. wall 제어 제한은 별도 최대 180초.
- `--stop-file <project/results/STOP>`: 기존 중단 계약. 파일이 있으면 새 실행을 시작하지 않는다.

Windows에서는 PowerShell로 WSL Ubuntu-24.04의 기존 `.runtime/steve1` 환경을 호출한다.
공식적인 전체 배치 진입점은 `scripts/Invoke-RMGeneralS2.ps1`이다. 완료/준비된 배치를 자동 재실행하지 않는다.
명세에 고정한 새 12개 학습 실행, 네 후보 학습, 25개 개발 비교, 실제 STOP 검사를 순차 실행한다.
원본/명령/버전/해시/프로그램/관측/후보는 `results/rm-general-s2-20261101`에 남는다.

## 학습은 별도 프로세스

`scripts/train_s2_conditions.py --root <project> --runs <new-train-run> ... --output <fresh-results-path>
--stop-file <project/results/STOP>`을 격리된 WSL Python에서 실행한다.
Minecraft나 외부 API를 시작하지 않는다. 게임 중 가중치를 업데이트하지 않는다.
기존 후보의 rollout을 초기 정책 자료라고 위장하거나, 개발 자료를 학습에 사용하면 거부한다.
현재 2×2 trainer는 동일 초기 정책의 첫 업데이트를 위한 것이며, 후보의 반복 PPO 수집을 암묵적으로 허용하지 않는다.
각 후보의 다음 반복 학습에는 별도 명세와 해당 후보로 수집한 on-policy 데이터가 필요하다.

## 관측과 종료

관측기와 게임 JAR는 고정한다. 서버 관측을 RM/학습 context에 쓰므로 영상 전용 결과가 아니다.
wood의 세부 단계와 dirt의 큰 조작 단계를 구분하고, 미지원 known/reachable은 unknown mask를 전달한다.
모니터는 정확히 같은 행동 반복, 인벤토리 진전, 기존 상호작용 진행 근거를 기록한다.
반복 자체는 실패가 아니다. 현재 600tick 정체/300tick 미회수 한도는 개발 설정이며 교정된 탐지기 성능으로 보고하지 않는다.

- 최종 목표 완료/환경 종료/정체 과업 실패: terminal, bootstrap 0.
- 행동 예산/제어 시간 제한: truncation, 마지막 실제 state에서 bootstrap.
- 외부 STOP/기술 오류: 학습 제외. 새 행동 금지, 입력 해제, 소유 자식 프로세스 정리.
- `learning-request.json`: LLM에 전달 가능한 실패 근거. `executed=false`, API 0.
  자동 LLM 호출/지식 획득/온라인 재학습이 실행됐다는 뜻은 아니다.

## 결과 해석

`scripts/analyze_rm_general_s2.py <batch>`는 서버 인벤토리와 원본 이벤트를 독립 감사한다.
목표별 획득량·실제 행동·완료 순서·틱·지연·GPU·초기 화면 차이를 기록한다.
완료된 [실제 비교 보고서](rm-general-s2-results-2026-11-01.md)에 성능과 한계를 기록했다.
원래 마지막 STOP 사전검사의 기술 실패와 별도 R1 실제 통과를 구분한다. 완료된 배치를 재실행하지 않는다.
모델 로드/단위 테스트/가중치 변화만으로 성능 개선을 주장하지 않는다.
현재 executor가 없는 도구/제작/돌 목표는 계속 미지원이다. 후보는 자동 승격하지 않는다.
