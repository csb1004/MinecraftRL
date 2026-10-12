# 정책 선행 검증 프로토콜

조합 후속 실험은 [별도 조건](craft-recovery-plan.md)으로 구분한다. 원래 무정지 정책
기준선, 인벤토리+통계 witness 정지 조건, GUI/서버 확정 회수 보조 조건을 합산하지 않는다.
조합 수정 조건은 첫 물품 관측 뒤 새 정책 행동 0회, 서버 인벤토리·커서 확정, 저장 NBT
일치를 각각 검증한다. 정리 입력은 정책 행동과 별도로 기록한다. 초기화 중 취소와
미실행은 정책 과업 성공/실패 표본에서 제외한다. [관측 경계의 원인 분석](craft-rm-root-cause.md).

## 평가 행렬

| 정책 | 조건 | 시드 | 제어 제한 |
|---|---|---|---|
| VPT foundation 1x | 자연어 없이 자발적 행동 | 동일 5개 | 회당 180초 |
| STEVE-1 | `collect wood` | 동일 5개 | 회당 180초 |
| STEVE-1 | `collect dirt` | 동일 5개 | 회당 180초 |

world seed와 random/numpy/torch RNG를 매 episode 설정한다. env.seed는 reset보다 먼저 호출한다. MineRL은 seed를 reset 후 지울 수 있으므로 에피소드마다 다시 설정한다. 프레임/인벤토리/위치 관측을 비교해 동일 seed가 실제 같은 초기 월드를 만드는지도 검증한다. 단순 seed 설정은 완전 결정론의 증거가 아니다.

VPT에선 이동·시점/채집 여부와 아이템별 최대 순증가를 관찰한다. STEVE-1은 인벤토리의 목표 아이템 최대 순증가 1개 이상을 최소 과업 기준으로 한다. wood는 1.16.5 오버월드 6종 log로 한정한다. 나무 판자나 임의 블록을 나무 획득으로 간주하지 않는다.

## 단계별 게이트

1. 대상 PC 경로/환경, WSL NVIDIA 전달, glxinfo/렌더링 확인.
2. 고정 소스 설치와 전체 버전/빌드 patch 기록.
3. 공식 가중치 다운로드 receipt, SHA256 및 MineCLIP 공식 MD5 검증.
4. 모델 로딩과 첫 act; 메모리 초과는 traceback 보존.
5. HumanSurvival reset/step/인벤토리 검증. 여기까지 실게임 진행 전에는 rollout 성공이라고 부르지 않는다.
6. 조건당 5개 시드 실행. 시스템 차단이면 나머지는 unrun으로 남긴다.
7. 별도 실제 중단 실험: 추론 중, step 중, 초기화 중에 STOP 요청. 로그에서 acknowledgment 이후 새 policy dispatch가 없는지 확인하고 neutral release를 별도 확인한다.
8. 환경/패키지/모델 receipt/로그를 보존하고 결과를 해석한다.

## 중단의 정확한 의미

Python unit test는 추론 도중 취소된 행동을 전송 전에 폐기하고, stop 이후 dispatch가 막히는지를 확인한다. 실제 게임 중단 보증을 대신하지 않는다.

이미 전송된 step은 되돌릴 수 없다. `stop_acknowledged` 이후 새 정책 행동을 보내지 않으며 중립 입력 해제는 controller=cleanup으로 한 번 시도한다. 환경 done이 관측된 경우 MineRL은 추가 step을 거부하므로 `environment_done_no_neutral_step`을 기록하고 환경을 닫는다. 실행 중인 환경이 응답하지 않아 해제에 실패하면 release 오류 또는 강제 종료 미검증으로 기록한다.

실제 단계별 시험은 PowerShell에서 `scripts/Invoke-StopProbe.ps1 -Policy vpt -Phase inference`로
실행한다. Policy는 `vpt`/`steve1`, Phase는 `initialization`/`inference`/`step`이다.
다른 정책 실행과 동시에 수행하지 않는다. 시험은 STOP 파일을 남기며 자동 재무장하지 않는다.
`stop-test-evaluation.json`에서 요청이 실제 해당 호출 시간 안에 있었는지, acknowledgment,
후속 정책 dispatch 수, 입력 해제 결과와 잔존 프로세스를 확인한다. 확인 후 시험의 STOP을
결과 폴더에 보존 이동한 다음 다음 시험을 실행한다. 이 진단은 수집 성공률 표본에서 제외한다.
`-TriggerStep 2`를 지정하면 첫 정책 행동이 이미 실행된 상태에서 두 번째 추론/step을
중단한다. 호출 시작 이벤트만으로 시점을 단정하지 않고 실제 call_started_s/call_finished_s
구간 안에 STOP 파일 생성이 완료됐는지 대조한다.

독립 감독 프로세스는 초기화 600초, 제어 180초, 종료 유예 10초를 관리한다. 추론/환경 호출이 멎으면 Linux subreaper와 PID/시작 시각으로 worker와 해당 실행의 자식 프로세스를 추적해 종료한다. 강제 종료 중인 시뮬레이터를 정상 종료 성공으로 보고하지 않는다. 180초는 **새 정책 행동을 허용하는 제어 예산**이며 기존 호출/종료 정리까지 전체 프로세스가 정확히 180초 안에 종료된다는 뜻이 아니다.

## 측정

- 추론 시간: CUDA synchronize로 비동기 큐를 정리한 뒤 act의 실제 경과 시간을 측정. 중앙값과 선형 보간 p95; 첫 추론 포함.
- 제어 주기: 화면 저장 + 추론 + env.step + 관측 로그 flush. 타이밍 이벤트 자체 기록 비용과 최종 정리는 제외. 가능한 한 실제 제어 부담을 반영한다.
- GPU: PyTorch peak allocated/reserved는 이 프로세스이며 모델 로딩을 포함한다. nvidia-smi device 0 전체 사용량은 약 250ms 주기 표본 최대값으로 다른 프로세스도 포함한다. 드라이버/할당자/렌더러 메모리를 동일 지표로 섞지 않는다.
- 획득: 초기 인벤토리 대비 최대 순증가, 최종 순증가. 재획득/소비를 추정해 합산하지 않는다. fullstats pickup은 보조 관측으로 기록될 수 있지만 성공 판정은 인벤토리다.
- 기술 실패율: attempted 중 실행 차단/오류/강제 종료/유효하지 않은 실행 비율. 사용자의 정상 취소는 따로 센다.
- 지시 실패율: 유효 실게임 episode 중 목표 아이템 증가가 없었던 비율. 분모 0이면 null, 미실행은 분모 제외. VPT는 지시 실패율 자체가 null.

이미지마다 PNG를 저장하므로 실행 속도와 디스크에 영향을 준다. 프레임 저장 빈도 변경 시 실험 설정 변경으로 기록해야 한다. 결과는 시간·호스트별 독립 디렉터리에 저장하며 실행 로그를 원문 연구 명세에 덮어쓰지 않는다.

## 결과 이후

로컬 메모리 부족이 관측되면 동일 시드/지시/시간을 Colab에서 실행한다. 제공 노트북은 아직 실행되지 않았다. Colab 결과는 로컬 실행 가능성 증거가 아니며, Colab에서 로컬 PC를 실시간 원격 제어하는 지연은 별도 실험이다. 유료 자원을 자동 선택하지 않는다.

정책 결과 다음 통합 순서는 Commitment의 task/revision 제어 → 검증된 이벤트 vocabulary → RM Runtime → 재계획 → 정책/API Skill routing이다. 해당 통합과 persona/memory/social/long-horizon/ablation은 이번 실행 하네스만으로 완료되지 않는다.
