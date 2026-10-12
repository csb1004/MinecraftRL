# 정체 감지·복구 경로 검증 — 수집 실패 포함

## 결론

**정체 감지, 입력 해제, 실패 상태 저장·복원, 명시적 계획 수정·재개의 제어 경로는
통과했다. 재개 후 흙 1개 수집은 실패했다.** 통합 실행의 원본 `passed=false`와
`replanned dirt task did not complete` 오류를 보존했다. 실제 수집 성공으로 집계하지 않는다.

추가 유료 API 0회. 원래 연구 명세와 기존 정책 기준선은 유지했다.

## 구현 및 로컬 검사

- 기본 60초 동안 목표 아이템 순증가의 최고치가 늘지 않으면 자동 paused.
- revision 증가, 실패 이유·관측 번호·경과 시간·임계값 기록, 중립 입력 해제.
- 추론 중 정체에 도달한 행동 폐기. 대기 중 STOP/명령을 우선 처리.
- schema 3으로 실패 근거 저장·복원, 이전 schema 1/2 읽기 호환.
- 명시적 resume은 baseline 보존, modify_plan은 완료 이력 보존 및 남은 목표 변경.
- 대화 콘솔 현재 과업에 실패 근거를 반영하고 이전 확인을 무효화.

새 기능 검사 4개가 실패하는 것을 먼저 확인한 뒤 구현해 통과시켰다. 별도 자체 검토에서
취소 우선순위와 단일 과업의 실패/취소 저장 검사를 2개 더 추가했다.
최종 Windows 95개 중 91개 통과·Linux 전용 4개 제외(1.911초), WSL 95개 모두 통과
(26.703초). 수정 모듈 및 실행기의 Python 구문 검사도 통과했다.
사용자 순차 실행 지침에 따라 하위 에이전트 없이 자체 검토했으며 독립 리뷰는 아니다.

## 실제 실행

경로: `results/failure-recovery-20261004T032512557525Z/`.
공식 MineRL 계열 기존 STEVE-1 환경, GTX 1060 3GB, seed 130363, 제어 상한 180초.
시작 전 GPU 443 MiB 사용 / 2523 MiB 여유, 드라이버 560.94를 재확인했다.

| 구간 | 설정·결과 | 정책 step | 제어 / 종료 처리(초) |
|---|---|---:|---:|
| 0 | wood 64→dirt 1 계획, 검사 임계값 5초에서 5.086초 정체 감지, paused 저장 | 14 | 6.255 / 4.464 |
| 1 | 실패 상태 복원, 남은 계획을 dirt 1로 교체, 명시적 resume, 60.052초 무획득 후 다시 paused | 534 | 82.371 / 5.346 |

구간 0은 정체를 의도적으로 유도한 제어 검사다. 구간 1은 기본 60초 임계값을 사용했고
실제 흙 순증가가 0개였다. 두 번째 정체 확인 후 외부 검사자가 세션 STOP 파일을
기록해 저장·종료했다. 제어 시간 82.371초에는 재개 전/정체 후 대기 시간도 포함된다.
두 구간 모두 원본 session status는 `cancelled`, task status는 `paused`다.
정체는 terminal 성공/취소로 위장되지 않는다.

| 측정 | 구간 0 | 구간 1 |
|---|---:|---:|
| 추론 중앙값 / p95(ms) | 45.203 / 383.667 | 42.492 / 51.800 |
| 제어 주기 중앙값 / p95(ms) | 144.192 / 624.233 | 102.383 / 129.844 |
| PyTorch allocated / reserved peak(MiB) | 2097.327 / 2144 | 2098.717 / 2144 |
| 전체 장치 관측 VRAM peak(MiB) | 2793 | 2797 |

첫 구간은 추론 15회뿐이며 초기 추론을 포함한다. GPU 수치는 다른 프로세스를 포함한
전체 장치 표본과 해당 프로세스의 PyTorch 측정을 구분했다. 자원 부족/OOM은 관측되지 않았다.

## 제어 근거

전체 이벤트를 별도 분석한 `control-evidence.json`에서 다음을 확인했다.

- 두 구간 모두 no_progress 이벤트 직전에 중립 입력 해제 성공.
- 각 정체 이후 새 정책 dispatch 0회. 두 구간에서 계산 중이던 행동 각각 1개 폐기.
- 복원 후 새 resume acknowledgment 전 dispatch 0회.
- 저장한 실패 근거 일치, paused 복원, 인벤토리 일치.
  위치 차이 0.275524블록으로 기존 허용 오차 2블록 이내.
- 계획 교체만으로 실행하지 않음. 별도 resume 후 실제 dirt 정책 행동 수행.
- 두 STOP acknowledgment 이후 dispatch 0회, 월드 저장 성공, 소유 자식 잔존 0개.
- 종료 후 별도 프로젝트 Python/Java 프로세스 조회도 잔존 0개였고 STOP을 다시 잠갔다.

제어 계약의 `control_passed=true`와 수집 결과 `policy_recovery_completed=false`를
분리했다. 통합 원본 acceptance는 계속 false다.

## 정책 실패 분석과 한계

실제 인벤토리로 흙 0개를 확인했다. `1/frames/000221.png`와 `000535.png`에는
발밑 구멍 안의 떨어진 흙 아이템이 보인다. 마지막 100개 행동 모두 attack 입력을
포함했고, 이동/점프 입력은 5개였다. 같은 구간의 수평 위치 변화 폭은 x 약 0.134,
z 약 0.216블록이었다. **채굴 후 아이템에 접근해 줍는 행동이 부족했던 정황**이다.
이것은 관측 기반 해석이며, 텍스트 조건/학습 분포 등 모델 내부 원인을 확정하지 않는다.

바닥의 드롭을 인벤토리 획득으로 바꾸거나, API 이동·아이템 지급으로 대체하지 않았다.
실행기는 이유를 `no_progress`로만 기록하며 경로 막힘 등의 원인을 단정하지 않는다.

이번 구현은 실패를 알리고 과업을 보존하는 기반이다. 자율적인 대체 스킬 선택이나
줍기 정체 해결은 남아 있다. 다음 정책/스킬 작업은 이 저장 월드와 로그를 재현 사례로
사용할 수 있다. 기본 60초의 최적성/오탐률, 여러 시드의 복구 성공률은 평가하지 않았다.

## 파일

- 위 실험 폴더의 `acceptance.json`, `control-evidence.json`, 각 구간의
  `events.jsonl`, `summary.json`, `task-checkpoint.json`, `frames/`.
- `results/failure-recovery-live.log`: 통합 실패 원본 종료 코드 2.
- `results/failure-recovery-windows.log`, `failure-recovery-wsl.log`: 최종 회귀.
- `results/failure-recovery-red.log`, `failure-recovery-green.log`: 최초 실패/통과.
- `results/analyze-failure-recovery.py`: 독립적인 이벤트 사후 검사.
- `results/failure-recovery-process-check.json`, `failure-recovery-final-receipt-20261004.json`:
  종료 확인과 성공/실패를 분리한 최종 기록.
- [실행 및 복구 사용법](failure-recovery.md).
