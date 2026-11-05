# 문서 지도

[프로젝트 시작](../README.md) · [에이전트 인계](handbook/agent-start.md)

## 현재 기준 문서

| 필요한 정보 | 먼저 읽을 문서 |
|---|---|
| 지금 끝난 일·미해결 과제 | [progress.md](progress.md), 맨 위 최신 항목 |
| 처음 작업 시작 | [agent-start](handbook/agent-start.md) |
| 코드 책임·데이터 흐름 | [architecture](handbook/architecture.md) |
| 환경·회귀 검사·변경 절차 | [development](handbook/development.md) |
| 새 PC 설치 | [new-pc-setup](new-pc-setup.md) |
| 최신 연구 결과 | [E2 결과](execution-contract-e2-results-2026-11-03.md) |
| 최신 실행·독립 학습 | [E2 사용법](execution-contract-e2-usage.md) |

## 보존할 설계 기준

- [원문 연구 명세](research-spec.md): 별도 지시 없이 축소·수정하지 않는다.
- [4단계 구현 방향](specs/2026-10-31-four-stage-research-spec.md): 사람 RM → LLM RM → 이상 감지/학습 → SQL.
- [1단계 RM→RL](specs/2026-10-31-stage1-rm-conditioned-rl-spec.md), [과업 분해](task-decomposition.md).
- [목표 조건 작성 규칙](specs/2026-10-31-goal-condition-authoring-rules.md): 대상·결과·목적·상위 제약 보존.
- [LOGO 어댑터](specs/2026-10-31-logo-validator-adapter.md): 고정 관측·호출자 성공 기준·반례 검사.
- [E2 실행 계약](specs/2026-11-03-execution-contract-e2.md): 현재 효과/학습 경계.

## 현재 경로의 연구 계보

| 실험 | 문서 | 상태 |
|---|---|---|
| 목표 큐 | [사용법](goal-queue-s1-usage.md) | 공용 큐 구현; 지원 스킬 경계를 지킨다 |
| U1/U2 | [통합 계약](unified-goal-contract-2026-11-01.md), [비교](unified-goal-u2-results-2026-11-02.md) | 완료된 개발 비교 |
| E1 | [사용법](effect-skill-e1-usage.md), [결과](effect-skill-e1-results-2026-11-03.md) | 완료, 후보 미승격 |
| E2 | [사용법](execution-contract-e2-usage.md), [결과](execution-contract-e2-results-2026-11-03.md) | 최신 완료, 후보 미승격 |
| 유지보수 | [결과](project-maintenance-2026-11-03.md), [작업 계획·기록](superpowers/plans/2026-11-03-project-maintenance.md) | 완료, 코드/문서 정리이며 정책 성능 실험과 별개 |

## 과거 기록 읽는 법

`specs/`는 프로토콜/설계, `superpowers/plans/`와 `plans/`는 계획·작업 기록,
`reports/`와 `evidence/`는 과거 결과/증거다. 루트의 날짜별 문서는 당시의 판단을 남긴 기록이다.
날짜가 오래된 설치/명령 예시는 현재 경로로 추측해 실행하지 않는다.

2026-10-25의 사용자 지시로 예전 전용 학습기와 많은 자료가 삭제됐다. 해당 문서와
루트 `README-history-2026-10-10.md`는 현행 안내가 아니다. 완료된 배치를 재실행하지 않는다.
문서를 대규모 이동하면 실험/대화에서 기록한 경로가 깨지므로 원래 위치를 유지하고 이 지도로 구분한다.
새 안내는 `handbook/`, 고정 연구 명세는 `specs/`, 일회성 상세 결과는 날짜가 있는 문서로 작성한다.
