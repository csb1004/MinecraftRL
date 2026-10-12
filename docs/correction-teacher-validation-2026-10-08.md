# 막힌 출발 교정 시연 결과 — 2026-10-08

학습용 월드 2개에서 **privileged scripted teacher의 실제 우회·원목 획득 시연**을 확보했다.
학생 정책의 학습·평가 성공은 0회이며 이번에 optimizer를 실행하지 않았다.
[사전 조건](correction-teacher-protocol-2026-10-08.md).

| 학습 월드 | 지형 | 실제 행동 | 제어 시간 | 서버 획득 | 벽 통과 행동 |
|---|---|---:|---:|---|---:|
| 104743 | 막힌 출발, 정방향 | 205 | 23.039초 | oak_log +1 | 73 |
| 104759 | 같은 구조의 z축 반전 | 205 | 27.679초 | oak_log +1 | 73 |

초기 canonical x=6.5에서 최소4.564까지 물러났고, z 최대5.226으로 벽 끝을 돌아
x≥8.3에 도달한 뒤 나무를 캐서 획득했다. 두 경로는 반전하면 거의 동일하며, 서로 다른
복잡한 지형 2종에 일반화한 결과가 아니다. 교사는 서버 위치·ray hit·알려진 fixture
좌표를 사용했다. API 우회 대체를 학습 정책 성과로 세지 않는다. 행동은 실제 MineRL
이동/시점/공격 입력이며 순간이동·아이템 지급 없이 완료했다.

## 보존과 검증

- `results/correction-teacher-20261008/experiment.json`: completed=true, errors=[], active=null.
- `results/correction-teacher-20261008/audit.json`: 410개 화면·행동·전이, 화면 크기640×360와
  PNG 무결성/SHA256, 전송 행동 일치, 이전 post-state=다음 pre-state, 최종 인벤토리 증가 확인.
- 원본 run: `results/rl-train-104743-20261008T024538270180Z/run/`,
  `results/rl-train-104759-20261008T024849167920Z/run/`.
- 각 run `teacher-rows.json`에 행동 전후 서버 관측, 행동 전 RM context, 보상 단계 기록.
  두 시연 각각 search66/approach3/mine129/collect6/complete1(전이 후 단계).
  화면은 `frames/`, 전송 및 STOP 이벤트는 `events.jsonl`.
- 두 run 모두 STOP 확인 뒤 새 dispatch0, 입력 해제 verified=true/held_key_count0,
  소유 자식 remaining_live_pids=[]; 실제 획득 뒤 추가 행동 없음.
- MineRL 종료 중 kill command timeout 경고가 있었지만 소유 프로세스 정리 결과는 모두0.
  실험 종료 후 별도 WSL 프로세스 조회에서는 systemd 사용자 세션 시작 경고가 나타났다.
  조회 목록에 게임 Java 프로세스는 없었으며, WSL 전역 서비스 설정은 변경하지 않았다.
- 제어 주기 중앙값/p95: 정방향60.84/72.79ms, 반전73.96/159.35ms.
  학습 모델 추론 성능 측정으로 해석하지 않는다.
- Windows 전체246검사 중230통과·16제외, WSL 신규2검사 통과. 신규2검사는 구현 전
  모듈 없음으로 실패한 뒤 구현 후 통과했다. 전역 시스템 변경·공개/유료 API 사용 없음.
- 원본 월드·부모 checkpoint·연구 명세 해시 유지. 기존 일회용 배치는 재실행하지 않았다.

## 한계와 다음 작업

`demonstration_candidate=true`, **`bc_training_ready=false`**다. 현재 상태는 교사 원본 확보다.
이번 2개는 모두 train 원천이다. 기존 평가 월드를 학습에 옮기지 말고 별도 개발용 시연을
만들어야 한다. 화면 전체와 행동의 정렬은 확인했지만 학생의 시점 이산화와 버튼 표현으로
변환할 때의 손실은 미검증이다. 런타임 wood embedding 일치·연속 메모리·기록된 RM context
입력 연결을 검증한 뒤 BC 후보를 학습하고, 이후 실제 학생 정책으로 분리 평가한다.

이 수집은 실패 정책과 같은 초기 fixture에서 교사가 실행한 별도 episode다. 정책이 방문한
중간 실패 상태에 전문가가 라벨을 추가한 DAgger 데이터는 아니다. 긴 벽 복구 일반화도
아직 해결되거나 검증되지 않았다.
