# 상대 경유점 이동 Implementation Plan

사용자 지정 방식: 주 실행 흐름의 순차 구현. 기존 실행 승인에 따라 재승인 없이 진행한다.
Goal: STEVE-1 시각/기억 특징과 상대 목표 조건을 실제 행동·도달 판정에 연결한다.
Architecture: 기존 수집 actor 보존, 이동 actor 추가, 기존 run_episode/STOP/서버 RM 재사용.
Tech Stack: Python3.10 WSL/PyTorch/MineRL, Windows PowerShell.
Spec: [고정 조건](waypoint-learning-protocol-2026-10-10.md).

## 검토 항목

음수/mirror 좌표, yaw wrap, 도착 직후 추가 행동, 학생/개발 정답 혼입, 다른 부모 캐시의
오용을 검사한다. 이미 충족된 경유점은 행동 없이 넘기고 STOP은 경유점 전환보다 우선한다.

- [x] `waypoints.py`와 검사: 상대 입력, handoff 정답 정렬, 분할/도착 경계.
  실패 검사를 먼저 실행하고 최소 구현 후 Windows 회귀를 실행한다.
- [x] `prepare_waypoint_features.py`: 감사 원본과 현재 부모 가중치로 순차 특징 재계산,
  source 해시/목표/행동/시각 프레임을 묶고 train/dev 고정 데이터 생성.
- [x] `train_waypoint_actor.py`: 별도 이동 actor BC, train만 optimizer, 목표 반전/zero/reload
  검사와 가중치·보존 보고서. 실제 GPU 메모리/시간 기록.
- [x] `waypoint_control.py`, `run_waypoint_validation.py`: 기존 수집→정체→경유점 이동→복귀,
  fixture 공급 출처·최대 예산·STOP·키/프로세스 종료·실제 궤적 감사.
- [x] 실제2개 개발 실행과 실패 원인 분리. 첫 후보2회,보강 정체 조건2회(이동미실행),
  별도 초기 목표 진단2회를 각각 보존했다. 실제4754전이 전수 감사. 정책 일반화 완료 선언은 아니다.
- [x] docs/progress·후속 단위·검증 보고서·원본 해시 갱신.
