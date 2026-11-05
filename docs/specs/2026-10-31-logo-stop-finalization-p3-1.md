# P3.1 — 취소 후 RM 정리 경계 수정 및 별도 STOP 재검사

2026-10-31. P3 성능 15회와 첫 STOP 진단이 끝난 뒤 작성한 기술 수정 부록이다.
원래 P3 프로토콜·가중치·15회 성능 결과·실패 STOP 기록을 보존한다.
사용자의 현재 로컬 실험/문제 수정 승인 범위에서 수행하며 API/학습/게시를 사용하지 않는다.

## 관측 및 원인

첫 STOP 진단(`results/rm-logo-conditioning-20261031/stop-probe-audit.json`)은
10 transition 후 취소되었다. 요청 가시화/ack 이후 새 정책 dispatch 각각 0,
서버 입력 해제 확인, held keys 0, 남은 소유 PID 0이었다. 그러나 worker exit -15로
강제 종료되어 전체 진단은 **실패**다. 이 기록을 성공으로 재분류하지 않는다.

RM 실행기는 `run_episode`의 기존 `on_control_closed` callback을 연결하지 않았다.
환경 close를 먼저 기다리고, 취소 실행에는 `execution_valid=False`이므로 제어 종료 marker도
쓰지 않았다. 환경/인터프리터 종료가 감독자의 기본 STOP 정리 시간 10초를 초과하면
이미 행동이 중단된 worker도 SIGTERM을 받는다. 첫 실행은 제어 후 cleanup만 약 7.55초였고
MineRL kill command timeout 로그와 이어지는 종료 지연이 있었다.

## 수정 범위

기존 goal recovery/waypoint와 같은 callback을 RM에 연결한다. STOP gate가 잠긴 뒤
입력 해제 성공 → 정책 close 성공 → `rm-control-finished.json` 작성 → 환경 close 순서다.
취소의 marker에는 `cancelled_archive=true`를 기록한다. 취소를 유효 성능/학습 표본으로
바꾸지 않는다. release/정책 close/실행 오류가 있으면 marker를 만들지 않는다.

감독자의 기존 제한을 재사용한다. 증명된 제어 종료 이후에만 현재 기본 60초의 정리
유예를 허용하고, 그 안에 끝나지 않으면 소유 프로세스를 정리한다. 증명이 없는 STOP의
10초 제한은 유지한다. 정책 행동을 재개하거나 전역 제한을 늘리지 않는다.

## 확인 및 사전 고정 재검사

- 회귀 테스트: 취소 시 release/정책 close/marker/환경 close 순서, 취소 상태 보존,
  release 실패/정책 close 실패 시 marker 없음, 일반 marker는 수동 STOP 유예를 늘리지 않음,
  취소 정리 유예도 시간 제한 적용.
- 성능 15회는 재실행하지 않는다. STOP 진단만 별도
  `results/rm-logo-stop-r1-20261031`에 **1회** 수행한다.
- 기존 `logo-staged` 은행을 그대로 복사, 공식 동결 가중치, seed 81031999,
  제어 상한 180초, 완료 transition 10회 뒤 STOP, 기존 helper의 합격 기준 유지.
- 변경 소스 snapshot/해시/명령/환경/원본 STOP/재가동 receipt를 새 폴더에 보존한다.
  기존 프로브가 남긴 것으로 확인된 STOP만 한 번 보존 이동하며 새 STOP은 덮어쓰지 않는다.
- 합격: cancelled, worker exit 2, 요청 가시화/ack 이후 dispatch 0, 실제 서버 입력 해제,
  held keys 0, 남은 소유 PID 0, 최종 STOP 유지. 종료 marker와 제한 시간도 확인한다.
- 취소 진단은 어느 결과든 성능/학습 표본 수와 성공률에 포함하지 않는다.

첫 실패와 수정 후 결과를 함께 보고한다. 재검사가 실패하면 기록을 보존하고 원인을
재분석하며, 자동 반복 실행하지 않는다.
