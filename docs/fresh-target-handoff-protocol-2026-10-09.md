# 새 목표 발견 시 즉시 수집 인계 — 2026-10-09

직전 고정 응답 실제 실행은800정체→803탐색→922새 원목 잠금→1002탐색 종료→
1327재정체/중지였다. `frames/000922.png`에는 나무가 보이며 `frames/001002.png`에는
벽만 보인다. 잠금 후80틱을 더 탐색한 것이 확인됐다.
원본: `results/goal-recovery-eval-104761-20261009T144548946737Z/run`.

`WorkshopObservation.java:wood()`는16블록 중심 ray로 처음 찾은 원목을 잠근 뒤,
시선을 돌려도32블록 거리 내에서 잠금을 보존한다. 따라서 `wood_v3_kind=1`만으로
현재 화면에 보인다고 판단할 수 없다. 실제 사례는 기존 종료 조건의 누락을 드러냈다.
기존 실행 당시 소스는 `results/recovery-outcome-fix-20261009/source-after-first-game/`에 보존했다.

수정은 관측/Java/RM/보상을 바꾸지 않는다. 한 실제 전이에서 kind0→1인 경우는
새 ray 관측 또는 accepted mining으로 원목을 새로 얻은 잠금이므로 바로 wood로 인계한다.
mine/collect 인계는 유지한다. 오래 유지된 잠금/과거 접근만으로 예산 종료 시 재개하지 않는다.
새 판정은 `target_acquired`, 목표 획득은 별도 실제 인벤토리 조건이다.
기존 목표/기억/정체 이력은 보존하고, 복귀 뒤 skill 구간 정체 감시도 유지한다.

검증: 실제922전이 경계 재생에서 즉시 인계,1002의 오래된 잠금은 인계 불허;
최초 발견 다음부터 탐색 행동0, STOP/완료 우선, 남은 호출0의 회귀 검사.
이후 기존 checkpoint/원래 wood 및 explore_9711 embedding으로 새 실제1회만 실행한다.
같은 corner archive, seed104761/sampling9721, 고정 응답5초, 제어180초,
탐색 최대200tick/30초/1회. 새 API/학습0, 최종5시드 미사용.
새 실행: `scripts/run_fresh_target_handoff.py`, 결과 `results/fresh-target-handoff-game-20261009/`.
서버 인벤토리, 실제 첫 발견/인계 tick, STOP뒤 행동0·해제키·소유PID, 추론/주기/GPU를 감사한다.
요청 전 픽셀이 달라지면 이전 실행과 성공률/시간의 인과 비교를 하지 않는다.
첫 실행의 판정·성공 여부를 소급 수정하지 않는다.
