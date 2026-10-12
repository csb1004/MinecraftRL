# 조합 수량 초과와 서버 확정 — 2026-10-05

후속으로 Shift 제한 조건을 실제5회 비교했다. 정확 막대기4·판자2 보존0/5→2/5,
과다 제작2/5→0/5이며, 다른 레시피 선택과 미제작은 남았다.
[구현과 실제 대조](craft-shift-guard-validation-2026-10-05.md).

막대기 실험에서 정책 정지 뒤 수량이 4→8로 바뀌었다. 이전 판자 문제와 마찬가지로
클라이언트 예측을 서버 확정과 구분해야 하며, 이번에는 입력 한 번의 일괄 제작 효과도 있다.
성공 판정은 정확 재료 예산을 만족하지 않아 false로 유지했다.

## 관측한 경로

`results/craft-sticks-20261004T162709295721Z/130363/run`의 events.jsonl 및 craft-cleanup.json:

| 시점 | 정책 입력 | 클라이언트 막대기 | 입력 칸 판자 | 서버 막대기/제작 통계 |
|---|---|---:|---:|---:|
| step86 | 클릭 없음 | 0 | 4 | 0/0 |
| step87 | sneak=1, attack=0 | 0 | 4 | 0/0 |
| step88 | sneak=1, attack=1 | 4 | 2 | 0/0 |
| neutral_release | 중립 입력 | 4 | 2 | 0/0 |
| 두 번째 서버 대기 | 중립 입력 | 8 | 0 | 8/8 |

첫 실제 목표 관측 step88에서 정책을 정지했다. 이후 새 추론/정책 전달은 0회다.
커서 배치 보조도 사용하지 않았다. 최종 저장 NBT도 막대기8/흙1이며 판자는 없다.
시드155921도 목표 관측 step96 이후 새 정책 전달 없이 최종 막대기8을 저장했다.

## 설치된 Minecraft 소스 확인

다음은 실행한 MineRL 패키지의 로컬 원본 소스이며 변경하지 않았다.
공통 경로: `.runtime/steve1/lib/python3.10/site-packages/minerl/MCP-Reborn/src/main/java/`

- `net/minecraft/client/gui/screen/inventory/ContainerScreen.java:320`:
  Shift가 눌린 클릭을 QUICK_MOVE로 변환한다.
- `net/minecraft/inventory/container/Container.java:253`:
  QUICK_MOVE에서 같은 결과를 꺼낼 수 있는 동안 transferStackInSlot을 반복한다.
- `net/minecraft/inventory/container/PlayerContainer.java:123`:
  결과 슬롯0의 아이템을 주 인벤토리로 옮긴다.
- `net/minecraft/inventory/container/PlayerContainer.java:100` 및
  `net/minecraft/inventory/container/WorkbenchContainer.java:57`:
  재료 변경 시 결과 슬롯을 다시 계산하는 경로는 `!world.isRemote`인 서버에서만 실행된다.
  클라이언트의 첫 결과4개 이동 예측과, 결과 슬롯을 갱신하며 반복하는 서버의8개 처리가
  다른 중간 수량을 만들 수 있다. 이 차이는 새로운 정책 호출을 필요로 하지 않는다.

마지막 Shift+클릭, 정책 정지 후 서버 8개 확정, 원본 코드의 반복 제작 경로가 일치한다.
관측 뒤 새 행동을 막아도 이미 전달한 일괄 제작 요청의 결과를 4개로 제한할 수는 없다.
서버를 기다리면 오판은 막을 수 있지만 이미 소비한 판자2를 되돌릴 수는 없다.

## 해석과 후속 제어 경계

현재 프롬프트는 `craft sticks`이며 수량을 전달하지 않았다. 막대기 제작 능력과
제어기의 정확 수량/재료 보존 능력을 구분해야 한다. 8개 제작을 제작 불가능의 근거나
언어로 전달하지 않은 수량을 이해하지 못했다는 근거로 사용하지 않는다.

정확한 예산을 보장하려면 행동 전달 전에 결과 슬롯에서 일괄 제작을 제한하거나,
입력 칸에 한 번 분량만 넣도록 제어해야 한다. 이는 정책 출력에 대한 별도 제약/보조이며,
원래 정책 평가 도중 조용히 추가하지 않는다. 현재 고정 조건의 실험을 보존하고 후속 조건을
분리해야 한다. 서버 확정과 저장 NBT 검사는 그런 제어를 추가하더라도 유지한다.
