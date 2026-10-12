# MineRL 서버 블록 관측의 스레드 교착 수정

2026-10-05 작업대 배치 관측을 추가하면서 실제 Windows/WSL 실행에서 발견했다.

## 증상과 근거

`results/workshop-20261005T013342880891Z`의 첫 104729 조건은 저장 월드를 로드했지만
첫 관측 응답을 반환하지 못했다. `events.jsonl`은 비어 있어 정책/GUI 행동은 0회다.
진단 STOP 후 supervisor가 소유 프로세스 6개를 정리했으며 잔존 PID는 없었다.
중립 입력 해제는 확인되지 않았으므로 정상 cleanup 성공으로 기록하지 않는다.
이 조건은 초기화 기술 오류이며 제작 성공/실패 표본에 포함하지 않는다.

`104729/java-startup-threads.txt`의 스택:

- `EnvServerSocketHandler`: `WorkshopObservation.read → World.getBlockState →
  ServerChunkProvider.getChunk → CompletableFuture.join`에서 게임 스레드를 기다린다.
- `Render thread`: `ReplaySender.tick → Object.wait`에서 다음 입력을 기다린다.

고정된 MineRL 1.16.5의 `ServerChunkProvider.getChunk`는 자신의 mainThread가 아닌
스레드에서 호출되면 executor에 작업을 예약한 뒤 join한다. 읽기 전용 호출이더라도
동기식 입력 대기와 결합하면 교착한다. VRAM 부족이나 정책 행동 실패가 원인이 아니다.

## 수정

게임 스레드의 `PlayRecorder.recordTickImpl`에서 관측 프레임 카운터를 증가시키기 **전**에
`WorkshopObservation.capture()`를 실행한다. 새 JsonObject를 volatile 참조로 발행하고,
통신 스레드의 `getInfo()`는 이 불변 스냅샷만 읽는다.
블록·아이템·해금 정보를 변경하거나 소켓 스레드에서 청크 작업을 기다리지 않는다.

중간 수정본은 `ReplaySender`의 입력 대기 전에 캡처했고 실제 1회 제작·저장을 통과했다
(`workshop-20261005T014059033512Z`, 183행동, 약21.07초).
그러나 프레임 카운터 발행과 캡처 사이에 경합 가능성이 있어 최종본은 카운터 발행 이전으로 옮겼다.
최종 5회 결과는 별도 배치로 기록한다. 중간 성공과 최종 5회를 하나의5개 분모로 섞지 않는다.

별도 JAR만 빌드했으며 upstream JAR은 변경하지 않았다.
최초 컴파일의 Slot.getSlotIndex 매핑 오류도 빌드 로그로 남겼다.
고정 버전의 실제 `Slot.slotIndex` 필드를 읽도록 수정했으며 게임 인벤토리를 쓰지 않는다.
