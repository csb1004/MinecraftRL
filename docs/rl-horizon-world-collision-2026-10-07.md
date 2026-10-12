# 새 개발세계 준비 실패와 복구 — 2026-10-07

첫 준비(2100011)는 정책 행동 0회에 `Cannot identify one owned Minecraft world`로 중단됐다. 평가 30회는 모두 미실행이며 실패 표본으로 세지 않는다. 원본 결과는 `results/rl-eval-2100011-20261006T232742759861Z/run`에 보존했다. 자식 프로세스 잔존은 0이었다.

직접 원인은 새 월드 저장 이름 충돌이다. upstream `EnvServer.createWorld`는 `RandomHelper.getRandomHexString()`으로 이름을 만들며 RandomHelper의 초기 난수원은 `new Random(0)`이다. 이번 이름 `mcpworlde3ee3add24b4`는 이미 10월 4일 생성된 폴더다. 당시 `results/semantic-world-scenario-20261004T013043802327Z/0/minecraft-latest.log`와 이번 Java 로그에 같은 이름이 있다. 기존 region/data 디렉터리의 생성 시각은 10월 4일, 이번 수정 시각은 10월 7일이다. 따라서 reset 전후의 새 level.dat 부모 폴더 집합 차이가 비어 안전 검사가 거부했다. Bad file descriptor는 이후 정리 단계의 이차 오류다. 해당 폴더의 현재 NBT 시드는 2100011이지만 기존 chunk가 섞일 가능성이 있으므로 재사용하지 않는다.

수정은 별도 session JAR의 새 월드 저장 이름에 UUID를 덧붙이고 기존 경로면 생성 전에 거부하는 것이다. 기존 게임 RNG 호출 횟수와 월드 seed 인자는 유지한다. 복원 경로는 변경하지 않는다. 준비 완료 후 및 ZIP 복원 전에 level.dat의 실제 WorldGenSettings.seed를 요청 시드와 대조한다. 원본 설치 JAR와 과거 ZIP은 변경하지 않는다.

복구는 사전 등록한 같은 5시드/3정책/2반복/180초를 유지한다. 첫 준비의 0행동 이름 충돌 실패만 확인하는 명시적 runner 옵션으로 재개하고, 기존 experiment JSON과 attempt-1 로그를 보존한다. 실제 평가 전에 모든 정책에 같은 수정 JAR가 적용된다. 학습 업데이트는 0이며 보상/정책/행동 경계는 변경하지 않는다. 기존 비교의 월드 ZIP 시드도 추가 감사한다. 이 수정과 테스트 통과 자체는 실제 게임 성공이 아니다.

실제 복구 결과: 같은5시드의 준비·NBT seed 검증과30회 정책 비교가 완료됐다. 모든 정상 준비/평가의 JAR SHA256은 `20b8fbe33500ba7932d5787782d66a81c98db6fd3e61ab67e0623e225d37f2b5`로 같았다. Windows181통과/7제외·WSL188통과, 원본 보존·STOP·잔존 프로세스0 최종 검사도 완료했다. [최종 결과](rl-horizon-new-worlds-validation-2026-10-07.md).
