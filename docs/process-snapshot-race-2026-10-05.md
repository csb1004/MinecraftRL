# 초기화 중 프로세스 감시 경합 — 2026-10-05

레시피북 작업대 실험의 마지막 시드262147에서 게임 초기화가 끝나기 전에 종료됐다.
조합 행동은0회다. 최초 원자료는
`results/craft-table-recipe-book-20261005T003358551921Z/262147`에 그대로 보존했다.

## 확인한 원인

감독 프로세스의 `supervisor.log`가 `ProcessOwner.capture()` → `snapshot()` →
`Path('/proc').glob('[0-9]*/stat')`의 내부 `exists/stat` 호출에서 발생한
`ProcessLookupError: [Errno 3] No such process: '/proc/5951/stat'`를 기록했다.
프로세스 목록을 찾는 동안 해당PID가 종료돼 사라진 것이다.

기존 코드는 `path.read_text()`의 OSError를 처리했지만, glob 이터레이터가 그 전에
실행하는 stat은 try 블록 밖이었다. 따라서 감시기 예외가 밖으로 나갔고, episode의 finally가
소유 프로세스 정리를 실행하면서 작업자에 SIGTERM을 보냈다. 작업자 기록의 SIGTERM과
MineRL launch EOFError는 이 종료 경로와 일치한다. 조합 선택이나 GPU 모델 추론은 시작되지 않았다.
감독기 예외 때문에 해당 시행의 정상 process_cleanup 요약도 작성되지 않았다.

## 수정과 검증

`src/minecraftrl/processes.py`에서 `/proc`의 바로 아래 이름만 iterdir로 열거하고,
각 PID의 stat 읽기는 기존 OSError 처리 안에서 수행하도록 바꿨다. 사라진PID는 건너뛰며
나머지PID를 계속 확인한다. PID+시작시각 소유권 검사와 비소유 프로세스 제외는 유지한다.

`tests/test_process_snapshot_race.py`는 열거 중 프로세스가 사라지는 상황을 재현한다.
수정 전 같은 ProcessLookupError로 실패했고, 수정 후 사라진PID를 제외한 두 생존PID를
모두 반환했다. Windows139개 통과/4개 제외, WSL143개 통과. 기존 자식 종료 검사도 포함한다.

재실행 전 프로젝트 관련 WSL Python/Java 잔존0을 확인했다.
`results/recipe-book-after-startup-failure-process-check.json`에 기록했다.
해당 시드만 새 독립 복사본에서 재시도하며 원래 초기화 오류를 성공 기록으로 덮어쓰지 않는다.
최종 재시도 결과와 집계는 레시피북 검증 보고서에 기록한다. 초기화0행동 회차는
제작 성공/실패 표본에서 제외하고 기술 오류1회로 별도 센다.
