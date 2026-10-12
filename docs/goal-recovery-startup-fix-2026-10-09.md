# 실제 대조의 기술 오류와 재개 기록

고정 대조의 corner-wood는1668행동/200개입tick/원목0으로 정상 종료했다.
감사에서1411전이의 target tuple과 JSON list를 서로 다른 값으로 비교한 문제가 있었다.
실제 보상값과 대상 좌표는 같았다. 모든 표현을 JSON 의미로 정규화하고 값/좌표 변조 거부 검사2개를
추가했다. 원본 감사 코드·실패 배치 상태/로그를 보존하고 같은 게임을 재감사했다. head logp 오차0,
RM/그래프/200tick/STOP 이후추론·전송0/키0/소유PID0을 확인했다. 게임은 재실행하지 않았다.

corner-explore 첫 기동은 MineRL process watcher의 `Unable to acquire PID file`로 실패했다.
프로젝트 공용 `logs/minerl_watchers/watcher_596-887.pid`는2026-10-03에 작성됐고 내용은939다.
2026-10-09의 새 부모596/자식887 조합이 같은 파일명을 사용했다. daemoniker는 기존 파일 때문에
daemon 생성을 거부했다. 단계0/추론0/제어 시작 전 오류이며 정책 성능 분모에서 제외한다.
원본 log/PID/소스/실패 상태를 `results/goal-recovery-implementation-20261009/startup-fix-originals/`에 보존했다.

설치 패키지나 오래된 PID 파일을 삭제하지 않고, 공식 watchdog 명령의 cwd만 각 결과 폴더의
`watcher-runtime/`으로 격리했다. 원래 watchdog, CLI의 자식 subreaper/PID+시작시간 소유권 정리는
유지한다. 부모 worker가 다른 실험의 PID 파일을 재사용하지 않는다. 프로젝트 전역 설정은 바꾸지 않았다.
동일 PID 조합의 두 실행 경로 분리·원본 보존·실패 전파 회귀 검사2개를 추가했다.

같은 seed9721/모델/임베딩/월드/180초/개입200tick 조건으로 explore 기동만1회 재시도한다.
완료된 wood는 보존해 비교하며 시작 상태·픽셀·개입 전 prefix 불일치가 있으면 인과 대조로 주장하지 않는다.
원래 batch 상태는 보존본에 남기고 `technical_attempts`로 기술 오류와 실제 성능 실행을 분리한다.
입장 기준, 남은4조건, 최종5시드와 연구 명세는 변경하지 않는다.
