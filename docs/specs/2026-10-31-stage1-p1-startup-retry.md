# P1.2 Startup Technical Retry

2026-10-31. [원래 P1](2026-10-31-stage1-rm-conditioned-rl-spec.md)과
[P1.1 종료 수정](2026-10-31-stage1-p1-finalization-amendment.md)을 보존한다.

7개 유효 실행 이후 `81031009-fixed`가 정책 행동 시작 전에 600초 startup 제한에 걸렸다.
감독 결과는 `hard_timeout/startup_timeout`, worker exit -9, 소유 자식 잔존 0이다.
`events.jsonl`이 생성되지 않았으며 정책 행동 0회다. 입력 해제는 초기화 전이어서 미검증이며,
정상 게임 실행/정책 실패/원목 획득 표본으로 계산하지 않는다. 원본 summary와 VRAM/콘솔/IO 관측을 보존한다.

관측한 worker 대기 위치는 Windows 드라이브의 9P 접근(`p9_client_rpc`)이며 열린 파일은
공식 `steve1.weights`였다. GPU OOM은 없었고 당시 WSL swap 사용은 0이었다.
그 후 독립 읽기 검사는 순차 32 MiB 약 0.120초, 임의 128개 4 KiB 읽기 약 0.050초로 끝났다.
같은 파일·환경의 실제 공식 모델 분리 로딩도 성공했다: import 약 99.635초, policy load 약 22.088초,
PyTorch peak 약 1915 MiB. 이 진단은 game step 0/optimizer update 0이며 실게임 성공이 아니다.
진단의 60초 faulthandler 출력은 중간 stack 표본이며 최종 결과는 정상 exit 0이다.

지연의 정확한 호스트 원인은 확정하지 못했다. 단순 읽기/모델 로딩에서 600초 지연이 재현되지 않았다고
기록하며, 환경 문제가 영구 해결됐다고 주장하지 않는다. 시스템 설정 변경, 강제 WSL 종료, 패키지 교체,
모델 변경 또는 프로젝트 이전을 하지 않았다. 추가 동작 코드 수정도 없다.

독립 로딩 성공 및 policy load 120초 미만을 확인한 뒤, `-ResumeStartupAfterIOCheck`로 한 번 재개한다.
기술 오류 조건만 `81031009-fixed-startup-r1`에 재시도하고 아직 실행하지 않은 81031013 A/B를 이어간다.
이미 유효한 7개 실행은 반복하지 않는다. 다시 기술 오류가 나면 자동 추가 재시도 없이 중단한다.
문구·임베딩·RM·난수·서버 관측·보상·제어 180초·startup 600초는 유지한다.

계획은 `results/rm-conditioning-20261031/pilot-startup-r1-plan.json`, 소스는
`source-at-game-startup-r1`, hash/기준 Git HEAD는 `work-startup-r1.json`에 보존한다.
첫 종료 오류와 이번 시작 오류를 모두 별도 제외 attempt로 보고하고, 초기 상태/렌더링의 비동일성도 유지한다.
