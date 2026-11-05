# MinecraftRL 작업 지침

- 실제 프로젝트는 `D:\DATA\Desktop\MinecraftRL`이다. Windows에서는 PowerShell을 쓰고 `gh`를 쓰지 않는다.
- 최신 상태는 `docs/progress.md` 및 거기서 링크한 최신 결과/프로토콜로 복원한다. 완료된 일회용 배치를 다시 실행하지 않는다.
- 새 에이전트는 `docs/handbook/agent-start.md`부터 읽는다. 폴더 역할은 `docs/README.md`, 코드 책임은 `docs/handbook/architecture.md`, 검사 명령은 `docs/handbook/development.md`를 따른다.
- 공용 코드는 `src/minecraftrl`, 실행 진입점은 `scripts`에 둔다. 기존 CLI/import/checkpoint 경로와 완료 실험의 `source-at-game`·해시를 보존한다. 기본 코드 검사는 `scripts/Invoke-Checks.ps1`이며 게임/학습 재개 명령이 아니다.
- 이미 승인된 프로젝트 내부 작업·격리 환경·로컬 실험은 단계별 재승인을 요구하지 않는다. 유료 자원·실제 Colab 연결·GitHub 공개는 별도 범위다.
- 사용자가 요청한 과업 분해 원칙은 `docs/task-decomposition.md`를 따른다. 새 과업은 검증 가능한 상태 변화 단위로 나누되 개별 키 입력을 RM 단계로 만들지 않는다.
- 상위 목표 RM과 skill 내부 단계 RM을 구분한다. 단계 정보는 학습 입력과 로그에 실제 연결하고, 미지원 과업을 구현된 것처럼 LLM 실행 vocabulary에 추가하지 않는다.
- 새 정책 목표 작성은 `docs/specs/2026-10-31-goal-condition-authoring-rules.md`를 따른다. 대상·기대 결과·현재 목적·상위 목표/제약을 보존하고 사람/LLM에 같은 검증을 적용한다. RM 상태 변화가 반드시 임베딩 교체를 뜻하지 않는다. 완료한 P1 조건을 소급 수정하지 않는다.
- LOGO 판별 함수와 구체적 정책 문구 v2는 `docs/specs/2026-10-31-logo-validator-adapter.md`를 따른다. 생성식은 고정 관측·호출자 소유 최종 성공 기준·반례 검사를 통과해야 한다. P1/P2는 유지하고 메타데이터 보존과 정책 문자열 표현을 구분한다.
- 성공은 실제 서버 상태/인벤토리 또는 저장 NBT로 판정한다. 화면 예측, 모델 로드, 합성 테스트, 준비용 scripted probe를 학습 정책 성공으로 세지 않는다.
- 학습/개발/최종 평가를 분리하고 원본 궤적·체크포인트·월드는 보존한다. privileged server 관측은 영상 전용 조건과 구분한다.
- STOP 이후 새 정책 행동 금지, 입력 해제와 소유 자식 PID 정리를 확인한다. 정책 실패를 알리지 않고 API/스크립트 제어로 대체하지 않는다.
- 연구 명세 원문 `docs/research-spec.md`는 별도 사용자 지시 없이 축소·수정하지 않는다.
- 서브에이전트를 자동으로 사용하지 않고 주 실행 흐름에서 순차 처리한다.
