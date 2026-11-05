# 검사 지도

표준 명령은 프로젝트 루트의 `scripts/Invoke-Checks.ps1`이다.
[환경과 명령 설명](../docs/handbook/development.md)을 먼저 읽는다.

- **Unit**: WSL STEVE-1 환경에서 `unittest discover -s tests`. CUDA를 비활성화한다.
- **Host**: Windows fixture-tools Python에서 PowerShell 계약과 오프라인 NBT 초기화 검사.
- **All**: 둘 다 실행한다. PowerShell 구문 검사는 모든 profile에서 실행한다.

대부분 fake 환경·임시 폴더로 실행한다. 프로세스 종료 검사는 테스트가 만든 짧은 자식 프로세스만 사용한다.
유료 API나 실제 Minecraft를 실행하지 않는다. CUDA/F3 실제 캡처 검사는 CPU 검사에서 skip될 수 있으며
실행하지 않은 검사를 성공으로 세지 않는다. Windows 도구가 필요한 항목은 Host에서 별도 확인한다.

| 변경 영역 | 먼저 실행할 파일 |
|---|---|
| 실행 계약/보상/학습 | `test_execution_contract.py`, `test_execution_learning.py`, `test_execution_training.py` |
| 학습 entrypoint/STOP/소스 해시 | `test_training_entrypoints.py`, `test_recovery_learning.py` |
| 목표 의미/큐 | `test_goal_spec.py`, `test_goal_queue.py`, `test_rm_program.py` |
| 종료와 자식 프로세스 | `test_rm_finalization.py`, `test_rl_finalization.py`, `test_supervision.py` |
| 월드/정책 기억 | `test_world_store.py`, `test_policy_memory.py` |

`retired/*.py.txt`는 2026-10-25에 제거된 실행기 전용 검사 원본이다.
[보관 목록과 사유](retired/README.md)에 현재 검사에서 빠진 내용을 명시했다.
실패를 skip으로 숨기지 않는다. 기존 공용 runtime/world-store 검사는 의존성을 정리해 계속 실행한다.
