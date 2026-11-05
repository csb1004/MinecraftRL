# 폐기된 실행기의 검사 원본

2026-11-03 정리 전 전체 검사: 1,128개, 오류 5개, skip 38개.
오류 5개는 이미 삭제된 전용 실행기/테스트 도우미를 참조했다.
skip 중 20개는 실제 Torch 부재가 아니라 삭제된 `train_navigation_refinement` import 실패를
`Torch unavailable`로 잘못 표시하고 있었다. 이를 현재 검사의 성공/환경 skip으로 남기지 않는다.

아래 원본 파일을 바이트 그대로 `.py.txt`로 보존했다. Python test discovery에서 실행하지 않는다.
실행기를 되살린 것이 아니며, 현재 정책 코드가 해당 과거 기능을 제공한다고 주장하지 않는다.

| 원본 | 보관한 내용 / 현행 검사에 남긴 내용 |
|---|---|
| `test_blind_recovery_metrics.py.txt` | 삭제된 `summarize_blind_recovery`의 2개 검사 전체 |
| `test_navigation_mixed_supervision.py.txt` | 삭제된 trainer의 8개 검사 전체; 잘못된 Torch skip 제거 |
| `test_navigation_sampling.py.txt` | 삭제된 trainer의 12개 검사 전체; 잘못된 Torch skip 제거 |
| `test_inplace_recovery.py.txt` | 삭제된 `run_inplace_recovery.OneRetry` 검사 1개 보관; 공용 runtime 재개 검사는 현행에 유지 |
| `test_navigation_observation.py.txt` | 삭제된 trainer의 `load_run` 검사 1개 보관; 관측/이동 입력 검사는 유지 |
| `test_step_training.py.txt` | 삭제된 trainer의 `measure` 검사 1개 보관; 공용 step 수학 검사는 유지 |
| `test_policy_memory.py.txt` | 삭제된 `clone_slot` 호출/검증 부분만 보관; 원자적 world-store 저장·손상 검사는 유지 |

합계 25개 과거 실행기 전용 test method와 clone 보조 검증 부분을 현행 범위에서 분리했다.
원본 사본/해시와 정리 전 로그는 `results/project-maintenance-20261103`에도 보존한다.
과거 기능을 복원하려면 당시 코드/자료를 독립된 재현 조건으로 복원한 뒤 이 검사를 다시 검토한다.
