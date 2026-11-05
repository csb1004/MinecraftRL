# 독립 adapter 학습

R1/E1/E2는 동일 trainer를 사용하고 profile로 컨텍스트·return·검증 경계를 정한다.
공식 motor/기존 정책 클래스는 원래 모듈에 남긴다.

| 파일 | 책임 |
|---|---|
| `adapter.py` | `build_parser`, `main`, `train`; replay 검사 → update → 후보 저장 |
| `validation.py` | train split, parent, 실제 완료/행동 수/컨텍스트/terminal과 profile 검사 |
| `provenance.py` | 작업 소스 해시와 저장 전 변경 검사 |

기존 `scripts/train_recovery_adapter.py`, `train_effect_adapter.py`, `train_execution_adapter.py` 명령은 유지한다.
`train_recovery_adapter.validate`도 호환 import로 유지한다.
입력은 검증된 새 train 실행 폴더, 출력은 새 `results/` 폴더다. 개발/최종 평가 자료를 학습하지 않는다.

STOP, 데이터 admission, replay gate, GAE/PPO 수식, RNG/optimizer 순서, checkpoint scope는
이번 코드 분리로 바꾸지 않았다. `source_code` 기록과 학습 중 소스 변경 시 저장 거부가 추가됐다.
현재 코드로 학습을 재실행한 것은 아니다. 자세한 절차는 [E2 안내](../../../docs/execution-contract-e2-usage.md)를 따른다.
