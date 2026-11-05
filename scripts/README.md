# 실행 스크립트 지도

[처음 시작](../docs/handbook/agent-start.md) · [설치](../docs/new-pc-setup.md) · [코드 검사](../docs/handbook/development.md)

스크립트는 실행 진입점이다. 공용 정책/학습 로직은 `src/minecraftrl`에 둔다.
Python 스크립트의 `PYTHONPATH`는 프로젝트 `src`를 포함해야 한다. 정책/학습은 기존 WSL 환경을 쓴다.

| 목적 | 진입점 | 주의 |
|---|---|---|
| 코드 상태 검사 | `Invoke-Checks.ps1` | 기본 All: CPU unit + Windows 임시 fixture + PowerShell 구문; 게임 시작 없음 |
| 환경 확인/설치 | `Test-LocalEnvironment.ps1`, `Get-OfficialSources.ps1`, `Invoke-PolicyProbe.ps1` | setup/download는 설치 안내에 따라 필요한 경우만 |
| 공식 정책 실행 | `Invoke-PolicyProbe.ps1`, `wsl-probe.sh` | STOP 상태와 새 실행 계획 확인 |
| 중단·진단 | `Stop-PolicyProbe.ps1`, `Invoke-StopProbe.ps1`, `probe_rm_conditioning_stop.py` | 실제 STOP probe는 게임 실행을 동반하므로 unit 검사와 다름 |
| 독립 R1/E1/E2 학습 | `train_recovery_adapter.py`, `train_effect_adapter.py`, `train_execution_adapter.py` | 현재 구현은 `minecraftrl.training`; 새 train 자료만 사용 |
| 앞선 S1/S2 학습 | `train_s1_heads.py`, `train_s2_conditions.py` | 각각의 고정 프로토콜 필요, 최신 E2 명령으로 오인하지 않음 |
| 목표/회수 준비 | `prepare_rm_goal_bank.py`, `prepare_rm_purpose_bank.py`, `prepare_logo_rm.py`, `prepare_recovery_assets.py`, `prepare_execution_pilot.py` | 기존 입력 덮어쓰기/완료 배치 재생성 금지 |
| 결과 분석 | `analyze_execution_contracts.py`, `analyze_effect_skills.py`, `analyze_rm_*.py`, `replay_logo_validators.py` | 기록된 원본을 읽는 분석; 정책 성공을 새로 만들어내지 않음 |
| 진단/보조 | `diagnose_*.py`, `read_f3_coordinates.py`, `Read-F3Text.ps1`, `Send-TaskCommand.ps1` | 진단 종류에 따라 모델/실행 중 세션에 접근할 수 있음 |
| 모델/빌드 보조 | `download_models.py`, `setup_dependencies.py`, `Install-MineRLWheel.ps1`, `minerl_build_cache.py` | 현재 설치 절차를 통해 사용 |
| 게시 보조 | `GitHub-ArtifactHelpers.ps1`, `Measure-ArtifactHash.ps1`, `check_publish_content.py` | 별도 게시 승인 필요; 마지막 검사는 Git index 기준 |

## 완료된 일회용 배치

다음은 고정 날짜/경로·조건을 가진 연구 드라이버다. **다음 작업을 이어가는 명령이 아니다.**

- `Invoke-RMConditioningPilot.ps1`, `Invoke-RMPurposePilot.ps1`, `Invoke-RMLogoPilot.ps1`.
- `Invoke-RMGoalQueuePilot.ps1`, `Invoke-RMGeneralS2.ps1`, `Invoke-UnifiedGoalU2.ps1`.
- `Invoke-RecoveryPreparation.ps1`, `Invoke-RecoveryDiagnostic.ps1`, `Invoke-CommonRecoveryR1.ps1`.
- `Invoke-EffectSkillE1.ps1`, `Invoke-ExecutionContractE2.ps1`.

일부 드라이버는 claim/work 파일로 재실행을 차단한다. 그 파일을 지우거나 경로만 바꿔 재실행하지 않는다.
새 연구는 새로운 프로토콜과 결과 경로를 사용한다. 과거 드라이버는 소스 고정과 명령 재현의 근거라 이동하지 않는다.
