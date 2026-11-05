# 공식 모델 설치와 실행 — 2026-10-25 전환 안내

기존 추가 학습 가중치·실험 자료·전용 실행기는 사용자 요청으로 로컬에서 제거한다.
과거 `research-20261015` 릴리스는 이전 연구의 별도 스냅샷이다. 현재 작업 트리에 예전
`Sync-Artifacts.ps1` 명령을 적용하거나 그 가중치를 새 기본 정책으로 불러오지 않는다.
새 RL 자동 학습/감시 루프는 아직 구현되지 않았다. 과거 재현 안내는 현재 가중치/실행기 구성에 적용되지 않아 제거했다.

현재 코드에서 유지하는 공식 모델 경로는 `weights/steve1`과 `weights/vpt`이며,
모델 출처는 `configs/models.json`, upstream 소스 버전은 `configs/sources.lock.json`에 있다.
Windows PowerShell7에서 실제 프로젝트 경로로 이동하고 다음 진입점을 사용한다.

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
& .\scripts\Test-LocalEnvironment.ps1
& .\scripts\Get-OfficialSources.ps1
# 정책 환경이 없는 새 PC에서만 setup. 기존 환경에는 다시 실행하지 않는다.
& .\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation setup
& .\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation download
& .\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation preflight
# 사용자가 게임 재개를 요청하고 이전 종료/STOP 상태를 확인한 뒤에만 실행한다.
& .\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation episode -Condition wood -Seconds 180
```

위 명령은 설치/실행 방법이며 이번 정리에서 실행한 명령이 아니다. 현재 PC의 `results/STOP`은
유지되어 preflight/episode가 차단될 수 있다. 이 안내를 이유로 자동 해제하지 않는다.
새 PC 전체 설치 성공이나 현재 정책 게임 성능을 이번 파일 정리 결과로 주장하지 않는다.
