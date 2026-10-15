# 실험 자료 공유와 복원

코드는 Git, 추가 학습 가중치·월드·실행 기록은 GitHub Releases로 공유한다.
`git pull`은 Release 자산을 내려받지 않는다. 아래 동기화 명령을 별도로 사용한다.
실험마다 코드·원목 정책·이동 정책·시작 월드·임베딩을 고정한다.

현재 공개 버전은 [research-20261015](https://github.com/csb1004/MinecraftRL/releases/tag/research-20261015)다.
8개 자료 묶음과 manifest의 합계는 약 871MB이며, 공개 다운로드로 5,631개 파일을 복원·검증했다.
검증 범위와 결과는 [발행 영수증](reports/2026-10-15/artifact-sharing-publication.json)에 기록한다.

## 받는 PC

PowerShell 7과 Windows Python 3.9 이상으로 자료를 동기화할 수 있다.
정책 실행은 기존 문서대로 별도 WSL Python 3.10 환경에서 수행한다.

```powershell
Set-Location 'D:\DATA\Desktop\MinecraftRL'
git pull --ff-only
.\scripts\Sync-Artifacts.ps1 -Tag research-20261015 -Experiment oct15-motion-corner
```

10/12 실패를 보려면 `-Experiment oct12-fixture` 또는 `oct12-visit-memory`,
세 실행을 모두 받으려면 `-Experiment all`을 사용한다.
이미 있는 파일은 SHA256을 검사하여 재사용한다. 다른 내용이 있으면 전체 복원 전에 충돌로 중단한다.
원래 파일은 삭제하거나 덮어쓰지 않는다. 다운로드는 `.cache/release-artifacts`에 저장한다.
같은 명령을 다시 실행하면 검증된 기존 파일/다운로드를 재사용한다.
실행 코드, pickle/checkpoint, 설치기, 게임은 동기화 중 실행하지 않는다.

복원 뒤 `reproduction/START-HERE.md`와
`reproduction/Show-ReproductionCommand.ps1 -Run oct15-motion-corner`를 확인한다.
명령 출력기는 당시 Python snapshot을 사용하고 새 출력 폴더를 지정한다. 출력된 명령을 실행하는 것은 별도 작업이다.
기존 일회용 실험 배치를 원래 결과 경로에서 다시 실행하지 않는다.

## 공개 자산과 공식/로컬 준비의 구분

- 추가 학습 checkpoint 3개, 학습 설정/결과, 개발 시작 월드 2개, 목표/skill 임베딩,
  10/12 실패 두 폴더 전체, 10/15 원목 목표 통합 실패 폴더 전체를 제공한다.
- 기본 모델은 `reproduction/base-models.json`과 manifest의 `base_models`에 공식 URL/ID·SHA256을 기록한다.
  기존 `scripts/download_models.py --policy steve1` 또는 `Invoke-PolicyProbe.ps1 -Policy steve1 -Operation download`로
  공식 배포처에서 받은 후 기록된 SHA256과 비교한다. 기본 모델 사본은 이 공개 릴리스에 재업로드하지 않는다.
- Minecraft 실행 JAR와 JAR 포함 MineRL wheel은 첨부하지 않는다.
  Minecraft EULA는 변형된 게임 실행 파일의 재배포를 제한한다.
  [Minecraft EULA](https://www.minecraft.net/en-us/eula)의 Java Edition 부분을 참조한다.
- `Get-OfficialSources.ps1`로 `configs/sources.lock.json`의 외부 소스를 받는다.
  새 환경은 `Invoke-PolicyProbe.ps1 -Policy steve1 -Operation setup`으로 프로젝트 내부에서 생성/빌드한다.
  기존 환경에는 setup을 반복하지 않는다. 초기 설치 기록과 conda explicit 목록도 자산에 포함한다.
  이는 인터넷에서 의존성을 받으며 새 PC에서 검증된 설치 결과를 뜻하지 않는다.
- `world_save_patch.py`와 보충 Java 소스가 별도 패치 JAR를 로컬 생성한다.
  당시 JAR SHA256은 `afeb88a9dd572bad19f25cf8d0a7346c08ef17ab5993e9de5b9da1ebee40688c`,
  patch source SHA256은 `58b912b13fbd74933c510c70ba14f9bc911266ce97eb8bc02196422f96e2fc49`이다.
  다른 환경에서 재빌드한 JAR가 byte 단위로 같다고 가정하지 않는다.

기본 모델 출처: [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1),
[VPT](https://github.com/openai/Video-Pre-Training), [MineCLIP](https://github.com/MineDojo/MineCLIP).
이 프로젝트가 해당 원본 모델이나 Minecraft의 권리를 새로 부여하는 것은 아니다.

## 다음 학습 결과를 발행하는 PC

`configs/artifact-release-20261015.json`을 새 파일로 복사하고 tag, 새 checkpoint/기록 경로,
실험별 group 연결을 갱신한다. 학습 중 저장되는 모든 checkpoint를 자동 공개하는 동작은 없다.
공유 대상으로 선택한 완료 실험만 아래 명령으로 게시한다.

```powershell
# 먼저 코드/설정 변경을 검토·검증·커밋하고 해당 커밋을 원격에 push한다.
.\scripts\Prepare-Artifacts.ps1 -Spec configs/artifact-release-20261015.json -OutputDirectory results/release-research-20261015
.\scripts\Publish-Artifacts.ps1 -BundleDirectory results/release-research-20261015 -NotesFile docs/release-notes-2026-10-15.md
```

다음 버전에서 준비할 때 `-PreviousManifest <이전 artifact-manifest.json>`을 추가하면
파일 경로·크기·SHA256이 모두 같은 묶음은 이전 릴리스의 자산을 참조한다.
checkpoint가 바뀌면 해당 묶음 전체를 새로 업로드한다. 바뀌지 않은 월드·기록은 다시 올리지 않는다.
이전 릴리스 자산은 이를 참조하는 실험이 있는 동안 유지한다.
직접 압축한 임의 ZIP을 자동으로 찾아 올리지 않으며 공개 목록 밖 파일은 포함하지 않는다.

발행기는 기존 Git credential 또는 프로세스 `GITHUB_TOKEN`을 메모리에서 사용한다.
토큰은 manifest/로그/명령 인수에 기록하지 않으며 `gh`를 사용하지 않는다.
새 릴리스를 draft로 만들고 모든 자산의 GitHub SHA256/크기를 확인한 뒤 prerelease로 공개한다.
업로드 도중 실패하면 같은 명령으로 draft를 재개할 수 있다. 동일 파일은 재업로드하지 않는다.
다른 내용의 기존 자산이나 태그는 덮어쓰지 않는다. `-DraftOnly`는 업로드 검증 후 draft를 유지한다.

GitHub 문서 기준 자산 하나는 2GiB 미만이다. 큰 묶음은 spec에서 여러 group으로 나눈다.
[Releases 제한](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases),
[Release API](https://docs.github.com/en/rest/releases/releases).

## 기록 해석

현재 코드 커밋과 당시 게임 코드는 구분한다. manifest의 code_commit은 전달 도구와 현재 연구 코드 버전이며,
각 게임의 실제 Python은 `source-at-game`에 보관돼 있다. 10/12 실행별 HEAD 영수증은 없다.
원래 snapshot에 누락됐던 Java 2개는 소스 조합 해시를 당시 JAR 영수증과 대조한 보충 사본이다.
기록된 게임 3개는 모두 `planning_paused`, 원목 0, `execution_valid=false`인 개발 진단이다.
이 릴리스의 공유/다운로드 성공은 정책 성능이나 새 PC 게임 실행 성공을 뜻하지 않는다.
