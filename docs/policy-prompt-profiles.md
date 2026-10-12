# 정책 조건 문구 비교

STEVE-1 어댑터의 `short` 기본값은 기존 `collect dirt` / `collect wood`다.
비교용 `official` 프로필은 고정된 공식 평가 코드의 같은 수집 목적 문구를 사용한다.
RM의 목표 아이템·목표 수량·baseline·인벤토리 완료 판정은 바꾸지 않는다.

- CLI 옵션: `--steve-prompt-profile short` 또는 `--steve-prompt-profile official`.
- guidance는 두 경우 모두 6.0이며 행동은 STEVE-1의 원래 출력이다.
- policy reset마다 실제 문구, 요청 목표, profile, guidance 및 조건 embedding SHA256을
  `policy_reset_finished.condition`에 남긴다.
- `official`은 지원하는 wood/dirt 목표만 받는다. 기본 `short` 동작은 유지한다.
- 이것은 조건 문구 비교 옵션이며 별도의 줍기 motor skill이나 자동 복구 기능은 아니다.

참조한 고정 소스는 `external/steve1/steve1/run_agent/paper_prompts.py`이며,
commit은 `903b244796322f4d0073a8f62c05f51eac3aed52`다. 공식 text guidance는
`external/steve1/run_agent/2_gen_vid_for_text_prompt.sh`와 일치시켰다.
text prior는 확률적으로 embedding을 샘플링하므로 reset/시드의 영향도 구분해야 한다.

## 실패 월드의 격리된 복사본

`--world-store-root`는 세션용 프로젝트 내부 저장소를 선택한다. 기본은 기존
`data/worlds`다. 비교 실행기는 각 시행의 results 폴더 안에 새로운 저장소를 만들고
원래 archive/manifest/current pointer의 복사본을 넣는다.
슬롯 이름은 유지하므로 같은 플레이어 UUID로 불러온다. 새 저장 세대는 복사본 쪽에만
생성한다. 원본 current pointer, manifest, world.zip의 해시/바이트가 유지됐는지 검사한다.
프로젝트 밖의 저장소 경로는 거부한다.

## 묶음 실행

프로젝트 Windows PowerShell에서, 기존 실행이 종료되어 STOP이 잠긴 상태로 실행한다.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python scripts/run_pickup_comparison.py
```

이 실행기는 문서화된 특정 실패 세대를 확인하고 총 10회만 실행한다.
동일 저장 월드에서 고정 시드 5개 × 두 문구를 비교한다. 각 구간은 최대 180초,
무획득 60초면 저장·종료한다. 기술 오류는 남은 시행을 진행하지 않고 보존한다.
원시 과업/정책 기록과 GPU·지연·제어 주기·획득량·정체·프로세스 종료 근거는
`results/pickup-comparison-<UTC>/`에 저장한다. 유료 API 호출은 없다.

완료 후 다음으로 집계한다.

```powershell
python scripts/analyze_pickup_comparison.py results/pickup-comparison-<UTC>
```

`completed`는 비교 실행의 완료 여부다. 실제 정책 성공은 각 시행의 실제 인벤토리와
`task_completed`로 구분한다. 미실행·기술 오류를 정책 실패 표본에 넣지 않는다.
같은 저장 월드의 소규모 결과를 여러 지형에서의 일반 수집 능력으로 해석하지 않는다.
