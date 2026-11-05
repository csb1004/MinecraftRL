# RM Goal Conditioning Pilot P1 — 2026-10-31

공식 STEVE-1 동결 정책의 실제 Minecraft 비교 10회를 완료했다. 고정 목표 A는 4/5회,
RM 단계별 목표 B는 3/5회 원목을 획득했다. 목표 임베딩의 행동 확률 영향은 확인했지만,
이번 문구·전이 구성에서 수행 개선은 확인하지 못했다. **optimizer update 0이며 RL 학습 결과가 아니다.**

[전체 4단계 명세](specs/2026-10-31-four-stage-research-spec.md),
[S1 상세/P1 사전 조건](specs/2026-10-31-stage1-rm-conditioned-rl-spec.md),
[종료 처리 수정 P1.1](specs/2026-10-31-stage1-p1-finalization-amendment.md),
[시작 오류 재시도 P1.2](specs/2026-10-31-stage1-p1-startup-retry.md).

## Actual Gameplay

동일한 공식 가중치, guidance 6, 각 조건 제어 180초 상한, 개발 seed 5개를 사용했다.
A는 항상 `collect wood`, B는 RM의 search/approach/mine/collect에 따라
`go explore`/`approach a tree`/`chop down the tree`/`pick up wood`를 사용했다.
MineCLIP 텍스트 인코더와 prior가 생성한 512차원 벡터를 캐시하고 실제 정책 입력에 연결했다.
완료는 화면 유사도나 shaping 점수가 아니라 **실제 서버 인벤토리의 첫 원목 증가**로 판정했다.

| Seed | A 원목 | A 정책 step | A 제어 초 | B 원목 | B 정책 step | B 제어 초 | B 조건 전환 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 81031001 | 1 | 124 | 21.441 | 1 | 686 | 101.015 | 20 |
| 81031003 | 1 | 87 | 16.750 | 2 | 1189 | 175.656 | 24 |
| 81031007 | 0 | 1322 | 180.104 | 0 | 1061 | 180.082 | 2 |
| 81031009 | 1 | 729 | 106.592 | 0 | 1299 | 180.024 | 0 |
| 81031013 | 1 | 97 | 17.870 | 1 | 108 | 19.725 | 3 |

원목 0인 세 실행은 유효한 시간 초과/정책 실패다. B의 81031003은 마지막 한 step에서 원목 2개를
획득했고 완료 뒤 추가 정책 행동은 없었다. 성공한 공통 seed 세 개 모두 B의 획득 시간이 더 길었다.
180초는 새 dispatch 허용 상한이므로 진행 중인 step의 완료로 실측이 소폭 넘을 수 있다.

총 6702개의 유효 정책 step을 감사했다. 각 step은 서버 1 tick이며, 추론/렌더링을 포함한 실제
제어 시간과 다르다. 예를 들어 1322 step은 20 tick/s 기준 시뮬레이션 66.1초에 해당한다.
따라서 180초 벽시계 예산이 게임 시간 180초를 의미하지 않는다. 후속 학습 비교는 경험 tick과
실제 소요 시간을 함께 제한/보고해야 한다.

같은 seed의 초기 위치·시선은 5쌍 모두 같았다. 초기 PNG는 81031003만 동일했고 나머지 4쌍은 달랐다.
새 월드 생성 방식이며 완전히 동일한 상태를 복원한 쌍 비교가 아니다. 시작 전 월드 ZIP은 만들지 않았고,
seed·초기 상태/PNG와 플레이 후 월드를 보존했다. 플레이 후 월드를 시작 상태라고 부르지 않는다.
개발 표본 5개이므로 통계적 우월성, 미지 RM 일반화, 인간다움에 관한 결론을 내리지 않는다.

## Goal Influence, Separate From Competence

A의 초기 화면 5개 각각에 대해 초기 순환 기억을 같게 두고 목표만 바꿨다. 동결 정책의 결정적
forward 30회를 실행했으며 게임 step/가중치 업데이트는 0이다. `root_repeat`은 동일 조건 재계산 대조다.

| 조건 대 root | 버튼 분포 TV 중앙값 | 카메라 분포 TV 중앙값 | 결정적 최종 행동 변경 / 5 |
|---|---:|---:|---:|
| search | 0.588344 | 0.473507 | 1 |
| approach | 0.345566 | 0.280830 | 0 |
| mine | 0.067013 | 0.233742 | 0 |
| collect | 0.318338 | 0.383742 | 1 |
| root_repeat | 0 | 0 | 0 |

TV는 두 확률분포의 총변동거리이며 0은 동일함을 뜻한다. 모든 다른 목표/화면 조합에서 분포가 달랐지만,
가장 확률 높은 행동까지 달라진 것은 20개 비교 중 2개다. 실제 게임은 확률적으로 행동을 샘플링했다.
수치상 조건이 연결된 증거이며 문구 이해의 정확성·도달 능력·학습 효과의 증거는 아니다.
원본에는 head별 KL/TV, 행동, 화면 hash와 순환 입력 상태 불변 검사가 있다.

## Failure Evidence and Next Work

1. **탐색 목표의 의미:** B의 81031009는 1299 step 전체가 search였고 원목 0이었다.
   81031007은 search 984/approach 36/mine 41 step 후 시간 초과였다. RM에는 원목 목표가 남아도
   현재 정책 벡터 `go explore`에는 원목 수집 목적이 명시되지 않는다. 목적을 유지하는 탐색 문구나
   제한된 전환을 별도 P2에서 비교해야 한다. P1 문구를 결과에 맞춰 소급 변경하지 않았다.
2. **관측과 단계 의미:** 기존 wood_v3의 초기 나무 발견은 현재 시선의 16블록 ray로 판정한다.
   화면 전체의 나무 인식이 아니며 미관측은 나무가 없다는 뜻이 아니다. 저장 대상은 거리/파괴 등의
   규칙으로 유지된다. 접근 가능은 눈-대상 중심 거리·가림 검사 또는 서버에서 승인된 채굴 상태에
   의존한다. collect는 실제 아이템 entity가 아니라 이전 파괴 위치를 나타낸다.
3. **전환 불안정:** 성공한 B 두 실행에서 조건이 20/24번 바뀌었다. 3회 연속 확인만으로 접근/채굴
   왕복이 없어지지 않았다. 81031001의 로그에는 가까운 거리에서도 mine→approach가 있고 대상 블록도
   바뀐다. 거리 하나만의 원인으로 단정하지 않는다. 대상 유지·가림·도달 기준을 분리해 검사해야 한다.
4. **기초 이동 한계:** A의 81031007은 y=63에서 시작해 마지막 y≈41까지 내려갔고 원목 0이었다.
   고정 목표 정책에서도 지형 회복 실패가 남는다. A는 RM 단계를 목표 선택에 사용하지 않으므로 이 실패를
   단계 전환 탓으로 설명할 수 없다. 공통 이동/회복 능력은 실제 RL C/D의 검증 과제로 남긴다.

다음은 목표 의미를 유지하는 조건 표현과 안정된 과업 전이를 먼저 별도 프로토콜로 잠근 뒤,
공식 가중치에서 출발한 고정/단계 조건의 **실제 RL C/D**를 같은 경험 예산으로 비교하는 순서다.
독립 학습기의 수집 확률 재현, value/advantage, 종료 bootstrap, gradient/저장·재개 검사가 필요하다.
동결 A/B 연결 진단만 끝났으며 S1의 전체 G1/G2, S2~S4, D01~D13은 완료하지 않았다.

## Runtime and Timing

실제 로컬 Windows PowerShell → WSL Ubuntu-24.04에서 실행했다. GTX 1060 3GB/driver 560.94,
격리 Python 3.10.14, torch 2.0.1/CUDA 11.7, gym 0.19.0, MineRL 1.0.0/Minecraft 1.16.5,
Zulu OpenJDK 8u472를 사용했다. GLX에서 D3D12 GTX 1060 가속을 확인했고 실제 게임 PNG를 저장했다.
시스템 Python/Java/드라이버를 교체하지 않았다. API 호출/유료 연산/Colab 연결 0.

| Seed/조건 | 추론 중앙값 / p95 ms | 전체 제어 주기 중앙값 / p95 ms | 장치 전체 VRAM 표본 peak MiB |
|---|---:|---:|---:|
| 81031001 A | 48.848 / 56.909 | 150.482 / 179.612 | 2761 |
| 81031001 B | 52.702 / 60.460 | 139.208 / 165.873 | 2754 |
| 81031003 A | 50.730 / 133.302 | 153.148 / 251.561 | 2758 |
| 81031003 B | 51.369 / 60.114 | 138.325 / 181.029 | 2791 |
| 81031007 A | 50.661 / 59.624 | 129.860 / 163.050 | 2776 |
| 81031007 B | 49.570 / 58.883 | 161.283 / 234.739 | 2812 |
| 81031009 A | 49.367 / 57.745 | 138.887 / 163.716 | 2763 |
| 81031009 B | 52.262 / 59.924 | 133.038 / 156.139 | 2779 |
| 81031013 A | 53.426 / 129.596 | 154.571 / 235.356 | 2746 |
| 81031013 B | 53.110 / 134.050 | 152.611 / 247.902 | 2754 |

PyTorch process peak allocated는 약 1915 MiB이며 위 장치 전체 값과 다르다. 장치 표본은 다른
프로세스 사용량도 포함하며 순간 peak를 놓칠 수 있다. GPU OOM은 관측하지 않았다. 이 사실만으로
backpropagation이나 동시 플레이/학습의 메모리 적합성을 확인한 것은 아니다.

## Technical Errors and Safety

두 기술 오류의 원본과 수정/재시도를 모두 보존했다. 유효 성능 표본 10개 외에 아래 2개 attempt가 있다.

- `81031001-staged`: 실제 1180 step/180.023초 동안 원목 0. 입력 해제와 환경 close 이후 기록/인터프리터
  종료가 기존 10초 유예를 넘어 worker exit -15. 성능 표본에서 제외했지만 실제 미획득 결과도 공개한다.
  종료 표식 이후에만 별도 60초 마무리 시간을 허용하도록 수정했으며 제어 시간을 늘리지 않았다.
  이후 실제 180초 시간 초과 실행들은 worker exit 0으로 마쳤다. 재시도 이름은 `staged-finalization-r1`.
- `81031009-fixed`: startup 600초 timeout, 게임/정책 step 0, worker exit -9. Windows 파일의 9P 대기와
  열린 공식 가중치 파일을 관측했다. 독립 로딩은 import 99.635초/load 22.088초로 성공했으나 정확한
  지연 원인은 미확정이다. 이를 영구 해결했다고 하지 않는다. 재시도 이름은 `fixed-startup-r1`.

유효 10회 모두 행동·관측·RM 조건/벡터 hash·희소 보상의 대응, 서버 tick 1 증가, 완료/중단 인지 이후
새 dispatch 0, 실제 중립 입력 후 held key 0, 소유 자식 PID 0을 감사했다. 모두 worker exit 0이다.
각 기술 오류의 소유 자식도 정리됐다. 시작 전 실패에는 입력 해제 검증을 주장하지 않는다.

별도 실제 게임 STOP 진단(seed 81031999)은 정책 행동 10회 후 중단을 요청했다.
요청 파일이 보인 뒤와 stop acknowledgment 뒤 새 dispatch 모두 0회, 실제 held key 0,
소유 자식 잔존 0을 확인했다. 요청 파일 가시화→인지 약 42.1ms이며 파일 생성 시작 기준 상한은
88.4ms다. 상태는 cancelled, worker/supervisor exit 2(의도된 취소), 진단 helper exit 0이다.
이 한 번의 지연 실측을 실시간 최악 지연 보장으로 일반화하지 않는다. 성능 표본에 포함하지 않으며
`results/STOP`을 유지했다. 원본은 `stop-probe`, `stop-probe-request.json`, `stop-probe-audit.json`에 있다.

코드 검사: Windows 조건/공식 재시작/런타임 24개, WSL 조건 7개,
WSL 종료/감독 회귀 17개 통과. 같은 테스트의 플랫폼별 실행을 서로 다른 48개 테스트로 세지 않는다.
PowerShell 파서, 관련 Python 컴파일과 diff 공백 검사를 통과했다. 합성 검사와 실게임 결과는 별도다.

## Provenance and Reproduction Boundary

공식 파일 SHA256:

| 파일 | SHA256 |
|---|---|
| 2x.model | `89f9c5dc7c73484350a4055e842952ba5b70f524126fbc147edf7a4b6f27ccb8` |
| steve1.weights | `20f57f58985d338f1fccfc54d7f6407db0468656b697492f3bfb46ffade5c4cf` |
| attn.pth | `599cc7f51622fbcbab6db6517e6348ca36cab6ea89388b991d798a6ad68a1c24` |
| steve1_prior.pt | `e6496c3d926b02d391e5a1d54cce2a07409cdef9febdc5008543ad2ffdc39630` |

기준 Git HEAD는 `de1d2f932918382dba76ab1a790b788ad00f4346`이지만 작업 트리에 기존 수정/삭제와
이번 조건 실행 구현이 있으므로 이 commit만 checkout해 같은 실행을 재현할 수 있다고 주장하지 않는다.
명세/결과를 공개한 PR은 문서 전용이다. 새 실행 코드·모델·월드·대용량 로그는 이 PR에 포함하지 않았다.

로컬 자료는 `D:\DATA\Desktop\MinecraftRL\results\rm-conditioning-20261031`에 있다.

- 최종 실행 목록/판정: `pilot-startup-r1-plan.json`, `audit-startup-r1.json`. 초기 계획/감사도 보존.
- 원본 구현: `source-at-game`, `source-at-game-finalization-r1`, `source-at-game-startup-r1`와 각
  `work*.json`의 소스/프로토콜 hash. 실행별 `*-command.json`에 실제 argv와 출력 경로.
- 최종 진단/분석/드라이버/새 테스트 7개 파일은 `diagnostic-source-final`과 동명 JSON hash 목록에
  별도 보존했다. 마지막 게임 snapshot의 소스 115개와 현재 소스가 모두 같은 hash인지 확인했다.
- 실행별 `summary.json`, `initial-state.json`, `events.jsonl`, `frames/*.png`, `rm-rows.json`,
  `rm-switches.json`, `input-release-state.json`, supervisor/VRAM 기록. 목표 캐시는 `goal-bank`.
- `preflight/preflight.json`에 패키지/공식 저장소 commit/모델 hash,
  `preflight/glxinfo-d3d12.txt`에 렌더링 실측. 게임 runtime JAR SHA256은
  `dac9a1c9cb6b0a7d848559cd7f55d88f7d5bd89db4bad4d5ed35cf8fe9a530a2`.
- `goal-sensitivity/report.json`은 게임 없는 조건 민감도 결과,
  `startup-io-observation.json`, `model-io-probe.json`, `model-loading-diagnostic.*`는 시작 오류 진단.

완료한 일회용 배치를 다시 실행하지 않는다. 감사 재생은 게임/학습 없이 아래 명령으로 가능하다.

```powershell
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path (Get-Location) 'src'
    python scripts/analyze_rm_conditioning.py --batch results/rm-conditioning-20261031 --plan pilot-startup-r1-plan.json
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
```

이 명령은 현재 로컬 구현과 보존된 자료가 있는 프로젝트에서 사용하는 감사 명령이다.
GitHub 문서만으로 모델/시작 월드/실행 소스까지 배포됐다는 뜻이 아니다.
원문 `docs/research-spec.md`는 SHA256
`4af3ad00c5bab9b9d08b2e23fe768e140944118aa85cf412c45a67bb032275ad`로 변경 없이 보존했다.
