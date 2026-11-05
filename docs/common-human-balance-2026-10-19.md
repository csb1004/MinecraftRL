# 사람 이동 감독 비중 비교 — 2026-10-19

## 관측된 문제와 실험 범위

자연 train104743에서 d613과3ad8 모두4/6 도착, 오르막2개에서 jump0으로 실패했다.
첫 화면이 달라 두 실행을 통제된 성능 비교로 해석하지 않는다. [자연 평가 기록](common-natural-navigation-2026-10-19.md).

이미 검증된 사람 시연의 점프는 train33/development19개다. d613, 사람 BC ebd13,
보존 BC3ad8 모두 이 관측들에서 delivered mode 점프 선택0이었다.
점프 버튼을 포함하는 모든 latent encoding의 확률을 합하면 train 점프 시연에서 평균
d613 .01025, ebd13 .03018,3ad8 .02959였다. development에서는 .00311/.00779/.00770이다.
학습 후 확률은 증가했지만 실제 선택으로 이어지지 않았다. 이는 원인 후보이며 감독 비중만으로
실제 오르막 실패가 해결된다는 증거는 아니다.

증거는 `results/common-human-retention-training-20261019/human-jump-prediction-diagnosis.json`과
`jump-probability-diagnosis.json`이다. CPU 진단의 최초 합계 오차1e-5 단정은 실패했고 원본 스크립트를
보존했다. float32 분포를 double로 재검산한 log-normalizer 오차는 약5.2e-5 이하였으며,
1e-3 이상이면 중단하는 별도 진단으로 완료했다. 운영 확률 계산이나 실제 행동은 수정하지 않았다.

## 사전 고정 조건

`configs/common-human-balance-training-v1.json` SHA256
`734ee7e02a1c1b26d519decf47108b325162ca1267964a08e999969af62fdc71`.

비교 기준은 [보존 BC](common-human-retention-2026-10-19.md)이며, 새 학습은 동일한 d613에서 시작한다.
사람 손실 비중만 .10→.25로 바꾼다. 로컬 감독 비중은 정규화 계약에 따라 .90→.75가 된다.
추가 local KL1.0, 전체 KL.01, LR1e-4, seed14101,4epochs/예정2184updates, batch32,
fresh AdamW, 자료17452행(로컬17240+사람212), monitor254/98, frozen encoder/root/value를 유지한다.
사람 개발98행과 실제 개발 실행은 gradient에 사용하지 않는다.
자연 실행 지원 코드가 추가된 이후이므로 운영 소스354개를 새로 고정했다. trainer 변경은 없으며
종료 후 데이터·초기 actor·학습 전 지표·실제 갱신 예산을 기존 보존 결과와 비교한다.

학습 관측에서 점프 true-positive≥11/33, 비점프 false-positive≤35/179 및 평지 mode near4/4·far4/4를
다음 실제 검사의 준비 기준으로 미리 기록했다. 준비 기준은 실제 성공이나 일반화 판정이 아니다.
미달 시 실패를 보존하고 추가 구간 자료/학습 배치를 검토하며, 동일 후보로 게임을 반복하지 않는다.
기본 정책 교체/유료 호출/최종5시드 사용은0이다.

## 실행

02:28 `results/common-human-balance-preparation-20261019/run_pipeline.py`를 최초 실행했다.
exec97347, 상태는 같은 폴더의 `pipeline.json` 및 `results/common-human-balance-training-20261019/workflow.json`.
자료 검사212/98과 예정2184updates를 확인한 뒤 학습에 진입했다. 현재 결과는 아직 없다.
운영 src/scripts와 해당 준비 파일은 배치 종료까지 고정한다.

별도 `results/common-human-broader-preparation-20261019`에서 기존 정렬 녹화의 미사용10구간을
검수하고 있다. 이는 이번 비중 비교에 섞지 않는다. 새로운 독립 녹화가 늘어난 것은 아니다.

## 추가 진단과 다음 비교 준비 — 02:42

`gradient-mixture-diagnosis.json`은 초기 d613과 별도 고정 seed17571의 학습 미니배치3개만 검사했다.
점프 시연이 든 배치에서 비중 .10→.25에 따라 clip 전 norm39.88→99.67,
적용 배율 .02507→.01003을 기록했다. 비중 증가가 clip 뒤 영향 증가로 이어지지 않을 수 있다는
근거다. 실제 전체 optimizer 이력을 재생한 결과나 실패 원인의 확정 증거는 아니다.
진단 optimizer updates0/개발 gradient0/소스 파라미터 변경0이다.

`results/common-human-stratified-preparation-20261019/staging`에 매 배치24local/8human 샘플링을 준비했다.
원래 데이터 집단 비율/새 배치 집단 비율의 중요도 보정을 BC·전체 KL·local KL에 함께 적용한다.
따라서 기대 손실은 유지하고 배치 구성과 반복 노출을 바꾸는 별도 실험이다. 4epochs/2184updates일 때
local52416/human17472회 노출이 예정되며 기존의 모든 행4회 순회와 동일하다고 부르지 않는다.
기울기 clip 통계도 기록한다. 기대 손실/기울기 일치, 집단 배치/순환/시드, 사람 KL 제외와 기존 검사를
포함한 WSL CPU13개가 통과했다. 아직 운영 미반영·새 학습0이다. 현재 실행의 소스는 수정하지 않았다.

## 비중 비교 완료 — 02:46

exec97347 completed=true/error=null.4epochs/2184갱신, 체크포인트
`results/common-human-balance-training-20261019/training/mlp/checkpoint.pt`, SHA256
`3ccc96c2af39aca60ab72d30c33faabd7f6bbd52433d5044eb337451ef77d779`.
실제 actor 변경, root/value 보존, reload error0, GPU peak allocated449.997MiB.
학습 시간889.162초는 WSL 측정이며 다른 플랫폼 시간과 통제된 속도 비교에 사용하지 않는다.

보존 .10→.25 비교의 delivered NLL은 사람train8.1241→7.8973, 사람monitor8.9347→8.5555,
평지monitor1.05458→1.04898, 전체train1.32357→1.34363이다. 전체train 손실은 증가했다.
독립 expert 감사 통과. `controlled-training-comparison.json`에서 trainer·초기 actor·학습 전 모든 지표·
optimizer 예산·실제 정답 내용 동일성을 확인했다. 자연 지원 guard 추가의 소스 SHA가 바뀐 metadata 때문에
14sidecar의 바이트 동일성은 주장하지 않으며, 모든 labels는 정확히 같다.

`human-prediction-audit.json`: train33/development19 기록된 점프에서 선택0/0, 오탐0/0.
train 점프 시연 평균 점프 확률 .03040으로 이전 .02959와 큰 차이가 없었다.
따라서 사전 train 점프11/33 기준 미달이며 새 게임0, 자연/최종 평가0, 기본 승격0이다.
별도 기존 확률 준비 기준도 near .81167<.85로 미달했다. 평지 mode 검사는 별도로 진행 중이다.

다음 sampler 비교는 `configs/common-human-stratified-training-v1.json` SHA256
`d62a5177389eaf9f66c40daa79d1699208a12961e5a1ce64b0d7f6bf285cfeb9`에 고정했다.
자료212/98·출발d613·비중.25·예산2184를 유지하고 샘플링과 기대 손실 보정만 바꾼다.
확장523행 자료는 이 비교에 섞지 않는다. 기존 영상 확장 특징 추출을 먼저 순차 실행 중이다.

02:49 평지 mode는near4/4·far4/4로 통과했다. 점프 준비 기준 미달은 그대로이며 실제 게임을 시작하지 않는다.
`results/common-human-balance-training-20261019/evidence-freeze.json`에 진단/학습/감사/소스354불변을 고정했다.
