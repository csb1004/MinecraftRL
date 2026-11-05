# 공통 이동 시연의 구간 확장 — 2026-10-19

새 독립 녹화를 추가한 실험이 아니다. 이미 영상/행동 정렬이 확인된 train2개/development1개 녹화에서
기존9구간과 겹치지 않는 추가 이동 구간을 찾았다. 개발 녹화는 새 조건에 맞는 미사용 구간0개라
기존 개발98행동을 그대로 유지한다. 최종5월드는 열지 않았다.

`results/common-human-broader-preparation-20261019/selection.json`은40frame 구간을5frame 간격으로 검사한다.
GUI/비이동 입력/클리핑 카메라/관측된 피해 증가/큰 위치 불연속/큰 높이 이탈을 제외하고,
수평 변위1.5–8블록·마지막4frame 높이 동일·기존/신규 구간 비중복 조건을 적용했다.
각 train 녹화 최대8개를 점프/비점프 교대로, 기존 선택과 시간상 거리가 큰 순서로 고정 선택했다.
이 필터는 접지·안전·사람 목표 의도의 증거가 아니다.

10개 후보의 모든410frame contact sheet를 직접 확인했다. 모래 계단/선인장 근처, 숲의 풀·흙 계단,
나무 사이 이동, 물가와 어두운 잎 가림 등을 기록했다. 안전 경로나 의도적 정지 label은 만들지 않았다.
공식 행동 사상에서 `054834d1b04a`의170–210 구간182frame의 좌·우 동시 키가 사라져
해당40frame 구간 전체를 제외했다. 최초 assertion 실패 스크립트와 후속 전수 감사 결과를 보존했다.

최종 준비 자료는 기존9+신규9=18구간720frame이다. 신규360frame에서 예정311학습행동,
점프41행동을 추가해 train212→523/개발98이 된다. 원본 영상/JSON/기존features와 labels는 보존한다.
`qualified-review/`의 manifest·visual-review·action-projection만 특징 추출 입력으로 사용한다.
원래10구간의 검토 결과와 제외 근거는 `review/`에 남긴다.

02:45 손실 비중 비교가 완료된 뒤 `run_qualification.py`를 최초 실행했다(exec40310).
기존360frame의 특징은 정확히 재사용하고 신규360frame만 GPU로 추출한다. 기존 root encoder와
실제 저장 game64frame의 재현을 다시 확인한다. 사후 목표 좌표·공식 정답·lineage·분할 검증과
기존9구간 labels/360features 동일성 검사를 통과해야 학습 계약을 만들 수 있다.
현재 추가 학습은0이고 특징 추출/자격 검사는 진행 중이다.

계획된 계약의 사람 BC 손실 비중은.25다. 별도의 sampler 비교는 기존212행 자료를 유지하므로
데이터 확장 효과와 샘플링 효과를 한 번에 섞지 않는다. 이 소규모 녹화만으로 공통 스킬의 일반화를
주장하지 않으며 실제 분리 게임 평가가 필요하다.

## 경로 오류 및 재시도 — 02:49

첫 exec40310은 특징 재현64frame 오차0 뒤 새 검수 경로의 날짜가20261017로 남아 있어 중단됐다.
실패 출력 `results/common-human-broader-features-20261017`과 원본 스크립트 `failed-attempt/`를 보존했다.
학습/게임 행동0, 신규 특징 저장0이며 자격 검사 완료로 세지 않는다.
세 준비 스크립트의 새 입력/출력 날짜를20261019로 수정했고, 원래 diverse20261017 원본 참조는 유지했다.
5입력 해시와 새 extractor 해시를 `retry1-prepared.json`에 바인딩했다.
`run_qualification_retry1.py`를 최초 실행했다(exec19888). 현재 상태는 `feature-workflow-retry1.json`,
로그는 retry1-*.log이며 원래 실패 ledger/log를 덮어쓰지 않는다.

## 자격 검사 완료 — 02:52

retry exec19888이 종료0으로 완료됐다. `feature-workflow-retry1.json`의 completed=true,
active=null,error=null을 확인했다. 최종 특징 파일은
`results/common-human-broader-features-20261019/features.pt`, SHA256
`50c7400028ee1010a12391de22ab74871c9fc372421a835bfcc1b72bf27ba035`다.
720frame 중 기존360개 특징과 기존310labels가 정확히 보존됐다. 새360frame에서311학습labels를
추가해 train523/development98이며, 새 점프41개를 더한 학습 점프는74개다.
저장된 실제 game64frame의 특징 재현 오차는0이다.

`extension-continuity.json`과 `human-reference-qualification.json` 모두 verified=true이며
`configs/common-human-navigation-broader-references-v1.json`을 만들었다. 자료 자격 검사는
실제 정책 실행 성공이 아니다. 현재 샘플링 비교는 기존212행을 유지하므로 확장 자료 학습은 아직0이다.

## 후속 학습 계획 고정 — 03:02, 실행 전

`configs/common-human-broader-training-v1.json`에 d613 출발·human 비중.25·local KL1.0·24/8배치·
2184갱신·local52416/human17472방문을 고정했다. SHA256은
`68b98c05111c5d274e30a16ae28cfbc43c5c8baa54fc86d637826131ce51b689`다.
train523개로 늘어도 학습량이 함께 늘지 않도록 가상 epoch당546갱신 옵션을 별도로 준비했다.
기존 기본546배치와 같은 seed의 전체 샘플 순서/RNG 상태 일치 및 확장 자료의 동일 방문 예산을
포함한 WSL CPU17검사가 통과했다. 현재 운영 코드에는 아직 적용하지 않았다.

현재 샘플링 비교의 종료·증거 고정과 코드 설치 검증 뒤에만 실행할 수 있다.
개발98행은 그대로다. 준비 기준은 전체 학습 점프≥25/74·비점프 오탐≤89/449,
기존212행 부분집합의 점프≥11/33·오탐≤35/179와 평지 mode4/4+4/4를 모두 요구한다.
자료 수에 맞춘 같은 비율 기준을 결과 전에 고정했으며 게임 성공 기준으로 쓰지 않는다.

KL 평균의 데이터 구성 비율은 local17240/17452에서17240/17763으로 변한다.
따라서 고정 BC 그룹 비중·갱신량 조건의 자료 구성 비교이며 순수한 다양성 효과만 분리했다고
주장하지 않는다. 새로운 독립 train 녹화0, 최종 월드0, 현재 확장 자료 optimizer0이다.

## 실제 비교 착수 — 03:12

앞선 sampler 비교는 train 점프8/33·오탐2/179, 개발0/19로 준비 기준 미달이었다.
결과 고정 후 갱신 예산 옵션을 설치했다. 최초 설치 검증의 검사 파일 LF/CRLF 차이는 텍스트
동일성을 확인해 검토한 CRLF 바이트로 맞췄고 실패 영수증을 남겼다. 재검증에서 실제 설치 해시와
17검사·다른353운영 파일 불변을 확인했다.

exec99272의 readiness가 train523/dev98, 학습 점프74·수직 목표277, 예산2184를 확인했고
study/training을 시작했다. 상태는
`results/common-human-broader-training-preparation-20261019/pipeline.json` 및
`results/common-human-broader-training-20261019/workflow.json`이다. 결과는 아직 미정이다.

## 완료 — 03:31, 실제 게임 진입 기준 미달

exec99272 정상 종료,2184갱신·root/value 보존·reload오차0·expert 감사 통과다.
checkpoint SHA256 `957dc4704c72f6a2470a115d514478fddd6e92b16049b5c78aeba1ad4dfe4e02`.
`controlled-training-comparison.json`에서 출발 actor/초기 모니터·14정답파일 바이트·갱신/방문 예산
일치를 확인했다. `evidence-freeze.json`으로 운영354소스 불변과 결과를 고정했다.

같은523행에서 대조ed31은 점프8/74·오탐3/449·NLL5.42443,
후보957은4/74·오탐0/449·NLL4.23089다. 개발98행은 대조0/19·오탐0·NLL5.82549,
후보1/19·오탐0·NLL3.95499다. 서로 크기가 다른 이전212행의 train NLL과 직접 비교하지 않는다.
후보의 원래212행 부분집합은 점프1/33·오탐0/179다.
평지 mode near4/4+far4/4는 통과했지만 점프25/74 및11/33 기준에 미달했다.
따라서 game_admitted=false, 새 게임0/기본승격0/최종월드0이다.
후속은 같은 자료·학습량의 [물리적 이동 버튼 보조 BC 비교](common-human-marginal-bc-2026-10-19.md)다.
