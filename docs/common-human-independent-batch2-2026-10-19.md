# 독립 녹화 두 번째 묶음과 특징 자격 준비 — 2026-10-19

현재 보조 BC 비교에 새 자료를 섞지 않는다. 아래는 후속 공통 이동 자료의 자격 검사이며
새 정책 학습/게임 성공 결과가 아니다. 원본은 Git에서 제외한 data/results에 보존한다.

## 수집과 분리

고정된 공식 VPT10.0 index에서 기존 사용/선택 그룹을 제외하고 다음12녹화를 선택했다.
내용을 보기 전에 순서대로 train2/development1로 나눴다. 녹화 그룹 구분이며 서로 다른 사람 수를 주장하지 않는다.
메타데이터12개 수신 완료,404/오류0. 기존40frame 이동 구간 필터를 그대로 사용했다.
학습4녹화/개발1녹화에 후보가 있었고, 유지되는 점프 정답 수 기준으로 train상위2/dev상위1을 골랐다.
정책 출력/점수는 자료 선택에 사용하지 않았다. 새 영상3개,총322935165bytes를 받았다.

자료: `data/common-human-navigation-independent-batch2-20261019`.
준비/검사: `results/common-human-independent-batch2-preparation-20261019`.
다운로드 전에 선언한 `video-protocol.json` SHA256:
`5653131af9d54f8385d78583383e679bd157d53fba5f049f731f2a757d6e04ea`.

## 시점과 화면 검수

기준은20FPS,영상frame수=기록행수+1,영점시차 상관≥.8/다른시차 대비차이≥.1/표본≥50이다.
영상의 처음 최대1100frame에서 같은 CPU optical-flow 검사기를 사용했다.

| 녹화 그룹 | 분리 | 영점 상관 | 판정 |
|---|---|---:|---|
|3ef2fda79f91|train|.935215|통과|
|416482eef705|development|.942554|통과|
|4cf439b98bb9|train|.698581|미달,모든후보제외|

10구간410화면을 모두 확인했다. 3ef의120–160구간은 화면 아래 채팅/상태 문자가 보여
이번 pilot에서 추가 텍스트 조건이 없는 시연과 분리해 제외했다. 416의3035–3075는 물이 있는
좁은 통로 이동이어서 육상 이동 자료에서 제외했다. 상단 레시피 알림도 원본/검수 기록에 남겼다.
남은 train300–340,345–385와 development810–850,1640–1680의160입력은
공식 mapper 왕복 및 물리적 버튼 보존 검사를 통과했고 손실0이다.
학습 후보71행/점프26,개발 후보56행/점프5이며 아직 특징/사후목표 자격 검사는 실행 전이다.

영상 SHA256:

- 3ef: `90ddd8d772f57e3f7a6e87e1149e4e08883908b03ded67cf5df07042423b3f61`
- 416: `04fe1a281f29c0a95de61f06c2109eb4cb6298986e02d50b3d9998f99fe204c9`
- 4cf: `1ad3fa63b532ad47c9066ff2906ea1536c59828ec6a3f03da888abd5c498bed8`

시점 미달은 완전한 비정렬의 증명이 아니다. 현재 계약의 채택 기준 미달로 남기며 기준을 낮추지 않았다.
화면/메타데이터에서 지면 접촉,인간의 실제 의도,안전한 경로,목표 달성을 새 정답으로 만들지 않았다.

## 독립 녹화 특징 자격 준비 — 03:53

첫 묶음의197c train35행과 이번 train71/dev56을 합친 새106train/56dev가 후보이다.
기존720feature/621labels를 정확히 재사용하고 새200frame만 동결 인코더로 계산할 준비를 했다.
예정 합계920frame,train629/development154,train4녹화/development2녹화다.
기존 독립 녹화2train/1dev에서2train/1dev를 추가하는 것이며 이전 개발 자료를 학습으로 옮기지 않는다.

`results/common-human-independent-features-preparation-20261019/prepared.json`에 원본 해시,
화면/시점/공식 행동 검사,증분 특징 추출기 및 순차 검토를 고정했다.
실제 저장game64frame의 특징 재현오차0과 부모/기본모델/임베딩/tail/temporal 계보 검증을 다시 요구한다.
`run_qualification.py`는 현재 보조 BC 학습 pipeline이 완전히 종료된 경우에만 GPU를 사용할 수 있다.
현재03:53: 준비 완료,특징추출0/새 자료optimizer0/게임0/API0. 자격 통과 여부는 아직 미정이다.

## 특징 자격 및 후속 학습 완료 — 04:39

`results/common-human-independent-features-preparation-20261019/feature-workflow.json`이 완료됐다.
기존720frame/621labels 보존+새200frame, 실제64저장화면 특징 재현오차0,
사후 목표 재구성 최대오차1.11e-16을 확인했다. train629/dev154(4train/2dev녹화) 계약은
`configs/common-human-navigation-independent-references-v1.json` SHA
`733db7d34869b85a79f26d0fdba53360e9c896b2db3ec29dbc503f5473606053`이다.

고정 특징 SHA `14e026b6462a8ba15b85291b6d951a662b46b24125ce5c01310a02a5d5820bb6`.
[후속 BC 결과](common-human-independent-training-2026-10-19.md)는2184갱신·점프13/100·오탐3으로
고정 준비 기준 미달이다. 이 자료는 자격을 유지하며 같은 자료의 시각 경로 비교에 사용한다.
독립 녹화 추가는 성공적인 게임 일반화가 아니며 API/이 자료 후보의 게임 실행0이다.
