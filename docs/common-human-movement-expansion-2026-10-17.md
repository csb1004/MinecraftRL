# 공개 사람 이동 자료의 범위 확장과 검토

기존3영상은 자연 지형의 다양성과 안전한 개발 표본이 부족했다. 공식
[VPT contractor 공개 자료](https://github.com/openai/Video-Pre-Training#contractor-demonstrations)의
보존한10.0 index에서 이전 녹화ID를 제외한 다음12녹화를 선택했다. 내용을 읽기 전에
녹화 단위2train:1development로 고정했다. 데이터는 Git 제외 경로에 두고 유료API는 사용하지 않았다.

출력은 `data/common-human-navigation-20261017`, 검토는
`results/common-human-movement-coverage-20261017`이다. `selection.json`/manifest에
원본URL·SHA256·ETag·byte수·split과 가용성 문제를 기록했다.

- 12개 중11개 메타데이터 다운로드,1개 공식 blob404. 없는 파일은 정책 실패 표본이 아니다.
- 1개 비UTF8 기록과1개 지원하지 않는 mouse button4 기록은 원본/사유를 보존하고 이번 엄격한
  선별에서 제외했다. 인코딩을 추측해 바꾸거나 미지원 입력을 정상 행동으로 바꾸지 않았다.
- 남은9개에서 겹치는2초 이동창 train11개·development14개를 찾았다.
  원래6녹화의 미다운로드 train 영상에서도14개 후보를 찾았다.
- 후보5개 이상인 미보유 영상4개, 총690,551,090byte를 공식 주소에서 받아 SHA를 기록했다.
  기존 원본 JSONL은 그대로 두고 감사용 사본을 영상 옆에 만들어 해시 일치를 확인했다.

## 실제 검토에서 드러난 제한

새4영상은 각각6,000행동/6,001영상frame,20FPS다. 앞1,100프레임의 optical-flow 검사를
`alignment.json`에 저장했다. 이는 경험적인 시점 증거이며 모든 프레임의 완벽한 정렬 증명은 아니다.
각 영상의 처음/중간/마지막 후보에서3프레임씩 총36프레임을 직접 확인했다.
범위와 판단은 `manual-visual-review.json`에 남겼다.

032f4eb01e68에는 물·용암 동굴과 실제 F3 overlay가 있었다. 평범한 지형 이동 pilot에 바로
넣지 않는다.176978c155fd는 사막/사바나 계단과 선인장 주변 이동이지만 낮은 체력/허기가 보인다.
개발181b0454f4f4는 깊은 동굴 입구와 어두운 동굴이며, 개발22bbc5451fc6는 낮의 풀 계단/숲이다.
화면 표본이나 피해 카운터 불변만으로 안전한 교사라고 선언하지 않는다.

새36개 후보창 중 끝12틱의 위치가 안정된 것은1개뿐이며 접지 증거는 없다. 대부분 계속
걷고 있으므로 임의로 고른 미래 위치를 **안정 정지 시연**의 목표로 붙이면 잘못된 감독이 된다.
일반 이동의 먼 목표 접근 자료와 실제 안정 정지 자료의 계약을 구분해야 한다.
원래 사람이 의도한 목표와 hindsight 좌표 목표도 구분한다.

현재 `training_ready=false`, 이 자료의 optimizer 갱신0이다. 현재 진행 중인 공통 이동 BC에는
감사한 로컬 교정 궤적만 들어간다. 사람 자료를 활용하려면 전체 선택 구간의 시각/overlay 확인,
공식 action projection, 현재 부모의 특징/기억 재생, 모르는 RM 상태를 꾸미지 않는 입력 계약,
분리된 실제 자연 지형 평가를 먼저 연결해야 한다.

## 전체 선택 구간과 공식 행동 사상 추가 확인

`complete-window-review`에서 train176978c155fd의3840..3880/5860..5900,
development22bbc5451fc6의3160..3200을 전체123프레임으로 확인했다.
이 세2초 구간에는 F3/열린 인벤토리가 보이지 않았다. 경사면의 블록 면에 반복 접근하는
장면, 물가 단차와 낮은 체력·허기, 개발 구간의 조합법/발전과제 overlay를 기록했다.
원본 녹화 전체를 검토하거나 안전한 전문가로 판정한 것은 아니다.

120개 실제 입력을 공식 mapper로 사상·역사상해 버튼 손실0/재사상 일치를 확인했다.
카메라 양자화의 최대 오차는 구간별1.24052/1.84052/0.31539도이며 원래 연속 카메라
입력과 완전히 같다고 주장하지 않는다. 실제 위치 끝점이 사람의 의도한 목표인지,
현재 부모의 특징·기억 재생과 미관측 RM 문맥을 어떻게 표현할지는 아직 미확정이다.
`training_ready=false`, 이 사람 자료의 optimizer0은 유지한다.

현재 부모 `onset-coverage-training-20261009/checkpoint.pt`의 scope는
`head_tail_last_recurrent`이고 추가 학습된 `backbone_tail`/`backbone_temporal`이 있다.
기존 `scripts/extract_bc_features.py`는 공식 `steve1.weights`만 읽고 RM 문맥을 영벡터로
채우므로 이번 공통 이동의 feature lineage와 미관측 문맥 계약에 그대로 사용할 수 없다.
기존 추출 결과를 새로운 부모의 학습 자료로 재분류하지 않는다. 후속 추출은 부모의 tail/마지막
기억층을 실제 로드하고, 별도2048시각 특징과 미관측 RM 문맥을 구분하여 실행 경로와 대조해야 한다.

## 별도 형식 후보군 확장

원래15개 유효 녹화의 보수적 후보71창은 보존했다. `expanded-movement-pool.json`은 별도
형식 후보군438창이다. 겹치는2초 창이며 독립 시연/성공438개라는 뜻이 아니다.
실제 이동 반경 상한을8→12블록으로 넓히고, 피해 카운터 누락은0이나 안전으로 간주하지 않고
`unverified_missing_counter`로 남겼다. 관측된 피해 카운터 변화는 여전히 제외한다.
공식 카메라 범위를 벗어나는 입력은 해당 틱의 loss 제외 후보로 표시하고 전체40틱 중
최소30개가 표현 가능해야 한다. 원본 행동/영상/녹화 split은 바꾸지 않는다.

예컨대 train054834d1b04a는31→195창, train0ac982de1b70는0→49창,
train1931b8e22571는0→39창으로 늘었다. 후자의 카운터 누락은 미검증 상태이며 더 좋은 교사라는
의미가 아니다. 앞서 F3/용암을 확인한032f4eb01e68의 제외 판단도 유지한다.
새 후보도 전체 구간·문맥의 영상 검토, 공식 사상, 부모 특징 재현과 목표 계약을 통과해야 한다.
아직 추가 영상 다운로드/새 후보 학습0이고 `training_ready=false`다.

## 추가3영상 확보·선택369화면 검토 — 17:38 후속

위 후보군에서 기존 recording split을 유지한 train0ac982de1b70/1931b8e22571,
development1d6469cf1ed4의 공식 영상517,867,820bytes를 추가 확보했다.
`data/common-human-navigation-expanded-20261017/download-retry3.json`이 완료 receipt다.
첫 다운로드는 진행 확인 중 소유 worker를 중단했을 때 이미2파일을 완료했고 세 번째는 부분 저장이었다.
완료한2파일의 해시를 확인해 재사용했다. 후속에서 기존 파일 거부와 If-Match ETag 인용 형식 오류를
각각 보존·수정하고, 마지막 파일만 새 임시 파일에 받아 길이/범위/ETag/SHA를 기록했다.

형식상 카메라가 모두 표현 가능한 첫/중간/마지막 창을 화면 확인 전에 골라9개2초 구간,
369화면을 전부 thumbnail로 확인했다. `expanded-window-review/visual-review.json`에 판단을 남겼다.
사막 구조물/계단, 숲의 나무 사이, 초지 단차를 확보했다. train0ac4300의 물 진입,
development340/1540의 해안 낭떠러지·깊은 구덩이 접근은 초기 일반 육지 pilot에서 제외한다.
검토한369화면에 F3/열린 인벤토리는 보이지 않았지만 전체 녹화 검토나 안전성 증명은 아니다.

세 영상 모두20FPS/6000입력/6001frame이다. 기존 앞1100frame optical-flow 검사의
0lag 상관도는0.7386/0.3147/0.7497이며 모두 인접lag보다 높지만 기존0.8 기준에는 미달한다.
평행 이동의 영향을 줄인 같은 제자리 회전 frame 쌍 검사에서는0lag0.6716/0.7062/0.9154였다.
새 train2개는 여전히 이전 기준 미달이고, 전체 영상의 시점 검증 완료로 표시하지 않는다.
별도 metadata의 yaw 변화와 같은 index의 mouse 입력은0lag 상관도0.9953/거의1/거의1이다.
이는 원시 입력 기록의 일관성 근거이며 화면 정렬을 대신하지 않는다.

기존에 검토한3구간의 현재 부모 특징 추출/실게임64frame 대조를
`results/common-human-feature-preparation-20261017/extract_qualification.py`로 준비했다.
아직 실행 전이며 목표/RM이 미관측이면 null로 남기고 시각2048채널만 분리한다.
이 시점에도 새 사람 자료 optimizer0, `training_ready=false`다.

17:44 후속: 위 특징 추출 검증을 실제 실행했다. 현재 부모 tail/마지막 기억층/초기 기억을
실게임 기록과 맞추고64프레임의2048시각 특징을 비교해 최대 오차0을 확인했다.
기존 검토 train2창80frame/development1창40frame의 특징·공식 행동도 저장했다.
`results/common-human-features-20261017/features.json`과 `features.pt`에 부모/모델/embedding/원본 해시를 남겼다.
기억은 각 선택창 시작에서 초기화하며 검토하지 않은 prefix 화면은 넣지 않았다.
미관측 RM/사람 의도 목표는 null, 모든 loss_eligible=false다. 특징 일치가 목표 감독이나 실제 학습
성공을 뜻하지 않으므로 이 자료의 `training_ready=false`, optimizer0은 유지한다.

18:27 후속: 원본 특징은 그대로 두고 기존3창 중 참고점 높이가 안정적이지 않은40frame을 제외했다.
나머지 train36/development35행동에만 먼 사후 참고점을 붙여 독립 재현과 CPU gradient 검사를 완료했다.
[공통 이동 입력·자료 계약](common-movement-input-contract-2026-10-17.md)을 따른다. 사람 optimizer0이다.

추가3영상의 정렬 진단에서 수직에 가까운 시선이 수평 median flow 비교를 흐릴 가능성을 점검했다.
검사 전에 |pitch|≤30도·공통 frame 집합·최소50표본·zero-lag 상관0.8·다른 lag 대비 차0.1을 고정했다.
이는 기존 결과를 본 뒤 만든 제한된 진단이며 기존 전체 구간 실패를 지우지 않는다.
`expanded-window-review/level-view-alignment.json` 결과: train0ac109쌍0.79098, train1931254쌍0.78308로
둘 다 미달했다. development1d64102쌍0.96929는 제한된 조건만 통과했다. 새 train 영상2개는 여전히
학습에 넣지 않는다. 전체 녹화/안전성/성공을 인증하는 결과가 아니다.

18:53 후속: 기존에 검증된 train36/monitor35행동을 별도 입력 계약으로 연결한
[혼합 BC pilot](common-human-navigation-pilot-2026-10-17.md)은 실제1,504갱신을 완료했다.
원본 feature bundle의 null/RM/loss 마스크는 보존하며 별도 검증 sidecar만 사용했다.
사람 모니터 NLL은17.9476→16.7709였지만 원거리 mode 준비3/4로 미달해 실제 게임0이다.
추가 train 영상2개는 여전히 학습에 사용하지 않았다.
