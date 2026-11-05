# 공통 이동 정책의 시각 특징 경로 확인

2026-10-23 자율 착지 자료 학습 중 수행한 코드/기존 로그 읽기다. 새로운 모델 추론·가중치 갱신·
게임 실행은 이 확인에 없으며, 실행 중인 학습 조건을 바꾸지 않았다. 아래 구조만으로 실패 원인을 확정하지 않는다.

`src/minecraftrl/policies.py`의 Steve1Policy는 공식 `weights/steve1/steve1.weights`를 먼저 불러온다.
이어 `RLPolicy`가 원목 학습 체크포인트의 action head와 `policy_tail.load_tail`이 지정한
`lastlayer`, `final_ln`, 마지막 recurrent block을 불러온다. 이 경로는 CNN 전체를 원목 자료로
다시 학습한 모델을 로드하는 경로가 아니다. CNN과 앞쪽 recurrent 층은 기본 가중치 경로를 유지한다.
현재 이동 학습에서는 이 특징 추출 경로를 동결하고 이동 보정/시각 투영/adapter1,471,828개만 학습한다.

실제 새 수집 로그 `results/common-student-landing-curriculum-20261023/correction-stone-east-upper/run/rl-condition.json`은
`prompt=collect wood`, `cond_scale=6`, `head_tail_last_recurrent` 계열의 학습된 후단을 기록한다.
실제 reset은 전달된 `goal-embed.npy`를 읽으며, 실행 임베딩 배열 SHA는
`d7fe31382a33c9e685ea84b5b438762857ebced6cb693d25eb43e062b28fb301`이다.
배열 해시와 NPY 파일 바이트 해시는 서로 다른 대상이다.

`RLPolicy.features`는 같은 화면의 조건부/무조건부 두 가지 특징과 기억 상태를 생성한다.
두 번째 가지의 `mineclip_embed`는0이며 `guided`는 두 head 출력을7/-6으로 결합한 뒤
log_softmax를 적용한다. 모든 행동 주체의 실제 관측 화면에서 기억을 갱신하고, 전달한 행동의
commit 시점에만 다음 기억 상태를 확정한다.

공통 이동의 `root_context_mode=ignored`는 별도로 덧붙인 상위RM6채널을 이동 head에서 제외한다.
STEVE-1 특징 생성에 쓰는 고정 조건 임베딩은 유지된다. 상대 목표·pitch·수평 움직임·남은 예산의
10채널은 양쪽 특징에 별도로 연결된다. 접지/수직 속도는 명시적인10채널에 포함되지 않는다.
시각/기억 특징이 그 정보를 어느 정도 담는지는 코드 경로 확인만으로 알 수 없다.

현재 대조는 자료 추가를 검증하므로 임베딩·guidance·특징 추출 가중치·입력을 유지한다.
향후 원인 분석에서는 관측 자료 부족, 고정된 조건/학습 후단의 영향, 시각 특징의 정보 부족을
서로 다른 가설로 구분해야 한다. CNN이 학습되지 않았거나 원목에 과적합됐다고 이 기록으로 주장하지 않는다.
