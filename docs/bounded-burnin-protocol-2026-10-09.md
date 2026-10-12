# 마지막 기억층 burn-in 계산 동일성 확인 — 2026-10-09

현재 학습 중인 onset/uniform 배치는 조건과 코드를 변경하지 않는다. 후속 반복의 계산 비용을
줄일 수 있는 별도 구현/검사다. 행동·정답·보상·학습률·gradient window를 바꾸지 않는다.

실제2x 모델 설정은 transformer4층, attention_memory_size256, timesteps128이다.
공식 MaskedAttention은 이 차이인128프레임의 K/V와 reset mask를 저장한다.
마지막 층의 K/V는 해당 층 입력의 정규화/선형 변환이며, 그 층의 과거 attention 출력으로
다음 K/V를 만드는 구조가 아니다. 앞선 층의 입력은 기존 전체 문맥에서 추출한 고정 특징이다.

따라서 각 window 전에 현재 가중치로 마지막128개의 **층 입력**을 재계산하면 모든 K/V와
reset mask가 교체된다. 이는 정책의 원시 영상 기억을128프레임으로 자르는 변경이 아니다.
초기128프레임 이내에는 원래와 같은 episode0부터 계산한다. 다른 recurrent 형식/attention
구조에는 적용하지 않고 명시적으로 거부한다.

1. 공식 구조의 작은 CPU 모델에서 cache 경계 전/후·현재 가중치 변경 후 출력과 모든
   parameter/input gradient의 정확한 일치를 검사한다. burn-in gradient 없음과 STOP 중단도 확인한다.
2. 현재 uniform 학습 종료 후, 실제 부모·집중 후보의 GPU 모델과 원본6시연에서 첫 행동 window 및
   뒤쪽 교사 window를 비교한다. 출력·기억 상태·정답 logprob·gradient가 일치하지 않으면 사용하지 않는다.
3. 동기화한 GPU 시간으로 같은 window의 기존/새 계산 시간을 측정하고 peak 메모리를 기록한다.
   새 optimizer 갱신/게임은0이다. STOP은 유지하고 원본/체크포인트 해시를 보존한다.

통과한 경우에만 이후 별도 학습 조건에서 사용한다. 완료된 onset/uniform 학습을 빠른 경로로
다시 실행하거나 시간 차이를 이미 얻은 것처럼 기록하지 않는다.
