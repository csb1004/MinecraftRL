# 시연 보강·사전학습 내부 층 검사 결과

**학습6/개발2조건의 교정 시연 확보와 기존 STEVE-1 마지막 표현층의 실제 미세조정을
완료했다. 실제4회 개발 실행에서 긴 벽은 둘 다 실패, 자연 원목은 둘 다 획득했다.
우회 개선은 없으며 자연 시작 픽셀 차이로 속도 개선도 주장하지 않는다. 기본 정책을 유지한다.**
[수집 조건과 범위](diverse-correction-protocol-2026-10-08.md).

## 실제 시연

모든 조건은 새 synthetic fixture이며 교사는 privileged 서버 좌표와 알려진 지형을 썼다.
평가 episode를 학습에 넣지 않았다. 최종5시드는 미사용이다. 테두리 출발 조건에서는
나무가 아니라 바깥 벽을 바라보도록 초기 yaw를 지정했다. 실제 첫 서버 자세와 일치함을
검증했다. 정방향과 반전은 동일 구조의 대칭 조건이며 독립 자연 지형으로 세지 않는다.

| 분할/시드 | 출발·벽 끝 | 행동 | 제어 시간 | 서버 획득 |
|---|---|---:|---:|---|
| train104743 | 먼 출발 /5 | 210 | 23.744초 | oak_log+1 |
| train104743 | 테두리 정면 /8 | 237 | 24.146초 | oak_log+1 |
| train104743 | 먼 테두리 /11 | 271 | 27.853초 | oak_log+1 |
| train104759 | 반전 먼 출발 /5 | 210 | 24.244초 | oak_log+1 |
| train104759 | 반전 테두리 정면 /8 | 237 | 25.928초 | oak_log+1 |
| train104759 | 반전 먼 테두리 /11 | 271 | 28.504초 | oak_log+1 |
| development104761 | 다른 먼 출발 /6 | 281 | 28.389초 | oak_log+1 |
| development104761 | 반전 다른 테두리 /10 | 320 | 31.772초 | oak_log+1 |

총2037전이: train1436/development601. 교사8/8획득은 **학생 정책 성공으로 세지 않는다**.
`results/diverse-teacher-20261008/experiment.json` completed=true/errors=[]/active=null.
모든 전이의 실제 전송행동/프레임 대응·이전 post=다음 pre·STOP 이후 dispatch0·키0·소유PID0
감사 통과. 별도 audit에서 화면 크기/PNG 무결성/해시, 실제 출발 위치·yaw, 전체 RM/보상 재계산
일치 확인. 원본 월드·부모 체크포인트·연구 명세 해시 유지.

`configs/diverse-teacher-bc-20261008.json`에 세션 split/seed와 모든 입력 화면·행동·이벤트
SHA256을 고정했다. `results/diverse-teacher-action-audit-20261008.json`에서8조건 모두
버튼 왕복 손실0·유효 camera, 최대 회전 양자화 오차2.273705도(기존2.5도 기준 이내).
이는 변환한 행동을 게임에서 재실행한 검증과는 다르다. 새2037전이 특징 추출/BC 학습을 실행했다.

완료된 prepare_diverse_teachers.py, run_diverse_teachers.py,
prepare_diverse_teacher_plan.py 배치를 다시 실행하지 않는다.

## 가져와서 미세조정한다는 의미

현재도 [공개 STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1)의 사전학습 가중치를 사용한다.
STEVE-1은 VPT와 MineCLIP을 바탕으로 조건부 정책을 학습한 모델이다. 이전 로컬 BC/PPO는
그중 action head만 갱신했으므로 내부 표현까지 미세조정한 실험은 아니었다.
[공식 VPT](https://github.com/openai/Video-Pre-Training)에도 BC 미세조정 경로가 있다.

[MineStudio](https://github.com/CraftJarvis/MineStudio)도 조사했다. VPT/STEVE-1 계열
가중치와 offline/online 학습 도구를 제공하고 VPT contractor 데이터를 변환해 제공한다.
검토한 자료만으로 이번 막다른 길을 해결하는 별도 검증된 모델이라고 판단할 수는 없다.
새 패키지·가중치로 바꾸지 않고 우선 보유한 사전학습 표현을 재사용하는 범위를 점검했다.

## 마지막 표현층 자원 검사

`results/tail-finetuning-probe-20261008/probe.json` completed=true/error=null.
실제 train104743 시연 첫 프레임으로 기존 recurrent 계산을 한 뒤, lastlayer 직전 특징을
분리하고 **사전학습된 lastlayer/final_ln과 부모 action head**를 연결했다.

- 분리 경로의 runtime 특징 재현 최대 오차 **0**.
- 마지막 표현층 파라미터 **4,202,496개**, action head **18,005,910개**.
- 마지막 표현층 gradient L1=17270.53, 1회 AdamW 후 실제 최대 weight 변화≈1.00136e−5.
- 시각·recurrent 층 gradient 없음. 이 두 부분은 동결했다.
- PyTorch peak allocated **1512.59MiB(약1.48GiB)**, reserved1548MiB. OOM 없음.
- 이는 **한 프레임의 폐기용 optimizer1회**이며 배포용 checkpoint를 만들지 않았다.
  새 학생 episode0, 기본 정책 변경0. 전체 데이터 학습·일반화·최대 batch 성능을 검증하지 않았다.

원래 전체 정책249,545,918개 파라미터에 FP32 weight/gradient/Adam 두 상태를 GPU에 모두
두면 산술적으로만3807.77MiB(약3.72GiB)가 필요하다. activation을 포함하기 전부터3GB GPU
용량을 넘는다. 이는 전층 학습을 실행해 OOM을 재현한 결과가 아니며 AMP/offload 방식의
가능성을 배제하지도 않는다. 이번 제한된 tail 검사에는 로컬 메모리가 충분했고 Colab에
실제 연결하지 않았다.

## 실제 미세조정과 체크포인트

lastlayer 직전 특징을 전체 episode 순서로 추출했다. 각 행의 실제 RM6 입력을 유지한다.
CNN/recurrent는 원본 사전학습 가중치로 동결하고, head-only 대 lastlayer/final_ln+head를
동일 부모/데이터/2epoch/180갱신/순서seed9202/KL계수1로 비교한다. 원본 특징 재현0,
학습한 tail과 head를 모두 재로드한 logp 오차0. value 가중치만 동결이며 tail 변화에 따른
value 출력 변화는 가능하다. tail scope/base weight hash를 검사하고 head-only PPO 재개를 거부한다.
기준 KL은 원본 특징의 부모 정책과 계산한다. 변경된 표현을 기준 정책에도 넣지 않는다.

lr1e-5 두 후보는 각각 trainKL .124534/.281421로 gate 미통과, 비교4조건 미실행.
lr2e-6는 head .006992 통과/tail .072940 미통과, 비교4조건 미실행.
모두 원본 결과 보존. 이는 게임 실패8회가 아니다. 개발 손실은 모두 줄었지만 정책 변화 폭을
제한하는 사전 기준을 유지했다. 마지막 지정 조정은 두 조건 모두 lr5e-7이다.
각 학습은 새 디렉터리이며 기존 완료 배치 재실행이 아니다. 개발 튜닝으로 최종 평가와 구분한다.
캐시 학습 GPU 최대할당량 head500.68/tail596.89MiB. 전체 모델+단일프레임 probe와 조건이 다르다.
게임 GPU 총사용량과도 구분한다. 클라우드/유료 자원/새 외부 가중치 다운로드 없음.

Windows252검사 중235통과·17제외. WSL 새tail3검사 통과. 격리 NBT 환경에서 새 yaw 검사를
포함한9개 통과(신규 검사는 구현 전 실패 확인). 전역 Python/Java/드라이버 변경 없음.

## 실제 게임 비교 — 4회 실행 종료, 자연 비교 조건 불일치

`results/diverse-bc-lr5e7-comparison-20261008/experiment.json`은 실행4회 종료,
active=null/execution_finished=true. **completed=false**는 자연 초기 픽셀 비교 실패를
보존한 것이다. 재실행하지 않는다. 원본 종료 전 상태는 experiment-before-close.json.
4회 모두 실제 gameplay valid/서버 보상·입력 해제 감사 통과; 이 비교 오류를 게임 실행
실패나 미획득으로 바꾸지 않는다. audit.json의 comparison_valid는 long_wall=true/natural=false.

최종 lr5e-7 head/tail 각180갱신. devNLL2.197233→2.163623/2.066149,
trainKL .000452/.005544. 둘 다 gate 통과. tail 실제최대변화8.90642e-5.
사전학습 tail을 포함한 재로드 logp0. 여섯 후보 총1080BC갱신과 폐기용 probe1회는 별도다.

| 개발 조건 / 정책 | 행동 | 제어 시간 | 실제 인벤토리 | 중앙 벽 공격 | 통로 진입 |
|---|---:|---:|---|---:|---|
| 긴 벽 / head | 1478 | 180.080초 | 획득0 | 1453 | 없음 |
| 긴 벽 / tail+head | 1450 | 180.071초 | 획득0 | 1425 | 없음 |
| 자연 / tail+head | 283 | 45.086초 | oak_log+1 | 해당 없음 | 해당 없음 |
| 자연 / head | 474 | 73.992초 | oak_log+1 | 해당 없음 | 해당 없음 |

긴 벽은 양쪽 모두 canonical z=3.5를 벗어나지 못했고 외곽 공격0, 중앙 벽 공격만
반복했다. 초기 서버/메모리/픽셀 동일. head1453/1478, tail1425/1450 중앙 벽 공격.
RM 복구→search 전환은 전달됐지만 행동 우회는 발생하지 않았다. 모델 로드나 loss 개선으로
이 실패를 성공으로 바꾸지 않는다.

자연은 서버 위치/인벤토리/시간/초기 memory가 같지만 첫 화면 bbox(617,121,640,141)의
오른쪽 나뭇잎/하늘 경계가 달랐다. RGB 평균 절대차 .068199/.070243/.122873(0..255).
차이가 작아도 영상 정책 입력은 달랐으므로45.086초 대73.992초를 학습의 인과 효과로
해석하지 않는다. 렌더링 동기화/가시성 변동 가능성은 가설이며 원인 확정이나 수정 완료 아님.
반복1회·개발 튜닝 조건이라는 추가 한계가 있다.

| 조건 / 정책 | 추론 중앙값 / p95 ms | 전체 제어 주기 중앙값 / p95 ms | 장치 VRAM 최대 MiB |
|---|---:|---:|---:|
| 긴 벽 / head | 51.07 / 60.75 | 117.18 / 136.09 | 2756 |
| 긴 벽 / tail+head | 53.07 / 62.66 | 120.50 / 136.73 | 2756 |
| 자연 / tail+head | 49.26 / 58.57 | 148.90 / 203.89 | 2767 |
| 자연 / head | 48.92 / 59.94 | 146.31 / 191.50 | 2766 |

장치 VRAM은 다른 프로세스를 포함한 표본 최대이며 짧은 peak를 놓칠 수 있다.
CUDA OOM/표본 오류 없음. 게임 중 별도 학습/무거운 진단을 실행하지 않았다.
총3685전이 감사, STOP 후 dispatch0, 실제 held_key_count0, 소유 자식PID0.
게임4회 종료 후 tail-runtime-audit.json에서 실제 head/value/tail과 저장 가중치가 정확히
일치함을 확인했다. tail fingerprint는 두 실제 후보 실행 모두 동일하다.
원본 월드/부모/두 후보/goal embedding/연구 명세 해시는 종료 후 다시 확인했다.

## 게임 종료 후 단계 진단

`results/diverse-bc-lr5e7-comparison-20261008/stage-loss.json`: optimizer0/게임0,
시연 상태에서만 계산했다. 모든 단계 손실은 줄었지만 학생이 방문한 정체 상태에서의
올바른 행동이나 장기 우회 성공을 보장하지 않는다. train1436 중 mine676/search466/
approach258/collect36으로, 개선이 채굴에만 있었던 것은 아니다.

| 개발 단계 / 행 수 | 원본 NLL | head NLL | tail+head NLL |
|---|---:|---:|---:|
| 탐색 /118 | 2.144991 | 2.099726 | 2.004786 |
| 접근 /125 | 4.082058 | 4.003708 | 3.747533 |
| 채굴 /340 | 1.476208 | 1.461954 | 1.418394 |
| 수집 /18 | 3.070030 | 3.057902 | 3.027523 |

최종 Windows252검사 중235통과/17제외를 다시 확인했다.
로그: `results/diverse-bc-final-tests-20261008.log`. 모든 게임·학습·후처리 프로세스가
종료됐으며 STOP은 collection-ended 상태로 유지했다.

## 해석과 다음 단위

공개 STEVE-1의 내부 표현층 미세조정 경로는 실제 학습·게임 실행까지 연결했다.
전체 CNN/Transformer 학습은 아니며 privileged RM6 입력 조건이다. 전층 학습 필요성이나
시연 보강만으로 해결 가능하다는 결론은 내리지 않는다. 강한 학습률은 KL gate에서 멈췄고,
통과한 작은 변화는 벽 정체를 해소하지 못했다. **KL 제한 자체가 복구 행동 학습을 지나치게
억제하는지**도 아직 분리하지 못했다. epoch만 추가하거나 보상 단계만 늘리는 것으로
원인을 단정하지 않는다.

다음 단위는 (1) 출구 방향 선택/정체 상태의 교사 행동을 단계별로 평가,
(2) 학생이 실제 방문한 정체 상태의 데이터 범위 확인, (3) BC에 맞는 변화 제한과
기존 수집 보존을 별도 프로토콜로 검증, (4) 자연 초기 화면 재현성 원인 분리다.
최종5시드와 원문 연구 범위는 유지한다. 이번 로컬 결과를 최신 버킷 호환·일반화·인간다움의
증거로 확대하지 않는다. 실패를 API 제어로 대체하지 않았다.
