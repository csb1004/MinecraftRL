# 학생 문맥을 보존한 교정 BC — 2026-10-09

문맥을 유지하면서 교사 행동만 학습하는 BC와 실제 학생 게임4회를 완료했다. 개발 행동 예측
손실은 줄었고 자연 월드의 이번1회 대조에서는 수집이 빨라졌지만 긴 벽 우회는 해결하지 못했다.
기본 정책은 유지한다. 학습·모델 로드를 우회 성공으로 해석하지 않는다.

[학습 전 고정 조건](masked-correction-bc-protocol-2026-10-09.md),
[원본 교정 시연](student-correction-validation-2026-10-09.md).
이번 갱신은 BC이며 PPO 갱신·실제 LLM 호출·Colab 연결은 없다. 기존 목표 RM/보상/기본 정책을
변경하지 않았으며 원문 연구 명세와 최종 평가5시드를 보존한다.

## 문맥과 교사 정답의 분리

원본6시연 전체4961프레임을 순서대로 재생했다. train3260/개발1701프레임 중 교사 정답은
train1005/개발619개이다. 나머지3337개는 학생3331개와 인계 neutral6개이며 action target은
None으로 저장한다. 학생의 실패 반복 행동을 정답으로 복제하지 않는다.
학생 수집 당시 특징3331개와 새 추출 결과의 최대 오차는0이다.
[특징 추출](../results/masked-correction-features-20261009/features.json),
[실제 학생 입력 대조](../results/masked-correction-features-20261009/source-replay-audit.json).

마지막 기억블록·표현층·actor를 수집 당시 부모에서 복원하고 앞쪽 CNN/기억층은 동결했다.
원본 action/frame/해시/실제 RM6 입력과 train/development 분할을 유지한다. 교사는 privileged
서버 상태를 이용한 scripted 시연이며, 학생도 privileged RM6 입력을 받으므로 영상 전용 조건이 아니다.
새 schema2와 마스크 검증으로 기존 무마스크 학습기에 이 자료를 넣는 것을 거부한다.
원본 manifest에서 이어받은 준비 여부/optimizer0 필드는 수집 시점의 상태다. 이번 masked
학습의 실행 여부·갱신 수는 별도 training.json 및 supervision-audit.json으로 판정한다.

교사 정답이 포함된8프레임 창130개를1epoch 학습했다. 모든 창 앞은 현재 가중치로 episode0부터
no-grad 재생하고 창 안에서는 기억을 통한 gradient를 유지했다. 교사 위치만 BC/KL 손실에
포함한다. 실제 optimizer 기록에서 고유 train 교사 정답1005개를 각각1회 사용했고 개발 정답,
학생/neutral 정답, 문맥-only optimizer step은 모두0이다. 학습 창에 포함된 문맥15개는 이후
교사 행동에 영향을 주는 간접 gradient만 받을 수 있다.
[실제 갱신 감사](../results/masked-correction-training-20261009/supervision-audit.json).

## 학습 결과

AdamW lr5e-7, weight decay0, KL계수1, gradient clip1, window8, order seed9602.
학습 파라미터72,883,510개 중 기억블록50,675,104개이다. 부모는 기존 기억층 후보
`results/temporal-bc-temporal-20261008/checkpoint.pt`이고 새 후보는
`results/masked-correction-training-20261009/checkpoint.pt`이다.

| 지표 | 학습 전 | 학습 후 |
|---|---:|---:|
| train 교사 NLL,1005개 | 1.995253 | 1.734968 |
| 개발 교사 NLL,619개 | 1.901921 | 1.690696 |
| train 교사 KL | 0 | .028744 |
| 개발 교사 KL | 0 | .020539 |
| train 전체/문맥 KL | 0 / 0 | .009514 / .000944 |
| 개발 전체/문맥 KL | 0 / 0 | .008636 / .001827 |

이번 부모 대비 strict KL<=.03 gate는 통과했다. 이는 원래 공식 모델 대비 누적 KL 상한이나
이전 부모가 실패했던 gate의 소급 통과를 뜻하지 않는다. 실제 게임은 미리 정한 별도 진단 조건에
따른다. 학습 후 lr/epoch/마스크를 다시 탐색하지 않았다.

actor/tail/기억층 최대 가중치 변화는 각각 .000050915/.000056505/.000053588이다.
value 가중치와 condition은 그대로이며 바뀐 표현에서 value 출력까지 같다는 뜻은 아니다.
부모 순차 특징 오차0, reset+reload 뒤 개발1701프레임 특징·전체 측정 결과 불일치0이다.
460프레임 burn-in 뒤 문맥7개+첫 교사 정답1개의 gradient 검사에서 앞선 문맥 gradient
L1=96.818987, 기억층 gradient L1=129469.2791, 미래 입력의 과거 출력 영향0을 확인했다.
이 진단은 optimizer를 갱신하지 않았다.

학습760.179초, PyTorch peak allocated2015.886MiB/reserved2286MiB, OOM0.
8프레임은 gradient 범위이며 기억이 과거8프레임만 보는 것은 아니다. 전체 episode BPTT나
전체 모델 미세조정이 아니고 기존 head-only PPO optimizer 재개는 허용하지 않는다.
[학습 원본 기록](../results/masked-correction-training-20261009/training.json).

## 전체 추론·실제 학생 게임

부모/후보 각각 개발 far 시연653프레임 전체를 원본 영상 모델로 순차 추론했다.
캐시 기반 경로와 특징·마지막 기억 상태·행동 log-prob 차이는 모두0이다.
검사 중 optimizer/game action0, PyTorch peak allocated1116.821MiB/reserved1136MiB.
[전체 모델 대조](../results/masked-correction-runtime-20261009/report.json).

긴 벽2000003은 부모→후보, 자연2000029는 후보→부모 순서로4회 실행했다. 각 제어 예산180초,
sampling seed=world seed, 같은 보존 월드·wood embedding·CFG6·RMv4를 사용했다.
교사/스크립트의 자동 복구 개입은0이다. 두 월드 모두 초기 서버/기억/픽셀이 완전히 같았다.
이전 배치의 자연 초기 픽셀 불일치를 소급해 해결한 것으로 해석하지 않는다.

| 세계 | 정책 | 행동 수 | 제어 시간 | 실제 서버 획득 | 수평 이동 경로 |
|---|---|---:|---:|---|---:|
| 긴 벽 | 부모 | 1613 | 180.072초 | 0 | 3.683블록 |
| 긴 벽 | masked BC | 1586 | 180.045초 | 0 | 6.618블록 |
| 자연 | masked BC | 198 | 28.346초 | oak_log+1 | 35.701블록 |
| 자연 | 부모 | 700 | 98.982초 | oak_log+1 | 134.383블록 |

자연 조건에서는 같은 초기 상태/픽셀에서 후보가 더 적은 행동과 시간으로 원목을 얻었다.
각1회이고 이미 재사용한 개발 세계이므로 일반적인 속도 향상이나 성공률 개선의 증거는 아니다.
긴 벽은 둘 다 미획득이며 후보의 이동 경로 증가를 우회 성공으로 세지 않는다.

| 조건 | 추론 중앙값 / p95 | 제어 주기 중앙값 / p95 | 장치 VRAM 표본 최대 |
|---|---:|---:|---:|
| 긴 벽 부모 | 46.27 / 55.16ms | 107.83 / 127.26ms | 2572MiB |
| 긴 벽 후보 | 50.11 / 58.19ms | 111.57 / 126.28ms | 2572MiB |
| 자연 후보 | 46.48 / 56.73ms | 128.79 / 172.66ms | 2608MiB |
| 자연 부모 | 49.80 / 59.55ms | 135.33 / 158.99ms | 2607MiB |

게임 PyTorch peak allocated1915.001MiB/reserved2056MiB, OOM/VRAM 표본 오류0.
모델·Java·월드 초기화는 제어 시간과 분리된다. 장치 표본은 다른 프로세스 메모리도 포함하며
짧은 peak를 놓칠 수 있다. 제한 직전 이미 시작한 추론/제어의 종료 때문에 elapsed는180초를
소폭 넘을 수 있으며 STOP 이후 새 정책 전송은0이었다.

[실제4회 배치](../results/masked-correction-comparison-20261009/experiment.json)는
completed=true/errors=[]/active=null이다. [실제 로드·획득·중단 감사](../results/masked-correction-comparison-20261009/audit.json)에서
4097전이, actor/value/tail/기억 가중치 일치, STOP 뒤 새 정책 전송0, neutral 뒤 held_key_count0,
소유 live PID0을 확인했다. 미실행 조건0이며 최종 평가5시드는 사용하지 않았다.

## 긴 벽에서 남은 문제

부모의 z 범위는963.2384~963.5067이고 후보는961.3000~963.7974였다. 후보는 x방향 통과 없이
z가 감소하는 닫힌 모서리에 접근했고1586전이 중1144개가 z<=961.35에 머물렀다. 저장 월드
NBT에서 z=960 경계와 x=-713,z=961~967의 벽은 bedrock, x=-713,z=968은 air임을 확인했다.
통로를 통과하려면 반대인 z증가 방향으로968.3 이상까지 나가야 한다. 두 정책 모두 그 위치나
벽 뒤 x>=-711.7에 도달하지 못했다.

부모는297행동에서 재탐색에 들어갔다. 후보는290행동에서 재탐색→412행동 target_changed→
630행동 재탐색으로 이어졌지만 획득하지 못했다. 공격은 부모1570회/후보1523회로 여전히 대부분이다.
이는 **정체 감지·부분 이동이 있어도 열린 쪽을 골라 우회를 지속하지 못한 실패**라는 실행 수준의
분류다. [서버 경로·NBT 감사](../results/masked-correction-comparison-20261009/route-audit.json),
[실제 경로 그림](../results/masked-correction-comparison-20261009/long-wall-routes.png).

기억층 미갱신/추론 누락/학생 반복 행동을 정답으로 복제한 문제는 이번 검사에서 배제했다.
왜 방향 선택이 충분히 학습되지 않았는지는 아직 분리하지 못했다. 문맥 KL은 교사 위치 KL보다
작았지만 그것만으로 단일 원인을 확정하지 않는다. 교정 시작 행동의 희소성·학생 방문 상태의
범위·8프레임 gradient·동결 prefix·짧은1epoch·KL제약의 영향을 구분해야 한다.

다음 단위는 보존된 시연에서 **교정 시작의 방향 선택→지속 이동→통로 진입**을 구분해 정답
확률과 학생 행동을 분석하는 것이다. 이를 확인한 뒤 필요한 train 자료/학습 범위를 정하며,
전체 NLL 감소만으로 출구 선택이 개선됐다고 판단하지 않는다. 이번 평가 궤적을 train으로
바꾸거나 새 우회 skill을 실행 vocabulary에 등록하지 않는다. LLM이 미지원 우회를 실행할 수
있다고 가정하지 않으며 기존 RM/목표는 보존한다.

## 코드 검사

Windows 전체336검사 중319통과/17환경 제외, WSL 신규6검사 통과.
마스크/역할/중립 경계·시드 분할·프레임 누락·문맥 target 차단·교사-only 직접 손실·문맥의
간접 기억 gradient·문맥-only 창 생략을 검사했다. 주 실행 흐름에서 순차 검토했으며
사용자 지침에 따라 서브에이전트는 사용하지 않았다.

[최종 보존 검사](../results/masked-correction-comparison-20261009/final-verification.json)에
원본36파일/4961프레임/월드/모델/명세 해시와 실제 학습·게임·STOP 결과를 기록한다.
완료된 추출·학습·전체 추론 대조·게임·감사는 재실행하지 않는다. 로컬 Windows/WSL 결과이며
유료 자원·실제 LLM·Colab 연결·GitHub 공개는0이다.
