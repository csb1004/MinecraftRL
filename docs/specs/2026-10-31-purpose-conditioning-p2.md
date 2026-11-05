# P2: 대상과 목적을 보존하는 목표 조건 — 사전 고정 프로토콜

## 목적과 범위

공통 목표 작성 계약을 실제 MineCLIP → prior → 공식 STEVE-1 행동에 연결한다.
학습된 공통 스킬 또는 RL 개선을 주장하는 실험이 아니다. optimizer update는 0이다.
사람이 작성한 proposal에 미래 LLM과 같은 compiler/검증을 적용한다. 유료 API는 호출하지 않는다.
완료한 P1 원본 조건·결과는 변경하거나 재실행하지 않는다.

## 세 조건

| 조건 | 정책에 들어가는 문구 | 변경 규칙 |
|---|---|---|
| A `official` | `chop down the tree, gather wood, pick up wood, chop it down, break tree` | 고정 |
| B `purpose-fixed` | `collect wood. Purpose: collect at least one wood log. Expected result: wood is added to the inventory.` | 고정 |
| C `purpose-switch` | `find a tree. Purpose: collect at least one wood log. Expected result: a tree is located.` → B의 문구 | 대상 확인 후 최대 1회 전환 |

A는 공식 paper_prompts.py 원문을 보존하는 기준군이다. 새로운 작성 계약의 출력으로 위장하지 않는다.
B/C는 일반 compiler로 생성하며, 대상을 바꾸는 규칙을 나무 전용으로 구현하지 않는다.
이 실험의 context/predicate 어댑터만 기존 wood 과업으로 한정한다. 미지원 과업의 실행을 허용하지 않는다.

관측 단계 search/approach/mine/collect와 실행 단계 seeking/gathering을 구분한다.
기존 server wood_v3 센서로 판정한 비탐색 단계가 3회 연속 확인되면 gathering으로 간다.
초기 상태에서 대상이 이미 확인되면 바로 gathering으로 시작한다. 이후 대상이 사라져도
수집 목적을 유지한다. complete는 즉시 흡수 상태가 되어 새 행동을 차단한다.
이 센서는 privileged server 관측이다. 픽셀만으로 목표 발견을 판정한 조건으로 해석하지 않는다.
나무를 찾았다는 것은 파괴/획득 성공이 아니다. 성공은 서버 인벤토리 원목 증가 ≥1로만 판정한다.

## 고정 변수와 실행

- 개발 시드: 81031001, 81031003, 81031007, 81031009, 81031013. P1 개발 시드 재사용, 최종 평가 아님.
- 각 시드 새 월드, A/B/C 순서를 시드마다 ABC/BCA/CAB/ABC/BCA로 회전. 총 15조건.
- 실행당 최대 제어 wall-clock 180초, startup 600초. 행동당 서버 1tick. 실제 tick 수를 별도 보고.
- 공식 2x STEVE-1 가중치 고정, guidance 6, prior RNG 31031, 정책 sampling RNG=world seed.
- embedding 생성은 게임과 분리. 실제 tokenizer의 BOS/EOS 포함 77token 초과 또는 잘림은 거부.
- RNN은 조건 교체 때 유지하며 실제 dispatch된 행동만 hidden state에 commit.
- 입력은 기존 이미지, 목표 벡터와 RNN 메모리. 좌표·RM stage vector를 actor에 새로 넣지 않는다.
- 보상은 최초 원목 획득 +1. shaping, 시간 페널티, 새로운 학습 없음.
- 중단·기술 오류는 성공/실패 표본에 넣지 않는다. 기술 오류 발생 시 배치를 멈추고 증거를 보존한다.
  성능 실패를 이유로 재시도하지 않는다. 필요할 때만 기술 수정과 별도 amendment를 기록한다.
- 초기 pose/화면, 명령, source snapshot, 공식 모델/문구/vector/token 해시, 서버 상태, PNG와 이벤트 보존.

## 분석과 종료

주 지표는 원목 획득 성공/수량, 성공까지 제어 시간이다. 추론 지연 중앙값/p95,
전체 제어 주기, GPU 메모리, tick 수, 목표 전환, 정체/실패 관측을 함께 기록한다.
A↔B는 공식 문구 대 계약 문구, B↔C는 목적 있는 탐색 조건 추가를 비교한다.
여전히 작은 개발 파일럿이며 여러 문구 요소를 함께 바꾼 A↔B로 특정 단어의 인과 효과를 주장하지 않는다.
5시드 결과로 일반화/학습 효과를 입증하지 않는다. 초기 pose가 같더라도 동적 월드가 완전히 같다고 가정하지 않는다.

별도 seed 81031999에서 10번 실제 transition 후 STOP을 요청해 새 dispatch, 입력 해제와
소유 자식 프로세스 정리를 검사한다. 성능 표본에는 제외한다. 종료 후 STOP을 유지한다.
기존 STOP은 현재 사용자 실험 지시에 따라 해시/내용을 보존한 명시적 재가동 기록 후 한 번만 해제한다.
이후 STOP은 배치가 자동으로 지우지 않는다.
