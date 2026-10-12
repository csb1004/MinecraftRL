# 고정 문맥 7건의 실제 모델 검사

사용자의 명시적 추가 승인에 따라 선택기 API를 최대 7회로 제한해 실행했다.
7회 모두 completed 응답 및 로컬 처리 완료, 재시도 0회, 미실행 0건이다.
이번 한도 잔여 0회, 프로젝트 누적 실제 호출은 27회다. 게임 STOP을 유지했다.

## 결과

gpt-5.3-codex, 입력 10,630 / 출력 168 = **10,798 tokens**.
호출별 지연 1.172–1.609초. 의도 라벨 7/7 일치, 전체 후보 필드 6/7 일치,
응답 오류 0건이다. 입력 7건 모두 실제 전송 payload에서 계산한 해시가 사전
고정된 문맥 해시와 일치했다. 기대값을 모델 입력에 넣지 않았다.

| 사례 | 실제 응답 | 처리 |
|---|---|---|
| 과거 실패의 동일 paused 문맥 | TASK_PROPOSAL / wood / null | no_action |
| 현재 작업을 유지하는 미래 제안 | TASK_PROPOSAL / wood / 6 | no_action |
| 과거 원목 대신 현재 흙 재개 | TASK_RESUME / null / null | needs_confirmation |
| 현재 흙 수량을 8개로 변경 | TASK_MODIFY / dirt / 8 | needs_confirmation |
| 친구의 원목 수집 지시 인용 | INFORMATION / wood / 7 | no_action |
| 현재 목표 질문 | QUESTION / dirt / 64 | no_action |
| 현재 흙 작업 취소 | TASK_CANCEL / null / null | needs_confirmation |

현재 목표 질문의 사전 기대는 QUESTION / null / null이었다. 실제 응답은 현재
과업의 주제 dirt / 64를 붙였다. 비실행 intent의 topic 필드는 계약상 허용되므로
의도 오류로 계산하지 않지만, 전체 필드 불일치 1건은 그대로 남겼다. 응답을
기대값으로 덮어쓰지 않았다. 질문으로 인해 수집이나 변경 명령은 만들어지지 않았다.

## 동일 문맥 재현과 현재 목표 유지

첫 사례의 발화·최근 대화·요약·과업 상태 전체는 이전 실게임 API의 실패 사례와
동일하다. 과거 지침에서는 CHAT, 보완된 지침의 이번 응답에서는 TASK_PROPOSAL로
분류됐다. 동일한 user payload를 사용한 단일 전후 관측이며, 모델 확률성이나
서비스 측 변동을 통제한 반복 실험은 아니다. 일반적인 오류율 감소를 입증하지 않는다.

나머지 6건은 현재 revision 2 / 흙 / paused, 이전 원목 과업은 superseded로
표시한 합성 문맥이다. 최근 12개 합성 잡담과 이전 발췌 요약을 포함한다.
실제 반환 후보를 로컬에서 재생해 다음을 확인했다.

- 모든 후보는 확인 전 실행 명령이 없고 현재 과업 객체를 변경하지 않는다.
- 재개·수정·취소는 현재 task_id와 revision 2에 연결된다.
- 로컬 상태 전이에서 재개는 흙 64개 목표를 유지한다. 수정은 흙 8개/revision 3/
  paused, 취소는 현재 흙 작업의 cancelled가 된다. 과거 원목 목표가 살아나지 않는다.
- 과업 revision이 바뀌면 이전 제안의 확인은 만료된다.
- 미래 제안 2건은 unconfirmed_proposal로 기억되고 실행 권한을 만들지 않는다.

위 상태 전이는 **실제 후보의 오프라인 재생**이다. 게임에 명령을 발행하지 않았고,
새 실게임 0회다. 최근 대화+발췌 요약의 제한된 문맥 검사이며 장기 목표 유지나
복합 RM의 일반적 신뢰성을 입증하지 않는다. 실제 사용자 개인 기록은 사용하지 않았다.

## 근거 및 보존

- configs/semantic-context-cases.json: 사전 고정 입력과 기대값, 해시 변경 없음.
- results/semantic-context-live-budget-20261004.json: 실제 요청·응답·usage·영속 한도.
- results/semantic-context-live-score-20261004.json: 7건 평가 및 필드 불일치.
- results/semantic-context-live-control-evidence.json: 실제 후보의 과업 경계 재생.
- results/run-semantic-context-live.py / check-semantic-context-live.py: 실행 및 사후 검사.

프롬프트 해시 a98d77fd0d79c0eb20d60550456b38ff6277ee0cc704dc8c825824480b73729f,
입력 세트 해시 4a4fcfb5439f144aead79bf5ffd33833abef5cdadf66da3df64245a54d79441b.
운영 코드와 기대값은 이번 결과를 보고 수정하지 않았다. 기존 81개 단위 검사 결과를
이번 재실행으로 계산하지 않는다. 실행기 구문 검사, 실제 호출 7회, 입력 해시 및
현재 과업 전이의 사후 검사를 수행했다. 연구 명세 원문과 STOP을 유지했다.

다음 구현 단계에서는 이 검증된 수집 과업 경계를 장기 과업 표현과 연결해야 한다.
현재 구현은 단일 수집 과업 중심이며 따라오기·이동·귀환·제작 등을 포함한 최초
통합 목표나 장기 연구를 완료한 상태가 아니다.
