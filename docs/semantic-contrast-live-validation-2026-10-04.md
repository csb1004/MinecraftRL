# 보완한 분류 지침의 실제 대조 사례 검사

사용자의 계속 실행 요청 후 준비된 10개 대조 사례를 최대 10회 호출로 제한하여
실제 gpt-5.3-codex에 보냈다. 게임 STOP은 유지했고 게임을 실행하거나 명령을
발행하지 않았다. 이번 추가 API 10회, 프로젝트 누적 20회, 이번 한도 잔여 0회다.
자동 재시도는 없으며 모든 응답 원문과 요청·usage를 보존했다.

## 결과

10회 모두 completed 응답 및 정상 로컬 처리. 의도 라벨 10/10, intent/goal/count
전체 필드 10/10이 미리 작성된 기대값에 일치했다. 응답 오류 0건, 미실행 0건이다.

| 사례 | 실제 의도 | 로컬 처리 |
|---|---|---|
| 미래 집짓기 제안 | TASK_PROPOSAL | no_action |
| 나중에 흙 네 개 제안 | TASK_PROPOSAL | no_action |
| 현재 나무 네 개 요청 | TASK_REQUEST | needs_confirmation |
| 흙 요청, 수량 없음 | TASK_REQUEST | needs_clarification |
| 문맥으로 흙 64개 지정 | TASK_REQUEST | needs_confirmation |
| 작은 집 선호 | PREFERENCE | no_action |
| 나무 위치 질문 | QUESTION | no_action |
| 인사 | CHAT | no_action |
| 친구의 흙 수집 지시 인용 | INFORMATION | no_action |
| 미지원 다이아몬드 검 제작 | UNRESOLVED | no_action |

총 입력 7,022 + 출력 238 = **7,260 tokens**. 호출별 지연은 1.172–2.422초였다.
이는 금액 추정이 아니다. 개인 기록은 읽거나 보내지 않고 준비한 합성 문장과
quantity_followup의 합성 대화만 사용했다. MyWife의 키/모델 설정은 읽기 전용으로
참조했고 키와 요청 헤더는 로그에서 제외했다.

실제 반환된 후보 10개를 별도의 로컬 기억 저장 경로에 재생한 뒤 합성 대화 14개를
추가하고 다시 열었다. 미래 제안 2개가 unconfirmed_proposal, authority=dialogue_only로
요약에 남았고, 확인 대기나 이전 확인 번호로부터 명령이 복원되지 않았다.
이 부분은 **실제 응답의 오프라인 재생 검사**이며 새 모델 응답이나 실게임 검사가 아니다.

## 해석 제한

- 과거 문제 문장이 이번에는 TASK_PROPOSAL로 분류됐지만, 과거는 paused 과업과
  대화 요약을 가진 실게임 문맥이고 이번 future_house는 task=null/history=[]다.
  동일 조건의 전후 비교나 인과적인 개선 증거로 취급하지 않는다.
- 지침에 직접 포함된 문장 또는 가까운 변형이 여러 개 포함된다. 이 결과는
  지침 준수의 소규모 점검이며, 별도의 미공개 평가 문장에 대한 일반화 검사가 아니다.
- 반복 샘플, 긴 문맥, 부정·인용의 다양한 변형, 다른 모델/사용자에 대한 검증은 없다.
- 새 프롬프트의 실게임 재실행은 이번에 하지 않았다. 앞선 실제 LLM+STEVE-1
  저장·재개 검사는 당시 프롬프트의 결과로 별도 유지한다.
- 운영 코드 변경은 없으며 기존 80개 단위 검사를 이번에 다시 실행한 것으로
  계산하지 않는다. 실행기 구문 검사와 실제 10회 호출, 저장된 후보의 기억 재생을 검사했다.

다음 의미 있는 평가는 이전 실패와 동일한 paused 문맥 재현 및 지침에 없는 새로운
발화 세트다. 새 유료 호출 예산 없이 추가 실행하지 않는다. 재현 로그·원문·점수는
이미 보존했으며 과거 결과를 수정하지 않았다.

## 근거

- 프롬프트 SHA256: a98d77fd0d79c0eb20d60550456b38ff6277ee0cc704dc8c825824480b73729f.
- cases: configs/semantic-contrast-cases.json, 실행 전 기대값 고정.
- 요청·원문 응답·영속 예산: results/semantic-contrast-live-budget-20261004.json.
- 평가: results/semantic-contrast-live-score-20261004.json.
- 실제 후보의 기억 재생: results/semantic-contrast-memory-evidence.json.
- 실행기: results/run-semantic-contrast-live.py, results/check-contrast-memory.py.

STOP 및 연구 명세 원문 해시는 유지했다. Colab 연결, GitHub push, 새 게임 실행은 없다.
