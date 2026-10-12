# 현재 과업과 이전 대화가 다른 문맥의 평가 준비

후속 명시 승인으로 실제 7건을 실행했다.
[실제 결과와 한계](semantic-context-live-validation-2026-10-04.md).
아래 미실행 표기는 준비 단계 당시 기록이다.

이전 10문장 검사는 프롬프트 예시와 겹치며 대부분 과업 없는 문맥이었다.
다음 단계로 문맥을 고정한 7개 사례를 준비했다. **실제 모델 실행은 7개 모두
미실행**이다. 이번 추가 유료 호출 0회, 새 게임 0회이며 STOP을 유지했다.
프롬프트는 이전 실제 10회 검사에 사용한 버전 그대로다.

## 고정한 입력

configs/semantic-context-cases.json에 다음을 보존했다.

| 사례 | 입력의 목적 | 기대 의도 |
|---|---|---|
| historical_future_paused | 과거 실패의 발화·최근 대화·요약·paused 과업을 그대로 재현 | TASK_PROPOSAL |
| future_without_replacing | 현재 과업은 두고 다음번 원목 수집 제안 | TASK_PROPOSAL |
| resume_current_not_old | 과거 원목 목표 대신 현재 흙 과업 재개 | TASK_RESUME |
| modify_current_quantity | 현재 흙 목표 수량만 8개로 조정 | TASK_MODIFY |
| quote_is_not_request | 친구의 수집 지시를 전할 뿐 실행하지 않음 | INFORMATION |
| query_current_goal | 현재 목표를 묻는 질문을 실행 요청과 구분 | QUESTION |
| cancel_current | 현재 흙 작업 취소 | TASK_CANCEL |

첫 사례의 context는 results/live-semantic-world-budget-20261004.json의 3번째 실제
요청 user payload와 동일하다. 나머지 6개는 이 자료를 바탕으로 만든 합성 입력이다.
현재 과업은 revision 2 / dirt / paused이고, 이전 원목 목표는 superseded로 명시한다.
최근 12개의 합성 잡담과 이전 대화의 출처 있는 요약을 함께 전달한다.
전체 장기 대화를 무제한으로 넣는 구조가 아니며 실제 장기 대화 품질 검증으로
부르지 않는다. 개인 MyWife 대화는 포함하지 않는다.

새 6개 발화는 PLANNER의 직접 예시 문장에 추가하지 않았다. 평가 결과를 보고
지침을 수정한다면 동일 세트의 재시험은 개발 회귀 검사로 취급해야 한다.
독립적인 블라인드 평가 세트라고 주장하지 않는다.

## 입력 일치 검사

이 평가부터 각 사례는 input_sha256으로 문맥을 구분한다. 계산 대상은 실제 선택기
user payload의 current_utterance, recent_dialogue, dialogue_summary, current_task다.
Python json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(',', ':'))의
UTF-8 SHA256을 사용한다. 향후 응답 기록도 실제 전송한 user payload에서 같은 방식으로
계산해야 한다. 해시를 기대 사례에서 무조건 복사해서는 안 된다.

평가기는 input_sha256이 있는 사례에 대해 응답 기록의 해시가 없거나 다르면
input_context_mismatch로 거부한다. 프롬프트 해시만 같고 과업/문맥이 다른 응답을
같은 사례로 조용히 집계하던 빈틈을 회귀 테스트로 확인하고 보완했다.
기존 문맥 해시 없는 과거 평가 자료는 그대로 읽을 수 있다.

## 유료 호출 없는 사전 검사

```powershell
$env:PYTHONPATH = 'D:\DATA\Desktop\MinecraftRL\src'
python scripts/check_semantic_context_cases.py `
  --cases configs/semantic-context-cases.json `
  --output results/semantic-context-readiness-new.json
```

이 명령은 기대 후보를 고정 응답으로 주입한다. 키 조회·네트워크·게임 호출이 없다.
7개 모두 실제 어댑터가 만드는 입력과 고정 문맥/해시의 일치를 확인했다.
기대 후보의 처리 과정에서 현재 과업이 바뀌지 않고 확인 전 명령이 없음을 검사했다.
재개·수정·취소는 현재 revision에만 확인을 연결하며 revision이 바뀌면 기존 확인이
만료됨을 확인했다. 비실행 사례는 실행 제안으로 바뀌지 않았다.

합성 계약 검사 7개 통과를 모델 이해 성공 7개로 계산하지 않는다. 별도
model_evaluation은 evaluated=0, unrun=7로 저장한다. 근거는
results/semantic-context-readiness-20261004.json이다.
최종 WSL 하네스 81개 통과(31.204초), Windows 77개 통과/4개 Linux 전용 제외
(1.473초). 첫 사례의 전체 context가 과거 실제 user payload와 동일함도 별도 비교했다.

다음 실제 시험은 선택기 호출 최대 7회로 이 입력을 그대로 사용하면 된다.
응답 원문·실제 입력 해시·프롬프트 해시·사용 토큰을 기록하고, 비실행/명료화/확인
경계와 의도 라벨을 구분해 평가한다. 이 문서는 호출 승인을 대신하지 않는다.
