# 의도 분류 지침 보완 및 기록 재평가

후속 실제 호출: 준비된 대조 사례 10개를 실행했다.
[실제 결과 및 일반화 한계](semantic-contrast-live-validation-2026-10-04.md).
아래 미실행 표기는 준비 단계 당시의 이력이다.

직전 실제 API에서 미래 집짓기 제안이 TASK_PROPOSAL 대신 CHAT으로 분류됐다.
실행은 차단됐지만 대화 기억에는 unconfirmed_proposal이 아닌 context로 남았다.
이번에는 선택기 지침을 보완하고 유료 호출 없이 재평가 도구와 대조 사례를 추가했다.
**새 지침의 실제 모델 정확도 개선은 아직 미검증이다.**

## 변경

기존 지침은 미래 제안의 TASK_REQUEST 금지만 명시했다. 이제 TASK_PROPOSAL과
CHAT, 선호, 질문, 사실 보고, 게임 지식, 미해결 발화를 구분하는 정의를 제공한다.
정중한 의문형 요청과 정보 질문, 미래 과업 제안과 현재 실행 지시, 인용과 실행 권한을
대조한다. 현재 과업이 있어도 미래의 변경 제안만으로 수정·재개하지 않도록 명시했다.

모델이 반환한 라벨을 로컬 키워드로 덮어쓰지 않는다. 실제 이전 CHAT 응답이나
이전 기억 파일도 변경하지 않았다. 따라서 모델이 같은 오분류를 반복하면 기억의
세부분류 한계는 남는다. TASK_PROPOSAL이 반환됐을 때 기존의 unconfirmed_proposal
저장과 비실행 경계가 최근 대화에서 밀려난 후 재시작에도 유지되는지 검사했다.

`configs/semantic-contrast-cases.json`에 미래 제안 2개, 현재 요청, 수량 누락,
문맥 기반 수량 보완, 선호, 질문, 인사, 인용, 미지원 요청의 대조 사례 10개를 정의했다.
명시된 expected는 평가 기준이며 모델 응답이나 합성 성공 표본이 아니다.

## 평가 도구

`minecraftrl.semantic_evaluation`은 저장된 Responses 응답만 읽는다. 네트워크,
키 조회, 게임 명령 발행 기능은 없다. 기록마다 case_id, 실제 사용한 프롬프트의
UTF-8 SHA256, response가 필요하다. 중복·알 수 없는 ID와 해시 누락을 거부한다.
출력 파일은 새 파일로만 생성해 기존 근거를 덮어쓰지 않는다.

```powershell
$env:PYTHONPATH = 'D:\DATA\Desktop\MinecraftRL\src'
python -m minecraftrl.semantic_evaluation `
  --cases results/semantic-baseline-cases.json `
  --records results/semantic-baseline-records.json `
  --output results/semantic-baseline-score-new.json
```

- evaluated: 응답 기록이 있는 시도 수. 유효 응답 및 응답 오류를 포함한다.
- unrun: 기록이 없는 사례. 성공/실패로 계산하지 않는다.
- response_errors: 미완료·잘못된 응답 계약. 의미 라벨 불일치와 구분한다.
- label_matches / label_mismatches: 유효 응답의 intent 일치 여부.
- candidate_matches: intent/goal/count 세 필드 모두의 엄격한 일치.
  비실행 응답의 선택적인 주제 필드 차이만으로 의도가 틀렸다고 주장하지 않는다.
- current_prompt_evaluated: 현재 프롬프트 해시와 일치하는 기록 수.
  과거 응답을 재채점해도 새 프롬프트의 실행 결과가 되지 않는다.

## 결과

직전 실제 요청의 원문 문맥을 그대로 보존해 선택기 4건을 재평가했다.
라벨 일치 3건, 불일치 1건, 응답 오류 0건이다. CHAT으로 분류된 미래 제안은
그대로 불일치로 남는다. 페르소나 응답은 선택기 점수에서 제외했다.
현재 프롬프트로 평가된 응답은 0건이다.

새 대조 사례는 10개 모두 unrun이다. 과거 사례와 새 대조 사례의 대화 맥락을
서로 같다고 간주하지 않는다. 프롬프트 SHA256:
`a98d77fd0d79c0eb20d60550456b38ff6277ee0cc704dc8c825824480b73729f`.

근거:

- results/semantic-baseline-cases.json 및 semantic-baseline-records.json:
  기존 실제 입력·출력·출처 해시를 추출한 읽기 전용 재평가 자료.
- results/semantic-baseline-score-final.json: 과거 모델 응답 점수.
- results/semantic-current-unrun-score-final.json: 새 10개 사례의 미실행 상태.
- results/semantic-taxonomy-windows-tests-final.log 및 semantic-taxonomy-wsl-tests-final.log:
  하네스 검사. 모델 이해 정확도 평가와는 별개다.

추가 유료 API 호출 0회, 새 실게임 0회. 기존 STOP과 연구 명세 원문을 유지한다.
최종 WSL 하네스 80개 통과(25.796초), Windows 76개 통과/4개 Linux 전용 제외
(1.540초). 평가기 미구현과 주제 필드/의도 점수 혼동을 먼저 실패하는 검사로
확인한 뒤 구현·수정했다. 이는 모델 오분류의 실제 개선 검사가 아니다.
다음 실제 모델 검증은 별도 승인된 호출 예산으로 대조 사례를 실행하고, 해당 요청의
프롬프트 해시와 원문 응답을 보존한 뒤 같은 평가기로 비교해야 한다.
