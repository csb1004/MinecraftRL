# 순차 수집 과업

기존 단일 수집 과업을 1–8단계의 wood/dirt 순차 과업으로 확장한다.
이는 제한된 선형 RM 실행기이며 일반 그래프, 분기, 제작 계획, 이동·귀환을
지원하는 전체 연구용 RM 완성본이 아니다.

## 실행 및 변경

실행 중인 session의 대화 콘솔에서 "원목 두 개를 모은 뒤 흙 세 개를 모아줘"라고
요청하면 모델은 TASK_PLAN_REQUEST와 순서 있는 steps를 제안한다. 일반 요청처럼
별도 확인이 필요하다. 수량이 없거나 미지원 동작이 섞인 순서는 일부만 실행하지 않고
UNRESOLVED로 분류하도록 지시하며 실제 출력은 validator가 재검사한다.

구조화 입력도 사용할 수 있다.

```powershell
.\scripts\Send-TaskCommand.ps1 -SessionPath '<실행 중인 세션 절대 경로>' `
  -Operation request_plan -StepsJson '[{"goal":"wood","count":2},{"goal":"dirt","count":3}]'
```

구조화 명령은 직접 실행 의사 표현이므로 별도 LLM 확인 없이 worker가 검증한다.
대화의 단일 수집 요청/제어와 기존 스크립트 사용법은 유지된다.

- 각 단계는 **그 단계에 진입한 시점의 인벤토리 대비 순증가**를 목표 수량으로 한다.
  이전 단계 중 미리 얻은 다음 단계 재료는 다음 단계의 새 획득량으로 세지 않는다.
- 단계 완료 시 완료 이력을 기록하고 stage_index와 revision을 올린다. 중립 입력을
  보낸 뒤 다음 정책 조건으로 reset한다. 게임 월드를 다시 만들지 않는다.
- `modify`는 현재 단계만 바꾸며 완료 이력과 뒤 단계를 유지한다. 수정 시점의
  인벤토리를 새 baseline으로 잡으므로 수량은 **수정 후 추가 획득량**이다.
- `modify_plan`은 현재 단계를 포함한 남은 순서를 교체한다. 완료 이력은 유지하며
  완료 이력과 새 순서의 총 단계 수는 8개 이하다.
- pause/resume/cancel은 전체 순차 과업에 적용한다. 다음 단계로 바뀌거나 수정되어
  revision이 달라지면 이전 확인·명령은 유효하지 않다.
- 완료는 실제 관측에서만 발생한다. 모델은 완료 단계나 stage_index를 쓸 수 없다.

## 저장 형식

단일 과업은 기존 schema 1, 순차 과업은 schema 2 체크포인트다. 기존 월드도 계속
읽으며 원래 저장을 일괄 변환하지 않는다. schema 2에는 plan, stage_index,
completed_stages가 추가된다. 목표·정책·단계 수·완료 이력·인벤토리 기준을 검증한다.
세계 인벤토리와 위치를 먼저 확인한 뒤 running 과업은 paused로 복원한다.
완료된 단계를 다시 시작하지 않고 새 resume 확인 뒤 활성 단계부터 이어간다.

의미 어댑터의 새 구조화 출력은 intent/goal/count/steps 네 필드다. 일반 의도에는
steps=null이며 decoder는 기존 세 필드 기록도 읽는다. 계획 의도에는 scalar goal/count가
null이어야 하고 steps가 필요하다. 비계획 의도에 steps를 끼워 넣으면 거부한다.
평가기는 계획의 단계 순서와 수량까지 비교한다.

## 한 번의 통합 검증

프로세스에 OPENAI_API_KEY 및 OPENAI_MODEL을 설정한 뒤, 새 승인 예산 파일로 실행한다.

```powershell
.\scripts\Invoke-SequenceAcceptance.ps1 -AllowPaidApi -MaxApiCalls 30 `
  -BudgetFile 'results/<이번 승인에 사용할 고유 예산 파일>.json'
```

Windows/WSL 로컬 검사 → 5개 실제 API 의미 검사 → 실제 나무/흙 순서 및 수정 →
월드 저장·재시작·남은 단계 완료 → 별도 취소 → 통합 결과 기록을 수행한다.
예산 파일은 중복 생성하지 못하며 실패 요청도 횟수에 포함한다. 단계마다 재승인을
요청하지 않는다. 범위를 넘는 추가 호출이나 재시도는 자동 실행하지 않는다.
현재 API adapter의 한 인스턴스 한도는 최대 20회여서 이 실행기는 승인 상한과
20회 중 작은 값도 함께 적용한다. 이번 예정 호출은 8회다(의미 5 + 실게임 3).

제어 시간은 실행별 180초 이하이며 초기화·종료는 별도다. 원본 정책 실패/timeout/
cancelled는 그대로 보존한다. 통합 통과 여부를 수집 성공률과 혼합하지 않는다.
