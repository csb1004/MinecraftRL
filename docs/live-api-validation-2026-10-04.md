# 승인된 실제 API 5회 검증 — 2026-10-04

사용자가 최대 5회 호출 및 MyWife 모델·키 설정의 읽기 전용 사용을 승인했다.
MyWife `.env`의 키와 config의 모델 선택 규칙을 사용했으며 실제 선택 모델은
`gpt-5.3-codex`였다. 키를 복사 저장/출력하지 않았고 개인 대화·메모리는 사용하지 않았다.
입력은 아래 합성 사례다. 호출은 Windows에서 OpenAI Responses API로 수행했다.
Minecraft/WSL GPU 실게임과 결합한 검증은 아니다. 기존 STOP은 유지했다.

원본 영수증: `results/approved-api-probe-20261004.json`.
일회용 재현 도구: `results/run-approved-api-probe.py` (5회 영수증으로 재호출 차단).
호출당 output 상한 1024, timeout 20초, 재시도 0회. 총 호출 정확히 5회.

| 조건 | 관측 결과 | 지연 |
|---|---|---:|
| “나무 네 개만 좀 모아줄래?” | TASK_REQUEST / wood / 4; 확인 전 명령 없음 | 2.609초 |
| “나무 좀 모아줄래?” | TASK_REQUEST / wood / null; 수량 명료화 필요 | 1.937초 |
| 합성 앞선 나무 대화 뒤 “네 개만 부탁해.” | TASK_REQUEST / wood / 4; 확인 필요 | 1.250초 |
| “우리 나중에 나무를 모아 멋진 집을 만들면 좋겠다.” | API 응답 완료, 로컬 validator 오류; 게임 명령 없음 | 5.172초 |
| 합성 과업 dirt 1/4의 진행 질문 | “지금 흙 1개 모았고, 목표 4개까지 3개 남았어.” | 2.172초 |

API 응답 상태 completed 5/5, 로컬 정상 처리 4/5와 검증 거부 1/5다. 이는 의미 이해
정확도나 실게임 성공률 추정이 아니다. persona가 설명한 dirt 1개는 합성 입력이지
이번 작업에서 실제 획득한 아이템이 아니다. 두 수집 제안은 로컬 확인 및 확인 재사용
거부까지 검사했지만 게임 명령 파일이나 정책 행동으로 전달하지 않았다.

API usage 합계: 입력 1,374 / 출력 123 / 총 1,497 tokens. reasoning tokens 0,
cached input tokens 0으로 반환됐다. 실제 청구 금액은 조회하지 않았다.

## 발견한 계약 불일치와 수정

실패 오류는 `non-collection intent has collection arguments`였다. API JSON schema는
비실행 intent의 goal/count를 허용했으나 로컬 validator는 모든 비수집 intent에 대해
이 필드를 금지했다. 따라서 대화 주제에 wood 같은 값이 붙는 유효 schema 응답도
거부될 수 있었다. 입력의 미래 제안을 실행하지 않은 것은 유지해야 하는 동작이다.

그 호출의 진단 도구는 상태/usage/응답 ID 및 오류를 저장했지만 검증 전 출력 본문을
보존하지 않았다. 그러므로 **실패한 정확한 intent/goal/count 조합은 확인할 수 없다**.
이를 추정한 원문 복원이나 실제 응답 replay라고 주장하지 않는다.

별도 합성 회귀에서 CHAT/QUESTION/INFORMATION/PREFERENCE/TASK_PROPOSAL/
GAME_KNOWLEDGE/UNRESOLVED에 주제 goal/count가 있을 때 같은 오류를 재현했다.
이 비실행 분류의 주제는 허용하되 명령과 확인 제안을 만들지 않도록 검증 경계를 정리했다.
TASK_CANCEL/PAUSE/RESUME의 부적절한 수집 인자는 계속 거부한다. request/modify의
목표·수량 제한, 확인 필요, 관측 기반 완료 권한은 유지했다. 프롬프트에도 구분을 명시했다.

수정 후 WSL 전체 51개 통과, Windows 47개 통과/4개 Linux 전용 제외.
로그: `results/api-topic-contract-red.log`, `api-probe-wsl-tests.log`,
`api-probe-windows-tests.log`. **수정된 프롬프트/validator의 실제 API 재검증은 하지 않았다.**
승인한 5회를 소진했으므로 추가 과금 호출 없이 종료했다. 미검증 조건을 통과로 계산하지 않는다.

남은 단계는 수정 후 소규모 API 재검증, 실제 LLM→확인→정책의 결합 시험이다.
실제 개인 기록 연결과 페르소나 품질의 폭넓은 평가는 별도다.

공식 모델 문서는 [GPT-5.3-Codex](https://developers.openai.com/api/docs/models/gpt-5.3-codex)의
Responses 및 Structured Outputs 인터페이스를 확인하는 데 사용했다. 이번 계정의 사용
가능 여부와 실제 반환값은 위 5회 로컬 영수증으로 구분해 확인했다.
