# 의미 해석·페르소나·과거 대화 연결

최신 문맥 평가: 동일 실패 문맥 1건과 새 표현 6건의 실제 API에서 의도 7/7 일치,
선택적 주제 필드를 포함한 전체 후보는 6/7 일치했다. 실제 후보의 현재 과업 결합을
로컬 재생으로 확인했다. [결과](semantic-context-live-validation-2026-10-04.md).

최신 후속 검사: 보완된 지침의 실제 대조 문장 10개가 사전 기대값과 일치했다.
[실제 호출·기억 재생·해석 제한](semantic-contrast-live-validation-2026-10-04.md).
아래 미검증/0건 표기는 각 단계 당시 기록이며 새 실게임 검사는 수행하지 않았다.

후속 분류 지침 보완: 미래 제안과 CHAT의 정의를 명확히 하고 기록 기반 평가기를
추가했다. 새 지침의 실제 모델 응답은 아직 0건이다.
[변경 및 미검증 경계](semantic-taxonomy-validation-2026-10-04.md).

최신 실제 검증: 추가 승인된 API 5회를 실제 STEVE-1과 연결해 수집·저장·복원·문맥
재개를 확인했다. 미래 제안은 비실행을 지켰지만 CHAT으로 분류한 차이가 남는다.
[실제 결과와 한계](live-semantic-world-validation-2026-10-04.md).
아래의 LLM+게임 미검증 표기는 당시 이력이며 장기 기억·개인 기록 검증은 여전히 남아 있다.

2026-10-04 후속 통합: 고정 합성 Responses 응답을 사용해 실제 STEVE-1 수집과
대화 확인, 기억·월드 재시작을 연결했다. 추가 API 호출 없이 흙 2→4개와 중단 경계를
검증했다. [범위·결과·재현](semantic-world-validation-2026-10-04.md).
이 결과는 실제 원격 LLM의 이해 품질 검증을 대체하지 않는다.

최신 추가: 최근 대화와 분류별 발췌 요약을 디스크에 유지하고 현재 과업 상태와 함께
모델에 전달한다. 게임 세션 간 대화 유지, 동일 실행의 과업 기록 복원 및 조회 경계는
[기억 저장 문서](memory-persistence.md)를 따른다. 아래는 최초 구현 및 후속 API 검증 이력이다.

후속 실제 검증: 사용자 승인으로 API 5회를 호출했다. 정상 처리 4회와 검증 거부 1회를
관측했으며 비실행 분류의 주제 필드 처리 불일치를 로컬에서 수정했다. 수정 후 API 재시험은
미실시다. [실제 결과 및 제한](live-api-validation-2026-10-04.md). 아래 미검증 설명은
최초 구현 당시 상태이며, 실제 개인 기록 및 LLM+GPU 결합 시험은 여전히 미검증이다.

## MyWife에서 참고한 구조

원본 프로젝트 `D:/DATA/Desktop/MyWife`를 읽기 전용으로 참고했다.

- `firefly/ai.py`, `prompt_tool_planner.txt`: 페르소나와 분리된 선택기, Responses API,
  별도 응답 생성 경로. MinecraftRL은 의미 후보와 페르소나 응답을 각각 호출한다.
- `firefly/prompts.py`: 최근 사용자/assistant 대화만 선택하고 명령 텍스트를 제외하는 방식,
  별도 페르소나 파일과 사용자 맥락 조합. 최근 모델 입력 12개 제한을 적용했다.
- `firefly/storage.py`, `firefly/brain.py`: 사용자 ID별 기록과 `brain_keywords` 점수 구조.
  `docs/bandi_memory_system.md`의 예전 단기/장기 구조보다 현재 실행 코드의 키워드 구조를
  기준으로 삼았다. 방/투표/뉴스 기록은 Minecraft 대화에 자동 합치지 않는다.

MyWife의 실제 `.env`, 개인 memory JSON, Discord 연결은 읽거나 복사하지 않았다.
코드 import도 하지 않아 봇 시작·환경변수 로드·기억 마이그레이션 부작용이 없다.
특정 캐릭터와 사용자 관계를 자동 복제하지 않는다. 기본 페르소나는 Minecraft 동료이며,
`-PersonaFile`로 사용자가 선택한 파일을 연결할 수 있다.

## 현재 구현

`semantic.py`의 선택기는 intent/goal/count만 반환한다. Responses API의 엄격한
JSON schema를 사용하고 로컬에서 다시 검사한다. 페르소나 답변은 표시용 문자열이며
명령 파서에 전달하지 않는다. 목표 수량 누락은 명료화 질문으로 반환한다. 최근 문맥은
다음 모델 호출에 전달하지만, 장기 기억 자동 갱신이나 MyWife에 역기록하지 않는다.

기존 정확한 한국어 문법은 유료 호출 없이 바로 검증된 명령으로 연결한다. 자유 문장의
모델 실행 제안은 명시적인 제안 ID 확인을 요구한다. 이는 이번 초기 LLM 연결의 검증 모드이며
장기 연구의 최종 commitment 알고리즘이나 의도 분류 정확도 평가를 대체하지 않는다.
확인에는 목표/수량/task ID/revision/status와 이벤트 epoch를 묶고 120초 만료를 둔다.
pause→resume으로 status가 돌아와도 중간 이벤트가 있으면 이전 확인은 만료된다.
새 발화는 앞선 제안을 폐기한다. 단독 ‘응’으로 실행하지 않는다.

worker acknowledgment 전에는 추가 명령을 발행하지 않는다. 파일 발행은 제출일 뿐이며,
worker가 최종 상태 검사·입력 해제 후 수락해야 실행된다. 완료 권위는 기존 실제 관측이다.
콘솔은 별도 Windows 프로세스이므로 API 대기가 WSL 정책 inbox를 막지 않는다.
API 대기 중에도 기존 STOP과 PowerShell 제어 스크립트를 사용할 수 있다.

## 실행 방법

실행 중인 STEVE-1 session의 실제 결과 경로를 사용한다. 기본값은 API 호출 없음이다.

```powershell
.\scripts\Invoke-DialogueConsole.ps1 -SessionPath 'D:\DATA\Desktop\MinecraftRL\results\session-steve1-실제ID'
```

실제 유료 API는 **사용자의 별도 승인 후** `OPENAI_API_KEY`, `OPENAI_MODEL`을 해당 프로세스에
설정하고 `-AllowPaidApi -MaxApiCalls 5`를 지정한다. `.env` 자동 로딩은 하지 않는다.
MyWife 키나 모델을 자동 복사하지 않으며 모델 기본값을 임의로 선택하지 않는다.
선택한 모델의 실제 사용 권한/구조화 출력 지원/응답 품질은 아직 API로 검증하지 않았다.

요청당 output token 상한 1024, 네트워크 대기 제한 20초, 기본 총 호출 수 5회,
자동 재시도 없음. 실패도 호출 예산에 포함하며 잡담은 선택기+페르소나로 최대 2회다.
이는 금액 한도가 아니다. 입력은 현재 발화, 최근 대화, 과업 요약이며 페르소나 응답에는
선택 사용자 profile도 포함한다. `store=false`로 요청한다. 원격 데이터 보존 전반의 보장을
뜻하지 않는다. 키와 원격 오류 본문은 로그에 남기지 않는다.

기존 기록 연결은 별도 `-MemoryFile <선택한 conversation_memory.json> -UserId <대상>`으로
명시한다. 선택 사용자 name/nickname/last_seen, brain_keywords 최대 64개, 최근 대화 최대
12개만 반영한다. 역할은 user/assistant만 허용하고 슬래시 명령은 제외한다.
파일 자체는 10MiB 이하만 읽으며 선택 사용자 외 기록은 모델 입력에 포함하지 않는다.
해당 옵션으로 API를 켜면 선택 기록 일부가 외부 API에 전송되므로 실제 기록 연결 범위를
사용자가 정한 뒤 사용한다. 이번 작업에서는 합성 사용자 데이터만 테스트했다.

콘솔의 `확인 <제안ID>`가 표시된 제안을 확정하고 `/exit`는 콘솔만 닫는다. 게임을 멈추려면
기존 `작업 취소해줘` 또는 STOP을 사용한다. 세션 종료/STOP 상태에서는 명령을 발행하지 않는다.
로그는 세션 폴더의 `dialogue-console-*.jsonl`로 분리하며 Git 제외를 적용한다.

## 검증 상태와 남은 범위

신규 로컬 테스트는 모델 응답 합성 fixture/주입 transport를 사용한다. 실제 API 호출,
실제 개인 기록 연결, 실제 LLM+GPU 정책의 end-to-end 실행은 미검증이다.
이 결과를 앞선 실게임 정책 성공이나 자연어 품질 평가와 혼합하지 않는다.
페르소나 프롬프트가 사실 왜곡을 완전히 방지한다고 보장하지 않는다. 실제 품질 평가는
승인된 API 시험에서 해야 한다. MyWife의 별도 내부 기억 updater, 기억 승격/감쇠,
호감도 변경과 장기 대화 저장은 이번 읽기 전용 연결에 포함하지 않는다.

공식 근거(2026-10-04 확인):
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Responses 전환 안내](https://developers.openai.com/api/docs/guides/migrate-to-responses).
`text.format` JSON schema 및 명시적 응답 상태/거부 처리에 맞춰 구현했다.
