# 대화 입력과 과업 확정 경계

후속 추가: MyWife를 참고한 의미 어댑터와 별도 대화 콘솔은
[semantic-integration.md](semantic-integration.md)에 정리했다. 아래는 기존 offline-ko-v1
경로의 설명이며, 실제 유료 LLM 호출은 아직 검증하지 않았다.

이 단계는 **offline-ko-v1의 제한된 한국어 명령 문법**을 구현한다. LLM, 일반 대화 생성,
문맥 기억, 다국어/범용 의도 이해, 연구 명세의 전체 semantic parser를 구현한 결과가 아니다.
연구 목표는 유지하며, 추후 LLM의 의미 후보도 별도 검증 뒤 과업 제어기로 전달해야 한다.
현재 `decide(..., candidate=...)`는 직접 문장에서 얻은 명령과 후보가 일치하는지 검증하는
로컬 접점이며, 모델 공급자나 유료 API에 연결되지 않았다.

## 입력

실행 중인 STEVE-1 session의 기존 commands 폴더를 사용한다. 한국어는 UTF-8 JSON으로
기록하며 완성된 임시 파일을 rename한 뒤 읽는다. 다음은 PowerShell 사용 예다.
세션 경로는 실제 실행이 출력한 경로로 지정한다. 이미 종료된 세션은 거부한다.

```powershell
$session='D:\DATA\Desktop\MinecraftRL\results\session-steve1-실제실행ID'
.\scripts\Send-DialogueMessage.ps1 -SessionPath $session -Text '지금 원목 4개 모아줘'
# command_acknowledged의 task.revision=1 확인 후:
.\scripts\Send-DialogueMessage.ps1 -SessionPath $session -Text '목표를 흙 4개로 바꿔줘' -ExpectedRevision 1
# 수정 수락 후 revision=2:
.\scripts\Send-DialogueMessage.ps1 -SessionPath $session -Text '잠시 멈춰줘' -ExpectedRevision 2
.\scripts\Send-DialogueMessage.ps1 -SessionPath $session -Text '다시 계속해줘' -ExpectedRevision 2
.\scripts\Send-DialogueMessage.ps1 -SessionPath $session -Text '작업 취소해줘' -ExpectedRevision 2
```

각 제어는 앞선 acknowledgment와 상태를 확인한 뒤 보낸다. 발행 성공은 실행 수락이 아니다.
revision을 자동으로 추측하지 않는다. task ID는 기본 gather, 다른 ID는 `-TaskId`로 지정한다.
재전송에는 같은 `-MessageId`를 사용한다. 같은 ID/내용은 한 번만 처리하며 같은 ID에 다른
내용을 넣으면 거부한다. 재시작 간 메시지 기억은 제공하지 않는다(현재 세션은 재시작 복구 미지원).

| 입력 | 처리 |
|---|---|
| `[지금 ]원목/나무/흙 N개 모아줘` | 수집 request; N은 정수 1–64 |
| `목표를 원목/나무/흙 N개로 바꿔줘` | modify; 명시한 revision 필요 |
| `잠시 멈춰줘`, `일시 중단해줘` | pause |
| `다시 계속해줘`, `작업 재개해줘` | resume |
| `작업 취소해줘`, `작업 취소해 주세요` | cancel |
| 질문·제안·선호·인용·부정·복합 문장·수량 누락·미지원 목표·단독 ‘응’ | 실행 명령 없음 |

수집 동사는 `모아 주세요`, `수집해줘`, `수집해 주세요`도 허용한다. 수정은 `바꿔 주세요`도
허용한다. 문장 전체가 일치해야 하며, 지원하지 않는 표현을 비슷한 명령으로 추정하지 않는다.
원목/나무는 기존 wood 목표군으로 연결한다. 하나의 요청만 수행하는 세션 제한은 그대로다.

## 권한과 기록

`dialogue_decision`에 원문, message ID, 분류, 이유, 후보 명령이 기록된다.
`eligible`은 해석 경계 통과일 뿐이다. `command_queued` 뒤 worker가 task ID/revision/상태를
검증하고 중립 입력을 전달한 다음 `command_acknowledged`를 기록해야 수락된 것이다.
부적절한 상태나 오래된 revision은 `command_rejected`다. 비실행 문장은 CommandBus의
generation을 바꾸지 않으며, 진행 중인 행동을 중단시키거나 새 목표를 등록하지 않는다.

`needs_context`, `needs_clarification`, `UNRESOLVED`도 명령을 만들지 않는다. 현재 응답 문장
생성 UI는 없으므로 호출자가 이 구조화 결과와 지원 문법을 표시해야 한다. 분류 자체는 제한적
휴리스틱이며 INFORMATION/GAME_KNOWLEDGE 등 명세 전체 분류의 정확도를 주장하지 않는다.
비실행 문장을 모두 정확히 이해했다는 뜻도 아니다.

메시지에 completed, inventory, reward 등의 필드를 넣으면 거부한다. 완료 권위는 실제
인벤토리 관측이다. 자연어 `다 했어`로 완료되지 않는다. 채팅 텍스트는 로컬 results 로그에
남으며 기존 Git 제외 규칙을 적용한다. 이 입력 통로는 신뢰된 로컬 운영자용이며 원격 사용자
인증이나 Minecraft 플레이어별 권한 체계를 제공하지 않는다.

## 검증

```powershell
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation dialogue-scenario -Seed 130363 -Seconds 180
```

이 시나리오는 비실행 문장 6개 → 원목 4개 → 흙 4개 수정 → pause → resume → 인벤토리
완료를 검사한다. 인벤토리/정책 행동은 주입하지 않는다. 기존 5시드 정책 벤치마크와 별도인
단일 기능 통합 시험이며 자연어 성공률 추정이나 인간다움 평가로 사용하지 않는다.
