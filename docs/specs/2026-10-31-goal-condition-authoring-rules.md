# 목표 조건 작성 규칙 — 사람·LLM 공통

2026-10-31 사용자 합의. 특정 자원이나 과업에 종속된 문구 규칙을 만들지 않는다.
[4단계 명세](2026-10-31-four-stage-research-spec.md)와
[S1 명세](2026-10-31-stage1-rm-conditioned-rl-spec.md)의 후속 작성 계약이다.
원문 연구 명세, 완료한 P1의 문구·데이터·판정을 소급 변경하지 않는다.

**후속 v2:** [LOGO 판별 함수·행동 문구 adapter](2026-10-31-logo-validator-adapter.md)의
`goal_actions.py`에서는 전체 의미/수량/성공 기준은 계약에 보존하고, 정책 입력에는 구체적인
행동·대상·목적 절과 적용 제약을 렌더링한다. 아래 `Purpose:`/`Expected result:` 형식은 v1/P2의
역사적 구현이며 v2 정책 입력에 그 필드 설명을 강제하지 않는다. 사람/LLM 모두 같은 검사를 받는다.

## 반드시 보존할 정보

모든 정책 지시는 **행동 + 대상 + 기대 결과 + 현재 목적**을 포함한다. 현재 목적이 상위 목표와
다르면 상위 목표도 포함하고, 적용되는 제약은 빠뜨리지 않는다. 생성자가 달라도 같은 규칙을 적용한다.

| 상황 | 반드시 명시할 정보 | 금지하는 생략 |
|---|---|---|
| 탐색·정보 수집 | 무엇을 찾거나 알아내는지, 그것이 필요한 목적, 발견 판정 | 대상 없는 탐험·둘러보기로 대체 |
| 이동·접근·복귀 | 도착 대상/장소와 목적, 도착 판정 | 목적지 없는 이동, 근거 없이 특정 방향/좌표 생성 |
| 조작·채집·제작 | 수행 행동, 영향을 받는 대상, 기대 상태 변화 | 행동 입력 자체를 목표 달성으로 간주 |
| 대기·관찰 | 기다리는 사건/조건, 기다리는 목적, 해제/중단 조건 | 무기한 대기 또는 목적 없는 관찰 |
| 복구·재계획 | 회복할 조건, 원래 수행 목적, 복귀/실패 기준 | 원래 목표 삭제, 별도 목적 없는 배회로 자동 전환 |

사용자의 목표 자체가 탐험이라면 탐험할 영역/정보 공백과 관측 가능한 진전 기준을 명시한다.
탐험이라는 단어 자체를 금지하지 않는다. 탐색 대상은 아직 존재가 확인되지 않은 목표 개념일 수 있다.
찾을 대상의 이름을 제공하는 것과 그 대상이 특정 위치에 있다고 주장하는 것을 구분한다.

## 의미와 전이 규칙

1. 상위 목표·현재 목적·수량/제약·완료 판정의 소유자는 RM/런타임이다. 작성자는 이를 바꾸지 않는다.
2. 대명사나 빈 대상만으로 지시하지 않는다. 대상/장소를 현재 관측·사용자 요청의 참조에 연결한다.
3. 기대 결과는 관측 가능한 상태 변화로 표현한다. 실제 판정은 등록된 관측 predicate가 담당하며
   모델의 설명, 화면 유사도, 버튼 입력을 자동 성공 판정으로 쓰지 않는다.
4. 하위 행동이 현재 목적에 어떻게 기여하는지 설명 가능해야 한다. 단순히 목적 문장을 덧붙였다고
   무관한 행동·대상·결과 조합이 올바르게 되는 것은 아니다.
5. RM 관측 상태와 정책 실행 목표를 분리한다. 여러 RM 상태가 같은 condition을 공유해도 된다.
   관측 단계가 바뀌었다는 이유만으로 임베딩을 바꾸지 않는다. 목적·대상·제약이 실질적으로 달라질 때
   전환 후보를 만들고, 관측 안정화/대상 소실/기억 유지 규칙은 런타임에서 별도로 검사한다.
6. 목표 충족/STOP/취소가 새 조건 선택보다 우선한다. 종료 상태에는 정책 조건을 배정하지 않는다.
7. 필요한 정보나 능력이 없으면 `unsupported`와 누락 이유를 반환한다. 기본 탐험 문구나 임의
   실행 스킬로 대체하지 않는다. 관측되지 않은 사실, 좌표, 인벤토리, 지식의 진실성을 만들어내지 않는다.
8. 짧게 쓰되 대상·목적·제약을 잘라내지 않는다. 길면 context를 명시적으로 재작성하고 새 revision을
   발급한다. 실제 임베딩 생성 전에 해당 tokenizer로 토큰 한도를 확인하고 조용한 잘림을 금지한다.

## 작성·검증 인터페이스

구현: `src/minecraftrl/goal_contract.py`. 대상명이나 Minecraft 아이템명을 검사 분기에 넣지 않는다.

| 런타임 소유 context | 의미 |
|---|---|
| `root_goal`, `active_goal`, `constraints` | 보존할 목적과 제약. 정책용 영어 표현을 준비하며 한국어 대화문은 별도 유지 |
| `actions`, `targets` | 이번 작성 요청에서 참조할 행동 표현과 구체적 대상/장소 표현 |
| `outcomes` | 기대 결과 설명과 `predicate_ref`. 실제 predicate 지원/권한은 호출자가 확인 |
| `states`, `terminal_states` | 조건을 배정할 활성 RM 상태 및 행동이 금지된 종료 상태 |

이 context는 LLM 응답에서 가져오지 않는다. 호출자가 요청/관측/허용 registry로 구성한다.
새 대상 개념이나 predicate를 LLM이 제안할 수는 있지만, 기존 요청의 registry를 자기 승인으로
확장할 수 없다. 별도 검토 후 새 context/revision으로 다시 요청한다. registry는 세계의 모든 지식이나
모든 스킬을 미리 열거하는 KB가 아니라 **이번 요청의 작성 범위**다.

사람/LLM이 제출하는 proposal은 동일하다.

```json
{
  "schema": 1,
  "context_sha256": "요청에 포함된 값",
  "status": "candidate",
  "reason": "",
  "conditions": [
    {"id": "seek", "action_ref": "find", "target_ref": "resource", "outcome_ref": "located"}
  ],
  "state_bindings": [
    {"state": "search", "condition_id": "seek"},
    {"state": "inspect", "condition_id": "seek"}
  ]
}
```

컴파일러는 action/target/outcome을 context에서 해석하고 다음 구조로 문구를 만든다.

```text
{action} {target}. Purpose: {active_goal}. Expected result: {outcome}.
Overall goal: {root_goal}. Constraints: {constraints}.
```

상위/현재 목적이 같으면 중복된 Overall goal만 생략하고, 제약이 없으면 Constraints만 생략한다.
기본 512자 상한은 작성 길이 검사이며 MineCLIP 토큰 한도 통과를 뜻하지 않는다.
LLM의 자유 형식 `prompt`, 목표/완료 조건 덮어쓰기, 누락/미등록 참조, 중복 JSON key,
오래된 context hash, 누락/중복/종료 상태 binding, 미사용 condition은 거부한다.
`unsupported`는 이유를 필수로 쓰고 conditions/bindings를 비운다.

## 현재 구현의 검증 범위

작성 요청에는 LLM용 지시와 참조 enum이 포함된 JSON Schema가 생성된다. 실제 API 호출은 하지 않는다.
컴파일 결과는 항상 `execution_authorized=false`, `capability_verified=false`, `tokenizer_checked=false`인
**검토용 후보**다. 문자열에 목적이 있다는 것만으로 의미 적합성·정책 수행 능력이 검증됐다고 주장하지 않는다.

현재 기계 검사는 필드·참조·목적/제약 보존·버전·binding을 담당한다. 다음 단계는 행동-대상-결과의
의미 일치, predicate의 실제 구현/정보 권한, 모델 tokenizer, 미지원 스킬 여부와 실제 행동 효과 검사다.
수량·부정 제약의 문구 포함만으로 정책 준수를 보장하지 않으며 RM/런타임의 독립 판정이 필요하다.
의미 검토를 같은 LLM이 자가 승인한 것만으로 통과시키지 않는다.

P1의 `PROMPTS`/캐시/실행기/분석기는 역사적 비교 조건으로 유지한다. 이 계약 파일은 P1 goal bank가
아니며 그대로 로드해 게임을 시작할 수 없다. 후속 실험에서는 이 공통 작성 경로를 사용하고 새로운
조건·전환·tokenizer 검사와 runtime adapter를 별도 protocol revision에 고정해야 한다.
향후 LLM 연동도 `request → proposal → 동일 compile/검토 → 승인된 임베딩/실행` 경로를 사용한다.

## 로컬 사용

프로젝트 루트에서 `PYTHONPATH=src`를 설정한 격리 환경 또는 기존 Windows Python으로 사용한다.
두 명령 모두 모델·게임·API·학습을 실행하지 않으며 STOP을 해제하지 않는다. 출력 파일이 있으면 덮어쓰지 않는다.

```powershell
python -m minecraftrl.goal_contract request --context context.json --output request.json
python -m minecraftrl.goal_contract compile --context context.json --proposal proposal.json --output candidate.json
```

`tests/test_goal_contract.py`에서 나무/광물/복귀 장소/물 공급원으로 대상을 바꿔 같은 계약을 검증한다.
이는 일반적인 계약 검사이며 그 과업들의 정책 실행 능력이나 새로운 predicate 구현을 주장하지 않는다.
