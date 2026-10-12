대화 기반 Reward Machine 및 인간형 플레이 정책을 이용한 Minecraft Embodied Agent 연구 명세서

1. 연구 개요

1.1 연구 주제

본 연구는 실제 Minecraft 환경에서 인간 플레이어와 자연어로 지속적으로 상호작용하면서, 대화에서 발생하는 명령·요청·선호·상황 정보를 해석하고, 실행이 필요한 장기 목표를 Reward Machine(RM) 형태로 구조화하여 수행하는 embodied agent의 개발을 목표로 한다.

에이전트는 단순히 자연어 명령을 Minecraft API 명령으로 변환하는 것이 아니라,

- 인간 플레이어와 지속적으로 대화하고,
- 캐릭터성을 유지하며,
- 대화 중 실제로 수행해야 할 목표를 구별하고,
- 복잡한 장기 목표를 명시적인 상태 구조로 변환하며,
- 실제 Minecraft 상황에 따라 계획을 수정하고,
- 인간 플레이 데이터에서 학습한 Minecraft 특유의 행동 방식과 플레이 관습을 활용하여

실제 플레이어와 함께 게임을 하는 동료 에이전트로 동작하는 것을 목표로 한다.

---

2. 핵심 연구 질문

본 연구는 다음 질문들을 중심으로 진행한다.

RQ1. 자연어 대화에서 실제 실행해야 할 목표를 안정적으로 식별할 수 있는가?

모든 대화가 명령은 아니다.

예:

- "다이아 예쁘다." → 일반 대화
- "다이아 집 한번 만들어보고 싶긴 하다." → 선호 또는 잠재적 목표
- "우리 다이아 집 만들까?" → 제안
- "다이아 블록으로 집 지어줘." → 실행 요청
- "아, 그냥 돌로 만들어." → 기존 목표 수정
- "됐어. 그만해." → 목표 취소

따라서 LLM이 대화를 직접 실행 명령으로 변환하는 구조 대신, 별도의 Task Commitment Layer를 사용하여 실행 여부를 결정한다.

---

RQ2. 자연어로 표현된 장기 목표를 Reward Machine으로 구조화하면 장기 과업 수행 안정성을 높일 수 있는가?

일반적인 LLM agent에서는 현재 진행 상태가 언어 context나 내부 memory에 암묵적으로 저장되는 경우가 많다.

본 연구에서는 목표 진행 상태를 Reward Machine으로 명시적으로 표현한다.

예:

Goal:
"철 장비 맞추고 네더 가자."

q0
 ↓ iron_pickaxe acquired
q1
 ↓ iron_armor acquired
q2
 ↓ flint_and_steel acquired
q3
 ↓ nether_portal constructed
q4
 ↓ enter_nether
SUCCESS

이를 통해 에이전트가 현재 무엇을 완료했고 무엇이 남았는지를 명시적으로 추적한다.

---

RQ3. 인간 Minecraft 플레이 데이터를 이용하여 에이전트의 행동을 보다 인간다운 방식으로 만들 수 있는가?

Minecraft에서 인간은 단순히 최단 경로 또는 최적 보상 행동만 수행하지 않는다.

예:

- 블록을 발밑에 설치하며 빠르게 올라가기
- 높은 곳에서 물을 이용해 착지하기
- 광산에서 횃불을 일정한 방향에 설치하기
- 마을 발견 시 상자를 확인하기
- 필요 이상으로 독특한 건축물을 만들기
- 특정 플레이 상황에서 널리 사용되는 관습적 행동을 수행하기

본 연구는 이를 단순한 목표 보상과 분리하여 Human Play Prior로 모델링한다.

---

RQ4. Persona를 대화뿐 아니라 행동 선택에도 반영할 수 있는가?

에이전트의 Persona가 단순한 말투 설정에 머무르지 않고 실제 행동 성향에도 영향을 미치도록 한다.

예:

Persona A
- 신중함 높음
- 전투 선호 낮음
- 탐험 선호 중간

Persona B
- 신중함 낮음
- 전투 선호 높음
- 탐험 선호 높음

동일한 목표라도 서로 다른 행동 전략이 선택될 수 있도록 한다.

---

3. 전체 시스템 구조

                    Human Player
                         │
                  Chat + Interaction
                         │
                         ▼
              ┌─────────────────────┐
              │ Dialogue / Persona  │
              │        LLM          │
              └──────────┬──────────┘
                         │
                 Semantic Interpretation
                         │
                         ▼
              ┌─────────────────────┐
              │ Task Commitment     │
              │ Layer               │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        CHAT         PREFERENCE        TASK
          │              │              │
          ▼              ▼              ▼
      Dialogue         Memory       RM Compiler
                                         │
                                         ▼
                                  RM Validator
                                         │
                                         ▼
                                  Reward Machine
                                         │
                 ┌───────────────────────┼───────────────────┐
                 │                       │                   │
             World State          Human Play Prior       Persona
                 │                       │                   │
                 └──────────────┬────────┴───────────────────┘
                                ▼
                         High-level Planner
                                │
                           Skill Selection
                                │
                                ▼
                    Skill / RL Controller
                                │
                                ▼
                       Minecraft Environment
                                │
                      observation / events
                                │
                    ┌───────────┴──────────┐
                    ▼                      ▼
               RM Transition          Replanning

---

4. 모듈별 상세 명세

4.1 Dialogue / Persona LLM

목적

인간 플레이어와 자연스럽게 대화하고, 대화 내용을 의미적으로 해석한다.

입력

- 플레이어 채팅
- 최근 대화 context
- 현재 게임 상황
- 에이전트 Persona
- 사용자와 관련된 장기 memory
- 현재 수행 중인 목표

출력

구조화된 semantic representation을 생성한다.

예:

{
  "utterance_type": "request",
  "target": "diamond",
  "action": "collect",
  "quantity": 3,
  "constraints": [],
  "confidence": 0.94
}

추가 역할

- 자연스러운 채팅
- 질문에 답변
- 현재 행동 설명
- 실패 상황 설명
- 도움 요청
- 제안
- 농담 및 Persona 유지

단, LLM 자체는 최종적인 task 등록 권한을 갖지 않는다.

---

4.2 Task Commitment Layer

목적

LLM이 해석한 대화가 실제 행동 목표로 등록되어야 하는지를 판단한다.

기본 분류

CHAT
QUESTION
INFORMATION
PREFERENCE
TASK_PROPOSAL
TASK_REQUEST
TASK_MODIFY
TASK_CANCEL
GAME_KNOWLEDGE

필요에 따라 multi-label classification을 허용한다.

예시

"다이아 집 예쁘긴 하겠다."
→ PREFERENCE

"우리 나중에 다이아 집 한번 만들어보자."
→ TASK_PROPOSAL

"지금 다이아 집 만들어줘."
→ TASK_REQUEST

"집 크기는 좀 더 작게 해."
→ TASK_MODIFY

구현 후보

다음 구조를 우선 고려한다.

LLM Semantic Parser
        +
Small Intent Classifier
        +
Rule-based Validator
        ↓
Task Commitment Decision

LLM 한 모델의 판단에 전적으로 의존하지 않음으로써 오해에 의한 잘못된 장기 행동 실행을 줄인다.

---

4.3 RM Compiler

목적

확정된 자연어 목표를 Reward Machine 구조로 변환한다.

입력 예시

"철 장비를 준비하고 네더로 가자."

중간 Task Representation

{
  "goal": "enter_nether",
  "requirements": [
    "iron_pickaxe",
    "iron_armor",
    "flint_and_steel",
    "nether_portal"
  ]
}

생성 RM 예시

q0
 │ HAS_ITEM(iron_pickaxe)
 ▼
q1
 │ EQUIPMENT_READY(iron_armor)
 ▼
q2
 │ HAS_ITEM(flint_and_steel)
 ▼
q3
 │ PORTAL_EXISTS(nether)
 ▼
q4
 │ ENTER_DIMENSION(nether)
 ▼
SUCCESS

---

4.4 RM Validator

LLM이 자유로운 자연어를 RM으로 변환할 경우 존재하지 않는 이벤트 또는 논리적으로 불가능한 조건을 생성할 가능성이 있다.

따라서 RM은 predefined event vocabulary를 사용한다.

예:

HAS_ITEM(item, count)
ITEM_CRAFTED(item)
BLOCK_MINED(block, count)
BLOCK_PLACED(block)
ENTITY_FOUND(entity)
ENTITY_KILLED(entity)
LOCATION_REACHED(location)
DIMENSION_ENTERED(type)
HEALTH_BELOW(value)
PLAYER_DISTANCE_BELOW(value)
TIME_STATE(day/night)
STRUCTURE_FOUND(type)
STRUCTURE_BUILT(type)

RM Validator는 다음을 검사한다.

- 존재하지 않는 transition condition
- unreachable state
- 동일 조건의 모순
- 종료 state 부재
- 무한 loop 가능성
- Minecraft mechanics 위반
- 현재 환경에서 수행 불가능한 goal

---

4.5 Reward Machine Runtime

RM은 단순 reward generator가 아니라 Agent Execution State Manager의 역할도 수행한다.

관리 정보:

현재 RM state
완료된 subgoal
남은 subgoal
현재 활성 목표
실패한 transition
반복 수행 횟수
조건부 branch

예:

Current task:
"diamond 3개 획득"

Current RM state:
q4 = iron_pickaxe acquired

Next:
LOCATE_DIAMOND

실제 Minecraft world state가 transition 조건을 만족할 때만 상태를 변경한다.

LLM의 언어적 판단만으로 목표를 완료했다고 판단하지 않는다.

---

4.6 High-Level Planner

역할

Reward Machine이 "무엇을 달성해야 하는가"를 정의한다면 Planner는 "어떤 방식으로 달성할 것인가"를 결정한다.

예:

Goal:
IRON_ACQUIRED

가능한 계획:

Plan A
동굴 탐색
→ 철광석 발견
→ 채굴

Plan B
마을 탐색
→ 상자 탐색
→ 철 획득

Plan C
현재 알려진 철 위치로 이동
→ 채굴

Planner 입력:

Current RM State
World State
World Memory
Available Skills
Human Play Prior
Persona
Previous Failures

출력:

Selected Skill Sequence

---

4.7 Replanning

Minecraft 환경은 동적이기 때문에 초기 계획을 끝까지 고정해서는 안 된다.

Replanning 조건 예:

Path blocked
Target resource unavailable
Unexpected enemy
Low health
Tool destroyed
Player changed request
Night started
Required resource discovered elsewhere
Skill repeated failure

예:

기존 계획:

mine iron
→ smelt iron
→ craft armor

하지만 마을 상자에서 iron armor 발견

↓

Planner replanning

craft armor 단계 제거
→ 다음 RM state로 이동

---

5. Skill System

Agent의 실제 Minecraft 행동은 reusable skill 단위로 구성한다.

5.1 Primitive Skills

move
look
jump
attack
use
place_block
break_block
select_item
open_inventory

---

5.2 Mid-Level Skills

navigate_to(location)
follow_player()
escape_enemy()
fight(entity)
mine(block)
collect(item)
craft(item)
smelt(item)
eat()
equip(item)

---

5.3 High-Level Skills

find_tree()
collect_wood()
explore_cave()
mine_iron()
build_shelter()
find_village()
prepare_for_nether()
protect_player()

---

6. Human Play Prior

목적

목표 달성 여부와 별개로, Minecraft 플레이어가 일반적으로 사용하는 행동 패턴을 학습한다.

Human Play Prior는 Reward Machine과 분리한다.

RM
= 무엇을 달성해야 하는가

Human Play Prior
= 사람들이 일반적으로 어떻게 행동하는가

학습 대상

- 이동 패턴
- 시점 조작
- mining 방식
- combat pattern
- building pattern
- exploration strategy
- inventory handling
- Minecraft 특유 행동

데이터

우선적으로 대규모 인간 플레이 trajectory를 활용한다.

가능한 방법:

Human Gameplay
      ↓
Behavior Cloning / VPT-style Pretraining
      ↓
Base Human Policy
      ↓
RL Fine-tuning

---

7. Minecraft Culture / Community Knowledge

Human Play Prior와 문화적 지식을 구별한다.

예를 들어 다음은 직접적인 motor policy라기보다 Minecraft 플레이 문화에 해당한다.

다이아 블록으로 집 짓기
특정 challenge 수행
초반에 나무부터 확보
마을을 발견하면 loot 확인
특정 건축 양식
speedrun에서 일반적으로 사용하는 전략
Minecraft meme
일반적인 multiplayer 관습

이 정보는 별도의 Knowledge Base 또는 RAG 형태로 관리한다.

Minecraft Wiki
Community knowledge
Human gameplay descriptions
Gameplay guides
Strategy documents
       ↓
Minecraft Knowledge Base
       ↓
LLM / Planner

---

8. Memory System

Agent memory를 단일 memory로 관리하지 않고 목적별로 분리한다.

8.1 Episodic Memory

과거 경험.

"지난번 북쪽 동굴에서 Creeper에게 죽음."

---

8.2 World Memory

Minecraft 세계 정보.

Village = (340, 72, -120)
Iron cave = (-82, 31, 400)
Home = (0, 65, 0)

---

8.3 User Preference Memory

플레이어의 게임 내 선호.

건축을 좋아함
자동화 시설을 선호함
특정 건축 스타일 선호

---

8.4 Skill / Experience Memory

과거 행동 성공 및 실패 정보.

특정 cave 접근 경로 성공률
특정 전투 전략 실패
특정 위치의 이동 비용

---

9. Persona System

Persona는 두 부분으로 분리한다.

Persona
├── Dialogue Persona
└── Behavioral Persona

Dialogue Persona

- 말투
- 표현 방식
- 감정 표현
- 농담
- 사회적 행동

Behavioral Persona

예:

cautiousness
curiosity
combativeness
resource_conservatism
exploration_preference
risk_tolerance
helpfulness

Planner의 action/plan preference에 영향을 준다.

단, Persona는 Reward Machine의 목표 자체를 임의로 변경하지 않는다.

---

10. Skill Policy 학습

모든 skill을 처음부터 RL로 학습하지 않는다.

기본 구조:

Human Gameplay
      ↓
Pretrained Policy
      ↓
Skill Specialization
      ↓
RL Fine-tuning

RL은 특히 다음과 같은 환경 적응에 사용한다.

- navigation
- combat
- exploration
- resource search
- dynamic obstacle handling
- player protection

crafting recipe 또는 단순 inventory operation처럼 deterministic한 행동은 API 또는 rule 기반 skill 사용을 허용한다.

---

11. Reward 구성

Skill RL reward와 Reward Machine reward를 분리한다.

예:

R_total
=
R_skill
+
lambda_rm * R_RM
+
lambda_human * R_human

개념적으로:

R_skill

현재 low-level skill의 성공도.

예:

navigation → target distance 감소
combat → enemy damage
mining → target block 획득

R_RM

현재 장기 목표에서의 progress.

subgoal transition 성공
task completion

R_human

인간 플레이 방식과의 유사성 또는 preference.

단, 초기 연구에서는 Human Play Prior를 reward로 직접 넣기보다 pretrained policy prior 또는 planner preference 형태로 사용하는 방법을 우선 검토한다.

---

12. 대화와 행동의 동시 처리

Agent가 과업을 수행하는 동안에도 플레이어와 대화할 수 있어야 한다.

예:

Player:
"지금 뭐해?"

Agent:
"철 찾는 중. 이 동굴 생각보다 깊네."

Player:
"철 말고 그냥 집으로 와."

↓

TASK_CANCEL(current)
+
TASK_REQUEST(return_home)

↓

현재 RM 종료
↓
새 RM 생성
↓
RETURN_HOME

따라서 Dialogue Agent와 Action Agent는 비동기적으로 작동해야 한다.

---

13. Multi-task 관리

한 번에 하나의 목표만 존재한다고 가정하지 않는다.

예:

Primary Task
"철 가져오기"

Persistent Task
"나 따라오기"

Conditional Task
"몬스터 나오면 나 보호하기"

따라서 RM을 하나의 linear sequence로 제한하지 않고,

Active RM Set

형태로 확장할 가능성을 고려한다.

Task priority:

Safety
>
Explicit urgent command
>
Current primary task
>
Background task
>
Autonomous behavior

와 같은 우선순위 시스템을 사용할 수 있다.

---

14. Autonomous Behavior

플레이어의 명시적인 명령이 없을 때도 아무 행동도 하지 않는 agent로 제한하지 않는다.

예:

Player nearby
→ follow / observe

Night
→ seek safety

Hungry
→ acquire/eat food

Idle
→ exploration / social interaction

Player building
→ assist or observe

단, autonomous goal과 사용자가 명시적으로 요청한 goal을 구별한다.

---

15. 연구 평가

단순 task success rate만 사용하지 않는다.

15.1 Task Completion

Success Rate
Partial Completion Rate
Steps to Completion
Time to Completion

---

15.2 Long-Horizon Performance

subgoal 수 증가에 따른 성능을 평가한다.

예:

2-step
5-step
10-step
20-step task

---

15.3 Task Interpretation

자연어 대화에서 task를 얼마나 정확하게 구분하는지 평가한다.

Task Detection Accuracy
False Task Activation Rate
Task Modification Accuracy
Task Cancellation Accuracy

특히 False Task Activation이 중요하다.

---

15.4 RM Evaluation

RM generation validity
Transition correctness
Task coverage
Execution consistency

---

15.5 Human-Likeness

Human gameplay와 비교한다.

가능한 지표:

Human evaluator preference
Trajectory similarity
Action distribution similarity
Behavior classifier score

---

15.6 Social Evaluation

실제 사용자 평가.

"함께 플레이하는 느낌이 드는가?"

"대화를 이해한다고 느끼는가?"

"명령을 지나치게 문자 그대로 해석하지 않는가?"

"행동과 대화가 일관적인가?"

"캐릭터성이 유지되는가?"

---

16. 주요 Ablation Study

연구 기여를 확인하기 위해 다음 비교를 수행한다.

A. Reward Machine 제거

LLM + Planner + Skill
vs
LLM + Planner + RM + Skill

→ 장기 task 안정성 비교.

B. Task Commitment Layer 제거

LLM → RM 직접 생성
vs
LLM → Commitment Layer → RM

→ 잘못된 task activation 비교.

C. Human Play Prior 제거

RL only
vs
Human pretraining + RL

→ 성공률 및 human-likeness 비교.

D. Persona 행동 영향 제거

Dialogue-only Persona
vs
Dialogue + Behavioral Persona

E. Replanning 제거

Fixed plan
vs
Dynamic replanning

---

17. 연구 단계

전체 연구를 한 번에 구현하지 않고 단계적으로 진행한다.

Stage 1 — Agent Infrastructure

Minecraft 연결
World State 추출
Basic Skill Library
Chat interface

---

Stage 2 — Dialogue & Task Understanding

Dialogue LLM
Task Commitment Layer
Task Representation

---

Stage 3 — Reward Machine

RM Compiler
RM Validator
RM Runtime
Minecraft Event Mapping

---

Stage 4 — Planning

Skill Graph
High-level Planning
Replanning

---

Stage 5 — Human Policy

Human gameplay pretraining
Skill policy training
RL fine-tuning

---

Stage 6 — Memory & Persona

World memory
Episodic memory
User preference
Behavioral Persona

---

Stage 7 — Integrated Evaluation

Natural Language
      ↓
Task Commitment
      ↓
Reward Machine
      ↓
Planning
      ↓
Human-like RL Skill
      ↓
Minecraft

전체 시스템 평가.

---

18. LWM 확장

초기 연구 범위에서는 Large World Model을 필수 구성요소로 사용하지 않는다.

추후 다음 구조로 확장할 수 있다.

Planner
   ↓
candidate plans
   ↓
Large World Model
   ↓
Imagined Rollout

Plan A → predicted outcome A
Plan B → predicted outcome B
Plan C → predicted outcome C

   ↓
Plan selection

이를 통해 실제 Minecraft에서 행동하기 전에 미래 결과를 예측할 수 있다.

이는 후속 연구에서 Model-Based RL과 결합할 수 있다.

---

19. 핵심 연구 기여 후보

본 연구가 최종적으로 주장할 수 있는 연구 기여는 실험 결과에 따라 다음과 같이 구성한다.

Contribution 1

자연어 대화에서 실제 수행할 과업을 구별하고 이를 Reward Machine으로 변환하는 대화 기반 task formalization architecture.

Contribution 2

Reward Machine을 reward shaping뿐 아니라 장기 embodied task의 실행 상태 관리 및 검증 구조로 활용하는 방법.

Contribution 3

LLM 기반 high-level reasoning과 인간 플레이로 학습된 low-level Minecraft policy를 결합하는 hierarchical agent architecture.

Contribution 4

언어 Persona와 행동 Persona를 함께 사용하여 동일한 목표에 대해 서로 다른 플레이 스타일을 생성하는 방법.

Contribution 5

Minecraft 플레이어와 지속적으로 상호작용하며 task 생성, 수정, 취소 및 재계획을 수행하는 interactive embodied agent.

---

20. 연구의 최종 형태

본 연구에서 목표로 하는 에이전트는 다음과 같은 상호작용을 수행한다.

Player:
"우리 이제 네더 갈까?"

Agent:
"지금 철 장비는 거의 됐는데 부싯돌이 없어.
그것만 구하면 바로 갈 수 있을 것 같아."

Player:
"그럼 준비해줘."

→ Task Commitment
→ RM 생성
→ Nether preparation 수행


[수행 도중]

Player:
"아 근데 밤이네. 일단 집으로 와."

→ 기존 Task 일시 중단
→ Return Home RM 활성화
→ 귀환


Player:
"아까 하던 거 계속하자."

→ 이전 RM 복구
→ 남은 state부터 수행

최종적으로 본 시스템은 단순한 명령 수행형 Minecraft agent가 아니라,

대화를 통해 목표를 이해하고, 목표의 진행 상태를 명시적으로 관리하며, 인간 플레이 방식에 기반해 실제 Minecraft 세계에서 장기 행동을 수행하는 지속적 embodied companion agent

를 지향한다.
