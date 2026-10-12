# MinecraftRL — 정책 실행 및 과업 제어

**2026-10-08 기억층 BC·실제 진단:** STEVE-1의 마지막 기억블록까지 연속 시연으로 학습하고
실제4회 비교했다. 긴 벽은 두 모델 모두 실패, 자연은 각각 원목+1이었다. 기억층 후보는
이동 행동이 생겼지만 우회는 미해결이다. 원래 KL 기준 미통과를 보존한 별도 탐색 실험이며
기본 정책은 유지한다. [학습·실제 결과·다음 단위](docs/temporal-bc-validation-2026-10-08.md).

**2026-10-08 정체 이력·재계획 검증:** 기존 RM/완료 증거를 보존하며 같은 실패·오래된 제안·
미지원 행동을 거부하는 모듈을 추가했다. 실제 정체5건에 가상 제안35건을 대조했다.
실제 LLM/게임 집행은 아직 연결하지 않았으며 우회 행동 학습이 남아 있다.
Windows298통과/17제외·WSL신규32통과. [결과·범위](docs/replanning-contract-validation-2026-10-08.md).

**2026-10-08 그래프 반복·무진전 감지:** 기존14궤적8788전이의 관찰 전용 재생에서 알려진 정체
4개를 검출했고 성공 참고10궤적에는 경보가 없었다. 사이클과 실제 진전을 함께 판정하며,
실시간 재계획·우회 개선은 아직 연결하지 않았다. Windows266통과/17제외·WSL신규31통과.
[결과·오류 교정·적용 범위](docs/graph-stall-validation-2026-10-08.md).

**2026-10-08 원본 정책·기억층 검사:** 원본 STEVE-1과 기존 로컬 정책 모두 자연 월드에서
원목을 얻었지만, 인공 긴 벽에서는 이동 명령 없이 공격을 반복했다. 공식/로컬 추론 경로를
128화면으로 대조했고, 별도8프레임 검사에서 마지막 기억층의 실제 갱신을 약2.28GiB GPU
할당량으로 확인했다. 우회 개선은 미해결이며 정식 연속시연 학습기 연결은 다음 작업이다.
기본 정책 유지. [실제 결과·범위·다음 단위](docs/base-policy-baseline-validation-2026-10-08.md).

**2026-10-08 사전학습 내부 표현층 미세조정:** 교사2037전이를 분리해 STEVE-1의 마지막
표현층+head와 head-only를 학습·실제4회 비교했다. 긴 벽은 둘 다 정체했고 자연은 둘 다
원목을 얻었다. 자연 시작 화면 차이로 속도 개선 주장은 하지 않는다. 기본 정책 유지.
[실제 결과·제약·다음 단위](docs/diverse-correction-validation-2026-10-08.md).

**2026-10-08 교사 BC 실제 비교:** train410/development283으로 BC52회 갱신,
실제 RM·동일 wood embedding 연결과 개발 손실 감소 확인. 긴 벽 전후 모두 실패하며 후보는
중앙 장애물 대신 테두리를 공격했다. 자연은 둘 다 획득했지만 후보가 더 오래 걸렸다.
양 조건 초기 화면 동일·각1회. 기본 정책 유지. [결과·원인 진단](docs/teacher-bc-validation-2026-10-08.md).


**2026-10-08 BC 학습·실제 비교:** 공개 탐색 시연 train621/development414로 BC78회 갱신,
개발 예측 손실은 감소했다. 실제 긴 벽은 부모/BC 모두 실패, 자연 세계는 둘 다 원목 획득.
자연 시작 화면 차이와 소규모 표본으로 개선을 주장하지 않으며 기본 정책을 유지한다.
[실제 결과와 한계](docs/bc-navigation-validation-2026-10-08.md).

**2026-10-08 통로 진입 학습·평가:** 쉬운 훈련 배치2/2획득,622전이로 PPO78회 갱신.
긴 벽 별도 평가는 전후 모두0/2라 기본 정책을 유지했다. 실제6회 중단 뒤 서버 키0·프로세스0 확인.
지연 편차와 문맥 반응 한계도 보존했다. [실행 결과](docs/recovery-gap-validation-2026-10-08.md).
[LLM 상위 플래너 제안](docs/llm-high-level-planner-2026-10-07.md)과 실제 snapshot3건을 준비했으며,
실시간 LLM 실행 연결은 아직 구현하지 않았다.

**2026-10-07 복구 curriculum 수집:** 짧은 벽2세계×2회, 실제6,948전이에서 원목 획득0회였다.
RM 재탐색은 전달됐으나 벽 공격이 계속됐다. 사전 기준으로 PPO0·후속 평가 미실행·기본 정책 유지.
실패 원본/경로 분석과 검사 결과를 보존했다. 다음 단위는 정책 우회 성공 경험 확보다.
[결과·계측 한계](docs/recovery-curriculum-validation-2026-10-07.md), [남은 작업](docs/next-task-units.md).

**2026-10-07 접근 복구 진단:** v4 정체 감지·재탐색 입력 연결은 실제 검증했지만,
v3/v4 모두 우회 시험장에서180초 미획득했다. 고정features의 문맥 교체가 행동 분포를 거의
바꾸지 못함을 확인했다. 기본 정책 유지·추가 학습0; 다음 단위는 별도 세계의 복구 행동 학습이다.
[결과와 한계](docs/approach-recovery-validation-2026-10-07.md).

**2026-10-07 실제 채굴 대상 교정:** 원목을 캐는데 RM이 접근으로 남는 원인을 서버 채굴 좌표로
확인하고 `wood-potential-v3`에 연결했다. 실제 채굴296관측에서 교정 좌표/단계 불일치0,
원목 획득2회·실제 사망 종료1회 감사 완료. 추가 학습0·기본 정책 유지.
[결과와 한계](docs/wood-target-validation-2026-10-07.md), [다음 작업 단위](docs/next-task-units.md).

**2026-10-07 원목 후속 검증·사망 경계 수정:** 파괴/우회 비용과 정책 선택, 자연 세계5개 전후 비교를 실행했다.
부모4/5·v2 5/5 획득이지만 반복1회와 시작 픽셀 편차로 학습 효과를 단정하지 않는다.
실제 사망 시 자동 재생성되어 종료가 누락되는 결함을 별도 JAR에서 수정하고,
noop·학습 정책 사망 및 정상 원목 획득을 재검증했다. 추가 학습0·기본 정책 유지·STOP·잔존 실행 프로세스0.
[결과와 한계](docs/wood-followup-validation-2026-10-07.md), [남은 작업 단위](docs/next-task-units.md).

**2026-10-07 원목 단계 RM·거리·틱 보상 연결:** 탐색/접근/채굴/수집 상태와 거리를 actor/value에 연결하고 실제 수집·개발 평가·시간 제한 종료를 감사했다. 초기/수정 버전 PPO 합계18회 갱신, 실제 원목 획득 확인. 시간 비용의 조기 실패 유인을 발견해 `wood-potential-v2`에서 실제 실패 종료 비용으로 수정했다. Windows192통과/7제외·WSL199통과. 효율 개선은 미입증이며 기본 sparse/정책은 유지한다. 새 실험 플래그는 `scripts/run_rl_pilot.py --reward-mode wood-potential-v2`. [결과·한계](docs/wood-shaping-validation-2026-10-07.md), [보상 정의](docs/wood-shaping-protocol-2026-10-07.md).

**2026-10-07 새 개발세계 30회 비교 완료:** 부모와 γ=.99 대조는 각각 9/10, γ=.999 후보는 8/10 원목 획득. 실패를 포함한 평균 제한 시간 점수는 53.18·43.82·59.23초로 후보는 사전 채택 기준을 통과하지 못했다. 기본 정책을 유지한다. 실제 11,539행동/확률 재현 오차 0, 중단·입력 해제·프로세스 정리 감사 통과. Windows 181개 통과/7개 제외, WSL 188개 통과. 첫 준비의 월드 이름 충돌을 수정하고 원본 실패를 보존했다. 추가 학습 0, 최종 5시드 미실행. [결과·실패 분석·한계](docs/rl-horizon-new-worlds-validation-2026-10-07.md).

아래는 이전 단계의 기록이다.

**2026-10-07 할인·GAE 비교 완료:** 중간 보상 없이9조합 진단,4분기 각304회 학습,부모 포함 실제20회 평가를 완료했다(20/20원목 획득). γ=.999·λ=.95 후보는 한 개발세계에서338·345tick으로 부모448·448tick보다 짧았고 다른 세계는모두86tick이었다. 실제4,866행동/확률을 오차0으로 재현했지만 화면 편차가 초기 분기에 개입해 순수 학습 효과는 미확정이다. 후보만 보존하고 기본 정책·최종5시드는 유지했다. Windows174통과/7제외·WSL181통과. [결과·원인 분석·후속 조건](docs/rl-horizon-validation-2026-10-07.md).

**2026-10-07 첫 추론 준비 경계 검증:** RLPolicy의 첫 합성곱 초기화를 제어 시작 전에 분리했다. 두 고정 세계의 실제 첫 추론은 0.275초·0.195초로 사전 기준 2초 미만을 충족했고 서버 원목 획득도 확인했다. 준비 도중 취소는 월드 reset 전 0행동으로 종료했다. 원시 감사 통과, WSL 178개 통과·Windows 171개 통과/7개 제외. 100초대 초기화 편차의 단일 근본 원인과 정책 학습 개선은 미해결이며, 추가 학습 0·최종 평가 5시드 미실행이다. [결과와 한계](docs/rl-backend-preparation-validation-2026-10-07.md).

**2026-10-06~07 가치함수 교정 비교 완료:** 선형·비선형 교정을 실행하고 새 8세계(획득 7회·시간 초과 1회)를 더해 총 16세계·24개 궤적·4,855행동으로 검증했다. 전체 예측 오차는 줄었지만 초반 구간 기준은 미통과해 actor 학습에 적용하지 않았다. 실제 정책 개선은 미입증이다. WSL 176개 통과, Windows 169개 통과/7개 제외, 원시 로그·원본 보존 감사 완료. [실행 결과와 남은 문제](docs/rl-value-repair-validation-2026-10-06.md), [조건과 판정 기록](docs/rl-value-repair-protocol-2026-10-06.md).

아래는 이전 단계의 기록이다.

**2026-10-06 학습 개선 부재 원인 점검:** 저장된5,598행동/확률 재현으로 학습 경로를 대조하고, 보상 전달·가치 예측·advantage 보정과 검증 관측 편차를 분석했다. 새 게임/학습은 실행하지 않았다. 다음 우선순위는 actor를 고정한 가치함수 진단이다. [원인 근거와 해결 순서](docs/rl-root-cause-2026-10-06.md).

**2026-10-06 확대 RL 배치 완료:** 8시드×2회에서 실제3,149 transition, 출력층·가치함수394회 추가 갱신(누적472회). 검증2시드×전후3회는 전후 각각6/6 성공했지만 행동 중앙값은86→86, 283→361로 개선 근거가 없다. 새 모델은 기본 정책으로 승격하지 않았다. 별도 기술 실패1회와 수정·재실행을 보존했으며 실제35초 종료 진단과 원본 로그 감사도 완료했다. 최종 평가5시드는 미실행이다. [결과·한계](docs/rl-expanded-validation-2026-10-06.md), [프로토콜](docs/rl-expanded-protocol-2026-10-06.md).

아래는 앞선 단계의 기록이다.

**2026-10-05 RL 반복 편차 진단 완료:** 같은 검증 시드에서 추가4회, 기존 포함 전후 각3회 모두 원목 획득. 행동 수는 전211·312·517, 후295·423·406으로 개선 근거가 없다. 저장 행동506개/확률을 정확히 재현한 교차 분석에서 최초4번째 시선 분기는 actor 변경이 아니라 관측 특징 차이로 발생했다. 엔진 내부 화면 차이의 원인은 미확정이며 추가 학습·기본 모델 변경은 없다. [실제 결과·원인 범위](docs/rl-variance-diagnosis-2026-10-05.md).


**2026-10-05 다중 시드 RL 배치 완료:** 실제3개 시드452 transition으로 행동 출력층·가치함수58회 추가 갱신(누적78회). Backbone은 고정했다. 별도2개 시드에서 시작 화면·상태가 전후 일치했고 모두 원목 획득에 성공했지만, 행동 수는86→86과211→295로 개선되지 않았다. 새 체크포인트는 개선판으로 채택하지 않는다. 최종 평가5시드는 미실행이다. [결과·한계](docs/rl-batch-validation-2026-10-05.md), [확정 프로토콜](docs/rl-batch-protocol.md).

아래는 앞선 단계의 기록이다.

**2026-10-05 첫 실제 RL 완료:** STEVE-1 backbone을 고정하고 행동 출력층·가치함수에 PPO 업데이트20회를 수행했다. 실제 원목 채집159 transition을 학습했고, 별도5개 시드의 성공률은 학습 전후 모두5/5였다. 네 시드는 빨라졌고 한 시드는 느려졌다. 초기 화면도4쌍에서 달라 일반 성능 향상은 미입증이다. [학습·평가 결과](docs/rl-pilot-validation-2026-10-05.md), [실험 프로토콜](docs/rl-pilot-protocol.md).

아래 작업대 검증부터는 이전 단계의 기록이다. 당시의 RL 미실행 표시는 위 결과로 갱신되었다.

**2026-10-05 작업대·3×3 GUI 보조:** 자연 채집한 원목3개로 작업대를 제작·설치·열고,
나무 곡괭이1개와 나무 도끼1개를 저장하는 고정5조건이 모두 통과했다.
레시피 JSON 기반 재료 계획·공통 grid 검사·서버 커밋을 사용하며, 저장 청크의 작업대도 확인했다.
커서에 곡괭이를 든 상태의 별도 중단 시험도 물품 회수·저장과 입력 해제를 통과했다.
명시적 GUI 보조이며 RL은 아직 실행하지 않았다. 다음 연구 단계는 작은 채집 과제의
실제 가중치 갱신이다. [실제 결과](docs/workshop-validation-2026-10-05.md),
[첫 RL 단계](docs/first-rl-stage.md), [실험 실행법](docs/workshop-protocol.md).

아래는 이전 단계의 기록이다.

**2026-10-05 레시피북 GUI 보조:** 실제 책 클릭·자동 배치·검사·회수로 막대기4·판자2
보존5/5, 작업대1 보존5/5를 확인했다. 학습 정책을 로드하지 않은 명시적 GUI 보조다.
초기화 오류1회는 프로세스 감시 경합을 수정하고 별도 재시도했다. 저장 NBT10/10 일치,
막대기 획득 후 곡괭이 레시피 해금5/5, WSL143개 통과. 3×3과 대화/RM 제작은 미지원이다.
[실제 결과](docs/recipe-book-validation-2026-10-05.md), [해금·표시·재료 조건](docs/recipe-book-unlocks.md).

**2026-10-05 결과 칸 입력 제한:** Shift 제한을 결과 칸과 GUI 전환으로 좁힌 실제5회에서
막대기4·판자2 저장3/5, 과다 제작0/5를 확인했다. 성공3회는 커서 회수 보조를 사용했고,
작업대/버튼 오제작2회가 남았다. 저장 NBT5/5 일치, WSL137개 검사 통과.
기본 정책과 대화/RM 제작 미지원 범위는 유지한다.
[실제 비교·측정](docs/craft-result-slot-validation-2026-10-05.md).

**2026-10-05 일괄 제작 입력 제한:** 별도 조합 실험에 Shift 제한을 추가해 실제5회에서
막대기4개·판자2개 보존0/5→2/5, 과다 제작2/5→0/5를 확인했다. 두 성공에는 커서 회수
보조가 사용됐다. 미제작1회와 다른 레시피2회가 남아 기본 동작은 바꾸지 않았다.
[수정·실제 비교](docs/craft-shift-guard-validation-2026-10-05.md).

**2026-10-05 독립 조합 검증:** 실제 10회에서 작업대1개 제작·저장4/5, 막대기 제작·저장2/5.
막대기는 모두8개로 과다 제작해 정확4개·판자2개 보존은0/5다. 첫 목표 관측 후 정책 행동0회,
저장 NBT10/10 일치. Shift+클릭의 서버 일괄 제작과 다른 레시피 선택을 별도 실패로 확인했다.
[결과·측정](docs/craft-recipes-validation-2026-10-05.md), [서버 수량 차이 원인](docs/craft-batch-click-cause.md).

이 저장소는 지속적 embodied companion agent 연구의 **공개 정책 실행 가능성 선행 검증**을 수행한다. 연구 범위를 API 봇으로 축소하지 않는다.

**직접 사용:** 프로젝트 PowerShell에서 `.\scripts\Start-MinecraftRL.ps1`로 게임과 대화 콘솔을 함께 시작한다. 같은 슬롯의 월드·목표를 복원하고 `/status`로 상태를 확인하며 `/exit`로 저장 후 종료한다. 기본은 API 호출 없는 제한된 한국어 명령이며 세션 제어 상한은 180초다. [사용법](docs/companion-use.md). **대화/RM의 조합 목표는 아직 미지원**이다.

**2026-10-04 조합 정지·완료 판정 수정:** 조합 실험이 수집 RM 밖에 있었고, 늦은 제작 통계와 클라이언트 예측을 완료 판단에 섞은 문제를 확인했다. 제작→회수→완료 상태와 서버 확정 검사를 추가해 실제 5회 모두 판자 4개를 저장했다. 첫 생성 관측 후 정책 행동 0회, 이전 실패 2회는 명시적 커서 회수 보조 사용. WSL 123개 검사 통과, 추가 API 0회. [원인](docs/craft-rm-root-cause.md), [최종 검증](docs/craft-recovery-validation-2026-10-04.md), [남은 범위](docs/crafting-readiness.md).

**앞선 판자 회수 후 정지 실험:** 5회 중 3회에서 판자 4개를 회수·저장했다. 당시 추가 행동 0회는 인벤토리와 제작 통계가 모두 충족된 witness 이후 기준이었다. 인벤토리에 먼저 4개가 들어온 뒤에는 4회씩 더 실행됐음을 후속 감사에서 확인했다. [당시 결과와 정정](docs/craft-retention-validation-2026-10-04.md).

**앞선 판자 제작 선행 실험:** STEVE-1 5회에서 판자 제작 통계는 5/5 증가했고, 판자 4개를 인벤토리로 회수한 증거는 3/5에서 확인했다. 당시에는 회수 후에도 실행을 계속해 종료 시 판자가 모두 0개였다. [이전 결과](docs/crafting-validation-2026-10-04.md).

**2026-10-04 같은 프로세스 재초기화 실험:** 자연 정체 뒤 한 번만 재개하는 실험기를 추가하고 고정 시드 5회를 실행했다. 모두 재시도 없이 흙을 획득해 실제 복구 시도는 0회이며, **재초기화의 복구 효과는 미검증**이다. 기본 자동 재시도는 켜지 않았다. WSL 101개 검사 통과, 추가 API 0회. [결과·재현·한계](docs/inplace-recovery-validation-2026-10-04.md).

**2026-10-04 줍기 정체 후 복구 비교:** 같은 실패 월드의 복사본에서 5개 시드 × 두 문구, 총 10회를 실행했다. 기존 문구는 5/5, 공식 평가 문구는 4/5로 실제 흙 획득에 성공했다. 기본 문구는 유지한다. 이는 저장·재시작 후의 소규모 비교이며 자동 복구나 일반 줍기 능력의 검증은 아니다. [결과·실패·측정](docs/pickup-comparison-validation-2026-10-04.md), [재현 도구](docs/policy-prompt-profiles.md).

**현재 추가 지원: 수집 정체 감지·복구.** 기본 60초 동안 실제 획득 증가가 없으면 실패 근거와 과업을 보존하고 일시정지한다. 저장·복원·계획 수정·명시적 재개의 제어 경로를 실제 게임에서 확인했다. 이번 재개 후 흙 수집은 실패했고, 60초 무획득 후 다시 멈췄다. 자동 대체 스킬 선택은 아직 지원하지 않는다. [사용법](docs/failure-recovery.md), [성공한 제어 검사와 정책 실패](docs/failure-recovery-validation-2026-10-04.md).

**현재 지원: 순차 수집 RM.** 1–8개 나무/흙 단계, 현재 단계 수정·남은 순서 교체, 중단·재개·취소, 월드와 완료 이력 저장을 지원한다. 실제 STEVE-1에서 나무→흙, 목표 수정, 재시작 후 남은 단계 완료와 취소를 검증했다. 전체 연구용 그래프 RM이나 이동·제작은 아직 지원하지 않는다. [사용법·한 번에 검증](docs/sequence-rm.md), [실제 결과](docs/sequence-rm-validation-2026-10-04.md). 아래는 이전 단계 이력이다.

**2026-10-04 고정 문맥 실제 검사:** 추가 API 7회에서 의도 7/7, 전체 필드 6/7이 기대값에 일치했다. 이전 실패와 동일 문맥에서도 미래 제안으로 분류됐으며, 실제 후보 재생에서 과거 원목 목표 대신 현재 흙 과업을 유지했다. 새 실게임 실행은 없다. [결과·필드 차이·한계](docs/semantic-context-live-validation-2026-10-04.md).

**2026-10-04 문맥 평가 준비:** 과거 실패와 동일한 paused 문맥 1건과 새 표현 6건을 고정하고, 입력 해시를 검사하도록 평가기를 보완했다. 합성 계약 검사 7건 통과이며 실제 모델 7건은 모두 미실행이다. [준비된 입력·검증 경계](docs/semantic-context-evaluation.md).

**2026-10-04 실제 분류 대조 검사:** 보완된 지침으로 실제 API 10회를 실행해 준비한 10개 문장의 의도·필드가 기대값과 일치했다. 미래 제안의 미확정 기억 보존도 실제 후보 재생으로 확인했다. 지침 예시와 겹치는 소규모 검사이며 일반화나 동일 문맥 전후 개선을 입증하지 않는다. [결과·한계](docs/semantic-contrast-live-validation-2026-10-04.md).

**2026-10-04 분류 지침 보완:** 미래 제안과 잡담 등의 의도 경계를 명시하고, 프롬프트 해시를 보존하는 오프라인 응답 평가기 및 대조 사례 10개를 추가했다. 새 지침의 실제 정확도 개선은 미검증이며 추가 API 호출은 없다. [변경·평가 범위](docs/semantic-taxonomy-validation-2026-10-04.md).

**2026-10-04 실제 LLM·게임 통합:** 추가 승인된 API 5회로 수량 명료화·문맥 기반 수집·미래 제안 비실행·복원 후 재개를 실제 STEVE-1과 연결했다. 흙 2개 수집 → 저장·복원 → 총 4개 수집, 확인 전·중단 후 새 정책 행동 0회. 미래 제안은 비실행을 지켰지만 CHAT으로 세부분류한 한계가 있다. [실제 응답·측정·한계](docs/live-semantic-world-validation-2026-10-04.md). 아래 미검증 표기는 각 단계 당시 기록이다.

**2026-10-04 대화·기억·월드 통합:** 고정 합성 모델 응답과 실제 STEVE-1을 연결해 확인 후 흙 2개 획득 → 저장·재시작 → 기억·과업 복원 → 새 확인 후 총 4개 획득을 검증했다. 확인 전·중단 후 새 정책 행동은 0회, WSL 75개 테스트 통과. 추가 유료 호출은 없으며 실제 LLM+게임 결합과 언어 이해 품질은 미검증이다. [실행 결과·재현 명령](docs/semantic-world-validation-2026-10-04.md). 아래는 각 단계 당시의 검증 이력이다.

**2026-10-04 월드 저장·재개:** `session`은 같은 저장 슬롯의 월드와 과업을 다시 불러온다. 최종 실제 STEVE-1 검증에서 흙 2개 획득 → 프로세스 종료 → 인벤토리·위치·플레이어 UUID 유지 → 명시적 재개 → 흙 총 4개 획득·저장을 확인했다. MineRL 종료 저장 누락은 세션 전용 JAR 사본에서 수정했고 시작 UUID도 고정했다. WSL 72개 통과, 추가 API 호출 없음. [사용법·제한](docs/world-persistence.md), [실게임 실패·수정·재검증](docs/world-validation-2026-10-04.md).

**2026-10-04 기억 저장:** 최근 대화 12개와 분류별 발췌 요약을 사용자별로 유지하고, 과업 체크포인트 및 동일 실행의 이벤트 복원을 추가했다. WSL 61개 통과, Windows 57개 통과/4개 제외. 추가 API·실게임 실행은 없으며 게임 월드 자동 복원은 지원하지 않는다. [저장·복원 사용법과 제한](docs/memory-persistence.md). 아래 검증 수치는 각 단계 당시 기록이다.

**2026-10-04 의미 어댑터:** MyWife의 선택기/페르소나 분리와 사용자별 최근 대화·키워드 기억 구조를 참고해 OpenAI Responses 어댑터, 제안 확인·명료화, 별도 대화 콘솔을 구현했다. 승인된 실제 API 5회는 정상 처리 4회/로컬 검증 거부 1회였다. 발견한 계약 불일치를 로컬에서 수정했고 WSL 테스트 51개가 통과했다. **수정 후 API 재시험, 개인 기록 연결, LLM+실게임 통합은 미검증**이다. [실제 API 결과](docs/live-api-validation-2026-10-04.md), [구조·사용법](docs/semantic-integration.md).

**2026-10-04 대화 입력 경계:** 제한된 한국어 수집·수정·중단·재개·취소 문장을 기존 과업 제어기에 연결했다. 실제 STEVE-1에서 원목 요청 → 흙으로 수정 → 중단/재개 → 인벤토리 흙 5개 획득(목표 4개)을 49.9초에 확인했다. 비실행 문장, 중복/상충 메시지 및 별도 실게임 취소도 검사했다. 최신 WSL 하네스 39개 통과. [사용 방법과 제한](docs/dialogue-interface.md), [검증 결과](docs/dialogue-validation-2026-10-04.md). LLM 자유 대화·의미 해석·기억은 아직 연결되지 않았다.

앞선 2026-10-04 구조화 과업 통합에서는 동일 월드 원목 4개 요청 → 흙 4개 수정 → 중단/재개 → 실제 흙 4개 획득을 검증했다. 당시 WSL 하네스 33개 통과. [구조화 명령](docs/task-session.md), [당시 실게임 결과](docs/task-session-validation-2026-10-04.md).

2026-10-03 정책 선행 검증: 지정 Windows 경로에 설치·해시 검증을 완료했다. GTX 1060 3GB/WSL의 공식 MineRL 환경에서 VPT 5회와 STEVE-1 나무·흙 각 5회(각 제어 예산 180초)를 완료했다. STEVE-1은 각 조건 5/5에서 실제 목표 아이템을 획득했다. 두 정책의 초기화·추론·step 중 STOP도 확인했고 당시 하네스 테스트 22개가 통과했다. 초기화 중단은 약 10초 유예 후 강제 종료였으며 즉시 정상 종료로 해석하지 않는다.

결과: [VPT](docs/local-vpt-results-2026-10-03.md), [STEVE-1](docs/local-steve1-results-2026-10-03.md), [실제 중단 시험](docs/local-stop-results-2026-10-03.md), [로컬 설치·검증](docs/local-validation-2026-10-03.md). 이전 클라우드 전달 기록은 `docs/validation-report.md`로 구분한다. 최신 Bukkit 호환성, 인간다움, 장기 연구 목표 달성을 입증한 결과는 아니다.

실험 종료 후 프로젝트 관련 Python/Java 잔존 0개를 확인했으며 `results/STOP`을 유지한다. 재실행 시 직전 종료 결과를 검토하고 STOP을 해당 결과 폴더에 보존 이동해 명시적으로 재무장한다. 현재 PC의 환경은 이미 설치돼 있으므로 아래 setup 절차는 새 설치용이다.

## Windows에 설치

전달 ZIP을 임시 폴더에 풀고, 압축을 푼 폴더에서 PowerShell로 실행한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install-MinecraftRL.ps1
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
.\scripts\Test-LocalEnvironment.ps1
```

`ExecutionPolicy Bypass`는 해당 프로세스만 대상으로 하며 시스템 실행 정책을 변경하지 않는다. 설치 대상은 위 경로로 고정돼 있다. 설치기는 SHA256을 검사하고 다른 내용의 기존 파일은 덮어쓰지 않는다. 원본 프로젝트에 자동 Git 초기화/커밋/푸시는 하지 않는다.

## VPT foundation 1x부터 실행

Windows Git과 WSL Ubuntu-24.04가 필요하다. WSLg의 DISPLAY/렌더링과 NVIDIA GPU 전달이 실제로 작동해야 한다. 사용자 보고 사양만으로 준비 완료라고 간주하지 않는다.

```powershell
.\scripts\Get-OfficialSources.ps1 -Policy vpt
.\scripts\Invoke-PolicyProbe.ps1 -Policy vpt -Operation setup
.\scripts\Invoke-PolicyProbe.ps1 -Policy vpt -Operation download
.\scripts\Invoke-PolicyProbe.ps1 -Policy vpt -Operation preflight
.\scripts\Invoke-PolicyProbe.ps1 -Policy vpt -Operation episode -Seconds 180
.\scripts\Invoke-PolicyProbe.ps1 -Policy vpt -Operation suite
```

setup/download는 명시적으로 실행할 때만 패키지/모델을 받는다. setup은 프로젝트의 `.runtime`에 별도 Python/Java/CUDA 런타임을 만든다. 기존 Windows Python 3.11, 드라이버, 시스템 Java를 변경하지 않는다. conda 환경 후보는 아직 실게임으로 검증한 lock이 아니다. 설치에는 `configs/conda-*-linux-64.candidate.txt`의 dry-run 해석 결과(URL/MD5)를 사용하고, YAML은 그 선택의 입력 명세로 보존한다. 실패 시 `results/setup-*.log`를 보존하고 임의 버전 교체로 넘어가지 않는다. 일부 실패 후 생성된 환경은 재실행 시 자동 덮어쓰기하지 않는다.

로컬 import 검사에서 두 후보의 MKL 충돌을 확인해 MKL 2024.0.0으로 수정했다.
현재 explicit 후보는 이 수정이 적용된 로컬 conda export이며 실제 gameplay 검증은 별도다.
환경 생성 이후 패키지 빌드만 재시도하려면 로그를 검토하고 기존 빌드 복사본을 보존한 뒤
`-Operation resume-setup`을 사용한다. 환경을 삭제/재생성하지 않으며 매번 새 임시 빌드
복사본을 사용한다. 공유 Forge 캐시의 동시 쓰기를 방지하기 위해 setup/resume-setup은
프로젝트 잠금으로 순차 실행한다.

WSL drvfs의 작은 write에서 실제 병목이 관측돼 컴파일 복사본과 기본 Gradle 캐시는
Linux 임시 폴더에 둔다. 프로젝트와 최종 설치 환경은 지정한 D: 경로에 유지한다.
`GRADLE_USER_HOME`을 명시하면 해당 캐시를 사용하며 빌드 위치와 JAR SHA256은 results에
기록한다. 성공한 전체 MCP 빌드의 JAR 해시를 확인해 pip의 metadata/wheel 단계에서
중복 컴파일을 방지한다. 임시 폴더는 WSL 재시작 시 사라질 수 있다.

이전 VPT 공식 기준은 torch 1.9.0이며 이번 후보는 Python 3.9/CUDA 11.1이다. VPT에 자연어 목표를 전달하지 않는다. 실행 결과의 획득량과 이동/시점 조작은 자발적 행동 관찰로만 해석한다.

## STEVE-1은 별도로 실행

```powershell
.\scripts\Get-OfficialSources.ps1 -Policy steve1
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation setup
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation download
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation preflight
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation episode -Condition wood
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation episode -Condition dirt
.\scripts\Invoke-PolicyProbe.ps1 -Policy steve1 -Operation suite
```

STEVE-1 후보는 Python 3.10/PyTorch 2.0.1/CUDA 11.7이다. 공식 2x 모델과 STEVE-1 가중치, MineCLIP, text prior를 사용한다. 지도 학습용 데이터셋과 RL base 2x 가중치는 추론에 불필요하므로 다운로드하지 않는다. 3GB VRAM 수용 여부는 미확인이다. OOM 시 로그를 남기며 API 제어로 대체하지 않는다.

MineCLIP/MineDojo의 gym==0.21 메타데이터와 MineRL gym fork 사이의 충돌은 공식 STEVE-1 문서에도 언급된다. 후보 설치기는 이 두 알려진 충돌만 기록하고 허용한다. 그 외 충돌은 중단한다. `pip check` 전체 통과로 표현하지 않으며 실제 import/환경 reset/rollout으로 추가 검증해야 한다.

## 중단

별도 PowerShell 창에서:

```powershell
.\scripts\Stop-PolicyProbe.ps1
```

`results/STOP`은 래치다. 추론 중 만들어진 행동도 전송 전에 폐기한다. 이미 시작된 환경 step은 완료될 수 있다. 이후 새 정책 행동을 보내지 않고, 별도의 중립 입력 해제 step을 보낸다. 무응답 시 감독 프로세스가 해당 실행에서 시작한 worker/Java 자식 프로세스만 추적해 종료한다. 강제 종료는 입력 해제를 입증하지 못하므로 결과에 미검증으로 남긴다. 기존 Java/게임 프로세스 전체를 종료하는 명령은 없다.

재실행 전에 종료 로그를 확인하고 명시적으로 다시 활성화한다.

```powershell
Remove-Item -LiteralPath .\results\STOP
```

## 결과와 해석

각 실행은 `results/` 아래 새 디렉터리를 만든다. `request.json`, `preflight.json`, `console.log`, `events.jsonl`, `frames/*.png`, `summary.json`, 가용 시 `traceback.txt`와 패키지/GPU 기록을 남긴다.

- 게임 제어 시간: 최대 180초. 초기화 최대 600초와 종료 유예 10초는 별도다.
- 시드 5개: 104729, 130363, 155921, 196613, 262147. 월드 seed를 reset 전에 설정하고 정책 RNG도 seed한다.
- 화면/행동/인벤토리와 기타 관측을 기록한다. 정책에는 화면만 전달한다.
- CUDA 동기화 후 추론 지연 중앙값/95백분위와 전체 제어 주기를 기록한다. 모델 로딩 시간은 추론 통계에 포함하지 않으며 첫 추론은 포함한다.
- PyTorch 최대 allocated/reserved와 nvidia-smi로 표본 측정한 전체 GPU 메모리를 구별한다. 후자는 다른 프로세스를 포함하고 순간 peak를 놓칠 수 있다.
- 인벤토리 시작 대비 최대 순증가와 최종 순증가를 기록한다. 첫 inventory 보유량은 획득으로 세지 않는다. 환경 인벤토리가 없으면 오류다.
- `execution_valid`와 `target_observed`는 다르다. foundation에는 지시 성공률이 없다.
- 실행되지 않은 시드는 성공/실패 분모에 넣지 않는다. 기술 실패율과 지시 과업 실패율을 분리한다.

## 구조

- `docs/research-spec.md`: 사용자 제공 20개 항목 본문
- `docs/design-decisions.md`: 추가 합의와 변경 이유
- `docs/policy-sources.md`: 확인한 공식 출처와 환경 제약
- `docs/experiment-protocol.md`: 실험·중단 검증·측정 해석
- `configs/sources.lock.json`: 공식 소스 커밋 고정
- `configs/models.json`: 모델 출처, 실제 다운로드 후 해시 receipt
- `src/minecraftrl/`: 정책 어댑터, 실행/중단/기록, 사전 점검, 보고
- `notebooks/colab-policy-probe.ipynb`: 동일 실험의 Colab 후보 절차; 미실행

기본 하네스 테스트는 GPU/게임 없이 실행한다. 이것을 정책 성공으로 보고하지 않는다.

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m unittest discover -s tests -v
```

OpenAI API 키는 이번 단계에 필요하지 않다. 향후 통합에서 `OPENAI_MODEL`을 설정 가능하게 유지한다. 다음 연결은 Dialogue → Commitment → RM → 정책/API Skill이며 최신 버킷 서버 버전/클라이언트 연결과 관측 bridge는 별도 검증 대상이다.
