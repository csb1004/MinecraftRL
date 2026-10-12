# 실행 기록 — 사용자 승인 계획, 2026-10-03

## 정체 후 복구·벽 정보 제거 비교 — 2026-10-12 (실게임·감사 완료)

[고정 프로토콜](blind-recovery-protocol-2026-10-12.md), [실제 결과·실패 분석](blind-recovery-validation-2026-10-12.md).
개발2월드×정보2조건,4게임4,599전이 완료. **완료 배치는 재실행하지 않는다.**
`results/blind-recovery-20261012/{far,corner}/{fixture,visit_memory}`에 원본/실행 소스를 보존한다.
far/visit_memory는 실제800정체 뒤 좌표·방문·실패 기록만으로 이동하며898번째에 벽을 횡단했다.
실제 목표2개 도달, 외벽 너머 후보 반복/정밀 도착 실패로400행동 종료·원목0.
far/fixture는 국소 이탈했지만 첫 목표0.357669(반경0.35) 미달·횡단0·원목0.
corner/fixture는 root가 이미 횡단한 뒤800정체→이동201행동/두 목표→root406행동/원목+1.
corner/visit_memory는 원래 root가792행동에 원목+2, 복구 미호출이므로 복구 분모에서 제외한다.
조건 전 root 화면/행동이 달라 인과 개선 주장은 하지 않는다. 좌표/root RM은 privileged server,
후보 선택은 규칙 기준선, 실제 모터는 신경망이다. F3-only/LLM 플래닝 성공으로 해석하지 않는다.
4,599전이 저장특징 행동확률0오차·정체/후보 결정 전수 일치·STOP뒤 전달0/키0/소유PID0.
Windows548검사/21skip·최종 판정2개 재검사·WSL관련23 통과. 임시폴더 권한/준비환경/JSON형식
감사 오류는 원본 보존 후 수정했으며 게임 재실행0. 원문·가중치·훈련캐시·원본월드 SHA256 보존.
자동 STOP 유지, API/optimizer/최종5시드 사용0·기본/LLM vocabulary 승격0.
완료 근거 `results/blind-recovery-20261012/final-verification.json`, 작은 요약/그림은 `docs/reports/2026-10-12`.
사용자 승인 GitHub 1차 업로드 준비: README·코드·문서·작은 근거를 정리하고 원본 results/가중치/
외부 코드/런타임/비밀 파일을 제외했다. README 원본은 `README-history-2026-10-10.md`에 보존한다.
다음 단위는 실패한 방향/연결 기억, 정밀 도달 학습, 접근 회복 근거에 따른 root 재개다.

## 상대 경유점 이동 학습·실제 개발 — 2026-10-10 (이번 배치 완료)

[고정 조건](waypoint-learning-protocol-2026-10-10.md), [현재 기록](waypoint-learning-validation-2026-10-10.md).
train1394/dev273 실제 teacher 이동 정답, 별도 actor528갱신 완료. 개발NLL0.9905→0.3684,
목표 반전 TV0.01533/재로드0오차. 원래 root actor/RM·backbone 보존.
전체 영상653프레임/캐시/기억/분포0오차. 첫 CUDA 경로 누락 시도는 보존하고 v2에서 수정 확인.
첫 후보 far902/corner1270전이, 각각 이동400행동·첫 경유점0·원목0으로 종료·전수 감사했다.
완료 배치는 재실행하지 않는다. 실행 디렉터리 `results/waypoint-validation-20261010/{far,corner}`.
far에서 약90도 회전 후 목표가 아직 옆인데 전진하는 실패를 관측했다.
[목표 조건 보강](waypoint-goal-focus-protocol-2026-10-10.md)1056갱신/TV0.8010 완료.
보강 far717/corner1028전이에서 원래 root가 원목1개씩 획득해 이동 skill은 두 곳 모두 미호출이다.
이 둘은 경유점 성공/실패 분모에서 제외한다. [초기 상태 직접 목표 진단](waypoint-initial-diagnostic-protocol-2026-10-10.md)도
별도 `results/waypoint-initial-validation-20261010/{far,corner}`에서 완료했다.
far: 이동337행동/경유점4개→root100행동/원목+1. corner: 이미 충족1개 생략→실제 도달2개,
마지막 목표2.340블록까지 접근했으나400행동 예산 소진/원목0. 보강 정체 후 skill은 여전히 미평가다.
실제6게임4754전이 감사/logp 오차0·STOP뒤 전달0·해제키0·소유PID0. Windows538/21skip·WSL관련13 통과.
원문/부모/원본 시연·월드 보존, 자동 STOP 유지. 기본 정책/LLM 실행 목록 승격0.
완료 근거: `results/waypoint-comparison-20261010/final-verification.json` 및 실제 경로 그림.
완료 배치를 재실행하지 않는다. 다음 단위는 학생의 방향 재정렬·도착 효율 교정이다.
API0/최종5시드 미사용. 실제 게임에는 privileged 좌표와 알려진 fixture 경로를 공급하며,
F3/자연 지형의 자동 경로 계획이나 탐험 일반화로 해석하지 않는다.

## 방문 기억·경로 우선순위·실제 F3 좌표 — 2026-10-10 (관측/계획 입력 단위 완료)

[상세 결과](spatial-exploration-validation-2026-10-10.md), [실행 계획](spatial-exploration-plan-2026-10-10.md).
관측한 구역/방향별 이동 기억, 명명 장소 저장·복원, 탐험/귀환 비용을 구현했다.
controller의 매 전이와 LLM 요청 payload에 지도 요약을 연결했다. 기존 RM/보상은 불변이다.
기존 실제1025전이 재생:32구역/50방향 연결, 이탈23행동의 신규 구역0·탐색200행동의 신규18.
실제 나무 추적/획득0이라는 원본 결과는 유지한다. 지도만으로 통행 가능성이나 정책 성공을 주장하지 않는다.
실제 MineRL UI 진단1회/7전이: F3 열기/닫기, 두 화면에서229.500/201.00000/-785.300 판독·서버 오차0.
기본 한/영 OCR 실패는 보존·거부하고 고정 공식 글꼴의 픽셀 대조로 판독했다. 닫은 화면은 거부했다.
F3 좌표→별도 지도 저장을 확인했으며, 현재 live planner 지도 출처는 privileged server다.
Windows 전체525/21skip, WSL 관련42 통과. 마지막 설명문/전송 assertion 후 양쪽42 재통과.
입력 해제키0·소유PID0·자동 STOP 유지. API/학습/새 정책 평가0, 최종5시드 미사용.
근거: `results/spatial-exploration-20261010/final-verification.json`. 완료 UI/아카이브 배치는 재실행하지 않는다.
상대 경유점 추종 학습, 이동 중 F3 위치 공급, 지속 월드 세션의 자동 지도/집 복원은 다음 과제다.

## 실제 LLM의 신경망 하위 계획 실행 — 2026-10-10 (완료)

[상세 결과](subgoal-live-validation-2026-10-10.md).
사용자 `승인할게`로 v3 준비본의 개발 게임1회/API최대1회를 실행했다.
gpt-5.3-codex 실제1회/입력3611·출력523token/6.812초/재시도0.
800정체→중립/9.634초 계획 대기→LLM의 이탈→탐색→접근→root 계획을 실제 설치했다.
803~825 이탈23행동/3.154블록 완료,826~1025 나무 찾기200행동/목표 추적0으로 예산 소진.
추가 호출 한도에 따라 planning_paused, 원목0, 접근/root 재개는 미실행이다.
실제 LLM 계획의 조건 전환/게임 연결은 확인했으나 과업 성공·우회 일반화는 아니다.
149.391초/1025전이 전수 감사/actor 오차0, 대기·STOP뒤 전송0·키0·소유PID0.
저장된 실패로 만든 archived_review에서 동일 locate_wood 재시도 차단과 기존 계획/RM 보존 확인.
새 API/게임 추가0, optimizer0, 최종5시드 미사용. 실행 소스/원본 실패/가중치/월드 보존.
실행: `results/goal-recovery-eval-104761-20261009T225334338893Z/run`.
완료 근거: `results/subgoal-skills-20261010/live-prepared-v3/live-run/final-verification.json`.
승인1회 소진, 자동 STOP 유지. 다음은 하위 탐색·통로 진입 능력 보강이며 같은 일회 배치를 재실행하지 않는다.

## LLM 선택용 신경망 하위 스킬·추상 상태 확장 — 2026-10-10 (로컬 완료)

[상세 기록](subgoal-skills-validation-2026-10-10.md), [고정 범위](subgoal-skills-protocol-2026-10-10.md).
leave_stalled_area/locate_wood/approach_wood를 실제 조건 벡터·완료 조건·스케줄·실패 이력에 연결했다.
원래 root RM/보상/기억을 유지하며 관측 상태를 LLM 입력과 행동 로그에 기록한다. 새 학습/API0.
첫191전이 GUI 기술 오류를 보존하고 수정했다. 새 고정 계획 게임은1315전이/175.628초/oak_log+3으로 완료했다.
41행동 이동→82행동 나무 찾기→이미 접근 가능하여 접근0행동 생략→원래 수집. GUI 열린 관측16개 처리.
실제 LLM 성과나 접근 스킬 능력 입증으로 해석하지 않는다.
최종 Windows516검사/21skip·WSL521검사/9skip 통과. 실제1506전이(오류191+성공1315) 감사/actor 오차0.
같은803번째 화면·신경망 기억을 고정한 비교에서 세 조건 모두 행동 확률을 바꿨고 실제 조건 특징 오차0이었다.
게임 뒤 행동0 예산 만료를 unrun으로 수정했다. 별도 재현/회귀 검사와 게임 당시/최종 소스를 구분해 보존한다.
완료 근거: `results/subgoal-skills-20261010/final-verification.json`. 자동 STOP 유지, 게임/GPU 비교 종료.
이 로컬 완료 시점에는 새 API 승인 잔여0회였다. v3 준비본은 이후 별도 승인으로 위 실제 LLM 검사에서 사용했다.
새 개발 게임1회·API최대1회/2048출력token/20초/재시도0·제어최대180초의 원래 한도를 지켰다.
실행 상태: `results/subgoal-skills-20261010/game-gui-fixed/work.json` completed=true.
성공 궤적: `results/goal-recovery-eval-104761-20261009T174948864354Z/run`.
기술 오류 궤적: `results/goal-recovery-eval-104761-20261009T174202267743Z/run`.
두 일회 게임과 완료된 감사/조건 비교를 재실행하지 않는다. 새 학습/Colab/기본 정책 승격0·최종5시드 미사용.

## LLM 하위 계획·기존 RM 스케줄 연결과 실제 제안 확인 — 2026-10-10

[상세 결과](rm-planner-validation-2026-10-10.md), [고정 범위](rm-planner-protocol-2026-10-10.md).
기존 두 선택지 복구와 별도로 최대8노드의 하위 목표 DAG, 선행 조건·우선순위·예산·실패 후 처리를
실제 RL controller에 연결했다. 원래 RM/진척/완료 조건·신경망 기억은 보존하며 임시 계획만 추가한다.
실패·부분 진전·이전 계획은 다음 요청에 전달하고, 같은 방법을 문구/노드 ID/우선순위로 반복하지 못한다.
현재 실행 skill은 collect_wood/explore_then_resume이며 미지원 제안은 보존하고 실행을 중지한다.
에피소드 root+하위 계획의 제한된 스케줄러다. 여러 사용자 과업 전체의 RM 관리는 미완료다.

고정 계획 실제1회:701정체→중립/5.002초 대기→704~787탐색→788기존 수집 복귀→1004전이/원목+1.
전수 감사/actor 확률 오차0. 직전 실행과5번째 화면·6번째 행동부터 달라 인과 성능 개선으로 해석하지 않는다.
계획 근거/단계/완료 판정은 privileged server 조건이며, 이 획득은 실제 LLM 계획의 성과가 아니다.

사용자의 `API 1회 검사 승인`으로 준비된 실제 실패2건/partial1건을 gpt-5.3-codex에 전달했다.
장애 원인 확인→검증된 통로/장애물 끝 이동→우회→원래 수집의 새 계획을 생성했다.
API1회/재시도0,9.359초,입력3131/출력584token. 첫 응답은 내부 root 이름에 목표 참조 해시를 넣어
거부됐으며, 정확히 bound된 참조만 정규화하도록 수정한 뒤 같은 응답의 로컬 재검증이 통과했다.
원본/첫 거부 영수증은 보존했다. 미지원3단계이므로 제안은 실행하지 않았고 새 API 호출도 없다.
`results/rm-planner-20261010/api-prepared-v2/paid-probe/`에 근거가 있다. 추가 승인 잔여0회.
Windows 브로커 `--rm-plan`의 제공자/스키마/출력 한도 연결도 수정하고 모의 전송으로 확인했다.

STOP 후 저장 유예 누락과 추론 전 STOP 예외 오분류를 실제 재현·수정했다. 최종300전이 취소는
worker exit2·궤적 완전 저장, STOP뒤 추론/전송0·키0·소유PID0 전수 감사 통과였다.
실제 게임5회는 고정 계획 획득1, 첫 저장 손상1, STOP 조건 미발생1, 취소 오분류1, 최종 취소1로 분리한다.
완전 감사는1004+1083+300=2387전이다. 미발생/기술 오류/제어 취소를 획득 성공률에 섞지 않는다.
최종 Windows499검사/21skip·WSL504검사/9skip 통과. 명세/가중치/월드와 실행 소스 보존 확인은
`results/rm-planner-20261010/final-verification.json`에 기록했다. 자동 STOP 유지, 게임/감사 종료.
새 학습/Colab/기본 정책 승격0·최종5시드 미사용. 완료/실패한 일회 배치는 그대로 두고 재실행하지 않는다.
다음은 LLM이 제안한 목적을 수행할 관측·우회 지점 선택·이동 skill의 실제 검증이다.

## 실패 이력→LLM 입력→중복 방지 일반화 — 2026-10-10

사용자 요청에 따라 [재계획 기억 연결](recovery-memory-validation-2026-10-10.md)을 구현했다.
기존 실제 LLM packet에서 빠졌던 실패 목록을 연결하고, PlanHistory schema2에 실제 방법/인자·전후 관측·
결과 영수증을 저장한다. 같은 상황/방법을 문구·위치 cell·선택 슬롯·세대만 바꿔 재시도하지 못한다.
관측된 목표/자원/도달 가능성 또는 실제 방법이 바뀌면 재검토하며, 원래 RM·실패 이력은 삭제하지 않는다.
프롬프트는 최근 실패4건/실행 결과 최대3건과 생략 건수를 받고 중복 검사는 전체 기록을 쓴다.
새 실행 대안이 없으면 API/행동 없이 중지한다. live는 기존 wood 탐색/중지 vocabulary와 호출1회 한도를 유지한다.

실제 새 고정 응답1회:800정체→803~980탐색178→새 나무 발견/partial→wood 재개→1082시간 제한·원목0.
실제 요청에 실패1건, 종료 이력에 부분 진전 영수증1건을 확인했고1082전이 전수 감사를 통과했다.
대기/STOP뒤 추론/전송0·키0·소유PID0. 게임/감사 완료: `results/recovery-memory-game-20261010/work.json`.
기존 실제2궤적의 다음 요청 재생에서는 실패2건/시도1건을 전달하고 같은 탐색을 차단했다(새게임/API0).
게임 뒤 일반 실패 이벤트의 인자 보존도 재현·수정했다. 실행 소스와 최종 검토 소스를 구분해 보존한다.
최종 Windows456검사/19skip·WSL461검사/9skip 통과. 완료 근거/보존 해시는
`results/recovery-memory-20261010/final-verification.json`에 기록한다. 자동 STOP 유지·소유 게임/학습 프로세스 정리 완료.
추가 유료 API/새 학습/Colab0·기본 정책 승격0·최종5시드 미사용. 완료 배치는 재실행하지 않는다.
현재 능력으로 다양한 우회 방안을 실행하는 하위 skill·다회 LLM 실행·복수 RM 스케줄러는 미완료다.

## 복구 종료/새 목표 인계 수정·실제 후속 검증 완료 — 2026-10-10

[최종 기록](recovery-outcome-fix-validation-2026-10-09.md). 새 후속 게임은400정체→중립/5.015초 대기→
403~478탐색→478새 나무 발견→479부터 wood→742재정체/계획 중지였다. 발견 뒤 추가 탐색0,
원목0·제어86.249초. 재개264전이 모두 approach·reachable0/채굴0, 벽 앞 공격 반복이 남았다.
따라서 제어 오류는 수정했지만 우회/수집 일반화는 해결되지 않았다. 실패2건과 원래 RM/목표는 보존했다.
`results/fresh-target-handoff-game-20261009/work.json` 완료, 같은 디렉터리 `analysis/`에 경로/실패 근거가 있다.
실제742전이의 actor 확률/RM/graph/tick 감사 통과, 대기/STOP뒤 새 추론·전송0·키0·소유PID0.
직전 게임과10번째 픽셀부터 달라 완료 시간/행동 수의 인과 성능 비교는 하지 않는다.
Windows433검사/19skip·WSL438검사/9skip 통과. 실제 게임2회와 KL 대조 단위는 모두 완료했다.
추가 API/Colab0·기본 정책 승격0·최종5시드 미사용. 자동 STOP 유지, 완료 배치는 재실행하지 않는다.
최종 근거는 `results/recovery-outcome-fix-20261009/final-verification.json`에 보존한다.

첫 수정의 로컬 검증1회는800정체→탐색200→원목 후보 발견에 따른 복귀→1327재정체/중지였다.
전체1327전이/신경망1325·중립2의 head 확률오차0, 키0·소유PID0·STOP뒤 전송0 감사 통과.
`results/recovery-outcome-game-20261009/work.json` 완료, 원목0, API0이다.

화면 대조에서 추가 원인을 찾았다.922에 새 원목을 보고도1002까지80틱 더 탐색했다.
서버의 wood_v3 목표는 시선을 돌려도 잠금이 유지되므로 현재 시야와 동일하지 않다.
[후속 조건](fresh-target-handoff-protocol-2026-10-09.md)에 따라 kind0→1의 실제 새 잠금 시
즉시 수집 조건으로 인계한다. 오래된 잠금/과거 접근만으로 예산 끝에 복귀하지 않는다.
원본922경계 재생은923부터 wood 인계,1002의 오래된 잠금은 중지를 확인했다.
이전 실제 실행 소스는 `results/recovery-outcome-fix-20261009/source-after-first-game/`에 보존했다.
후속 실제1회와 자동 전수 감사도 완료했다. `scripts/run_fresh_target_handoff.py`는 소비된 일회 실행기다.

KL1→0.1 대조662갱신·reload오차0·감사 완료. train 첫 회전26.60%→90.65%,
개발 corner0.1369%→1.2379%이나 far0.02268%→0.01352%로 악화했다.
개발 입장 미달로 새 후보4게임은 전부 미실행, 기본 정책 승격0이다.
`results/coverage-anchor-ablation-20261009/comparison.json`, `anchor-comparison.png`,
`results/anchor-bc-comparison-20261009/audit.json`에 근거를 보존했다.
이는 학습 상태의 손실 충돌을 확인했지만 복구 일반화를 해결한 결과는 아니다.

## 복구 결과 소비 수정·KL 대조 착수 기록 — 2026-10-09

사용자 `그래 해결해줘`에 따라 [고정 조건](recovery-outcome-fix-protocol-2026-10-09.md)으로 진행한다.
기존 `assess_attempt`가 실시간 제어에서 소비되지 않아200tick 뒤 무조건 wood로 복귀하던 경로를 수정했다.
현재 target+접근/발견 진전 또는 현재 mine/collect가 있을 때만 재개하고, 그 외에는 시도 결과와
기존 목표/요청을 보존한 뒤 입력을 해제한다. 재개 후 재정체도 새 실패 이력으로 남기고 요청을 늘리지 않는다.
CLI의 정상 계획 중지→worker_failed 오표기와 집계의 기술 실패 오분류도 별도로 재현/수정했다.
변경 전 소스: `results/recovery-outcome-fix-20261009/source-before/`.
이 절 작성 당시 실제 추가 게임0·추가 유료 호출0이었다. 이후 경과는 위 최신 기록을 따른다.

`results/coverage-anchor-ablation-20261009/`에서 KL1→0.1 단일 변수 대조를 착수했다.
같은 masked 부모/자료/seed9612/순서/662갱신이며 개발 궤적은 optimizer에서 제외한다.
진행 로그/프로세스: `results/recovery-outcome-fix-20261009/anchor-training.*.log`, `anchor-process.json`.
완료된 과거 배치를 재실행하지 않는다. 진행 중 GPU 작업과 새 GPU 게임을 겹치지 않는다.
기존 최종 평가5시드·연구 명세·원본·가중치·유료 사용 범위는 유지한다.

## 실시간 LLM1회 인계 완료·원목 미획득 — 2026-10-09

직전 최대1회 유료 연결 질문에 대한 사용자의 `계속해줘`로 해당1회 실행을 승인받았다.
승인 기록: `results/planning-recovery-live-authorization-20261009.json`.
준비 manifest SHA256 `1e1cfea42e3843f7c378825ea3e4c3b7acff187232d017b5b58bac40f0e18aaa`와
현재 소스/가중치/월드를 확인한 뒤 전용 일회 실행기를 시작했다. 기존 완료 배치를 재실행하지 않는다.
[실제 결과](planning-recovery-live-validation-2026-10-09.md): API1회/재시도0, explore_then_resume,
입력988/출력137tokens. 실제800정체→801중립→4.4459초 대기→802현재 검토 수락→803~1002탐색→
1003wood 복귀→1579전이/시간 제한·원목0. 제어/대기/개입 한도를 유지했다.
약11블록 이동했지만 벽 횡단·원목 발견·채굴/회수 단계 모두0이고 복귀 후577전이도 search였다.
LLM 전달·계약 검토는 정상이며, 현재 탐색 skill의 목표 재획득 한계가 남았다. 새 학습0이다.

실제1579전이의 head logp오차0·행동/RM/graph/tick 감사 통과. 대기/STOP뒤 추론/전송0·키0·소유PID0,
워커/브로커/부모exit0. 고정 응답 실행과는 요청 전5번째 화면부터 달라 LLM의 인과 효과로 비교하지 않는다.
실행/영수증: `results/planning-recovery-live-20261009/work.json`, `exchange/api-receipt.json`.
분석/경로: `results/planning-recovery-live-20261009/analysis/report.json`, `analysis/route.png`.
최종 해시·호출 한도·정리 확인: `results/planning-recovery-live-20261009/final-verification.json`.
모든 프로세스 정리·자동 STOP 유지. 완료된 게임/API 일회 배치를 재실행하지 않는다.
이번 추가 승인 잔여0회. 다음 로컬 과제는 복구 종료/목표 재획득 판정의 스케줄러 연결과 하위 skill 보완이다.
복수 RM 스케줄러나 실제 RL 개선이 완료된 것으로 표기하지 않는다.

## 계획 대기/복구·Windows 브로커 로컬 검증 완료 — 2026-10-09

[실행 기록](planning-recovery-validation-2026-10-09.md): 실제5회/4509전이 감사 완료.
첫430행동 원목+1은 정체 없어 인계 미실행. 실제 held_key_count0.0/int0 오류1건은 보존·수정했다.
수정 후 고정 응답은800정체→중립/5.02초 대기/재검토→탐색200tick→원목 복귀→1583에서 원목+1.
별도 대기 중 STOP은401전이/응답·개입0, 완전 저장 뒤 SIGTERM(-15) 정리됐다.
Windows↔WSL 실제 브로커도1요청/5.105초/200tick/wood 복귀를 통과했으나1657전이에서 시간 종료·원목0.
후자는 벽 횡단 뒤 원목 재발견에 실패하고800에서 재정체했다. 앞선 성공 실행은 탐색 전에 이미 벽을
건넜으므로 탐색의 우회 인과 효과로 주장하지 않는다. 두 고정 응답 실행을 LLM 성과/학습 개선으로 세지 않는다.

5회 모두 head logp오차0·공식 행동/RM/graph/서버tick 감사, STOP뒤 새 추론/전송0·해제키0·소유PID0.
Windows 전체415검사(18skip)/WSL420검사(9skip), 추가 준비 검사2개는 각각 통과.
이번 단위 실제 API/새 학습/Colab0, 이전 승인 API2회는 별도 기록이며 잔여 승인0.
모든 일회 배치가 완료됐다. 재실행하지 않는다. 현재 자동 STOP 유지, 게임/브로커 프로세스 정리 완료.
최종 기록: `results/planning-recovery-implementation-20261009/final-verification.json`.

다음 결정은 [실시간 API 최대1회 연결 검사](planning-recovery-live-protocol-2026-10-09.md)의 유료 사용 승인이다.
소스/가중치/월드/프롬프트 고정: `results/planning-recovery-live-prepared-20261009/manifest.json`.
최대1호출/1024출력tokens/20초 대기/재시도0, 게임180초·탐색200tick/30초·기존 RM 보존.
실제 유료 연결은 미실행이다. 장기 후속 과제는 목표 재획득·복구 후 재정체 처리 및 그 학습/일반화다.
복수 RM 우선순위 스케줄러 전체의 구현 완료를 뜻하지 않는다.

## 실제 LLM 복구 제안2회·플래너 대조·완료 우선 수정 — 2026-10-09

[결과](recovery-draft-api-validation-2026-10-09.md): 추가 승인된 API2회/재시도0 완료, 남은 승인0.
기록된 corner 정체 두 건 모두 `explore_then_resume`, 계약2/2통과, 지연5.375/3.454초,
입력1594/출력261 tokens. 기존 원목 RM/목표를 보존했다. 고정 탐색 규칙과 같은 선택이며
LLM의 추가 효과나 실제 복구 성능을 입증하지 않는다. 새게임/행동/학습/Colab0이다.
최초 기록 래퍼의 envelope 비교 오류는 실제 전송0에서 재현·수정했고 원본을 보존했다.
실제 전송 영수증은 `results/recovery-draft-preparation-20261009/paid-probe-transportfix/report.json`이다.

실제 응답을 기존 ReplanGuard에22개 오프라인 조건으로 대조해 stale/STOP/완료/예산/미지원 거부,
응답 재사용 금지·원래 목표/실패 이력 보존을 확인했다. 실시간 generation/중립 해제 증거를 만들지 않았다.
정체 표기와 완료 근거가 모순되면 완료를 우선해 API 요청 전에 거부하도록 추가 수정했다.
Windows386검사(18skip)/WSL 관련40검사 통과. 호출 당시 코드와 수정 후 코드를 구분해 보존했다.
이번 작업 기록은 `results/recovery-draft-followup-20261009/work.json`이다.

다음은 실제 계획 대기의 중립 전이/RM/기억/로그 연결과 실행 세대/현재 상태 재검증이다.
아직 실시간 LLM·자동 우선순위 스케줄러 연결 완료가 아니다. 기존 STOP을 유지하며 새 유료 호출은 없다.
완료된 API/게임 일회 배치를 재실행하지 않는다.

## 일회 목표 조건 전환 실제6회·개입 중 취소·실패 근거 완료 — 2026-10-09

[최종 로컬 결과](goal-recovery-validation-2026-10-09.md): 성능5378전이/6회와 별도 취소722전이/1회.
corner wood0개/1668행동, explore1개/945행동. 탐색군은603정체→709벽 횡단→784상호작용/wood 복귀→945획득.
far wood0개/1668, explore설정1개/567; 자연wood1개/201, explore설정1개/329.
far/explore와 자연 두 조건은 개입0회이며 원래wood 수집이다. 이3회는 복구 성공으로 세지 않는다.
최초 서버/기억/픽셀은 같지만 이후 입력이 달라 세 쌍 모두 조건 전환만의 인과 효과를 주장하지 않는다.
새 PPO/BC 갱신·실제LLM·Colab0, 최종5시드·기본 정책·연구 명세 보존.

[고정 조건](goal-recovery-pilot-protocol-2026-10-09.md): 그래프 정체 뒤 같은 신경망의 목표 임베딩만
최대200tick/1회 변경, mine/collect 또는 예산 종료 뒤 원래 wood 복귀. 상위 RM/기억/행동 생성 보존,
교사/API 대체0, eval 전용/학습 적격성false. 실제 가중치·head 확률 오차0·공식 행동 변환·RM/그래프·tick 감사 통과.
6회 모두 STOP뒤 추론/전송0·해제key0·잔존 소유PID0. 별도 실제 취소도 첫 개입722전이 뒤 새 전송0을 확인했다.
취소 워커는10초 유예 뒤 exit -9로 강제 정리됐고,722개 궤적 완전 저장을 별도 로드로 확인했다.

[실패 판정→LLM/플래너 근거](recovery-failure-evidence-2026-10-09.md): 정체 의심/관측 오류/부분 진전/
상호작용 재개/예산 소진/목표 완료를 분리했다. 실제6경보와 남은 제어 시간, 복귀 후 재정체를 기록했다.
일반 경로 불가능이나 채굴 손상량을 센서 근거 없이 단정하지 않는다. 실제 LLM·실시간 스케줄러·다중 RM 우선순위는 미연결이다.
최종 Windows384검사(18skip)/WSL389검사(9skip) 통과. [기술 오류 수정](goal-recovery-startup-fix-2026-10-09.md)의
기동0step PID 파일 충돌1건은 성능 분모 밖에 보존했다. 완료된 게임은 재실행하지 않았다.

최종 감사: `results/goal-recovery-comparison-20261009/final-verification.json`.
다음 유료 단계는 `results/recovery-draft-preparation-20261009/manifest.json`의 기록된 정체2건 제안 검사다.
기존모델식별자 gpt-5.3-codex, 최대2호출/각1024출력token/20초/재시도0, 실행권한false.
이 절 작성 당시 입력/프롬프트/코드를 준비했고 실제 호출0이었다. 이후 추가 승인으로 위의2회 검사를 완료했다.
이 검사는 과거 관측에 대한 제안 품질 검사이며 실제 게임 자동 집행이 아니다.
현재 STOP을 유지하며 GPU 게임 프로세스는 정리됐다. 작업 기록은 `results/goal-recovery-implementation-20261009/work.json`이다.

## 목표 조건 영향 진단 완료 — 2026-10-09

[결과](goal-conditioning-diagnosis-validation-2026-10-09.md): 같은13시연×7조건, prefix7652/분기728프레임,
동일 latent3쌍 공식 임베딩 오차0·원래 wood의 첫 지표13곳 오차0·공통 기억/원본 hash 감사 통과.
corner 첫 공격 해제+회전 wood0.130~0.140%→explore18.35~18.89%, far0.0165~0.0246%→0.162~0.205%.
개발6쌍 모두 증가, train33쌍 중26쌍 증가. far는 여전히낮고 교사 후속화면의 높은 평균은 자율 실행 증거가 아니다.
학습/게임/LLM/Colab0. 별도 제한된 실제 정체 전환 진단 구현으로 진행했다.

## 복구 시작 상태8개 보강·662갱신 완료, 새 게임 미실행 — 2026-10-09

[수집·학습 결과](onset-coverage-validation-2026-10-09.md): 실제8회/6437전이, 학생 단독1회·교사보조7회.
새 교사 정답1560개를 추가했다. 전체10836특징/원본 학생7639입력 오차0, train2565정답으로662갱신 완료.
문맥 전용7652개·개발619정답 분리, 정답 노출2896회, 저장/reset/reload오차0 및 원본 hash 감사 통과.
train 첫 회전26.60%, 개발 far0.02268%/corner0.13691%로 각0.5% 입장 기준에 미달했다.
개발 첫 NLL21.220→12.410, 계속 NLL1.627→0.921 개선에도 기본 정책 승격 없음. 엄격KL gate도 미통과.
계획된 새4게임은0실행/4미실행으로 기록하고 감사했다. 성공/실패 표본으로 세지 않는다.
학습1504.201초, PyTorch peak allocated2016.194MiB/reserved2306MiB, 오류/OOM0.
[부모 유지 손실 분석](bc-anchor-objective-analysis-2026-10-09.md)과 복구/search 입력 중첩은 추가 원인 후보이며
현재 학습 계약을 소급 변경하지 않는다. 다음 목표 조건 진단으로 진행했다. 실제 PPO/LLM/Colab0.

## 복구 시작 BC 대조 완료 — 2026-10-09

사용자가 실제 결정 사안 전까지 로컬 개선을 계속하도록 승인했다.
[고정 조건](onset-bc-protocol-2026-10-09.md): masked BC 후보에서 동일lr5e-6/각260갱신으로
uniform(기존 교사 window2회) 대 onset(기존1회+첫 정답 집중130회)을 비교한다.
시작 window는 문맥7+진짜 교사 정답1이고, 개발 자료/학생 정답 주입은 금지한다.
KL 참조 특징도 이번 부모의 전체 문맥에서 새로 계산한다. 기존 feature 부모와 혼동하지 않는다.
기본 정책/원문/최종5시드 보존. 실제 게임은 사전 정의 첫 행동/계속 행동/추론 일치 조건을
통과한 후보만 진행한다. 신규 sampling/결합확률 검사5개 Windows·WSL 각각통과,
Windows 전체349검사 중332통과/17환경 제외. `results/onset-bc-focused-20261009/` 및
`results/onset-bc-uniform-20261009/` 각260갱신 완료, 두 supervision 감사 통과·reload오차0.
train 시작4곳 공격 해제+회전은 집중22.8788%/고른 반복0.0745%, 개발2곳은0.00456%/0.00279%이다.
두 조건 모두 사전 진단 입장 기준 통과, 기존 엄격KL gate는 미통과. 기본 정책으로 승격하지 않는다.
전체 영상 추론은 각653프레임 특징·기억·행동확률 오차0으로 통과했다.
두 후보와 공통 부모의 실제 학생6게임/5838전이 감사 완료. 긴 벽 세 정책 모두 미획득,
자연 세 정책 모두 oak_log+1. 자연 집중 후보는 초기10×6픽셀 영역 차이로 시간 인과 비교 무효,
그 밖의 대조는 서버/기억/첫 화면 동일. 모든 STOP 뒤 정책전송0·해제키0·소유PID0.
고른 반복은 통로 높이까지 갔으나 반대편 횡단0, 집중은 통로 높이에 미도달.
[상세 결과·지연·경로·한계](onset-bc-validation-2026-10-09.md).
별도로 마지막 기억층의128개 유효cache만 재생하는 최적화를 준비했다. 작은 공식 모델 CPU5검사 통과,
실제 GPU24쌍 출력·기억·gradient 차이0, window 계산 속도비 중앙값4.52배로 검증 완료했다.
[계산 결과와 적용 범위](bounded-burnin-validation-2026-10-09.md). 현재 대조 학습은 변경·재실행하지 않았다.
[추가 상태 범위 조건](onset-coverage-protocol-2026-10-09.md)에 따른 train8개 월드 준비/NBT 검사 완료.
실제 비교에서도 미해결이어서 수집으로 진행했다. 준비 월드 자체는 실행 표본으로 세지 않는다.
작업 기록은 `results/onset-bc-work-20261009/work.json`이며 완료된 배치를 다시 실행하지 않는다.

## 교정 행동 구간별 예측 분석 완료 — 2026-10-09

[분석 조건](correction-action-diagnosis-protocol-2026-10-09.md),
[원본 인계 경유점 정정](correction-action-diagnosis-clarification-2026-10-09.md).
교정6시연의 실제 경유점/회전/전진 행동 복원과 Windows·WSL 신규8검사가 각각 통과했다.
초기 분석기의 고정z 가정은 원본 handoffs[].waypoints와 달랐으며 v1/오류를
`results/correction-action-diagnosis-precheck-20261009/`에 보존하고 실제 인계 위치를 사용하도록 교정했다.
학생/중립 구간은 계속 문맥 전용이다. 마지막 기억층+표현층+actor를 각각 복원해4961프레임씩
재생하고9922예측을 저장했다. 기존 긴 벽2게임3199전이의 own-policy logprob 오차0,
전체 NLL 재집계 오차 최대4.31e-9, 소스 해시 보존. 결과는 `results/correction-action-diagnosis-20261009/`.
개발 전체NLL1.901921→1.690696 개선의70.41%는 통로 횡단/나무 재접근에서 발생했다.
첫 복구 행동의 정답 방향 회전 확률은 far .809390%→.511039%, corner .000309%→.000235%로
오히려 감소했다. 교사가1~3번 회전한 뒤에는 후보가28~96%를 주지만 교사가 바꾼 화면의 예측이다.
시작 구간 첫8평균73.50%만 보면 첫 행동 병목이 가려진다. 훈련 첫 정답은4/1005개뿐이다.
실제 후보 모서리1143행동의 공격 명령 확률99.858%/실제100%, 이동 실제0이었다.
이전1144와의 차이1은 행동 전/후 위치 분류 기준이다. 학습 실패의 유일한 원인이나 자율 성공은 주장하지 않는다.
새 게임/학습/LLM0, STOP 해제0. 다음 단위는 진짜 교사 시작 정답의 표집/비중과 첫 행동 지표를
분리한 복구 시작 BC이며 아직 미실행이다. 개발 자료/최종5시드/기본 정책/원문 보존.
[상세 결과·그래프·한계](correction-action-diagnosis-validation-2026-10-09.md). 완료된 분석은 재실행하지 않는다.

## 학생 문맥을 보존하는 masked BC·실제4회 완료 — 2026-10-09

[고정 조건](masked-correction-bc-protocol-2026-10-09.md)에 따라 새 plan
`configs/masked-correction-bc-20261009.json`을 만들었다. 원본 교정 시연은 재실행하지 않는다.
학생/중립 action target=None, BC schema2로 기존 무마스크 학습기를 차단한다.
전체 문맥 replay·교사만 loss·문맥-only window 생략·부모 기억층 복원을 구현했다.
Windows336검사 중319통과/17환경 제외, 신규6검사 WSL통과. 특징 추출
`results/masked-correction-features-20261009/` 완료:4961프레임, 학생 수집 당시3331입력과
특징 오차0. 교사 정답1624개/문맥 전용3337개를 확인했다. train1005정답으로
`results/masked-correction-training-20261009/`의130개 교사 포함 window 갱신 완료.
개발 NLL1.901921→1.690696, 부모 대비 train 교사 KL.028744, 저장/reload오차0.
문맥7프레임의 미래 교사 행동 gradient·미래→과거 영향0을 확인했다. 전체 영상 추론도
부모/후보 각각653프레임 특징·기억·확률 오차0. `results/masked-correction-comparison-20261009/`
실제4회 중 긴 벽2회 종료: 부모1613/후보1586행동, 둘 다180초 미획득. 초기서버/기억/픽셀 동일.
자연은 후보198행동28.346초/부모700행동98.982초, 둘 다 서버 oak_log+1. 자연도 초기서버/
기억/픽셀 동일. 각1회 개발 대조이므로 일반화나 일반적인 속도 개선을 주장하지 않는다.
배치 completed=true/errors=[]/active=null, 실제4097전이·로드 가중치 일치·STOP뒤dispatch0·
입력 해제키0·소유PID0 감사 완료. VRAM 최대2572~2608MiB/OOM0. 원문/기본 정책/최종5시드 보존.
긴 벽 후보는 닫힌 쪽 z=961.3으로 이동해1144전이 정체, 열린 통로 z>=968.3에는 도달하지 못했다.
이동은 바뀌었지만 출구 방향 선택·지속 이동은 미해결이다. 다음은 교정 시작/지속 이동/통로 진입의
행동별 예측 분석이며 이번 평가 궤적을 train으로 전환하지 않는다. 실제 LLM/Colab0.
완료된 게임/학습/추출/원본 시연은 재실행하지 않는다.
[상세 결과·한계](masked-correction-bc-validation-2026-10-09.md).

## 학생 방문 정체 상태 교정 시연 수집·감사 완료 — 2026-10-09

[사전 조건](student-correction-protocol-2026-10-09.md)에 따라 기존 기억층 탐색 후보로
train4/development2의 새 실행6회를 완료했다. `results/student-correction-20261009/experiment.json`
completed=true/errors=[]/active=null, final-audit.json verified=true. 완료된 배치를 재실행하지 않는다.
기존 그래프 정체→neutral1tick·서버 키0 확인→명시적 교사 인계를 같은 월드에서 연결했다.
학생/neutral은 문맥 전용, 교사만 loss_eligible=true인 `correction-rows.json`으로 보존한다.
6회 모두 교사 구간에서 통로를 통과하고 서버 oak_log+1; 학생 성공으로 세지 않는다.
전체4961전이(train3260/dev1701), 교사 정답1624개(train1005/dev619), 문맥 전용3337개.
`configs/student-correction-data-20261009.json` 분할/해시/mask 고정. 기존 teacher 추출기에
그대로 넣지 않는다. PPO/optimizer0·bc_training_ready=false. 기본 정책/원문/최종5시드 보존.
가중치·PNG·행동·서버 tick·RM6/보상 전수 감사, STOP뒤dispatch0·키0·소유PID0.
최초 감사의 tuple/list 비교 오류3곳은 원본/v1 보존 후 검사기만 교정했고4961전이 정확 일치.
Windows330검사 중313통과/17환경 제외, WSL신규7통과. 장치VRAM최대2628~2754MiB/OOM0.
[실제 결과·지연·한계](student-correction-validation-2026-10-09.md). 다음은 학생 prefix를 유지하고
교사 위치에만 손실을 주는 masked 연속 BC 연결과 분리된 실제 학생 평가다.

## 연속 복구 기억층 BC·실제 진단 완료 — 2026-10-08

[사전 조건](temporal-bc-protocol-2026-10-08.md)에 따라 새로운 temporal 입력2037개를 추출하고,
같은 train1436/dev601·8프레임·lr5e-7·순서로 tail/마지막 기억층+tail을 각각182갱신했다.
`results/temporal-bc-tail-20261008/`, `results/temporal-bc-temporal-20261008/` 학습 완료.
개발 NLL2.197233→2.122717/1.811346, reload0, 기억층 실제 변경·앞7프레임 gradient 확인.
기억층 후보 trainKL .060300으로 원래 .03 gate 미통과. 통과로 바꾸거나 재학습하지 않는다.
[별도 진단 조건](temporal-bc-diagnostic-protocol-2026-10-08.md)을 게임 전에 고정했다.
원래4조건은 미실행으로 닫고 같은 후보의 제한된 탐색 게임4회를 별도 완료했다.
전체 추론 대조1차는 검사기의 CPU 초기 기억/GPU 모델 불일치로 중단했으며 오류 보존.
초기 기억 device 교정 후 `results/temporal-bc-runtime-20261008-v2/` 완료: 두 후보 각각281프레임
전체 추론/캐시 특징·기억·행동 log-prob 불일치0. 완료된 추론 대조를 재실행하지 않는다.
`results/temporal-bc-diagnostic-20261008/` 긴 벽 두 게임 완료: 대조1413/기억층1378행동,
둘 다180초 미획득. 초기서버/기억/픽셀 동일. 기억층 이동명령34회지만 우회 미해결.
자연 후보391행동62.049초 birch_log+1/대조209행동30.256초 oak_log+1.
자연 서버/기억 동일·초기23×20픽셀 차이로 시간/인과 비교는 부적격. 각1회 일반화 주장 없음.
`results/temporal-bc-diagnostic-20261008/experiment.json` completed=true/errors=[]/active=null,
실제3391전이·로드된 가중치 일치·STOP뒤dispatch0·키0·소유PID0 감사 완료.
기억층 벽 이동경로5.774블록이지만 옆 방향 범위.524블록으로 우회 미해결; 기본 정책 유지.
장치VRAM최대2821~2855MiB/OOM없음, 학습 PyTorch최대2015.89MiB. 실제 LLM/Colab0.
Windows306통과/17환경 제외·WSL신규8통과. 아직 이번 후보의 실제 우회 성공 근거는 없다.
완료된 추출/학습/추론/게임/감사를 다시 실행하지 않는다. 원문 명세/최종5시드 유지.
[결과·지연·오류 교정·한계](temporal-bc-validation-2026-10-08.md). 다음은 훈련 월드의 학생 방문
정체 상태 교정 시연이며, 이번 개발 평가 궤적을 학습으로 전환하지 않는다.

## 정체 이력·재계획 제안 검증 연결 완료 — 2026-10-08

[계약](replanning-contract-protocol-2026-10-08.md), [결과·한계](replanning-contract-validation-2026-10-08.md).
정체 snapshot→실패 이력→제안 검증의 오프라인 경계를 구현했다. 기존 RM/완료 증거 불변,
동일 실패·stale·STOP/완료·미지원 skill·허구 근거·보상 수정·입력 해제 미확인을 거부한다.
요청3회/응답1회/TTL30초/전체180초 예산과 소비 상태를 저장 복원한다. 자동 pause revision
증가가 같은 실패를 새 시도로 오인하게 하는 문제를 회귀 검사 후 수정했다.
실제 정체5event/4궤적에 가상 제안35건 모두 예상 사유 거부, 정상 참고10궤적 요청0.
원본에 없는 task/generation/neutral 증거는 생성하지 않았다. 현재 조건은 동일 collect_wood를
제외하면 검증된 대안이 없어 model 호출 생략으로 판정. 우회 skill을 허위 등록하지 않았다.
`results/replanning-contract-20261008-v2/report.json` 최종 완료. 새 게임/학습/LLM0,
모든 판정 execution_authorized=false. Windows298통과/17제외·WSL신규32통과.
STOP/연구명세/원본/기본 정책 보존. 완료된 감사는 재실행하지 않는다.
실제 우선순위 스케줄러·LLM/dispatch 연결은 미구현. 다음은 연속 복구 시연의 기억층 BC
학습기/체크포인트/추론 연결과 분리된 실제 대조이며, 이번 검사를 우회 개선으로 세지 않는다.

## 그래프 반복·무진전 감지 오프라인 검증 완료 — 2026-10-08

[계약](graph-stall-protocol-2026-10-08.md), [결과·한계](graph-stall-validation-2026-10-08.md).
시간 순서의 방향 그래프 사이클과 실제 무진전을 함께 확인하는 관찰 전용 감지기를 추가했다.
새 위치/시야/대상/거리 최고 기록/채굴 진행/블록 소실/인벤토리 증거를 사용하며 RM·보상·행동은 변경0.
`configs/graph-stall-replay-20261008.json` 원본14궤적 해시/분할 고정. 최종 v3에서8788전이
재생 완료: 알려진 정체4/4검출, 성공 참고10궤적 경보0, 관측 누락0. 일반 오탐률/재현율 증거 아님.
긴 벽 원본/로컬 모두200tick, 탐색 정체400tick, 다른 막힌 출발421tick 검출(기존 접근 감지687tick).
기존 감지기가 놓친 search 정체를 검출했다. 전체 경로 감사에서 반복 점프 사이클2종과 자연 성공의
단발 사이클2종 확인. 실제 수평 왕복 실패 분류는 미검증이며 해당 경계는 합성 검사로만 확인했다.
v1은 render_ready 의미 오해로 관측을 누락하여 성능 근거에서 제외, v2는 교사/정책 행동 shape
차이로 중단했다. 원본 오류를 보존하고 회귀 검사 후 v3로 교정. 임계값은 변경하지 않았다.
Windows283검사266통과/17환경 제외, WSL새31통과. STOP/연구명세/원본 해시 보존.
새 게임·학습·LLM 호출0, 기본 정책 유지. 완료된 원본 게임 배치/오프라인 산출물 재실행 금지.
다음은 감지 근거의 계획 검증/실패 이력 연결과 연속 복구 시연의 기억층 BC다. 실제 planner 집행,
RM 우선순위 전환, 우회 skill 개선은 아직 완료하지 않았다.

## 원본 정책 대조·기억층 학습 가능성 검사 완료 — 2026-10-08

[고정 프로토콜](base-policy-baseline-protocol-2026-10-08.md),
[실제 결과·한계](base-policy-baseline-validation-2026-10-08.md).
기존 개발화면128장의 공식/로컬 특징·기억상태·동일 RNG 행동 표본 불일치0.
AMP/FP32 출력층의 확률 차이는 남으며 모든 입력의 행동 동일성을 보장하지 않는다.
`results/base-policy-parity-20261008/original-head.pt`는 원본head+RM0열이며 학습 후보가 아니다.
원본 대 기존 gap부모 실제4회: 긴 벽은 모두180초 미획득·이동 명령0·서버 이동0,
자연은 원본236행동33.072초/로컬283행동44.512초에 각각oak_log+1.
긴 벽 초기서버/메모리/픽셀 동일. 자연은10×6영역 픽셀 차이로 인과·속도 비교 부적격.
`results/base-policy-comparison-20261008/experiment.json` completed=true/errors=[]/active=null,
comparison_valid long_wall=true/natural=false. completed는4회 실행·감사 완료만 뜻한다.
3541전이 감사·STOP뒤dispatch0·키0·소유PID0. 장치 표본 VRAM최대2790~2807MiB, OOM없음.
별도 train교사8연속프레임으로 마지막 기억블록+표현층+head 폐기용AdamW1회 실행.
이전7프레임 gradient·기억블록 실제 가중치 변화·재현0·미래→과거 영향0 확인.
기억블록5067만/전체학습대상7288만파라미터, 게임 종료·동결prefix CPU 이동 조건에서
PyTorch peak allocated2334.70MiB. `results/temporal-finetuning-probe-20261008/probe.json` 완료.
이는 부분 기억층 학습의 기술 검사이며 실제 우회 개선/긴 시퀀스/RL/전체모델학습 증거가 아니다.
기억블록 체크포인트와 정식 연속시연 학습기 연결은 미구현; 다음 단위는 그 계약과 BC 대조다.
기본 정책/최종5시드/연구명세 유지. 완료된 대조·probe 일회용 스크립트는 재실행하지 않는다.

## 기본 행동 추가 학습 경로 조사 — 2026-10-08

[코드·공식 자료 조사](base-policy-learning-assessment-2026-10-08.md): STEVE-1/VPT 추가 학습은
가능하다. 현재 PPO는head/value, 최근BC는tail/head만 갱신하며 CNN/recurrent 동결과
RM6의 마지막 출력층 연결, BC KL gate의 제약을 확인했다. 기본 행동 전부 미학습으로 단정하지 않는다.
원본 정책/로컬 실행 경로 대조→학습 제한 분리→연속 시연/temporal 학습→교정·RL을 후속 순서로 제시.
이번 턴은 조사·문서만이며 새 학습/게임/대용량 수집/클라우드 연결0. 기존 실행 배치 재실행 금지.

## 러프 명세·RM 우선순위 보존 요구 반영 — 2026-10-08

[러프 명세](research-spec-rough.md)를 추가했다. LLM은 기존 RM/진행도를 지우지 않고
우선순위를 변경하며, 취소/대체/사용자 중단은 자동 재개하지 않는 요구를 명시했다.
원문20항목은 보존했다. 이번 변경은 문서만이며 다중 RM 스케줄러의 구현·실행 검증은 미완료다.

## 시연 보강·사전학습 표현층 미세조정·실제 비교 종료 — 2026-10-08

교사train6/development2에서2037전이(train1436/dev601), 실제원목8회 획득/감사 완료.
교사 성공은 학생 성공으로 세지 않는다. `configs/diverse-teacher-bc-20261008.json` 분할/해시 고정.
공개 STEVE-1 lastlayer/final_ln+head의 실제 BC 연결. CNN/recurrent 동결, privileged RM6.
같은 데이터/부모/순서로 head 대 tail+head 비교: lr1e-5와2e-6는 KL gate 미통과 조건이 있어
해당 게임 비교는 미실행. lr5e-7 각180갱신, devNLL2.163623/2.066149, KL.000452/.005544 통과.
6개 학습 총1080갱신, 별도 폐기용 probe1회. 원본/tail 특징 재현0, checkpoint reload0.
실제 긴 벽 head1478/tail1450행동 모두180초 미획득, 중앙벽 공격1453/1425, 이동·통과0.
자연 tail283행동45.086초/head474행동73.992초 모두oak_log+1. 각1회로 개선 주장 없음.
긴 벽 초기서버/메모리/픽셀 동일. 자연 서버/메모리 동일하지만 우측23×20영역 픽셀 차이로
자연 속도 비교 부적격. `results/diverse-bc-lr5e7-comparison-20261008/experiment.json`은
execution_finished=true/active=null/completed_rollouts=4/**completed=false**와 원래 비교 오류를 보존.
기존4게임을 다시 실행하지 않는다. audit.json에 long_wall=true/natural=false 기록.
3685전이 감사, STOP뒤dispatch0·키0·소유PID0. VRAM최대2756~2767MiB, OOM없음.
Windows252검사 중235통과/17제외, WSLtail3·NBT9통과. 게임 후 실제 head/tail 가중치 일치·단계별 손실 감사 완료. 탐색/접근 loss도 줄었으나 우회 미해결.
기본 정책/최종5시드/연구명세 유지. 다음 단위는 복구행동 supervision·BC변화제한·화면재현성 분리.
[결과와 한계](diverse-correction-validation-2026-10-08.md), [고정 조건/조정 기록](diverse-correction-protocol-2026-10-08.md).
완료된 시연/특징추출/학습/비교 일회용 스크립트 재실행 금지.

## 교사 BC 학습·실제 비교 완료, 우회 미해결 — 2026-10-08

`results/teacher-bc-comparison-20261008/experiment.json` completed=true/errors=[]/active=null.
완료된 개발 준비/교사/특징추출/학습/비교 배치는 다시 실행하지 않는다.
새 개발104761 교사283행동29.317초 oak_log+1; train410/dev283 분리, 동일 wood embedding과
실제 RM6차원 연결/재계산693개 일치. `configs/teacher-bc-20261008.json` source/frame 해시 고정.
`results/teacher-bc-features-20261008/`, `results/teacher-bc-training-20261008/` 완료:
BC52갱신 devNLL2.498326→2.304929, trainKL.014499/reload0/RM열 갱신/value 불변.
실제 긴 벽 부모1465/후보1500행동 모두180초 통과/획득0. 후보 중앙 벽 공격16회로 줄었지만
반대쪽 테두리 공격1343회(89.53%), 통로에 도달하지 못했다. 자연 후보351행동56.099초
birch_log+1, 부모209행동31.911초 oak_log+1. 두 조건 모두 초기 서버/메모리/픽셀 동일.
4회3525전이 감사·STOP 뒤 dispatch0·키0·소유PID0. OOM 없음. Windows232통과/16제외,
WSL BC7+교사3통과. 게임 종료 후 단계별 CPU loss 진단 완료. 기본 정책·최종5시드 유지.
다음은 다양한 train 출발/벽 길이/막다른 테두리 상태의 교정 시연과 출구 선택·통로 진입
분리 검증이다. epoch만 늘리거나 이번 결과를 우회 성공으로 해석하지 않는다.
[결과와 진단](teacher-bc-validation-2026-10-08.md), [사전 조건](teacher-bc-protocol-2026-10-08.md).

## 막힌 출발 교사 시연2개 확보 — 2026-10-08

`results/correction-teacher-20261008/experiment.json` completed=true/errors=[];
완료된 `scripts/run_correction_teachers.py`를 재실행하지 않는다.
명시적 privileged scripted teacher가 train104743/104759 반전 월드에서 각각205행동,
23.039/27.679초에 실제 oak_log+1. 벽에서 물러나 끝으로 우회, 둘 다73번째 행동에 통과.
학습 정책 성공 아님·optimizer0·PPO eligible=false·BC training-ready=false.
410화면/전송행동/전후상태/RM context와 SHA 보존, STOP 이후 dispatch0·키0·소유PID0 감사.
Windows230통과/16제외, WSL신규2통과. 원본 월드/부모/연구명세 불변·최종시드 미사용.
다음은 별도 개발 시연 확보와 학생 행동 표현/wood embedding/연속 문맥을 검증한 BC 연결.
[실제 결과와 한계](correction-teacher-validation-2026-10-08.md),
[사전 조건](correction-teacher-protocol-2026-10-08.md). 기본 학습 정책은 유지한다.


## 나무 복구 시연 탐색 완료·학습 진입 보류 — 2026-10-08

공식 VPT10 독립 세션6개 JSONL, 후보3개 MP4를 확보했다. 사전 train4/development2 분리,
기존 probe 세션 제외. 위치/원목 변화 후보와 스토리보드36프레임 검토 결과 일반 채집은
보이지만 긴 벽 앞 정체→우회→수집 시연은 미확인. 공개 데이터 전체에 없다는 뜻은 아니다.
추가 BC/게임평가0·기본 정책 유지·최종 시드 미사용. 부적합 목표를 붙여 학습하지 않았다.
`data/bc-wood-search-20261008/` 원본/해시, `results/bc-wood-search-20261008/` 후보/영상판독 보존.
같은 녹화 식별자의 다른 조각이 train/dev에 섞이는 BC 계약 누수를 회귀 검사 후 차단했다.
Windows228통과/16제외, WSL 관련6통과. 다음은 학습용 막힌 상태에서 실제 교정 시연 확보.
[상세와 한계](bc-wood-data-review-2026-10-08.md). 완료된 다운로드와 이전 BC 배치는 재실행 금지.


## BC 학습·실제 비교 완료, 긴 벽 개선 미입증 — 2026-10-08

`results/bc-comparison-20261008/experiment.json` completed=true, errors=[]. 완료된
`scripts/run_bc_comparison.py`를 재시작하지 않는다. 공식 FindCave 세션3개(약194MB)를 확보해
train2/development1로 분리, 지형 가장자리 이동 구간을 검토했다. frame/action 광류 시점 확인.
순차 특징 train621/development414개로 BC78회 갱신, 개발 NLL5.04695→4.73488,
train KL.01029, reload오차0, RM 열/value 불변.
`results/bc-features-20261008/`, `results/bc-training-20261008/` 완료 산출물을 다시 만들지 않는다.
긴 벽2000003 부모1465/BC1490행동 모두180초·통과/원목0. 자연2000029는 부모300행동48.600초
birch_log+1, BC225행동38.805초 oak_log+1. 자연 쌍 시작픽셀 차이·각1회·CPU 진단 중첩으로
개선 인과 주장 금지. 긴 벽 쌍 초기 서버/메모리/픽셀 동일.4회3480전이 감사·STOP/키0/소유 PID0.
고정 wood 평가 특징64개에서 KL4.988e−5/작은 TV라 과업 간 전이도 미미했다(학습0).
Windows227통과/16제외·WSL새5검사 통과. 기본 정책 유지·최종5시드 미사용.
다음은 wood 또는 명시적 이동 목표의 실제 복구 시연 보완이며 FindCave epoch만 늘리지 않는다.
[조건](bc-navigation-protocol-2026-10-08.md), [결과 기록](bc-navigation-validation-2026-10-08.md).

## 막힌 출발2회 완료·공개 모방 데이터 샘플 확보 — 2026-10-08

추가 합의: 과업에 적합하면 BC만으로 해당 skill을 채택할 수 있다. 목표 조건/시간 문맥을
유지한 BC부터 비교하며 관련 우회 시연의 존재를 먼저 확인한다. BC 이후 모든 skill에 RL을
의무 적용하지 않는다. 실제 분리 평가와 기존 능력 보존으로 판단한다.
[선택 기준](imitation-data-plan-2026-10-08.md). 이 합의 기록 시 추가 학습/rollout은 실행하지 않았다.

`results/blocked-entry-20261008/experiment.json` completed=true. 일회용
`scripts/prepare_blocked_entry.py`와 `scripts/run_blocked_entry.py`를 재실행하지 않는다.
기존 gap 후보로 train104743/104759·sampling801/802, 벽 앞6.5/3.5에서1536/1674행동을
수집했다. 각180초, 통과0·원목0. 벽 공격93.2%/99.7%. 첫째 복구 진입, 둘째 원목 목표를
잡지 못해 전부 search로 접근 정체 감지 대상이 아니었다. 지연 낮아도 이동 실패.
optimizer0·평가/최종시드0·기본 정책 유지. STOP/키0/소유 PID0·원시 감사 통과.
[실제 결과](blocked-entry-validation-2026-10-08.md), [사전 조건](blocked-entry-protocol-2026-10-08.md).

사용자의 모방 데이터 제안에 따라 공식 VPT 영상/키·마우스 샘플184MB와8.x/10.x index를
확보했다. `data/vpt-contractors-preflight-20261008/` 원본/출처/해시 보존, Git 제외.
`results/contractor-audit-20261008-v2/` 영상6001프레임·행동6000개 검사, 형식 후보5060개.
JSONL 내부 마우스/다음 yaw 일치 확인, 영상 offset/목표·RM 라벨은 미확정.
원목 증가21건·이동17구간 색인만 추출했으며 모방 학습0·모방 정책 성공0이다.
초기 감사의 N=N 실패 산출물은 `results/contractor-audit-20261008/`에 보존했다.
새 변환 검사6개 Windows/WSL 통과, 전체 Windows222통과/16제외, NBT8통과.
[공개 출처·확인 결과·학습 연결 계약](imitation-data-plan-2026-10-08.md).
다음 단위는 시연 구간/목표 조건/시점 검증→별도 BC 후보→복구 RL/분리 평가다.
RM마다 새 모델을 처음부터 학습하지 않는다. 현재 런타임 planner/BC trainer 추가는 미구현.

## 통로 진입 학습·분리 평가 완료, 정체 복구 개선 미입증 — 2026-10-08

`results/recovery-gap-20261007/experiment.json`의 일회용6회 배치 완료. 완료된
`scripts/run_recovery_gap.py`와 준비 스크립트를 재시작하지 않는다. 통로 가까운 새 배치에서
세계104743/104759·정책 난수701/702를 분리해122/500전이, 원목1/3개를 획득했다.
실제 통과는30/31전이이며 정체 복구 진입0이었다.622전이로 PPO78회 갱신, 확률 재현/reload
오차0, KL.000664로 비교 후보를 보존했다. 긴 벽 별도2000003/2000029 평가는 전후 모두0/2,
실제 통과0회. 두 쌍 초기 서버/메모리/픽셀은 동일했지만 제어 지연 편차가 커 시간당tick이 달랐다.
총4,468전이 감사. 고정features64개 문맥 TV는 약2배 증가했으나 여전히 작고 획득 개선 없음.
기본 정책 유지·최종5시드 미실행. 평가로 재학습하지 않았다.
neutral 이후 서버 키0 기록/검사를 추가해 실제6회 통과, STOP 이후 전송0·잔존 프로세스0.
Windows216통과/16제외·WSL224통과/8제외·격리 NBT8통과. 원본24궤적+직전 실패4궤적,
월드/부모/embedding/JAR/명세 보존 검사 완료. nvidia-smi 계측 timeout4기록·OOM 없음.
[결과·경로·지연 한계](recovery-gap-validation-2026-10-08.md), [사전 조건](recovery-gap-protocol-2026-10-07.md).
[LLM 상위 플래너 구조 제안](llm-high-level-planner-2026-10-07.md)과 실제 snapshot3건도 준비했다.
실시간 planner 구현·API 호출은 이번0이다. 다음은 정체 후 통로 진입 경험/이동 skill의 관측 계약,
LLM 제안 검증 계약과 지연 원인 격리다. 존재하지 않는 skill을 실행 vocabulary에 추가하지 않는다.

## 복구 curriculum 수집 완료·성공 경험 확보 실패 — 2026-10-07

`results/recovery-curriculum-20261007/experiment.json`의 일회용 배치 완료. 완료된
`scripts/run_recovery_curriculum.py`를 재시작하지 않는다. 별도 훈련104743/104759에서
짧은 벽을 각2회 실행해6,948전이를 수집했지만 원목 획득/벽 통과 모두0회였다.
모두 RM 재탐색에 진입했으나 행동의92.4–97.5%가 벽을 향한 공격이었다.
사전 획득 기준으로 PPO0·새 checkpoint 없음·중간 단계 및 부모/후보 평가 미실행.
준비 ZIP6개와 실패 궤적을 보존했다. 같은 seed 반복을 독립 세계로 계산하지 않는다.
shaping trainer 최종 KL 검사는 연결했지만 실제 optimizer 경로는 이번 미실행이다.
Windows212통과/15제외·WSL220통과/7제외·격리 NBT7통과. 원본24궤적/월드/부모/JAR/명세
보존 검사, STOP 이후 정책 전송0·neutral cleanup 응답·프로세스0 확인. neutral 이후 키 개수
재관측 로그는 없으므로 그 측정은 미완료다. GPU 계측1회 timeout도 보존; CUDA OOM 없음.
최종5시드 미실행·기본 정책 유지. 다음 단위는 더 쉬운 시작 조건에서 정책 우회 획득 경험 확보,
세계/정책 난수 분리, 그 뒤 동일 behavior 학습→분리 평가다.
[결과·경로·한계](recovery-curriculum-validation-2026-10-07.md),
[사전 조건](recovery-curriculum-protocol-2026-10-07.md).

## 접근 정체·재탐색 입력 검증 완료, 행동 복구 실패 — 2026-10-07

`results/approach-recovery-20261007/experiment.json`의4회 실행 완료. 완료된 일회용
`scripts/run_approach_recovery.py`를 재시작하지 않는다. v4는200tick 위치/거리 무진전 때
search 문맥으로 전환하고 실제 진전·다른 대상·채굴 준비/회수/종료에 해제한다. 행동 주입 없음.
기존 실패 부모는 sparse 문맥 열이0이어서 RM 입력만 바꿔 복구됐다고 할 수 없다.
오프라인 legacy 대입에서482/1175전이 정체, 정상 최근2실행 경보0. 실제 bedrock 우회 준비
344tick 획득/감사 통과·정체 오경보0. 정책 v3/v4는1561/1671tick 각각180초 미획득.
v4는266전이에서 재탐색 입력으로 전환했지만 접근 회복/획득 없음. 실제사망467tick,
health0/alive0/held keys0·실패−1/bootstrap0·추가 행동0·소유 PID0 감사 통과.
초기 상태/픽셀 동일하지만 첫 행동 차이37은 복구 진입보다 먼저라 경로 차이를 입력 효과로 볼 수 없다.
고정features64표본의 문맥 교체 TV 평균 camera8.466e-6/buttons4.945e-6, logp 재현 오차0.
RM 입력은 전달됐지만 현재 head의 반응이 매우 작다. 문맥 열 추가 후 학습은98·161전이뿐으로
200tick 정체 경험이 없었다. 별도 훈련 세계의 복구/우회 경험 학습이 다음 단위다.
224검사 Windows212통과/12제외·WSL220통과/4제외, 격리 fixture4통과.
[결과/한계](approach-recovery-validation-2026-10-07.md), [사전 조건](approach-recovery-protocol-2026-10-07.md).
eval만 수집하고 optimizer0·최종5시드 미실행·기본 정책 유지. 이번 eval 자료는 학습하지 않는다.

## 원목 실제 채굴 대상 교정 완료 — 2026-10-07

`results/wood-target-20261007/experiment.json`의 일회용3회 실행 완료; 재실행하지 않는다.
최초 시야의 원목을 계속 잠가 다른 원목을 캐는 동안 접근 단계로 남는 원인을 실제 서버
PlayerInteractionManager의 채굴 좌표로 확인했다. v2 수집에서145채굴 관측 중143접근 오판,
v3 수집에서151관측 중 기존 센서150접근 오판. 병렬 새 센서는296관측 모두 채굴 좌표/단계 일치.
기존 센서/v1/v2는 보존하고 wood-potential-v3에 서버 진행 중 채굴 대상 전환을 연결했다.

같은 v2 checkpoint·흙 벽 월드의 v2/v3는256/252tick, 실제 서버 원목1/2개 획득.
초기 상태·픽셀 동일, actor/value 양 CFG 문맥/보상 원시 감사 통과. 추가 학습0·기본 정책 유지.
v3 사망도467tick, health0/alive0/held keys0, 실패−1/bootstrap0, 종료 뒤 행동0·소유 PID0.
217검사 Windows205통과/12제외, WSL213통과/4제외. 연구 명세·checkpoint·world·기존 학습24궤적 보존.
이번 eval은 학습하지 않는다. 최종5시드 미실행. 실제 드롭 추적과 막힌 접근 복구는 남았다.
[결과와 다음 단위](wood-target-validation-2026-10-07.md), [조건](wood-target-protocol-2026-10-07.md).

## 원목 후속 비교·사망 경계 수정 완료 — 2026-10-07

`results/wood-followup-20261007/experiment.json`이 현재 배치 상태다. 일회용 runner를 재실행하지 않는다.
준비 scripted 경로4개·시험장 학습 정책4개는 실제 획득했다. 흙 벽은 파괴207/우회344tick,
돌 벽은 파괴347/우회273tick. 정책 비교는 흙 부모405/v2 495, 돌 부모750/v2 528tick;
두 비교의 초기 상태/픽셀은 동일했다. 학습 후 v2 별도 개발 확인도134tick 획득/원시 감사 통과.
기본 정책 승격·추가 학습0. 최종5시드는 미실행.

사망 noop 진단은466전이 뒤 서버 tick6 증가로 오류 중단됐고 실제 게임 로그/저장 NBT에서
사망 후 자동 재생성을 확인했다. 해당 무효 시행과 정리 로그를 보존한다.
원본 비교 코드/체크포인트를 유지하면서 `scripts/resume_wood_followup.py`로 미실행 자연 세계
10회만 이어서 완료했다. 이 resume 역시 완료된 일회용이며 재실행하지 않는다.
자연 세계 부모4/5, v2 5/5 획득. 시간 초과180초 포함 평균62.113/23.639초.
초기 서버 상태는7대조 모두 동일, 픽셀은6/7동일(마지막 자연 세계만 작은 차이)이므로
성공률 차이를 학습 효과만으로 단정하지 않는다. 배치19기록 원시 감사 완료:18유효·사망 진단1무효.

비교 종료 후 RL 전용 사망 종료 설정·자동 재생성 차단·마지막 관측 발행을 별도 JAR로 수정했다.
동일 noop 사망 재현은467전이, 실제 server health0/alive0/held keys0, 실패 비용-1,
추가 행동0·잔존 PID0으로 통과했다. 행동당1tick 검사는 유지한다.
학습 정책 사망도467전이/terminal bootstrap0/실패 비용-1로 원시 감사 통과했다.
같은 수정 JAR의 일반 원목 회귀는86tick 획득/감사 통과. 두 사망과 정상 회귀 모두
입력 해제·이후 새 정책 행동0·소유 PID0을 확인했다. Windows/WSL 실행 프로세스0, STOP 유지.
결과는 `results/death-boundary-*.log`와 `results/wood-followup-20261007/final-verification.json`에 기록한다.
현재211검사 중 Windows199통과/12제외, WSL207통과/4제외. 원본 설치 JAR는 보존한다.

[후속 결과](wood-followup-validation-2026-10-07.md), [사망 경계 원인](rl-death-boundary-2026-10-07.md),
[앞으로의 작업 단위](next-task-units.md), [공통 단계 계약](task-decomposition.md).
루트 AGENTS.md에 후속 과업의 상태 변화 단위 분해를 연결했다.

## 원목 단계 RM·거리/틱 비용 연결 완료 — 2026-10-07

사용자 승인에 따라 고정 목표 중심 ray 관측과 탐색/접근/채굴/수집/완료 상태를 추가하고,
단계 one-hot5개+거리1개를 STEVE-1 actor/value head에 연결했다. 영상/recurrent backbone 고정,
직접 행동 API 대체 없음. 서버 보조 입력 조건이며 영상 단독 인식으로 해석하지 않는다.
v1 실제 학습 수집98tick/16.446초 원목 획득, PPO7회 갱신. 학습 제외 개발 월드2000003에서
86tick/12.747초 획득했으나 과거에도86tick이었으므로 개선 근거 없음. 5초 제한 진단31tick,
정상 time_limit/미완료 bootstrap .194996 보존. 진단은 성공률에서 제외한다.

시간 비용 목적함수에서 조기 실패 유인을 발견하고 v1 원본을 보존했다. v2는 실제 환경 실패
종료에만 -1, 사용자 STOP/기술 오류는 학습 제외, 시간 초과는 bootstrap이다. 수정 수식/단위
검증 통과. v2 실제 수집161tick/24.233초 획득, 다섯 단계 모두 관측, 추가 PPO11회(이번합계18회).
이 수집은 v1에서 갱신된 head를 사용했으므로98→161을 보상 버전만의 효과로 해석하지 않는다.
v2 마지막 학습 checkpoint는 tensor/optimizer 재로딩 검증까지이며 학습 후 gameplay 평가는 남았다.

실제376transition의 보상/상태 입력/행동을 원시 로그와 재대조했다. 목표/중단 확인 후 행동0,
입력 해제·소유 PID0. 각 학습 확률 재계산/재로딩 오차0. Windows192통과/7제외, WSL199통과.
연구 명세·이전 checkpoint3개·학습 궤적24개 보존 해시 확인. 기본 정책 승격 없음, STOP 유지.
파괴/우회 배치의 학습 성능 비교·다중 시드 평가·실제 실패 종료 분기 검증은 미실행이다.
[결과](wood-shaping-validation-2026-10-07.md), [조건·수정](wood-shaping-protocol-2026-10-07.md).
근거 `results/wood-shaping-verification.json` 및 각 실행의 `shaping-audit.json`.

## 새 개발세계 고정 비교 완료 — 2026-10-07

새 개발세계 2100011/2100023/2100037/2100041/2100059에서 부모78회·γ=.99/λ=.95 대조·γ=.999/λ=.95 후보를 각2회, 실제30회 평가했다. 부모/대조 각9/10, 후보8/10 획득. 정상적인180초 미획득에180초를 부여한 평균은53.18/43.82/59.23초다. 후보는 정글 세계에서0/2로 양쪽 대조1/2보다 적었고, 양쪽보다 빠른 세계는2/5였다. 사전 채택 기준 미통과로 기본 정책 유지; 대조도 사후 자동 승격하지 않는다. 추가 optimizer 업데이트0, 중간 보상·API 대체 없음.

첫 준비는 기존 월드 이름 충돌로0행동 중단됐다. 별도 JAR의 새 저장 이름에 UUID를 추가하고 실제 NBT seed를 검증해 같은 시드로 복구했다. 실패와 복구 전 JSON·로그 보존. 정상 준비5회와 평가30회는 동일 수정 JAR를 사용했다. 실제11,539행동/로그확률 재현 오차0. 최초 분기20대조 중9개에는 출력층 교체 효과가 없었지만 정글 세계의 첫 camera 선택에는 직접 효과가 있었다. 부모·대조의 같은 정책 반복에서도 성공/실패가 갈려 전체 실패의 단일 원인으로 단정하지 않는다. 원시 감사상 목표 승인·중단 확인 뒤 새 정책 전달 없음, 입력 해제·잔존 PID0. 초기 측정 상태25/25, 시작 픽셀23/25 동일. 첫 추론1.42~2.92초로 일부2초 초과도 기록했다.

Windows181통과/7제외·WSL188통과. 기존 가짜 NBT 테스트 자료1개를 실제 형식으로 수정하고 최초 실패 로그를 남겼다. 기존 학습 궤적24개·체크포인트·연구 명세 원문 보존 검사 완료. STOP 유지, Windows/WSL 실행 프로세스0. 최종5시드·유료·Colab·GitHub 공개 미실행. 근거: `results/rl-horizon-new-worlds-20261007/{experiment,audit,final-verification,recommendation}.json` 및 `action-diagnosis/diagnosis.json`, `failure-diagnosis.json`. 완료된 runner를 재실행하지 않는다. [결과와 다음 범위](rl-horizon-new-worlds-validation-2026-10-07.md), [사전 조건](rl-horizon-new-worlds-protocol-2026-10-07.md), [이름 충돌 수정](rl-horizon-world-collision-2026-10-07.md).

## 보상 시간 규모 비교 완료 — 2026-10-07

사용자가 야간 자율 실행을 승인했다. 중간 보상 없이 γ/λ9조합과 새 discount별 critic을 비교하고, γ=.99/.999 × λ=.95/1의 네 분기를 각304회(합계1,216회) 동일 조건으로 학습했다. 확률 재현 오차0, checkpoint reload 통과, 최종 평균 KL .0015~.0024다. 부모+4분기×개발2시드×2반복 실제20회 모두 원목 획득, 기술 실패/재시도0이었다. γ=.999·λ=.95는2000029에서338·345tick, 부모448·448tick으로 평균23.77% 단축을 관측했으나2000003은모두86tick이었다. 원시 감사 통과, 실제4,866행동/확률 재생 오차0. 첫 행동 분기10대조 중9개는 양쪽 입력에서 actor 교체 효과가 없어 관측 편차가 개입했고, 전체 성능 향상의 인과 효과는 미확정이다. 후보만 `recommendation.json`에 보존하며 기본 정책 승격 없음. Windows174통과/7제외·WSL181통과. 최종5시드/유료/Colab/공개는 미실행이다. 근거 `results/rl-horizon-20261007/{experiment,audit,final-verification}.json`; 단일 실행용 runner를 재시작하지 않는다. STOP 유지. 후속 새 개발세계 비교는 상단 완료 기록 참조. [사전 조건](rl-horizon-protocol-2026-10-07.md), [완료 보고서](rl-horizon-validation-2026-10-07.md).

## 첫 추론 준비 경계 검증 — 2026-10-07

실제 첫 Conv2d의 준비를 RLPolicy 생성 시 수행하고 완료 후 STOP을 재확인한다. 준비 시간은 1.789초·7.374초로 별도 기록했고, 실제 104869/104831 첫 추론은 0.275초·0.195초였다. 두 시행 모두 서버 원목 획득(300행동/43.796초, 244행동/48.752초), 학습 제외·optimizer 갱신 0이다. 준비 도중 취소는 월드 reset 이전/0행동/잔존 PID 0으로 검증했다. Java 버전 검사 시간 초과 1회는 blocked/0행동 원본을 보존하고 별도 폴더에서 한 번 복구했다. 원시 감사 통과, WSL 178개 통과·Windows 171개 통과/7개 제외. 초기 상태는 원본과 일치했으나 두 번째 시작 화면은 달랐다. 장시간 변동의 정확한 원인은 미확정이고 가치함수 채택 기준은 유지한다. STOP 유지, 최종 평가 5시드 미실행. [보고서](rl-backend-preparation-validation-2026-10-07.md), [사전 조건](rl-backend-preparation-protocol-2026-10-07.md). 근거: `results/rl-backend-preparation-recovery-20261007`.

## 가치함수 교정 해결안 실행 완료 — 2026-10-06~07

선형 교정·64-unit MLP 비교 후 고정 부모 78회 actor로 새 8세계를 각각 한 번 수집했다. 새 수집은 서버 원목 획득 7/8, 유효 시간 초과 1/8이며, 기존 포함 16세계·24개 궤적·4,855행동으로 재비교했다. 전체 OOF MSE는 기존 .123595→선형 .056771/MLP .057850으로 낮아졌지만, 초반 MSE는 .015243→.066668/.067844로 악화해 채택 기준 미통과다. 초반 정규화 advantage 양수는 26/2,500→995/1,194로 늘었으나 행동 개선 증거는 아니다. Actor 추가 갱신 0, 진단 MLP 갱신 총 7,660회. 새 궤적 원시 감사·보존 해시 검사 통과, WSL 176개 통과·Windows 169개 통과/7개 제외. 첫 추론 9.42~137.24초 지연의 정확한 원인과 실제 정책 개선은 미해결이다. 기본 정책·최종 평가 5시드는 유지하며, 추가 actor 비교와 반복 학습은 조건 미충족으로 미실행이다. STOP 유지·잔존 실행 프로세스 0. [실제 결과와 한계](rl-value-repair-validation-2026-10-06.md), [사전 조건](rl-value-repair-protocol-2026-10-06.md). 근거: `results/rl-value-data-expansion-20261006/final-verification.json`.

## RL 개선 부재의 원인 점검 완료 — 2026-10-06

저장 학습·검증 행동5,598개를 원래 특징/난수/actor로 전부 재현했고 logp 오차0을 확인했다. PPO·GAE 식의 해당 경로는 독립 계산과 일치했다. 끝 보상 직접 영향의 짧은 시간 범위, 가치 예측 의존성과 배치 평균 보정 후 초반 음수 집중을 확인했다. lambda만 높이는 것으로 그 현상이 해결되지 않는다는 계산도 남겼다. 어려운 검증 시드의 세 비교 모두 최초 분기는 actor 변경이 아니라 특징 차이에서 발생했다. 원인 전체의 인과 비중과 해결 효과는 아직 미검증이다. 정책/학습 코드와 checkpoint 변경0, 새 gameplay/optimizer0. 권장 다음 단계는 actor를 고정한 critic 예측 진단이며, 반복 학습 확대에 앞서 학습 신호를 점검한다. [근거와 해결 순서](rl-root-cause-2026-10-06.md).

## 확대 RL 배치 완료 — 2026-10-06

학습8시드×2회, 실제3,149 transition으로 출력층·가치함수를394회 추가 갱신했다(누적472회). 검증2시드×전후3회는 전후 각각6/6 성공이지만, 행동 중앙값은 첫 시드86→86, 두 번째283→361로 개선 근거가 없다. 기본 정책으로 승격하지 않는다. 원목 미획득·종료 후처리 기술 실패1회는 원본을 보존하고 별도 집계했으며, 감독기 수정 후 같은 조건을 정확히1회 재실행했다. 별도35초 실제 시간 초과 진단은 정상 종료·중단 이후 행동0·입력 해제·소유 프로세스 정리를 확인했다. 정상28개와 기술 실패 원본 감사 통과, WSL172개 통과·Windows165개 통과/7개 제외. 최종 평가5시드는 미실행이다. [전체 결과](rl-expanded-validation-2026-10-06.md), [고정 조건과 결함 수정](rl-expanded-protocol-2026-10-06.md). 실행 근거: `results/rl-expanded-20261006T095005817265Z`.


**2026-10-05 RL 반복 편차 진단 완료:** 같은 검증 시드에서 추가4회, 기존 포함 전후 각3회 모두 원목 획득. 행동 수는 전211·312·517, 후295·423·406으로 개선 근거가 없다. 저장 행동506개/확률을 정확히 재현한 교차 분석에서 최초4번째 시선 분기는 actor 변경이 아니라 관측 특징 차이로 발생했다. 엔진 내부 화면 차이의 원인은 미확정이며 추가 학습·기본 모델 변경은 없다. [실제 결과·원인 범위](rl-variance-diagnosis-2026-10-05.md).


**2026-10-05 다중 시드 RL 배치 완료:** 실제3개 시드452 transition으로 행동 출력층·가치함수58회 추가 갱신(누적78회). Backbone은 고정했다. 별도2개 시드에서 시작 화면·상태가 전후 일치했고 모두 원목 획득에 성공했지만, 행동 수는86→86과211→295로 개선되지 않았다. 새 체크포인트는 개선판으로 채택하지 않는다. 최종 평가5시드는 미실행이다. [결과·한계](rl-batch-validation-2026-10-05.md), [확정 프로토콜](rl-batch-protocol.md).

아래는 앞선 단계의 기록이다.

## 첫 실제 로컬 RL 및 분리 평가 완료 (2026-10-05)

실제 원목 채집159 transition으로 STEVE-1 행동 출력층·가치함수 PPO20회 갱신, checkpoint/optimizer 재로딩을 검증했다. Backbone은 고정했다. 학습 제외5시드에서 전후 모두5/5 성공했고 네 시드는 빨라졌으나 한 시드는24.19→80.43초로 느려졌다. 초기 화면4쌍 불일치 때문에 엄밀한 동일 초기 상태 비교도 미확보다. 일반 향상 주장은 보류한다. 보상·실제 전달 행동·종료 이후 차단·평가 데이터 제외 감사는 통과했다. [전체 결과](rl-pilot-validation-2026-10-05.md).

아래 내용은 각 단계 당시의 기록이다. RL 업데이트0이라는 과거 상태는 위 실험으로 갱신되었다.

## 작업대 제작·설치·3×3 도구 제작 검증 완료 (2026-10-05)

STEVE-1 자연 채집으로 원목3개를 확보한 뒤 불변 월드를 복제했다.
명시적 GUI 보조로 작업대 제작·배치·열기→나무 곡괭이1·도끼1 보존5/5.
각183행동/제어19.10–19.99초, 결과 클릭7회, Shift0, 완료 후 새 행동0.
서버 인벤토리·저장 NBT와 작업대 저장 청크5/5 일치, 소유 프로세스 잔존0, 추가 API0.
별도 중단 시험은 곡괭이를 커서에 든153단계에서 멈추고 곡괭이1·판자3·막대기2를 보존했다.
초기화 교착1건은 행동0의 기술 오류로 제외하고 게임 스레드 관측 발행으로 수정했다.
공식 레시피 JSON의 의존 재료 계획·shaped/shapeless·tag·2×2/3×3 공통 검사를 추가했다.
Windows145통과/4제외, WSL149통과. RL 업데이트는0이며 다음 연구 단계는 실제 채집 보상
연결과 첫 backward/optimizer 갱신이다. 연구 명세20항목은 유지한다.
[결과](workshop-validation-2026-10-05.md), [교착 원인](workshop-observation-threading.md),
[첫 RL 단계](first-rl-stage.md).

## 레시피북 GUI 경로·해금 조건 검증 완료 (2026-10-05)

사용자 승인에 따라 학습 정책과 분리한 명시적 레시피북 GUI 보조를 추가했다.
서버/클라이언트 해금·표시·재료 검사→책 클릭→실제 배치/출력 검사→일반 클릭1회→회수·저장.
막대기4·판자2 보존5/5, 작업대1 보존5/5. 유효10회 모두14행동, 목표 이후 새 행동0,
저장 NBT10/10 일치. 막대기5회는 곡괭이 레시피가 새로 해금됐고 작업대5회는 미해금 유지.
초기화1회가/proc 순회 중 ProcessLookupError로 종료돼0행동 표본을 제외했다.
프로세스 감시 경합을 재현·수정하고262147만 별도 재실행해 성공했다. 최초 오류는 보존했다.
Windows139개 통과/4개 제외, WSL143개 통과. 원본 월드·연구 명세 유지, 추가 API0,
실행 프로세스 잔존0, STOP 유지. 검색/페이지/그룹,3×3,대화/RM 제작 연결은 남았다.
[실제 결과](recipe-book-validation-2026-10-05.md), [해금 조건](recipe-book-unlocks.md),
[감시 경합 수정](process-snapshot-race-2026-10-05.md).

## 결과 칸 Shift 제한 비교 완료 (2026-10-05)

결과 칸의 예상 커서 위치에만 Shift 제한을 적용하는 선택적 조건을 추가했다.
GUI 전환/초기 관측 전에도 제한하며, 다른 재료 이동과 월드 행동은 보존한다.
고정5회에서 막대기4·판자2 저장3/5, 과다 제작0/5, 다른 레시피 실패2/5.
성공3회는 커서 회수 보조 사용. 원본과130363의 정책 제안88개가 모두 동일한 상태에서
막대기8개 대신4개를 저장했다. 전체 Shift 제한2/5와 비교해 성공 시드도 달라졌다.
3274개 행동의 다른 필드 불변, 실제 Shift17회 수정/563회 보존, 커서2524개 좌표 대조 통과.
성공 관측 뒤 새 정책 행동0, 저장 NBT5/5 일치, 기술 오류·미실행·추가 API0.
Windows133개 통과/4개 제외, WSL137개 통과, Java 빌드 성공. 원본 월드·연구 명세 유지,
Windows/WSL 실행 프로세스 잔존0, STOP 유지. 레시피 선택과 재료 예산 판정이 남았다.
[최종 결과](craft-result-slot-validation-2026-10-05.md), [사전 조건](craft-result-slot-protocol.md).

## 일괄 제작 입력 제한 비교 완료 (2026-10-05)

실험 전용 --craft-no-quick-move를 추가해 정책 sneak/Shift를 전달 전에0으로 제한했다.
고정5회에서 막대기4·판자2 보존0/5→2/5, 과다 제작2/5→0/5. 목표 아이템 제작·저장 자체는
두 조건 모두2/5이며, 미제작1회/작업대 오제작2회가 남았다. 두 성공은 커서 회수 보조 사용.
시드155921에서96개 정책 제안 중95개가 동일하고 나머지도Shift 값만 달랐으며,
마지막 클릭의 Shift 제한 후 서버·저장 수량이8개 대신4개로 확정됐다.
5054개 전달의 다른 필드 불변, Shift573회 수정 확인. 목표 관측 후 새 정책 행동0,
저장 NBT5/5 일치, 원본 월드·연구 명세 유지, 추가 API·기술 오류·미실행0.
과거 제작 통계 지연에 의한 분석 오분류도 서버 기준으로 수정했다.
Windows127개 통과/4개 제외, WSL131개 통과. 기본 동작과 대화/RM 제작 미지원 범위는 유지한다.
[최종 비교](craft-shift-guard-validation-2026-10-05.md), [사전 조건](craft-shift-guard-protocol.md).

## 막대기·작업대 독립 실제 검증 완료 (2026-10-05)

중지 위치에서 재개해 판자4/흙1 저장본의 독립 복제본으로 고정 시드5개×두 조건을 실행했다.
작업대1개 제작·저장4/5, 막대기 제작·저장2/5이나 모두8개여서 정확4개·판자2개 보존0/5.
막대기 조건의 나머지는 작업대2회/버튼1회, 작업대 조건의 나머지는 막대기8개 제작이었다.
첫 목표 관측 후 새 정책 행동0회이나, 이미 전달한 Shift+클릭의 서버 일괄 제작이 수량을
4→8로 확정하는 사례를 원본 코드와 로그에서 확인했다. 레시피·예산의 행동 전 제약은 남았다.
저장 NBT10/10 일치, 기술 실행 실패·미실행·추가 API0. 커서 배치 보조 실제 사용0회.
Windows123개 통과/4개 제외, WSL127개 통과, Java 패치 빌드 성공.
원본 월드·연구 명세 유지, 실행 자식 잔존0, STOP 유지. 대화/RM 제작은 계속 미지원이다.
[상세 결과](craft-recipes-validation-2026-10-05.md), [수량 차이 원인](craft-batch-click-cause.md).

## 조합 정지 경계·서버 확정·저장 판정 수정 완료 (2026-10-04)

조합 실험은 수집 RM 밖의 경로였고, 인벤토리+늦은 제작 통계를 동시에 기다려 기존
성공 3회도 인벤토리 4개 이후 행동 4회가 추가됐음을 확인했다. 제작→회수→완료/실패
경계를 추가해 첫 판자 생성 관측 이후 정책 호출을 차단했다. 첫 회수 수정의 저장
불일치를 NBT 검사로 검출했고, 클라이언트 예측과 서버 확정의 차이를 직접 관측해
서버 인벤토리·커서·통계를 확인한 뒤 종료하도록 수정했다.
최종 고정 시드 5회 모두 판자 4개 회수·저장 NBT 일치, 이후 정책 행동 0회.
2회는 명시적 커서 배치 보조, 3회는 정책 인벤토리 회수 후 서버 확정 대기였다.
정책 자체 학습 개선이나 일반 대화/RM 제작 지원으로 해석하지 않는다.
Windows 119개 통과/4개 제외, WSL 123개 통과. 추가 API·최종 기술 오류·미실행 0.
원본 월드·연구 명세 유지, STOP 유지. 최초 수정 실패와 초기화 중 취소는 별도 보존했다.
[근본 원인](craft-rm-root-cause.md), [최종 결과](craft-recovery-validation-2026-10-04.md).
아래는 각 단계 당시 기록이다.

## 판자 회수 즉시 정지·저장 검증 완료 (2026-10-04)

선택적 완료 검사와 craft-probe 전용 옵션을 추가했다. 같은 원목 저장본의 독립 복사본에서
고정 시드 5회·제어 상한 180초를 실행했다. 판자 4개 회수 3/5, 회수 성공 사례의 저장
보존 3/3이며 성공 관측 뒤 새 추론·정책 행동 0회다. 나머지 2회는 회수 기준 미달이다.
저장 level.dat/playerdata를 직접 검사해 모든 최종 인벤토리와 일치함을 확인했다.
기술 오류·미실행·추가 API 0, 입력 해제·GUI 닫기·저장·자식 정리 확인, 잔존 0개,
STOP 유지, 원본 월드·연구 명세 유지. Windows 110개 통과/4개 제외, WSL 114개 통과.
일반 대화/RM 제작은 아직 미지원이며 회수 이전 실패는 남아 있다.
[실제 결과](craft-retention-validation-2026-10-04.md). 아래는 이전 단계 당시 기록이다.

## 판자 제작·회수 선행 실험 완료 (2026-10-04)

자연 수집 원목 1개가 있는 월드의 복사본으로 고정 시드 5회×180초를 실행했다.
판자 제작 통계는 5/5 증가, 원목 소비·실제 판자 4개 회수 증거는 3/5였다. 이후
작업대·버튼·막대기 제작으로 판자를 소비해 최종 판자는 0/5였다. 목표 달성 즉시
중단·보존은 다음 보완 항목이다. 기존 수집 RM에 조합 지원을 추가하지 않았다.
기술 오류/미실행 0, 전 시행 입력 해제·GUI 닫기·저장·STOP 후 행동 0회·자식 정리 확인.
저장 NBT 인벤토리도 최종 관측과 일치. 커서 물품 자체의 직접 관측은 미지원으로
완전한 반환 검증을 주장하지 않는다. Windows 109개 통과/4개 제외, WSL 113개 통과.
추가 API 0회, 원본 월드·연구 명세 유지, STOP 유지. [실제 결과](crafting-validation-2026-10-04.md).

## 직접 실행·대화·저장·재개 통합 (2026-10-04)

Start-MinecraftRL.ps1을 추가했다. 기존 대화·기억·과업·월드를 재사용하며 상태 표시와
저장 후 종료를 제공한다. 출력 폴더 충돌, WSL 표준 입력 공유, 완료 과업의 새 요청 거부,
자연 완료가 cancelled로 기록되는 경합을 발견해 수정했다. 시험 전용 실제 월드에서
원목 목표 실행·일시정지·저장→복원·흙으로 수정·1개 획득→완료 과업 복원·새 요청·총 2개를
확인했다. Windows 104개 통과/4개 제외, WSL 108개 통과. API 추가 0회, 잔존 0개,
STOP 유지. 조합은 현재 미지원·미검증이며 판자부터 별도 검증할 대상으로 정리했다.
[사용법](companion-use.md), [실제 실패·수정·검증](companion-validation-2026-10-04.md),
[조합 검토](crafting-readiness.md).

## 같은 프로세스 재초기화 실험 종료, 복구 효과는 미검증 (2026-10-04)

자연 정체 시 한 번만 resume하는 실험기와 worker PID 증거, 진행도/완료 이력 보존
검사 2개를 추가했다. 실제 고정 시드 5회 모두 흙 1개를 처음부터 획득했고 정체·재시도는
0회였다. 따라서 초기 수집 5/5이며 복구 성공률의 분모는 0이다. 같은 프로세스에서의
실제 복구 효과는 미검증으로 유지한다. 기본 자동 재시도는 켜지 않았다.
Windows 97개 통과/4개 제외, WSL 101개 통과. 원본 월드·연구 명세 유지, 전 시행
중립 해제·저장·자식 종료 확인, 잔존 프로세스 0개, STOP 유지. 추가 API 0회.
[실행·측정·미검증 범위](inplace-recovery-validation-2026-10-04.md).

## 실패 월드의 정책 문구 비교 완료 (2026-10-04)

원래 실패 상태를 독립 저장소로 복사해 기존 5개 시드 × short/official 두 문구를
모두 실행했다. short 5/5, official 4/5 수집 성공. 공식 문구의 마지막 시드는 흙 0개,
밀 씨앗 1개 및 60초 no_progress로 기록했다. 미실행/기술 오류는 0개이며 원본
월드 해시·current pointer 유지, 전 구간 입력 해제·저장·자식 정리, STOP 유지 확인.
기본 short 문구와 기존 명시적 재개 경계를 유지한다. 자동 복구는 추가하지 않았다.
Windows 95개 통과/4개 제외, WSL 99개 통과. 새 옵션에 의한 기존 Namespace 호환
회귀를 검사에서 발견해 수정했다. 추가 API 0회, 누적 37회 유지.
[실제 비교와 한계](pickup-comparison-validation-2026-10-04.md),
[격리 저장소·문구 옵션](policy-prompt-profiles.md).

## 정체 감지·복구 경로 구현, 실제 수집 실패 보존 (2026-10-04)

no_progress 감지·paused/새 revision·입력 해제·실패 근거 schema 3 저장/복원 및
대화 상태 갱신을 구현했다. Windows 91개 통과/4개 제외, WSL 95개 통과.
실제 STEVE-1에서 5초 검사 임계값으로 정체 후 저장, 같은 월드의 실패 근거 복원,
남은 계획 수정·명시적 재개까지 확인했다. 재개 후 흙 1개 수집은 60초 동안 0개로
실패했고 다시 paused가 됐다. 바닥의 드롭을 줍지 못한 정황을 프레임과 행동 로그로
보존했다. 전체 acceptance=false, 제어 증거 control_passed=true로 구분했다.
각 정체 및 STOP 후 새 행동 0회, 입력 해제·월드 저장·소유 자식 정리 확인.
추가 API 0회, 기존 누적 37회 유지. [사용법](failure-recovery.md),
[측정·실패 원인 분석](failure-recovery-validation-2026-10-04.md).

## 순차 수집 RM 구현·종합 검증 완료 (2026-10-04 12:09 KST)

추천안 A 및 추가 API 최대 30회 승인에 따라 1–8단계 선형 수집 과업, 단계별
관측 전환, 완료 이력, 현재/남은 단계 수정, 체크포인트 schema 2, 대화/PowerShell
연결 및 한 번에 실행하는 검증 명령을 구현했다. Windows 85개 통과/4개 제외,
WSL 89개 통과. 실제 게임에서 원목 1개→흙 단계→수량 수정→저장·복원→흙 총 3개
및 전체 과업 완료를 확인했다. 별도 취소는 추론 결과 1개 폐기, 게임 dispatch 0회.
STOP/취소 후 새 행동과 잔존 프로세스 0개. 추가 API 총 10/30회, 누적 37회.
최종 문구 정리 뒤 추가 의미 검사 2회와 관련 로컬 검사 24개를 포함한 수치다.
상세: [구현·사용법](sequence-rm.md), [종합 검증 결과](sequence-rm-validation-2026-10-04.md).

## 고정 문맥 7건 실제 호출 (2026-10-04 11:41 KST)

사용자 명시 승인으로 추가 API 7회 완료. 의도 라벨 7/7, 전체 필드 6/7 일치.
목표 질문의 dirt/64 주제 필드 차이는 비실행 계약상 허용되지만 불일치로 보존했다.
이전 실패와 동일한 전체 user payload에서는 이번에 TASK_PROPOSAL을 반환했다.
실제 후보의 오프라인 재생에서 현재 흙 과업 재개/수정/취소 및 stale 확인 차단을
검사했다. 입력 10630/출력 168 tokens, 새 게임 0회, STOP 유지, 누적 호출 27회.
이번 한도 잔여 0회. [실제 문맥 검사 결과](semantic-context-live-validation-2026-10-04.md).

## 현재 과업과 과거 목표가 다른 문맥 평가 준비 (2026-10-04)

과거 실패와 동일한 실제 user payload 1건 및 새 표현의 합성 문맥 6건을 고정했다.
과거 원목 목표와 현재 paused 흙 과업을 분리한다. 평가기에 입력 문맥 해시 검사를
추가해 같은 프롬프트라도 다른 문맥의 응답이면 거부한다. 합성 기대 응답을 주입한
어댑터/과업 확인 계약 검사 7건 통과, 실제 모델 7건은 전부 미실행이다.
추가 API·실게임 0회, STOP 유지. [입력·재현·한계](semantic-context-evaluation.md).
최종 WSL 81개 통과, Windows 77개 통과/4개 Linux 전용 제외.

## 보완된 지침의 실제 10개 대조 사례 (2026-10-04 11:31 KST)

후속 계속 요청에 따라 실제 API 추가 10회, 자동 재시도 없이 완료했다. 의도 라벨과
전체 후보 필드 모두 사전 기대값 10/10 일치. 입력 7022/출력 238 tokens, 새 게임 0회.
실제 후보를 로컬 기억에 재생하고 재시작해 미확정 제안 2개 보존·무행동을 확인했다.
지침에 나온 예시와 겹치고 과거 실패와 문맥이 달라 일반화/인과적 개선 증거로 보지 않는다.
STOP 유지, 누적 유료 호출 20회, 이번 한도 잔여 0회. 상세:
[실제 대조 사례 결과](semantic-contrast-live-validation-2026-10-04.md).

## 의도 분류 지침 및 오프라인 평가 (2026-10-04)

TASK_PROPOSAL과 CHAT 등 라벨 정의·대조 예시를 PLANNER에 보완했다. 실제 응답
라벨이나 과거 기억은 덮어쓰지 않았다. 기록 기반 평가기에 프롬프트 해시, 미실행/
응답 오류/의도 라벨/전체 필드 일치 구분을 추가했다. 과거 실제 선택기 4건 재평가는
일치 3/불일치 1 그대로이며 새 지침의 대조 사례 10개는 모두 미실행이다.
추가 API·실게임 0회, STOP 유지. 상세:
[분류 지침·재평가](semantic-taxonomy-validation-2026-10-04.md).
최종 WSL 80개 통과, Windows 76개 통과/4개 Linux 전용 제외.

## 실제 LLM·실게임 통합 (2026-10-04 10:50 KST)

사용자의 후속 승인으로 실제 API를 추가 5회 호출했다(기존 5회와 별도, 잔여 0회).
gpt-5.3-codex의 수량 명료화 → 문맥 기반 흙 64개 제안 → 확인 → 실제 흙 2개
획득·저장 → 복원 → 미래 집짓기 제안 비실행 → 문맥 기반 resume → 흙 총 4개
획득·저장을 검증했다. 모든 API 응답 및 로컬 처리 완료, 입력 5075/출력 162 tokens.
미래 제안을 TASK_PROPOSAL 대신 CHAT으로 분류한 1건은 명시적 품질 한계로 기록했다.
확인 전 및 pause/STOP 후 새 정책 행동 0회, 입력 해제 확인, 프로젝트 잔존 프로세스
0개, STOP 유지. 개인 기록은 사용하지 않았다. 이번 운영 코드 변경/단위 검사 재실행은
없다. 상세: [실제 LLM·월드 통합](live-semantic-world-validation-2026-10-04.md).

## 대화·기억·실게임·월드 복원 연결 (2026-10-04 10:37 KST)

Responses 어댑터에 고정 합성 응답을 주입하고 실제 STEVE-1에서 제안 확인 → 흙
2개 획득 → pause/저장/종료 → 월드·과업·대화 기억 복원 → 새 확인으로 재개 →
흙 총 4개 획득·저장을 검증했다. 세션 STOP 미검사 및 중첩 폴더의 공통 STOP
전달 누락을 회귀 검사로 수정했다. 확인 전, pause/STOP 후 정책 행동 모두 0회,
입력 해제 확인, 프로젝트 Python/Java 잔존 0개. STOP 유지.
WSL 75개 통과, Windows 71개 통과/4개 제외. 합성 응답 10회, 추가 유료 API 0회.
실제 LLM의 이해/기억 품질 검증은 아니다. 다음은 별도 호출 예산 승인 후 실제
LLM+게임 결합 검사이며, 개인 기록 연결은 하지 않았다.
상세: [대화·월드 통합 결과](semantic-world-validation-2026-10-04.md).

## 월드 저장·재시작 및 과업 재개 (2026-10-04 09:34 KST)

후속 최종 검사에서 시작 UUID가 실행마다 달라지는 문제도 발견해 슬롯별로 고정했다.
새 시나리오 `world-scenario-20261004T004053114851Z`에서 실제 흙 2→4개, 같은 UUID의
NBT/playerdata 유지, 위치 차이 0.116890블록, 명시적 재개 전 무행동을 재검증했다.
최종 WSL 72개 통과, Windows 68개 통과/4개 제외. 아래 09:34 결과는 UUID 수정 전 단계다.

사용자 요청으로 session의 기본 companion 저장 슬롯과 자동 불러오기를 추가했다.
최초 실제 재시작에서 인벤토리 불일치를 검출하고 정책 행동 0회로 중단했다. 저장 NBT와
고정 MineRL 소스를 통해 종료 시 최신 플레이어/월드 저장 누락을 확인했다. 설치 원본을
보존하고 별도 세션 JAR에 저장 flush만 추가했다.
수정 후 새 실험에서는 실제 흙 2개/씨앗 1개 복원, 위치 차이 0.011796블록, paused 복원과
명시적 resume 뒤 흙 총 5개 획득·재저장까지 통과했다. 중단 후 정책 행동 및 소유 프로세스
잔존 0개. 의도적 STOP 시나리오로 raw cancelled/invalid는 보존하고 정상 정책 표본에 합치지 않는다.
WSL 71개 통과, Windows 67개 통과/4개 제외. 추가 API 호출 0회. 상세:
[사용법](world-persistence.md), [실게임 결과](world-validation-2026-10-04.md).

## 대화 기억 및 과업 기록 복원 (2026-10-04)

최근 대화 12개 + 출처 있는 분류별 발췌 최대 24개를 사용자별로 저장한다. 과업 상태는
별도 체크포인트와 이벤트 기록으로 유지한다. 콘솔 재시작 시 발행 직후 명령 중복과
진행도 지연을 재현해 수정했다. 이전 확인 토큰은 복원하지 않으며 새 월드에 이전
획득량을 적용하지 않는다. 게임 월드/worker 자동 복원은 미구현이다.
WSL 61개 통과, Windows 57개 통과/4개 제외. PowerShell 구문 및 합성 기록 조회 통과.
이번 추가 유료 호출 0회, 새 실게임 0회. STOP 및 연구 명세 유지.
상세: [memory-persistence.md](memory-persistence.md). 아래 항목은 이전 단계 이력이다.

## 승인된 실제 API 5회 검증 (2026-10-04 08:30 KST)

MyWife 설정을 읽기 전용으로 사용해 gpt-5.3-codex를 정확히 5회 호출했다.
API 응답 완료 5회, 로컬 정상 처리 4회와 검증 거부 1회. 수집 요청/수량 명료화/합성
대화 문맥/관측 상태 기반 persona 응답을 확인했다. 개인 기록 및 새 실게임은 사용하지 않았다.
입력 1,374 / 출력 123 tokens. 비실행 intent에 주제 goal/count가 붙을 때 로컬 validator가
거부하는 schema 불일치를 합성 회귀로 재현/수정했다. 정확한 실패 응답 본문은 진단 도구가
저장하지 않아 확인 불가임을 보고서에 명시했다. WSL 51개 통과, Windows 47개 통과/4개 제외.
수정 후 실제 API 재시험은 5회 한도 소진으로 미실시. 추가 호출 승인을 추정하지 않는다.
상세: live-api-validation-2026-10-04.md. 기존 STOP 및 연구 명세 해시 유지.

## MyWife 참고 의미 어댑터 및 대화 맥락 준비 (2026-10-04)

MyWife 실행 코드를 읽기 전용으로 참고했다. Responses API 의미 선택기와 별도 persona 응답,
최근 대화 12개, 명시적으로 선택한 사용자 기록 읽기 연결을 구현했다. 모델 제안은 과업
revision/status/epoch 및 120초 유효 기간에 묶인 확인 뒤 기존 파일 명령으로 전달한다.
수량 누락은 명료화, persona 출력은 명령으로 해석하지 않는다. 기존 제한 문법은 보존했다.
WSL 전체 50개 통과, Windows 46개 통과/4개 Linux 검사 제외. API malformed 응답 오류도
재현 후 수정했다. 상세는 semantic-integration.md 및 semantic-progress.md 참조.

실제 API 호출 0회, 실제 MyWife 개인 데이터 접근/복사 0회, 새 실게임 실행 0회다.
기존 STOP 유지 및 연구 명세 해시 동일. 유료 API 사용은 사용자가 정한 별도 승인 범위다.
남은 확인: 사용할 모델/프로세스 키 설정, 승인된 소규모 실제 API 검증, 이후 LLM+정책 통합.
장기 기억 updater/호감도 변경/자율 기억 축적 및 전체 자유 대화 품질은 아직 검증하지 않았다.

## 대화 실행 경계 완료 (2026-10-04 07:28 KST)

제한된 한국어 문법 offline-ko-v1, 별도 commitment 검증, 메시지 중복/상충 처리,
PowerShell 대화 입력을 기존 과업 제어기에 연결했다. 실제 동일 월드 STEVE-1에서 비실행
문장 6개로 인한 명령 없음, 요청/수정/pause/resume 수락, 49.912초에 실제 흙 5개 획득으로
목표 4개 충족을 검증했다. 수정 후 이전 revision 행동 및 pause 중 행동은 0회다.
별도 PowerShell 한국어 취소 시험은 22 step 뒤 수락, 이후 정책 행동 0, 입력 해제/자식 정리
및 사용자 CLI exit 0을 확인했다. 중복 요청 1개와 같은 ID의 변경 메시지 1개도 각각
중복/거부로 처리했다. 마지막 프로젝트 Python/Java 검사 잔존 0개, STOP 유지.

WSL 39개 테스트 통과. 기존 100ms 시작 제한의 테스트 경합을 재현/수정했으며 실제 감독
코드는 변경하지 않았다. 연구 명세 SHA256 동일. 작업 기록 `dialogue-progress.md`,
사용법 `dialogue-interface.md`, 실게임 근거 `dialogue-validation-2026-10-04.md` 참조.
이는 LLM 기반 자유 대화의 완성이 아니다. 다음 미구현 범위는 실제 LLM semantic adapter,
확인/명료화 대화 및 과업 맥락·기억이며 원래 장기 연구 목표를 유지한다.
유료 API/Colab/GitHub push는 수행하지 않았다. 아래 기록은 이전 단계 이력이다.

## 과업 제어 통합 완료 (2026-10-04)

사용자가 승인한 후속 시나리오를 구현하고 실제 STEVE-1/MineRL에서 검증했다.
동일 월드 reset 1회, 원목 revision 1 → 흙 revision 2, pause/resume, 실제 흙 4개 획득으로
104.713초에 완료했다. 오래된 추론 2개 폐기, 수정 후 이전 revision 행동 0, pause 중 행동 0.
별도 PowerShell 과업 취소 및 정상 CLI 반환도 확인했다. 최신 WSL 하네스 33개 통과.
두 건의 기록/반환 오류를 회귀 테스트 후 수정했다. 전체 작업·검토 기록은
`task-session-progress.md`, 사용 방법은 `task-session.md`, 결과는
`task-session-validation-2026-10-04.md`에 있다.
기존 정책 벤치마크 15회와 신규 기능 검증은 분리한다. 원문 연구 명세 SHA256은 그대로다.
최종 프로세스 점검에서도 잔존 0개이며 실행 중인 실험은 없다. 완료 표시 STOP을 유지한다.
다음 범위는 구조화 명령 앞의 대화 의도/commitment 해석이며 아직 구현했다고 주장하지 않는다.

## 이번 로컬 선행 검증 완료 (2026-10-03 21:28 KST)

VPT 고정 5개 시드의 180초 실행 완료: 유효 5/5, 총 5798 step, 자발적 아이템 획득
4/5. 모두 neutral_command_sent, 소유 자식 잔존 0개다. 상세는
`local-vpt-results-2026-10-03.md`에 기록했다. STEVE-1은 나무 5/5 및 흙 5/5 실행 완료,
10회 모두 목표 아이템의 실제 인벤토리 순증가를 확인했다. 총 12721 step, 기술 실패 0,
시간 제한 이후 새 정책 dispatch 0, neutral_command_sent, 소유 자식 잔존 0개다.
세부 수치는 `local-steve1-results-2026-10-03.md`에 기록했다. suite 경로는
`results/suite-steve1-20261003T111126674108Z`다.

두 정책의 초기화/추론/step 외부 STOP 6조건을 확인했다. 요청 및 acknowledgment 이후
새 정책 dispatch 0, 소유 자식 잔존 0개다. 추론/step에서는 neutral_command_sent를
확인했고 초기화 전에는 정책 입력이 없어 해제가 해당되지 않았다. 초기화는 약 10초 유예 후
감독 강제 종료였다. 세부 지연과 원본 오류/재시험은 `local-stop-results-2026-10-03.md` 참조.
실제 추론 중단에서 요약 기록 손실을 발견·수정했고 회귀 포함 22개 하네스 검사가 통과했다.

최종 별도 WSL 프로세스 점검에서도 프로젝트 Python/Java 잔존 0개였다
(`results/final-process-check.json`). 마지막 진단의 `results/STOP`은 유지하며 실행 중인
실험은 없다. 다음 실행 전에 마지막 STOP 결과를 검토하고 파일을 보존 이동한다.
정상 실게임의 OOM은 없었으므로 Colab 이전은 필요하지 않았다. Colab 연결/유료 자원/GitHub
push는 수행하지 않았다. 연구 명세 20개 항목의 SHA256은 원본과 일치한다. 장기 통합 연구와
최신 Bukkit 호환성은 이번 선행 검증으로 완료됐다고 간주하지 않는다.
아래 사용자 일시 중단은 과거 이력이다.

## 재개 (2026-10-03 19:46 KST)

사용자의 재개 요청으로 STOP 파일을 results에 보존 이동하고 VPT seed 104729,
제어 제한 180초의 실제 실행을 시작했다. 재개 직전 GPU 사용 691 MiB / 여유 2275 MiB.
아래 중단 기록은 이전 시점의 이력이며 현재 실행 상태를 뜻하지 않는다.

## 사용자 요청으로 일시 중단 (2026-10-03 저녁)

사용자가 실행 중단을 요청했다. `results/STOP`을 유지하며 자동 재시작하지 않는다.
최신 실행 `results/episode-vpt-20261003T091816811746Z`는 cancelled이고 소유 자식
프로세스 잔존 0개다. 중립 입력 해제는 미검증이며 실제 정책 gameplay 표본은 0개다.
`results/user-pause-process-check.json`에 별도 프로세스 종료 확인을 기록했다.

재개 위치:
- 두 정책의 격리 환경, 공식 가중치, MineRL/MCP JAR 및 cp39/cp310 wheel 설치 완료.
  JAR SHA256은 `f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48`.
  wheel은 `.runtime/artifacts`에 보존했고 setup/installed-packages receipt는 results에 있다.
- VPT pip check 통과. STEVE-1은 문서화한 MineCLIP/MineDojo gym==0.21.0 충돌만 남는다.
  cloudpickle 1.6.0, cmake 3.26.4, lit 16.0.6을 보완했다.
- 빌드/런타임 JAR의 WSL drvfs I/O 병목을 실제 스택으로 확인했다. 빌드 scratch 및
  동일 SHA의 실행 JAR 캐시만 Linux 임시 폴더에 둔다. 프로젝트/설치 prefix/게임 cwd는 D:다.
  `scripts/minerl_build_cache.py`, `scripts/Install-MineRLWheel.ps1`,
  `src/minecraftrl/minerl_runtime.py`에 재현 가능한 처리 추가.
- 첫 실행 `episode-vpt-20261003T091018259199Z`는 초기화 STOP 진단이다.
  Java JAR 스캔 중 중단, 정책 행동 0, 소유 프로세스 잔존 0. 초기화 전 release 오류를
  재현해 수정했다. 원본 결과는 error 그대로 보존했으며 정상 5-seed 실험 표본에 넣지 않는다.
- 최신 실행은 JAR 캐시 복사를 검증한 뒤 사용자 요청으로 중단됐다. 캐시 적용 후 실제
  Minecraft reset/step 성공, 추론/step 중 STOP, 5개 seed 행렬은 여전히 미검증이다.
- 변경 후 WSL 하네스 20개 통과 (`results/wsl-unit-tests-runtime-cache.log`).
  이는 실게임 검증이 아니다. 실행 중 다른 게임의 GPU 사용도 관측됐으므로 새 실험 때
  가용 VRAM과 동시 사용 프로그램 상태를 다시 기록한다. 정책 OOM은 아직 관측하지 않았다.
- WSL 재시작 후 /tmp는 지워질 수 있다. 설치된 패키지/가중치/wheel과 결과는 D:에 남는다.
  이전 임시 빌드 경로를 재사용하지 말고 보존 artifact 및 설치 환경에서 재개한다.
- 재개를 요청받으면 최신 종료 로그 확인 후 STOP을 보존 이동하고 VPT 실제 실행부터
  계속한다. 재승인/초기 설계 질문, 전역 설정 변경, Colab 실제 연결, GitHub push는 하지 않는다.

## 로컬 작업 재개

Windows 머신 DESKTOP-V0HJPC2에서 지정 경로에 실제 설치했다. 전달본 57개 파일의
설치 전후 해시를 검증했고 충돌은 없었다. WSL Ubuntu에서 하네스 15개 테스트 통과,
프로세스 범위 D3D12 설정에서 GPU 가속 렌더링을 확인했다. VPT 환경 설치와 가중치
다운로드를 시작했다. 세부 근거와 미실행 항목은 `local-validation-2026-10-03.md` 참조.

## 이전 Linux 클라우드 전달 작업 기록

계획 재승인 없이 실행했다. 원문 연구 명세가 권위 문서이며 이번 범위는 정책 선행 검증이다.

- Task 1 문서: 20개 최상위 항목 보존, 추가 결정 별도 작성. 현재 파일 SHA256은 research-spec.integrity.json에 기록. 독립 원본 업로드와의 byte 비교는 없으며 수신 본문 전사본이다.
- Task 2 구성: 소스 commit/model 출처 manifest, 환경 후보, Python 패키지/PowerShell/WSL 스크립트 작성. 모델·데이터·비밀·결과 Git 제외.
- Task 3 공개 정책: 두 공식 추론 어댑터 작성. cloud 실제 진입은 blocked. Windows/GPU 실제 실험 미완료.
- Task 4 실행 제어: 제한·취소·입력 해제·프로세스 소유권·로그/지연/인벤토리/메모리 계측 구현. 합성 테스트 통과; 실게임 중단은 미검증.
- Task 5 실험: 5개 seed/조건 manifest 및 실행기 작성. 실제 rollout 0회. null/unrun과 기술 실패 구분.
- Task 6 Colab: 동일 실험 노트북 작성/구문 검사. 실제 연결/실행 미착수.
- Task 7 보고: validation-report.md에 차단 원인/증거/한계/다음 실행 기록.

Ruling: Windows 경로가 없는 클라우드에서 전달 준비본을 작성 — 사용자 최신 '알아서 하도록' 지시에 따라 가능한 작업 진행. 실제 Windows 설치로 오인하면 안 되며 설치기는 원래 경로로 고정.
Ruling: 원문 Stage 5를 삭제/축소하지 않고 공개 정책 검증만 선행 — 사용자 명시 추가 합의. 장기 연구 기여는 아직 평가 전.
Ruling: torch/Python을 정책별 후보 환경으로 분리 — VPT/STEVE-1 공식 요건 차이. 후보 설치 및 행동 재현은 후속 검증 필요.
Ruling: MineRL fork gym 커밋 고정 및 알려진 MineCLIP/MineDojo gym 메타데이터 충돌 기록 — 공식 의존성 충돌. 예상 밖 충돌은 중단; 이 선택이 작동하지 않으면 설치/실행 단계에서 실패를 보고.
Ruling: MineRL build 복사본의 cache 경로/MCP revision만 patch — 기존 환경 오염 감소 및 버전 재현. 실제 빌드는 미검증.
Ruling: 최신 버킷 버전을 지금 임의 고정하지 않음 — 실제 공통 지원/연결이 미확인. 정책 실험과 별도의 서버 테스트 필요.

최종 독립 리뷰: 중요 3건, 수정 전 재현 → 수정 후 15/15 통과. 추가 집계 누락 회귀 검사 포함. 미해결 minor 지적 없음. 전체 기록은 docs/validation-report.md 및 docs/evidence/ 참조.
