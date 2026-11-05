# 목표 수량 이어하기 — 완료 검증, 2026-10-16

목표는 같은 월드/원래 목표 이력에서 실제 원목64개·흙4개를 보유하는 것이다.
최종 memory-resume/1은 **원목64개(참나무52+자작나무12)·흙25개·체력8.167**다.
서버와 저장 NBT 재고 일치, 두 목표 completed, 완료 뒤 새 추론/정책 전송0·서버 키0·잔존 소유 PID0을 확인했다.
수집 목표 완료이며, 앞선 사망/저체력 분기와 명시적 되감기를 포함하는 개발 이어하기 결과다. 장기 연구 완료로 확대하지 않는다.
최종 세대 `results/goal-completion-20261016-memory-resume/worlds/priority-dev/2b50d8ba1ad24cb7873266d456d29b4c`.
전수 완료 감사 `results/goal-completion-20261016-preparation/goal-completion-audit.json`.
최신 상태는 [progress](progress.md), 조건 변경 전 기록은 [프로토콜](goal-completion-protocol-2026-10-16.md).

모두 로컬 WSL/GTX1060 3GB, 개발 seed104761의 이어하기다. 독립 성공률 표본이나
최종5시드 평가로 세지 않는다. 새로운 학습 갱신·유료 API·Colab·GitHub 발행은0이다.

| 결과 폴더 (`results/goal-completion-20261016-` 뒤) | 실제 정책 행동 | 원목 보유 변화 | 판정 |
|---|---:|---:|---|
| baseline/0 | 677 | 2→3 | 흙 목표 충족, 최종 실제 흙8 |
| checkpoint/0 | 0 | 미실행 | 로딩 중 흙8→9 차이로 기술 차단; 정책 실패 표본 제외 |
| resume/0 | 817 | 3→3 | 명시적 startup pickup 검증, 최종 흙21 |
| official/0 | 846 | 3→5 | 공식 나무 prompt, 최종 흙26 |
| official/1 | 815 | 5→5 | 순증가 없음, 배치 종료 후 진단 |
| recovery/0 | 790 | 5→7 | 서버 재고 판정/키 해제 계측, 복구 미호출 |
| recovery/1 | 793 | 7→10 | 복구 미호출 |
| recovery/2 | 845 | 10→10 | 수집777+탐험68, 회차 제한에 의한 복구 중단; 흙19 |
| recovery-resume/0 | 802 | 10→12 | 중단된 복구22행동 재개+다른 위치에서 탐험51행동, 아래 참조 |
| recovery-resume/1 | 838 | 12→16 | 정상 수집, 저장 NBT/서버 키0/잔존 소유 PID0 |
| recovery-resume/2 | 799 | 16→15 | 수집668+탐험131, 이탈했으나 순증가 없음; 최종 흙17 |
| prewarm/0 | 866 | 15→16 | 준비 비용 분리, 저장 NBT/서버 키0/잔존 소유 PID0 |
| prewarm/1 | 801 | 16→20 | 같은 조건, 저장 NBT/서버 키0/잔존 소유 PID0 |
| prewarm/2 | 795 | 20→24 | 같은 조건, 저장 NBT/서버 키0/잔존 소유 PID0 |
| prewarm/3 | 867 | 24→26 | 같은 조건, 저장 NBT/서버 키0/잔존 소유 PID0 |
| prewarm/4 | 876 | 26→32 | 같은 조건, 저장 NBT/서버 키0/잔존 소유 PID0 |
| prewarm/5 | 896 | 32→35 | 중간36개에서 감소, 최종 저장 NBT/서버 키0/잔존 소유 PID0 |
| prewarm/6 | 834 | 35→34 | 복구 경보/전환0, 순증가 없음으로 종료; 저장/중단 정상 |
| visual/0 | 747 | 34→40→0 | 실제 사망+server/client 종료 지연 처리 오류; 실행 유효성 false, 정상 저장 없음 |
| damage/0 | 845 | 34→36 | 명시적 체크포인트 재시도, 체력19.167 유지; NBT/STOP/키 해제/소유 PID 정리 정상 |
| damage/1 | 184 | 36→36 | 피해 복구2회 이탈 뒤 재피해; 체력8.167에서 예산 중단, NBT/STOP/키 해제/소유 PID 정리 정상 |
| damage-stability/0 | 814 | 36→38 | 피해 복구100틱·21.282블록 이탈 후 수집, 추가 피해0; NBT/STOP/키 해제/소유 PID 정리 정상 |
| damage-stability/1 | 310 | 38→38 | 나무 옆 좀비 접근 및4.5HP 감소 후 저체력 중단; 최종 체력3.667/흙18, NBT/STOP/키0/소유 PID0 |
| threat/0 | 879 | 38→45 | 명시적 직전 체크포인트 비교, 추가 피해0; 위협 센서 양성0, NBT/STOP/키0/소유 PID0 |
| threat/1 | 742 | 45→45 | 피해 전13.784블록 위협 감지→탐험125행동, 체력 유지/안정 조건 미달; 정상 저장·중단 |
| threat-budget/0 | 265 | 45→45 | 누적 예산 재개 후 안정 기준 충족→즉시 재위협→재회피 중5HP 피해, 체력3.167에서 정상 저장·중단 |
| flee/0 | 119 | 45→45 | 명시적 조건 비교,118행동 무입력/수평 이동0,3HP 피해 후 체력5.167에서 정상 저장·중단 |
| flee-guidance/0 | 313 | 45→39 | guidance0,14.952블록 이동했으나 추격 지속/3HP 피해; 체력5.167에서 정상 저장·중단 |
| pursuit/0 | 365 | 45→45 | 명시적45개 복원 후41.672블록 이동/피해0; watcher OSError61 기술 취소, 실행 유효성false; 정상 저장·STOP/키0/PID0 |
| pursuit-resume/0 | 108 | 45→45 | 추격 안정 기준 충족→수집 복귀 뒤 낙하6HP 피해; 실제 저장 체력2.167, 정상 저장·STOP/키0/PID0 |
| ground/0 | 732 | 45→44 | guidance2/추격0·피해0,9.604블록 이후 점프 반복; 거리 복귀 조건 진단 중단cancelled, 정상 저장·STOP/키0/PID0 |
| clear-handoff/0 | 471 | 44→28 | 접지·추격 해소 뒤 수집 복귀; 원목17개 소비/1개 회수, 진단cancelled, 정상 저장·STOP/키0/PID0 |
| collection-resume/0 | 825 | 28→34 | 조건 유지/되감기0, 감소0·피해0; 정상 저장·STOP/키0/PID0 |
| collection-resume/1 | 856 | 34→30 | 소비5/획득1, 원목 드롭5개 미회수; 무순증가 배치 종료, 정상 저장·STOP/키0/PID0 |
| collection-guidance/0 | 471 | 30→29 | root guidance2, 이동 후 몹 접근/4HP 피해, 저체력 중단; 정상 저장·STOP/키0/PID0 |
| retention/0 | 879 | 30→38 | 명시적 피해 전 복원, 소비0/획득8·피해0·복구 전환0; 정상 저장·STOP/키0/PID0 |
| retention/1 | 839 | 38→46 | 소비0/획득8·피해0·복구 전환0; 정상 저장·STOP/키0/PID0 |
| retention/2 | 813 | 46→43 | 소비8/획득5·피해0, 실제800틱 보유량 정체 경보→탐험13틱 후 회차 제한; 정상 저장·STOP/키0/PID0 |
| memory/0 | 810 | 43→45 | 첫 cold 회차, 정책 기억 최초 저장·정상 저장/STOP/키0/PID0 |
| memory/1 | 815 | 45→50 | 같은 기억 실제 복원, 소비0·피해0·정상 저장/STOP/키0/PID0 |
| memory/2 | 832 | 50→54 | 기억 복원, 획득5/소비1·피해0·정상 저장/STOP/키0/PID0 |
| memory/3 | 미기록 | 미실행 | CUDA 준비 중 호스트 재시작, 정상 종료/정책 성공·실패 표본 제외 |
| memory-resume/0 | 794 | 54→60 | 재부팅 전 기억 실제 복원, 소비0·피해0·정상 저장/STOP/키0/PID0 |
| memory-resume/1 | 549 | 60→64 | 두 목표 완료, 완료 뒤 추론/행동0·NBT 일치·키0/PID0 |

prewarm까지 정상 완료된 실행은 제어180초 상한 뒤 중립 입력, 정상 월드 저장, 잔존 소유 PID0을 확인했다.
상한 전에 전달된 진행 중 호출의 반환/정리 시간은 별도이며, 상한 뒤 새 정책 행동을 허용하지 않는다.
recovery 조건부터 실제 서버 held_key_count0을 기록했다. 이전 조건의 중립 응답을 그 계측으로 소급하지 않는다.
서버/클라이언트 재고 차이는 별도로 남긴다. recovery/0에서는 서버 원목6·클라이언트5인4관측이 있었다.
월드 NBT의 Time/DayTime도 prewarm/0~2에서9687→10514→11335로 이어졌다.
현재 생존 GameType0/Difficulty3, doDaylightCycle=true, keepInventory=false이며 이 조건을 변경하지 않았다.
근거는 `results/goal-completion-20261016-preparation/world-clock-audit.json`과 원본 저장본이다.

## 개선과 확인 범위

- 중복 checkpoint 저장을 제거하되 매 관측/명령/전환/종료 내구성 저장은 유지했다.
  동일20행동 모형45→25회,2.220→1.094초. 실제 resume 추론 시작 간격 중앙값186.162ms.
- 다른 목표 수행 중 획득한 실제 재고를 실행 가능한 목표에도 반영한다. 기존 흙 목표 완료 지연26.107초를
  재현했고 로그 재생에서 첫 충족 관측에 완료한다. pause/선행 조건은 우회하지 않는다.
- 이어하기에서만 명시적으로 근처 저장 드롭으로 설명 가능한 자연 획득을 허용한다.
  원래 정확 일치 검사는 기본으로 유지하고 자연 획득을 정책 행동 기여로 계산하지 않는다.
- 기존 그래프 정체 감지와 공식 `go explore` embedding을 다중 목표 실행에 연결했다.
  모터 행동은 신경망 출력이며 목표 수량/기준 재고를 수정하지 않는다. 전환 안에서 정책 기억을 보존한다.
- 서버 계측 조건의 목표 판정은 서버 인벤토리를 사용한다. 클라이언트69/서버5의 잘못된 완료를
  모형으로 재현·수정했다. 이 값 자체가 실제 게임에서 관측됐다는 뜻은 아니다.
- 중단된 복구를 실패와 분리하고 원본 영수증을 유지한다. `continued_from` 연결의 전체200틱/30초를
  사용하며 재시작으로 예산을 새로 주지 않는다. 새로운 프로세스의 신경망 기억은 초기화된다.
- 전환용 중립 입력에서 목표를 채우면 추가 탐험 행동이 없다. STOP 직전에 확인된 하위 목표 완료도 보존한다.

## 실제 복구 결과

첫 시도는 recovery/2의 정체200틱 뒤 시작했다. 68행동/14.976초에서 전체 회차 제한으로 중단됐다.
다음 실행이 남은132틱/15.024초를 이어받아22행동 뒤 시간 예산으로 복귀했다.
원래 진입점 기준 수평 이탈1.125블록이므로 이 시도는3블록 이탈 실패다.
두 구간의 실행 시간 외에 중립 전환/조건 변경 시간이 있어 영수증 벽시계 합은30.104초이며,
정책 행동 시간과 정리 시간을 혼동하지 않는다. 마지막 탐험 전달은 복귀 전 중립 입력보다 앞선다.

이후 다른 위치의 `stationary_repetition` 경보에서 새로운 탐험51행동/약11.047초로
**실제 수평3.245블록 이탈**, GUI 닫힘, 원래 수집 조건 복귀를 확인했다.
이것은 국소 이탈 기준 충족이며 원목 획득/벽 우회 일반화의 증거로 확대하지 않는다.
해당 전체 실행의 실제 저장 인벤토리는 참나무9+자작나무3·흙19다.

첫 재개 추론은9,935.987ms, 이후 중앙값51.534ms였다. 기존 RL 경로의 CUDA convolution
준비 함수를 공식 task session 경로에 연결했다. 준비 비용은 제어 시작 전 별도 JSON에 기록한다.
CPU/CUDA 난수와 가중치/eval 상태 보존 검사를 통과했고 실제 첫 추론 지연을 prewarm 배치에서 확인했다.

prewarm/0의 준비6.968초는 제어 시작 전이다. 첫 실제 추론991.381ms로 이전9,935.987ms보다 짧다.
이후 전체 추론 중앙값49.705/p95 58.398ms, control-cycle 중앙값189.540/p95 285.886ms,
추론 시작 사이 전체 간격 중앙값191.324/p95 290.725ms, 장치 GPU 사용 peak2793MiB다.
원목15→16이며 초기 상태/궤적이 달라 수집 성능의 인과 개선은 주장하지 않는다.
비교 근거는 `results/goal-completion-20261016-preparation/prewarm-latency-comparison.json`에 있다.

recovery-resume/2에서는 서버 tick156~556의401관측 중 공격247/사용15/점프8, 최장 연속 공격48행동,
서버 채굴134관측이었다. 같은 원목/위치 주변 반복 뒤 탐험131행동으로 이탈했으나 최종 원목15개다.
이탈만으로 원래 채집 목표를 달성했다고 보지 않는다. 최신15개·흙17개 저장본을 이어간다.
완료된 이탈 기록이 복귀 중립 처리 지연으로 timeout으로 바뀌던 별도 경계도 모형으로 재현·수정했다.

준비·검사·독립 NBT/전이 감사는 `results/goal-completion-20261016-preparation/`에 보존했다.
각 새 배치의 `source-at-game`, `work.json`, 원본 events/frames/worlds도 유지한다.
영상 목표 조건 추가 후 현재 회귀는 Windows 전체696개 중674통과·22제외, WSL 관련39개 통과다.
앞선 CUDA 준비 변경의 실제 CUDA 난수/가중치 보존 검사를 포함한38개 통과 기록도 보존했다.
이전 넓은 WSL45개와 중단 경계12개 검사 기록도 보존했다.

## 피해와 서버 사망 경계

visual/0에서 체력19.167→15.167→10.167→6.167→2.167→0을 관측했다.
frame724에는 스켈레톤, 마지막에는 약6블록 하강이 보인다. 최종 피해 원인을 단정하지 않는다.
747번째 정책 행동의 서버 사망과 달리 client life2.167/alive=true/done=false가 남아
navigation_state 검증에서 오류가 났다. 원본 error/execution_valid=false를 사후 성공/정상 실패로 바꾸지 않는다.
STOP 뒤 새 dispatch0, 서버 held_key_count0, 잔존 소유 PID0. 정상 저장 응답은 없어 세대에 승격하지 않았다.
잔여 파일 ZIP/NBT/해시는 준비 폴더 visual-0-residual-world.*에 별도 보존했다.

서버 사망 즉시 종료·전환 중 사망 차단 검사를 추가했다. 수집 중 피해≥2HP이면 기존 neural escape를
우선 요청하고, 피해 복구 중 새 원목 획득만으로 수집에 돌아가지 않는다. 체력≤8 또는 피해 복구 예산
소진 시 recovery_blocked로 제어를 멈춘다. 3블록 이탈은 여전히 국소 이동 기준이며 생존 보장은 아니다.
실제 로그를 오프라인 재생하면 첫 피해 frame537에서 전환을 요청한다. 반사실적 생존 증거가 아니다.
Windows 전체701개(679통과·22제외), WSL/CUDA 관련34개 통과; 실제 수정 효과는 다음 실행 대상이다.
첫 WSL 검사 명령의 CUDA 라이브러리 경로 누락 실패는 별도 로그에 보존했고 기존 프로젝트 환경으로 재검사했다.
명시적 재시도는 prewarm 최종34개 세계의 바이트를 그대로 복사하고 실패 분기의 목표 시간/복구 이력을
함께 보존한다. 시간/물리 상태의 체크포인트 되감기를 공개하며 uninterrupted 성공으로 세지 않는다.

damage/1에서 실제 피해에 따른 전환은 동작했으나 첫 수집 복귀2.970초 뒤2HP,
두 번째 복귀17.866초 뒤4HP 감소가 있었다. 첫 피해5HP를 포함해19.167→8.167이다.
두 번째 피해 때3블록 하강이 동반됐고 후반 화면에 스켈레톤이 있다. 모든 피해를 동일 원인으로 단정하지 않는다.
제어43초 수준에서 recovery_blocked로 종료하고 정상 저장했다. 목표 수량·기준 재고·이력은 유지된다.
수정 기준은 피해 복구에만12블록 수평 이탈+체력 감소 없는100서버틱을 적용한다. 미래 안전 보장은 아니다.
미해결 피해와 예산 소진도 복원하므로 재시작만으로 수집 재개/새 복구 예산이 생기지 않는다.
이 기준 수정 후 Windows 전체703(681통과·22제외), WSL 관련27검사 통과.
실제 damage-stability/0은100틱 동안 추가 피해 없이21.282블록 이동해 damage_stabilized를 관측했다.
이어 원목2개가 늘었으며 전체814행동 동안 추가 피해0, 최종38개/흙17/체력8.167이다.
복구→수집 재개 사례이며 기존 조건과 무작위 대조나 생존 일반화로 해석하지 않는다.

다음 damage-stability/1은 나무 옆 좀비 접근 화면과 함께 체력4.5 감소로 중단됐다.
이 원본을 보존하고 공격 대상 몹의 선제 관측을 opt-in으로 구현했다. 사전 전환/지속 위협 때 복귀 차단/
잘못된 센서 차단 모형을 포함해 Windows708(686통과·22제외), WSL39검사 통과, 별도 JAR 컴파일 통과.
threat 비교는 직전38/17/8.167 물리 상태에서 시작한다. 이후 실패 이력/목표 시간은 보존하고 되감기를 명시한다.
실제 회피 결과는 아직 진행 중이다. 초기 WSL 잘못된 검사 모듈 이름 실패 및 수정 재검사 로그도 보존했다.

threat/1에서 공격 대상 감지→전환은 피해 전에 동작했다. 최초거리13.784블록, 최소2.765블록,
탐험 종료 때 최초지점에서15.501블록 이동했고 전 회차 체력8.167을 유지했다.
하지만 마지막 양성이frame741, 마지막 정책frame742이므로 안정 구간은1서버틱이다. 완료 판정은 하지 않았다.
총30.239초/125틱의 위험 복구 뒤 recovery_blocked로 정상 저장했다. 단순 정체 탐험90행동과 구분한다.
근거는 preparation/threat-1-timing-diagnosis.json, threat-1-audit.json 및 원본 사건/화면이다.

threat-budget/0은 기존125틱/30.239초를 차감한 나머지275틱/59.761초로 재개했다.
100틱/18.598블록의 안정 관측 뒤 수집으로 돌아갔지만 약0.33초 후 새 위협으로 다시 탐험했다.
265행동에서5HP 피해, 최종45원목/18흙/체력3.167로 정상 중단·저장했다. 스켈레톤 근접 화면을 보존했다.
서버/NBT 일치·STOP 뒤 dispatch0·서버 held keys0·잔존 소유 PID0이다. 안전 일반화 실패로 기록한다.

위험 전용 실험 조건을 공식 prior로 생성했다. 원래 float16 출력을 수치 변경 없이 float32로 저장했다.
생성 준비의 dtype 가정 오류와 수정 성공 로그를 모두 보존했다. GPU peak725MiB, 생성45.518초이며
실제 정책 행동/최적화 업데이트/유료 호출0이다. 실험 조건은 validated_skill=false다.
일반 탐험→위험 조건 전환, 신경망 기억/누적 예산 보존, 저장 이력의 방법 hash와 재개 연결 검사를 추가했다.
마지막 수정 후 Windows 전체715개(693통과·22제외), WSL 관련33개 통과.
근거: preparation/flee-final-windows.log, flee-final-wsl.log, flee-method-chain-green.log.
이는 소프트웨어 검증이며 실제 회피 성공은 `goal-completion-20261016-flee`에서 별도 판정한다.

flee/0은119행동 중118개가 모든 키/카메라0, 나머지1개는use였다. 서버 위치 범위가 세 축 모두0이며
실험 조건으로는 실제 회피에 실패했다. 최초 위협7.241블록, 최소1.551블록, 체력8.167→5.167이다.
원목45/흙18, 저장 generation3a9df915f24742fda8963f607ef7a792, NBT/해시 일치,
STOP 뒤 새 dispatch0·서버 held keys0·잔존 소유 PID0이다.
근거 preparation/flee-0-action-diagnosis.json, flee-0-threat-diagnosis.json, flee-0-health-diagnosis.json, flee-0-audit.json.

동일 실패 화면64장/각각 초기 RNN/고정 난수의 오프라인 비교:

| 조건 | 무입력 버튼 확률 평균 | 이동/점프 버튼 확률 평균 | 샘플된 완전 무입력/64 |
|---|---:|---:|---:|
| flee guidance6 | 98.763% | 0.322% | 63 |
| flee guidance2 | 51.907% | 28.834% | 30 |
| flee guidance0 | 39.481% | 45.538% | 20 |
| 기존 explore guidance6 | 89.083% | 9.745% | 59 |
| 공식 영상 explore guidance7 | 98.941% | 1.021% | 63 |

강한 guidance에서 정지 쪽으로 출력이 집중됨을 확인했다. 화면을 고정한 비교이므로 실제 이동/안전/획득
성과는 판정할 수 없다. 원본 분포와 화면 해시는 `goal-completion-20261016-hazard-offline/diagnosis.json`에 있다.
위험 복구에만guidance0을 명시하며 기존 조건6 이력을 지우지 않는다. embedding과guidance 조합으로
실패 중복 및 중단 재개를 검사한다. Windows718(696통과·22제외), WSL36검사 통과 후 실제 비교를 시작했다.

실제 guidance0은313행동/67.910초의 위험 복구에서14.952블록 이동했다. 최초 위협7.277블록,
최소2.122블록, 최종 체력5.167/원목39다. 마지막 화면은 나뭇잎에 가까우며 안전 이탈을 충족하지 못했다.
정상 저장1bd602b5374b42c9b5184334f0f063c3, 서버/NBT 일치·STOP 뒤 dispatch0·키0·소유 PID0.
소프트웨어 회귀와 오프라인 움직임 확률 개선이 실제 회피 성공을 보장하지 않는 사례로 남긴다.
이후 pursuit-v5는 별도 활성 추격자 센서의 가림 상태를 포함한다. Windows721(699통과·22제외), WSL46검사 통과.

pursuit/0의 실제365관측 중41개는 LOS 위협0인데 활성 추격자가 있었다. 최소2.421블록이며
종료 때 활성 추격자1/14.983블록이다. 이동41.672블록·피해0이지만 안전 이탈은 완료하지 못했다.
감시 스레드의OSError61로 자동STOP 후cancelled/execution_valid=false였다. 원본 상태는 수정하지 않는다.
재개 명령 승인 후에도events를 읽던 불필요한 경로를 제거하고 STOP 전달만 유지한다.
원본은 상세 traceback이 없어 커널/파일시스템 원인을 단정하지 않는다. 같은OSError 주입으로
승인 전 오류는STOP+진단, 승인 뒤 읽기0과STOP 전달 유지 경계를 검사했다.
근거 preparation/pursuit-0-audit.json, pursuit-0-sensor-audit.json, watcher-green.log.

pursuit-resume/0은 남은435틱/98.802초만 재개했다. 추가101틱에서 원래 진입점54.559블록,
연속100틱 동안 활성 추격자/체력 감소 없음으로 복귀했으나 공중 상태를 제외하지 못했다.
약8블록 하강 후 실제 체력8.167→2.167, 총108행동/26.260초로 저체력 중단했다.
마지막 client life8.167을 실제 저장 체력으로 오인하지 않는다. 저장 NBT2.167/원목45/흙18,
generation961bb125a07546c4ab03019bb4fb407b이며STOP 뒤 dispatch0·서버 키0·소유 PID0이다.
새 watcher의resume-receipt를 확인했고 오류는 없었다. 이 한 회차로DrvFS 오류 자체가 완전히 해결됐다고 주장하지 않는다.
접지/낙하값을 검증해 공중에서 복귀하지 않는 검사를 추가했다. Windows726(704통과·22제외), 관련46검사 통과.

ground/0은732개 관측 모두 활성 추격자0/체력8.167이었다. 접지·낙하0·추격0인76개 관측 중
100틱 이후 첫 관측은frame110이다. 수평 거리9.604블록이12블록 기준에 미달해 복귀하지 못했다.
732행동/154.388초에서 agent 진단STOP을 걸었다. 원본cancelled/execution_valid=false를 재분류하지 않는다.
실제 저장44원목/18흙/체력8.167/food5, generation41dea1ed58fa4ce7a5dc54fd5f53cc9e,
NBT/STOP 뒤 dispatch0/키0/소유 PID0. 근거 preparation/ground-0-audit.json, ground-0-handoff-diagnosis.json.
새 직접 관측 기준의 오프라인 설명과 실제 다음 수집 성과를 분리한다. 과거 궤적을 사후 성공 표본으로 세지 않는다.
이 변경 후 Windows727(705통과·22제외), WSL47검사 통과.

clear-handoff/0은 위험 복구100행동 뒤 원래 수집 조건으로 전환했다. 남은 예산과 원래 진입점/이력을
유지했고 접지·추격자0·무피해 기준을 만족했다. 이후 원목 설치로44→27, 마지막28개가 됐다.
471행동에서 진단STOP을 요청해 cancelled/execution_valid=false로 보존했다. 저장e3fc71eb09de474297a4e2b7c797d2c5,
실제28원목/18흙/체력8.167/food5, NBT/해시 일치·STOP 뒤 dispatch0·키0·잔존 소유 PID0이다.
추론 중앙값51.141ms/p9561.916ms, 제어 주기198.203/302.546ms, 장치 VRAM peak2626MiB다.
근거 preparation/clear-handoff-0-audit.json, 원본 events/frames/source-at-game.

같은471장의 저장 화면을 다섯 조건으로 재생했다(각각 recovery100+collection371).
collection 구간 평균 use 확률은 visual7/RNN유지18.954%, visual7/초기화18.663%,
visual2/유지14.800%, text6/유지18.509%, text6/초기화17.537%다.
목표 조건이나 RNN초기화만으로 설치 문제가 해결된다는 근거는 없다. 마지막128화면에서
visual7의 attack93.059%/use0.448%였고 마지막 실제 화면도 설치 원목을 채굴 중이었다.
고정된 화면에 대한 반사실적 출력 비교이며 실제 획득 성과가 아니다. 총172.460초/게임 행동0/학습0/유료 호출0.
원본 분포: results/goal-completion-20261016-collection-offline/diagnosis.json.

collection-resume/0은825행동에서28→34, 원목 감소0·피해0이었다.61회 연속 같은 서버 채굴 대상
관측 구간이 여러 번 있었고,400행동 이후 실제 재고가 늘었다. 원목 개별 기원을 추적하지 않았으므로
설치 원목 회수와 자연 원목 신규 획득을 구분해 집계할 수 없다. 이전44개 대비 아직10개 부족하다.
저장3aac1fab81ff43adbdf29ced629b850f, 서버/NBT/해시 일치·STOP 뒤 dispatch0·키0·소유 PID0.
추론51.944/60.684ms(중앙/p95), 전체 제어 주기200.578/289.627ms, 장치 VRAM peak2634MiB.
근거 preparation/collection-resume-0-audit.json, collection-resume-0-mining-audit.json.

collection-resume/1은856행동에서34→30(소비5/획득1), 피해0, 서버/NBT/해시·STOP 뒤 dispatch0·키0·소유 PID0이었다.
저장f916bf3203864656b249dae9122c232c의 반경32블록 원목 드롭5개(거리2.725~4.653블록)를 확인했다.
첫 회차의 회수 성공만으로 전체 회수 문제가 해결됐다고 해석하지 않는다. 배치는 무순증가로 종료했다.
근거 preparation/collection-resume-1-audit.json, mining-audit.json, drops-audit.json.

root guidance 옵션 변경 후 Windows730개(708통과·22제외), WSL 관련56개 통과.
텍스트/영상 reset의 명시값·기본값, 0/2/7 복구 후 복원·RNN 유지, 유한 범위/미지원 경로 거부를 검사했다.
가중치/목표 embedding 변경은 없으며 실제 효과는 collection-guidance에서 별도 확인한다.

collection-guidance/0은471행동 후 몹 접근/체력8.167→4.167로 recovery_blocked였다. 재고30→29,
정상 저장7e5244da75cd4c7aa4ea5ae6e512c596, NBT/해시·STOP 뒤 dispatch0·키0·잔존 소유 PID0이다.
추론52.398/59.850ms, 제어 주기211.299/317.359ms, 장치 VRAM peak2636MiB. root guidance2를 성공으로 채택하지 않는다.

collection-resume/1 고정 관측 재생에서 기존 정체 경보0, 실제 보유 최고치만 목표 진전으로 쓰면801행동/800틱에
no_goal_progress를 확인했다. 그때 재고30, 마지막 단계 진행은37틱 전 블록 파괴였다. 실제 복구 성과와 구분한다.
파일 preparation/collection-resume-1-retention-replay.json. 이 결함 수정 후 Windows732(710통과·22제외), WSL78개 통과.
프로토콜의 retention 비교는 실패 세계를 보존한 명시적 피해 전 체크포인트 재시도이며 연속 생존이나 인과 대조가 아니다.

retention/0은879행동으로30→38, 소비0·피해0·복구 전환0이었다. 초기100~200행동에서4개를 주웠고,
이후 채굴/획득이 이어졌다. 저장3b859ae26f3a44c7b3900737e2700f92, NBT/해시·STOP 뒤 dispatch0·키0·소유 PID0.
추론49.519/57.897ms, 제어 주기191.064/274.529ms, 장치 VRAM peak2640MiB.
정체 복구가 호출되지 않아 이번 획득을 감지 수정의 효과로 볼 수 없다. 준비 시 복원1개와 실행 획득8개를 구분한다.
근거 preparation/retention-0-audit.json, retention-0-mining-audit.json.

retention/1은839행동으로38→46, 소비0·피해0·복구 전환0이었다. 실제 저장3c059993e6a94833aa82dd371fbf9886,
NBT/해시·STOP 뒤 dispatch0·키0·소유 PID0. 추론50.124/59.620ms, 제어198.721/295.438ms, VRAM peak2605MiB.
근거 preparation/retention-1-audit.json 및 mining-audit.json. 독립 평가 표본이 아닌 같은 개발 세계 이어하기다.

retention/2은813행동에서46→43, 소비8/획득5·피해0이었다. 서버tick824에서no_goal_progress_ticks800으로
실제 경보가 발생했다. 마지막 단계 진전은54틱 전 거리 개선이므로 단계 진행과 목표 보유 진전을 구분한 사례다.
실제 공식 go explore로 전환해13틱/3.157445941초를 수행한 뒤 회차 제한으로 저장했다.
저장ea600be62fdb4901ab7e92e9ebcc3c8b,43원목/22흙/체력8.167/food4, NBT/해시·STOP 뒤 dispatch0·키0·소유 PID0.
추론52.542/60.971ms, 제어207.037/300.608ms, VRAM peak2648MiB. 복구의 이탈/획득 성공은 아직 미확인이다.
근거 preparation/retention-2-audit.json 및 mining-audit.json, 실제 recovery_detector 사건.

회차 간 기억 진단은 retention/1의840추론 화면을 재생한 뒤 retention/2의320고정 화면을 비교했다.
첫64화면 use 확률 cold12.499%/기억 유지3.168%, attack36.723%/41.261%였다.
최초128화면 use6.586%/1.805%, 전체320화면8.197%/6.413%다. 샘플링 seed104761은 두 조건에 동일했다.
고정 화면 반사실적 출력이므로 게임 수집/인과 효과가 아니다. 실제 행동·학습·유료 호출0.
파일 goal-completion-20261016-session-memory-offline/diagnosis.json, 총186.560초.

같은 월드·목표 revision·root condition·모델·Torch·tensor 구조일 때만 NPZ RNN/RNG를 복원하도록 구현했다.
Windows739(717통과·22제외), 최종 경로 검증7, WSL 관련72검사 통과. 잘못된 WSL 모듈명으로 최초 명령1오류는 보존했다.
STOP 중 추론 결과를 전달하지 않고 입력 해제/월드 save 뒤 기억을 캡처하는 순서, 파일 손상/원자적 세대 게시를 검사했다.
실제 GTX1060/STEVE1/torch2.0.1에서16화면 추론 후 NPZ292818bytes를 복원해 다음 행동 및 다음 RNN SHA가 일치했다.
GPU검사49.352초, 기억SHA10470ed390f2b67ee78dad3bc2ef2fc82a64a85a91dedf04edfb695980f14e6a.
출력 goal-completion-20261016-memory-gpu-check/report.json. 게임 월드 없는 직렬화 검사를 실제 정책 성과로 세지 않는다.

memory/0은810행동으로43→45(소비0/획득2), 피해0이었다. 남은 탐험을 이어가 displaced로 root 수집 복귀했다.
정상 저장a3b9794ce51a4ffeba8ab882f0374ca9의 실제45/23/체력8.167/food4를 NBT로 대조했다.
기억810추론/9970597bytes를 같은 세대에 저장하고 file/world/RNN SHA 일치 및 STOP 뒤 새 추론/dispatch0을 확인했다.
서버 키0·소유 PID0, 추론50.426/58.781ms, 전체 제어208.276/301.326ms, 장치 VRAM peak2681MiB.
원래180초 뒤 새 dispatch는 금지됐으며 이미 전달된 제어 종료까지180.169초였다. 첫 회차는 cache 없는 cold start다.
근거 preparation/memory-0-audit.json, memory-0-cache-audit.json, memory-0-mining-audit.json.

memory/1은 실제 시작 시 직전 memory/file SHA 일치·restored=true를 확인했다.815행동에서45→50, 감소0·피해0이다.
저장31c147329e2d43ad9c1984e0ebba90e6, 실제50원목/23흙/체력8.167/food4, 서버/NBT/월드·기억 해시 일치.
STOP 뒤 새 추론/dispatch0·서버 키0·소유 PID0.815추론 NPZ9987197bytes를 다음 회차용으로 저장했다.
추론51.165/59.280ms, 제어206.166/302.327ms, 장치 VRAM peak2659MiB.
근거 preparation/memory-1-audit.json, memory-1-cache-audit.json, memory-1-mining-audit.json.
동일 조건의 cold 대조 rollout이 없으므로 획득량을 기억 유지의 인과 효과로 보지 않는다.

memory/2는 실제 기억 복원 뒤832행동50→54(획득5/소비1), 피해0이었다. 기억 유지 후에도 원목 설치 소비가 남는다.
저장617b7322cbdf43e9a07ee0b8aab4df46의54원목/24흙/체력8.167/food4, NBT/월드·기억 해시 일치.
STOP 뒤 새 추론/dispatch0·키0·소유 PID0,832추론 NPZ9984954bytes를 보존했다.
추론51.144/59.478ms, 제어205.323/298.711ms, 장치 VRAM peak2659MiB.
근거 preparation/memory-2-audit.json, memory-2-cache-audit.json, memory-2-mining-audit.json.

memory/3은 backend-preparation-started(08:06:03)까지만 있고 완료 marker/events/summary/새 저장이 없다.
09:53 재접속에서 원래 실행 handle이 없고 WSL 관련 Python/Java 프로세스도 없었다. Windows 마지막 부팅08:09:21을 확인했다.
재부팅 원인은 규명되지 않았다. 실제 정책 실패/성공 표본으로 세지 않고 STOP/정상 저장이 검증됐다고 주장하지 않는다.
원본 디렉터리는 보존하고 별도 preparation/memory-host-restart-receipt.json에 기술적 중단을 기록했다.
마지막 정상617b 세대의 NBT/기억/소스 해시를 재확인하고 memory-resume에서 같은54/24를 이어간다.

memory-resume/0은 재부팅 후 CUDA 초기화16.833초를 완료하고 직전 기억 hash 일치 상태로 실제 재개했다.
794행동54→60(획득6/소비0), 피해0, 저장4bc6f32f4f874621b68c0881cfcf4bc8의60원목/25흙/체력8.167/food4를 검증했다.
서버/NBT/월드·기억 해시 일치·STOP 뒤 새 추론/dispatch0·키0·소유 PID0.794추론 NPZ9985457bytes.
추론54.659/63.354ms, 제어212.816/302.933ms, 장치 VRAM peak2617MiB.
근거 preparation/memory-resume-0-audit.json, memory-resume-0-cache-audit.json, memory-resume-0-mining-audit.json.

memory-resume/1은549행동/120.108초에서60→64(획득4/소비0)로 두 목표 completed가 됐다. 체력8.167·food4를 유지했다.
최종 저장2b50d8ba1ad24cb7873266d456d29b4c의64원목/25흙, 서버·playerdata·level.dat NBT/월드·기억 해시를 대조했다.
완료 사건 뒤 새 추론/dispatch0, STOP 뒤 새 추론/dispatch0, 독립 서버 held_key_count0, 소유 PID0이다.
추론51.470/60.647ms, 전체 제어201.780/302.282ms, 장치 VRAM 표본 peak2593MiB.
최종 기억549추론 NPZ9970934bytes, SHA7493dd77ce16ec361b3a5359fc409918d106cff5b2643152d02a7996735110f2.
근거 preparation/memory-resume-1-audit.json, memory-resume-1-cache-audit.json, memory-resume-1-mining-audit.json.

완료 감사는 최초716407a9 목표의world_id/task_id/revision1/target_count64·4/baseline0/대상 아이템 불변,
기존 versions/failures/복구 history·되감기 provenance 보존 및 원본 연구 명세 해시 불변을 확인했다.
정상 memory5실행의3800행동은43→45→50→54→60→64였고, 모두 마지막 dispatch가180초 이내였다.
기억 없는 첫 회차 뒤4번 실제 같은 기억/file hash 복원이 확인됐다. memory/3 호스트 재시작은 성공/정책 실패 표본에서 제외했다.
이5회는 같은 개발 세계의 순차 이어하기이고 마지막 구간에 추가 물리 되감기는 없다. 전체 연구의 이전 되감기는 여전히 포함된다.
정책 기억 유지의 인과 성능 향상·인간다움·최신 서버 호환·새 학습 완료를 주장하지 않는다. 최종 STOP latch를 유지하고 추가 실행하지 않는다.
