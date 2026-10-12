# 실제 사망 전이의 경계 오류

## 재현

후속 시험의 명시적 noop 진단 `results/rl-eval-104729-20261007T045428744248Z/run`.
플레이어의 체력은 실제로 감소했고 마지막 정상 전이는 체력약1, alive=true였다.
466전이 뒤 다음 호출에서 `expected one server tick, observed 6`으로 중단됐다.
실행 무효이며 정책 과업 실패·학습 표본으로 세지 않는다. 중단 뒤 neutral 해제와 소유 PID0은 확인했다.
예외를 일으킨 반환 관측은 원시 transition으로 저장되지 않아 해당 호출의 done/체력을 직접 단정하지 않는다.

독립 증거: 해당 실행의 원본 게임 로그13:57:38에 화상 사망이 기록됐다. 동일 임시 월드의 저장
플레이어는 Health20/DeathTime0/Fire-20, 원래 지상 스폰 위치로 이동해 있었다.
저장 통계는 deaths1/time_since_death2다. 따라서 자동 재생성은 소스 추측이 아니라 실제 발생했다.
`results/wood-followup-20261007/death-game-log.txt`, `death-saved-player.json`에 원본 로그와
저장 NBT 추출값·통계·파일 해시를 보존했다.

## 설치된 고정 소스에서 확인한 경로

- `human_survival_specs.py`는 `DoneOnDeath` handler를 기본으로 추가하지 않는다.
- `EnvServer.loadOrCreateWorld`의 저장 ZIP 경로는 `createNewWorld`를 건너뛴다.
  `doneOnDeath` 설정과 respawn flag 초기화는 현재 `createNewWorld` 안에만 있다.
- `DeathScreen.render`는 멈춤을 피한다는 주석과 함께 자동 respawn을 호출한다.
- `PlayRecorder.tick`은 죽은 플레이어의 전이를 기록하지 않고 episode를 reset한다.
- `EnvServer.waitForNextObservation`은 recorder의 tick counter 변화에 의존한다.

따라서 RL에서 기대한 ‘사망은 실제 실패 종료’ 계약이 환경 설정·복원·관측 handshake 전체에
연결되지 않았다. 틱 검사를 완화하거나 새 생명의 관측을 같은 궤적으로 학습하는 것으로 해결하지 않는다.

기존 단위 검사는 `done=True`가 주어졌을 때의 실패 보상/종료 처리를 검사했다.
엔진의 자동 respawn과 저장 월드의 설정 누락까지 실행하지 않았기 때문에 이 결함을 잡지 못했다.

## 수정·재검증 단위

1. RL 환경에만 명시적 사망 종료 조건을 설정하고, 새 월드/저장 월드 양쪽에 적용한다.
2. 해당 조건일 때 자동 respawn을 막고 마지막 죽은 플레이어의 관측을 발행한다.
   게임 스레드의 기존 snapshot 경계와 한 행동/한 서버 tick 계약을 보존한다.
3. 실제 server health/alive 및 종료 전이를 기록하고, 실패 비용-1·추가 행동0·정리를 검사한다.
4. 같은 수정 JAR에서 일반 원목 획득도 재검증한다. 기존 배치 결과와 JAR 버전을 분리한다.

자연 월드10회 비교는 원본 실행 코드/체크포인트를 고정해 완료한 뒤 수정했다.
수정 전 비교와 수정 후 회귀를 합쳐 같은 실행 조건의 표본이라고 해석하지 않는다.

구체적 경계: RL 전용 `DoneOnDeath` handler → 공통 월드 로드의 설정 → recorder의
mission별 terminal 설정으로 전달한다. 사망 화면의 자동 respawn은 이 설정일 때만 막는다.
죽은 플레이어도 마지막 recorder 전이를 발행하고, 발행 전 mouse/keyboard 상태와 KeyBinding을
해제한다. 읽기 전용 서버 health/alive와 눌린 키 개수를 기록해 검증한다.
원본 설치 JAR/소스는 수정하지 않고 기존 방식의 별도 패치 JAR를 생성한다.

## 수정 및 실제 회귀 완료

`world_save_patch.py`의 terminal patch는 저장/새 월드 공통 설정, recorder의 마지막 사망 프레임,
DeathScreen의 조건부 자동 재생성 차단을 연결한다. `workshop_survival(terminate_on_death=True)`는
RL에만 적용한다. 관측에 server health/alive·held key count를 추가했고 실제 키 상태는
terminal 프레임 발행 전에 해제한다. 1tick 검사는 완화하지 않았다.

| 검사 | 실행 폴더 | 결과 |
|---|---|---|
| 같은 noop 화상 사망 | `rl-eval-104729-20261007T054445759039Z` | 467전이, environment_done, 실패 비용-1 |
| v2 학습 정책 화상 사망 | `rl-eval-104729-20261007T055032753815Z` | 467전이, terminal=true, bootstrap0, 실패 비용-1 |
| 일반 원목 회귀 | `rl-eval-2000003-20261007T055528809596Z` | 86전이, 실제 원목 획득 |

사망 두 경로의 실제 server health0/alive0, client life0/is_alive=false, held key count0을 확인했다.
모든 전이는1서버 tick, 중단 확인 뒤 새 정책 행동0, 소유 PID0이었다. 사망 후 neutral step은
실행하지 않았고 `environment_done_no_neutral_step`과 게임 스레드의 키 해제 증거를 구분해 기록한다.
모두 eval이며 추가 학습0이다. 단위 검사만으로 사망 환경을 검증했다고 주장하지 않는다.

수정 JAR SHA256: `2de8cff9236d41fc3abcf12ee1788210016bc45a10bd83eff7af6734bc9a9677`.
이전 비교 JAR SHA256: `7055b595f897f72a815e6d0ccc30a01e2fcb51c7aaa6e7d5f2dc48f4ad153454`.
원본 설치 JAR SHA256은 `f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48`로 유지한다.
첫 빌드 명령은 격리 JDK 경로가 빠져 javac를 찾지 못했고, micromamba 환경에서 컴파일해 해결했다.
실행·원시 감사·빌드·검사 로그를 모두 보존했다.

Windows211검사 중199통과/12제외, WSL211중207통과/4제외. 새 회귀 검사는 저장 월드 설정,
사망 프레임 발행 순서·입력 해제, opt-in 설정, 예상하지 못한 upstream 소스 거부를 검사한다.
실제 재현과 저장 파일의 독립 증거로 이 화상 사망 경로의 원인·수정은 확인했다.
다른 Minecraft 버전의 종료 동작까지 검증한 것은 아니다.
