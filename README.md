# MinecraftRL

말로 받은 목표("철 캐줘")를 Minecraft 안에서 끝까지 수행하는 에이전트를 연구한다.
공식 STEVE-1은 고정해 두고, 그 위에 **LLM이 제안하고 프로그램이 검증한 지식·계획**, **실수 직전에 멈추는 감시**,
**기술 종류별로 학습한 작은 헤드**를 얹는다. 장기 목표와 [원문 연구 명세 20개 항목](docs/research-spec.md)은 유지한다.

**처음 온 사람·에이전트는 [작업 시작 안내](docs/handbook/agent-start.md)부터 읽는다.**
[최신 진행 기록](docs/progress.md) · [문서 지도](docs/README.md) ·
[코드 구조](docs/handbook/architecture.md) · [개발·검사 방법](docs/handbook/development.md)

## 한눈에 보기

| 단계 | 하는 일 | 코드 |
|---|---|---|
| 1. 목표 해석 | 말을 LLM이 목표로 제안하고, 프로그램이 허용 목표·수량으로 검증한다 | `pipeline/goal.py` |
| 2. 지식과 계획 | SQLite 지식 저장소(조건부 지식, 지지·반박 기록)에서 플래너가 단계 계획을 만든다. 부족한 지식은 게임 전에 LLM 웹 검색(인용 검증)으로 채운다 | `pipeline/kb.py`, `knowledge.py`, `planner.py`, `oversight.py` |
| 3. 단계 관리 | Reward Machine이 지금 단계를 관리한다. 인벤토리 수량과 블록 설치 기록이 바뀌면 다음 단계로 넘어간다 | `pipeline/rm.py`, `session.py` |
| 4. 행동 | 고정한 STEVE-1 + 그 단계 카테고리의 기술 헤드 | `reacquire_head.py`, `head_switch.py` |
| 5. 감시 | 되돌릴 수 없는 행동(맨손 채굴 등)은 보내기 전에 가드가 막는다. 별도 스레드의 감시가 이상·정체를 잡으면 정책을 멈춘다 | `pipeline/guard.py`, `monitor.py`, `monitor_thread.py` |
| 6. 멈춘 뒤 | 지식 조회 → 없으면 LLM 웹 검색 → 재계획. 정체·반복 실패는 해결책을 골라 그 단계만 수리한다 | `pipeline/oversight.py`, `remedy.py`, `feedback.py` |
| 7. 판정 | 성공은 서버 인벤토리와 블록 기록으로만 판정한다 | `pipeline/session.py`, `analysis.py` |

## 현재 상태 — 2026-11-25

### 최근에 정한 것

- **헤드는 손동작 카테고리 7개로 나눈다** (11-22). 이동, 탐색, 블록 × 공격, 블록 × 사용, 개체 × 공격, 개체 × 사용, 원거리.
  창 조작·먹기는 규칙 기술이고, 기다리기는 RM 단계다. RM 단계마다 그 단계의 카테고리 헤드로 바꾸고, 이어지는 단계의 카테고리가 같으면 그대로 둔다.
  [카테고리 헤드 설계](docs/specs/2026-11-18-online-category-learning-design.md)
- **실제 센서 대응은 뺀다** (11-25). 11-19에는 "실제 센서로 잴 수 있는 값만 정책 입력으로"를 세웠지만, 센서 모형·잡음 조건 같은 실사화 작업은 하지 않기로 했다.
  남긴 원칙은 하나다. **정책 입력은 화면과 게임이 주는 구조화된 상태(조준선 블록, 손 아이템, 인벤토리, 자세)이고, 가려진 대상의 위치 같은 숨은 정보는 학습 신호로만 쓴다.**
  코드 이름(`sensor.py`, `sensor_monitor`)은 예전 경로를 지키려고 그대로 둔다. [결정 기록](docs/specs/2026-11-19-sensor-tier-observation-design.md)
- **정체 뒤 복구와 시작 전 계획을 넣었다** (11-25). MineEvolve(arXiv 2603.13131)의 Monitor·Inducer·Curator·Adaptor 구조를 따랐다.
  [정체 뒤 복구 설계](docs/specs/2026-11-25-stagnation-remedy-design.md)
  - 정책 단계 시도마다 실행 피드백을 남기고, 30초 동안 진척이 없으면 멈춘다.
  - 해결책은 지식 저장소에서 먼저 찾고, 없으면 LLM이 실행 가능한 수리 6가지 중 하나를 고른다. 프로그램이 다섯 가지 검사를 한 뒤 저장하고, 게임 결과로 지지·반박한다.
  - 게임을 시작하기 전에 계획에 부족한 지식을 먼저 물어 채운다. 전에는 가드가 멈춘 뒤에야 물었다.

### 무엇이 어디까지 됐나

| 구성요소 | 상태 | 근거 |
|---|---|---|
| 목표 해석·검증, RM, 가드, 지식 저장소, LLM 웹 지식, 재계획 | 게임에서 확인 | "철 캐줘" 평가(11-05): 빈 지식 6/10, 이어받은 지식 1/3. 흐름 점검(11-24, 평가 아님): 6판 중 5판 철광석 |
| 감시를 별도 스레드로 | 게임에서 확인 | 11-16 점검, 11-24 점검(샘플 버림 0, 늦은 경계 0) |
| 정체 뒤 복구(해결책) | 구현·단위 검사. 제자리 정체 보류와 임시 수리만 게임에서 확인 | 11-25 무료 점검 3판. LLM이 해결책을 만드는 경로는 게임에서 아직 못 돌림 |
| 시작 전 계획 | 구현·단위 검사 | 빈 지식 + LLM 경로는 게임에서 아직 못 돌림 |
| 기술 헤드 | 아래 표 | |
| RM 단계마다 헤드 교체 | 일부 | 신호기 복구(건축 ↔ 캐기)에서만 돈다. "철 캐줘" 흐름은 아직 공식 STEVE-1만 쓴다 |
| 판 사이 헤드 갱신, 버전 관문, 보상 악용 감시 | 설계만 | [11-18 설계](docs/specs/2026-11-18-online-category-learning-design.md) |
| 탐색 헤드, 지식 Q | 없음 / 설계만 | [지식 Q 설계](docs/specs/2026-11-20-knowledge-q-design.md) |

### 기술 헤드

헤드는 고정한 STEVE-1의 행동 헤드에 더하는 작은 보정이다. STEVE-1 본체는 바꾸지 않는다.

| 카테고리 | 헤드 | 입력 | 결과 (서버 판정, 괄호는 같은 조건의 공식 STEVE-1) |
|---|---|---|---|
| 블록 × 사용 | 건축 `s-final` (11-22) | 클라이언트가 보는 블록·조준선, 설계도 | 개발 8/8 (0/8), 최종 8/8 (0/8) |
| 블록 × 공격 | 캐기 `f-final` (11-24) | 위 + 조준선이 목표를 겨누는지 | 개발 8/8 (0/8). 최종 평가는 안 함 |
| 블록 × 사용 + 공격 | 신호기 복구 시연: 건축 ↔ 캐기 교체 | 이상감지가 헤드를 고른다 | 시연 7/8 (판정 아님). 설계도 블록 부숨 0 |
| 블록 × 공격 | 채굴 v3 `mining-v3` | 특권(fixture 좌표, 서버 자세) | 최종 8/8 (8/8), 채굴 시도가 파괴로 끝난 비율 53% (5%), 성공 단계 시간 11초 (38초) |
| 블록 × 사용 | 놓기 `place`, 건축 `build-r5`, 다리 `bridge` | 특권 | 최종 5/8, 7/8, 5/8 (모두 0/8) |
| 이동 | 점프 `jump` | 특권 | 최종 8/8 (0/8) |
| 탐색, 개체 × 공격, 개체 × 사용, 원거리 | 없음 | | |

- 특권 입력을 받는 다섯 헤드(11-14–15)는 참고 조건으로만 쓴다. 가중치와 상세는 [checkpoints/README](checkpoints/README.md)에 있다.
- 11-22 이후 헤드(`s-final`, `f-final`)는 숨은 정보를 입력으로 받지 않는다. 가중치는 로컬 `results/`에만 있고 저장소에는 없다.
- 캐기 헤드 3차: 목표를 겨눴을 때 공격 97.4%, 다른 블록을 겨눴을 때 공격 0/106. 다만 조준선 입력은 교사의 공격 조건과 같은 신호다.
- 근거는 [진행 기록](docs/progress.md)의 11-22–11-24 항목과 [센서 복구 설계](docs/specs/2026-11-23-sensor-repair-design.md)에 있다.

### 알려진 한계

- 감시와 성공 판정은 서버 상태를 읽는다. 정책 입력과는 구분한다.
- "철 캐줘" 흐름의 정책은 공식 STEVE-1이라 탐색에 오래 걸린다. 정체 기준은 제자리 정체만 잡고, 계속 돌아다니는 탐색은 잡지 못한다.
- 점검(11-24, 11-25)은 흐름이 도는지 보는 것이고 정책 평가가 아니다. 평가 수치는 표본이 작다.
- 판 사이에 스스로 학습하는 루프는 아직 없다. 헤드 학습은 사람이 정한 배치와 사전 등록한 판정으로 했다.
- 유료 LLM 호출은 `--allow-paid`를 줄 때만 일어난다.

## 빠른 진입

실제 프로젝트는 `D:\DATA\Desktop\MinecraftRL`이다. Windows에서는 PowerShell을 사용하고,
정책 환경은 WSL의 격리 환경을 사용한다. `gh`를 사용하지 않는다.

이미 설치된 PC에서 게임 없이 코드 상태만 확인한다.

```powershell
Set-Location -LiteralPath 'D:\DATA\Desktop\MinecraftRL'
& .\scripts\Invoke-Checks.ps1
```

검사는 기존 WSL/Windows 환경만 사용하며 결과를 새 `results/checks-*`에 남긴다.
새 PC는 먼저 [공식 모델 설치 안내](docs/new-pc-setup.md)를 따른다.

목표 흐름 한 판은 WSL에서 프로젝트 폴더를 현재 위치로 두고 이렇게 돌린다.

```bash
export PYTHONPATH=$PWD/src LD_LIBRARY_PATH=/usr/lib/wsl/lib GALLIUM_DRIVER=d3d12 XDG_CACHE_HOME=$PWD/.cache
.runtime/bin/micromamba run -p .runtime/steve1 python scripts/run_iron_demo.py probe --root . \
  --output results/<새 폴더>/<판 이름> \
  --config configs/iron-demo.json --fixture configs/iron-demo-fixture-v2.json \
  --seed 130363 --sampling-seed 1 --limit-s 900 --frame-every 2 \
  --goal iron_ore:1 --kb results/<새 폴더>/kb.sqlite --stop-file results/<새 폴더>/STOP
```

- 말로 시작하려면 `--goal` 대신 `--utterance "철 캐줘" --allow-paid`를 준다. `OPENAI_API_KEY`, `OPENAI_MODEL` 환경 변수가 필요하고 호출은 유료다.
- 프롬프트 임베딩(`results/iron-demo-prompts/` 등)은 저장소에 없다. `scripts/prepare_demo_prompts.py`로 만든다.
- 저장소 코드만으로 과거 실험의 가중치·월드·실행 로그가 복원되지는 않는다. 정확한 재현에는 해당 실행의 별도 자료와 `source-at-game`이 필요하다.
- 완료된 배치는 다시 실행하지 않는다. `results/STOP`은 자동으로 풀지 않는다.

## 폴더 역할

| 경로 | 역할 |
|---|---|
| [src/minecraftrl](src/minecraftrl/README.md) | 실행, RM, 관측, 정책, 학습 라이브러리. 목표 흐름은 `pipeline/` |
| [scripts](scripts/README.md) | 설치·검사·실행·독립 학습 진입점, 완료 실험 드라이버 |
| [tests](tests/README.md) | 현재 회귀 검사와 별도로 보관한 폐기 실행기 검사 |
| [configs](configs/README.md) | 공식 소스/모델 출처, 환경, 검증용 목표 예제 |
| [checkpoints](checkpoints/README.md) | 학습한 기술 헤드 다섯 개(특권 참고 조건) |
| [docs](docs/README.md) | 현재 안내, 고정 명세, 결과, 과거 연구 기록 |
| `results/` | 실행별 원본 증거·체크포인트·월드·소스 사본, Git 제외 |
| `weights/`, `external/`, `.runtime/` | 공식 가중치, upstream 소스, 격리 환경, Git 제외 |
| `data/`, `logs/`, `.cache/` | 로컬 자료·로그·캐시, Git 제외 |
| `notebooks/` | 별도 클라우드 실행 준비 자료; 실제 연결은 별도 승인 |

과거 문서는 날짜가 있는 연구 기록이다. 특히 2026-10-25에 폐기한 학습기/자료의
실행 명령은 현재 안내로 사용하지 않는다. 과거 경로는 재현 근거를 보존하기 위해 유지한다.
