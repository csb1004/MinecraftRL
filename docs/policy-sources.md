# 공식 소스 조사 — 2026-10-03

## 공개 모방 데이터 추가 조사 — 2026-10-08

[VPT contractor/BASALT와 STEVE-1 변환 경로](imitation-data-plan-2026-10-08.md)를 재확인했다.
공식 VPT 샘플184MB를 실제 다운로드해 영상6,001프레임·행동6,000개를 검사했다.
형식 검사는 통과했지만 관측/행동 시점과 구간별 목표 라벨이 미확정이므로 학습 준비 완료는 아니다.
BC를 기존 PPO 궤적과 분리하며 일반 이동 경험과 실패 복구 경험을 보충하는 방향이다.
아래는 최초 설치 당시의 소스 기록이며 현재 실행 상태는 `progress.md`를 따른다.

`configs/sources.lock.json`에 조회한 정확한 커밋을 기록했다. README 조회와 코드 검토는 실행 성공의 증거가 아니다.

| 후보 | 확인한 공식 근거 | 이번 결정 |
|---|---|---|
| [VPT](https://github.com/openai/Video-Pre-Training) | README foundation 1x `.model`/`.weights`, `agent.py`의 reset/get_action, torch==1.9.0 | 첫 로컬 메모리/행동 기준선. Python 3.9 후보 |
| [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1) | README Linux/Python 3.10/PyTorch 2.0, MineRL v1.0.1; `MineRLConditionalAgent`와 text prior | 별도 환경; 2x + MineCLIP + prior, guidance 6.0 |
| [MineRL](https://github.com/minerllabs/minerl/tree/v1.0.1) | HumanSurvival/HumanControl: POV와 FlatInventoryObservation; env.seed → reset | 실제 인벤토리 판정. gym fork 커밋도 고정 |
| [MineStudio](https://github.com/CraftJarvis/MineStudio) | 공식 README와 커밋 확보 | 실행/학습 환경 후속 후보. 이번 baseline 대체 안 함 |
| [Mineflayer](https://github.com/PrismarineJS/mineflayer) | 공식 docs/README.md와 package.json | API skill/관측 후보. 정책 가중치 실행기로 간주하지 않음 |

## 구체적인 호환성 쟁점

- VPT 공식 README는 torch 1.9.0과 Python 3.10+ 비호환, torch 교체 시 행동 재현성 차이를 명시한다. Windows Python 3.11을 그대로 쓰는 것으로 검증을 생략할 수 없다.
- STEVE-1 공식 README는 MineDojo/MineCLIP와 VPT의 gym 요구 충돌을 명시한다. 실제 MineRL v1.0.1 requirements에는 `minerllabs/gym@minerl-fix`가 들어 있다. 따라서 가변 브랜치를 커밋으로 고정한 fork를 두 정책의 환경에 각각 설치하는 후보로 선택했다. 이는 검증된 환경으로 간주하지 않는다.
- MineRL `setup.py`는 설치 중 MCP/Java/Gradle 자산을 빌드한다. `scripts/setup_mcp.sh`의 실제 MCP 기준은 **1.16.5-20210115**다. 저장소에 남아 있는 구형 Malmo build.gradle의 1.11.2 문자열을 현재 실행 버전으로 오인하지 않는다.
- 빌드 복사본에서 Gradle 자산 경로만 프로젝트 `.cache/gradle`을 사용하게 수정하고, MCP 태그를 조회한 커밋으로 고정한다. 원본 외부 checkout은 수정하지 않으며 변경 diff/해시는 results에 기록한다. MCP 빌드와 렌더링은 아직 실행 검증 전이다.
- MineCLIP의 공식 체크섬은 MD5다. 이를 확인하고 모든 실제 다운로드에는 SHA256을 추가로 기록한다. 나머지 모델의 로컬 SHA256은 받은 바이트의 식별자이지 독립적인 upstream 진위 증명이 아니다.
- `.model`/가중치는 pickle/torch 역직렬화를 사용한다. 고정된 공식 출처와 receipt 검사를 통과한 파일만 실행 경로에 넣는다.
- 코드/환경 후보를 고정한 것과 설치가 성공한 것, 모델 로드와 실제 rollout 성공은 각각 다른 검증 단계다.

## 버킷 연결은 별도 과제

공식 MineRL에서 실행돼도 최신 버킷 서버 호환은 입증되지 않는다. 실제 통합에는 지원 Minecraft 버전의 클라이언트, 화면 캡처/키·마우스 bridge, 인벤토리/이벤트 관측 bridge, task revision에 연결한 입력 소유권, 재접속/입력 해제 검증이 필요하다. API skill과 정책 제어 구간은 별도로 기록해야 한다.

서버와 봇의 공통 최신 안정 버전은 실제 서버 종류/빌드와 Mineflayer 지원표를 확인하고 로그인·이동·관측·끊김 테스트를 통과한 뒤 lock한다. 현재 특정 최신 버전이 검증됐다고 주장하지 않는다.

conda dry-run으로 선택된 linux-64 패키지 URL과 upstream MD5는 configs/conda-*-linux-64.candidate.txt에 고정했다. pip 전이 의존성은 설치 뒤 freeze로 기록하며, 전체 검증 환경 lock은 아직 없다.
