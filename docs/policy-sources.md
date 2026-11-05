# 공식 소스 조사 — 2026-10-03

## STEVE-1 호환 Minecraft 버전 재확인 — 2026-10-25

사용자는 환경 복잡도를 줄이기 위해 1.7.10을 제안했고, 다른 낮은 버전도 STEVE-1과 맞으면
대체해도 된다고 보완했다. 확인한 공식 실행 조합은 **Minecraft 1.16.5 / MineRL v1.0.1**이다.
현재 설치본도 이 조합을 기준으로 하므로 이번 검토에서는 버전을 교체하지 않는다.

- [고정 STEVE-1 README](https://github.com/Shalev-Lifshitz/STEVE-1/blob/903b244796322f4d0073a8f62c05f51eac3aed52/README.md)는 MineRL v1.0.1을 지정한다.
- [MineRL v1.0.1 빌드 스크립트](https://github.com/minerllabs/minerl/blob/v1.0.1/scripts/setup_mcp.sh)는 MCP `1.16.5-20210115`를 사용한다.
- 로컬 `configs/sources.lock.json`, `external/minerl/scripts/setup_mcp.sh`와 설치된
  `.runtime/steve1/lib/python3.10/site-packages/minerl/MCP-Reborn/build.gradle:193`,
  `.runtime/vpt/lib/python3.9/site-packages/minerl/MCP-Reborn/build.gradle:193`에서도 1.16.5를 확인했다.
- [MineRL 버전 설명](https://github.com/minerllabs/minerl/blob/v1.0.1/README.md)은 v1.0에서 게임 버전과
  행동/관측 인터페이스가 바뀌었음을 명시한다. 구형 MineRL 설치만으로 STEVE-1의 하위 버전 호환을
  확보했다고 볼 수 없다. 이번에 확인한 공식 경로에서는 1.16.5보다 낮은 STEVE-1 지원 조합을 찾지 못했다.

1.7.10 이식 시 관측/입력/상태 판정 bridge의 재작업과 시각·조작 차이에 대한 정책 검증이 필요하다는
것은 위 의존 구조에 따른 판단이며, 실제 이식 실패나 성능 저하를 측정한 결과가 아니다.
초기 복잡도는 1.16.5 안에서 과업과 월드 조건을 제한해 낮추는 방향을 추천한다. 지형·배치·도구 조건과
평가 시드는 다양하게 유지하고 장기 연구 목표는 축소하지 않는다. 구체적 월드 설정은 아직 변경하지 않았다.
설치본 설정/공식 소스 확인과 문서 기록만 수행했다. 게임·학습 재개 없음, STOP 유지.

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
