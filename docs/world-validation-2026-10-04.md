# 월드 저장·재시작 실제 검증 — 2026-10-04

Windows DESKTOP-V0HJPC2 / WSL Ubuntu-24.04 / GTX 1060 3GB에서 STEVE-1을 실행했다.
추가 API 호출, Colab 연결, MyWife 개인 데이터 사용은 모두 0회다. 테스트 시드는 130363,
실행별 제어 예산은 최대 180초다. 이번 검증은 저장 기능의 단일 시드 통합 시험이며
기존 5-seed 정책 실험의 성공률에 합치지 않는다.

## 첫 시도에서 발견한 실패

`results/world-scenario-20261004T001349946251Z`: 첫 프로세스에서 흙 2개를 실제 획득한
뒤 `<Quit/>` 응답을 받고 ZIP을 만들었다. 그러나 새 프로세스의 초기 인벤토리가 저장 직전
관측과 달라 `restored_inventory_mismatch`로 중단했다. 복원 실패 후 정책 행동은 0회다.
저장 ZIP의 `level.dat`와 `playerdata/*.dat`를 직접 읽어 Inventory가 비어 있고 시작 위치가
남아 있음을 확인했다. 재시작 실패와 잘못된 저장본은 덮어쓰지 않고 보존했다.

고정 MineRL의 동기식 서버 종료 경로에 최신 상태 저장이 빠져 있었다. 프로젝트 전용
`ReplaySender.stop()` 사본에 서버 플레이어 데이터·월드 flush를 추가하고 Java 8로 컴파일했다.
원본 JAR는 그대로이며, 일반 episode/suite는 수정본을 사용하지 않는다.

- 원본 SHA256: `f7a1e0aa8bee8a0ff9479856e807be6d8aa30373db8934d5b093342690ae9a48`
- 세션용 수정본 SHA256: `984f6336b9e0e851a3c9f586c6d3f13dffbcba277485601a025233d2d36599a6`
- 빌드 기록: `.runtime/world-save/df018cefa323df997b1a5b105b9f5e34113abf6494c6600fbf7ce5d62f2374ec.json`
- NBT 진단: `results/world-first-save-nbt.json`, `results/world-fixed-save-nbt.json`.

## 수정 후 실제 결과

`results/world-scenario-20261004T002800879251Z/acceptance.json`: `passed=true`.
서로 다른 worker/Java 프로세스 사이에서 같은 월드 ID와 과업을 이어갔다.

| 항목 | 첫 실행 | 새 프로세스에서 복원 후 실행 |
|---|---:|---:|
| 실제 흙 수량 | 0 → 2 | 2 → 5 |
| 정책 step | 186 | 175 |
| 제어 시간 | 26.718초 | 26.441초 |
| 종료·저장 시간 | 5.086초 | 5.413초 |
| 추론 중앙값 / p95 | 43.306 / 53.939 ms | 42.816 / 54.056 ms |
| 제어 주기 중앙값 / p95 | 117.945 / 155.474 ms | 122.681 / 164.228 ms |
| PyTorch 최대 allocated / reserved | 2098.67 / 2144 MiB | 2098.67 / 2144 MiB |
| 전체 GPU 표본 최대 사용량 | 2777 MiB | 2775 MiB |

복원 초기 인벤토리는 흙 2개와 씨앗 1개로 저장 관측과 정확히 일치했다. 위치 차이는
0.011796블록(허용치 2블록)이었다. MineRL 초기 중립 프레임·저장 종료 처리 사이의
미세한 물리 변화가 있으므로 프레임 단위 동일성을 주장하지 않는다.

과업은 paused, revision 1, baseline 0, progress 2로 복원됐다. 명시적 resume 수락 전
약 1.200초 동안 정책 행동 0회, 이후 흙을 총 5개까지 획득했다. 두 실행 모두 pause 및
STOP 수락 이후 새 정책 행동 0회, 중립 입력 해제 성공, 소유 자식 프로세스 잔존 0개다.
근거: 같은 결과 폴더의 `control-evidence.json`, 각 실행의 `events.jsonl`, `summary.json`.

검증기는 수집량 확인 후 의도적으로 pause와 STOP을 보냈다. 따라서 raw summary의
`status=cancelled`, `execution_valid=false`, worker exit 2는 보존하며 정상 180초 정책 표본으로
계산하지 않는다. 설정한 흙 64개 과업을 완료했다는 뜻도 아니다. 저장·복원 시나리오의
통과 여부는 별도 acceptance로 판정했다.

이 단계의 검증 저장 슬롯은 `validation-20261004T002800879251Z`, 마지막 세대는
`b5c84a00aedf48eaa459f70f1c8ffd86`이며 `data/worlds/` 아래에 보존한다.
기본 사용자 슬롯 `companion`과 구분한다. 이 저장본에는 흙 5개까지의 중단된 과업이 남는다.

## 플레이어 UUID 고정 후 최종 검증

추가 NBT 검사에서 위 두 저장본의 플레이어 UUID가 서로 다름을 발견했다. upstream 실행
기본 이름이 `Player<시간>`이라 시작 UUID가 매번 바뀌었다. 인벤토리·위치 복원은 통과했지만
장기 플레이어별 기록에 영향을 줄 수 있어, 세션 시작 옵션에 슬롯별 고정 UUID를 추가했다.
새 수정은 Java 저장 패치를 바꾸지 않고 프로젝트 launcher의 선택적 실행 인자로 적용한다.

최종 `results/world-scenario-20261004T004053114851Z/acceptance.json`도 통과했다.
두 번의 실제 저장 NBT에서 동일한 UUID `241fd488-51a0-563f-b0c9-a6e9be02c499`를 확인했고
각 저장본의 playerdata 파일도 같은 UUID의 단일 파일이다. 흙은 0→2, 재시작 후 2→4개가
됐다. 씨앗 1개도 유지됐다. 초기 위치 차이는 0.116890블록, 재개 전 1.215초간 정책 행동 0회,
두 실행의 pause/STOP 수락 후 정책 행동 0회다. 중립 입력 해제 및 소유 프로세스 정리 성공.

| 최종 실행 지표 | 저장 전 | 새 프로세스에서 재개 후 |
|---|---:|---:|
| 정책 step | 319 | 143 |
| 제어 / 종료·저장 시간 | 42.222 / 5.073초 | 23.238 / 4.512초 |
| 추론 중앙값 / p95 | 42.477 / 51.568 ms | 44.999 / 54.329 ms |
| 제어 주기 중앙값 / p95 | 112.530 / 163.564 ms | 132.486 / 157.468 ms |
| PyTorch 최대 allocated / reserved | 2098.72 / 2144 MiB | 2098.67 / 2144 MiB |
| 전체 GPU 표본 최대 사용량 | 2782 MiB | 2763 MiB |

최신 슬롯은 `validation-20261004T004053114851Z`, 세대는
`f68f6702d0a94351b7cb4d7f74274bfe`다. 흙 4개 및 중단된 과업이 남는다.
NBT 근거는 `results/world-identity-before-nbt.json`, `results/world-identity-after-nbt.json`.
이전 단계 receipt는 `results/world-pre-identity-receipt-20261004.json`에 보존했다.
UUID 동일성을 확인한 것이며 실제 동물 소유권·모든 업적을 별도로 시험한 것은 아니다.

## 회귀 검사와 한계

WSL 테스트 72개 통과, Windows 68개 통과/4개 Linux 전용 제외. PowerShell 구문과
WSL shell 구문을 확인했다. 손상·동시 쓰기·저장 발행 중 오류·다른 월드·진행도 불일치·
오래된 확인/명령 재발행·복원 후 일시정지·새 과업 baseline을 검사했다.

전원 차단 또는 강제 종료 때 진행 중 상태까지 저장하는 기능은 아니다. 마지막 완료 저장본으로
복원하며 주기적 중간 저장, GUI 상태와 정책 은닉 상태 복원은 아직 없다. 긴 플레이·다른 차원·
다양한 아이템/NBT·멀티플레이어 서버의 광범위한 지속성은 이번 시험으로 입증하지 않았다.
LLM+실게임 결합 및 요약의 실제 모델 품질 검증도 이번 범위에 넣지 않았다.
