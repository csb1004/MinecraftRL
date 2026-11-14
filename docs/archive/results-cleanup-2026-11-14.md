# 결과 폴더 정리 보고 (2026-11-14)

- 이유: 11-14 15:23 D 드라이브가 가득 찼다(남은 공간 약 20MB). D:에 있는 WSL 디스크가 커지지 못해 읽기 전용이 됐고, 밤샘 사슬(건축 R0, 채굴 v3)이 멈췄다.
- 사용자 지시(11-14): "result중에 옛날꺼 있지? 네 판단 하에, 지금 학습하는데 필요없다고 생각되는 애들은 정리만 해서 보고서형태로 남기고 지워도 좋아".
- 지운 양 54.4GB. D: 남은 공간은 20MB에서 98.2GB가 됐다.
  - 첫 두 삭제(31.3GB) 뒤 이미 75.6GB가 비어 있었다. 지운 양보다 약 44GB 더 늘었는데, 그 사이 WSL이 멈췄다가 다시 시작하며 디스크를 복구했다. 원인은 확인하지 않았다.
- 지운 파일은 휴지통을 거치지 않았다(D:가 가득 차 있었다). 되살릴 수 없다.
- 기계용 목록: [results-cleanup-2026-11-14-provenance.json](results-cleanup-2026-11-14-provenance.json)(데이터셋·jar 해시),
  [results-cleanup-2026-11-14-folders.json](results-cleanup-2026-11-14-folders.json)(폴더별 확장자·크기·남긴 파일 수).

## 1. 채굴 배치의 옛 중간 데이터셋 16개 — 19.8GB

`results/multi-block-20261108/data/`의 `r0, r0-1, r0-2, a-r0-2, a-r0-3, a-r0-4, p-r0-4, p-r0-5, s-r0-5, s-r0-6, l-r0-7, l-r0-8, l-r0-9, l-r0-9g, m-r0-9, m-r0-9g`(.pt).

- 학습 실행 기록(latent 청크, 텔레메트리, fixture)에서 `scripts/build_reacquire_dataset.py`로 만든 파생 파일이다. 원본 실행 기록은 그대로 있다.
- 각 데이터셋의 `.summary.json`은 남겼다. 거기에 sha256, 검증 실행, 사용한 실행 목록과 청크 해시가 있다. 해시는 provenance 파일에도 옮겨 적었다.
- 체크포인트(`train/*/checkpoint.pt`)는 지우지 않았다.
- 가장 최근의 `m-r0-10.pt`와 `m-r0-10g.pt`(m-final 학습 자료)는 남겼다. 채굴 v3는 실행 기록에서 새 데이터셋(v3-r0-10)을 만들고, 오프라인 관문은 학습 보고서의 NLL 값을 비교하므로 옛 데이터셋이 필요 없다.

## 2. 10월에 만든 게임 jar 26개 — 11.5GB

`.runtime/world-save/<패치 해시>.jar`와 영수증 `.json`. 10-04부터 10-19 사이에 패치 변형마다 하나씩 만든 수정 Minecraft jar다(각 444MB).

- 채굴 배치(`multi-block-20261108`) 전체와 11-13·11-14 배치(점프, 놓기, 다리, 건축)의 실행 620판은 `c7c4fc2c…`(11-05)와 `b317d10a…`(11-14, 놓기 관측)만 썼다. 이 둘과 `ab17d96d…`(11-03)는 남겼다.
- 지운 jar의 영수증(sha256, upstream·source·patched source sha256)은 provenance 파일에 그대로 옮겼다. 그 jar로 돈 옛 실행의 `summary.json`에도 `jar_receipt`가 있다.
- 같은 패치가 다시 필요하면 `world_save_patch.build_save_jar`가 새로 만든다. 다만 새 jar의 sha256은 옛 것과 다를 수 있다.

## 3. 11월 1–3일 실험 폴더 5개의 무거운 파일 — 23.1GB

작은 글 기록(json, jsonl, md, txt, py, ps1, log 중 1MB 이하)은 남기고, 나머지(체크포인트 .pt, 큰 텔레메트리 .jsonl, 화면 .png, 압축 .zip, 큰 json)를 지웠다.
세 실험 모두 끝났고 후보를 기본 정책으로 승격하지 않았다. 지금 코드·설정·배치 스크립트는 이 폴더를 읽지 않는다(옛 실행 스크립트 `Invoke-ExecutionContractE2.ps1`, `Invoke-EffectSkillE1.ps1`, `prepare_execution_pilot.py`만 참조하며, 끝난 배치라 다시 돌리지 않는다).

| 폴더 | 실험과 결론(결과 문서) | 실행 수 | 전체 | 지움 | 남김 |
|---|---|---|---|---|---|
| `execution-contract-e2-20261103` | E2 실행 계약·전체 과업 학습. 후보 미승격([결과](../execution-contract-e2-results-2026-11-03.md)) | 29 | 9.9GB | 9.86GB(14572개) | 29MB(1122개) |
| `execution-contract-e2-20261103-retention-check` | E2 회수 보충 비교 8회(같은 문서) | 8 | 1.7GB | 1.71GB(2263개) | 6MB(210개) |
| `common-recovery-r1-20261101` | 공통 회수 R1. 후보 채택 없음([결과](../common-recovery-r1-results-2026-11-01.md)) | 56 | 7.6GB | 7.60GB(10763개) | 45MB(1722개) |
| `effect-skill-e1-20261102` | E1 공통 결과 스킬과 RM→RL 연결. 후보 승격 없음([결과](../effect-skill-e1-results-2026-11-03.md)) | 15 | 3.4GB | 3.34GB(5999개) | 15MB(774개) |
| `effect-skill-e1-20261103-resource-check` | E1 자원 간섭 보충 한 쌍(같은 문서) | 2 | 0.6GB | 0.56GB(1088개) | 2MB(86개) |

- 남은 것: 실행별 `summary.json`·`request.json`, 계획(`plan.json`), 보고서, 감사 스크립트(`audit_tensors.py` 등), 작은 로그.
- 사라진 것: 학습 체크포인트와 후보 가중치, 원본 텔레메트리(큰 `telemetry.jsonl`), 화면 프레임. 이 실험의 결과를 다시 계산하거나 화면으로 확인할 수는 없다. 결론은 위 결과 문서에 있다.

## 4. 지우지 않은 것

- 지금 학습·평가가 쓰는 배치: `multi-block-20261108`(데이터셋 2개 외 전부), `jump-2026111[34]`, `place/bridge/build-20261114`, 모든 `*-prompts`.
- 설정·코드가 참조하는 폴더: `iron-demo-probe-20261104-arena-v2`, `pit-escape-probe-20261104`, `pit-ceiling-probe-20261104`, `reacquire-20261107`, `reacquire-head-check-20261107`, `iron-demo-baseline-20261103`.
- `iron-demo-eval-20261105*`(약 3.7GB): 목표 모니터 재생([monitor-spec-replay](../monitor-spec-replay-2026-11-05.md))에 다시 쓸 수 있어서 남겼다.
- `.runtime/mamba/pkgs`(17.4GB, 패키지 캐시): steve1 환경이 하드링크로 공유하고 있을 수 있다. WSL 안에서 `micromamba clean`으로만 정리해야 해서 손대지 않았다.
- `.runtime/steve1`, `.runtime/vpt`, `.runtime/artifacts`(환경, 모델, minerl wheel), `data/`(월드), `logs/`.
- WSL 디스크 파일은 117.8GB지만 안에서 실제로 쓰는 공간은 9.2GB다. 압축(compact)은 시스템 작업이라 하지 않았다.
