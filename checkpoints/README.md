# 학습된 기술 헤드 (2026-11-20 스냅샷)

공식 STEVE-1(고정) 위에 붙는 보정 헤드 다섯 개의 최종 체크포인트다.
헤드만 학습했고 STEVE-1 본체와 공식 행동 헤드는 바꾸지 않았다. 파일마다 약 1MB다.

- **들어 있지 않은 것:** STEVE-1·VPT·MineCLIP 공식 가중치(`weights/steve1/`). 따로 받아야 한다([새 PC 설치](../docs/new-pc-setup.md)).
- **원본:** 로컬 `results/` 아래 학습 폴더에 그대로 있다. 이 사본은 같은 바이트다. 복사 뒤 sha256이 원본과 `train-report.json`의 `checkpoint_sha256`에 모두 맞는 것을 확인했다.
- **입력 등급:** 다섯 헤드 모두 fixture 좌표와 서버 자세로 계산한 특권 특징을 입력으로 받는다. [센서 등급 관측 설계](../docs/specs/2026-11-19-sensor-tier-observation-design.md)에서는 이 헤드들을 특권 참고 조건(R)으로 분류한다.
- **붙이는 법:** skill 명세의 `reacquire.checkpoint`에 경로를 적으면 `pipeline/cli.py`가 `reacquire_head.install`로 정책에 붙인다.

| 폴더 | 기술 | 원본 학습 폴더 | sha256 | 결과 (서버 판정) | 결과 문서 |
|---|---|---|---|---|---|
| `heads/mining-v3` | 채굴 v3 (조준점 입력) | `results/multi-block-20261108/train/v3-final` | `40c930c92352414119690ff988523539155f8781e34762b8559db0e960aa356b` | 개발·최종 판정 통과. 최종 8/8(공식 8/8), 채굴 시도 중 파괴 비율 5%(공식 53%), 성공 단계 시간 중앙값 11초(공식 38초) | [결과](../docs/mining-aim-results-2026-11-14.md) |
| `heads/place` | 블록 놓기 | `results/place-20261114/train/p-final` | `844f435624e2f0d1bc6a86b83988dff425f2c5b831ea1fe68497391c3517a4b3` | 개발 6/8, 최종 5/8(공식 0/8) | [결과](../docs/place-block-results-2026-11-14.md) |
| `heads/build-r5` | 건축 5차 (잘못 놓인 블록 제거 후 건축) | `results/build5-20261115/train/p5-final` | `b74495b8fb0c2c80678f137a9c7ec647e38af8ce9a9a37c95214e3e9eab902d6` | 개발 7/8, 최종 7/8(공식 0/8) | [결과](../docs/build5-results-2026-11-15.md) |
| `heads/bridge` | 다리 놓기 | `results/bridge-20261114/train/b-final` | `32ae5c2e3fe75285897fd6b78849c7177093f52d33c0f5130e952cda1f001056` | 개발 5/8, 최종 5/8(공식 0/8) | [결과](../docs/bridge-results-2026-11-14.md) |
| `heads/jump` | 점프 코스 (방향 입력) | `results/jump-20261114/train/j2-final` | `b98e3c2d06153c3a41b5328a501a5912f3fac7e0ba6ca1807b2f0230929322f5` | 개발 7/8, 최종 8/8(공식 0/8) | [결과](../docs/jump-direction-results-2026-11-14.md) |

- 각 폴더의 `train-report.json`은 학습 자료 해시, 하이퍼파라미터, 검증 NLL, 특징 목록을 담은 원래 학습 보고서다.
- **공개 예외:** `.gitignore`는 `*.pt`를 막는다. `checkpoints/heads/*/checkpoint.pt`만 예외로 올린다(사용자 결정, 2026-11-20). `scripts/check_publish_content.py`는 이 다섯 파일을 계속 표시한다. 의도한 예외다.
