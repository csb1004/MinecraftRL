# 센서 등급 S0–S1 기반 구현 계획 (2026-11-22)

> 실행: 주 흐름에서 순서대로 구현한다(AGENTS.md, 서브에이전트 없음). 각 단계는 검사를 먼저 쓰고 실패를 본 뒤 구현한다.

**목표:** [센서 등급 관측 설계](../../specs/2026-11-19-sensor-tier-observation-design.md)에서 정해진 것 중 게임 없이 만들 수 있는 기반을 구현한다. 모드 센서(S0 코드), `SensorFrame`, 본 지도, 가드 1 센서 경로와 가드 2–4, 누출 검사 1·2·4·8이다.

**구조:** 모드에는 새 선택 관측 `SensorObservation`을 더한다. `sensor_monitor`를 켠 jar에만 들어가므로 기존 jar 해시는 그대로다. 파이썬 쪽은 actor 입력 타입 `SensorFrame`과 본 지도 `SeenMap`을 만들고, 가드는 이 둘과 지시(계획)만 읽는다. 센서 값은 행 JSON이 아니라 묶음 npz(`sensor_v1`)로 기록한다.

**범위 밖 (다음 묶음):** 헤드별 센서 특징·단계와 센서 교사, S0 게임 1판, S2 그림자 기록. 게임 실행은 사용자의 시작 지시 뒤에 한다.

## 지키는 규칙

- 사용자 요청(11-22): "정해진 것들 구현해줄래? 그걸 보고 좀 정하면 나을려나?"
- `sensor_monitor`를 켜지 않은 실행은 지금과 바이트 단위로 같아야 한다. jar 패치 해시, 행 JSON, 가드 동작이 모두 그대로다.
- 센서 묶음은 클라이언트 월드(`mc.world`)와 클라이언트 플레이어만 읽는다.
- 가드는 행동을 지우기만 한다. 행동을 대신 만들지 않는다.
- 완료 실험의 코드 경로·해시·CLI는 바꾸지 않는다.

## 작업

### 1. 모드 센서 `SensorObservation.java` (S0 코드)

- 새 파일 `src/minecraftrl/SensorObservation.java`.
  - 키: `sensor_available`, `sensor_client_tick`, `sensor_eye_x/y/z`, `sensor_yaw`, `sensor_pitch`, `sensor_foot_x/y/z`, `sensor_hfov`, `sensor_vfov`, `sensor_range`, `sensor_cols`, `sensor_rows`
  - 광선 136개(15×9 격자, 마지막은 조준선 `objectMouseOver`): `sensor_ray_kind`, `sensor_ray_dist`, `sensor_ray_x/y/z`, `sensor_ray_dx/dy/dz`, `sensor_ray_fluid`, `sensor_ray_fluid_dist`
  - 주변 27칸: `sensor_voxel_kind`, `sensor_voxel_fluid`. 순서는 (dy+1)·9 + (dz+1)·3 + (dx+1)이다.
  - 손 도구: `sensor_tool_item`, `sensor_tool_damage`, `sensor_tool_max`
- 종류 어휘는 기존 `block()`에 `fire`, `soul_fire`를 더한다(가드 4). 유체는 `none`/`water`/`lava`로 따로 적는다.
- `world_save_patch.py`: `sensor_monitor` 인수. 켰을 때만 클래스와 `infoJson.add("sensor", ...)`, `capture` 호출을 넣는다.
- `workshop_monitor.workshop_survival(sensor=...)`: `sensor` 모니터 스키마.
- `pipeline/cli.py`: 설정 `sensor_monitor`를 jar와 환경에 전달한다.
- 검사: `test_world_save_patch`에 "끄면 해시가 같고, 켜면 클래스와 호출이 들어간다". WSL에서 javac로 새 클래스를 upstream jar에 대고 컴파일한다(게임 없음).

### 2. `pipeline/sensor.py`: `SensorFrame`, `SensorLog`

- `sensor_frame(info, t) -> SensorFrame | None`: `info['sensor']`와 `info['workshop']`의 센서 등급 키만 읽는다. 위치는 `client_x/y/z`, 시선은 `yaw/pitch`다.
- `ACTOR_TIERS`: 필드마다 등급 태그. `check_actor_tiers()`는 오라클·심판 태그가 있으면 예외를 낸다.
- `SensorLog(folder)`: 묶음 npz(`sensor-000.npz`)와 `manifest.json`(schema `sensor_v1`, 광선 배치, 어휘, 등급표)이다. `add(step, raw) -> row index`.
- 세션: `sensor_monitor`가 켜지면 행에 `sensor_row`를 적고 묶음에 쓴다.
- 검사: 같은 info에서 오라클·심판 키(`demo_mining_*`, `server_x`, `pursuit_*`, `demo_last_destroy_harvestable`, fixture)만 바꾸면 `SensorFrame`이 같다(누출 검사 3). 로그 왕복.

### 3. `pipeline/seen_map.py`: 본 지도

- `SeenMap.update(frame)`: 광선이 지나간 칸은 공기, 닿은 칸은 종류, 유체가 고체보다 가까우면 그 칸은 유체로 적는다. 주변 27칸도 적는다. 칸마다 근거(틱, 출처)를 남긴다.
- `kind(cell)`: 본 적 없으면 `unknown`이다. 다시 잰 값은 덮는다. 판마다 새로 만든다.
- `cast(origin, direction, reach)`: 본 지도 위의 광선이다. 처음 만난 고체, 공기만 지나 끝, 또는 `unknown`을 돌려준다.
- `depth_below(cell, limit)`: 발 아래 고체까지의 깊이다. 바닥을 모르면 `None`이다.

### 4. `pipeline/sensor_guard.py`: 가드 1 센서 경로와 가드 2–4

- `SensorGuards(guard, needs, *, skill_kind, planned_uses, wear_margin)`
  - `pre_dispatch(frame, seen, action, prev_frame) -> list[event]`
  - 사건마다 `mask`(지울 키 목록)와 `hold`가 있다.
- 가드 1: 지금 조준선 칸과, 이번 행동의 카메라 변화만큼 돌린 방향을 본 지도에서 4.5블록까지 쏜 칸을 함께 판단한다(`Guard.judge`). 종류가 `other`·`unknown`이고 손 도구가 지키는 블록 하나에라도 모자라면 공격을 지운다(`guard_aim_uncertain`, 헛막음으로 따로 센다).
- 가드 2 `guard_drop_needed`: 손 아이템이 필요 목록에 있거나 핫바 키를 같이 누르면 `drop`을 지운다.
- 가드 3 `guard_tool_wear`: 남은 횟수 < 계획에 남은 사용 수 + 여유분이면 공격을 지우고 hold한다. 여유분 기본값 2는 임시값이다. S2 전에 지난 기록으로 정한다(내 결정).
- 가드 4 `guard_hazard_move`: 땅에 있을 때, 키별로 2틱 뒤 발 칸을 예측한다. 용암·불, 체력 6 이하가 되는 낙하, 바닥 모름, (다리·점프가 아니면) 피해 낙하면 그 이동 키와 달리기·점프를 지운다.
- 같은 가드가 같은 단계에서 `hold_s` 넘게 이어지면 hold한다(가드 1과 같은 방식).
- 세션: 센서가 켜지고 감독이 있을 때만 쓴다. 지운 키는 행의 `masked`와 사건으로 남긴다.

### 5. 누출 검사 (설계 §6의 1·2·4·8)

- import 검사: `sensor.py`, `seen_map.py`, `sensor_guard.py`가 fixture·`stage_replay`·과제 교사 모듈을 import하지 않는다.
- 반사실 검사: 가드 판단이 서버 harvestable·서버 채굴 대상만 바꿔서는 달라지지 않는다.
- 등급표: 세션이 센서 경로를 켤 때 `check_actor_tiers()`를 통과해야 한다.

## 검사 명령

WSL `.runtime/steve1`, `PYTHONPATH=<worktree>/src:<worktree>/tests`, `CUDA_VISIBLE_DEVICES=`:
`python -m unittest test_pipeline_sensor test_pipeline_seen_map test_pipeline_sensor_guard test_world_save_patch test_pipeline_guard test_pipeline_session -v`.
마지막에 전체 unit discover를 돌린다(기존에 알려진 `reacquire_batch` 오류 9건은 따로 적는다).
