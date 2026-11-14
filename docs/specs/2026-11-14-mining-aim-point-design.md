# 채굴 v3: 교사 조준점 방향을 특권 입력으로 — 설계 (2026-11-14)

- 상태: 사용자 지시(11/13 "블럭을 제대로 캐는 것도 잘 안됐잖아? 밤동안 계속해서 이것들 학습해줘", 11/14 "계속해줘. 중간에 멈추지 말고
  네 판단에 필요하다고 판단되면 해"). 세부는 내가 정했다(**(내 결정)**). 브랜치 `mining-aim`(`jump-map` 위, 워크트리 `../MinecraftRL-mining3`).
- 앞선 결과: [채굴 v2](../mining-direction-results-2026-11-14.md). 대상 블록 중심 방향을 넣자 학습 배치 수집은 17/18이 됐다.
  하지만 학습 안 한 발 높이 철광석의 최종 평가에서 2/8(공식 8/8)에 그쳤다.
  블록 바로 옆에서는 중심이 위의 돌에 가린다. sight 교사는 보이는 면을 겨누는데, 그 점은 입력이 가리키는 중심보다 17도 아래였다.
- 과업, 단계, 교사(sight), 성공 판정(서버 인벤토리)은 그대로 둔다. 바꾸는 것은 헤드 특징의 기준점뿐이다.

## 1. 특징 (특권 입력, 기록에 밝힘)

`AIM_FEATURES = ('aim_sin_yaw', 'aim_cos_yaw', 'aim_pitch', 'aim_dist')`. sight 교사가 그 행에서 시선을 돌리려는 곳을 그대로 쓴다.

| 행 | 기준(교사 정답과 같음) | 거리 |
|---|---|---|
| near, 조준 대상 아님, 중심이 보임 | 대상 블록 중심 | 눈 → 중심 |
| near, 중심이 가림, 보이는 면 있음 | 그 면 안쪽 점(`sight_plan` 'face') | 눈 → 그 점 |
| near, 면도 안 보임, 설 자리 있음 | pitch는 블록 중심, yaw는 설 자리(`sight_plan` 'walk') | 발 → 설 자리(수평) |
| near의 나머지('hidden'), aimed, mining, collect | v2와 같은 대상 블록 중심 | 눈 → 중심 |

- 값: yaw 차이의 sin·cos, pitch 차이 / 60(±1로 자름), 거리 / 9(0..1로 자름). 대상이 없거나 단계를 모르거나 gui면 0 벡터다.
- 성질: near 행에서 교사 카메라 정답은 이 방향을 축마다 10도로 자른 값과 같다. 테스트로 묶는다.
- 기하는 fixture 블록에서 그때까지 부서진 블록을 뺀 것이다(교사 정답과 같은 tracker). 서버 자세·fixture 좌표를 쓰는 특권 입력이다.
- v2 특징(`MINING_FEATURES`)과 그 기록 검사는 그대로 둔다. 기록의 특징 이름으로 다시 계산해 비교하고, 출력할 특징은 따로 고른다.
- 헤드 체크포인트의 특징 이름이 `AIM_FEATURES`이면 게임에서도 이 특징을 계산한다(`LiveStageFeed(feature_names=...)`).

## 2. 학습

- 자료: m-r0-10과 같은 실행(R0–R6 벽 + L7–L10, 교사 sight, 철광석 제외)에 조준점 특징을 넣는다(`v3-r0-10`). 배치 균형 사본 `v3-r0-10g`.
  검증 실행은 m-r0-10과 같다(L10의 6판, 샘플링 12).
- v3-a: v3-r0-10g로 학습(KL 10, 균형, patience 2, 특징 배율 10).
- **오프라인 관문(게임 전, 미리 정함):** v3-a의 검증 NLL이 같은 행에서 m-final(1.555)보다 낮아야 한다. **(내 결정)**
- DAgger 한 라운드: L11 = v3-a로 학습 배치 3개 × 블록 6개(샘플링 13) → v3-r0-11 → 배치 균형 사본 → v3-final. 검증은 L11의 6판.
- 발 높이(low)는 계속 학습에서 뺀다. 일반화를 보는 자리다.

## 3. 평가 (게임 전에 정함)

- **개발:** 철광석, low 배치, seed 130363, 샘플링 1–4, 갈래 `aimpoint`(v3-final) 4판. 공식은 11-13 개발 결과(`dev-low/official`)를 쓴다.
  이 seed·배치는 이 질문에 네 번째로 쓴다. 개발용이라 허용하지만 최종만 일반화 증거로 본다.
- **판정(공식 대비, v2의 약점을 고침):** `dev_verdict(공식, aimpoint)` = 성공 수 ≥ 공식, 파괴 비율 ≥ 공식의 2배이고 공식보다 큼,
  성공 단계 시간 중앙값 ≤ 공식. 개발에서 공식은 4/4이므로 4/4가 필요하다.
- **최종(개발 통과 때만):** 새 seed **264575(처음 씀)**, 샘플링 61–68, 갈래 aimpoint·공식 각 8판, 같은 규칙.
- 함께 보고: 단계 시간 비율, 실패 판의 sight 계획(centre/face/walk) 비율.

## 4. 실행

- `results/multi-block-20261108/launch/aim.sh`. 건축 배치(`results/build-20261114/launch/build.log`)가 끝난 뒤 시작한다(게임은 한 번에 하나).
- `mining-aim` 브랜치가 `jump-map`에 병합돼 있어야 시작한다(로컬 병합만).

## 5. 게임 전 확인 (11-14 08:45, 게임 없음)

- v2 실패 판 분석: `results/multi-block-20261108/analysis/occlusion-v2.txt`(v2 결과 문서 §5).
- 실제 학습 기록 3판(L10 m-a: 머리 위 양털·모래, 바닥 속 청금석)에서 조준점 특징을 다시 계산했다(`aim_real_check.py`, 분석용).
  중심 특징과 다른 행은 교사가 face·walk 계획을 낸 행과 정확히 같았다(38, 32, 11행).
  그 행들에서 교사 정답의 시선 방향(yaw·pitch 부호)이 조준점 특징과 맞는 비율은 1.00, 중심 특징과는 0.61·0.56·0.82였다.

## 6. 구현

- `reacquire_teacher.py`: `AIM_FEATURES`, `aim_view`, `aim_features`, `feature_values(names, ...)`.
- `stage_live.LiveStageFeed(feature_names=...)`, `ReacquireHook`이 헤드의 특징 이름을 feed에 넘긴다.
- `reacquire_data.run_examples/build_dataset(features=...)`: `True`는 v2 특징, 이름 튜플은 그 특징. `build_reacquire_dataset.py --features aim`.
