# 공통 결과 스킬 E1 실행과 독립 학습

현재 지원: `goal_spec_v1`의 원목/흙 인벤토리 증가. 기존의 전체 목표 큐와
내부 탐색/접근/상호작용/회수 단계는 유지한다. E1은 그 위에 실제 획득 결과의
학습 경계를 제공한다. 특정 월드의 좌표나 고정 키 조작을 정책으로 사용하지 않는다.

## 데이터 흐름

|구성|소유 정보|다음 구성에 전달하는 정보|
|---|---|---|
|사람/향후 LLM|목표 제안|동일 검증을 거친 대상·수량·우선순위·의존성|
|상위 목표 큐|최종 성공, 순서, 활성화 기준|목표 ID/개정/해시, 실제 기준 인벤토리|
|결과 스킬 계약|다음에 넘을 인벤토리 최고치|고정 요청, 기대 결과, 증거 종류|
|조건부 정책|공식 영상 특징 + 목표 임베딩 + 구조화 조건|일반 게임 행동|
|실제 게임/고정 관측|행동 이후 인벤토리·tick|확정 스킬 결과, 별도 최종 목표 결과|
|독립 PPO 작업|새 train 궤적, 부모 checkpoint|검증된 후보 가중치와 학습 기록|
|향후 모니터/지식 저장소|요청/결과 이벤트의 소비자|추가 학습 제안 및 근거; 아직 미구현|

`skill_reward=1`은 상위 목표 성공률과 별개다. 3개 목표에서 2개만 획득하면 전체 목표는
실패할 수 있지만 획득한 스킬 구간은 학습한다. 같은 행동에서 2개를 얻으면 보상은 1회이며,
버렸다 다시 주운 수량이 이전 최고치를 넘지 않으면 새 보상은 없다.

## 실행

기존 MineRL/STEVE-1 WSL 환경에서 `python -m minecraftrl.cli episode`에 다음을 지정한다.
전체 명령과 해시는 `results/effect-skill-e1-20261102/*-command.json`에 보존된다.

```text
--rm-conditioning goal-queue --rm-program <goal_spec_v1.json>
--rm-goal-bank <검증된 recovery goal-bank> --effect-learning
--recovery-variant nonlinear-on --recovery-world <검증된 시작 월드.zip>
--recovery-task collection --recovery-split train
--recovery-action-budget 1200 --recovery-sampling-seed <고정 정수>
--limit-s 180 --stop-file <프로젝트 results/STOP>
```

실행 시 `--seed`는 저장 월드의 seed와 같아야 한다. 별도 pickup 조건은
`--recovery-task pickup`이며 준비용 월드라는 점을 기록한다. 개발에는 split을
`development`로 설정한다. 재사용할 새 후보는 `--recovery-checkpoint`로 명시한다.
E1과 기존 R1/S2 checkpoint는 서로 호환되지 않는다. 공식 파일은 수정하지 않는다.

실행 기록의 `effect-trajectory.pt`는 실제 입력 tensor, 원래 목표 보상/종료,
별도 스킬 보상/종료, 요청/결과/근거와 모델 정체성을 담는다.
`skill-events.jsonl`은 첫 확정 행동 이후 시작과 완료/실패/중단을 기록한다.
사용하지 않은 추론이나 bootstrap으로 실행 이벤트를 만들지 않는다.

## 독립 학습

격리된 WSL Python에서 다음 작업을 별도로 실행한다. 실행 중 게임 정책 가중치를
교체하지 않는다. 다음 rollout이 새 checkpoint를 명시적으로 사용해야 한다.

```text
python scripts/train_effect_adapter.py --root <프로젝트 WSL 경로>
  --output <새 results 하위 출력> --stop-file <results/STOP>
  --variant nonlinear-on --runs <새 train 실행 폴더1> <새 train 실행 폴더2>
```

후속 업데이트는 `--parent <직전 checkpoint.pt>`와 **그 부모로 새로 수집한** 자료를 쓴다.
개발/평가/취소/기술 실패/다른 부모 궤적은 거부한다. 원본 서버 로그 감사에는
`scripts/analyze_effect_skills.py <배치 폴더>`를 사용한다.

학습기는 요청과 결과를 재계산하고 실제 입력 tensor, 행동 확률과 가치 재현을 검사한다.
스킬 완료에서 GAE를 끊고, 미완료 시간 제한에는 다음 상태의 가치로 bootstrap한다.
공식 backbone/motor는 고정, 작은 residual adapter와 critic만 업데이트한다.
이는 온라인 동시 학습·모델 기반 planning·LLM/SQL 통합을 완료했다는 의미가 아니다.

STOP은 학습과 게임 모두에 적용된다. 완료된 일회용 실험 드라이버는 재실행하지 않는다.
결과·월드·가중치는 Git 제외 대상이며 새 후보는 자동 승격하지 않는다.
