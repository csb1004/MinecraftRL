# 사용자 요청으로 일시 중지 — 2026-10-04

**2026-10-05 재개 및 해당 단계 완료:** 아래 중지 시점에서 이어서 실제10회를 실행했다.
작업대4/5, 막대기 제작·저장2/5(모두8개), 정확 막대기 예산0/5.
[최종 기록](craft-recipes-validation-2026-10-05.md)을 현재 상태로 사용한다. 아래는 중지 당시 이력이다.

막대기·작업대 독립 실험을 준비하던 중 사용자가 PC 사용을 위해 중지를 요청했다.
이번 작업에서 Minecraft/GPU 실험은 시작하지 않았다. results/STOP을 유지한다.

## 현재 변경

- crafting_probe.py: planks/sticks/table 고정 레시피, 재료 예산 검사, 목표 생성 즉시 정지,
  정확한 소비·생산 수량으로 최종 witness 판정. 기본값은 기존 planks.
- cli.py: 별도 craft-probe에서 sticks/table 조건 허용. 일반 대화/RM 제작 지원은 추가하지 않았다.
- CraftGuiObservation.java, craft_gui.py: 막대기·작업대 커서/입력 칸 및 서버 수량·통계 읽기 확장.
- tests/test_craft_recipes.py: 먼저 신규 인터페이스 부재 실패를 확인하고 구현 후 통과.
- Windows test_craft*.py 15개 통과. 전체 회귀, WSL 테스트, 새 Java 빌드 및 실게임은 **미실행**.

## 재개할 작업

1. 변경 사항 검토. 기존 판자 witness가 최소 수량에서 정확 수량으로 바뀐 영향도 확인.
2. scripts/run_crafting_probe.py 및 분석기는 아직 판자 전용이다. 레시피와 source 저장본을
   명시적으로 선택하고 조건별 기록·재료 잔량·저장 NBT를 판정하도록 확장한다.
3. results/craft-recovery-20261004T071020156492Z의 성공 저장본 중 하나를 선택한다.
   experiment.json의 world_save.generation으로 실제 경로를 확인하고 판자4/흙1 및 NBT를 검사한다.
   각 조건·시드는 해당 저장본의 독립 복사본에서 시작하며 원본은 유지한다.
4. 막대기: 판자2 소비→막대기4, 판자2 잔량. 작업대: 판자4 소비→작업대1.
   프롬프트는 각각 craft sticks / craft a crafting table. 고정 시드5개, 각 제어180초 상한.
5. 선행 테스트/Java 빌드 후 실제 실행. 정책 생성 후 즉시 정지, 명시적 커서 회수 보조,
   서버 확정·GUI 종료·저장 NBT·자식 종료를 분리 기록한다. 미실행은 표본으로 계산하지 않는다.

기존 최종 판자 결과와 근본 원인 수정은 craft-recovery-validation-2026-10-04.md 및
craft-rm-root-cause.md 참조. 추가 실험 결과는 아직 없다.
