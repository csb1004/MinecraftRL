# 벽 우회·원목 획득 개발 검증 자료

[상세 보고서](../../local-navigation-reliability-2026-10-19.md)의 공개용 요약이다.
최종 보완판을 고정한 6회는 모두 실제 원목 1~2개를 획득하고 39.324~76.218초에 정상 완료했다.
같은 개발 미로의 두 시작점과 세 행동 시드를 사용했다. 서버 통행·시야 관측 조건이며 영상 전용 또는 연구용 최종 평가가 아니다.

- [최종 판정](navigation-final-verification.json): 개발 2회와 고정 반복 6회, 실제 원목 증가·저장 NBT·정상 종료·STOP 확인.
- [실험 계획](navigation-protocol.json), [소스 고정 해시](navigation-source-freeze.json), [99개 테스트 결과](navigation-tests.txt).
- [실제 이동 경로](navigation-verified-paths.png): 사후 평가용 지도와 정책 궤적.
- [1차 실패](navigation-first-round-failure.json), [1차 배치 중단 기록](navigation-first-round-interruption.json): 주변 시야 센서 보완 전 실패도 보존한다.
- [자료 해시](navigation-evidence-manifest.json): 로컬 원본과 동일한 공개 사본의 경로·크기·SHA256.

원시 화면·행동 궤적·전체 월드·체크포인트는 로컬 `results/`에 보존하며 이 커밋에는 포함하지 않는다.
이 폴더의 문서·JSON·그림만으로 전체 실험을 다시 실행할 수 있다는 의미는 아니다.
`scripts/run_local_navigation_comparison.py`, `run_local_reliability_batch.py` 등은 준비된 로컬 입력과 프로토콜을 사용하는 연구 실행 보조 도구다.
새 PC에서는 먼저 기존 모델·월드와 로컬 프로토콜을 준비해야 하며, 완료된 배치를 덮어쓰거나 반복 실행하지 않는다.

개선 기능은 기존 CLI의 `--rl-navigation-strategy local_passability`로 선택한다.
기본 전략과 원래 목표 RM·보상·종료 판정은 유지한다. 모든 실제 행동은 기존 고정 신경망에서 생성하며 추가 학습은 하지 않았다.
