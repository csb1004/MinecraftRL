# P1.1 Technical Finalization Amendment

2026-10-31. [원래 P1 명세](2026-10-31-stage1-rm-conditioned-rl-spec.md)는 그대로 보존한다.
이 문서는 첫 두 실행 이후, 후속 실행 전에 작성한 종료 처리 수정 기록이다.

- `81031001-fixed`: 실제 서버 원목 1개, 124 step, 제어 21.441초, worker exit 0.
- `81031001-staged`: 1180 step 동안 획득 0, 제어 180.023초 후 time_limit.
  180초는 새 dispatch를 허용하는 제어 상한이며 진행 중인 step/종료 기록 때문에 실측은 소폭 넘을 수 있다.
  입력 해제 verified/held key 0, 종료 정리 7.602초, episode_finished 및 결과 파일까지 기록했지만
  감독자의 제어 종료 + 10초 deadline에서 worker SIGTERM(exit -15). 최종 판정은 worker_failed.
  원본은 기술 오류로 제외하며 유효한 정책 실패 표본으로 바꾸지 않는다.
- 원래 감독자는 새 RM 조건의 제어 종료 표식을 지원하지 않았다. 정상 짧은 성공은 180초 deadline까지
  시간이 남지만, 180초를 다 쓴 실행은 기록 후 interpreter 정리가 같은 10초 예산에 포함되어 강제 종료됐다.
  완료 요약을 먼저 쓴 뒤 지연 종료하는 별도 프로세스로 같은 실패를 재현했다.

수정: STOP gate가 잠기고 중립 입력/정책/환경 close까지 성공한 뒤 `rm-control-finished.json`을 기록한다.
감독자는 이 시점 이후 파일 기록·interpreter 종료에 최대 60초를 별도로 허용한다. 표식이 없으면 기존
10초 유예를 유지한다. 사용자 STOP은 이 추가 시간을 적용하지 않으며 새 정책 행동은 계속 금지한다.
60초 이후에도 종료되지 않으면 기술 오류이며 소유 프로세스를 정리한다. 종료 표식은 성공 획득의 증거가 아니다.

회귀 검증: 새 RM 종료 검사 4개(수정 전 정상 정리 케이스 1개 실패 → 수정 후 전부 통과),
기존 RL 종료 검사 6개, supervisor 검사 7개가 WSL에서 통과했다. 실제 게임 수정 검증은 후속 실행 대상이다.

정책/문구/임베딩/난수/관측/보상/180초 제어/준비 60 tick은 바꾸지 않는다. 변경은 제어 이후 수명주기뿐이다.
원래 유효한 fixed 실행은 재실행하지 않는다. invalid staged만 `81031001-staged-finalization-r1`에 기술 재실행하고,
나머지 8개 미실행 조건을 원래 순서로 수행한다. 첫 오류도 별도 attempt로 보고한다. 정책 성능 실패의 재시도는 없다.

실행: `scripts/Invoke-RMConditioningPilot.ps1 -ResumeFinalizationFix`.
고정 계획 `results/rm-conditioning-20261031/pilot-finalization-r1-plan.json`,
새 소스 snapshot `source-at-game-finalization-r1`, 원본은 `source-at-game`에 보존한다.
혼합 기술 revision은 결과 보고에 표시하고, 이를 논문의 봉인된 최종 평가로 간주하지 않는다.

첫 staged의 1180개 행동은 모두 search 조건이었다. 따라서 이 기록만으로 단계 전환 효과를 판단할 수 없다.
문구나 센서의 성능 수정은 P1.1에 섞지 않는다. 임베딩 전환 자체와 탐색 목표의 적절성은 결과에서 구분한다.
