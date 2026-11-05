# 공통 이동 v6: 동일 자료 추가 학습과 별도 실행 판정

v5는2,992갱신을 완료했으나 기존 제안 확률 gate와 별도 결정적 명령 검사를 모두 통과하지 못했다.
근거리 중립4/4, 원거리 전진3/4이며 실패 기록과 checkpoint는 보존한다. 실제 v5 게임은 아직0이다.

v6는 같은6개 train 실행의 expert2,989+상대 목표2,990정답으로16epoch를 추가 학습한다.
학습률0.0001, batch32, KL상한3.0, 실제 전달 명령 BC를 유지한다. 가중치는 이어받고 AdamW는
새로 시작하므로 연속 optimizer 실행과 같다고 주장하지 않는다. 원본 actor/value는 유지한다.
사람 영상과 개발 실패 궤적은 gradient에 넣지 않는다. 공통 입력 계약 변경도 이번에는 적용하지 않는다.

실행 전 고정한 `results/common-navigation-dagger-preparation-20261017-v6/mode-readiness-template.json`은
같은4개 실제 관측에서 근거리 중립4/4, 원거리 전진4/4를 요구한다. 이는 v5 때 선언한 사후 탐색적
프로토콜의 연속이며 기존 제안 확률 실패를 지우지 않는다. 통과해도 일반화 성공은 아니다.

통과하면 새 train-b calibration을 실행한다. 실제 서버/클라이언트의 안정 도착12tick,6목표 중5개 이상,
제어180초 이하, 교사0, 확률·명령 재생과 STOP·키해제·소유PID 감사를 요구한다. 그 다음에만 같은
delivered_mode 조건에서 v5/v6 두 후보를 개발2월드에서 새로 비교한다. 최소10/12 및 월드별 비악화,
도착 수 증가 또는 동률에서 행동 수 감소가 기준이다. 최종 시드는 사용하지 않고 기본 자동 승격도 없다.

v5와 v6의 같은12개 label sidecar SHA를 확인하며 결과는 아래에 보존한다.

- 학습: `results/common-navigation-dagger-20261017-v6/workflow.json`
- mode 진단: `results/common-navigation-dagger-preparation-20261017-v6/mode-readiness.json`
- 실제 calibration: `results/common-navigation-dagger-calibration-20261017-v6/workflow.json` (미실행 시 표본0)
- 실제 개발 비교: `results/common-navigation-dagger-evaluation-20261017-v6/workflow.json` (미실행 시 표본0)

## v5 출력 분기 진단의 한계

같은8입력에서 CFG를 빼면 카메라가0이 되는 사례가 있으나 근거리4개 중3개는 전진으로 바뀌었다.
conditional/unconditional 단독 분기는 실행 정책으로 학습한 조건이 아니다. 출력 분기 변경만으로
해결됐다고 결론내리지 않으며 현재 CFG와 학습 조건을 유지한다.

## 완료 결과

16epoch/2,992갱신, checkpoint `b25c960d20896bc1f1e085e07bd25fe6ffa2505c6a8c58aa97cd27cfb5ce69f8`.
학습 전달 NLL1.4994→1.1197, 별도 모니터1.4485→1.4736으로 모니터는 소폭 악화했다.
학습 정답 감사·root/value 보존·reload0, v5와12개 label sidecar의 byte 동일성을 확인했다.
기존 확률 gate는 근거리0.76555로 미달, 별도 mode도 근거리4/4·원거리3/4로 미달했다.
실제 calibration/개발 게임은 실행하지 않았으므로 v6 게임 표본0이다. 기본 정책 교체0이다.
같은 데이터를 다시 무조건 반복하지 않고, [공통 이동 입력 계약](common-movement-input-contract-2026-10-17.md)의
별도 root-context 조건으로 이어간다. 이것이 실패의 확정 원인이라는 뜻은 아니다.
