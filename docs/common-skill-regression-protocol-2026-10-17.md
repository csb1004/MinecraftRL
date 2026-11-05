# 공통 이동 후보의 반복 및 이전 조건 회귀

강도를 높인BC 후보가새개발2월드에서12/12도착했다. 같은후보를고정하고,
다른samplingseed와앞선개발조건에서도통하는지추가학습없이확인한다.
config `configs/common-navigation-regression-v1.json`, output `results/common-navigation-generalization-20261017`.

- 후보 `584e7cf507a9ce4376f542f46269660fc40fc9f900e3878f425bc9f727066fb1` 고정.
  기준은 기존PPO `e033f1ef7902102569d01459d85cdd80252c141f4e1346bdb40832502e8acaea`.
- repeat-a/b: v3의동일시작월드/목표에 **새samplingseed17359**, 기준/후보각2회(4게임).
- previous-a/b: v2의동일시작월드/목표/samplingseed17231, 후보2회.
  기준은이미감사한v2의2실행을재사용한다. 같은seed/actor/world/points/split과raw파일hash를검증하고,
  후보의 초기 PNG·pose·recurrent memory·주요 정책 source·JAR·embedding/본체 조건이 맞아야 비교로 인정한다.
- 새게임은총6회다. 재사용baseline2회를새실행으로세지않는다. 모든실제결과와경로를구분한다.
- 통과기준: 후보24목표중22개이상도착, 어느case에서도기준보다도착수가줄지않음,
  총도착증가또는같은도착수에서행동감소. 각각의6목표는연속되고독립표본이아니다.
- 이gate는평지목표추종의추가개발근거다. 자연지형/안전/제작등전체목표달성으로보지않고
  default도자동교체하지않는다. 결과에따라실제공통skill RL/안정도착/지형범위로계속한다.
- 새학습0/최종평가seed사용0/API·Colab·게시0. 실제제어175초예산/180초이하,
  STOP이후새추론/행동0·입력해제·소유PID정리검증을그대로적용한다.

완료된배치를다시실행하는것이아니라새가중치/새sampling조건을별도경로에평가한다.
추가실험은현재v3대조가종료된뒤GPU를순차사용한다.
