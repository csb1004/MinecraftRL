# 교정 구간 분석의 원본 계약 정정

분석 전 준비 검사에서 최초 문서의 첫 경유점 `(5.5,3.5)` 가정이 실제 student correction과
다른 것을 발견했다. `StudentCorrection.commit`은 neutral 인계 뒤 `recovery_waypoints`로
교사의 경유점을 바꾼다. 벽 앞 출발에서는 첫 점이 `(5.5,인계 시 canonical z)`이다.
원본6시연의 handoffs[].waypoints가 실행된 계약이며 이후3경유점은 최초 문서와 같다.

원본 프로토콜을 소급 수정하지 않는다. 이번 분석은 이 정정과 원본 인계 receipt를 함께 적용한다.
첫 실패는 train-104743-far의467프레임에서 고정z=3.5 대신 실제z=3.1508192341791528이었다.
가정한 위치로는 회전 부호까지 달라져 원본 행동 일치 검사가 실패했다. 분석기v1/오류를
`results/correction-action-diagnosis-precheck-20261009/`에 보존한다. 가중치·게임·시연 오류가 아니다.
원본 시연/모델은 수정하지 않았고 새 게임/optimizer는0이다. GPU 예측 분석 실행 전에 발견했다.
실제 인계 위치와 receipt 일치, 거울 반전, 잘못된 고정좌표 거부를 회귀 검사한다.

joint NLL과 exact_label_probability는 BC에서 사용한 이산 button/camera 정답의 확률이다.
camera gate가 꺼지면 서로 다른 camera code가 같은 환경 명령으로 합쳐질 수 있으므로,
이를 물리적 행동이나 경로 성공 확률로 해석하지 않는다. yaw 부호 확률은 gate를 포함해 별도로 계산한다.
