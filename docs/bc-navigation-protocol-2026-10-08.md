# 목표 조건 BC 전이 파일럿 — 2026-10-08

공식 FindCave index의 첫3개 세션을 다운로드 전에 train2/development1로 고정했다.
동일 contractor의 서로 다른 세션이므로 독립 플레이어 일반화로 해석하지 않는다.
영상 검토에서 언덕/단차 주변 이동 및 구덩이 둘레 이동을 확인했다.
긴 bedrock 벽에서 정체 후 탈출한 전문가 시연은 아니며 **지형 탐색 전이 후보**다.

`configs/bc-navigation-20261008.json`의 구간과 원래 목표 `find a cave`를 사용한다.
광류와 마우스 회전 대조에서 세 파일 모두 offset0의 상관이 최대(.962/.830/.979)였으며
공식 loader와 같은 frame[i]/action[i] 대응을 채택한다. N+1의 마지막 프레임은 행동으로 만들지 않는다.
40프레임부터 순서를 유지해 recurrent state를 계산하고 지정 구간만 loss에 포함한다.
GUI/미지원 입력/버튼 투영 손실/범위 밖 카메라는 loss에서 제외하고 시간열은 삭제하지 않는다.

사전학습 STEVE-1 vision/recurrent 고정, 기존 gap 후보 action head만 BC한다.
외부 자료의 RM 입력6열은 미관측0이며 해당 가중치 열과 value는 고정한다. 0을 search 라벨로
해석하지 않는다. 실제 실행 때에는 기존 서버 RM 입력을 유지한다. 목표 embedding은 원래
FindCave 문구로 고정 seed9201에서 생성한다. 실제 wood 평가의 embedding은 기존 것을 유지한다.
이 둘을 같은 과업 조건이라고 주장하지 않는다.

학습 seed9202, AdamW lr1e-5/weight_decay0, batch16, epochs2, gradient norm1,
loss=전문가 행동 NLL + 1.0×부모 분포 KL. PPO reward/old_logp를 만들지 않는다.
개발 NLL 비증가, train KL≤.03, 가중치 변화>0, reload logp 오차0,
RM 열/value 불변을 만족하면 실제 비교 후보다. 이 기준 통과는 성능 승격이 아니다.
실패하면 해당 후보를 보존하고 이 turn에서는 추가 하이퍼파라미터 탐색 없이 원인을 보고한다.

실제 비교 후보가 생기면 기존 별도 개발 긴 벽2000003과 자연 세계2000029에서 부모/후보 각1회,
동일 seed/embedding/월드/180초 제한으로 비교한다. 긴 벽은 통과+원목 획득, 자연 세계는 원목
획득을 확인한다. 시드2000029는 기존 개발 세계이며 최종 평가가 아니다.
초기 상태를 대조하고 지연/VRAM/행동수/원목/STOP/키0/자식 PID를 기록한다.
두 쌍만으로 기본 정책을 승격하거나 일반화를 주장하지 않는다. 최종5시드는 미실행 유지.
