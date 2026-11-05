# 자연 지형 공통 이동 자료의 준비 상태

기존 공개사람영상3세션을 검사했으나 아직 학습에 넣지 않는다. 입력 시점의 근거는 보강됐지만
보수적으로 선별한 이동 구간의 다양성과goal/RM 연결이 충분하지 않다.
결과 `results/common-human-movement-20261017`; 원본은 `data/bc-wood-search-20261008`에 그대로 보존했다.

| 녹화ID | 기존split | 행동/영상frame | 0lag 회전상관 | 이동 후보창 |
|---|---|---:|---:|---:|
|054834d1b04a|train|5279/5280|.8009|31|
|0a5ba522405b|train|6000/6001|.8194|0|
|0af0903eb6f4|development|6000/6001|.7115|1|

모두20FPS이고 앞1,100frame의 수평 optical flow와 기록된회전을 비교했을 때0lag가 ±1/±2보다
큰 상관을 보였다. 이는 시점정렬의경험적근거이며 모든frame의 완벽한 인과정렬 증명은 아니다.
기존manifest의영상/JSONL SHA256과일치한다. 원래세션분리를 유지했고 세션간자료를 섞지 않았다.

이동후보는40tick/2초,20tick간격의 **겹치는** 창이다. GUI·미지원입력·클릭상호작용·슬롯변경·
카메라clipping·damage counter변화·위치불연속·큰높이변화·작은순이동을 제외했다.
이는 준비용 보수적필터이며 남은구간이 모두 안전한 전문가 행동이라는뜻은 아니다.
특히향후지점을고른 hindsight endpoint는 사람이의도했던목표가 아니고, 좌표도 contractor의client metadata다.
이창수를독립성공표본으로세지않는다. 피해counter불변도안전정책이입증됐다는뜻이아니다.

`index_sources.py`와 `movement-candidates.json`에선별코드·원본hash·제외사유·미확인항목을보존했다.
training_ready=false, optimizer0/이자료의실게임평가0. 남은것은구간영상/F3overlay검토,
공식action projection검사,목표와RM문맥/특징추출계약,다양한세션확보다.
이준비검사는진행중인평지이동BC/실게임비교와구분한다.

후속 시각 검토: `movement-review.jpg`의 train 3개 창 및 development 1개 창에서 총 12프레임을
직접 확인했다. 표본 화면에는 F3가 없고 숲·야간·물가 이동이 보인다. 유일한 development 창은
깊은 협곡 가장자리에 접근해 아래를 보는 장면이므로, 피해 카운터가 그대로라는 이유만으로
안전한 이동 정답으로 분류하면 안 된다. 전체 구간을 검토한 것은 아니며 `visual-review.json`에
확인 범위를 기록했다. 여전히 training_ready=false, 해당 자료의 갱신 0이다.
