# 설정 지도

| 파일 | 역할 |
|---|---|
| `sources.lock.json`, `models.json` | 공식 upstream 소스 버전, 모델 출처·파일 identity |
| `environment-steve1.yml`, `environment-vpt.yml`, `pip-constraints.txt` | 분리된 WSL 정책 환경 |
| `fixture-tools.txt` | Windows 오프라인 NBT/자료 도구 환경 |
| `conda-*-linux-64.candidate.txt` | 환경 후보 기록; 실제 설치 검증을 대신하지 않음 |
| `experiment.json` | 공식 정책 기본 조건 |
| `goals-wood-then-dirt.json`, `unified-goals-wood-then-dirt.json` | 목표 큐/통합 요청 예제; 서로 다른 schema에 유의 |
| `goals-wood-then-stone.json` | 미지원 스킬 거부용 예제; 돌 수집이 구현됐다는 뜻이 아님 |
| `logo-wood-validators-v1.json` | 고정 LOGO 원목 판별식 |
| `dialogue-persona.txt` | 대화 페르소나; 자동 LLM 학습 루프와 구별 |

완료 실험의 실제 입력은 해당 `results/<batch>/`의 고정 사본·해시를 따른다.
여기의 예제를 고친 뒤 과거 실험 설정이 바뀐 것으로 소급 해석하지 않는다.
비밀정보/API 키는 이 폴더에 저장하지 않는다.
