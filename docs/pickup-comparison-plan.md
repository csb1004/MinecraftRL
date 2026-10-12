# 줍기 정체: 공식 평가 문구와 짧은 문구의 비교

가설: 화면상 채굴했으나 줍지 못한 현상은 실제 인벤토리/STOP 결함이 아니라 수집
정책의 출력 분포와 관련된다. 고정된 공식 평가 문구가 짧은 문구보다 이 상태에서
수집을 개선하는지 먼저 검사한다. 실패 내부 원인을 미리 확정하지 않는다.

참조: 고정된 external/steve1/steve1/run_agent/paper_prompts.py의 dirt 문구,
run_agent/2_gen_vid_for_text_prompt.sh의 text_cond_scale=6.0,
steve1/data/text_alignment/vae.py의 확률적 prior 샘플링.

- 출발점은 failure-20261004T032512557525Z 슬롯의
  3b2f51efcf7c4bbe8edfee14623eb5db 세대다. original pointer/manifest/world.zip은 보존한다.
- 결과 폴더 안 각 시행마다 world store를 복사하며 슬롯 이름/플레이어 UUID는 유지한다.
- 같은 저장 월드에 기존 시드 5개 × short/official 문구 2개 = 총 10회.
  새 지형 5개가 아니라 동일 실패 상태에서의 초기 난수 시드 비교다.
- 시드별 두 문구 순서는 번갈아 바꾼다. guidance=6.0, weights, 실제 목표 dirt 1,
  baseline 및 완료 조건은 동일하다. reset 후 문구와 embedding SHA256을 기록한다.
- 실행당 제어 상한 180초, 무획득 60초면 paused/저장 종료. 자동 재시도 없음.
- 모델/실환경 오류가 있으면 남은 시행을 미실행으로 남긴다. 정책의 무획득은 결과로
  기록하고 다음 미리 정한 조건으로 진행한다. 수집은 실제 인벤토리로만 판정한다.
- 기본 프로필은 short 그대로 둔다. 연구용 opt-in 옵션으로 비교하고 한 월드 결과를
  일반 정책 향상으로 해석하지 않는다. API 모터 대체/유료 LLM 호출 없음.

## 구현·검증 기록

공식 문구 적용/기본값/지원하지 않는 조건/격리 저장소 옵션을 실패 검사로 먼저
확인했다. 문구 선택, 실제 조건 기록, CLI 전달과 프로젝트 내부의 world-store-root를
추가했다. 연구용 저장소는 새 결과 경로에 만들고 기존 저장소는 덮어쓰지 않는다.
사용자 지침에 따라 단일 에이전트·기존 프로젝트에서 실행하고 자체 검토한다.

완료: WSL의 기존 Namespace 호출 회귀 4개를 기본값 호환으로 수정한 뒤 99개 통과.
Windows 95개 통과/4개 제외. 실제 10회 완료, short 5/5와 official 4/5. 기술 오류와
미실행은 0개. 마지막 official의 no_progress는 정책 미완료로 보존했다.
원본 세대와 pointer 보존, 제어/종료 검사 통과. 기본 문구 변경 근거가 없어 short 유지.
상세 결과와 해석 경계는 pickup-comparison-validation-2026-10-04.md에 기록했다.
