"""Finalize the fixed comparison and separately versioned death-boundary regression evidence."""
import datetime,json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
base=root/'results/wood-followup-20261007'
state=json.loads((base/'experiment.json').read_text())
audit=json.loads((base/'audit.json').read_text())
assert state['completed'] and not state['unrun'] and len(audit['rows'])==19
assert [r['label'] for r in audit['rows'] if not r['valid']]==['probe-death']
regression=[]
for name in ('noop','learned','normal'):
    run=Path((root/f'results/death-boundary-{name}.log').read_text().splitlines()[0])/'run'
    report=json.loads((run/'summary.json').read_text())
    check=json.loads((run/('probe-audit.json' if name=='noop' else 'shaping-audit.json')).read_text())
    assert report['execution_valid'] and report['worker_exit_code']==0 and check['verified']
    assert not report['process_cleanup']['remaining_live_pids'] and not report['rl_training_eligible']
    assert check['one_server_tick_per_action']
    if name!='normal':
        assert report['status']=='environment_done'
        if name=='learned':assert check['terminal'] and check['bootstrap']==0 and check['components']['failure']==-1
    else:assert report['status']=='target_reached'
    jar=json.loads((run/'runtime-jar-cache.json').read_text())
    assert jar['sha256']=='2de8cff9236d41fc3abcf12ee1788210016bc45a10bd83eff7af6734bc9a9677'
    regression.append(dict(label=name,path=str(run),report=report,audit=check,jar=jar))
prior=json.loads((root/'results/rl-horizon-new-worlds-20261007/final-verification.json').read_text())
assert digest(root/'docs/research-spec.md')==prior['research_spec_sha256']
for item in prior['checkpoints'].values():assert digest(item['path'])==item['sha256']
for item in state['checkpoints'].values():assert digest(item['path'])==item['sha256']
for path,sha in state['world_sha256'].items():assert digest(path)==sha
original=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
for item in original['training']['trajectories']:
    p=str(item['path']).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(root)+'/')
    assert digest(p)==item['sha256']
tests={}
for platform in ('windows','wsl'):
    p=root/f'results/death-boundary-tests-{platform}.txt';log=p.read_text()
    assert re.search(r'^OK(?: \(skipped=\d+\))?\s*$',log,re.M)
    skipped=re.search(r'OK \(skipped=(\d+)\)',log)
    tests[platform]=dict(run=int(re.search(r'Ran (\d+) tests',log)[1]),skipped=int(skipped[1]) if skipped else 0,sha256=digest(p))
assert (root/'results/STOP').read_text().startswith('RL collection ended')
changed={p:digest(root/p) for p,sha in state['source_sha256'].items() if digest(root/p)!=sha}
assert {Path(p).name for p in changed}=={'cli.py','fixture_probe.py','world_save_patch.py','workshop_monitor.py'}
result=dict(verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    comparison=audit,death_boundary_regression=regression,tests=tests,
    fixture_tests=dict(passed=4,log='results/wood-fixture-tests.txt'),
    changed_python_since_comparison=changed,workshop_java_sha256=digest(root/'src/minecraftrl/WorkshopObservation.java'),
    preserved_prior_training_trajectories=24,research_spec_sha256=prior['research_spec_sha256'],
    original_checkpoints_and_worlds_preserved=True,optimizer_updates=0,default_policy_promoted=False,
    final_seeds_unrun=state['final_seeds_unrun'],stop_latched=True)
write_json(base/'final-verification.json',result)

lines=['# 원목 RM 후속 검증 — 2026-10-07','',
 '실제 Windows/WSL 로컬 검증. 추가 학습0, 기본 정책 변경0, 최종 평가5시드 미실행. '
 '원문 연구 범위는 유지한다. [고정 조건](wood-followup-protocol-2026-10-07.md).','',
 '## 자연 월드5개 고정 비교','',
 '| 개발 시드 | 부모 결과/tick | v2 결과/tick |', '|---|---|---|']
for seed in (2100011,2100023,2100037,2100041,2100059):
    pair=[next(r for r in audit['rows'] if r['label']==f'natural-{seed}-{c}') for c in ('parent','v2')]
    cells=[('획득' if r['success'] else '180초 미획득')+f" / {r['steps']}" for r in pair]
    lines.append(f'| {seed} | {cells[0]} | {cells[1]} |')
lines+=['', '부모4/5·v2 5/5 획득. 시간 초과를180초로 계산한 평균 제어시간은62.113/23.639초다. '
 'v2는3세계 단축·1세계 증가·1세계 동일이었다. 각 조건 반복1회이며 자동 승격하지 않는다.',
 '초기 서버 상태·recurrent memory·sampling seed는7대조 모두 동일했다. 초기 픽셀은6/7동일; '
 '2100059의 차이는 bbox(460,130,463,135), 채널 평균 절대차(.00104,.00134,.00030)였다. '
 '작은 화면 차이라도 성공률 차이를 학습 효과만으로 단정하지 않는다.',
 '서버 보조 단계/거리 입력 조건이며 영상 단독 정책의 성능으로 해석하지 않는다. '
 '시행별 서버 획득량·GPU 메모리·추론 중앙값/p95·제어 주기는 `results/wood-followup-20261007/audit.json`에 있다.','',
 '## 파괴·우회 비용과 정책 선택','',
 '| 시험장 | scripted 파괴/우회 tick | 부모/v2 tick |','|---|---:|---:|',
 '| 흙 벽·긴 우회로 | 207 / 344 | 405 / 495 |',
 '| 돌 벽·짧은 우회로 | 347 / 273 | 750 / 528 |','',
 '모두 실제 원목을 획득했다. 정해 둔 두 경로의 비용이며 전역 최적 경로의 증명은 아니다. '
 '실제 할인 반환값도 각각 파괴(.068205 > -.027035), 우회(.006440 > -.027990)를 선호했다. '
 '그러나 두 정책 모두 돌 벽에서도 직접 통과했다. v2의 완료 시간 단축과 우회 선택 학습을 구분한다.',
 '시험장 두 정책 대조는 초기 서버 상태/픽셀 모두 동일했다. Scripted 비용 측정은 명시적 준비 진단이며 '
 '정책 성능·학습 데이터에 합산하지 않는다.','',
 '## 드러난 관측·복구 문제','',
 '- 처음 잠근 원목과 실제 공격 대상이 다르면 채굴 중에도 접근 단계에 남는다. 흙 벽 v2의 '
 '공격 입력 후 나무 hit 관측322개 중251개가 접근이었다. 이는 실제 블록 피해 이벤트 수가 아니다. '
 '초기 임시 집계의 `[0]` 버튼 배열 처리 오류는 수정했다.',
 '- 2100059 부모는 후반 돌 벽 주변에서 이동이 정체됐다. Search8/approach1397, '
 '원목 거리약19~22블록, 최종 체력20·서버 인벤토리0이었다. 부모는 새 context 열이0이므로 '
 '이를 RM 입력이 유발했다고 단정하지 않는다. 막힌 접근의 복구와 재탐색이 필요하다.',
 '- 현재 회수 거리는 실제 드롭 entity가 아닌 파괴 위치 근사다. 다른 skill에 그대로 일반화하지 않는다.','',
 '## 실제 사망 경계 오류와 수정','',
 '기존 사망 진단은466전이 후6서버 tick 증가로 오류 중단됐다. 원본 게임 로그와 저장 NBT에서 '
 '사망1회 후 체력20·원래 스폰 위치로 자동 재생성된 것을 확인했다. 과업 실패 표본에서 제외했다.',
 'RL에만 사망 종료를 지정하고 저장/새 월드 양쪽에 전달했다. 별도 패치 JAR에서 자동 재생성을 '
 '막고 죽은 플레이어의 마지막 관측을 발행하며 키를 해제한다. 원본 설치 파일은 보존했다.',
 '| 수정 후 검사 | 결과 | 실제 전이 |','|---|---|---:|']
for entry in regression:
    r=entry['report'];lines.append(f"| {entry['label']} | {r['status']} | {r['steps']} |")
lines+=['', '사망 두 경로는 실제 server health0/alive0, held keys0, 실패 비용-1이며 행동당1서버 tick을 '
 '유지했다. 학습 정책의 terminal bootstrap은0이다. 죽은 환경에는 neutral step을 추가하지 않고 '
 '게임 스레드에서 키를 해제한 증거와 프로세스 정리를 확인했다. 이후 새 정책 행동0, 소유 PID0.',
 '이 세 시행은 수정 전 비교와 JAR 버전을 분리한다. [원인·경계·증거](rl-death-boundary-2026-10-07.md).','',
 '## 준비 실패와 보존','',
 '초기 scripted 회수는 밑동 앞 정지로2108tick/180.057초, 점프 수정은 위 원목에 막혀 '
 '2889tick/180.036초 미획득했다. 밑동 제거→회수로 수정한 뒤 네 비용 진단이 성공했다. '
 '두 원본 실패는 `rl-eval-104729-20261007T041331833104Z`, `rl-eval-104729-20261007T042222467204Z`에 보존했다.',
 '학습 후 v2의 별도 기존 개발 확인은134tick/23.577초 획득 및 원시 감사 통과였다 '
 '(`rl-eval-2000003-20261007T040326524711Z`). 이전86tick과 직접 개선 비교하지 않는다.',
 'Windows211검사 중199통과/12제외, WSL211중207통과/4제외, 격리 NBT 검사4통과. '
 '원문 연구 명세·원본 체크포인트·이전 학습 궤적24개·원본 월드 해시를 보존했다. '
 '처음 빌드 호출의 JDK 경로 누락은 격리 환경 명령으로 수정했고 시스템 Java 설정은 바꾸지 않았다.','',
 '## 다음 작업 단위','',
 '[후속 목록](next-task-units.md)과 [공통 단계 계약](task-decomposition.md)을 루트 AGENTS.md에 연결했다. '
 '원목 목표 추적→접근 정체 복구→실제 아이템 회수 관측→동일 조건의 학습 대조 순으로 진행한다. '
 '그다음 흙·제작·작업대·복수 목표를 각각 완료 증거와 복구 조건이 있는 단위로 확장한다.',
 '이번 성공률만으로 장기 과업·인간다움·문화 이해 또는 최신 서버 호환을 주장하지 않는다.',
 '전체 증거: `results/wood-followup-20261007/final-verification.json`.']
(root/'docs/wood-followup-validation-2026-10-07.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(verified=True,comparison_records=19,regression_records=len(regression),tests=tests)))
