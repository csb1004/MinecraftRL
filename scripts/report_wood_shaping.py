"""Finalize local pilot evidence after all bounded runs and training finish."""
import datetime,json,re
from pathlib import Path
from minecraftrl.runtime import write_json
from minecraftrl.world_store import digest

root=Path(__file__).resolve().parents[1]
runs=[]
for name in ('collection','evaluation','truncation','v2-collection'):
    folder=Path((root/f'results/wood-shaping-{name}.log').read_text().splitlines()[0])/'run'
    summary=json.loads((folder/'summary.json').read_text())
    audit=json.loads((folder/'shaping-audit.json').read_text())
    assert audit['verified'] and summary['execution_valid'] and not summary['process_cleanup']['remaining_live_pids']
    runs.append(dict(label=name,path=str(folder),audit=audit,summary=summary,trajectory_sha256=digest(folder/'trajectory.pt')))
training=[]
for name in ('wood-shaping-training-20261007','wood-shaping-v2-training-20261007'):
    folder=root/'results'/name;record=json.loads((folder/'training.json').read_text())
    assert record['completed'] and record['behavior_logp_max_error']==record['checkpoint_reload_logp_error']==0
    assert digest(folder/'checkpoint.pt')==record['checkpoint_sha256']
    training.append(dict(path=str(folder),report=record))
tests={}
for platform in ('windows','wsl'):
    path=root/f'results/wood-shaping-tests-{platform}.txt';log=path.read_text()
    assert re.search(r'^OK(?: \(skipped=\d+\))?\s*$',log,re.M)
    skipped=re.search(r'OK \(skipped=(\d+)\)',log)
    tests[platform]=dict(run=int(re.search(r'Ran (\d+) tests',log)[1]),skipped=int(skipped[1]) if skipped else 0,sha256=digest(path))
prior=json.loads((root/'results/rl-horizon-new-worlds-20261007/final-verification.json').read_text())
for item in prior['checkpoints'].values():assert digest(item['path'])==item['sha256']
assert digest(root/'docs/research-spec.md')==prior['research_spec_sha256']
allowed={'cli.py','rl_collect.py','rl_core.py','rl_policy.py','workshop_monitor.py','WorkshopObservation.java'}
changed={}
for path,sha in prior['runtime_source_sha256'].items():
    current=digest(root/path)
    if current!=sha:
        assert Path(path).name in allowed,path
        changed[path]=current
original=json.loads((root/'results/rl-value-data-expansion-20261006/experiment.json').read_text())
for item in original['training']['trajectories']:
    path=str(item['path']).replace('/mnt/d/DATA/Desktop/MinecraftRL/',str(root)+'/')
    assert digest(path)==item['sha256']
assert (root/'results/STOP').exists()
result=dict(verified=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=runs,
            training=training,optimizer_updates=sum(t['report']['updates'] for t in training),tests=tests,
            changed_runtime_source_sha256=changed,preserved_prior_training_trajectories=len(original['training']['trajectories']),
            prior_checkpoints_preserved=True,research_spec_sha256=prior['research_spec_sha256'],
            stop_latched=True,default_policy_promoted=False,preferred_experimental_reward='wood-potential-v2',
            final_seeds_unrun=prior['final_seeds_unrun'])
write_json(root/'results/wood-shaping-verification.json',result)
lines=['# 원목 단계 RM·거리·시간 보상 연결 결과 — 2026-10-07','',
       '실제 Windows/WSL 로컬 실행. 단계 RM과 보상을 actor/value에 연결하고 PPO 학습까지 검증했다. '
       '효율 향상이나 우회/파괴 선택 학습의 증거는 아직 없다. 기본 정책 승격 없음.','',
       '[사전 조건 및 v2 수정](wood-shaping-protocol-2026-10-07.md). 원본 v1 결과도 보존한다.','',
       '| 실행 | 버전 | 실제 결과 | 정책 tick | 제어 초 |','|---|---|---|---:|---:|']
for run in runs:
    s=run['summary'];lines.append(f"| {run['label']} | {s['reward_config']['name']} | {s['status']} | {s['steps']} | {s['elapsed_control_s']:.3f} |")
lines+=['','5초 truncation은 의도적으로 제한한 종료 진단이며 성공률 표본에서 제외한다. '
        '학습/개발 평가도 서로 합산하지 않는다. 개발 시드2000003은 과거에도86tick으로, 개선 주장을 하지 않는다.','',
        'v1 수집은 원래 부모 head, v2 수집은 v1에서7회 갱신한 head를 사용했다. '
        '동일 학습 월드의98→161tick은 보상 버전만 바꾼 대조가 아니며 관측/반복 편차도 통제하지 않았다. '
        'v2에서 추가한 실패 비용은 이 성공 궤적에서는0이다.','',
        '## 구현과 검증','',
        '- 서버 중심 시선으로 확인한 원목을 잠그고 탐색/접근/채굴/수집/완료 상태를 기록한다.',
        '- 단계 one-hot 5개와 거리 1개를 actor/value 입력에 추가했다. 기존 열은 유지하고 새 열은 0으로 초기화한다. '
        '서버 보조 입력을 쓰므로 영상 전용 정책으로 해석하지 않는다.',
        '- 획득 +1, potential shaping .2*(.99*Phi(next)-Phi(current)), 틱당 -.0001을 분리 기록한다. '
        '종료 잠재값이0이므로 shaping 합이 음수일 수 있다. 총합만으로 행동 품질을 판정하지 않는다.',
        '- v1의 시간 비용은 긴 성공보다 조기 실패를 선호할 수 있다는 수식 검사를 보존했다. '
        'v2는 실제 환경 실패 종료에만 -1을 추가한다. 같은 초기 상태의 완전한 할인 반환값에서 '
        '성공과 실패의 순서를 바로잡았다. 사용자 STOP/기술 오류는 학습 제외, 시간 초과는 bootstrap이다.',
        '- 실제 수집 단계는 개발 평가에서10tick, v2 수집에서16tick 관측됐다. v2에서는 다섯 단계가 모두 관측됐다. '
        '거리는 실제 item entity가 아니라 파괴 위치까지이며 '
        '인벤토리 획득만 완료로 인정한다. v2 실제 사망 시험은 미실행; 실패 분기는 단위/수식 검사다.',
        '- 실제 모든 transition의 action/context/보상 항목을 서버 로그와 재대조했다. 목표/중단 뒤 새 정책 행동0, '
        '입력 해제 및 소유 PID 잔존0. 5초 진단은 마지막 미완료 상태의 bootstrap을 보존했다.',
        f"- Windows {tests['windows']['run']-tests['windows']['skipped']}통과/{tests['windows']['skipped']}제외; WSL {tests['wsl']['run']}통과.",
        f"- 실제 optimizer 업데이트 합계 {result['optimizer_updates']}회. 각 버전은 한 epoch, 새 optimizer이며 "
        '기존 부모의 누적 업데이트 수와 합산하지 않는다.',
        '- 각 학습의 행동 확률 재계산·체크포인트 재로딩 오차0. v1의 신규 RM 입력 열에 실제 gradient 업데이트가 '
        '반영된 것은 `context-learning.json`으로 별도 확인했다.',
        '- v2 최종 학습 checkpoint는 tensor/optimizer 재로딩까지 검증했다. 그 최종 checkpoint의 학습 후 '
        '별도 gameplay 평가는 미실행이다. 실제 개발 평가는 v1 학습 checkpoint다.',
        '- 원래 연구 명세, 이전 checkpoint3개 및 학습 궤적24개 해시 보존을 확인했다. 최종5시드/유료/Colab/공개 미실행.','',
        '## 남은 실험','',
        '짧은 연결 시험이므로 벽을 부수는 편이 빠른 배치/돌아가는 편이 빠른 배치의 실제 비교, '
        '탐색·탈출 경험 확대, sparse·단계만·거리 추가·시간 비용 추가 조건의 다중 시드 비교가 남았다. '
        'v2 새 checkpoint의 개선이나 일반화를 주장하지 않는다. 이번 RM은 첫 원목 skill 내부용이며 '
        '대화 기반 장기 RM 전체의 학습을 완료한 것이 아니다.','',
        '## 근거','',
        '`results/wood-shaping-verification.json`에 실행 경로, 원시 감사, 학습 보고서, GPU/지연/제어 주기, 해시를 수록했다. '
        '`results/wood-shaping-cost-analysis.json`은 v1 문제, `wood-shaping-v2-cost-analysis.json`은 수정 수식 검사다.']
(root/'docs/wood-shaping-validation-2026-10-07.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(verified=True,updates=result['optimizer_updates'],runs=len(runs),tests=tests)))
