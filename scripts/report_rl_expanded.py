"""Generate the completed expanded-batch report from audited measurements."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];base=a.batch.resolve();s=json.loads((base/'experiment.json').read_text());audit=json.loads((base/'audit.json').read_text());t=s['training']
assert s['completed'] and audit['verified']
train=[r for r in audit['runs'] if r['split']=='train'];ev=[r for r in audit['runs'] if r['split']=='eval']
rows=['# 확대 RL 배치 결과 — 2026-10-06','','상태: 실제 수집·학습·반복 검증 및 원본 로그 감사 완료.','','[고정 프로토콜](rl-expanded-protocol-2026-10-06.md). 결과 폴더: `'+str(base.relative_to(root)).replace('\\','/')+'`.','','## 실제 학습','',f"8시드×2회, 총{t['samples']} transition으로 행동 출력층·가치함수를{t['updates']}회 추가 갱신했다. 부모78회 포함 누적{t['total_updates']}회다. 영상·순환 backbone은 고정했고 실제 가중치 및 optimizer 상태를 저장·재로딩했다.",f"성공{sum(r['success'] for r in train)}/16, 시간 초과{sum(r['status']=='time_limit' for r in train)}/16. 보상 합{t['reward_sum']}, 확률 재계산 최대 오차{t['behavior_logp_max_error']}, 재로딩 오차{t['checkpoint_reload_logp_error']}.",f"actor 최대 절대 변화{t['actor_max_abs_change']}. Torch 학습 단계 peak allocated{t['torch_peak_allocated_mib']:.2f}MiB/reserved{t['torch_peak_reserved_mib']:.2f}MiB. 게임/backbone을 해제한 frozen-feature 출력층 학습 메모리이며 전체 모델 학습 메모리가 아니다.",f"checkpoint SHA256: `{t['checkpoint_sha256']}`.",'','| 학습 시행 | 결과 | 행동/tick | 현실 제어 초 |','|---|---|---:|---:|']
if audit.get('technical_attempts'):
 rows[5:5]=['','별도 기술 실패1회: 최초 train-104801-2는1206행동/180.0308초 동안 원목을 얻지 못했고, 게임 종료 후 감독기 유예 만료로 worker가 SIGTERM을 받았다. 원본은 보존하고 학습·정상 완료 표본에서 제외했다. 종료 후처리 결함을 수정한 뒤 같은 조건으로 정확히1회 재실행했으며, 아래16개 정상 완료 표본만으로 전체 시도의 성공률을 표현하지 않는다. 원목 미획득이 관측된 이 시도를 숨기지 않고 기술 실패로 별도 집계한다.','']
for r in train:rows.append(f"|{r['label']}|{r['status']}|{r['steps']}|{r['seconds']:.3f}|")
rows+=['','## 별도 반복 검증','','같은 검증2시드에서 각각 전후3회다. 독립12개 시드 표본이나 최종 평가로 해석하지 않는다. 시간 초과도 분모에 포함하며 성공 시행 중앙값은 성공률과 함께 해석한다.','','| 시드 | 단계 | 성공/시행 | 시간 초과 | 개별 행동 수(전체) | 성공 행동 중앙값 | 성공 현실 초 중앙값 |','|---|---|---:|---:|---|---:|---:|']
for seed,phases in audit['validation'].items():
 for phase,v in phases.items():
  median=v['successful_median_seconds'];median='—' if median is None else f'{median:.3f}'
  rows.append(f"|{seed}|{phase}|{v['successes']}/{v['attempted']}|{v['time_limits']}|{v['steps']}|{v['successful_median_steps']}|{median}|")
for phase in ('before','after'):
 group=[r for r in ev if r['phase']==phase];successful=[r for r in group if r['success']]
 rows.append(f"\n전체 {phase}: 성공 {len(successful)}/{len(group)}, 시간 초과 {sum(r['status']=='time_limit' for r in group)}. 성공 행동 중앙값 {statistics.median(r['steps'] for r in successful) if successful else '—'}, 성공 제어 초 중앙값 {statistics.median(r['seconds'] for r in successful) if successful else '—'}.")
rows+=['','## 자원·지연','','단위ms, 각 시행 중앙값/p95. GPU는 다른 프로세스를 포함한 장치 전체 사용량의 표본 최대MiB이며 순간 최대를 보장하지 않는다. 현실 제어 시간에는 첫 추론도 포함된다.','','| 시행 | 추론 중앙값/p95 | 전체 제어 주기 중앙값/p95 | 장치 GPU MiB | 최종 서버 원목 개수 |','|---|---:|---:|---:|---:|']
for r in audit['runs']:
 i=r['inference_ms'];c=r['control_cycle_ms'];logs=sum(v for k,v in r['final_server_inventory'].items() if k.endswith('_log'));rows.append(f"|{r['label']}|{i['median']:.2f}/{i['p95']:.2f}|{c['median']:.2f}/{c['p95']:.2f}|{r['device_vram']['sampled_peak_used_mib']:.0f}|{logs}|")
if (base/'measurement-note.txt').exists():
 rows+=['','측정 조건: validation-2000029-2-after 중 이전 실패 원본 로그를 읽는 별도 Windows Python 작업이 겹쳤다. 지연에 미친 영향의 크기는 측정하지 않았으며, 이 시행은 제외하거나 재실행하지 않았다. 현실 시간과 지연 수치는 로컬 시스템 부하를 포함한 관측값으로 해석한다. 자세한 동시 작업 기록은 결과 폴더 measurement-note.txt에 보존한다.']
rows+=['','## 감사와 해석 경계','',f"초기 서버/정책 상태 비교는 통과했다. 같은 시드 최초 실행 대비 픽셀이 다른 실행은{sum(not r['initial_pixels_match_first'] for r in audit['runs'])}개다. 픽셀 불일치를 이유로 표본을 교체하거나 재시도하지 않았다.",'','원본 전달 행동·보상·서버 인벤토리·매 행동 서버tick/월드시간 증가1·목표 후 새 행동 차단·입력 해제·실행별 자식 프로세스 정리·체크포인트 실제 적용·학습16개 파일 해시와 평가 제외를 감사했다. 최종 평가5개 시드는 미실행이다. 이번에 별도 수동 STOP을 유발한 실게임 시험은 하지 않았다.','', '기본 정책은 변경하지 않았다. 새 checkpoint는 연구 산출물이며 자동 승격하지 않는다. 새 checkpoint의 흙 수집·제작·다른 목표 성능은 이번 배치에서 평가하지 않았다. 단일 배치와 작은 검증 집합으로 일반 성능 향상·인간다움·문화 이해·최신 서버 호환성을 주장하지 않는다. 원 연구 명세와 장기 목표는 유지한다.','']
signal_path=base/'learning-signal.json'
if signal_path.exists():
 signal=json.loads(signal_path.read_text());mass=[e['last_64_advantage_abs_mass_fraction'] for e in signal['episodes']]
 rows+=['','## 판단과 학습 신호 진단','','전후 성공률은 각각6/6으로 같고, 시드별 성공 행동 중앙값은86→86과283→361이다. 이번 배치에서 효율 개선 근거를 얻지 못했으며 새 checkpoint를 개선판으로 채택하지 않는다. 두 개 검증 시드의 반복이므로 일반적인 성능 저하를 확정하는 결과도 아니다.',
 f"학습 가능 파라미터는{t['trainable_parameters']:,}개다. 실제 actor 변경·optimizer step·저장 재로딩·평가 시 해당 head 적용을 검증했으므로, 학습이 실행되지 않거나 잘못된 checkpoint를 평가한 경우는 이번 감사에서 배제됐다.",
 f"고정된 학습 return 목표에 대한 가치 예측 MSE는{signal['value_mse_against_fixed_training_targets_before']:.6f}→{signal['value_mse_against_fixed_training_targets_after']:.6f}로 감소했다. 학습 데이터 적합도이며 독립 검증 성능은 아니다. 원 advantage 양수{signal['positive_advantages']}개/음수{signal['negative_advantages']}개, 배치 정규화 후 양수{signal['normalized_positive_advantages']}개/음수{signal['normalized_negative_advantages']}개다. 모두 성공한 episode라도 모든 행동을 같은 방향으로 강화한 것은 아니다.",
 f"episode별 마지막64행동에 포함된 원 advantage 절댓값 합의 비율 중앙값은{statistics.median(mass):.1%}(범위{min(mass):.1%}–{max(mass):.1%})다. 이는 종료 근처 신호 분포의 기술통계이며 전체 gradient 기여율이나 개선 실패의 확정 원인은 아니다. gradient norm 제한은{signal['gradient_clip_activated_updates']}/{signal['optimizer_updates_inspected']}개 업데이트에서 작동했고, minibatch reference KL 최대는{signal['minibatch_reference_kl_max']:.6f}였다. reference는 이번 배치 시작 actor이며 최초 공개 정책과의 KL이 아니다.",
 '정상 학습 데이터16개가 모두 성공 episode이고 현재 검증 성공률도 이미100%여서 성공률 향상을 관측할 여지가 없었다. 원목1개 획득의 끝 보상, 제한된8개 학습 월드, 관측/실행 편차와 frozen backbone 범위가 결과 해석의 제약이다. 각각의 인과 기여는 이번 한 배치로 분리하지 못했다. 동일 설정의 학습을 자동 반복하거나 평가 결과로 학습률을 조정하지 않았다. 다음 단계에서는 더 학습할 횟수보다 개선을 구별할 과제·데이터·보상 기준을 먼저 검토하는 편이 타당하다. 최종 연구 범위를 원목1개로 축소하는 의미가 아니다.']
diagnostic_path=base/'timeout-diagnostic-audit.json'
if diagnostic_path.exists():
 d=json.loads(diagnostic_path.read_text());assert d['verified']
 rows+=['','## 별도 실제 시간 초과 진단과 자동 검사','',f"부모78회 checkpoint, 시드104801, 동일 저장 월드·임베딩, 평가 모드, 상한35초의 별도1회다. 결과는 {d['status']}, {d['steps']}행동/{d['seconds']:.4f}초, worker 종료 코드{d['worker_exit_code']}였다. 입력 해제 후 게임 정리{d['cleanup_seconds']:.4f}초, 실제 제어 종료 marker 및 감독기60초 후처리 경계를 확인했다. 시간 제한 이후와 중단 확인 이후 새 행동은 각각{d['dispatches_after_deadline']}회/{d['dispatches_after_ack']}회, 남은 소유 프로세스0이다. bootstrap={d['bootstrap']:.8f}로 유한하며 dispatch_commit을 사용했다.",
 '이221개 transition은 학습 데이터와 성능 검증12회에 포함하지 않는다. 원래1206행동/180초 실패 궤적을 다시 재현한 검사는 아니며, 짧은 실제 시간 초과와 자동 회귀 검사로 수정 경로를 검증했다.',
 '수정 후 전체 자동 검사: WSL172개 통과, Windows172개 중165개 통과/7개 제외. 검사 로그는 results/rl-expanded-tests-wsl-after-fix.txt 및 results/rl-expanded-tests-windows-after-fix.txt에 보존했다. 후처리 유예 초과·수동 STOP 우선·최종 supervisor 실패 표본의 학습 거부를 포함한다. 이번 배치에서 별도 수동 STOP 실게임 시험을 추가한 것은 아니다.']
final_path=base/'final-verification.json'
if final_path.exists():
 f=json.loads(final_path.read_text());assert f['verified']
 rows+=['','## 최종 보존·종료 확인','','프로젝트 소유 WSL/Windows Python·Java 프로세스 잔존0, STOP 유지. 원 연구 명세·부모 checkpoint·새 checkpoint·기존14개 학습 궤적·저장 월드10개의 해시를 확인했고 원 기술 실패 summary는 보존 manifest와 일치했다. 최종 평가5시드는 미실행, 기본 정책 승격 없음. 세부 근거는 결과 폴더 final-verification.json에 보존한다.']
(root/'docs/rl-expanded-validation-2026-10-06.md').write_text('\n'.join(rows),encoding='utf-8')
print(json.dumps(dict(updates=t['updates'],samples=t['samples'],validation=audit['validation'])))
