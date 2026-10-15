"""Audit completed trials sequentially in one GPU process; never rerun gameplay."""
import gc,json,sys
from pathlib import Path
import torch
from minecraftrl.world_store import digest
from audit_waypoint_run import main as audit


root=Path(__file__).resolve().parents[1];base=root/'results/adaptive-recovery-20261014'
index=json.loads((base/'completed-trials.json').read_text(encoding='utf-8'));assert len(index)==6
for key,relative in index.items():
    run=base/relative;proof=run/'waypoint-audit.json'
    if proof.exists():
        result=json.loads(proof.read_text(encoding='utf-8'));assert result['verified']
        for name,sha in result['source_sha256'].items():assert digest(run/name)==sha
        print(json.dumps(dict(trial=key,verified_existing=True,steps=result['steps'])),flush=True)
        continue
    sys.argv=['audit_waypoint_run.py',str(run),'--actor',str(root/'results/waypoint-training-goal-focused-20261010/checkpoint.pt')]
    audit();gc.collect();torch.cuda.empty_cache()
