"""One local official-prior preparation, with no Minecraft or paid API calls."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time

from minecraftrl.runtime import write_json
from minecraftrl.subgoal_skills import SKILLS
from minecraftrl.world_store import digest
from minecraftrl.rl_control import claim_automatic_stop,publish_completion


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve()
    if not out.is_relative_to(root/'results'):raise ValueError('project-local output required')
    out.mkdir(exist_ok=False);stop=root/'results/STOP'
    claim_automatic_stop(stop,out/'STOP-before.txt');started=time.monotonic()
    report=dict(completed=False,error=None,game_actions=0,api_calls=0,optimizer_updates=0,seed=9811)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0,str(root/'external/steve1'))
        from steve1.config import MINECLIP_CONFIG,PRIOR_INFO
        from steve1.mineclip_code.load_mineclip import load
        from steve1.data.text_alignment.vae import load_vae_model
        from steve1.utils.embed_utils import get_prior_embed
        config=copy.deepcopy(MINECLIP_CONFIG);config['ckpt']['path']=str(root/'weights/steve1/attn.pth')
        model=load(config,device='cuda').eval()
        prior=load_vae_model(dict(PRIOR_INFO,model_path=str(root/'weights/steve1/steve1_prior.pt')))
        items={}
        for name,spec in SKILLS.items():
            if stop.exists():raise InterruptedError('STOP during local embedding preparation')
            source=root/'results/goal-conditioning-diagnosis-20261009/explore_9711.npy'
            if name=='leave_stalled_area':
                assert digest(source)=='f8839fa46c254dbb7d8834b97d59afd30b995c53b671d22ae64b1ba9ea397d55'
                value=np.load(source,allow_pickle=False);seed=9711
            else:
                seed=9811;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                value=get_prior_embed(spec['prompt'],model,prior,'cuda')
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                repeat=get_prior_embed(spec['prompt'],model,prior,'cuda')
                assert np.array_equal(value,repeat)
            assert value.shape==(1,512) and np.isfinite(value).all()
            path=out/(name+'.npy');np.save(path,value)
            items[name]=dict(path=path.relative_to(root).as_posix(),prompt=spec['prompt'],seed=seed,
                file_sha256=digest(path),embedding_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                provenance='pinned official STEVE-1 text/prior; not a skill success claim')
        write_json(out/'catalogue.json',dict(schema=1,skills=items))
        report.update(completed=True,device=torch.cuda.get_device_name(),
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            source_sha256={str(path.relative_to(root)):digest(path) for path in (
                root/'weights/steve1/attn.pth',root/'weights/steve1/steve1_prior.pt',
                root/'external/steve1/steve1/utils/embed_utils.py',Path(__file__),out/'catalogue.json')})
    except BaseException as exc:report['error']=repr(exc)
    finally:
        report['elapsed_s']=time.monotonic()-started;write_json(out/'preparation.json',report)
        publish_completion(stop,'RL collection ended; neural subgoal embedding preparation finished.')
    print(report);return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
