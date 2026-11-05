"""Prepare immutable recovery worlds or goal vectors; never runs a policy."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from minecraftrl.runtime import write_json


def worlds(root,batch):
    from minecraftrl.recovery_fixture import make_fixture
    folder=batch/'worlds';folder.mkdir(exist_ok=False)
    items=[]
    train_seed=81111001
    plans=[
        ('train-r1-wood-front',train_seed,'train','oak_log',2.5,0,0,30,'none'),
        ('train-r1-dirt-front',train_seed,'train','dirt',2.7,90,0,30,'none'),
        ('train-r1-wood-side',train_seed,'train','oak_log',3.2,-70,40,25,'none'),
        ('train-r1-dirt-side',train_seed,'train','dirt',3.0,155,-40,25,'none'),
        ('train-r2-wood-far',train_seed,'train','oak_log',4.3,35,35,20,'none'),
        ('train-r2-dirt-turned',train_seed,'train','dirt',3.1,-45,135,25,'none'),
        ('train-r2-wood-step',train_seed,'train','oak_log',3.8,100,0,15,'step'),
        ('train-r2-dirt-occluded',train_seed,'train','dirt',4.5,-125,-25,20,'occluder'),
        ('dev-pickup-front',81112001,'development','birch_log',2.8,-25,10,32,'none'),
        ('dev-pickup-side',81112003,'development','dirt',3.6,75,-65,20,'none'),
        ('dev-pickup-turned',81112007,'development','birch_log',3.3,-150,-145,28,'none'),
        ('dev-pickup-step',81112009,'development','dirt',4.2,115,20,12,'step'),
        ('dev-pickup-occluded',81112013,'development','birch_log',4.9,10,45,20,'occluder'),
    ]
    for i,(name,seed,split,item,distance,bearing,view,pitch,obstacle) in enumerate(plans):
        source=batch/f'prepare-{seed}'/'initial-world.zip'
        layout=dict(item=item,distance=distance,bearing=bearing,view_offset=view,pitch=pitch,obstacle=obstacle)
        result=make_fixture(source,folder/(name+'.zip'),layout,split)
        items.append(dict(id=name,seed=seed,split=split,world=(folder/(name+'.zip')).relative_to(root).as_posix(),
            sha256=result['sha256'],task='pickup',program='dirt1' if item=='dirt' else 'wood1',sampling_seed=111200+i))
    for i,seed in enumerate((train_seed,81112001,81112003,81112007,81112009,81112013)):
        split='train' if seed==train_seed else 'development'
        source=batch/f'prepare-{seed}'/'initial-world.zip';dest=folder/f'natural-{seed}.zip'
        shutil.copyfile(source,dest)
        meta=json.loads(source.with_suffix('.json').read_text());meta['experimental_split']=split
        write_json(dest.with_suffix('.json'),meta)
        for target in (('wood1','dirt1') if split=='train' else (('dirt1',) if i in (2,4) else ('wood1',))):
            items.append(dict(id=f'{split}-natural-{seed}-{target}',seed=seed,split=split,
                world=dest.relative_to(root).as_posix(),sha256=meta['sha256'],task='collection',program=target,sampling_seed=111300+i*2+(target=='dirt1')))
    write_json(batch/'world-manifest.json',dict(schema='recovery_worlds_r1',cases=items,policy_actions=0,
        train_dev_separate=True,dev_birch_subtype_held_out=True,novel_item_kind_generalization=False))
    print(f'Prepared {len(items)} cases; no policy actions',flush=True)


def embeddings(root,batch):
    import numpy as np
    import torch
    from minecraftrl.recovery_learning import SCOPE,PROMPTS,load_bank
    from minecraftrl.rm_purpose_conditioning import checked_tokens
    folder=batch/'goal-bank'/'pickup';folder.mkdir(exist_ok=False)
    sys.path.insert(0,str(root/'external/steve1'))
    from mineclip.mineclip.tokenization import get_tokenizer,tokenize_batch
    from steve1.config import MINECLIP_CONFIG,PRIOR_INFO
    from steve1.mineclip_code.load_mineclip import load
    from steve1.data.text_alignment.vae import load_vae_model
    from steve1.utils.embed_utils import get_prior_embed
    tokens={k:checked_tokens(v,get_tokenizer('clip'),tokenize_batch) for k,v in PROMPTS.items()}
    config=copy.deepcopy(MINECLIP_CONFIG);config['ckpt']['path']=str(root/'weights/steve1/attn.pth')
    model=load(config,device='cuda').eval()
    prior=load_vae_model(dict(PRIOR_INFO,model_path=str(root/'weights/steve1/steve1_prior.pt')))
    entries={}
    for key,prompt in PROMPTS.items():
        torch.manual_seed(31031);torch.cuda.manual_seed_all(31031)
        with torch.no_grad():value=get_prior_embed(prompt,model,prior,'cuda').astype(np.float32)
        path=folder/(key+'.npy');np.save(path,value,allow_pickle=False)
        entries[key]=dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            vector_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    from minecraftrl.rm_program import load_bank as original_bank
    _,meta=original_bank(batch/'goal-bank')
    write_json(folder/'manifest.json',dict(schema=SCOPE,prompts=PROMPTS,tokens=tokens,seed=31031,
        embeddings=entries,official_weights=meta['official_weights'],policy_actions=0))
    load_bank(batch/'goal-bank');print('Pickup goal embeddings verified; no policy execution',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--mode',choices=['worlds','embeddings','diagnostic'],required=True)
    args=p.parse_args()
    if not args.batch.resolve().is_relative_to(args.root.resolve()/'results'):raise ValueError('project-local batch required')
    if args.mode=='diagnostic':
        from minecraftrl.recovery_fixture import make_fixture
        result=make_fixture(args.batch/'prepare-81111001'/'initial-world.zip',args.batch/'diagnostic-world.zip',
            dict(item='oak_log',distance=2.6,bearing=10.,view_offset=0.,pitch=32.,obstacle='none'),'development')
        print(json.dumps(result),flush=True)
    else:(worlds if args.mode=='worlds' else embeddings)(args.root.resolve(),args.batch.resolve())
