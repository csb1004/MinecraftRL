"""Immutable saved starts for RL; no task reward or training during preparation."""
import json,shutil,zipfile
from pathlib import Path
from .world_store import digest,validate_archive,observation_evidence
from .runtime import write_json,jsonable,save_png
from .rl_core import WoodReward


def verify_saved_seed(compressed,seed):
    from .saved_inventory import read_nbt
    actual=read_nbt(compressed)['Data']['WorldGenSettings']['seed']
    if actual!=seed: raise ValueError(f'Saved world seed {actual} != requested {seed}')
    return actual


def checked_world(path,seed):
    path=Path(path);manifest=json.loads(path.with_suffix('.json').read_text())
    if manifest['seed']!=seed or digest(path)!=manifest['sha256']:raise ValueError('RL initial archive mismatch')
    validate_archive(path)
    with zipfile.ZipFile(path) as archive:
        levels=[name for name in archive.namelist() if name.endswith('/level.dat')]
        if len(levels)!=1:raise ValueError('Expected one saved world level')
        verify_saved_seed(archive.read(levels[0]),seed)
    return manifest


def prepare_world(env,output,seed,gate,stop_file):
    report=dict(status='error',execution_valid=False,steps=0,learned_policy_loaded=False)
    try:
        env.seed(seed);obs=env.reset()
        for _ in range(20):
            if gate.requested.is_set() or (stop_file and Path(stop_file).exists()):raise InterruptedError('STOP during preparation')
            obs,_,done,info=env.step(env.raw.action_space.noop())
            if done or info['workshop']['error'] or not info['workshop']['server_available']:raise ValueError('invalid initial world')
        WoodReward(info['workshop']['server_inventory'])
        save_png(obs['pov'],output/'prepared.png')
        evidence=observation_evidence(obs);server=jsonable(info['workshop'])
        env.release();world=env.save_and_quit();archive=output/'initial-world.zip'
        actual_seed=verify_saved_seed((world/'level.dat').read_bytes(),seed)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for path in sorted(world.rglob('*')):
                if path.is_symlink():raise ValueError('world symlink')
                if path.is_file() and path.name!='session.lock':
                    entry=zipfile.ZipInfo('./saves/'+world.name+'/'+path.relative_to(world).as_posix());entry.compress_type=zipfile.ZIP_DEFLATED
                    with path.open('rb') as src,z.open(entry,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,1024*1024)
        validate_archive(archive)
        write_json(archive.with_suffix('.json'),dict(seed=seed,sha256=digest(archive),observation=evidence,server=server,
            saved_seed=actual_seed,world_directory=world.name,
            neutral_preparation_steps=20,scope='saved initial world; restoration has its own neutral startup ticks'))
        report.update(status='world_prepared',execution_valid=True,release='neutral_command_sent',archive=str(archive))
    finally:
        env.close();write_json(output/'summary.json',report)
    return report
