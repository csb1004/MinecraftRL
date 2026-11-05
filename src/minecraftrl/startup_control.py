"""Opt-in, audited random-block-tick pause during the pinned 60-step startup."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import zipfile

from .runtime import write_json
from .saved_inventory import read_nbt
from .world_store import digest,validate_archive

ENVIRONMENT_KEY='MINECRAFTRL_STARTUP_RANDOM_TICKS'
SCHEMA='startup_random_tick_control_v1'
REQUEST_SCHEMA='startup_random_tick_request_v1'
RESUME_CLIENT_TICK=80


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _inside(root,path):
    root=Path(root).resolve();path=Path(path)
    path=(root/path).resolve() if not path.is_absolute() else path.resolve()
    if path==root or root not in path.parents:
        raise ValueError('startup artifact must be inside the project')
    return path


def _speed(value):
    if type(value) is not int or not 0<=value<=4096:
        raise ValueError('unsupported randomTickSpeed')
    return value


def verify_world_pair(source,paused,restore_speed):
    """Check every archive member; only one NBT game rule may differ."""
    restore_speed=_speed(restore_speed)
    validate_archive(source);validate_archive(paused)
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(paused) as derived:
        if original.namelist()!=derived.namelist():
            raise ValueError('controlled startup archive members changed')
        levels=[p for p in original.namelist() if p.endswith('/level.dat')]
        if len(levels)!=1:raise ValueError('one world level required')
        level=levels[0]
        for name in original.namelist():
            a=original.read(name);b=derived.read(name)
            if name!=level:
                if a!=b:raise ValueError('controlled startup changed world member: '+name)
                continue
            before=read_nbt(a);after=read_nbt(b)
            if before['Data']['GameRules']['randomTickSpeed']!=str(restore_speed):
                raise ValueError('original randomTickSpeed does not match restoration')
            if after['Data']['GameRules']['randomTickSpeed']!='0':
                raise ValueError('startup world random block ticks are not paused')
            after['Data']['GameRules']['randomTickSpeed']=str(restore_speed)
            if before!=after:raise ValueError('controlled startup changed other NBT state')


def prepare_startup_world(root,source,output):
    """Offline only; nbtlib belongs in the existing isolated fixture-tools environment."""
    import nbtlib as n
    root=Path(root).resolve();source=_inside(root,source);output=_inside(root,output)
    metadata=_read(source.with_suffix('.json'))
    if metadata.get('experimental_split') not in ('train','development'):
        raise ValueError('controlled starts require explicit train/development split')
    if metadata.get('startup_control') is not None:
        raise ValueError('nested startup world preparation is unsupported')
    if metadata.get('sha256')!=digest(source):raise ValueError('original world hash mismatch')
    if ('navigation_goals' in metadata and
            metadata['navigation_goals'].get('source_archive_sha256')!=metadata['sha256']):
        raise ValueError('original navigation goals not bound to source archive')
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError(output)
    validate_archive(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as original:
        names=[p for p in original.namelist() if p.endswith('/level.dat')]
        if len(names)!=1:raise ValueError('one world level required')
        level_name=names[0];data=original.read(level_name)
        level=n.File.parse(io.BytesIO(gzip.decompress(data)))
        if type(metadata.get('seed')) is not int or int(level['Data']['WorldGenSettings']['seed'])!=metadata['seed']:
            raise ValueError('startup source saved seed differs from manifest')
        restore_speed=_speed(int(str(level['Data']['GameRules']['randomTickSpeed'])))
        level['Data']['GameRules']['randomTickSpeed']=n.String('0')
        buffer=io.BytesIO();level.write(buffer)
        replacement=gzip.compress(buffer.getvalue(),mtime=0)
        with zipfile.ZipFile(output,'x') as target:
            for entry in original.infolist():
                # writestr mutates ZipInfo.header_offset; preserve the original reader's objects.
                target.writestr(copy.copy(entry),replacement if entry.filename==level_name else original.read(entry))
    verify_world_pair(source,output,restore_speed)
    if metadata['sha256']!=digest(source):raise ValueError('original world changed during preparation')
    result=copy.deepcopy(metadata)
    result['sha256']=digest(output)
    if 'navigation_goals' in result:
        result['navigation_goals']['source_archive_sha256']=result['sha256']
    result['startup_control']=dict(schema=SCHEMA,source_archive=source.relative_to(root).as_posix(),
        source_archive_sha256=metadata['sha256'],source_manifest_sha256=digest(source.with_suffix('.json')),
        paused_speed=0,restore_speed=restore_speed,resume_client_tick=RESUME_CLIENT_TICK,startup_steps=60)
    result['scope']='Copied start with unchanged terrain/player/inventory; random block ticks paused during startup and explicitly restored before policy control'
    write_json(output.with_suffix('.json'),result)
    return result


def configure_startup_control(root,world,metadata,output,*,enabled=False):
    """Bind one worker to an audited world and request; never inherit ambient activation."""
    os.environ.pop(ENVIRONMENT_KEY,None)
    control=metadata.get('startup_control') if metadata else None
    if not enabled and control is None:return None
    if type(enabled) is not bool or not enabled or not isinstance(control,dict):
        raise ValueError('controlled startup world and explicit flag must be used together')
    root=Path(root).resolve();world=_inside(root,world);output=_inside(root,output)
    if metadata.get('experimental_split') not in ('train','development'):
        raise ValueError('controlled startup does not admit final evaluation')
    if (control.get('schema')!=SCHEMA or type(control.get('paused_speed')) is not int or control['paused_speed']!=0
            or type(control.get('resume_client_tick')) is not int or control['resume_client_tick']!=RESUME_CLIENT_TICK
            or type(control.get('startup_steps')) is not int or control['startup_steps']!=60):
        raise ValueError('unsupported startup control contract')
    restore_speed=_speed(control.get('restore_speed'))
    source=_inside(root,control['source_archive'])
    source_metadata=_read(source.with_suffix('.json'))
    if (digest(source)!=control['source_archive_sha256'] or digest(source.with_suffix('.json'))!=control['source_manifest_sha256']
            or source_metadata.get('sha256')!=control['source_archive_sha256'] or digest(world)!=metadata.get('sha256')):
        raise ValueError('startup control world/manifest hash mismatch')
    if (source_metadata.get('experimental_split')!=metadata['experimental_split'] or
            source_metadata.get('seed')!=metadata.get('seed') or source_metadata.get('startup_control') is not None):
        raise ValueError('startup control source split/seed mismatch or nesting')
    verify_world_pair(source,world,restore_speed)
    request=dict(schema=REQUEST_SCHEMA,paused_speed=0,restore_speed=restore_speed,resume_client_tick=RESUME_CLIENT_TICK,
        startup_steps=60,world_sha256=metadata['sha256'],source_archive_sha256=control['source_archive_sha256'],
        seed=metadata['seed'],split=metadata['experimental_split'])
    path=output/'startup-random-ticks-request.json'
    if path.exists():raise FileExistsError(path)
    write_json(path,request)
    os.environ[ENVIRONMENT_KEY]=str(path)
    return dict(request_path=str(path),request_sha256=digest(path),output=str(output),**request)


def verify_startup_receipt(control,server,*,first_step=False):
    if control is None:return None
    if digest(control['request_path'])!=control['request_sha256']:
        raise ValueError('startup request changed after launch')
    filename='startup-random-ticks-first-step.json' if first_step else 'startup-random-ticks-restored.json'
    path=Path(control['output'])/filename
    try:receipt=_read(path)
    except (OSError,ValueError) as exc:raise ValueError('startup restoration receipt unavailable') from exc
    expected_tick=control['resume_client_tick']+int(first_step)
    if (receipt.get('schema')!='startup_random_tick_receipt_v1' or receipt.get('restored') is not True
            or receipt.get('request_sha256')!=control['request_sha256']):
        raise ValueError('startup restoration receipt is not bound to request')
    required=dict(restore_speed=control['restore_speed'],current_speed=control['restore_speed'],
        client_tick=expected_tick,client_tick_before=expected_tick,extra_simulation_ticks=0,input_commands=0)
    if any(type(receipt.get(k)) is not int or receipt[k]!=value for k,value in required.items()):
        raise ValueError('startup restoration value or boundary mismatch')
    for key in ('client_tick','server_tick','world_time'):
        if (receipt.get(key)!=server.get(key) or receipt.get(key+'_before')!=receipt.get(key)):
            raise ValueError('startup restoration tick evidence mismatch')
    before=receipt.get('random_state_before');after=receipt.get('random_state_after')
    if type(before) is not int or type(after) is not int or not 0<=before<2**48 or before!=after:
        raise ValueError('startup restoration advanced random state')
    return dict(verified=True,file=filename,sha256=digest(path),client_tick=expected_tick,restore_speed=control['restore_speed'])
