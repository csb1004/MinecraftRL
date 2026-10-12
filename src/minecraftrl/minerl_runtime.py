"""Stage only the immutable game JAR off drvfs; worlds and results stay in place."""
import hashlib
import os
from pathlib import Path
import tempfile
import uuid
from .runtime import write_json


def launch_scoped_watcher(output, parent_pid, child_pid, *temp_dirs):
    """Keep upstream watchdog behavior but isolate PID files by episode.

    PID pairs can repeat after a WSL restart. Upstream daemoniker refuses a
    leftover file even when its old owner has gone; never delete that evidence.
    """
    import subprocess
    import sys
    scope = Path(output).resolve()/'watcher-runtime'
    scope.mkdir(exist_ok=True)
    env = os.environ.copy()
    project = Path(__file__).resolve().parents[2]
    env['PYTHONPATH'] = str(project/'src')+os.pathsep+env.get('PYTHONPATH','')
    write_json(Path(output)/'watcher-runtime.json',dict(cwd=str(scope),parent_pid=parent_pid,child_pid=child_pid,
        scope='Upstream watchdog, per-episode PID/log namespace; ownership cleanup unchanged'))
    subprocess.check_call([sys.executable,'-m','minerl.utils.process_watcher',str(parent_pid),str(child_pid),
                           '--child-dirs',*temp_dirs],cwd=str(scope),env=env)


def configure_player_identity(minecraft_dir,slot):
    """The upstream CLI otherwise derives a new offline UUID from Player<TIME>."""
    launcher=Path(minecraft_dir)/'launchClient.sh'; source=launcher.read_text()
    original='java -Xmx$maxMem -jar "$fatjar" --envPort=$port'
    replacement='''playerArgs=()
if [ -n "${MINECRAFTRL_PLAYER_UUID:-}" ]; then
    playerArgs=(--username MinecraftRL --uuid "$MINECRAFTRL_PLAYER_UUID")
fi
java -Xmx$maxMem -jar "$fatjar" --envPort=$port "${playerArgs[@]}"'''
    if source.count(replacement)==1: pass
    elif source.count(original)==1: launcher.write_text(source.replace(original,replacement))
    else: raise ValueError('unexpected_player_identity_launcher')
    player_uuid=uuid.uuid5(uuid.NAMESPACE_URL,'MinecraftRL:'+slot).hex
    os.environ['MINECRAFTRL_PLAYER_UUID']=player_uuid
    return player_uuid


def prepare_jar_cache(minecraft_dir, output,source_override=None):
    minecraft_dir = Path(minecraft_dir).resolve()
    source = Path(source_override) if source_override else minecraft_dir/'build/libs/mcprec-6.13.jar'
    scratch = Path(tempfile.mkdtemp(prefix='minecraftrl-runtime-'))
    cached = scratch/source.name
    digest = hashlib.sha256()
    with source.open('rb') as src, cached.open('xb') as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block: break
            digest.update(block); dst.write(block)
    actual = hashlib.sha256(cached.read_bytes()).hexdigest()
    if actual != digest.hexdigest(): raise RuntimeError('Runtime JAR copy hash mismatch')
    launcher = minecraft_dir/'launchClient.sh'
    before = launcher.read_text()
    old = 'fatjar=build/libs/mcprec-6.13.jar'
    new = 'fatjar="${MINECRAFTRL_RUNTIME_JAR:-build/libs/mcprec-6.13.jar}"'
    if before.count(old) == 1:
        after = before.replace(old, new).replace('-jar $fatjar ', '-jar "$fatjar" ')
        launcher.write_text(after)
    elif before.count(new) == 1:
        after = before
    else:
        raise RuntimeError('Unexpected MineRL launch script; refusing to patch')
    os.environ['MINECRAFTRL_RUNTIME_JAR'] = str(cached)
    write_json(Path(output)/'runtime-jar-cache.json', {
        'source':str(source), 'cached':str(cached), 'sha256':actual,
        'bytes':cached.stat().st_size, 'minecraft_working_directory':str(minecraft_dir),
        'launcher_sha256':hashlib.sha256(after.encode()).hexdigest(),
        'scope':'identical JAR read cache only; installed package and game working directory unchanged'})
    return cached
