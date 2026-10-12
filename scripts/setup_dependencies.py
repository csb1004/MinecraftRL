"""Build candidate dependencies in an already-isolated conda prefix.

Never run with a system interpreter. Original source checkouts remain untouched.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from minerl_build_cache import patch_setup


def main():
    p = argparse.ArgumentParser(); p.add_argument('--policy', choices=['vpt','steve1'], required=True)
    args = p.parse_args(); root = Path(__file__).resolve().parents[1]
    expected = (root/'.runtime'/args.policy).resolve()
    if Path(sys.prefix).resolve() != expected: raise SystemExit('Refusing installation outside the project-local environment')
    lock = json.loads((root/'configs/sources.lock.json').read_text())['sources']
    needed = [args.policy,'gym','minerl'] + (['mineclip','minedojo'] if args.policy == 'steve1' else [])
    for name in needed:
        repo = root/'external'/name
        actual = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if actual != lock[name]['commit']: raise RuntimeError('Source commit mismatch: '+name)
        subprocess.run(['git','-C',str(repo),'diff','--exit-code','HEAD','--'],check=True)
    constraints = str(root/'configs/pip-constraints.txt')
    def pip(*values): subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-c',constraints,*values],check=True,cwd=root)
    def filtered_requirements(source, exclude):
        return [line.strip() for line in source.read_text().splitlines() if line.strip() and not line.startswith('#')
                and not any(line.lower().startswith(name) for name in exclude)]
    pip('--no-deps','--no-build-isolation',str(root/'external/gym'))
    pip(*filtered_requirements(root/'external/minerl/requirements.txt',['gym']), 'gym3', 'attrs', 'cloudpickle')
    # Build intermediates on Linux's filesystem: drvfs single-byte JAR writes
    # were measured taking minutes per artifact. Installed prefixes stay on D:.
    scratch = Path(tempfile.mkdtemp(prefix='minecraftrl-build-'+args.policy+'-'))
    build = scratch/'minerl'
    (root/'results'/('build-location-'+args.policy+'.json')).write_text(json.dumps({
        'temporary_build':str(build), 'installed_prefix':str(expected),
        'project_root':str(root)}, indent=2)+'\n')
    shutil.copytree(root/'external/minerl',build,ignore=shutil.ignore_patterns('.git','__pycache__'))
    patches = []
    edits = {
        build/'setup.py': ("os.path.join(os.path.expanduser('~'), '.gradle', 'caches', 'forge_gradle', 'assets')",
                          "os.path.join(os.environ['GRADLE_USER_HOME'], 'caches', 'forge_gradle', 'assets')"),
        build/'scripts/setup_mcp.sh': ('git checkout 1.16.5-20210115','git checkout '+lock['mcp']['commit'])}
    for path, (old,new) in edits.items():
        before = path.read_text()
        if before.count(old) != 1: raise RuntimeError('Upstream build assumption changed: '+str(path))
        after = before.replace(old,new)
        if path.name == 'setup.py':
            after = patch_setup(after)
        if path.name == 'setup_mcp.sh':
            if not after.startswith('#!/bin/bash\n'): raise RuntimeError('Unexpected MCP script header')
            after = after.replace('#!/bin/bash\n', '#!/bin/bash\nset -euo pipefail\n', 1)
        path.write_text(after)
        patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(path.relative_to(build)),tofile=str(path.relative_to(build))))
    (root/'results'/('build-'+args.policy+'.patch')).write_text(''.join(patches))
    # The upstream build invokes Java/Gradle; process-local JAVA_HOME only.
    os.environ['JAVA_HOME'] = str(Path(sys.prefix))
    os.environ.setdefault('GRADLE_USER_HOME', str(root/'.cache/gradle'))
    pip('--no-deps','--no-build-isolation',str(build))
    if args.policy == 'steve1':
        # Upstream explicitly notes gym version conflicts. Use MineRL's pinned fork.
        pip(*filtered_requirements(root/'external/minedojo/requirements.txt',['gym']),
            *filtered_requirements(root/'external/mineclip/requirements.txt',['gym','torch','minedojo']),
            'gdown==5.2.0','accelerate==0.18.0','wandb','cmake','lit')
        pip('--no-deps','--no-build-isolation',str(root/'external/minedojo'),str(root/'external/mineclip'))
        pip('--no-deps','--no-build-isolation','-e',str(root/'external/steve1'))
    pip('--no-deps','--no-build-isolation','-e',str(root))
    check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (root/'results'/('pip-check-'+args.policy+'.txt')).write_text(check.stdout+check.stderr)
    lines = [x for x in check.stdout.splitlines() if x.strip()]
    known = args.policy == 'steve1' and lines and all(
        x.lower().startswith(('mineclip ', 'minedojo ')) and 'gym==0.21.0' in x for x in lines)
    if check.returncode and not known: raise RuntimeError('Unexpected dependency conflicts; see pip-check report')
    receipt = {'policy':args.policy,'candidate_only':True,'python':sys.version,
               'gradle_user_home':os.environ['GRADLE_USER_HOME'],
               'temporary_build':str(build),
               'minecraft_jar_sha256':json.loads((build/'.minecraftrl-mcp-built.json').read_text())['sha256'],
               'upstream_commits':{n:lock[n]['commit'] for n in needed},
               'mcp_commit':lock['mcp']['commit'],'build_patch_sha256':hashlib.sha256(''.join(patches).encode()).hexdigest(),
               'known_gym_metadata_conflict':bool(known),'gameplay_validated':False}
    (root/'results'/('setup-'+args.policy+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print('Candidate packages installed. Metadata conflict exceptions are recorded; gameplay is not validated.')

if __name__ == '__main__': main()
