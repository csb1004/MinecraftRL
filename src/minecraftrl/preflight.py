"""Read-only readiness check; never installs packages or downloads weights."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from .runtime import write_json


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'exit_code': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc: return {'error': str(exc)}


def inspect(root, policy):
    root = Path(root).resolve(); problems = []
    result = {'host': platform.platform(), 'python': sys.version, 'executable': sys.executable,
              'project_root': str(root), 'policy': policy, 'scope': 'this executor only; not Windows PC unless run there',
              'packages': {}, 'sources': {}, 'artifacts': {}}
    expected = (3, 9) if policy == 'vpt' else (3, 10)
    if sys.version_info[:2] != expected: problems.append('candidate baseline requires Python %s.%s' % expected)
    if platform.system() != 'Linux': problems.append('run policy under Linux/WSL')
    for name in ['torch', 'numpy', 'gym', 'gym3', 'minerl', 'Pillow', 'opencv-python', 'mineclip']:
        try: result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
            if name != 'mineclip' or policy == 'steve1': problems.append('package unavailable: ' + name)
    expected_torch = '1.9.0' if policy == 'vpt' else '2.0.1'
    if result['packages'].get('torch') and result['packages']['torch'].split('+')[0] != expected_torch:
        problems.append('torch differs from candidate baseline: ' + expected_torch)
    result['gpu'] = command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader'])
    if result['gpu'].get('exit_code') != 0: problems.append('nvidia-smi unavailable or failed')
    if importlib.util.find_spec('torch'):
        try:
            import torch
            result['cuda_available'] = torch.cuda.is_available()
            result['torch_cuda'] = torch.version.cuda
            result['torch_arch_list'] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
            if not result['cuda_available']: problems.append('torch CUDA unavailable')
            else: result['gpu_capability'] = torch.cuda.get_device_capability(0)
        except Exception as exc: problems.append('torch import/CUDA check failed: ' + repr(exc))
    result['java'] = command(['java', '-version'])
    if result['java'].get('exit_code') != 0: problems.append('Java unavailable')
    elif '"1.8.' not in result['java']['stderr'] + result['java']['stdout']: problems.append('candidate MineRL baseline requires JDK 8')
    result['display_configured'] = bool(os.environ.get('DISPLAY'))
    result['xvfb_available'] = bool(shutil.which('xvfb-run'))
    if not result['display_configured']: problems.append('DISPLAY unset; launch under WSLg or xvfb-run')
    sources = json.loads((root/'configs/sources.lock.json').read_text())['sources']
    required_sources = [policy, 'minerl', 'gym'] + (['mineclip', 'minedojo'] if policy == 'steve1' else [])
    for name in required_sources:
        path = root/'external'/name
        if not path.is_dir(): problems.append('source missing: ' + name); continue
        actual = command(['git', '-C', str(path), 'rev-parse', 'HEAD'])
        dirty = command(['git', '-C', str(path), 'diff', '--exit-code', 'HEAD', '--'])
        result['sources'][name] = {'expected': sources[name]['commit'], 'actual': actual, 'tracked_clean': dirty.get('exit_code') == 0}
        if actual.get('stdout') != sources[name]['commit']: problems.append('source revision mismatch: ' + name)
        if dirty.get('exit_code') != 0: problems.append('tracked source modified: ' + name)
    for spec in json.loads((root/'configs/models.json').read_text())['artifacts']:
        if spec['policy'] != policy: continue
        path = root/spec['file']; receipt = path.with_suffix(path.suffix+'.receipt.json')
        if not path.is_file() or not receipt.is_file(): problems.append('artifact/receipt missing: '+spec['file']); continue
        try:
            recorded = json.loads(receipt.read_text()); actual = sha256(path)
            result['artifacts'][spec['file']] = {'sha256': actual, 'bytes': path.stat().st_size}
            if actual != recorded['sha256']: problems.append('artifact hash mismatch: '+spec['file'])
            if recorded.get('url') != spec.get('url') or recorded.get('gdrive_id') != spec.get('gdrive_id'):
                problems.append('artifact source mismatch: '+spec['file'])
        except Exception as exc: problems.append('artifact check failed: '+repr(exc))
    result['problems'] = problems; result['status'] = 'blocked' if problems else 'ready_for_execution_attempt'
    result['gameplay_validated'] = False
    return result
