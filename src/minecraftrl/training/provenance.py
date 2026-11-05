"""Hash the local implementation separately from rollout and model identities."""
import hashlib
from pathlib import Path


def source_manifest(root):
    """Include nested package code and training entrypoints, never results or weights.

    This records the working source tree, including uncommitted changes. The caller
    captures it before training and checks it again before writing a candidate.
    Environment versions and rollout source snapshots remain separate evidence.
    """
    root = Path(root).resolve()
    package = root / 'src/minecraftrl'
    if not package.is_dir():
        raise ValueError('project source tree required for training provenance')
    paths = [p for p in package.rglob('*')
             if p.is_file() and p.suffix in ('.py', '.java') and '__pycache__' not in p.parts]
    paths.extend((root / 'scripts').glob('train_*.py'))
    if (root / 'pyproject.toml').is_file():
        paths.append(root / 'pyproject.toml')
    files = []
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError('training source outside project')
        files.append({'path': path.relative_to(root).as_posix(),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'schema': 'training_source_v1', 'files': files}


def verify_sources(root, expected):
    """Refuse to label a candidate with sources that changed during its update."""
    if source_manifest(root) != expected:
        raise ValueError('training source changed during update; candidate not written')
