"""Explicit, content-addressed release bundles and non-overwriting restoration.

Only standard-library file operations live here. GitHub credentials and network
operations belong to the PowerShell entry points. Checkpoints are never loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import uuid
import zipfile

LIMIT = 2 * 1024**3
TOKEN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
SHA = re.compile(r'^[0-9a-f]{64}$')
SECRET = re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b')
TEXT = {'.py', '.ps1', '.sh', '.json', '.jsonl', '.md', '.txt', '.log', '.yml', '.yaml', '.toml', '.patch'}


def digest(path):
    with Path(path).open('rb') as stream:
        return stream_digest(stream)[0]


def stream_digest(stream):
    value = hashlib.sha256(); size = 0
    for block in iter(lambda: stream.read(4 * 1024**2), b''):
        value.update(block); size += len(block)
    return value.hexdigest(), size


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or any(ord(c) < 32 for c in value):
        raise ValueError('unsafe_relative_path')
    parts = value.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if any(p in {'', '.', '..'} or p.lower() == '.git' or p.rstrip(' .') != p or p.split('.')[0].upper() in reserved
           or any(c in p for c in '<>"|?*') or p.lower().startswith('.env') for p in parts):
        raise ValueError('unsafe_relative_path: ' + value)
    return PurePosixPath(value)


def confined(root, relative):
    parts = relative_path(relative).parts
    root = Path(root).resolve(); path = root
    for part in parts:
        path = path / part
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_reparse_tag', 0):
                raise ValueError('link_in_path: ' + relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError('path_escape')
    return path


def inspect_public(path, destination):
    relative_path(destination)
    parts = PurePosixPath(destination).parts
    if parts[0] in {'data', 'logs'} or any(p in {'.runtime', '.cache', 'external', '.venv'} for p in parts):
        raise ValueError('excluded_public_path: ' + destination)
    if path.suffix.lower() in {'.jar', '.whl', '.exe', '.dll', '.pem', '.key', '.bundle'}:
        raise ValueError('excluded_public_binary: ' + destination)
    if path.suffix.lower() == '.zip':
        # World archives may contain NBT/chunks, never executable distribution.
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                name = member.filename.rstrip('/')
                # Historical start-world ZIPs use a harmless leading './'. This
                # is inspection only: preserve the original world ZIP bytes.
                while name.startswith('./'): name = name[2:]
                if name == '.' and member.is_dir(): continue
                if name: relative_path(name)
                if Path(name).suffix.lower() not in {'', '.dat', '.dat_old', '.mca', '.mcc', '.json', '.lock'}:
                    raise ValueError('unexpected_world_member: ' + destination)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('world_symlink')
    if path.suffix.lower() in TEXT:
        with path.open('rb') as stream:
            tail = b''
            for block in iter(lambda: stream.read(1024**2), b''):
                data = tail + block
                if SECRET.search(data): raise ValueError('potential_credential: ' + destination)
                tail = data[-512:]


def validate_manifest(value):
    if value.get('schema') != 1 or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value.get('repository', '')):
        raise ValueError('invalid_manifest')
    if not TOKEN.fullmatch(value.get('tag', '')) or not re.fullmatch(r'[0-9a-f]{40}', value.get('code_commit', '')):
        raise ValueError('invalid_release_identity')
    seen = set(); ids = set()
    for group in value['groups']:
        if not TOKEN.fullmatch(group['id']) or group['id'] in ids: raise ValueError('duplicate_group')
        ids.add(group['id'])
        if not TOKEN.fullmatch(group['release_tag']) or not TOKEN.fullmatch(group['asset']) or not group['asset'].endswith('.zip'):
            raise ValueError('invalid_asset')
        if not SHA.fullmatch(group['sha256']) or type(group['bytes']) is not int or not 0 < group['bytes'] < LIMIT:
            raise ValueError('invalid_asset_digest_or_size')
        for row in group['files']:
            relative_path(row['path'])
            key = row['path'].casefold()
            if key in seen: raise ValueError('duplicate_destination')
            seen.add(key)
            if not SHA.fullmatch(row['sha256']) or type(row['bytes']) is not int or not 0 <= row['bytes'] < LIMIT:
                raise ValueError('invalid_file_digest_or_size')
            if PurePosixPath(row['path']).parts[0] not in {'results', 'reproduction', 'weights'}:
                raise ValueError('destination_outside_artifact_roots')
    for path in seen:
        parts = path.split('/')
        if any('/'.join(parts[:end]) in seen for end in range(1, len(parts))):
            raise ValueError('file_directory_collision')
    for experiment in value['experiments'].values():
        if not set(experiment['groups']).issubset(ids): raise ValueError('unknown_group')
        evidence = {row['path']: row['sha256'] for group in value['groups'] if group['id'] in experiment['groups'] for row in group['files']}
        for path, sha in experiment.get('frozen_assets', {}).items():
            relative_path(path)
            if not SHA.fullmatch(sha): raise ValueError('invalid_frozen_asset_digest')
            if path != 'docs/research-spec.md' and evidence.get(path) != sha:
                raise ValueError('experiment_asset_mismatch: ' + path)
    return value


def prepare(root, spec, output, commit, previous=None):
    root = Path(root).resolve(); output = Path(output)
    if not TOKEN.fullmatch(spec['tag']) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', spec['repository']):
        raise ValueError('invalid_release_identity')
    if any(not TOKEN.fullmatch(g['id']) for g in spec['groups']): raise ValueError('invalid_group_id')
    if len({g['id'] for g in spec['groups']}) != len(spec['groups']): raise ValueError('duplicate_group')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'artifact-manifest.json').exists(): raise FileExistsError('manifest_already_exists')
    old = validate_manifest(previous) if previous else None
    if old and old['repository'] != spec['repository']: raise ValueError('previous_repository_mismatch')
    if old and old['tag'] == spec['tag']: raise ValueError('new_release_tag_required')
    groups = []
    for group in spec['groups']:
        sources = {}
        for item in group['sources']:
            base = confined(root, item['path'])
            if not base.exists(): raise FileNotFoundError(base)
            candidates = sorted(base.rglob('*')) if base.is_dir() else [base]
            for path in candidates:
                if path.is_dir():
                    confined(root, path.relative_to(root).as_posix())
                    continue
                if path.name in group.get('exclude_names', []) or '__pycache__' in path.parts or path.suffix == '.pyc': continue
                confined(root, path.relative_to(root).as_posix())
                dest = item.get('destination', item['path'])
                if base.is_dir(): dest += '/' + path.relative_to(base).as_posix()
                if dest in sources: raise ValueError('duplicate_source_destination')
                inspect_public(path, dest)
                sources[dest] = path
        rows = [dict(path=dest, bytes=path.stat().st_size, sha256=digest(path)) for dest, path in sorted(sources.items())]
        identity = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        asset = group['id'] + '-' + identity[:20] + '.zip'
        prior = next((g for g in old['groups'] if g['id'] == group['id'] and g['files'] == rows), None) if old else None
        if prior:
            groups.append(prior); print('REUSE ' + group['id'], flush=True); continue
        archive = output/asset
        if not archive.exists():
            with zipfile.ZipFile(archive, 'x', allowZip64=True) as z:
                for row in rows:
                    info = zipfile.ZipInfo(row['path'], date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED if sources[row['path']].suffix in {'.pt', '.png', '.zip'} else zipfile.ZIP_DEFLATED
                    info._compresslevel = 6
                    with sources[row['path']].open('rb') as src, z.open(info, 'w', force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 4*1024**2)
        entry = dict(id=group['id'], asset=asset, release_tag=spec['tag'], bytes=archive.stat().st_size,
                     sha256=digest(archive), files=rows)
        if entry['bytes'] >= LIMIT: raise ValueError('split_bundle_below_2GiB: ' + group['id'])
        verify_bundle(archive, entry)
        groups.append(entry); print('VERIFIED BUNDLE ' + group['id'], flush=True)
    manifest = dict(schema=1, repository=spec['repository'], tag=spec['tag'], code_commit=commit,
                    groups=groups, experiments=spec['experiments'], base_models=spec.get('base_models', []),
                    runtime=spec.get('runtime', {}), scope=spec.get('scope', ''))
    validate_manifest(manifest)
    write_json(output/'artifact-manifest.json', manifest)
    return manifest


def selected_groups(manifest, experiment):
    validate_manifest(manifest)
    if experiment == 'all': return manifest['groups']
    required = manifest['experiments'][experiment]['groups']
    return [g for g in manifest['groups'] if g['id'] in required]


def plan(root, manifest, experiment):
    needed = []; matched = 0
    for group in selected_groups(manifest, experiment):
        missing = False
        for row in group['files']:
            path = confined(root, row['path'])
            if path.exists():
                if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                    raise ValueError('existing_file_conflict: ' + row['path'])
                matched += 1
            else: missing = True
        if missing: needed.append(group)
    return dict(groups=needed, existing_files_verified=matched)


def verify_bundle(path, group):
    path = Path(path)
    if path.stat().st_size != group['bytes'] or digest(path) != group['sha256']:
        raise ValueError('bundle_hash_mismatch: ' + group['asset'])
    with zipfile.ZipFile(path) as z:
        expected = {row['path']: row for row in group['files']}
        if len(z.infolist()) != len(expected) or set(z.namelist()) != set(expected):
            raise ValueError('unexpected_bundle_members')
        for info in z.infolist():
            relative_path(info.filename)
            row = expected[info.filename]
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.file_size != row['bytes']:
                raise ValueError('invalid_bundle_member')
            with z.open(info) as stream:
                actual, size = stream_digest(stream)
            if actual != row['sha256'] or size != row['bytes']: raise ValueError('member_hash_mismatch')


def restore(root, manifest, experiment, cache):
    work = plan(root, manifest, experiment)
    # Check all bundles and all existing destinations before creating any files.
    for group in work['groups']: verify_bundle(Path(cache)/group['asset'], group)
    created = 0
    for group in work['groups']:
        with zipfile.ZipFile(Path(cache)/group['asset']) as z:
            for row in group['files']:
                dest = confined(root, row['path'])
                if dest.exists(): continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest = confined(root, row['path'])
                temporary = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.partial')
                try:
                    with z.open(row['path']) as src, temporary.open('xb') as dst:
                        shutil.copyfileobj(src, dst, 4*1024**2)
                    if digest(temporary) != row['sha256']: raise ValueError('copy_hash_mismatch')
                    # Atomic no-clobber installation (same-directory hard link).
                    os.link(temporary, dest)
                    created += 1
                finally:
                    if temporary.exists(): temporary.unlink()
    final = plan(root, manifest, experiment)
    if final['groups']: raise RuntimeError('incomplete_restore')
    return dict(completed=True, files_created=created, files_verified=final['existing_files_verified'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['prepare', 'plan', 'restore'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--manifest', type=Path); p.add_argument('--spec', type=Path)
    p.add_argument('--output', type=Path); p.add_argument('--cache', type=Path)
    p.add_argument('--experiment', default='all'); p.add_argument('--previous', type=Path)
    args = p.parse_args()
    if args.operation == 'prepare':
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.root, text=True).strip()
        result = prepare(args.root, read_json(args.spec), args.output, commit, read_json(args.previous) if args.previous else None)
        print(json.dumps(dict(manifest=str(args.output/'artifact-manifest.json'), groups=len(result['groups']))))
    elif args.operation == 'plan': print(json.dumps(plan(args.root, read_json(args.manifest), args.experiment)))
    else: print(json.dumps(restore(args.root, read_json(args.manifest), args.experiment, args.cache)))


if __name__ == '__main__': main()
