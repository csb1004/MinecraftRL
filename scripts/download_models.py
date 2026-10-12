"""Download only inference artifacts named by official, pinned source files."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--policy', choices=['vpt', 'steve1'], required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'configs/models.json').read_text())
    for spec in manifest['artifacts']:
        if spec['policy'] != args.policy: continue
        dest = root / spec['file']; dest.parent.mkdir(parents=True, exist_ok=True)
        receipt = dest.with_suffix(dest.suffix + '.receipt.json')
        if dest.exists():
            if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != digest(dest):
                raise RuntimeError('Existing artifact has no matching receipt: ' + str(dest))
            print('Verified existing:', dest.name); continue
        partial = dest.with_suffix(dest.suffix + '.partial')
        if partial.exists(): raise RuntimeError('Prior partial download exists; inspect before removing: ' + str(partial))
        print('Downloading:', dest.name, flush=True)
        if 'url' in spec:
            with urllib.request.urlopen(spec['url'], timeout=60) as response, partial.open('xb') as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block: break
                    out.write(block)
        else:
            import gdown
            # Public artifacts do not require reading/writing home-directory cookies.
            if not gdown.download(id=spec['gdrive_id'], output=str(partial), quiet=False, use_cookies=False):
                raise RuntimeError('Google Drive download failed: ' + dest.name)
        with partial.open('rb') as stream: head = stream.read(512).lower()
        if partial.stat().st_size < 100 or b'<html' in head or b'<!doctype html' in head:
            raise RuntimeError('Artifact looks like an error page: ' + str(partial))
        if spec.get('official_md5') and digest(partial, 'md5') != spec['official_md5']:
            raise RuntimeError('Official MineCLIP checksum mismatch')
        record = dict(spec, sha256=digest(partial), bytes=partial.stat().st_size,
                      downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      checksum_basis='official MD5 plus local SHA256' if 'official_md5' in spec else 'local SHA256; upstream does not publish a trusted hash here')
        partial.rename(dest); receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print('Recorded SHA256:', record['sha256'])

if __name__ == '__main__': main()
