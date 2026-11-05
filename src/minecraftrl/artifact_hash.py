"""Full SHA256 reads, using native Windows I/O for large WSL project files."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

MIN_NATIVE_BYTES=16*1024*1024


def windows_drive_path(path):
    parts=PurePosixPath(str(path)).parts
    if (len(parts)<4 or parts[:2]!=('/','mnt') or not re.fullmatch('[a-zA-Z]',parts[2])
            or any(p in ('.','..') or '\\' in p or ':' in p for p in parts[3:])):
        raise ValueError('not a standard WSL drive path')
    return parts[2].upper()+':\\'+'\\'.join(parts[3:])


def native_bridge(path):
    root=Path(__file__).resolve().parents[2]
    if (sys.platform!='linux' or not os.environ.get('WSL_DISTRO_NAME')
            or not path.is_relative_to(root) or path.stat().st_size<MIN_NATIVE_BYTES):return None
    executable=shutil.which('powershell.exe')
    if executable is None:return None
    script=root/'scripts/Measure-ArtifactHash.ps1'
    try:return executable,windows_drive_path(script.as_posix()),windows_drive_path(path.as_posix())
    except ValueError:return None


def sha256(path):
    path=Path(path).resolve();bridge=native_bridge(path)
    if bridge is None:
        value=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
        return value.hexdigest()
    before=path.stat();exe,script,target=bridge
    result=subprocess.run([exe,'-NoProfile','-NonInteractive','-File',script,'-LiteralPath',target],
        capture_output=True,text=True,timeout=120)
    if result.returncode!=0:raise ValueError('native artifact hash failed')
    data=json.loads(result.stdout.lstrip('\ufeff'))
    after=path.stat()
    if (set(data)!={'schema','sha256','bytes'} or data['schema']!=1 or type(data['bytes']) is not int
            or data['bytes']!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)
            !=(after.st_size,after.st_mtime_ns,after.st_ino)
            or not isinstance(data['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',data['sha256'])):
        raise ValueError('native artifact hash evidence mismatch')
    return data['sha256']
