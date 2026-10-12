"""A batch may consume its own completion latch, never a new user STOP."""
from pathlib import Path


def publish_completion(stop,message):
    try:
        with Path(stop).open('x',encoding='utf-8') as stream:stream.write(message)
    except FileExistsError:pass  # A late user STOP takes precedence over success.


def claim_automatic_stop(stop,claimed):
    stop=Path(stop);claimed=Path(claimed)
    if claimed.exists():raise FileExistsError(claimed)
    stop.rename(claimed)
    automatic=claimed.read_text(encoding='utf-8-sig').startswith(('RL collection ended','RL training ended'))
    if not automatic or stop.exists():
        # Never replace a stop that arrived concurrently with the claim.
        try:
            with stop.open('x',encoding='utf-8') as stream:stream.write(claimed.read_text(encoding='utf-8-sig'))
        except FileExistsError:pass
        raise InterruptedError('manual/concurrent STOP; refusing automatic re-arm')
