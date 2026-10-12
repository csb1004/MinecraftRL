"""Parse the visible XYZ line, never substitute privileged telemetry on failure."""
import math
import re


def parse_f3_xyz(text):
    if not isinstance(text,str) or len(text)>32768:raise ValueError('invalid F3 text')
    lines=[s.strip() for s in text.splitlines() if re.match(r'^\s*XYZ\s*:',s)]
    if len(lines)!=1:raise ValueError('one unambiguous F3 XYZ line required')
    number=r'([+-]?\d+(?:\.\d+)?)'
    match=re.fullmatch(r'XYZ\s*:\s*'+number+r'\s*/\s*'+number+r'\s*/\s*'+number,lines[0])
    if not match:raise ValueError('unreadable F3 XYZ line')
    xyz=[float(x) for x in match.groups()]
    if any(not math.isfinite(v) or abs(v)>30000000 for v in xyz):raise ValueError('invalid F3 coordinates')
    return xyz
