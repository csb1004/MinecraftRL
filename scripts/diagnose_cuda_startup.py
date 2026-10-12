"""Fresh-process CUDA startup isolation. No game, policy update or global settings."""
import argparse
import faulthandler
import json
import os
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--disable-cudnn', action='store_true')
args = parser.parse_args()
assert '/usr/lib/wsl/lib' in os.environ.get('LD_LIBRARY_PATH','').split(':'), 'Use the policy launcher WSL library environment'
out = args.output.resolve()
assert out.is_relative_to(ROOT/'results') and (ROOT/'results/STOP').exists()
out.mkdir(exist_ok=False)
records = []


def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_s=r.ru_utime,system_s=r.ru_stime,minor_faults=r.ru_minflt,major_faults=r.ru_majflt)


def measure(name, fn, sync=lambda: None):
    before = usage(); started = time.monotonic()
    with (out/(name+'-stack.txt')).open('w') as trace:
        faulthandler.dump_traceback_later(20, repeat=True, file=trace)
        try:
            value = fn(); sync()
        finally:
            faulthandler.cancel_dump_traceback_later()
    after = usage()
    row = dict(stage=name,seconds=time.monotonic()-started,usage_delta={k:after[k]-v for k,v in before.items()})
    records.append(row)
    (out/'progress.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(row),flush=True)
    return value


torch = measure('import_torch', lambda: __import__('torch'))
if args.disable_cudnn:
    torch.backends.cudnn.enabled = False
torch.manual_seed(40321)
device_tensor = measure('cuda_allocate', lambda: torch.randn(2,3,128,128,device='cuda'), torch.cuda.synchronize)
matrix = torch.randn(128,128,device='cuda')
measure('cuda_matmul', lambda: matrix@matrix, torch.cuda.synchronize)
conv = torch.nn.Conv2d(3,128,3,padding=1).cuda().eval()
torch.cuda.synchronize()
(out/'maps-before.txt').write_text(Path('/proc/self/maps').read_text())
with torch.no_grad(), torch.cuda.amp.autocast():
    for i in range(3):
        result = measure('conv_'+str(i), lambda: conv(device_tensor), torch.cuda.synchronize)
        assert torch.isfinite(result).all()
(out/'maps-after.txt').write_text(Path('/proc/self/maps').read_text())
report = dict(completed=True,scope='synthetic convolution only; not policy execution',
    torch_version=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
    cudnn_enabled=torch.backends.cudnn.enabled,cudnn_benchmark=torch.backends.cudnn.benchmark,
    device=torch.cuda.get_device_name(),threads=torch.get_num_threads(),records=records,
    environment={k:os.environ.get(k) for k in ('LD_LIBRARY_PATH','XDG_CACHE_HOME','CUDA_MODULE_LOADING')},
    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
(out/'diagnosis.json').write_text(json.dumps(report,indent=2))
assert (ROOT/'results/STOP').exists()
