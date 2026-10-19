"""Full saved-action audit, without gameplay or rearming STOP."""
import argparse
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    import torch
    torch.set_num_threads(1)
    from audit_fresh_target_handoff import audit
    audit(a.attempt/'run',root/'results/local-wall-diagnosis-20261017/assets/results/onset-coverage-training-20261009/checkpoint.pt')


if __name__=='__main__':main()
