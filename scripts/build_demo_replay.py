"""Build the offline replay page for one demo run folder (spec 5.11).

Usage: python scripts/build_demo_replay.py RUN_DIR [--out OUT_DIR]
Writes OUT_DIR/replay.html (default RUN_DIR/replay) with the data inline and the run's frames linked next to it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from minecraftrl.pipeline.viewer import build_replay  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not (args.run_dir / 'timeline.jsonl').exists(): parser.error('not a demo run folder (no timeline.jsonl): %s' % args.run_dir)
    print(build_replay(args.run_dir, args.out or args.run_dir / 'replay'))


if __name__ == '__main__':
    main()
