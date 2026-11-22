"""Run a script under the carbon tracker (tools/carbon.py), without editing it.

    python src/tools/run_tracked.py [--name NAME] [--out CSV] SCRIPT.py [ARGS...]

e.g.  python src/tools/run_tracked.py --name train src/train.py
      python src/tools/run_tracked.py --name benchmark benchmark/compare.py runs/model.pt

The script runs as if launched by `python SCRIPT.py ARGS...`: same sys.argv, its folder first on sys.path,
__name__ == "__main__". The row is written even if the script fails, and its exit code is kept.
"""
import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for `tools`
from tools.carbon import DEFAULT_CSV, track  # noqa: E402


def run_tracked(script, args=(), name=None, out=DEFAULT_CSV, **track_options):
    script = Path(script)
    saved_argv, saved_path = sys.argv, list(sys.path)
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(script.resolve().parent))
    try:
        with track(name or script.stem, out, **track_options):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", help="row name in the CSV (default: the script name)")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--mode", choices=["process", "machine"], default="process")
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    a = parser.parse_args(argv)
    run_tracked(a.script, a.args, a.name, a.out, tracking_mode=a.mode)


if __name__ == "__main__":
    main()
