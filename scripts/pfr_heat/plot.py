"""Plot saved canonical operator runs without loading models or datasets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chem_operator.plotting import plot_operator_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Canonical run directory; repeat to compare compatible models.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "comparison")
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()
    for path in plot_operator_runs(args.run, args.output_dir, cases=args.cases):
        print(path)


if __name__ == "__main__":
    main()
