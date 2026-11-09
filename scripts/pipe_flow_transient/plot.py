"""Plot saved canonical operator runs without loading models or datasets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chem_operator.plotting import add_plot_run_arguments, select_plot_runs, plot_selected_runs, print_plot_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_run_arguments(parser, root=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "comparison")
    parser.add_argument("--cases", type=int, default=1)
    args = parser.parse_args()
    runs = select_plot_runs(args, parser, problem_id="pipe_flow_transient")
    print_plot_paths(plot_selected_runs(runs, args.output_dir, cases=args.cases))


if __name__ == "__main__":
    main()
