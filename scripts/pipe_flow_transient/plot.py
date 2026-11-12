"""Plot saved canonical operator runs without loading models or datasets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import add_plot_run_arguments, select_plot_runs, plot_selected_runs, print_plot_paths

PATHS = ExamplePaths.from_script(__file__)
PROBLEM_ID = "pipe_flow_transient"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_run_arguments(parser, root=PATHS.root, problem_id=PROBLEM_ID,
                           cases=3)
    args = parser.parse_args()
    runs = select_plot_runs(args, parser, problem_id=PROBLEM_ID)
    print_plot_paths(plot_selected_runs(runs, args.output_dir, cases=args.cases))


if __name__ == "__main__":
    main()
