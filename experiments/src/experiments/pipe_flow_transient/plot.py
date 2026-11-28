"""Plot saved canonical operator runs without loading models or datasets."""
import argparse
from experiments.paths import example_paths
from chem_operator.plotting import add_plot_run_arguments, select_plot_runs, plot_selected_runs, print_plot_paths, use_publication_style

PATHS = example_paths(__file__)
PROBLEM_ID = "pipe_flow_transient"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_run_arguments(parser, root=PATHS.root, problem_id=PROBLEM_ID,
                           cases=3)
    args = parser.parse_args()
    use_publication_style(use_tex=args.use_tex)
    runs = select_plot_runs(args, parser, problem_id=PROBLEM_ID)
    print_plot_paths(plot_selected_runs(runs, args.output_dir, cases=args.cases))


if __name__ == "__main__":
    main()
