"""Plot saved canonical operator runs without loading models or datasets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chem_operator._experiments.comparison import load_run, read_reconstructions
from chem_operator.plotting import plot_operator_runs


def plot_pfr_profiles(run_path, output_dir, *, cases=2):
    """Plot the PFR flow rates and gas temperature in a two-panel figure."""
    import matplotlib.pyplot as plt

    run = load_run(run_path)
    data = read_reconstructions(run.path)
    required = {"case_ids", "pfr_reference", "pfr_prediction", "pfr_labels", "z"}
    missing = required.difference(data)
    if missing:
        raise ValueError(
            "PFR reconstruction data is missing: " + ", ".join(sorted(missing))
        )

    labels = tuple(str(label) for label in data["pfr_labels"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    expected = ("F_A", "F_B", "F_C", "T_gas")
    missing_labels = [label for label in expected if label not in label_to_index]
    if missing_labels:
        raise ValueError(
            "PFR reconstruction channels are missing: "
            + ", ".join(missing_labels)
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    count = min(cases, len(data["case_ids"]))
    colors = {"F_A": "tab:blue", "F_B": "tab:orange", "F_C": "tab:green"}
    for case_index in range(count):
        z = data["z"][case_index]
        reference = data["pfr_reference"][case_index]
        prediction = data["pfr_prediction"][case_index]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        for label in expected[:3]:
            channel = label_to_index[label]
            axes[0].plot(
                z,
                reference[channel],
                color=colors[label],
                label=f"{label} reference",
            )
            axes[0].plot(
                z,
                prediction[channel],
                color=colors[label],
                linestyle="--",
                label=f"{label} prediction",
            )
        axes[0].set(
            xlabel="Axial position [m]",
            ylabel="Molar flow rate [mol/s]",
            title="Species flow rates",
        )

        temperature = label_to_index["T_gas"]
        axes[1].plot(z, reference[temperature], label="Reference")
        axes[1].plot(
            z,
            prediction[temperature],
            linestyle="--",
            label="Prediction",
        )
        axes[1].set(
            xlabel="Axial position [m]",
            ylabel="Temperature [K]",
            title="Gas temperature",
        )

        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        path = output / f"case_{case_index:02d}_pfr_profiles.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved.append(path)
    return tuple(saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Canonical run directory; repeat to compare compatible models.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "comparison")
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()

    paths = list(plot_operator_runs(args.run, args.output_dir, cases=args.cases))
    for run_path in args.run:
        run = load_run(run_path)
        destination = (
            args.output_dir
            if len(args.run) == 1
            else args.output_dir / run.manifest["model_id"] / run.path.name
        )
        paths.extend(
            plot_pfr_profiles(run.path, destination, cases=args.cases)
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
