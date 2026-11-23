"""Plot saved canonical operator runs without loading models or datasets."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chem_operator._experiments.comparison import load_run, read_reconstructions
from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import (
    add_plot_run_arguments, member_plot_dirs, select_plot_runs, plot_selected_runs,
    print_plot_paths, use_publication_style,
)

PATHS = ExamplePaths.from_script(__file__)
PROBLEM_ID = "pfr_heat"
PFR_CHANNELS = ("F_A", "F_B", "F_C", "T_gas")
SPECIES_COLORS = {"F_A": "tab:blue", "F_B": "tab:orange", "F_C": "tab:green"}
# Hues kept clear of the RdYlBu temperature colormap drawn behind the lines.
FIELD_SPECIES_COLORS = {"F_A": "black", "F_B": "#7b3294", "F_C": "#1b7837"}


def _load_pfr_heat(run_path, extra_keys=()):
    """Load saved reconstructions and map each PFR channel to its index."""
    run = load_run(run_path)
    data = read_reconstructions(run.path)
    required = {
        "case_ids", "pfr_reference", "pfr_prediction", "pfr_labels", "z",
        *extra_keys,
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(
            "PFR reconstruction data is missing: " + ", ".join(sorted(missing))
        )

    labels = tuple(str(label) for label in data["pfr_labels"])
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing_labels = [
        label for label in PFR_CHANNELS if label not in label_to_index
    ]
    if missing_labels:
        raise ValueError(
            "PFR reconstruction channels are missing: "
            + ", ".join(missing_labels)
        )
    return data, label_to_index


def plot_wall_field_profiles(run_path, output_dir, *, cases=2):
    """Plot a 2x2 reference/prediction figure of wall and gas-side fields.

    The top row is the wall temperature over (z/L, r/R_i). The bottom row has
    the species molar flow rates drawn over a translucent band coloured by the
    gas temperature. All four panels share one reversed RdYlBu colour bar
    (red = hot) with a breakpoint at its yellow midpoint: the blue half is a
    linear scale for the
    wall temperature and the red half a linear scale for the gas temperature.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    data, label_to_index = _load_pfr_heat(
        run_path, extra_keys=("r", "wall_reference", "wall_prediction")
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    count = min(cases, len(data["case_ids"]))
    gas_channel = label_to_index["T_gas"]
    for case_index in range(count):
        z = data["z"][case_index]
        r = data["r"][case_index]
        columns = (
            ("Reference", data["wall_reference"][case_index, 0],
             data["pfr_reference"][case_index]),
            ("Prediction", data["wall_prediction"][case_index, 0],
             data["pfr_prediction"][case_index]),
        )
        walls = np.stack([wall for _, wall, _ in columns])
        gases = np.stack([profiles[gas_channel] for _, _, profiles in columns])
        # Each field gets its own linear range on one half of the colour bar.
        scales = {
            "wall": (walls.min(), max(walls.max() - walls.min(), 1e-12), 0.0),
            "gas": (gases.min(), max(gases.max() - gases.min(), 1e-12), 0.5),
        }

        def to_colour(values, field):
            low, span, offset = scales[field]
            return offset + 0.5 * (np.asarray(values) - low) / span

        norm = Normalize(0.0, 1.0)
        cmap = "RdYlBu_r"  # reversed so that red is hot
        figure, axes = plt.subplots(
            2, 2, figsize=(8.5, 3.7), sharex=True, sharey="row",
            gridspec_kw={"height_ratios": (1.0, 1.4)},
        )

        for column, (title, wall, profiles) in enumerate(columns):
            field_axis, profile_axis = axes[0, column], axes[1, column]
            field_axis.pcolormesh(
                z, r, to_colour(wall, "wall").T, cmap=cmap,
                shading="gouraud", norm=norm,
            )
            field_axis.set_title(title)
            for label in PFR_CHANNELS[:3]:
                profile_axis.plot(
                    z, profiles[label_to_index[label]],
                    color=FIELD_SPECIES_COLORS[label], label=f"${label}$",
                    zorder=2,
                )
            profile_axis.set_xlabel(r"$\xi = z/L$")

        # Lay the gas-temperature band under the lines once the shared
        # y-limits are known, so it fills each lower panel exactly.
        low, high = axes[1, 0].get_ylim()
        for column, (_, _, profiles) in enumerate(columns):
            axes[1, column].imshow(
                to_colour(profiles[gas_channel], "gas")[None, :],
                cmap=cmap, norm=norm,
                alpha=0.3, aspect="auto", interpolation="bilinear",
                extent=(z[0], z[-1], low, high), origin="lower", zorder=0,
            )
        axes[1, 0].set_ylim(low, high)
        axes[1, 0].set_xlim(z[0], z[-1])

        axes[0, 0].set_ylabel("$r/R_i$")
        axes[1, 0].set_ylabel("$F_i$ [mol/s]")
        axes[1, 0].legend(fontsize=8)

        figure.tight_layout()
        top = axes[0, 1].get_position()
        bottom = axes[1, 1].get_position()
        colorbar_axis = figure.add_axes(
            (top.x1 + 0.012, bottom.y0, 0.012, top.y1 - bottom.y0)
        )
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis,
        )
        ticks, tick_labels = [], []
        for field, values in (("wall", walls), ("gas", gases)):
            locator = MaxNLocator(nbins=4)
            for value in locator.tick_values(values.min(), values.max()):
                if values.min() <= value <= values.max():
                    ticks.append(float(to_colour(value, field)))
                    tick_labels.append(f"{value:g}")
        colorbar.set_ticks(ticks, labels=tick_labels)
        colorbar.ax.minorticks_off()
        colorbar.ax.axhline(0.5, color="black", linewidth=0.8)
        for position, text in ((0.25, "wall $T_s$ [K]"),
                               (0.75, "gas $T_{gas}$ [K]")):
            colorbar.ax.annotate(
                text, xy=(1.0, position), xycoords="axes fraction",
                xytext=(28, 0), textcoords="offset points",
                rotation=90, va="center", ha="left",
            )
        path = output / f"case_{case_index:02d}_wall_field_profiles.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)
    return tuple(saved)


def plot_pfr_profiles(run_path, output_dir, *, cases=2):
    """Plot the PFR flow rates and gas temperature in a two-panel figure."""
    import matplotlib.pyplot as plt

    data, label_to_index = _load_pfr_heat(run_path)
    expected = PFR_CHANNELS
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    count = min(cases, len(data["case_ids"]))
    colors = SPECIES_COLORS
    for case_index in range(count):
        z = data["z"][case_index]
        reference = data["pfr_reference"][case_index]
        prediction = data["pfr_prediction"][case_index]
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))

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
            xlabel="Axial position z/L",
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
            xlabel="Axial position z/L",
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
    add_plot_run_arguments(parser, root=PATHS.root, problem_id=PROBLEM_ID,
                           cases=3)
    parser.add_argument(
        "--wall-field-plot", action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot the 2x2 wall temperature / profile figure (default: on).",
    )
    args = parser.parse_args()
    use_publication_style(use_tex=args.use_tex)

    runs = select_plot_runs(args, parser, problem_id=PROBLEM_ID)
    paths = list(plot_selected_runs(runs, args.output_dir, cases=args.cases))
    for run in runs:
        destination = args.output_dir / run.manifest["model_id"] / run.path.name
        for member, member_dir in member_plot_dirs(run, destination):
            paths.extend(
                plot_pfr_profiles(member.path, member_dir, cases=args.cases)
            )
            if args.wall_field_plot:
                paths.extend(plot_wall_field_profiles(
                    member.path, member_dir, cases=args.cases
                ))
    print_plot_paths(paths)


if __name__ == "__main__":
    main()
