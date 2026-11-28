from __future__ import annotations

import csv
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cantera as ct
import numpy as np

from chem_operator.datasets import (
    CaseParameters,
    CaseSimulator,
    SimulationRecord,
)
from chem_operator.sampling import ParameterSpec


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_Q2D_EXPERIMENT_ROOT = PROJECT_ROOT / "Quasi-2D-packbed-experiement"
DEFAULT_Q2D_SOLVER_ROOT = (
    DEFAULT_Q2D_EXPERIMENT_ROOT / "external" / "Quasi-2D-packbed"
)
DEFAULT_TUTORIAL_ROOT = (
    DEFAULT_Q2D_SOLVER_ROOT / "example" / "Methane_reforming"
)
DEFAULT_TUTORIAL_CASES = {
    False: DEFAULT_TUTORIAL_ROOT / "CMR",
    True: DEFAULT_TUTORIAL_ROOT / "CMRwithAnnularChannel",
}
INPUT_SCHEMA_PATH = Path(__file__).resolve().parent / "q2d_input_schema.tsv"
DEFAULT_SOLVER_TIMEOUT_S = 8 * 60.0

GAS_SPECIES_FALLBACK = ["CH4", "CO", "O2", "CO2", "H2", "H2O", "AR"]
SURFACE_SPECIES_FALLBACK = [
    "H(s)",
    "O(s)",
    "CH4(s)",
    "H2O(s)",
    "CO2(s)",
    "CO(s)",
    "OH(s)",
    "C(s)",
    "HCO(s)",
    "CH(s)",
    "CH3(s)",
    "CH2(s)",
    "COOH(s)",
    "Ni(s)",
]

@dataclass(frozen=True)
class InputField:
    dat_key: str
    param_name: str
    value_type: str
    case_section: str
    case_key: str
    template_section: str
    comment: str

    def coerce(self, value: Any) -> Any:
        if self.value_type == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y", "on"}:
                    return True
                if lowered in {"0", "false", "no", "n", "off"}:
                    return False
            return bool(value)
        if self.value_type == "int":
            return int(float(value))
        if self.value_type == "float":
            return float(value)
        if self.value_type == "str":
            return str(value)
        return value


def load_input_schema(path: str | Path = INPUT_SCHEMA_PATH) -> list[InputField]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            InputField(
                dat_key=row["dat_key"],
                param_name=row["param_name"],
                value_type=row["value_type"],
                case_section=row["case_section"],
                case_key=row["case_key"] or row["dat_key"],
                template_section=row["template_section"],
                comment=row["comment"],
            )
            for row in reader
        ]


INPUT_SCHEMA = load_input_schema()
INPUT_FIELD_BY_DAT_KEY = {field.dat_key: field for field in INPUT_SCHEMA}
INPUT_FIELD_BY_PARAM = {field.param_name: field for field in INPUT_SCHEMA}
MECHANISM_FILE_KEYS = {
    "GASCTIFILE_LUMEN",
    "SURFCTIFILE_LUMEN",
    "GASCTIFILE_SUPPORT",
    "SURFCTIFILE_SUPPORT",
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _format_input_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def read_input_dat(path: str | Path) -> dict[str, Any]:
    """Read the quasi-2D solver input file into a key/value dictionary."""
    values: dict[str, Any] = {}
    with Path(path).open() as f:
        for line in f:
            data = line.split("//", 1)[0].strip()
            if not data:
                continue

            parts = data.split()
            if len(parts) < 2:
                continue

            key = parts[0]
            field = INPUT_FIELD_BY_DAT_KEY.get(key)
            if field is None:
                values[key] = _parse_scalar(parts[1])
            else:
                values[key] = field.coerce(parts[1])

    return values


def write_input_dat(
    template_path: str | Path,
    output_path: str | Path,
    values: Mapping[str, Any],
) -> None:
    """Render an input.dat file by replacing known values in a template."""
    rendered_lines = []

    with Path(template_path).open() as f:
        for line in f:
            body, sep, comment = line.rstrip("\n").partition("//")
            data = body.strip()
            if not data:
                rendered_lines.append(line.rstrip("\n"))
                continue

            key = data.split()[0]
            if key not in values:
                rendered_lines.append(line.rstrip("\n"))
                continue

            indent = body[: len(body) - len(body.lstrip())]
            replacement = (
                f"{indent}{key:<22} {_format_input_value(values[key]):<22}"
            )
            if sep:
                replacement = f"{replacement} //{comment}"
            rendered_lines.append(replacement.rstrip())

    Path(output_path).write_text("\n".join(rendered_lines) + "\n")


def _mechanism_species(
    mechanism_path: Path,
    phase_name: str,
    fallback: Sequence[str],
) -> list[str]:
    if not mechanism_path.exists():
        return list(fallback)

    try:
        return list(ct.Solution(str(mechanism_path), phase_name).species_names)
    except Exception:
        return list(fallback)


def _input_values_from_params(
    template_values: Mapping[str, Any],
    params: Mapping[str, Any],
    annular_channel: bool,
) -> dict[str, Any]:
    values = dict(template_values)
    values["SOLVE_CHANNEL"] = INPUT_FIELD_BY_DAT_KEY["SOLVE_CHANNEL"].coerce(
        annular_channel
    )

    for name, value in params.items():
        if name == "input_overrides":
            continue
        if name == "inlet_composition":
            values.update(value)
            continue

        field = _input_field_for_name(name)
        if field is not None:
            values[field.dat_key] = field.coerce(value)

    for name, value in params.get("input_overrides", {}).items():
        field = _input_field_for_name(name)
        if field is None:
            values[name] = value
        else:
            values[field.dat_key] = field.coerce(value)

    return values


def _input_field_for_name(name: str) -> InputField | None:
    return (
        INPUT_FIELD_BY_PARAM.get(name)
        or INPUT_FIELD_BY_DAT_KEY.get(name)
        or INPUT_FIELD_BY_DAT_KEY.get(name.upper())
    )


def _case_section_values(
    values: Mapping[str, Any],
    section: str,
) -> dict[str, Any]:
    return {
        field.case_key: values[field.dat_key]
        for field in INPUT_SCHEMA
        if field.case_section == section and field.dat_key in values
    }


def _constant_values(values: Mapping[str, Any]) -> dict[str, Any]:
    constant_sections = {"geometry", "controls", "mechanism_parameters"}
    return {
        field.case_key: values[field.dat_key]
        for field in INPUT_SCHEMA
        if (
            field.case_section in constant_sections
            and field.dat_key not in MECHANISM_FILE_KEYS
            and field.dat_key in values
        )
    }


def _mechanism_file_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field.case_key: values[field.dat_key]
        for field in INPUT_SCHEMA
        if field.dat_key in MECHANISM_FILE_KEYS and field.dat_key in values
    }


def make_cmr_tutorial_case(annular_channel: bool = False) -> CaseParameters:
    """Direct CaseParameters version of the methane CMR tutorial case."""
    return CMRSim(annular_channel=annular_channel).make_case({})


class CMRSim(CaseSimulator):
    """Catalytic membrane reactor wrapper for the external quasi-2D solver.

    The solver interface is file based: ``make_case`` prepares the input values,
    and ``run_case`` writes ``input.dat``, optionally runs the solver, then
    parses either the planned ``q2d_grid_*.csv`` export or the legacy tutorial
    CSV output.
    """

    name = "q2d_cmr"

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        annular_channel: bool = False,
        template_case_dir: str | Path | None = None,
        q2d_experiment_root: str | Path = DEFAULT_Q2D_EXPERIMENT_ROOT,
        solver_command: str | Sequence[str] | None = None,
        solver_timeout_s: float | None = DEFAULT_SOLVER_TIMEOUT_S,
        work_root: str | Path | None = None,
        keep_case_dirs: bool = False,
        use_reference_if_no_solver: bool = True,
    ):
        self._parameter_space = dict(parameter_space or {})
        self.annular_channel = annular_channel
        self.q2d_experiment_root = Path(q2d_experiment_root)
        self.template_case_dir = Path(
            template_case_dir or DEFAULT_TUTORIAL_CASES[annular_channel]
        )
        self.work_root = Path(work_root) if work_root is not None else None
        self.keep_case_dirs = keep_case_dirs
        self.use_reference_if_no_solver = use_reference_if_no_solver
        self.solver_timeout_s = solver_timeout_s

        env_command = os.environ.get("CHEM_OPERATOR_Q2D_SOLVER_COMMAND")
        self.solver_command = solver_command if solver_command is not None else env_command

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    def make_case(self, params: Mapping[str, Any]) -> CaseParameters:
        annular_channel = bool(params.get("annular_channel", self.annular_channel))
        template_case_dir = Path(params.get("template_case_dir", self.template_case_dir))
        template_input = template_case_dir / "input.dat"

        template_values = read_input_dat(template_input)
        values = _input_values_from_params(template_values, params, annular_channel)

        gas_species = _mechanism_species(
            template_case_dir / str(values["GASCTIFILE_LUMEN"]),
            str(values["GASPHASE_LUMEN"]),
            GAS_SPECIES_FALLBACK,
        )
        surface_species = _mechanism_species(
            template_case_dir / str(values["SURFCTIFILE_LUMEN"]),
            str(values["SURFPHASE_LUMEN"]),
            SURFACE_SPECIES_FALLBACK,
        )

        inlet_composition = {
            species: float(values[species])
            for species in gas_species
            if species in values
        }
        surface_coverages = {
            species: float(values[species])
            for species in surface_species
            if species in values
        }

        initial_conditions = _case_section_values(values, "initial_conditions")
        boundary_conditions = _case_section_values(values, "boundary_conditions")
        if "outlet/P" in boundary_conditions:
            initial_conditions.setdefault("gas/P", boundary_conditions["outlet/P"])
        initial_conditions["gas/X"] = inlet_composition
        initial_conditions["surface/coverages"] = surface_coverages

        mechanism_parameters = _case_section_values(values, "mechanism_parameters")
        mechanism_parameters.update(
            {
                "gas_species": gas_species,
                "surface_species": surface_species,
                "template_case_dir": str(template_case_dir),
                "template_input": str(template_input),
                "input_schema": [field.__dict__ for field in INPUT_SCHEMA],
                "input_values": values,
            }
        )

        return CaseParameters(
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            geometry=_case_section_values(values, "geometry"),
            controls=_case_section_values(values, "controls"),
            solver_parameters=_case_section_values(values, "solver_parameters"),
            mechanism_parameters=mechanism_parameters,
        )

    def run_case(self, case: CaseParameters) -> SimulationRecord:
        command_template = self._normalized_solver_command()

        if command_template is None and self.use_reference_if_no_solver:
            template_dir = Path(case.mechanism_parameters["template_case_dir"])
            return parse_legacy_solution(
                template_dir / "solution_files" / "solution_1173K_10Bar_500sccm.csv",
                input_path=template_dir / "input.dat",
                metadata_extra={
                    "mechanism": case.mechanism_parameters.get("GASCTIFILE_LUMEN"),
                    "solver_mode": "tutorial_reference",
                    "solver_parameters": deepcopy(case.solver_parameters),
                },
            )

        if command_template is None:
            raise ValueError(
                "CMRSim requires solver_command or "
                "CHEM_OPERATOR_Q2D_SOLVER_COMMAND unless "
                "use_reference_if_no_solver=True."
            )

        tic = time.perf_counter()
        if self.work_root is not None:
            self.work_root.mkdir(parents=True, exist_ok=True)

        if self.keep_case_dirs:
            case_dir = Path(
                tempfile.mkdtemp(prefix="q2d_cmr_", dir=self.work_root)
            )
            cleanup_context = None
        else:
            cleanup_context = tempfile.TemporaryDirectory(
                prefix="q2d_cmr_", dir=self.work_root
            )
            case_dir = Path(cleanup_context.name)

        try:
            self._prepare_case_directory(case, case_dir)
            command = self._solver_command_for_case(command_template, case_dir)
            completed = subprocess.run(
                command,
                cwd=case_dir,
                text=True,
                capture_output=True,
                check=True,
                timeout=self.solver_timeout_s,
            )

            record = parse_solver_output(case_dir, input_path=case_dir / "input.dat")
            toc = time.perf_counter()
            record.metadata.update(
                {
                    "wall_time": toc - tic,
                    "solver_command": command,
                    "solver_stdout": completed.stdout[-4000:],
                    "solver_stderr": completed.stderr[-4000:],
                    "case_dir": str(case_dir) if self.keep_case_dirs else None,
                    "solver_parameters": deepcopy(case.solver_parameters),
                }
            )
            return record
        except subprocess.TimeoutExpired as err:
            raise TimeoutError(
                "Q2D Docker solver timed out after "
                f"{self.solver_timeout_s} s in {case_dir}. "
                "Increase solver_timeout_s for deliberately hard cases, or reduce "
                "the sampled parameter range."
            ) from err
        except subprocess.CalledProcessError as err:
            stdout = (err.stdout or "")[-4000:]
            stderr = (err.stderr or "")[-4000:]
            raise RuntimeError(
                "Q2D Docker solver failed in "
                f"{case_dir} with exit code {err.returncode}.\n"
                f"Command: {err.cmd}\n"
                f"stdout tail:\n{stdout}\n"
                f"stderr tail:\n{stderr}"
            ) from err
        finally:
            if cleanup_context is not None:
                cleanup_context.cleanup()

    def _normalized_solver_command(self) -> list[str] | None:
        if self.solver_command is None:
            return None
        if isinstance(self.solver_command, str):
            return shlex.split(self.solver_command)
        return list(self.solver_command)

    def _solver_command_for_case(
        self,
        command: Sequence[str],
        case_dir: Path,
    ) -> list[str]:
        format_values = {"case_dir": str(case_dir.resolve())}
        return [part.format(**format_values) for part in command]

    def _prepare_case_directory(
        self,
        case: CaseParameters,
        case_dir: Path,
    ) -> None:
        case_dir.mkdir(parents=True, exist_ok=True)
        template_dir = Path(case.mechanism_parameters["template_case_dir"])

        write_input_dat(
            case.mechanism_parameters["template_input"],
            case_dir / "input.dat",
            case.mechanism_parameters["input_values"],
        )

        for key in (
            "GASCTIFILE_LUMEN",
            "SURFCTIFILE_LUMEN",
            "GASCTIFILE_SUPPORT",
            "SURFCTIFILE_SUPPORT",
        ):
            filename = case.mechanism_parameters.get(key)
            if not filename:
                continue
            source = template_dir / str(filename)
            destination = case_dir / str(filename)
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    value = str(value).strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _empty_field(n_z: int, n_r: int) -> np.ndarray:
    return np.full((n_z, n_r), np.nan)


def parse_q2d_grid_csv(
    csv_path: str | Path,
    manifest_path: str | Path | None = None,
    input_path: str | Path | None = None,
) -> SimulationRecord:
    """Parse the planned long CSV exporter: one row per ``(z, r)`` cell."""
    csv_path = Path(csv_path)
    rows: list[dict[str, str]] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items() if k and k.strip()})

    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")

    n_z = max(int(row["axial_index"]) for row in rows) + 1
    n_r = max(int(row["radial_index"]) for row in rows) + 1
    z = np.full(n_z, np.nan)
    r = np.full(n_r, np.nan)
    region_id = np.full((n_z, n_r), -1, dtype=int)
    region_labels: dict[str, int] = {}

    columns = list(rows[0])
    id_columns = {
        "axial_index",
        "radial_index",
        "region",
        "z",
        "r",
    }
    grouped_prefixes = [
        "theta_support_",
        "sdot_support_",
        "sdot_gas_",
        "axial_flux_",
        "radial_flux_",
        "theta_",
        "wdot_",
        "sdot_",
        "Y_",
        "X_",
    ]

    groups: dict[str, list[tuple[str, str]]] = {}
    scalar_columns = []
    for column in columns:
        if column in id_columns:
            continue

        matched = False
        for prefix in grouped_prefixes:
            if column.startswith(prefix):
                group = prefix[:-1]
                groups.setdefault(group, []).append((column, column[len(prefix) :]))
                matched = True
                break
        if not matched:
            scalar_columns.append(column)

    fields = {name: _empty_field(n_z, n_r) for name in scalar_columns}
    grouped_fields = {
        group: np.full((n_z, n_r, len(items)), np.nan)
        for group, items in groups.items()
    }

    for row in rows:
        j = int(row["axial_index"])
        i = int(row["radial_index"])
        z[j] = _to_float(row["z"])
        r[i] = _to_float(row["r"])

        region = row["region"]
        region_labels.setdefault(region, len(region_labels))
        region_id[j, i] = region_labels[region]

        for column in scalar_columns:
            fields[column][j, i] = _to_float(row.get(column))

        for group, items in groups.items():
            for k, (column, _) in enumerate(items):
                grouped_fields[group][j, i, k] = _to_float(row.get(column))

    fields["region_id"] = region_id
    fields.update(grouped_fields)

    manifest = {}
    if manifest_path is not None and Path(manifest_path).exists():
        import json

        manifest = json.loads(Path(manifest_path).read_text())
    input_values = {}
    if input_path is not None and Path(input_path).exists():
        input_values = read_input_dat(input_path)

    return SimulationRecord(
        coordinates={"z": z, "r": r},
        fields=fields,
        constants=_constant_values(input_values),
        metadata={
            "source_csv": str(csv_path),
            "input_path": str(input_path) if input_path is not None else None,
            "format": "q2d_grid_long_csv",
            "region_labels": region_labels,
            "field_species": {
                group: [species for _, species in items]
                for group, items in groups.items()
            },
            "mechanism_files": _mechanism_file_values(input_values),
            "manifest": manifest,
            "input_values": input_values,
        },
    )


def _read_legacy_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    with path.open() as f:
        headers = [part.strip() for part in f.readline().split(",") if part.strip()]

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] > len(headers):
        data = data[:, : len(headers)]
    return headers, data


def parse_legacy_solution(
    csv_path: str | Path,
    input_path: str | Path,
    metadata_extra: Mapping[str, Any] | None = None,
) -> SimulationRecord:
    """Parse the solver's legacy wide tutorial CSV into a Q2D-like record."""
    csv_path = Path(csv_path)
    input_path = Path(input_path)
    headers, data = _read_legacy_csv(csv_path)
    column_index = {name: i for i, name in enumerate(headers)}
    values = read_input_dat(input_path)

    gas_species = _mechanism_species(
        input_path.parent / str(values.get("GASCTIFILE_LUMEN", "")),
        str(values.get("GASPHASE_LUMEN", "gas")),
        GAS_SPECIES_FALLBACK,
    )
    surface_species = _mechanism_species(
        input_path.parent / str(values.get("SURFCTIFILE_LUMEN", "")),
        str(values.get("SURFPHASE_LUMEN", "Ni_surface")),
        SURFACE_SPECIES_FALLBACK,
    )

    z = data[:, column_index["z"]]
    reactor_radius = float(values.get("CHANNELRAD_LUMEN", 0.0)) / 2.0
    r_values = [reactor_radius]

    annular_columns = {
        "density": "Annular_Density",
        "velocity_axial": "Annular_Velocity",
        "gas_temperature": "Annular_Temperature",
    }
    has_annular = all(name in column_index for name in annular_columns.values())
    if has_annular:
        channel_inner = float(values.get("CHANNELRAD_LUMEN", 0.0))
        channel_outer = float(values.get("CHANNELRAD_ANNULUS", channel_inner))
        r_values.append(channel_inner + 0.5 * (channel_outer - channel_inner))

    n_z = len(z)
    n_r = len(r_values)

    fields: dict[str, np.ndarray] = {
        "pressure": _empty_field(n_z, n_r),
        "velocity_axial": _empty_field(n_z, n_r),
        "gas_temperature": _empty_field(n_z, n_r),
        "solid_temperature": _empty_field(n_z, n_r),
        "density": _empty_field(n_z, n_r),
        "region_id": np.zeros((n_z, n_r), dtype=int),
    }
    if has_annular:
        fields["region_id"][:, 1] = 1

    legacy_scalar_map = {
        "Pressure_0": "pressure",
        "Velocity_0": "velocity_axial",
        "GasTemperature_0": "gas_temperature",
        "SolidTemperature_0": "solid_temperature",
    }
    for legacy_name, field_name in legacy_scalar_map.items():
        if legacy_name in column_index:
            fields[field_name][:, 0] = data[:, column_index[legacy_name]]

    if has_annular:
        for field_name, legacy_name in annular_columns.items():
            fields[field_name][:, 1] = data[:, column_index[legacy_name]]

    fraction_group = "X" if "mole" in csv_path.name.lower() else "Y"
    fraction = np.full((n_z, n_r, len(gas_species)), np.nan)
    for k, species in enumerate(gas_species):
        column = f"{species}_0"
        if column in column_index:
            fraction[:, 0, k] = data[:, column_index[column]]
    fields[fraction_group] = fraction

    theta = np.full((n_z, n_r, len(surface_species)), np.nan)
    for k, species in enumerate(surface_species):
        column = f"{species}_0"
        if column in column_index:
            theta[:, 0, k] = data[:, column_index[column]]
    fields["theta"] = theta

    metadata = {
        "source_csv": str(csv_path),
        "input_path": str(input_path),
        "format": "legacy_wide_csv",
        "region_labels": {"lumen": 0, "annular_channel": 1}
        if has_annular
        else {"lumen": 0},
        "field_species": {
            fraction_group: gas_species,
            "theta": surface_species,
        },
        "mechanism_files": _mechanism_file_values(values),
        "input_values": values,
    }
    metadata.update(metadata_extra or {})

    return SimulationRecord(
        coordinates={"z": z, "r": np.array(r_values)},
        fields=fields,
        constants=_constant_values(values),
        metadata=metadata,
    )


def parse_solver_output(
    case_dir: str | Path,
    input_path: str | Path | None = None,
) -> SimulationRecord:
    case_dir = Path(case_dir)
    input_path = Path(input_path or case_dir / "input.dat")

    grid_files = sorted(case_dir.glob("q2d_grid_*.csv"))
    if grid_files:
        grid_file = grid_files[-1]
        manifest = grid_file.with_name(
            grid_file.name.replace("q2d_grid_", "q2d_grid_manifest_").replace(
                ".csv", ".json"
            )
        )
        return parse_q2d_grid_csv(grid_file, manifest, input_path)

    legacy_files = sorted(case_dir.glob("solution_*.csv"))
    if not legacy_files and (case_dir / "solution_files").exists():
        legacy_files = sorted((case_dir / "solution_files").glob("solution_*.csv"))
    if legacy_files:
        return parse_legacy_solution(legacy_files[-1], input_path)

    raise FileNotFoundError(f"No q2d_grid_*.csv or solution_*.csv found in {case_dir}")


def compare_record_to_legacy_csv(
    record: SimulationRecord,
    csv_path: str | Path,
) -> dict[str, float]:
    """Compare the parsed record against the tutorial CSV columns it came from."""
    headers, data = _read_legacy_csv(csv_path)
    column_index = {name: i for i, name in enumerate(headers)}

    comparisons = {
        "z": (
            np.asarray(record.coordinates["z"], dtype=float),
            data[:, column_index["z"]],
        ),
        "Pressure_0": (
            np.asarray(record.fields["pressure"], dtype=float)[:, 0],
            data[:, column_index["Pressure_0"]],
        ),
        "Velocity_0": (
            np.asarray(record.fields["velocity_axial"], dtype=float)[:, 0],
            data[:, column_index["Velocity_0"]],
        ),
        "GasTemperature_0": (
            np.asarray(record.fields["gas_temperature"], dtype=float)[:, 0],
            data[:, column_index["GasTemperature_0"]],
        ),
    }

    species = record.metadata["field_species"].get("Y", [])
    if "Y" in record.fields:
        for species_name in ("CH4", "H2", "CO2"):
            column = f"{species_name}_0"
            if column in column_index and species_name in species:
                comparisons[column] = (
                    record.fields["Y"][:, 0, species.index(species_name)],
                    data[:, column_index[column]],
                )

    if "Annular_Velocity" in column_index:
        comparisons["Annular_Velocity"] = (
            np.asarray(record.fields["velocity_axial"], dtype=float)[:, 1],
            data[:, column_index["Annular_Velocity"]],
        )

    return {
        name: float(np.nanmax(np.abs(actual - expected)))
        for name, (actual, expected) in comparisons.items()
    }


def default_docker_solver_command(image: str = "chem-operator-q2d") -> list[str]:
    """Docker command for running the Q2D solver in a generated case directory."""
    return ["docker", "run", "--rm", "-v", "{case_dir}:/work", image]
