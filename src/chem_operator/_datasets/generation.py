"""HDF5 simulation-dataset generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from tqdm import trange

from chem_operator.sampling import ParameterSpec

from .records import CaseSimulator, SimulationRecord


class SimulationDatasetGenerator:
    """
    may need to be refactored to save simulation trajecories one at a time to reduce RAM consumption
    currently all generated then all saved
    """
    def __init__(
        self,
        simulator: CaseSimulator,
        output_path: str | Path, # move to save method?
        seed: int = 0,
        # storage_format: Literal["hdf5", "zarr"] = "hdf5"
        # resume: bool = True
        # num_workers: int = 1
        # sampling_policy: SamplingPolicy
        # failure_policy: Literal["raise", "skip", "retry"] = "retry"
        # save_solver_diagnostics: bool = True
    ):
        self.simulator = simulator
        self.output_path = Path(output_path)
        self.seed = seed
        # self.overwrite = overwrite

        # if self.output_path.exists() and not self.overwrite:
        #     raise FileExistsError(f"{self.output_path} already exists.")

    @staticmethod
    def split_parameter_space(
        parameter_space: Mapping[str, ParameterSpec],
    ):
        grid_specs = {}
        sampled_specs = {}

        for name, spec in parameter_space.items():
            if spec.is_grid:
                grid_specs[name] = spec
            else:
                sampled_specs[name] = spec

        return sampled_specs, grid_specs

    @staticmethod
    def iter_grid_combinations(
        grid_specs: Mapping[str, ParameterSpec],
    ):
        if not grid_specs:
            yield 0, {}
            return

        names = list(grid_specs)
        value_lists = [grid_specs[name].grid_values() for name in names]

        for grid_idx, values in enumerate(product(*value_lists)):
            yield grid_idx, dict(zip(names, values))

    def generate_split(
        self,
        split: str,
        n_cases: int,
        seed: int,
    ) -> list[SimulationRecord]:
        rng = np.random.default_rng(seed)
        records = []

        sampled_specs, grid_specs = SimulationDatasetGenerator.split_parameter_space(
            self.simulator.parameter_space
        )

        record_idx = 0

        print(f"{split = }")

        for base_case_idx in trange(n_cases):
            sampled_params = {
                name: spec.sample(rng)
                for name, spec in sampled_specs.items()
            }

            for grid_idx, grid_params in (
                SimulationDatasetGenerator.iter_grid_combinations(grid_specs)
            ):
                params = sampled_params | grid_params

                case = self.simulator.make_case(params)
                try:
                    record = self.simulator.run_case(case)
                except Exception as e:
                    print(f"Simulation case [{base_case_idx = },{record_idx = }] failed")
                    print(e)
                    continue

                record.metadata.update(
                    {
                        "record_idx": record_idx,
                        "base_case_idx": base_case_idx,
                        "grid_idx": grid_idx,
                        "split": split,
                        "simulator": self.simulator.name,
                        "seed": seed,
                        "params": deepcopy(params),
                    }
                )

                records.append(record)
                record_idx += 1

        return records

    def generate_splits(
        self,
        n_cases: int = 10,
        train_fraction: float = 0.8,
        valid_fraction: float = 0.1,
        test_fraction: float = 0.1,
    ) -> dict[str, list[SimulationRecord]]:
        if not np.isclose(train_fraction + valid_fraction + test_fraction, 1.0):
            raise ValueError("Split fractions must sum to 1.")

        n_train = int(n_cases * train_fraction)
        n_valid = int(n_cases * valid_fraction)
        n_test = n_cases - n_train - n_valid

        return {
            "train": self.generate_split("train", n_train, self.seed),
            "valid": self.generate_split("valid", n_valid, self.seed + 1),
            "test": self.generate_split("test", n_test, self.seed + 2),
        }
    
    def save_split(
        self,
        split: str,
        records: list[SimulationRecord],
        overwrite: bool = False,
    ) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.simulator.name}_{split}.h5"

        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Use overwrite = True.")

        mode = "w" if overwrite else "x"

        with h5py.File(path, mode) as h5:
            h5.attrs["schema"] = "simulation-record-split-v0"
            h5.attrs["simulator"] = self.simulator.name
            h5.attrs["split"] = split
            h5.attrs["n_cases"] = len(records)

            if records:
                h5.create_dataset(
                    "field_names",
                    data=np.array(list(records[0].fields.keys()), dtype="S"),
                )
                h5.create_dataset(
                    "constant_names",
                    data=np.array(list(records[0].constants.keys()), dtype="S"),
                )

            cases_group = h5.create_group("cases")

            for i, record in enumerate(records):
                case_group = cases_group.create_group(f"{i:06d}")
                self._save_record(case_group, record)

    def save_splits(
            self, 
            records_splits: dict[str, list[SimulationRecord]], 
            overwrite: bool = False
    ):
        for split, records in records_splits.items():
            self.save_split(split, records, overwrite)
    
    def _save_record(
        self,
        group: h5py.Group,
        record: SimulationRecord,
    ) -> None:
        coordinates_group = group.create_group("coordinates")
        fields_group = group.create_group("fields")
        constants_group = group.create_group("constants")

        for name, value in record.coordinates.items():
            coordinates_group.create_dataset(name, data=np.asarray(value))

        for name, value in record.fields.items():
            fields_group.create_dataset(name, data=np.asarray(value))

        for name, value in record.constants.items():
            self._save_value(constants_group, name, value)

        metadata_json = json.dumps(record.metadata, default=self._json_default)
        group.attrs["metadata"] = metadata_json

    def _save_value(
        self,
        group: h5py.Group,
        name: str,
        value,
    ) -> None:
        if isinstance(value, dict):
            subgroup = group.create_group(name)
            for key, subvalue in value.items():
                self._save_value(subgroup, key, subvalue)
        elif isinstance(value, str):
            group.attrs[name] = value
        elif np.isscalar(value):
            group.attrs[name] = value
        else:
            group.create_dataset(name, data=np.asarray(value))

    def _json_default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)
