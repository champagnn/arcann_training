"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

Mock end-to-end test of the input generation for PolarMACE explorations with symmetrix.

Runs the real ArcaNN command line ('python -m arcann_training STEP PHASE -c hpc1') on a fake training
directory and checks the files it writes. Nothing is simulated or labeled: LAMMPS, CP2K, MACE and
symmetrix are never called (the converted '.model.json' models are empty placeholders), and the steps
in between (training, launch/check/deviate/extract) are replaced by writing their control JSON by hand.

Covered:
- initialization detects the PolarMACE model;
- exploration prepare writes symmetrix/mace/polar inputs with the charge/spin of each system, in the
  main run and in every committee rerun, keeping the user's options ('&'-continued line included);
- labeling prepare inherits the charge/spin of the exploration;
- the next exploration takes them back from the labeling;
- inconsistent inputs (wrong pair_style for the model, swapped placeholders) stop prepare;
- a non-polar MACE input still keeps its pair_style options in the reruns.

Run from the repository root (not part of the pre-commit unit tests, it takes a few seconds per run):
    python -m unittest discover -s tests/mock -v
Set ARCANN_MOCK_KEEP=1 to keep the generated directories (their path is printed) for inspection.
"""

# Standard library modules
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "sn2_ch3cl_br_mace" / "NR" / "user_files"
MACHINE = "hpc1"

# One structure, explored as two systems with different charge/multiplicity
SYSTEMS = {
    "sysa": {"charge": -1, "total_spin": 1},
    "sysb": {"charge": 0, "total_spin": 2},
}
NNP_COUNT = 3

POLAR_IN = """units metal
boundary p p p
atom_style atomic
atom_modify map yes
newton on
read_data _R_DATA_FILE_

{pair}

timestep _R_TIMESTEP_
dump dcd all dcd _R_PRINT_FREQ_ _R_DCD_OUT_
thermo _R_PRINT_FREQ_
velocity all create _R_TEMPERATURE_ _R_SEED_VEL_ dist gaussian
fix 1 all nve
fix 2 all temp/csvr _R_TEMPERATURE_ _R_TEMPERATURE_ $(100.0*dt) _R_SEED_THER_
run _R_NUMBER_OF_STEPS_
write_restart _R_RESTART_OUT_
"""

POLAR_PAIR = """pair_style symmetrix/mace/polar charge _R_CHARGE_ &
           spin _R_TOTAL_SPIN_ mesh 0.6 # user options must survive in the reruns
pair_coeff * * _R_MODEL_FILES_ _R_ATOM_LABELS_"""
POLAR_PAIR_EXPECTED = (
    "pair_style symmetrix/mace/polar charge {charge} spin {total_spin} mesh 0.6"
)


class MockTraining:
    """A fake ArcaNN training directory, driven through the real command line."""

    def __init__(self, polar: bool = True, pair: str = POLAR_PAIR):
        self.path = Path(tempfile.mkdtemp(prefix="arcann_mock_"))
        self.user_files = self.path / "user_files"
        self.user_files.mkdir()
        (self.path / "data").mkdir()
        (self.path / "NNP").mkdir()

        for name in [
            "properties.txt",
            "machine.json",
            f"job_lammps-mace_explore_gpu_{MACHINE}.sh",
            f"job-array_lammps-mace_explore_gpu_{MACHINE}.sh",
            f"job_CP2K_label_cpu_{MACHINE}.sh",
            f"job-array_CP2K_label_cpu_{MACHINE}.sh",
        ]:
            shutil.copy(EXAMPLE / name, self.user_files / name)

        mace_yml = (EXAMPLE / "mace_0.3.14.yaml").read_text()
        if polar:
            mace_yml = mace_yml.replace('model: "MACE"', 'model: "PolarMACE"')
        (self.user_files / "mace_0.3.14.yaml").write_text(mace_yml)

        for system in SYSTEMS:
            shutil.copy(
                EXAMPLE / "ch3cl_br_free_300K.lmp", self.user_files / f"{system}.lmp"
            )
            (self.user_files / f"{system}.in").write_text(POLAR_IN.format(pair=pair))
            for step in (1, 2):
                shutil.copy(
                    EXAMPLE / f"{step}_ch3cl_br_free_300K_labeling_XXXXX_{MACHINE}.inp",
                    self.user_files / f"{step}_{system}_labeling_XXXXX_{MACHINE}.inp",
                )

        # 'exploration prepare' only checks that atomsk exists
        self.atomsk = self.path / "atomsk"
        self.atomsk.write_text("#!/bin/sh\n")
        self.atomsk.chmod(0o755)

        (self.path / "input.json").write_text(
            json.dumps(
                {"nnp_program": "mace", "data_format": "extxyz", "nnp_count": NNP_COUNT}
            )
        )

    def cleanup(self):
        if os.environ.get("ARCANN_MOCK_KEEP"):
            print(f"\nMock training kept in: {self.path}")
        else:
            shutil.rmtree(self.path, ignore_errors=True)

    # ------------------------------------------------------------------------------------------ #
    def run(self, folder: str, step: str, phase: str, input_json: dict | None = None):
        """Run 'python -m arcann_training STEP PHASE' in a folder of the training, return the process."""
        cwd = self.path / folder
        cwd.mkdir(exist_ok=True)
        if input_json is not None:
            (cwd / "input.json").write_text(json.dumps(input_json))
        return subprocess.run(
            [sys.executable, "-m", "arcann_training", step, phase, "-c", MACHINE],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def failed(process) -> bool:
        """A phase failed: non-zero exit code (the error is also reported in the log)."""
        failed = process.returncode != 0
        logged = "encountered an error" in process.stdout + process.stderr
        assert failed == logged, (
            f"exit code {process.returncode} but error logged: {logged}"
        )
        return failed

    def control(self, name: str) -> dict:
        return json.loads((self.path / "control" / f"{name}.json").read_text())

    def write_control(self, name: str, data: dict):
        path = self.path / "control" / f"{name}.json"
        if path.exists():
            path.chmod(0o644)
        path.write_text(json.dumps(data, indent=4))

    def fake_training(self, iteration: int):
        """Replace a training step: the control JSON and placeholder symmetrix models."""
        padded = str(iteration).zfill(3)
        self.write_control(
            f"training_{padded}",
            {
                "mace_model_version": "0.3.14",
                "mace_dtype": "float64",
                "is_frozen": True,
                "is_compressed": True,
                "is_incremented": True,
            },
        )
        for nnp in range(1, NNP_COUNT + 1):
            (self.path / "NNP" / f"model_{nnp}_{padded}.model.json").write_text("{}")
        (self.path / f"{str(iteration + 1).zfill(3)}-exploration").mkdir(exist_ok=True)

    def fake_exploration_extract(self, iteration: int, candidates: int = 2):
        """Replace launch/check/deviate/extract: mark the exploration extracted and write candidates."""
        padded = str(iteration).zfill(3)
        exploration_json = self.control(f"exploration_{padded}")
        exploration_json.update(
            is_launched=True, is_checked=True, is_deviated=True, is_extracted=True
        )
        for system in exploration_json["systems_auto"]:
            exploration_json["systems_auto"][system].update(
                selected_count=candidates,
                candidates_count=candidates,
                rejected_count=0,
                total_count=candidates,
                disturbed_candidate_value=0.0,
                disturbed_start_value=0.0,
                ignore_first_x_ps=0.5,
                sigma_low=0.1,
                sigma_high=0.8,
                sigma_high_limit=1.5,
                max_candidates=candidates,
                mean_s_per_step=0.01,
            )
            frame = xyz_frame_from_lmp(
                self.user_files / f"{system}.lmp", self.control("config")
            )
            (
                self.path
                / f"{padded}-exploration"
                / system
                / f"candidates_{padded}_{system}.xyz"
            ).write_text(frame * candidates)
        self.write_control(f"exploration_{padded}", exploration_json)
        (self.path / f"{padded}-labeling").mkdir(exist_ok=True)


def xyz_frame_from_lmp(lmp: Path, config: dict) -> str:
    """One extended-XYZ frame with the atoms of a LAMMPS data file (orthogonal box)."""
    text = lmp.read_text()
    box = [
        list(map(float, m.groups()))
        for m in re.finditer(r"(\S+)\s+(\S+)\s+[xyz]lo\s+[xyz]hi", text)
    ]
    lengths = [hi - lo for lo, hi in box]
    lines_after = text.split("Atoms")[1].splitlines()[1:]
    atoms = []
    for line in lines_after:
        if line.strip():
            atoms.append(line)
        elif atoms:
            break
    lines = [str(len(atoms))]
    lines.append(
        f'Lattice="{lengths[0]} 0.0 0.0 0.0 {lengths[1]} 0.0 0.0 0.0 {lengths[2]}" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"'
    )
    for atom in atoms:
        fields = atom.split()
        symbol = config["properties"][fields[1]]["symbol"]
        lines.append(f"{symbol} {fields[-3]} {fields[-2]} {fields[-1]}")
    return "\n".join(lines) + "\n"


def lammps_inputs(folder: Path, system: str) -> list[Path]:
    return sorted(folder.glob(f"{system}/*/*/{system}_*.in"))


# ---------------------------------------------------------------------------------------------- #
class TestPolarMaceInputChain(unittest.TestCase):
    """initialization -> exploration 1 -> labeling 1 -> exploration 2, all prepare phases only."""

    @classmethod
    def setUpClass(cls):
        cls.mock = MockTraining()
        m = cls.mock

        cls.init = m.run(".", "initialization", "start")
        m.fake_training(0)
        cls.exp1 = m.run(
            "001-exploration",
            "exploration",
            "prepare",
            {
                "atomsk_path": str(m.atomsk),
                "traj_count": 1,
                "systems_charge": [s["charge"] for s in SYSTEMS.values()],
                "systems_total_spin": [s["total_spin"] for s in SYSTEMS.values()],
            },
        )
        if not m.failed(cls.exp1):
            m.fake_exploration_extract(1)
            # No charge/spin given to the labeling: they must come from the exploration
            cls.lab1 = m.run("001-labeling", "labeling", "prepare", {})
            m.fake_training(1)
            # Nor to the next exploration: they must come from the labeling
            cls.exp2 = m.run(
                "002-exploration",
                "exploration",
                "prepare",
                {
                    "atomsk_path": str(m.atomsk),
                    "traj_count": 1,
                    "previous_start": False,
                },
            )

    @classmethod
    def tearDownClass(cls):
        cls.mock.cleanup()

    def assertRan(self, process):  # noqa: N802
        self.assertFalse(
            MockTraining.failed(process),
            msg=process.stdout[-3000:] + process.stderr[-3000:],
        )

    def test_1_initialization_detects_polar(self):
        self.assertRan(self.init)
        self.assertTrue(self.mock.control("config")["polar_mace"])

    def check_exploration_inputs(self, folder: str):
        for system, values in SYSTEMS.items():
            inputs = lammps_inputs(self.mock.path / folder, system)
            self.assertEqual(len(inputs), NNP_COUNT, msg=f"{system}: {inputs}")
            for lmp_in in inputs:
                with self.subTest(file=lmp_in.name):
                    text = lmp_in.read_text()
                    self.assertNotRegex(text, r"_R_\w+_", msg="placeholder left")
                    commands = [
                        " ".join(c.split("#")[0].split())
                        for c in text.replace("&\n", " ").splitlines()
                        if c.strip().startswith("pair_")
                    ]
                    # main run + one rerun per other committee member
                    self.assertEqual(
                        [c for c in commands if c.startswith("pair_style")],
                        [POLAR_PAIR_EXPECTED.format(**values)] * NNP_COUNT,
                    )
                    models = [
                        c.split()[3] for c in commands if c.startswith("pair_coeff")
                    ]
                    # every committee member exactly once, models of the previous training
                    prev_iter = str(int(folder[:3]) - 1).zfill(3)
                    self.assertEqual(
                        sorted(models),
                        [
                            f"model_{n}_{prev_iter}.model.json"
                            for n in range(1, NNP_COUNT + 1)
                        ],
                    )
                    for model in models:
                        self.assertTrue(
                            (lmp_in.parent / model).exists(), msg=f"{model} not linked"
                        )
                    self.assertEqual(text.count("\nrerun "), NNP_COUNT - 1)

    def test_2_exploration_1_inputs(self):
        self.assertRan(self.exp1)
        self.check_exploration_inputs("001-exploration")
        exploration_json = self.mock.control("exploration_001")
        for system, values in SYSTEMS.items():
            self.assertEqual(
                {k: exploration_json["systems_auto"][system][k] for k in values}, values
            )

    def test_3_labeling_inherits_charge_spin(self):
        self.assertRan(self.lab1)
        labeling_json = self.mock.control("labeling_001")
        for system, values in SYSTEMS.items():
            self.assertEqual(
                {k: labeling_json["systems_auto"][system][k] for k in values}, values
            )
        # CP2K inputs were written for every candidate
        for system in SYSTEMS:
            self.assertEqual(
                len(
                    list(
                        (self.mock.path / "001-labeling" / system).glob(
                            "*/1_labeling_*.inp"
                        )
                    )
                ),
                2,
            )

    def test_4_exploration_2_takes_labeling_values(self):
        self.assertRan(self.exp2)
        self.check_exploration_inputs("002-exploration")


# ---------------------------------------------------------------------------------------------- #
class TestInconsistentInputs(unittest.TestCase):
    """Inputs that must stop 'exploration prepare'."""

    def prepare(self, polar: bool, pair: str):
        mock = MockTraining(polar=polar, pair=pair)
        self.addCleanup(mock.cleanup)
        self.assertFalse(mock.failed(mock.run(".", "initialization", "start")))
        mock.fake_training(0)
        return mock, mock.run(
            "001-exploration",
            "exploration",
            "prepare",
            {"atomsk_path": str(mock.atomsk), "traj_count": 1},
        )

    def test_polar_model_with_non_polar_pair_style(self):
        _, process = self.prepare(
            True,
            "pair_style symmetrix/mace\npair_coeff * * _R_MODEL_FILES_ _R_ATOM_LABELS_",
        )
        self.assertTrue(MockTraining.failed(process))
        self.assertIn("symmetrix/mace/polar", process.stdout + process.stderr)

    def test_non_polar_model_with_polar_pair_style(self):
        _, process = self.prepare(False, POLAR_PAIR)
        self.assertTrue(MockTraining.failed(process))
        self.assertIn("not a PolarMACE", process.stdout + process.stderr)

    def test_swapped_placeholders(self):
        _, process = self.prepare(
            True,
            "pair_style symmetrix/mace/polar charge _R_TOTAL_SPIN_ spin _R_CHARGE_\n"
            "pair_coeff * * _R_MODEL_FILES_ _R_ATOM_LABELS_",
        )
        self.assertTrue(MockTraining.failed(process))
        self.assertIn("use 'charge _R_CHARGE_'", process.stdout + process.stderr)

    def test_hardcoded_model(self):
        _, process = self.prepare(
            True,
            "pair_style symmetrix/mace/polar\npair_coeff * * model.json C H Cl Br N",
        )
        self.assertTrue(MockTraining.failed(process))
        self.assertIn("_R_MODEL_FILES_", process.stdout + process.stderr)


class TestNonPolarMace(unittest.TestCase):
    """Regression: a regular MACE input keeps its pair_style options in the reruns."""

    def test_options_kept(self):
        mock = MockTraining(
            polar=False,
            pair="pair_style symmetrix/mace/kk\npair_coeff * * _R_MODEL_FILES_ _R_ATOM_LABELS_",
        )
        self.addCleanup(mock.cleanup)
        self.assertFalse(mock.failed(mock.run(".", "initialization", "start")))
        mock.fake_training(0)
        process = mock.run(
            "001-exploration",
            "exploration",
            "prepare",
            {"atomsk_path": str(mock.atomsk), "traj_count": 1},
        )
        self.assertFalse(
            mock.failed(process), msg=process.stdout[-3000:] + process.stderr[-3000:]
        )
        for lmp_in in lammps_inputs(mock.path / "001-exploration", "sysa"):
            text = lmp_in.read_text()
            self.assertEqual(text.count("pair_style symmetrix/mace/kk\n"), NNP_COUNT)
            self.assertNotIn("_R_CHARGE_", text)
        self.assertNotIn(
            "charge", mock.control("exploration_001")["systems_auto"]["sysa"]
        )


if __name__ == "__main__":
    unittest.main()
