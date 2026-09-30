#!/usr/bin/env python3
"""
Smoke test on SYNTHETIC data (no study data required).

Builds a tiny fake project (3 participants x 2 runs, 10x10x10-voxel volumes, 25
random "components" and random NCT tables), then runs pipeline steps 04-07 through
their command-line interfaces and checks that the expected outputs exist.

    python tests/smoke_test.py

Steps 00, 01, 02, 03 and 08 need external software/data (TemplateFlow, FSL, ANTs,
CBIG NCT, nilearn's atlas download) and are not covered here.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
from nct_common import NETWORK_MAPPINGS  # noqa: E402

N_IC, N_T, SHAPE = 25, 60, (10, 10, 10)
SUBJECTS = [90001, 90002, 90003]
SESSIONS = ["ses-v1", "ses-v2o1"]


def build_project(root: Path, seed: int = 1):
    rng = np.random.default_rng(seed)
    affine = np.diag([2, 2, 2, 1.0])

    for sid in SUBJECTS:
        sub = f"sub-{sid}"

        # MELODIC output: 4-D components, each with one bright blob
        ics = rng.standard_normal(SHAPE + (N_IC,)) * 0.3
        for k in range(N_IC):
            c = rng.integers(2, 8, 3)
            ics[c[0]-1:c[0]+2, c[1]-1:c[1]+2, c[2]-1:c[2]+2, k] += 3
        gica = root / "melodic" / f"{sub}_subject_level_ICA.gica"
        gica.mkdir(parents=True)
        nib.save(nib.Nifti1Image(ics.astype(np.float32), affine), gica / "melodic_IC.nii.gz")

        # BOLD runs + run list
        runs = []
        for ses in SESSIONS:
            d = root / "bold" / sub / ses
            d.mkdir(parents=True)
            p = d / f"{sub}_{ses}_bold.nii.gz"
            bold = 100 + rng.standard_normal(SHAPE + (N_T,)) + rng.standard_normal(N_T)
            nib.save(nib.Nifti1Image(bold.astype(np.float32), affine), p)
            runs.append(str(p))
        lst = root / "fmriprep_lists" / sub
        lst.mkdir(parents=True)
        (lst / "subject_level_files.txt").write_text("\n".join(runs) + "\n")

        # Dual-regression stage-2 output (one 4-D file per run)
        dr = root / "dual_regression" / str(sid)
        dr.mkdir(parents=True)
        for k in range(len(runs)):
            m = (rng.standard_normal(SHAPE + (N_IC,)) * 2).astype(np.float32)
            nib.save(nib.Nifti1Image(m, affine), dr / f"dr_stage2_subject{k:05d}.nii.gz")

        # NCT output: one network_correspondence.csv per component
        for ic in range(N_IC):
            rows = []
            for atlases in NETWORK_MAPPINGS.values():
                for atlas, labels in atlases.items():
                    for label in labels:
                        strong = rng.random() < 0.09
                        rows.append({
                            "group": atlas, "name": label,
                            "dice": rng.uniform(.25, .55) if strong else rng.uniform(0, .28),
                            "p_value": rng.uniform(0, .04) if strong else rng.uniform(0, 1),
                        })
            d = root / "derived" / "nct_results" / sub / f"{sub}_IC{ic:02d}"
            d.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(d / "network_correspondence.csv")


def run(cmd, env):
    print("$", " ".join(str(c) for c in cmd))
    res = subprocess.run([sys.executable, *map(str, cmd)], env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-2000:])
        raise SystemExit(f"FAILED: {cmd[0]}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "project"
        build_project(root)
        env = {**os.environ, "CATD_ROOT": str(root)}
        assign = root / "derived" / "network_assignments"

        run([SCRIPTS / "04_assign_networks.py"], env)
        run([SCRIPTS / "05_qa_network_assignment.py"], env)
        run([SCRIPTS / "06_extract_dr_metrics.py",
             "--network-assignments", assign / "subject_network_assignments.csv",
             "--overwrite-masks"], env)
        bnc_csv = Path(tmp) / "bnc.csv"
        run([SCRIPTS / "07_between_network_connectivity.py", "--output-csv", bnc_csv], env)

        # ---- checks ----
        n_sub, n_runs = len(SUBJECTS), len(SUBJECTS) * len(SESSIONS)
        summary = pd.read_csv(assign / "subject_network_assignments.csv")
        assert len(summary) == n_sub
        assert (assign / "quality_plots" / "coverage_by_network.png").exists()

        dr = pd.read_csv(root / "derived" / "dualreg_nct_metrics.tsv", sep="\t")
        assert len(dr) == n_runs and {"DMN_mean", "DMN_b_mean", "DMN_c_mean"} <= set(dr.columns)

        bnc = pd.read_csv(bnc_csv)
        assert len(bnc) == n_runs and {"DMN_SN_Z", "DMN_SN_Z_comb"} <= set(bnc.columns)

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
