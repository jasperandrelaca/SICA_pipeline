#!/usr/bin/env python3
"""
Orchestration test for scripts/preprocessing/P2_denoise_bold.py.

FSL is replaced by tiny stub programs that log their arguments and copy their
input to the output path, so this checks file pairing, naming, command
arguments, cleanup, skipping and error handling -- NOT the image processing
itself (that needs real FSL). Run:  python tests/test_denoise_stubbed.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "scripts" / "preprocessing" / "P2_denoise_bold.py"
sys.path.insert(0, str(P2.parent))
from P2_denoise_bold import BASE_CONFOUNDS, GSR_CONFOUNDS  # noqa: E402

SPACE = "space-MNIPediatricAsym_cohort-1_res-2"
FSLMATHS = '#!/bin/bash\necho "fslmaths $*" >> "$STUB_LOG"\ncp "$1" "${@: -1}"\n'
REGFILT = ('#!/bin/bash\necho "fsl_regfilt $*" >> "$STUB_LOG"\n'
           'while [[ $# -gt 0 ]]; do case "$1" in -i) i="$2";; -o) o="$2";; esac; shift; done\n'
           'cp "$i" "$o"\n')


def make_runs(func, runs, n_vol=20):
    func.mkdir(parents=True)
    cols = BASE_CONFOUNDS + GSR_CONFOUNDS
    for run in runs:
        pre = f"sub-901_ses-1_task-rest_{run}"
        for name, shape in ((f"{pre}_{SPACE}_desc-preproc_bold.nii.gz", (4, 4, 4, n_vol)),
                            (f"{pre}_{SPACE}_desc-brain_mask.nii.gz", (4, 4, 4))):
            img = nib.Nifti1Image(np.random.rand(*shape).astype("float32"), np.eye(4))
            if len(shape) == 4:
                img.header.set_zooms((2, 2, 2, 2.5))
            nib.save(img, str(func / name))
        df = pd.DataFrame(np.random.randn(n_vol, len(cols)), columns=cols)
        df.loc[0, [c for c in cols if "derivative1" in c]] = np.nan
        df.to_csv(func / f"{pre}_desc-confounds_timeseries.tsv", sep="\t", index=False,
                  na_rep="n/a")


def run_p2(sub, gsr, env, *extra):
    return subprocess.run([sys.executable, str(P2), str(sub), "--gsr", gsr, *extra],
                          env=env, capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "bin").mkdir()
        for name, body in (("fslmaths", FSLMATHS), ("fsl_regfilt", REGFILT)):
            (tmp / "bin" / name).write_text(body)
            (tmp / "bin" / name).chmod(0o755)
        log = tmp / "stub.log"
        env = dict(os.environ, PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", STUB_LOG=str(log))

        # two runs in one folder, global signal regression on
        sub = tmp / "sub-901" / "ses-1"
        make_runs(sub / "func", ["run-1", "run-2"])
        r = run_p2(sub, "y", env)
        assert r.returncode == 0, r.stderr
        finals = sorted(p.name for p in (sub / "func").glob("*demeaned_masked.nii.gz"))
        assert len(finals) == 2 and all("_removed_GS_smoothed_filtered_demeaned_masked" in f
                                        for f in finals), finals
        assert not list((sub / "func").glob("*_smoothed.nii.gz")), "intermediates not cleaned"
        design = pd.read_csv(next((sub / "func").glob("*run-1_filtered_confounds.tsv")),
                             sep="\t", header=None)
        assert design.shape == (20, 36) and not design.isna().any().any()
        cmds = log.read_text().splitlines()
        assert sum(c.startswith("fsl_regfilt") for c in cmds) == 2
        assert all(" -f " + ",".join(map(str, range(1, 37))) in c
                   for c in cmds if c.startswith("fsl_regfilt"))
        assert any("-s 2.123" in c for c in cmds) and any("-bptf" in c for c in cmds)

        # re-running skips; the other GSR variant is treated as a separate output
        assert "skipping" in run_p2(sub, "y", env).stdout
        assert run_p2(sub, "n", env).returncode == 0
        assert len(list((sub / "func").glob("*_with_GS_*_masked.nii.gz"))) == 2

        # failures: missing mask and confound/volume mismatch -> non-zero exit, logged
        bad = tmp / "sub-902" / "ses-1"
        make_runs(bad / "func", ["run-1", "run-2"])
        next((bad / "func").glob("*run-2*brain_mask*")).unlink()
        f = next((bad / "func").glob("*run-1*confounds*"))
        pd.read_csv(f, sep="\t").iloc[:-2].to_csv(f, sep="\t", index=False, na_rep="n/a")
        r = run_p2(bad, "y", env)
        assert r.returncode == 1
        text = (bad / "preproc_denoising_log.tsv").read_text()
        assert text.count("FAILED") == 2 and "BOLD volumes" in text and "mask file not found" in text
    print("test_denoise_stubbed: all checks passed")


if __name__ == "__main__":
    main()
