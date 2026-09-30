#!/usr/bin/env python3
"""Test for scripts/preprocessing/P3_count_scans_build_lists.py on an empty-file tree.
Run:  python tests/test_scan_counts.py"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

P3 = Path(__file__).resolve().parents[1] / "scripts" / "preprocessing" / "P3_count_scans_build_lists.py"
SP = "space-MNIPediatricAsym_cohort-1_res-2"


def touch_runs(root, label, n, variant="removed_GS", size=10):
    for i in range(n):
        ses = f"ses-{i + 1}"
        f = (root / f"sub-{label}" / ses / "func" /
             f"sub-{label}_{ses}_task-rest_{SP}_desc-preproc_{variant}_smoothed_filtered_demeaned_masked.nii.gz")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * size)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fp = tmp / "fmriprep"
        for label, n in {"901": 1, "902": 2, "903": 3, "904": 4, "905": 6}.items():
            touch_runs(fp, label, n)
        touch_runs(fp, "903", 2, variant="with_GS")      # other variant must be ignored
        touch_runs(fp, "906", 3, size=0)                 # empty files must not count
        (tmp / "labels.txt").write_text("901\nsub-902\n903\n904\n905\n906\n907\n")
        with open(tmp / "map.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bids_label", "study_id"])
            for b, sid in zip(range(901, 908), range(10001, 10008)):
                w.writerow([b, sid])
        env = dict(os.environ, CATD_ROOT=str(tmp))
        base = [sys.executable, str(P3), "--gsr", "y", "--fmriprep-dir", str(fp),
                "--subject-list", str(tmp / "labels.txt"), "--id-map", str(tmp / "map.csv")]

        r = subprocess.run(base + ["--out-csv", str(tmp / "c.csv")], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = r.stdout
        # 901:1  902:2  903:3  904:4  905:6 -> 4+ = 2 ; 906, 907 -> 0 scans
        assert "Participants examined: 7" in out
        table = {}
        for ln in out.splitlines():
            parts = ln.split()
            if len(parts) == 2 and parts[0] in ("0", "1", "2", "3", "4+"):
                table[parts[0]] = int(parts[1])
        assert table == {"0": 2, "1": 1, "2": 1, "3": 1, "4+": 2}, table
        floors = {ln.split()[1]: (int(ln.split()[2]), int(ln.split()[3]))
                  for ln in out.splitlines() if ln.strip().startswith(">=")}
        assert floors["3"] == (3, 13), floors        # 903:3 + 904:4 + 905:6 scans
        assert "906, 907" in out
        rows = list(csv.DictReader(open(tmp / "c.csv")))
        assert {r["bids_label"]: int(r["n_scans"]) for r in rows}["903"] == 3

        # dry run writes nothing; --write-lists with floor 3 keeps 903, 904, 905
        r = subprocess.run(base + ["--min-scans", "3"], env=env, capture_output=True, text=True)
        assert "dry run" in r.stdout and not (tmp / "subjects.txt").exists()
        r = subprocess.run(base + ["--min-scans", "3", "--write-lists"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (tmp / "subjects.txt").read_text().split() == ["10003", "10004", "10005"]
        lst = (tmp / "fmriprep_lists" / "sub-10005" / "subject_level_files.txt").read_text().split()
        assert len(lst) == 6 and lst == sorted(lst) and all("removed_GS" in p for p in lst)
        assert "1-3" in r.stdout
        # refuses to overwrite
        r = subprocess.run(base + ["--min-scans", "3", "--write-lists"], env=env,
                           capture_output=True, text=True)
        assert r.returncode != 0 and "--overwrite" in r.stderr
    print("test_scan_counts: all checks passed")


if __name__ == "__main__":
    main()
