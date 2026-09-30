#!/usr/bin/env python3
"""Test for scripts/diagnostics/diagnose_min_dice_floor.py on hand-built NCT tables.
Run:  python tests/test_diagnose_floor.py"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnostics" / "diagnose_min_dice_floor.py"


def write_ic(nct, sub, ic, dmn_dice):
    """One IC table in which only the DMN labels carry signal."""
    rows = [("TY7", "Default", dmn_dice, 0.001), ("TY17", "DefaultA", dmn_dice, 0.001),
            ("TY17", "DefaultB", 0.0, 0.9), ("AS400Y17", "DefaultA", dmn_dice, 0.001),
            ("EG17", "Default", dmn_dice, 0.001), ("TY7", "Limbic", 0.05, 0.5)]
    d = nct / f"sub-{sub}" / f"sub-{sub}_IC{ic:02d}"
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=["group", "name", "dice", "p_value"]).to_csv(d / "network_correspondence.csv")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        nct = Path(tmp) / "nct"
        # subject 10001: winner 0.50, secondaries 0.32 and 0.35, one IC below the 0.3 gate
        for ic, d in enumerate([0.50, 0.32, 0.35, 0.10]):
            write_ic(nct, 10001, ic, d)
        # subject 10002: a single candidate -> winner only
        for ic, d in enumerate([0.45, 0.05]):
            write_ic(nct, 10002, ic, d)
        prefix = Path(tmp) / "out"
        r = subprocess.run([sys.executable, str(SCRIPT), "--nct-dir", str(nct), "--floors", "0.3",
                            "0.4", "--out-prefix", str(prefix)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        win = pd.read_csv(f"{prefix}_winners.csv")
        sec = pd.read_csv(f"{prefix}_secondary.csv")
        dmn_w, dmn_s = win[win.network == "DMN"], sec[sec.network == "DMN"]
        assert sorted(dmn_w.ic_num) == [0, 0] and len(dmn_w) == 2
        assert sorted(dmn_s.ic_num) == [1, 2], dmn_s          # ICs 1 and 2, not the sub-gate IC 3
        assert "DMN: 2 secondary candidates across 1 subjects" in r.stdout
        # floor 0.4 removes both secondaries (100%) and no winner (0.45, 0.50 stay)
        assert "min_dice=0.4  :   0 secondary candidates remain (2 removed, 100.0%)" in r.stdout
        assert "min_dice=0.4  : 0 winner(s) would be lost" in r.stdout
        # gate is the repository's, not a local copy
        assert "'DMN': 0.3" in r.stdout
    print("test_diagnose_floor: all checks passed")


if __name__ == "__main__":
    main()
