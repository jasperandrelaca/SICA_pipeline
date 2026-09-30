#!/usr/bin/env python3
"""
P3_count_scans_build_lists.py -- how many denoised scans does each participant have,
and which participants/runs enter the analysis?

Run after P2. Scans <fmriprep_out>/sub-*/ for the final denoised runs of the chosen
global-signal variant
    *_desc-preproc_removed_GS_smoothed_filtered_demeaned_masked.nii.gz   (--gsr y)
    *_desc-preproc_with_GS_smoothed_filtered_demeaned_masked.nii.gz      (--gsr n)
and counts one scan per file.

1. REPORT (always). Prints how many participants have 1, 2, 3 and 4+ scans, and how
   many participants and scans remain at each possible minimum ("floor"), so that
   you can choose an inclusion floor. Optionally lists participants with zero scans
   (--subject-list), which usually means a failed or never-submitted job.
2. BUILD (only with --write-lists and --min-scans N). Keeps participants with at
   least N scans and writes
     - $CATD_FMRIPREP_LIST_DIR/sub-<ID>/subject_level_files.txt  (one run per line,
       sorted by path; this order is used by MELODIC AND dual regression)
     - the participant list for steps 01-02 (default $CATD_ROOT/subjects.txt)
   The study used a floor of 3 scans.

Identifiers: fMRIPrep folders carry BIDS labels (sub-<label>). If the identifiers
used from step 01 on are different, pass --id-map, a CSV with columns
bids_label,study_id. Without it the BIDS label is used as the ID.

Usage
  python P3_count_scans_build_lists.py --gsr y                         # report only
  python P3_count_scans_build_lists.py --gsr y --min-scans 3 --write-lists
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catd_config as cfg  # noqa: E402

FINAL_SUFFIX = "_smoothed_filtered_demeaned_masked.nii.gz"


def final_pattern(gsr):
    tag = "removed_GS" if gsr == "y" else "with_GS"
    return f"*_desc-preproc_{tag}{FINAL_SUFFIX}"


def scan_participants(fmriprep_dir, gsr):
    """{bids_label: [sorted list of final denoised run paths]} (non-empty files only)."""
    out = {}
    for sub in sorted(p for p in Path(fmriprep_dir).glob("sub-*") if p.is_dir()):
        runs = sorted(str(f.resolve()) for f in sub.rglob(final_pattern(gsr))
                      if f.is_file() and f.stat().st_size > 0)
        out[sub.name[len("sub-"):]] = runs
    return out


def _strip(label):
    label = label.strip()
    return label[4:] if label.startswith("sub-") else label


def read_labels(path):
    with open(path) as fh:
        return [_strip(ln) for ln in fh if ln.strip()]


def read_id_map(path):
    with open(path, newline="") as fh:
        return {_strip(r["bids_label"]): r["study_id"].strip()
                for r in csv.DictReader(fh)}


def bin_label(n):
    return "4+" if n >= 4 else str(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsr", choices=["y", "n"], required=True,
                    help="which denoised variant to count: y = removed_GS (the study), n = with_GS")
    ap.add_argument("--fmriprep-dir", type=Path, default=None,
                    help="fMRIPrep output directory (default: $CATD_FMRIPREP_OUT_DIR)")
    ap.add_argument("--subject-list", type=Path, default=None,
                    help="BIDS labels that were preprocessed (P1 list); participants "
                         "with zero scans are reported")
    ap.add_argument("--id-map", type=Path, default=None,
                    help="CSV with columns bids_label,study_id")
    ap.add_argument("--min-scans", type=int, default=None,
                    help="inclusion floor: minimum number of scans per participant")
    ap.add_argument("--write-lists", action="store_true",
                    help="write subject_level_files.txt and the participant list "
                         "(requires --min-scans)")
    ap.add_argument("--list-dir", type=Path, default=None,
                    help="default: $CATD_FMRIPREP_LIST_DIR")
    ap.add_argument("--out-subjects", type=Path, default=None,
                    help="participant list for steps 01-02 (default: $CATD_ROOT/subjects.txt)")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="per-participant scan counts (default: not written)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.write_lists and args.min_scans is None:
        ap.error("--write-lists requires --min-scans")

    fmriprep_dir = args.fmriprep_dir or cfg.fmriprep_out_dir()
    if not Path(fmriprep_dir).is_dir():
        sys.exit(f"ERROR: fMRIPrep directory not found: {fmriprep_dir}")
    scans = scan_participants(fmriprep_dir, args.gsr)
    if not scans:
        sys.exit(f"ERROR: no sub-* folders in {fmriprep_dir}")

    if args.subject_list:
        for label in read_labels(args.subject_list):
            scans.setdefault(label, [])
    id_map = read_id_map(args.id_map) if args.id_map else {}
    if not args.id_map:
        print("NOTE: no --id-map given; BIDS labels are used as participant IDs.")
    unmapped = [b for b in scans if id_map and b not in id_map]
    if unmapped:
        sys.exit(f"ERROR: {len(unmapped)} label(s) missing from --id-map: {unmapped[:10]}")

    counts = {b: len(r) for b, r in scans.items()}
    variant = "removed_GS" if args.gsr == "y" else "with_GS"
    print(f"\nDenoised scans found ({variant}) in {fmriprep_dir}")
    print(f"Participants examined: {len(counts)}\n")

    dist = Counter(bin_label(n) for n in counts.values())
    print("Scans per participant   participants")
    for k in ("0", "1", "2", "3", "4+"):
        if k == "0" and not dist["0"]:
            continue
        print(f"  {k:<20}  {dist[k]:>5}")
    if dist["0"]:
        zero = sorted(b for b, n in counts.items() if n == 0)
        print(f"\n  {len(zero)} participant(s) with 0 scans (failed or unsubmitted P1/P2 job?): "
              f"{', '.join(zero)}")

    print("\nIf the minimum number of scans per participant were...")
    print("  floor   participants kept   scans kept")
    for floor in (1, 2, 3, 4):
        kept = [n for n in counts.values() if n >= floor]
        print(f"  >= {floor}    {len(kept):>10}          {sum(kept):>6}")

    rows = []
    for b in sorted(scans):
        rows.append({"bids_label": b, "study_id": id_map.get(b, b), "n_scans": counts[b],
                     "n_sessions": len({cfg.parse_session(r) for r in scans[b]}),
                     "included": (args.min_scans is not None and counts[b] >= args.min_scans)})
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-participant counts written to {args.out_csv}")

    if args.min_scans is None:
        print("\nChoose a floor and re-run with --min-scans N --write-lists to build the lists.")
        return

    keep = [r for r in rows if r["included"]]
    print(f"\nFloor = {args.min_scans} scans: {len(keep)} of {len(rows)} participants included, "
          f"{sum(r['n_scans'] for r in keep)} scans")
    if not args.write_lists:
        print("(dry run: add --write-lists to write the run lists)")
        return

    list_dir = Path(args.list_dir) if args.list_dir else cfg.fmriprep_list_dir()
    subjects_file = (Path(args.out_subjects) if args.out_subjects
                     else cfg._root() / "subjects.txt")
    targets = [list_dir / f"sub-{r['study_id']}" / "subject_level_files.txt" for r in keep]
    existing = [t for t in targets + [subjects_file] if t.exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: {len(existing)} output file(s) already exist (e.g. {existing[0]}); "
                 "use --overwrite")
    for r, target in zip(keep, targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(scans[r["bids_label"]]) + "\n")
    subjects_file.parent.mkdir(parents=True, exist_ok=True)
    subjects_file.write_text("\n".join(str(r["study_id"]) for r in keep) + "\n")
    print(f"Run lists written under {list_dir}; participant list ({len(keep)} lines) "
          f"written to {subjects_file}")
    print(f"Array range for steps 01-02: 1-{len(keep)}")


if __name__ == "__main__":
    main()
