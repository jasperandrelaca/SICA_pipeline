#!/usr/bin/env python3
"""
03_nct_network_correspondence.py
================================
Compare every participant-specific MELODIC component with canonical
resting-state network atlases using the CBIG Network Correspondence Toolbox
(NCT; Kong et al., 2025).

For each participant (one ``sub-XXXXX_subject_level_ICA.gica`` folder):

  1. Register the ICs from MNIPediatricAsym (the space of our data) to
     FSL/MNI152 2 mm space, which is what NCT expects. The registration is a
     template-to-template ANTs transform (pediatric template -> adult MNI
     template) applied to the 4-D melodic_IC.nii.gz.
  2. Split the 4-D IC file into one 3-D volume per component (NCT does not
     handle 4-D input).
  3. Threshold each component (default: z > 2.5) and run NCT against the atlases
     in ``--atlases``; NCT computes Dice overlap and a spin/permutation p-value
     for every atlas network.

Output (per participant)::

    {output_dir}/sub-XXXXX/
        sub-XXXXX_melodic_IC_FSLMNI2mm.nii.gz     registered 4-D ICs
        ica_components_3d/sub-XXXXX_ICnn.nii.gz   split 3-D components
        sub-XXXXX_ICnn/network_correspondence.csv NCT result for component nn
        sub-XXXXX_all_components_dice.tsv         (only if NCT wrote *summary*.csv)

Component numbering is 0-based (IC00 ... IC24 for 25 components) and matches the
volume index in melodic_IC.nii.gz and in the dual-regression stage-2 maps.

Requirements: cbig_network_correspondence, antspyx, templateflow, nibabel,
numpy, pandas, matplotlib (see requirements.txt).

Example (process participants 0-9 of the sorted list; run further batches to
stay within memory limits)::

    python scripts/03_nct_network_correspondence.py --start 0 --end 10
"""

import os
import sys

# --- Environment tweaks: must be set BEFORE importing cbig_network_correspondence ---
# Prevent NCT from trying to auto-update (git lock contention in parallel jobs).
os.environ["CBIG_NCT_NO_AUTO_UPDATE"] = "1"
# Some clusters lack a CA bundle in the default location.
if (not os.path.exists("/etc/ssl/certs/ca-certificates.crt")
        and os.path.exists("/etc/pki/tls/certs/ca-bundle.crt")):
    os.environ["GIT_SSL_CAINFO"] = "/etc/pki/tls/certs/ca-bundle.crt"
# Last-resort workaround for clusters with broken SSL interception. This disables
# certificate verification for git, so it is OFF unless explicitly requested:
#     export NCT_DISABLE_GIT_SSL_VERIFY=1
if os.environ.get("NCT_DISABLE_GIT_SSL_VERIFY") == "1":
    os.environ["GIT_SSL_NO_VERIFY"] = "true"

import argparse
import gc
import traceback
from pathlib import Path

import ants
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (lower memory, no display needed)
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from templateflow.api import get
import cbig_network_correspondence as cnc

import catd_config

plt.rcParams["figure.max_open_warning"] = 0

# NCT's "FSLMNI2mm" space is 91 x 109 x 91 voxels.
EXPECTED_SHAPE = (91, 109, 91)

# Atlases used in this study (NCT identifiers).
DEFAULT_ATLASES = ["TY7", "TY17", "AS400Y17", "EG17"]

AGGRESSIVE_MEMORY_CLEANUP = True


# ----------------------------------------------------------------------------
# NCT permutation-test patch
# ----------------------------------------------------------------------------
def patch_permutations(disable: bool, n_permutations: int):
    """
    Monkey-patch NCT's permutation routine.

    disable=True            -> skip permutation testing (no p-values; least memory)
    disable=False           -> force ``n_permutations`` permutations by overwriting
                               the 5th positional argument (n_perm) of
                               ``permute_overlap_data_all``.

    NOTE: the positional-argument index is an assumption about the NCT version
    used for this study; if you upgrade NCT, check the signature of
    ``compute_overlap_with_atlases.permute_overlap_data_all``.
    """
    module = cnc.compute_overlap_with_atlases

    if disable:
        def dummy_permute(*args, **kwargs):
            print("      [Skipping permutation test - memory saving mode]")
            return {}
        module.permute_overlap_data_all = dummy_permute
        print("[OK] NCT patched to skip permutation testing (no p-values will be computed)\n")
        return

    original_permute = module.permute_overlap_data_all

    def patched_permute(*args, **kwargs):
        args_list = list(args)
        if len(args_list) >= 5:
            original_n_perm = args_list[4]
            args_list[4] = n_permutations
            print(f"      [Running {n_permutations} permutations (NCT default was {original_n_perm})]")
        else:
            print("      [WARNING: could not locate n_perm argument; NCT default used]")
        return original_permute(*args_list, **kwargs)

    module.permute_overlap_data_all = patched_permute
    print(f"[OK] NCT permutation testing enabled with {n_permutations} permutations\n")


def cleanup_memory():
    """Close matplotlib figures and force garbage collection."""
    if AGGRESSIVE_MEMORY_CLEANUP:
        plt.close("all")
        gc.collect()
        print("    [memory cleanup]", end=" ")


# ----------------------------------------------------------------------------
# Atlas helpers
# ----------------------------------------------------------------------------
def get_available_atlases():
    """Atlas names available in the installed NCT data directory (or None)."""
    try:
        atlas_config_dir = Path(cnc.__file__).parent / "data" / "atlas_config"
        if atlas_config_dir.exists():
            return sorted(f.name for f in atlas_config_dir.iterdir()
                          if f.is_file() and not f.name.startswith("."))
        print(f"  [WARN] Atlas config directory not found: {atlas_config_dir}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Could not retrieve atlas list: {exc}")
        return None


def validate_atlas_names(requested):
    """Split ``requested`` into (valid, invalid) against the NCT installation."""
    available = get_available_atlases()
    if available is None:
        print("  [WARN] Cannot validate atlas names - NCT data not downloaded yet")
        return list(requested), []
    valid = [a for a in requested if a in available]
    invalid = [a for a in requested if a not in available]
    return valid, invalid


def print_available_atlases():
    atlases = get_available_atlases()
    print("\n" + "=" * 60)
    if atlases:
        print(f"Available atlases in NCT ({len(atlases)} total):")
        for atlas in atlases:
            print(f"  {atlas}")
    else:
        print("Atlas data not yet downloaded (NCT downloads it on first run).")
    print("=" * 60 + "\n")


# ----------------------------------------------------------------------------
# Templates and registration
# ----------------------------------------------------------------------------
def load_templates():
    """
    Load the registration target (adult MNI) and source (MNIPediatricAsym) templates.

    Target : MNI152NLin6Asym, 2 mm, T1w brain (FSL's standard MNI152 space).
             Falls back to MNI152NLin2009cAsym if unavailable.
    Source : MNIPediatricAsym cohort-1, 2 mm, T1w (the space of our data).

    Returns (target_template, source_template) as ANTs images.
    """
    print("  Loading templates from TemplateFlow...")

    try:
        target_path = get("MNI152NLin6Asym", resolution=2, suffix="T1w", desc="brain")
        target = ants.image_read(str(target_path))
        print("  [OK] Target template: MNI152NLin6Asym")
    except Exception:  # noqa: BLE001
        print("  [WARN] MNI152NLin6Asym unavailable; trying MNI152NLin2009cAsym...")
        target_path = get("MNI152NLin2009cAsym", resolution=2, suffix="T1w", desc=None)
        target = ants.image_read(str(target_path))
        print("  [OK] Target template: MNI152NLin2009cAsym")
    print(f"    Dimensions: {target.shape}")

    source_path = get("MNIPediatricAsym", cohort=1, resolution=2, suffix="T1w")
    source = ants.image_read(str(source_path))
    print("  [OK] Source template: MNIPediatricAsym cohort-1")
    print(f"    Dimensions: {source.shape}")

    if target.shape != EXPECTED_SHAPE:
        print(f"\n  [WARN] Target template shape {target.shape} != NCT-expected {EXPECTED_SHAPE};")
        print("         registered ICs will be resampled to the expected voxel dimensions.")

    return target, source


def register_ica_to_mni(ica_path, output_path, target_template, source_template,
                        transform_type="SyN"):
    """
    Bring a 4-D IC file from MNIPediatricAsym space into adult-MNI / NCT space.

      1. Register source_template (pediatric) -> target_template (adult MNI).
      2. Apply that transform to every IC volume (linear interpolation,
         time-series mode).
      3. If the result is not 91x109x91, resample it to those voxel dimensions
         (see docs/KNOWN_ISSUES.md, item 2, for a caveat about this step).

    Returns True on success, False otherwise.
    """
    try:
        print(f"  Registering {os.path.basename(ica_path)} (MNIPediatricAsym -> MNI152), "
              f"transform: {transform_type}")

        ica_img = ants.image_read(ica_path)
        if ica_img.dimension == 4:
            print(f"  Detected 4-D data (shape: {ica_img.shape})")

        print("  Computing template registration...")
        reg = ants.registration(
            fixed=target_template,
            moving=source_template,
            type_of_transform=transform_type,
        )
        print("  [OK] Template registration completed")

        warped = ants.apply_transforms(
            fixed=target_template,
            moving=ica_img,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",   # continuous-valued maps
            imagetype=3,             # 3 = time series (4-D)
        )
        print(f"  Warped dimensions: {warped.shape}")

        if warped.shape[:3] != EXPECTED_SHAPE:
            print(f"  Resampling to NCT-expected dimensions: {EXPECTED_SHAPE}")
            resized = ants.resample_image(
                warped,
                resample_params=EXPECTED_SHAPE,
                use_voxels=True,
                interp_type=4,  # ANTs ResampleImage code 4 (B-spline per ANTsPy docs)
            )
            if resized.shape[:3] == EXPECTED_SHAPE:
                warped = resized
                print(f"  [OK] Resampled to {warped.shape}")
            else:
                print(f"  [WARN] Resampling gave {resized.shape[:3]}, not {EXPECTED_SHAPE}; "
                      "NCT may raise dimension errors")
        else:
            print(f"  [OK] Dimensions already match NCT expectations: {warped.shape[:3]}")

        warped.to_filename(output_path)
        print(f"  [OK] Registered ICs saved (final dimensions: {warped.shape})")
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] Registration failed: {exc}")
        traceback.print_exc()
        return False


def split_4d_to_3d(input_4d_path, output_dir, subject_id):
    """Split a 4-D IC file into ``{subject_id}_ICnn.nii.gz`` 3-D volumes (0-based nn)."""
    print("  Splitting 4-D ICs into 3-D components...")
    img_4d = nib.load(input_4d_path)
    data_4d = img_4d.get_fdata()
    n_components = data_4d.shape[3]
    print(f"  Found {n_components} components")

    split_dir = os.path.join(output_dir, "ica_components_3d")
    os.makedirs(split_dir, exist_ok=True)

    paths = []
    for i in range(n_components):
        img_3d = nib.Nifti1Image(data_4d[..., i], img_4d.affine, img_4d.header)
        out = os.path.join(split_dir, f"{subject_id}_IC{i:02d}.nii.gz")
        nib.save(img_3d, out)
        paths.append(out)

    print(f"  [OK] Split into {len(paths)} 3-D volumes")
    return paths


# ----------------------------------------------------------------------------
# NCT
# ----------------------------------------------------------------------------
def create_config_file(name, output_dir, data_space="FSLMNI2mm", threshold=None):
    """
    Write the NCT ``config`` file for one component.

    threshold : [lower, upper] or None. None -> NCT default [0, Inf].
    """
    if threshold is None:
        threshold_str = "[0,Inf]"
    else:
        lower = threshold[0] if threshold[0] is not None else 0
        upper = "Inf" if threshold[1] is None or threshold[1] == float("inf") else threshold[1]
        threshold_str = f"[{lower},{upper}]"

    content = (
        "[data_info]\n"
        "Data_Category: UserData\n"
        f"Data_Name: {name}_ICA\n"
        f"Data_Space: {data_space}\n"
        "Data_Type: Soft\n"
        "Data_NetworkAssignment: 0\n"
        f"Data_Threshold: {threshold_str}\n"
    )
    config_path = os.path.join(output_dir, f"{name}_config.txt")
    with open(config_path, "w") as fh:
        fh.write(content)
    return config_path


def combine_component_results(results, output_dir, subject_id):
    """
    Concatenate any ``*summary*.csv`` files NCT wrote per component into one TSV.

    Depending on the NCT version there may be no such files; nothing downstream
    depends on this output (04_assign_networks.py reads network_correspondence.csv).
    """
    frames = []
    for result in results:
        if not result["success"]:
            continue
        for csv_file in Path(result["dir"]).glob("*summary*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df["Component"] = result["component"]
                frames.append(df)
            except Exception:  # noqa: BLE001
                continue
    if frames:
        out = os.path.join(output_dir, f"{subject_id}_all_components_dice.tsv")
        pd.concat(frames, ignore_index=True).to_csv(out, sep="\t", index=False)
        print(f"  [OK] Combined results: {out}")


def process_subject(subject_ica_path, output_base_dir, target_template, source_template,
                    atlas_list=None, threshold=None, skip_registration=False,
                    transform_type="SyN"):
    """Run the full register -> split -> NCT workflow for one participant."""
    subject_id = os.path.basename(os.path.dirname(subject_ica_path)).split("_")[0]  # sub-XXXXX

    print(f"\n{'=' * 60}\nProcessing subject: {subject_id}\n{'=' * 60}")
    subject_out = os.path.join(output_base_dir, subject_id)
    os.makedirs(subject_out, exist_ok=True)

    # Step 1: registration
    if not skip_registration:
        registered_path = os.path.join(subject_out, f"{subject_id}_melodic_IC_FSLMNI2mm.nii.gz")
        if not register_ica_to_mni(subject_ica_path, registered_path,
                                   target_template, source_template,
                                   transform_type=transform_type):
            print(f"  [FAIL] Registration failed for {subject_id}, skipping")
            return False
    else:
        registered_path = subject_ica_path
        print("  Skipping registration (assuming data already in FSL MNI 2 mm space)")

    # Step 2: split
    split_volumes = split_4d_to_3d(registered_path, subject_out, subject_id)

    # Atlas list
    atlas_list = atlas_list or DEFAULT_ATLASES
    valid_atlases, invalid_atlases = validate_atlas_names(atlas_list)
    if invalid_atlases:
        print(f"  [WARN] Ignoring unknown atlases: {invalid_atlases}")
    if not valid_atlases:
        print("  [FAIL] No valid atlases found")
        return False
    print(f"  Using atlases: {', '.join(valid_atlases)}")

    # Step 3: NCT per component
    print(f"\n  Running NCT on {len(split_volumes)} components individually...")
    results = []
    for idx, volume_path in enumerate(split_volumes):
        component = os.path.basename(volume_path).replace(".nii.gz", "")
        print(f"\n  [{idx + 1}/{len(split_volumes)}] {component}")
        try:
            config = create_config_file(f"{subject_id}_{component}", subject_out,
                                        threshold=threshold)
            comp_out = os.path.join(subject_out, component)
            os.makedirs(comp_out, exist_ok=True)

            params = cnc.compute_overlap_with_atlases.DataParams(config, volume_path)
            cnc.compute_overlap_with_atlases.network_correspondence(
                params, valid_atlases, comp_out
            )
            plt.close("all")  # NCT creates figures; close immediately
            print("    [OK] Done", end=" ")
            results.append({"component": component, "success": True, "dir": comp_out})
        except Exception as exc:  # noqa: BLE001
            print(f"    [FAIL] Error: {str(exc)[:80]}", end=" ")
            results.append({"component": component, "success": False})
        cleanup_memory()
        print()

    n_ok = sum(r["success"] for r in results)
    print(f"\n  Completed {n_ok}/{len(split_volumes)} components")
    if n_ok > 0:
        combine_component_results(results, subject_out, subject_id)
        cleanup_memory()
        return True
    return False


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ica-dir", default=None,
                        help="Directory containing sub-*_subject_level_ICA.gica folders "
                             "(default: $CATD_MELODIC_DIR)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: $CATD_NCT_RESULTS_DIR)")
    parser.add_argument("--start", type=int, default=None,
                        help="First participant index (0-based, inclusive) of the sorted list")
    parser.add_argument("--end", type=int, default=None,
                        help="Last participant index (exclusive)")
    parser.add_argument("--atlases", nargs="+", default=None,
                        help=f"NCT atlas identifiers (default: {' '.join(DEFAULT_ATLASES)})")
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="Lower z threshold applied to each IC by NCT; upper bound is Inf "
                             "(default: 2.5)")
    parser.add_argument("--transform", choices=["SyN", "Affine"], default="SyN",
                        help="ANTs transform type for pediatric -> adult MNI (default: SyN)")
    parser.add_argument("--skip-registration", action="store_true",
                        help="Assume ICs are already in FSL MNI 2 mm space")
    parser.add_argument("--no-cache-templates", action="store_true",
                        help="Reload templates for every participant")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of NCT permutations (default: 1000)")
    parser.add_argument("--no-permutations", action="store_true",
                        help="Skip permutation testing entirely (Dice only, no p-values)")
    parser.add_argument("--list-atlases", action="store_true",
                        help="Print the atlases available in the NCT install and exit")
    args = parser.parse_args()

    if args.list_atlases:
        print_available_atlases()
        return

    ica_parent = args.ica_dir or str(catd_config.melodic_dir())
    output_dir = args.output_dir or str(catd_config.nct_results_dir())
    threshold = [args.threshold, float("inf")]
    cache_templates = not args.no_cache_templates

    patch_permutations(args.no_permutations, args.n_permutations)

    os.makedirs(output_dir, exist_ok=True)

    all_subject_dirs = sorted(
        os.path.join(ica_parent, d) for d in os.listdir(ica_parent)
        if d.endswith("_subject_level_ICA.gica")
    )
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else len(all_subject_dirs)
        subject_dirs = all_subject_dirs[start:end]
        print(f"\n{'=' * 60}\nSubject range: indices {start}..{end - 1} "
              f"({len(subject_dirs)} of {len(all_subject_dirs)} participants)\n{'=' * 60}")
    else:
        subject_dirs = all_subject_dirs

    print(f"\n{'=' * 60}\nCBIG Network Correspondence Toolbox - ICA analysis")
    print("Registration: MNIPediatricAsym -> MNI152 (FSL 2 mm)\n" + "=" * 60)

    # Validate requested atlases up front
    atlas_list = args.atlases
    available = get_available_atlases()
    if available:
        print(f"[OK] Found {len(available)} atlases in NCT installation")
        if atlas_list:
            valid, invalid = validate_atlas_names(atlas_list)
            if invalid:
                print(f"\n[WARN] Unknown atlas names: {invalid}. Use --list-atlases to see valid names.")
            if not valid:
                print("\n[FAIL] None of the requested atlases are valid.")
                sys.exit(1)
            atlas_list = valid
    else:
        print("[WARN] NCT atlas data not yet downloaded; it will be fetched on first use")

    print(f"\n{'=' * 60}")
    print(f"Participants:      {len(subject_dirs)}")
    print(f"Output directory:  {output_dir}")
    print(f"NCT threshold:     [{threshold[0]}, Inf]")
    print(f"Skip registration: {args.skip_registration}")
    print(f"Transform type:    {args.transform}")
    print(f"Atlases:           {', '.join(atlas_list or DEFAULT_ATLASES)}")
    print("=" * 60 + "\n")

    target_template = source_template = None
    if not args.skip_registration and cache_templates:
        print("Loading templates (reused for all participants)...")
        target_template, source_template = load_templates()
        print()

    successful = failed = 0
    for subject_dir in subject_dirs:
        ica_file = os.path.join(subject_dir, "melodic_IC.nii.gz")
        if not os.path.exists(ica_file):
            print(f"\n[WARN] ICA file not found: {ica_file}")
            failed += 1
            continue

        if not args.skip_registration and not cache_templates:
            target_template, source_template = load_templates()

        ok = process_subject(
            ica_file, output_dir, target_template, source_template,
            atlas_list=atlas_list, threshold=threshold,
            skip_registration=args.skip_registration, transform_type=args.transform,
        )
        successful += int(ok)
        failed += int(not ok)

        print(f"  Cleaning up after {os.path.basename(subject_dir)}...")
        cleanup_memory()
        print()

    print(f"\n{'=' * 60}\nProcessing complete\n{'=' * 60}")
    print(f"Successful: {successful}\nFailed:     {failed}\nTotal:      {len(subject_dirs)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
