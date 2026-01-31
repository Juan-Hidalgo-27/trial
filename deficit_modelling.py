"""
deficit_modelling.py - DEBUG version

Deficit modelling for stroke lesion analysis.
This version includes debugging parameters for testing with small datasets.

Usage:
    # Normal run
    python deficit_modelling.py --lesionpath /path/to/lesions --discopath /path/to/disconnectomes

    # DEBUG run with 10 samples
    python deficit_modelling.py --lesionpath /path/to/lesions --n_samples 10 --verbose True
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm, trange


def str2bool(v):
    """Parse boolean from string for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


deficits = [
    "hearing",
    "language",
    "introspection",
    "cognition",
    "mood",
    "memory",
    "aversion",
    "coordination",
    "interoception",
    "sleep",
    "reward",
    "visual recognition",
    "visual perception",
    "spatial reasoning",
    "motor",
    "somatosensory",
]


class deficit_inference:
    def __init__(
        self,
        roipath,
        input_type,
        roi_thresh,
        images_loaded,
        disco_thresh=0.5,
        verbose=False,
    ):
        self.roipath = roipath
        self.input_type = input_type
        self.roi_thresh = roi_thresh
        self.images_loaded = images_loaded
        self.disco_thresh = disco_thresh
        self.verbose = verbose

        self.roipairs = {}
        rois = os.listdir(self.roipath)

        if self.verbose:
            print(f"Loading ROI pairs from: {self.roipath}")
            print(f"Found {len(rois)} files")

        for roipair in range(1, 17):
            roipair_str = str(roipair) + "_"
            file = False
            for roi in rois:
                if roi.startswith(roipair_str):
                    if "nii" in roi:
                        file = roi
            if file:
                roi_pair = nib.load(os.path.join(roipath, file)).get_fdata()
                if self.verbose:
                    print(f"  Loaded ROI {roipair}: {file}")
            else:
                if self.verbose:
                    print(f"    ROI {roipair} not found, skipping...")
                continue

            self.roipairs[roipair] = roi_pair

        self.deficits = deficits

    def find_deficits(self, df):
        n_deficits = len(self.roipairs)

        if self.verbose:
            print(
                f"\nFinding deficits for {len(df)} samples using {n_deficits} ROI pairs"
            )

        preallocate_regionmasks = {}
        for roipair in self.roipairs.keys():
            roi_pair = self.roipairs[roipair]

            # Create binary masks for each ROI within the pair.
            roi1 = (roi_pair == 1).astype(int)
            roi2 = (roi_pair == 2).astype(int)
            regionmasks = [roi1, roi2]

            # Now compute overlap between each lesion and the ROI mask. Exceeding a
            # threshold by proportion of the ROI will register is a 'hit' and the
            # associated deficit can therefore be simulated.
            roi_vols = [
                np.sum(regionmasks[region]) for region in range(len(regionmasks))
            ]
            treatments = ["1", "0"] if np.argmax(roi_vols) == 0 else ["0", "1"]

            inds1 = np.where(roi1 == 1)
            centroid1_x = inds1[0].mean() if len(inds1[0]) > 0 else np.nan
            centroid1_y = inds1[1].mean() if len(inds1[0]) > 0 else np.nan
            centroid1_z = inds1[2].mean() if len(inds1[0]) > 0 else np.nan

            inds2 = np.where(roi2 == 1)
            centroid2_x = inds2[0].mean() if len(inds2[0]) > 0 else np.nan
            centroid2_y = inds2[1].mean() if len(inds2[0]) > 0 else np.nan
            centroid2_z = inds2[2].mean() if len(inds2[0]) > 0 else np.nan

            roi_centroids_x = [centroid1_x, centroid2_x]
            roi_centroids_y = [centroid1_y, centroid2_y]
            roi_centroids_z = [centroid1_z, centroid2_z]

            preallocate_regionmasks[roipair] = [
                regionmasks,
                roi_vols,
                treatments,
                (roi_centroids_x, roi_centroids_y, roi_centroids_z),
            ]

        df.reset_index(inplace=True, drop=True)

        desc = "Finding deficits" if not self.verbose else "Finding deficits (verbose)"
        for i in trange(len(df), desc=desc):
            filename = df["filename"].iloc[i]

            # Check if image is loaded
            if filename not in self.images_loaded:
                if self.verbose:
                    print(f"    Image not found: {filename}")
                continue

            img = self.images_loaded[filename]

            for roipair in preallocate_regionmasks.keys():
                regionmasks, roi_vols, treatments, centroids = preallocate_regionmasks[
                    roipair
                ]

                # Handle case where roipair index exceeds deficits list
                if roipair - 1 >= len(self.deficits):
                    continue

                for region, vol, treatment in zip(regionmasks, roi_vols, treatments):
                    treatment_susceptibility = (
                        f"{self.deficits[roipair - 1]}_W{treatment}"
                    )
                    thresh = int(np.round(vol * self.roi_thresh))
                    if (
                        np.sum(((img > self.disco_thresh).astype(int) * region))
                        > thresh
                    ):
                        df.loc[i, treatment_susceptibility] = 1
                    else:
                        df.loc[i, treatment_susceptibility] = 0

        return df, preallocate_regionmasks


def harmonize_columns(
    main_df,
    other_dfs_path,
    latent_list,
    train_or_test,
    kf_count,
    input_type,
    reductions,
    verbose=False,
):
    """Harmonize columns from different latent dimension files."""

    main_df.reset_index(inplace=True, drop=True)

    if verbose:
        print(f"\nHarmonizing columns for {len(main_df)} samples")
        print(f"  Latent dimensions: {latent_list}")
        print(f"  Reductions: {reductions}")

    # Load other latent dimension files
    other_latent_dfs = {}
    for dim in latent_list[1:]:
        df = get_file(
            other_dfs_path, f"{train_or_test}_{kf_count}_dim_{dim}_{input_type}"
        )
        if df is not None:
            other_latent_dfs[dim] = df
            if verbose:
                print(f"  Loaded dim {dim}: {len(df)} samples")
        else:
            if verbose:
                print(f"    File not found for dim {dim}")

    other_reductions = {REDUCTION: pd.DataFrame() for REDUCTION in reductions}

    for i in trange(len(main_df), desc="Harmonizing columns"):
        img_name = main_df["filename"].iloc[i]
        for dim in latent_list[1:]:
            if dim not in other_latent_dfs:
                continue
            gt_train_dim = other_latent_dfs[dim]
            include = gt_train_dim.loc[gt_train_dim["filename"] == img_name]
            for REDUCTION in reductions:
                col_name = f"{REDUCTION}_{input_type}_{dim}_K{kf_count}"
                if len(include) and col_name in include.columns:
                    other_reductions[REDUCTION] = pd.concat(
                        [
                            other_reductions[REDUCTION],
                            pd.DataFrame(
                                {col_name: [include[col_name].item()]},
                                index=[i],
                            ),
                        ]
                    )
                else:
                    if verbose and i == 0:
                        print(f"      Column {col_name} not found")
                    other_reductions[REDUCTION] = pd.concat(
                        [
                            other_reductions[REDUCTION],
                            pd.DataFrame(
                                {f"{REDUCTION}_{input_type}_{dim}_{kf_count}": np.nan},
                                index=[i],
                            ),
                        ]
                    )

    for REDUCTION in reductions:
        for col in other_reductions[REDUCTION].columns:
            valid_mask = other_reductions[REDUCTION][col].isna() == False
            main_df[col] = other_reductions[REDUCTION].loc[valid_mask][col]

    return main_df


def get_file(path, name):
    """Load pickle file by name prefix."""
    if not os.path.exists(path):
        return None
    loadfiles = os.listdir(path)
    filetoload = None
    for file in loadfiles:
        if file.startswith(name):
            filetoload = file
            break
    return pd.read_pickle(os.path.join(path, filetoload)) if filetoload else None


def run(parameters):
    """Main run function."""
    paths, ground_truth, reductions, debug_params = parameters
    (lesionpath, discopath, path) = paths
    (latent_list, kfold_deficits, roi_threshs, names) = ground_truth
    (run_ae, run_vae, run_nmf, run_pca) = reductions
    (n_samples, verbose) = debug_params

    print(f"\n{'#'*60}")
    print("# DEFICIT MODELLING PIPELINE")
    print(f"{'#'*60}")

    if n_samples:
        print(f"  DEBUG MODE: Processing only {n_samples} samples")
    if verbose:
        print(f"Verbose output enabled")

    input_types = []
    if lesionpath:
        input_types.append("lesion")
    if discopath:
        input_types.append("disco")

    if verbose:
        print(f"\nInput types: {input_types}")
        print(f"Latent list: {latent_list}")
        print(f"K-folds: {kfold_deficits}")
        print(f"ROI thresholds: {roi_threshs}")
        print(f"Names: {names}")

    reductions_list = []
    if run_vae:
        reductions_list.append("vae_means")
    if run_ae:
        reductions_list.append("ae")
    if run_nmf:
        reductions_list.append("nmf")
    if run_pca:
        reductions_list.append("pca")

    if verbose:
        print(f"Reductions: {reductions_list}")

    roi_stem = os.path.join(os.getcwd(), "atlases", "2mm_parcellations")

    # Check if roi_stem exists, try alternative paths
    if not os.path.exists(roi_stem):
        alt_paths = [
            os.path.join(os.getcwd(), "../../atlases", "2mm_parcellations"),
            os.path.join(path, "atlases", "2mm_parcellations"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                roi_stem = alt
                break

    if verbose:
        print(f"ROI stem path: {roi_stem}")
        print(f"ROI stem exists: {os.path.exists(roi_stem)}")

    roipaths = [os.path.join(roi_stem, name) for name in names]

    for roi_thresh in roi_threshs:
        for input_type in input_types:
            print(f"\n{'='*60}")
            print(f"Processing: {input_type}, ROI threshold: {roi_thresh}")
            print(f"{'='*60}")

            image_path = lesionpath if input_type == "lesion" else discopath

            if not os.path.exists(image_path):
                print(f"  Image path does not exist: {image_path}")
                continue

            images_list = os.listdir(image_path)
            images_list = [f for f in images_list if "nii" in f]

            # DEBUG: Limit samples
            if n_samples is not None and n_samples < len(images_list):
                images_list = images_list[:n_samples]
                print(f"  DEBUG: Limited to {n_samples} images")

            print(f"Loading {len(images_list)} images...")
            images_loaded = {
                filename: nib.load(os.path.join(image_path, filename)).get_fdata()
                for filename in tqdm(images_list, desc="Loading images")
            }

            for kf_count in range(kfold_deficits):
                print(f"\n--- Fold {kf_count + 1}/{kfold_deficits} ---")

                for roipath, name in zip(roipaths, names):
                    if not os.path.exists(roipath):
                        print(f"  ROI path does not exist: {roipath}")
                        continue

                    print(f"Processing ROI: {name}")

                    generator = deficit_inference(
                        roipath, input_type, roi_thresh, images_loaded, verbose=verbose
                    )

                    savepath = os.path.join(
                        path, f"{input_type}_{roi_thresh}_{name}_{kf_count}"
                    )
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)

                    # Load train data
                    ground_truth_train = get_file(
                        path, f"train_{kf_count}_dim_{latent_list[0]}_{input_type}"
                    )

                    if ground_truth_train is None:
                        # Try alternative naming
                        ground_truth_train = get_file(path, f"train_split_{kf_count}")
                        if verbose:
                            print(f"  Trying alternative: train_split_{kf_count}")

                    if ground_truth_train is None:
                        print(f"    Train file not found for fold {kf_count}")
                        continue

                    # DEBUG: Limit samples in dataframe
                    if n_samples is not None and len(ground_truth_train) > n_samples:
                        ground_truth_train = ground_truth_train.head(n_samples)
                        if verbose:
                            print(f"  DEBUG: Limited train to {n_samples} samples")

                    print(f"  Train samples: {len(ground_truth_train)}")

                    ground_truth_train, susceptibility_information_train = (
                        generator.find_deficits(ground_truth_train)
                    )

                    if len(latent_list) > 1 and len(reductions_list) > 0:
                        ground_truth_train = harmonize_columns(
                            ground_truth_train,
                            path,
                            latent_list,
                            "train",
                            kf_count,
                            input_type,
                            reductions_list,
                            verbose=verbose,
                        )

                    ground_truth_train.to_pickle(
                        os.path.join(
                            savepath, f"train_{input_type}_{roi_thresh}_{kf_count}.pkl"
                        )
                    )

                    # Save centroids
                    centroids = {
                        roi: susceptibility_information_train[roi][-1]
                        for roi in susceptibility_information_train.keys()
                    }
                    pd.DataFrame(centroids).to_json(
                        os.path.join(savepath, "centroids.json")
                    )

                    del ground_truth_train

                    # Load test data
                    ground_truth_test = get_file(
                        path, f"test_{kf_count}_dim_{latent_list[0]}_['{input_type}']"
                    )

                    if ground_truth_test is None:
                        ground_truth_test = get_file(path, f"test_split_{kf_count}")

                    if ground_truth_test is None:
                        print(f"    Test file not found for fold {kf_count}")
                        continue

                    # DEBUG: Limit samples
                    if n_samples is not None and len(ground_truth_test) > n_samples:
                        ground_truth_test = ground_truth_test.head(n_samples)

                    print(f"  Test samples: {len(ground_truth_test)}")

                    ground_truth_test, _ = generator.find_deficits(ground_truth_test)

                    if len(latent_list) > 1 and len(reductions_list) > 0:
                        ground_truth_test = harmonize_columns(
                            ground_truth_test,
                            path,
                            latent_list,
                            "test",
                            kf_count,
                            input_type,
                            reductions_list,
                            verbose=verbose,
                        )

                    ground_truth_test.to_pickle(
                        os.path.join(
                            savepath, f"test_{input_type}_{roi_thresh}_{kf_count}.pkl"
                        )
                    )
                    del ground_truth_test

    print("\n Deficit modelling complete!")


def command_line_options():
    parser = argparse.ArgumentParser(
        description="Deficit Modelling for Stroke Lesion Analysis"
    )
    parser.add_argument("--path", type=str, default="", help="Base path for results")
    parser.add_argument(
        "--lesionpath", type=str, default="", help="Path to lesion nii files"
    )
    parser.add_argument(
        "--discopath", type=str, default="", help="Path to disconnectome nii files"
    )

    parser.add_argument(
        "--latent_list",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128, 256],
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--kfold_deficits", type=int, default=10, help="K-fold crossval"
    )
    parser.add_argument(
        "--roi_threshs",
        type=float,
        nargs="+",
        default=[0.05],
        help="Threshold for susceptibility modelling",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=["genetics", "receptor"],
        help="Ground truth rationales",
    )

    parser.add_argument(
        "--run_ae", type=str2bool, default=False, help="Run autoencoder"
    )
    parser.add_argument(
        "--run_vae", type=str2bool, default=False, help="Run variational autoencoder"
    )
    parser.add_argument(
        "--run_nmf",
        type=str2bool,
        default=True,
        help="Run non-negative matrix factorisation",
    )
    parser.add_argument(
        "--run_pca",
        type=str2bool,
        default=True,
        help="Run principal component analysis",
    )

    # DEBUG parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="DEBUG: Limit number of samples to process",
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=False, help="DEBUG: Verbose output"
    )

    args = parser.parse_args()
    paths = (args.lesionpath, args.discopath, args.path)
    ground_truth = (args.latent_list, args.kfold_deficits, args.roi_threshs, args.names)
    reductions = (args.run_ae, args.run_vae, args.run_nmf, args.run_pca)
    debug_params = (args.n_samples, args.verbose)
    return (paths, ground_truth, reductions, debug_params)


if __name__ == "__main__":
    parameters = command_line_options()
    run(parameters)
