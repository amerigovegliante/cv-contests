import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_detector(cfg: dict):
    """Instantiate a feature detector from the config's 'detector' section."""
    det_cfg = cfg.get('detector', {})
    det_type = det_cfg.get('type', 'SIFT').upper()

    if det_type == 'SIFT':
        detector = cv2.SIFT_create(
            nfeatures=det_cfg.get('nfeatures', 0),
            nOctaveLayers=det_cfg.get('nOctaveLayers', 3),
            contrastThreshold=det_cfg.get('contrastThreshold', 0.04),
            edgeThreshold=det_cfg.get('edgeThreshold', 10),
            sigma=det_cfg.get('sigma', 1.6),
        )
    elif det_type == 'ORB':
        detector = cv2.ORB_create()
    elif det_type == 'AKAZE':
        detector = cv2.AKAZE_create()
    elif det_type == 'BRISK':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Unsupported detector type: '{det_type}'. "
                         "Choose from: SIFT, ORB, AKAZE, BRISK.")

    return detector, det_type


def save_results(results_path: str, cfg: dict, logo_names: list,
                 AP_all: np.ndarray) -> None:
    """Save AP results and the parameters used to a .txt file."""
    mAP = float(np.mean(AP_all))
    det_cfg = cfg.get('detector', {})
    match_cfg = cfg.get('matching', {})

    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("           LOGO CONTEST - RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # --- Results ---
        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write("Average Precision (AP):\n")
        for name, ap in zip(logo_names, AP_all):
            f.write(f"  {name:12s}  ==>  {ap:.4f}\n")
        f.write(f"\nMean Average Precision (mAP)  ==>  {mAP:.4f}\n\n")

        # --- Parameters ---
        f.write("PARAMETERS USED\n")
        f.write("-" * 40 + "\n")
        f.write(f"  detector.type             : {det_cfg.get('type', 'SIFT')}\n")
        if det_cfg.get('type', 'SIFT').upper() == 'SIFT':
            f.write(f"  detector.nfeatures        : {det_cfg.get('nfeatures', 0)}\n")
            f.write(f"  detector.nOctaveLayers    : {det_cfg.get('nOctaveLayers', 3)}\n")
            f.write(f"  detector.contrastThreshold: {det_cfg.get('contrastThreshold', 0.04)}\n")
            f.write(f"  detector.edgeThreshold    : {det_cfg.get('edgeThreshold', 10)}\n")
            f.write(f"  detector.sigma            : {det_cfg.get('sigma', 1.6)}\n")
        f.write(f"  matching.ratio_thresh     : {match_cfg.get('ratio_thresh', 0.75)}\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to: {results_path}")


def main():
    # -------------------------------------------------------------------------
    # Parse CLI arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Logo Contest - Feature Matching")
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        '--output', '-o',
        default='results.txt',
        help="Path to the output results file (default: results.txt)"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------
    cfg = load_config(args.config)
    ratio_thresh = cfg.get('matching', {}).get('ratio_thresh', 0.75)

    print(f"Config loaded from : {args.config}")
    print(f"Detector type      : {cfg.get('detector', {}).get('type', 'SIFT')}")
    print(f"Ratio threshold    : {ratio_thresh}\n")

    #################################################################################
    #                        FIRST PART -  READING IMAGES                           #
    #################################################################################

    # List of query/database images
    query_images = []
    db_images = []
    logo_names = []

    # Resize images
    img_size = (400, 400)

    # Read query images
    # Order by name
    query_dir = sorted(os.listdir('query/'))
    for idx, img_name in enumerate(query_dir):
        # Read image
        img = cv2.imread(os.path.join('query/', img_name))
        # Resize image
        img = cv2.resize(img, dsize=img_size)
        # Save image
        query_images.append(img)
        # Save logo names (remove "_logo.txt" and capitalize)
        logo_names.append(os.path.splitext(img_name)[0][:-5].capitalize())
        #print("Image '" + img_name + "' loaded.")
    print("Query images loaded!")

    # Read db images
    # Order by number
    db_dir = sorted(os.listdir('database/'), key=lambda x: int(x.split('.')[0]))
    for idx, img_name in enumerate(db_dir):
        # Read image
        img = cv2.imread(os.path.join('database/', img_name))
        # Resize image
        img = cv2.resize(img, dsize=img_size)
        # Save image
        db_images.append(img)
    print("Database images loaded!")



    #################################################################################
    #               SECOND PART - GROUND-TRUTH                                      #
    #################################################################################

    # Ground-truth arrays per logo
    coca_cola_GT = np.zeros(len(db_images))
    marlboro_GT = np.zeros(len(db_images))
    starbucks_GT = np.zeros(len(db_images))
    heineken_GT = np.zeros(len(db_images))

    coca_cola_GT[18:25] = 1
    marlboro_GT[55:61] = 1
    marlboro_GT[100:] = 1
    starbucks_GT[70:100] = 1
    heineken_GT[25:55] = 1

    # Matrix containing ground-truth values for each query image
    GT_matrix = np.vstack([coca_cola_GT, heineken_GT, marlboro_GT, starbucks_GT])
    print(f"GT matrix shape: {GT_matrix.shape}")

    n_coca_cola = int(coca_cola_GT.sum())
    n_heineken = int(heineken_GT.sum())
    n_marlboro = int(marlboro_GT.sum())
    n_starbucks = int(starbucks_GT.sum())
    n_others = len(db_images) - (n_coca_cola + n_heineken + n_marlboro + n_starbucks)
    print(f"Number of database images (coca_cola): {n_coca_cola}")
    print(f"Number of database images (heineken): {n_heineken}")
    print(f"Number of database images (marlboro): {n_marlboro}")
    print(f"Number of database images (starbucks): {n_starbucks}")
    print(f"Number of database images (others): {n_others}")


    #################################################################################
    #                   THIRD PART - SIFT FEATURE EXTRACTION                        #
    #################################################################################

    # Build detector from config
    detector, det_type = build_detector(cfg)
    print(f"\nDetector: {det_type}")

    # Detect and compute interest points and their descriptors
    query_kps = []
    query_des = []
    db_kps = []
    db_des = []

    for i in range(len(query_images)):
        kp, des = detector.detectAndCompute(query_images[i], None)
        query_kps.append(kp)
        query_des.append(des)

    for i in range(len(db_images)):
        kp, des = detector.detectAndCompute(db_images[i], None)
        db_kps.append(kp)
        db_des.append(des)


    #################################################################################
    #                 FOURTH PART - FEATURE MATCHING                                #
    #################################################################################

    # Choose norm based on detector type
    norm = cv2.NORM_HAMMING if det_type in ('ORB', 'BRISK') else cv2.NORM_L2

    # Brute-force matcher
    bf = cv2.BFMatcher_create(norm)

    # Matrix containing the number of matches per logo
    num_matches = np.zeros((len(query_images), len(db_images)))

    # Save matches for each descriptor of query images
    # List of lists: 1 list per query image, containing n_db_images lists, one for each db image,
    # containing the selected matches
    logo_matches = []

    # For each query image
    for q in range(len(query_images)):
        q_matches = []
        # For each database image
        for db in range(len(db_images)):
            matches = bf.knnMatch(query_des[q], db_des[db], k=2)
            # Apply Lowe's ratio test
            n_matches = 0
            sel_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    n_matches += 1
                    sel_matches.append(m)
            # Save number of matches
            num_matches[q, db] = n_matches
            # Save matches
            q_matches.append(sel_matches)
        logo_matches.append(q_matches)

    for idx, logo in enumerate(logo_names):
        print("Number of matches (" + logo + "):")
        print(num_matches[idx])


    #################################################################################
    #                   FIFTH PART - PERFORMANCE EVALUATION                         #
    #################################################################################

    # Average precision for each logo
    AP_all = np.zeros(len(query_images))

    _ = plt.figure(figsize=(12, 12))
    for i in range(len(query_images)):
        # Rank images in descending order according to the number of detected matches
        # The idx (corresponding to database images) which have the largest number of valid matches will be situated towards
        # the start of the array
        # The idx containing the least number of valid matches will be situated towards the end of the array
        # In this manner, we arrange predictions based on its validity or confidence (here, measured by the number of matches)
        idx = np.argsort(-num_matches[i])

        # True Positives
        TP = GT_matrix[i, idx] == 1

        # False Positives
        FP = GT_matrix[i, idx] == 0

        # TP and FP for each first k database images --> using np.cumsum gives us different "clusters" of matched database images,
        # ordered by number of matches. NB: TP + FP = k for each k
        # Note that as we go from the start to the end of the array, we are effectively varying the confidence threshold
        # (here, we use the number of matches as our confidence measure).
        # This is possible since the idx is sorted, and correspondingly, so are the TP and FP arrays
        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        # Precision/Recall for each k
        prec = TP / (TP + FP)
        rec = TP / np.sum(GT_matrix[i])

        # (Interpolated) Average precision (as in Pascal VOC2007)
        AP = 0
        interp_prec = []
        # We evaluate the precision at 11 recall values to reduce ''wiggles'' in the precision/recall curve
        for t in np.linspace(0, 1, num=11):
            p = prec[rec >= t]
            if p.size == 0:
                p = 0
            else:
                p = p.max()
            interp_prec.append(p)
            AP = AP + p/11
        AP_all[i] = AP

        # Plot precision/recall
        _ = plt.subplot(2, 2, i+1)
        plt.title(f"{logo_names[i]} - Average Precision (AP): {AP:.3f}")
        _ = plt.plot(rec, prec, '-', linewidth=3)
        _ = plt.plot(np.linspace(0, 1, num=11), interp_prec, 'o', color='red', linewidth=2)
        plt.legend(['Precision', 'Interpolated AP'])
        plt.grid('on')
        plt.xlabel('Recall'), plt.ylabel('Precision')
        plt.axis((0, 1, 0, 1.1))


    #################################################################################
    #                 SIXTH PART - Print & save results                             #
    #################################################################################
    print('\nBASELINE RESULTS')
    print("Average Precision (AP):\n")
    for i in range(len(query_images)):
        print(f'{logo_names[i]:9s}  ==>' + ' ' * 2 + f'{AP_all[i]:.3f}')

    print(f"\nMean Average Precision (mAP) ==> {np.mean(AP_all):.3f}")

    # Save results and parameters to .txt file
    save_results(args.output, cfg, logo_names, AP_all)


if __name__ == "__main__":
    main()
    plt.show()
