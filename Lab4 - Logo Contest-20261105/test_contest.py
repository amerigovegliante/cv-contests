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


def build_detector(det_cfg: dict):
    """Instantiate a feature detector from a detector config dict."""
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
        detector = cv2.ORB_create(
            nfeatures=det_cfg.get('nfeatures', 500),
        )
    elif det_type == 'AKAZE':
        detector = cv2.AKAZE.create()
    elif det_type == 'BRISK':
        # OpenCV 5: BRISK lives under xfeatures2d (requires opencv-contrib-python)
        detector = cv2.xfeatures2d.BRISK.create()
    else:
        raise ValueError(f"Unsupported detector type: '{det_type}'. "
                         "Choose from: SIFT, ORB, AKAZE, BRISK.")

    return detector, det_type


def resolve_query_config(cfg: dict, logo_name: str) -> dict:
    """
    Return the config block for a specific query logo.

    Lookup order:
      1. cfg['queries'][<key>] where <key> is the lowercased logo name
         with spaces replaced by underscores.
      2. The top-level cfg itself (global fallback).

    The per-query block may contain its own 'detector' and 'matching'
    sections. Missing keys fall back to the global config.
    """
    key = logo_name.lower().replace(' ', '_')
    per_query = cfg.get('queries', {}).get(key)
    if per_query is None:
        return cfg

    # Merge: per-query values override global, but global fills gaps.
    merged = {
        'detector': {**cfg.get('detector', {}), **per_query.get('detector', {})},
        'matching': {**cfg.get('matching', {}), **per_query.get('matching', {})},
    }
    return merged


def save_results(results_path: str, cfg: dict, per_query_cfg: dict,
                 logo_names: list, AP_all: np.ndarray) -> None:
    """Save AP results and the per-query parameters used to a .txt file."""
    mAP = float(np.mean(AP_all))

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

        # --- Per-query parameters actually used ---
        f.write("PARAMETERS USED PER QUERY\n")
        f.write("-" * 40 + "\n")
        for name in logo_names:
            qcfg = per_query_cfg[name]
            det = qcfg.get('detector', {})
            mat = qcfg.get('matching', {})
            f.write(f"  [{name}]\n")
            f.write(f"    detector.type          : {det.get('type', 'SIFT')}\n")
            if det.get('type', 'SIFT').upper() == 'SIFT':
                f.write(f"    detector.nfeatures     : {det.get('nfeatures', 0)}\n")
                f.write(f"    detector.nOctaveLayers : {det.get('nOctaveLayers', 3)}\n")
                f.write(f"    detector.contrastThresh: {det.get('contrastThreshold', 0.04)}\n")
                f.write(f"    detector.edgeThreshold : {det.get('edgeThreshold', 10)}\n")
                f.write(f"    detector.sigma         : {det.get('sigma', 1.6)}\n")
            f.write(f"    matching.ratio_thresh  : {mat.get('ratio_thresh', 0.75)}\n\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to: {results_path}")


def main():
    # -------------------------------------------------------------------------
    # Parse CLI arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Logo Contest - Feature Matching")
    parser.add_argument('--config', '-c', default='config.yaml',
                        help="Path to the YAML configuration file (default: config.yaml)")
    parser.add_argument('--output', '-o', default='results.txt',
                        help="Path to the output results file (default: results.txt)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------
    cfg = load_config(args.config)
    print(f"Config loaded from : {args.config}")
    if 'queries' in cfg:
        print(f"Per-query overrides: {list(cfg['queries'].keys())}")
    print(f"Global detector    : {cfg.get('detector', {}).get('type', 'SIFT')}")
    print(f"Global ratio_thresh: {cfg.get('matching', {}).get('ratio_thresh', 0.75)}\n")

    #################################################################################
    #                        FIRST PART - READING IMAGES                            #
    #################################################################################
    query_images, db_images, logo_names = [], [], []
    img_size = (400, 400)

    query_dir = sorted(os.listdir('query/'))
    for img_name in query_dir:
        img = cv2.imread(os.path.join('query/', img_name))
        img = cv2.resize(img, dsize=img_size)
        query_images.append(img)
        logo_names.append(os.path.splitext(img_name)[0][:-5].capitalize())
    print("Query images loaded!")

    db_dir = sorted(os.listdir('database/'), key=lambda x: int(x.split('.')[0]))
    for img_name in db_dir:
        img = cv2.imread(os.path.join('database/', img_name))
        img = cv2.resize(img, dsize=img_size)
        db_images.append(img)
    print("Database images loaded!")

    #################################################################################
    #               SECOND PART - GROUND-TRUTH                                      #
    #################################################################################
    coca_cola_GT = np.zeros(len(db_images))
    marlboro_GT = np.zeros(len(db_images))
    starbucks_GT = np.zeros(len(db_images))
    heineken_GT = np.zeros(len(db_images))

    coca_cola_GT[18:25] = 1
    marlboro_GT[55:61] = 1
    marlboro_GT[100:] = 1
    starbucks_GT[70:100] = 1
    heineken_GT[25:55] = 1

    GT_matrix = np.vstack([coca_cola_GT, heineken_GT, marlboro_GT, starbucks_GT])
    print(f"GT matrix shape: {GT_matrix.shape}")

    # --- Sanity checks on ground truth ---
    n_coca_cola = int(coca_cola_GT.sum())
    n_heineken = int(heineken_GT.sum())
    n_marlboro = int(marlboro_GT.sum())
    n_starbucks = int(starbucks_GT.sum())
    n_others = len(db_images) - (n_coca_cola + n_heineken + n_marlboro + n_starbucks)
    print(f"Database size: {len(db_images)}")
    print(f"Number of database images (coca_cola): {n_coca_cola}")
    print(f"Number of database images (heineken): {n_heineken}")
    print(f"Number of database images (marlboro): {n_marlboro}")
    print(f"Number of database images (starbucks): {n_starbucks}")
    print(f"Number of database images (others): {n_others}")
    if n_others < 0:
        print("WARNING: ground-truth arrays overlap — some DB images are "
              "labelled with more than one logo. Check the GT index ranges.")

    #################################################################################
    #               THIRD PART - PER-QUERY FEATURE EXTRACTION                       #
    #################################################################################
    # Resolve config and build one detector per query.
    per_query_cfg = {}
    query_kps, query_des = [], []
    query_det_type = []

    for i, name in enumerate(logo_names):
        qcfg = resolve_query_config(cfg, name)
        per_query_cfg[name] = qcfg

        det, det_type = build_detector(qcfg.get('detector', {}))
        kp, des = det.detectAndCompute(query_images[i], None)
        query_kps.append(kp)
        query_des.append(des)
        query_det_type.append(det_type)
        print(f"Query '{name}': detector={det_type}, keypoints={len(kp) if kp else 0}")

    # Database features only need to be extracted once, with the global detector.
    # (If queries use different detectors, database descriptors must be
    # extracted per query's detector — see note below.)
    print("\nNOTE: each query uses its own detector; DB descriptors are "
          "computed per query as needed.\n")

    #################################################################################
    #                 FOURTH PART - FEATURE MATCHING (per query)                    #
    #################################################################################
    num_matches = np.zeros((len(query_images), len(db_images)))
    logo_matches = []

    for q in range(len(query_images)):
        # (Re)build the DB descriptors with THIS query's detector.
        # Detectors like SIFT, ORB, BRISK, AKAZE produce different descriptor
        # types; you cannot mix them across query and database.
        det_cfg = per_query_cfg[logo_names[q]].get('detector', {})
        det, det_type = build_detector(det_cfg)
        ratio_thresh = per_query_cfg[logo_names[q]].get('matching', {}).get('ratio_thresh', 0.75)

        db_des_q = [det.detectAndCompute(img, None)[1] for img in db_images]
        db_kps_q = [det.detectAndCompute(img, None)[0] for img in db_images]

        norm = cv2.NORM_HAMMING if det_type in ('ORB', 'BRISK') else cv2.NORM_L2
        bf = cv2.BFMatcher_create(norm)

        q_des = query_des[q]
        q_matches = []

        for db in range(len(db_images)):
            d_des = db_des_q[db]
            if q_des is None or d_des is None or len(q_des) < 2 or len(d_des) < 2:
                num_matches[q, db] = 0
                q_matches.append([])
                continue

            matches = bf.knnMatch(q_des, d_des, k=2)
            n_matches = 0
            sel_matches = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    n_matches += 1
                    sel_matches.append(m)

            num_matches[q, db] = n_matches
            q_matches.append(sel_matches)

        logo_matches.append(q_matches)

    for idx, logo in enumerate(logo_names):
        print(f"Number of matches ({logo}) [{query_det_type[idx]}]:")
        print(num_matches[idx])

    #################################################################################
    #                   FIFTH PART - PERFORMANCE EVALUATION                         #
    #################################################################################
    AP_all = np.zeros(len(query_images))

    plt.figure(figsize=(12, 12))
    for i in range(len(query_images)):
        idx = np.argsort(-num_matches[i])
        TP = GT_matrix[i, idx] == 1
        FP = GT_matrix[i, idx] == 0
        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        prec = TP / (TP + FP)
        rec = TP / np.sum(GT_matrix[i])

        AP = 0
        interp_prec = []
        for t in np.linspace(0, 1, num=11):
            p = prec[rec >= t]
            if p.size == 0:
                p = 0
            else:
                p = p.max()
            interp_prec.append(p)
            AP = AP + p / 11
        AP_all[i] = AP

        plt.subplot(2, 2, i + 1)
        plt.title(f"{logo_names[i]} ({query_det_type[i]}) - AP: {AP:.3f}")
        plt.plot(rec, prec, '-', linewidth=3)
        plt.plot(np.linspace(0, 1, num=11), interp_prec, 'o', color='red', linewidth=2)
        plt.legend(['Precision', 'Interpolated AP'])
        plt.grid(True)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis((0, 1, 0, 1.1))

    #################################################################################
    #                 SIXTH PART - Print & save results                             #
    #################################################################################
    print('\nBASELINE RESULTS')
    print("Average Precision (AP):\n")
    for i in range(len(query_images)):
        print(f'{logo_names[i]:9s} ({query_det_type[i]:5s}) ==>  {AP_all[i]:.3f}')

    print(f"\nMean Average Precision (mAP) ==> {np.mean(AP_all):.3f}")

    save_results(args.output, cfg, per_query_cfg, logo_names, AP_all)


if __name__ == "__main__":
    main()
    plt.show()