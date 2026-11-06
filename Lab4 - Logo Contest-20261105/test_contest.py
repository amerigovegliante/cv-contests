import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; remove if you want interactive plots
import matplotlib.pyplot as plt
import yaml
import argparse
from datetime import datetime


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def to_root_sift(descriptors, eps=1e-7):
    """Convert SIFT descriptors to RootSIFT (L1-normalize then sqrt)."""
    if descriptors is None or len(descriptors) == 0:
        return descriptors
    if descriptors.dtype not in (np.float32, np.float64):
        return descriptors
    desc = descriptors.astype(np.float32, copy=True)
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc)
    return desc


def count_ransac_inliers(sel_matches, q_kps, db_kps,
                         reproj_thresh=5.0, min_inliers=4):
    """
    Estimate a homography from ratio-filtered matches and return the
    number of inliers consistent with it.

    If there are fewer than `min_inliers` matches, or homography estimation
    fails, returns 0 (this pair contributes nothing to the ranking).
    """
    if len(sel_matches) < min_inliers:
        return 0
    if q_kps is None or db_kps is None:
        return 0

    src_pts = np.float32(
        [q_kps[m.queryIdx].pt for m in sel_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [db_kps[m.trainIdx].pt for m in sel_matches]
    ).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if M is None or mask is None:
        return 0
    return int(mask.sum())


def build_detector(det_cfg: dict):
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
        detector = cv2.ORB_create(nfeatures=det_cfg.get('nfeatures', 500))
    elif det_type == 'AKAZE':
        detector = cv2.AKAZE.create()
    elif det_type == 'BRISK':
        detector = cv2.xfeatures2d.BRISK.create()
    else:
        raise ValueError(f"Unsupported detector type: '{det_type}'.")

    return detector, det_type


def resolve_query_config(cfg: dict, logo_name: str) -> dict:
    key = logo_name.lower().replace(' ', '_')
    per_query = cfg.get('queries', {}).get(key)
    if per_query is None:
        return cfg
    return {
        'detector': {**cfg.get('detector', {}), **per_query.get('detector', {})},
        'matching': {**cfg.get('matching', {}), **per_query.get('matching', {})},
    }


def save_results(results_path: str, cfg: dict, per_query_cfg: dict,
                 logo_names: list, AP_all: np.ndarray) -> None:
    mAP = float(np.mean(AP_all))

    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("           LOGO CONTEST - RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write("Average Precision (AP):\n")
        for name, ap in zip(logo_names, AP_all):
            f.write(f"  {name:12s}  ==>  {ap:.4f}\n")
        f.write(f"\nMean Average Precision (mAP)  ==>  {mAP:.4f}\n\n")

        f.write("PARAMETERS USED PER QUERY\n")
        f.write("-" * 40 + "\n")
        for name in logo_names:
            qcfg = per_query_cfg[name]
            det = qcfg.get('detector', {})
            mat = qcfg.get('matching', {})
            f.write(f"  [{name}]\n")
            f.write(f"    detector.type           : {det.get('type', 'SIFT')}\n")
            if det.get('type', 'SIFT').upper() == 'SIFT':
                f.write(f"    detector.nfeatures      : {det.get('nfeatures', 0)}\n")
                f.write(f"    detector.nOctaveLayers  : {det.get('nOctaveLayers', 3)}\n")
                f.write(f"    detector.contrastThresh : {det.get('contrastThreshold', 0.04)}\n")
                f.write(f"    detector.edgeThreshold  : {det.get('edgeThreshold', 10)}\n")
                f.write(f"    detector.sigma          : {det.get('sigma', 1.6)}\n")
                f.write(f"    detector.root_sift      : {det.get('root_sift', False)}\n")
            f.write(f"    matching.ratio_thresh   : {mat.get('ratio_thresh', 0.75)}\n")
            f.write(f"    matching.ransac_reproj  : {mat.get('ransac_reproj_thresh', 5.0)}\n")
            f.write(f"    matching.ransac_min_inl : {mat.get('ransac_min_inliers', 4)}\n\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Logo Contest - Feature Matching")
    parser.add_argument('--config', '-c', default='config.yaml')
    parser.add_argument('--output', '-o', default='results.txt')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config loaded from : {args.config}")
    if 'queries' in cfg:
        print(f"Per-query overrides: {list(cfg['queries'].keys())}")
    print(f"Global detector    : {cfg.get('detector', {}).get('type', 'SIFT')}")
    print(f"Global ratio_thresh: {cfg.get('matching', {}).get('ratio_thresh', 0.75)}")
    print(f"Global root_sift   : {cfg.get('detector', {}).get('root_sift', False)}")
    print(f"Global RANSAC      : reproj={cfg.get('matching', {}).get('ransac_reproj_thresh', 5.0)}, "
          f"min_inliers={cfg.get('matching', {}).get('ransac_min_inliers', 4)}\n")

    #################################################################################
    #                        FIRST PART - READING IMAGES                            #
    #################################################################################
    query_images, db_images, logo_names = [], [], []
    img_size = (400, 400)

    for img_name in sorted(os.listdir('query/')):
        img = cv2.imread(os.path.join('query/', img_name))
        img = cv2.resize(img, dsize=img_size)
        query_images.append(img)
        logo_names.append(os.path.splitext(img_name)[0][:-5].capitalize())
    print("Query images loaded!")

    for img_name in sorted(os.listdir('database/'), key=lambda x: int(x.split('.')[0])):
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
        print("WARNING: ground-truth arrays overlap.")

    #################################################################################
    #               THIRD PART - PER-QUERY FEATURE EXTRACTION                       #
    #################################################################################
    per_query_cfg = {}
    query_kps, query_des = [], []
    query_det_type = []

    for i, name in enumerate(logo_names):
        qcfg = resolve_query_config(cfg, name)
        per_query_cfg[name] = qcfg

        det_cfg = qcfg.get('detector', {})
        use_root_sift = det_cfg.get('root_sift', False)

        det, det_type = build_detector(det_cfg)
        kp, des = det.detectAndCompute(query_images[i], None)
        if use_root_sift:
            des = to_root_sift(des)

        query_kps.append(kp)
        query_des.append(des)
        query_det_type.append(det_type)
        print(f"Query '{name}': detector={det_type}, "
              f"root_sift={use_root_sift}, "
              f"keypoints={len(kp) if kp else 0}")

    #################################################################################
    #                 FOURTH PART - FEATURE MATCHING + RANSAC                       #
    #################################################################################
    num_matches = np.zeros((len(query_images), len(db_images)))
    logo_matches = []

    for q in range(len(query_images)):
        det_cfg = per_query_cfg[logo_names[q]].get('detector', {})
        mat_cfg = per_query_cfg[logo_names[q]].get('matching', {})

        use_root_sift = det_cfg.get('root_sift', False)
        ratio_thresh = mat_cfg.get('ratio_thresh', 0.75)
        ransac_reproj = mat_cfg.get('ransac_reproj_thresh', 5.0)
        ransac_min = mat_cfg.get('ransac_min_inliers', 4)

        det, det_type = build_detector(det_cfg)

        # Compute DB features once per query, applying RootSIFT if enabled.
        db_des_q, db_kps_q = [], []
        for img in db_images:
            kp, des = det.detectAndCompute(img, None)
            if use_root_sift:
                des = to_root_sift(des)
            db_kps_q.append(kp)
            db_des_q.append(des)

        norm = cv2.NORM_HAMMING if det_type in ('ORB', 'BRISK') else cv2.NORM_L2
        bf = cv2.BFMatcher_create(norm)

        q_des = query_des[q]
        q_kps = query_kps[q]
        q_matches = []

        for db in range(len(db_images)):
            d_des = db_des_q[db]
            d_kps = db_kps_q[db]
            if (q_des is None or d_des is None
                    or len(q_des) < 2 or len(d_des) < 2):
                num_matches[q, db] = 0
                q_matches.append([])
                continue

            matches = bf.knnMatch(q_des, d_des, k=2)
            sel_matches = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    sel_matches.append(m)

            use_ransac = mat_cfg.get('use_ransac', False)

            if use_ransac:
                n_matches = count_ransac_inliers(
                    sel_matches, q_kps, d_kps,
                    reproj_thresh=ransac_reproj,
                    min_inliers=ransac_min,
                )
            else:
                n_matches = len(sel_matches)

            num_matches[q, db] = n_matches
            q_matches.append(sel_matches)

        logo_matches.append(q_matches)

    for idx, logo in enumerate(logo_names):
        rs = per_query_cfg[logo].get('detector', {}).get('root_sift', False)
        print(f"Number of RANSAC inliers ({logo}) [{query_det_type[idx]}"
              f"{', RootSIFT' if rs else ''}]:")
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

        rs_tag = " + RootSIFT" if per_query_cfg[logo_names[i]].get(
            'detector', {}).get('root_sift', False) else ""
        plt.subplot(2, 2, i + 1)
        plt.title(f"{logo_names[i]} ({query_det_type[i]}{rs_tag}) - AP: {AP:.3f}")
        plt.plot(rec, prec, '-', linewidth=3)
        plt.plot(np.linspace(0, 1, num=11), interp_prec, 'o', color='red', linewidth=2)
        plt.legend(['Precision', 'Interpolated AP'])
        plt.grid(True)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis((0, 1, 0, 1.1))

    plt.savefig('pr_curves.png', dpi=120, bbox_inches='tight')
    print("\nPR curves saved to: pr_curves.png")

    #################################################################################
    #                 SIXTH PART - Print & save results                             #
    #################################################################################
    print('\nBASELINE RESULTS')
    print("Average Precision (AP):\n")
    for i in range(len(query_images)):
        rs = per_query_cfg[logo_names[i]].get('detector', {}).get('root_sift', False)
        rs_tag = "+RS" if rs else "   "
        print(f'{logo_names[i]:9s} ({query_det_type[i]:5s}{rs_tag}) ==>  {AP_all[i]:.3f}')

    print(f"\nMean Average Precision (mAP) ==> {np.mean(AP_all):.3f}")

    save_results(args.output, cfg, per_query_cfg, logo_names, AP_all)


if __name__ == "__main__":
    main()