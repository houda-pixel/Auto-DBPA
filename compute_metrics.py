import os
import argparse
import trimesh
import numpy as np
from multiprocessing import Pool


from sklearn.neighbors import KDTree
import numpy as np


def _setup_parser():
    """Set up the argument parser for command-line execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path to the directory where predictions and ground truth files are stored.",
    )
    parser.add_argument(
        "--foldername",
        type=str,
        default="test",
        help="Label or name for the dataset being processed.",
    )
    parser.add_argument(
        "--num_processors",
        type=int,
        default=16,
        help="Number of processors to use for multiprocessing.",
    )
    return parser


def compute_angles(points, triangles):
    A = points[triangles[:, 0], :]  # triangle vertex A
    B = points[triangles[:, 1], :]  # triangle vertex B
    C = points[triangles[:, 2], :]  # triangle vertex C

    # calculate the angles
    l_AB = np.sqrt(np.sum((A - B) ** 2, axis=-1)) + 1e-10
    l_AC = np.sqrt(np.sum((A - C) ** 2, axis=-1)) + 1e-10
    l_BC = np.sqrt(np.sum((B - C) ** 2, axis=-1)) + 1e-10

    dot_A = np.sum((B - A) * (C - A), axis=-1) / (l_AB * l_AC)
    dot_B = np.sum((A - B) * (C - B), axis=-1) / (l_AB * l_BC)
    dot_C = np.sum((A - C) * (B - C), axis=-1) / (l_AC * l_BC)
    dot_A = np.clip(dot_A, -1.0, 1.0)  # to avoid complex values
    dot_B = np.clip(dot_B, -1.0, 1.0)
    dot_C = np.clip(dot_C, -1.0, 1.0)
    angle_A = np.arccos(dot_A)  # [0, PI] in radians
    angle_B = np.arccos(dot_B)
    angle_C = np.arccos(dot_C)
    angles = np.stack([angle_A, angle_B, angle_C], axis=1)
    return (angles.reshape(-1)) * 180 / np.pi


def get_metrics(gt_path, pred_path):

    sample_num = 100000
    f1_threshold = 0.003
    ef1_radius = 0.004
    ef1_dotproduct_threshold = 0.2
    ef1_threshold = 0.005
    sample_num = 100000

    print(f"Processing:\n  Ground Truth: {gt_path}\n  Prediction: {pred_path}")

    # Load ground truth mesh
    print("Loading ground truth mesh...")
    gt_mesh = trimesh.load_mesh(gt_path)
    print("Ground truth mesh loaded.")
    gt_points, gt_indices = gt_mesh.sample(sample_num, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_indices]

    # Load prediction mesh
    print("Loading prediction mesh...")
    pred_mesh = trimesh.load_mesh(pred_path)
    print("Prediction mesh loaded.")

    # Sample points from prediction mesh
    print("Sampling points from prediction mesh...")
    pred_points, pred_indices = pred_mesh.sample(sample_num, return_index=True)
    print(f"Sampled {len(pred_points)} points from prediction mesh.")
    pred_normals = pred_mesh.face_normals[pred_indices]
    print("Extracting vertices and triangles from prediction mesh...")

    pred_vertices = pred_mesh.vertices
    pred_triangles = pred_mesh.faces

    print("Computing angles for prediction mesh...")
    pred_angles = compute_angles(pred_vertices, pred_triangles)

    small_angles = []
    for i in range(40):
        small_angles.append(np.mean((pred_angles < i).astype(np.float32)))

    # Comment out the rest for now
    # Uncomment when debugging moves beyond sampling
    # pred_normals = pred_mesh.face_normals[pred_indices]
    # gt_normals = gt_mesh.face_normals[gt_indices]
    # cd and nc and f1
    # from gt to pred
    pred_tree = KDTree(pred_points)
    dist, inds = pred_tree.query(gt_points, k=1)
    recall = np.sum(dist < f1_threshold) / float(len(dist))
    gt2pred_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    gt2pred_mean_cd2 = np.mean(dist)
    neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(gt_normals * neighbor_normals, axis=1))
    gt2pred_nc = np.mean(dotproduct)
    gt2pred_nr = np.mean(np.degrees(np.arccos(np.minimum(dotproduct, 1.0))))

    gt2pred_na = []
    for i in range(90):
        gt2pred_na.append(
            np.mean((dotproduct < np.cos(i / 180.0 * np.pi)).astype(np.float32))
        )

    # from pred to gt
    gt_tree = KDTree(gt_points)
    dist, inds = gt_tree.query(pred_points, k=1)
    precision = np.sum(dist < f1_threshold) / float(len(dist))
    pred2gt_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    pred2gt_mean_cd2 = np.mean(dist)
    neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(pred_normals * neighbor_normals, axis=1))
    pred2gt_nc = np.mean(dotproduct)
    pred2gt_nr = np.mean(np.degrees(np.arccos(np.minimum(dotproduct, 1.0))))

    pred2gt_na = []
    for i in range(90):
        pred2gt_na.append(
            np.mean((dotproduct < np.cos(i / 180.0 * np.pi)).astype(np.float32))
        )

    cd1 = gt2pred_mean_cd1 + pred2gt_mean_cd1
    cd2 = gt2pred_mean_cd2 + pred2gt_mean_cd2
    nc = (gt2pred_nc + pred2gt_nc) / 2
    nr = (gt2pred_nr + pred2gt_nr) / 2
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0

    # sample gt edge points
    indslist = gt_tree.query_radius(gt_points, ef1_radius)
    flags = np.zeros([len(gt_points)], bool)
    for p in range(len(gt_points)):
        inds = indslist[p]
        if len(inds) > 0:
            this_normals = gt_normals[p : p + 1]
            neighbor_normals = gt_normals[inds]
            dotproduct = np.abs(np.sum(this_normals * neighbor_normals, axis=1))
            if np.any(dotproduct < ef1_dotproduct_threshold):
                flags[p] = True
    gt_edge_points = np.ascontiguousarray(gt_points[flags])

    # sample pred edge points
    indslist = pred_tree.query_radius(pred_points, ef1_radius)
    flags = np.zeros([len(pred_points)], bool)
    for p in range(len(pred_points)):
        inds = indslist[p]
        if len(inds) > 0:
            this_normals = pred_normals[p : p + 1]
            neighbor_normals = pred_normals[inds]
            dotproduct = np.abs(np.sum(this_normals * neighbor_normals, axis=1))
            if np.any(dotproduct < ef1_dotproduct_threshold):
                flags[p] = True
    pred_edge_points = np.ascontiguousarray(pred_points[flags])

    # ecd ef1

    if len(pred_edge_points) == 0:
        pred_edge_points = np.zeros([486, 3], np.float32)
    if len(gt_edge_points) == 0:
        ecd1 = 0
        ecd2 = 0
        ef1 = 1
    else:
        # from gt to pred
        tree = KDTree(pred_edge_points)
        dist, inds = tree.query(gt_edge_points, k=1)
        erecall = np.sum(dist < ef1_threshold) / float(len(dist))
        gt2pred_mean_ecd1 = np.mean(dist)
        dist = np.square(dist)
        gt2pred_mean_ecd2 = np.mean(dist)

        # from pred to gt
        tree = KDTree(gt_edge_points)
        dist, inds = tree.query(pred_edge_points, k=1)
        eprecision = np.sum(dist < ef1_threshold) / float(len(dist))
        pred2gt_mean_ecd1 = np.mean(dist)
        dist = np.square(dist)
        pred2gt_mean_ecd2 = np.mean(dist)

        ecd1 = gt2pred_mean_ecd1 + pred2gt_mean_ecd1
        ecd2 = gt2pred_mean_ecd2 + pred2gt_mean_ecd2
        if erecall + eprecision > 0:
            ef1 = 2 * erecall * eprecision / (erecall + eprecision)
        else:
            ef1 = 0

    # return only sampling results for now
    return (
        cd1 * 100,
        cd2 * (10**5),
        f1,
        nc,
        nr,
        ecd1 * 100,
        ecd2 * 10000,
        ef1,
    )


def compute_metrics(predictions_dir, foldername, num_processors):
    # Collect files from directories
    preds_dir_files = os.listdir(predictions_dir)
    predictions_ = [
        os.path.join(predictions_dir, file)
        for file in preds_dir_files
        if file.startswith("prediction")
    ]
    groundtruths_ = [
        os.path.join(predictions_dir, file)
        for file in preds_dir_files
        if file.startswith("groundtruth")
    ]

    # Sort files to ensure consistency
    predictions = []
    groundtruths = []
    point_clouds = []
    for i, pred_file in enumerate(predictions_):
        try:
            test_mesh = trimesh.load_mesh(pred_file)
            # Check if the loaded file is a valid mesh
            if not hasattr(test_mesh, "faces") or len(test_mesh.faces) == 0:
                print(f"Invalid mesh or point cloud detected: {pred_file}")
                point_clouds.append(pred_file)  # Store as point cloud
                continue
            predictions.append(pred_file)  # Valid mesh
        except Exception as e:
            print(f"Error loading file: {pred_file}, Error: {e}")
            point_clouds.append(pred_file)  # Store as point cloud
            continue

    for i, gt_file in enumerate(groundtruths_):
        try:
            test_mesh = trimesh.load_mesh(gt_file)
            if not hasattr(test_mesh, "faces") or len(test_mesh.faces) == 0:
                continue  # Ignore invalid ground truths
            groundtruths.append(gt_file)
        except Exception as e:
            continue

    # Match files based on the full identifier
    paired_groundtruths = []
    paired_predictions = []
    for gt_file in groundtruths:
        # Extract full identifier from the ground truth filename
        gt_identifier = (
            os.path.basename(gt_file).replace("groundtruth_", "").split(".")[0]
        )
        matching_preds = [
            pred_file
            for pred_file in predictions
            if gt_identifier in os.path.basename(pred_file).replace("prediction_", "")
        ]
        if matching_preds:
            paired_groundtruths.append(gt_file)
            paired_predictions.append(matching_preds[0])
        else:
            print(f"Warning: No prediction found for ground truth {gt_file}")

    # Log the matched pairs
    print("\n=== Verified File Pairing ===")
    for gt, pred in zip(paired_groundtruths, paired_predictions):
        print(f"Ground Truth: {gt} -> Prediction: {pred}")

    # Continue with sampling and metric computation only for valid meshes
    test_size = len(paired_predictions)
    results = {"surface": []}
    results_surface = np.zeros([test_size, 8], np.float32)

    with Pool(maxtasksperchild=1, processes=num_processors) as pool:
        results["surface"] = pool.starmap(
            get_metrics, zip(paired_groundtruths, paired_predictions)
        )

    for i in range(test_size):
        results_surface[i] = np.array(results["surface"][i])

    results_surface = np.around(np.mean(results_surface, axis=0), decimals=4, out=None)

    print("=============================\n")

    # Count and report invalid files
    print(f"Total files in predictions directory: {len(predictions_)}")
    print(f"Valid meshes for sampling: {len(predictions)}")
    print(f"Point clouds or invalid meshes: {len(predictions_) - len(predictions)}")

    print(f"Number of valid file pairs used for computation: {test_size}")
    print(
        "\n=================================================================================="
    )
    print(f"average results on all {foldername} samples:")
    print(
        "surface:   (CD1,CD2,F1,NC,NR,ECD1,ECD2,EF1)=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)"
        % tuple(results_surface)
    )


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    compute_metrics(args.predictions_dir, args.foldername, args.num_processors)
