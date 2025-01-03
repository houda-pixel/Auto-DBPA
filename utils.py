import os
from functools import partial
import pickle
import tempfile

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import torch
from torch import utils
import open3d as o3d
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import IO
import point_cloud_utils as pcu
import pymeshlab


class Globals:
    layers = [10, 10]
    h_cost = 1.0


def load_pickle(file_path):
    """"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def save_model(state, path):
    model = {}
    model["state"] = state
    save_pickle(path, model)


def py3d_to_o3d(pcd):
    """for ball pivoting we will use open3d"""
    points = pcd.points_list()[0].cpu().detach().numpy()
    normals = pcd.normals_list()[0].cpu().detach().numpy()
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d_point_cloud.normals = o3d.utility.Vector3dVector(normals)
    return o3d_point_cloud


def convert_to_py3d(o3d_mesh):
    verts = torch.tensor(np.array(o3d_mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.array(o3d_mesh.triangles), dtype=torch.int64)
    # Create a PyTorch3D Meshes object from the vertices and faces tensors
    mesh_pt3d = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
    return mesh_pt3d


def py3d_mesh_to_pcd(normalized_mesh, target_num_pts):
    verts = np.array(normalized_mesh.verts_list()[0])
    faces = np.array(normalized_mesh.faces_list()[0])
    normals = pcu.estimate_mesh_vertex_normals(verts, faces)
    fid, bc = pcu.sample_mesh_poisson_disk(verts, faces, num_samples=target_num_pts)
    rand_positions = pcu.interpolate_barycentric_coords(faces, fid, bc, verts)
    rand_normals = pcu.interpolate_barycentric_coords(faces, fid, bc, normals)
    pointcloud = Pointclouds(
        points=[torch.from_numpy(rand_positions)],
        normals=[torch.from_numpy(rand_normals)],
    )
    return pointcloud


def get_py3d_pointclouds(meshes, target_num_pts, num_processors):
    point_clouds = Parallel(n_jobs=num_processors)(
        delayed(py3d_mesh_to_pcd)(mesh, target_num_pts) for mesh in meshes
    )
    return point_clouds


def read_mesh(path):
    return IO().load_mesh(path)


def normalize_mesh(mesh):
    vertices = mesh.verts_list()[0].numpy()
    triangles = mesh.faces_list()[0].numpy()

    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)
    vertices[:, 0] = (vertices[:, 0] - x_mid) / scale
    vertices[:, 1] = (vertices[:, 1] - y_mid) / scale
    vertices[:, 2] = (vertices[:, 2] - z_mid) / scale

    # normalized_mesh = Meshes(verts=[verts], faces=[faces_idx])
    normalized_mesh = Meshes(
        verts=[torch.from_numpy(vertices)], faces=[torch.from_numpy(triangles)]
    )
    return normalized_mesh


def read_all_meshes(basedir, filepath):
    path = os.path.join(basedir, filepath)
    meshes = []
    with open(path) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        mesh_path = os.path.join(basedir, line).strip()
        mesh = read_mesh(mesh_path)
        normalized_mesh = normalize_mesh(mesh)
        meshes.append(normalized_mesh)
    return meshes


@torch.no_grad()
def compute_cost(source_mesh, target_mesh, n_points=2048):
    try:
        sample_trg = sample_points_from_meshes(target_mesh, n_points)
        sample_src = sample_points_from_meshes(source_mesh, n_points)
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        loss = loss_chamfer.item()
        return loss
    except ValueError:
        # Set it a very high value
        loss = Globals.h_cost

    return loss


def bpa_pymeshlab(point_cloud_path, radius, mesh_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(point_cloud_path)
    r = pymeshlab.AbsoluteValue(radius)
    ms.generate_surface_reconstruction_ball_pivoting(ballradius=r)
    ms.save_current_mesh(mesh_path)


def run_bpa(pcd, radius):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_path_input = os.path.join(tmpdirname, "input_pointcloud.ply")
        temp_path_output = os.path.join(tmpdirname, "ouput_mesh.ply")
        IO().save_pointcloud(pcd, temp_path_input)
        bpa_pymeshlab(temp_path_input, radius, temp_path_output)
        reconstructed_mesh = read_mesh(temp_path_output)
    return reconstructed_mesh


def uniformly_sample_points(pcd, num_points):
    sampled_points = np.asarray(
        pcd.uniform_down_sample(len(pcd.points) // num_points).points
    )
    tree = o3d.geometry.KDTreeFlann(pcd)
    indices = []
    for i in range(len(sampled_points)):
        k, idx, _ = tree.search_knn_vector_3d(sampled_points[i], 1)
        indices.append(idx[0])
    return indices


def compute_fpfh_feature(pcd, radius_feature, max_nn, num_points):
    o3d_pcd = py3d_to_o3d(pcd)
    o3d_pcd.estimate_normals()
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        o3d_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn),
    )
    feature_vector = np.asarray(fpfh.data)
    sampling_indices = uniformly_sample_points(o3d_pcd, num_points=num_points)

    feature_vector = feature_vector.reshape(
        feature_vector.shape[1], feature_vector.shape[0]
    )
    feature_vector = feature_vector[sampling_indices]
    return feature_vector


def get_fpfh_features(
    point_clouds,
    num_processors=2,
    radius_feature=0.5,
    max_nn=100,
    num_points=100,
):
    """"""
    feature_func = partial(
        compute_fpfh_feature,
        radius_feature=radius_feature,
        max_nn=max_nn,
        num_points=num_points,
    )
    features = Parallel(n_jobs=num_processors, verbose=30)(
        delayed(feature_func)(point_cloud) for point_cloud in point_clouds
    )

    return features


def create_clusters(features, codebook_size=33):
    features = np.concatenate(features)
    kmeans = KMeans(
        n_clusters=codebook_size,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=1234,
    ).fit(features)
    return kmeans


def transform(input_vector, kmeans, codebook_size=33, use_sq=False):
    if use_sq:
        distances = kmeans.transform(input_vector)
        soft_quantization = 1.0 / (distances + 1e-6)
        soft_quantization = normalize(soft_quantization, axis=1, norm="l2")
        features = np.sum(soft_quantization, axis=0)

    else:
        labels = kmeans.predict(input_vector)
        features = np.zeros(codebook_size)
        for i in range(len(labels)):
            features[labels[i]] += 1
    return features / np.sum(features)


def featurize_data(data, k_means, codebook_size=33, use_sq=False):
    transformed_data = []
    for feature in data:
        transformed_data.append(
            transform(feature, k_means, codebook_size=codebook_size, use_sq=use_sq)
        )
    return np.array(transformed_data)


class BPADataSet(utils.data.Dataset):
    def __init__(self, features, targets, point_clouds):
        self.features = features
        self.targets = targets
        self.point_clouds = point_clouds

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        point_cloud = self.point_clouds[idx]

        return {
            "feature": feature,
            "target": target,
            "point_cloud": point_cloud,
        }


def collate_function(batch):
    features = [x["feature"].reshape(1, -1) for x in batch]
    targets = [x["target"] for x in batch]
    point_clouds = [x["point_cloud"] for x in batch]
    features = np.concatenate(features)

    return {
        "features": features,
        "targets": targets,
        "point_clouds": point_clouds,
    }


def predict_radius(state, catx, obs):
    epsilon = 0
    state.network_extras["dropout_rate"] = 0.0
    actions, probabilities, state = catx.sample(obs=obs, epsilon=epsilon, state=state)
    return actions


def evaluate_predictions(
    point_clouds,
    actions,
    targets,
    num_processors,
    n_points=2048,
    save=False,
    experiment_name="",
    data_dir="",
    folder_name="test",
):
    rec_meshes_py3d = Parallel(n_jobs=num_processors)(
        delayed(run_bpa)(point_cloud, radius)
        for point_cloud, radius in zip(point_clouds, actions)
    )
    if save:
        print("saving predictions")
        outdir = data_dir
        pred_dir = os.path.join(outdir, experiment_name, folder_name)
        os.makedirs(pred_dir, exist_ok=True)
        for i in tqdm(range(len(rec_meshes_py3d))):
            IO().save_mesh(
                rec_meshes_py3d[i],
                os.path.join(pred_dir, f"prediction_{i}.obj"),
            )
            IO().save_mesh(
                targets[i],
                os.path.join(pred_dir, f"groundtruth_{i}.obj"),
            )
        np.save(os.path.join(pred_dir, "radius_values.npy"), actions)
    return np.array(
        [compute_cost(x, y, n_points) for x, y in zip(rec_meshes_py3d, targets)]
    )


def evaluate(
    dataloader,
    state,
    catx,
    obs,
    num_processors,
    n_points,
    save=False,
    experiment_name="",
    data_dir="",
    folder_name="test",
):
    predictions = []
    point_clouds = []
    meshes = []

    for batch in dataloader:
        predictions.extend(predict_radius(state, catx, batch["features"]).tolist())
        point_clouds.extend(batch["point_clouds"])
        meshes.extend(batch["targets"])

    return np.mean(
        evaluate_predictions(
            point_clouds,
            predictions,
            meshes,
            num_processors,
            n_points=n_points,
            save=save,
            experiment_name=experiment_name,
            data_dir=data_dir,
            folder_name=folder_name,
        )
    )


def preprocess_data(config):
    print("Reading Meshes")
    meshes = read_all_meshes(config["ds_folder"], "train.txt")
    test_meshes = read_all_meshes(config["ds_folder"], "test.txt")
    train_meshes, validation_meshes = train_test_split(
        meshes, test_size=config["val_proportion"], random_state=42
    )
    del meshes
    outdir = config["outdir"]
    os.makedirs(outdir, exist_ok=True)
    print("Generating Pointclouds")
    sub_size = 1000
    for i in range(0, (len(train_meshes) // sub_size) + 1):
        start = i * sub_size
        end = (i + 1) * sub_size
        end = end if end <= len(train_meshes) else len(train_meshes)
        train_meshes_subset = train_meshes[start:end]
        train_pointclouds = get_py3d_pointclouds(
            train_meshes_subset,
            config["target_num_pts"],
            config["num_processors"],
        )
        train_features = get_fpfh_features(
            train_pointclouds,
            num_processors=config["num_processors"],
            radius_feature=config["radius_feature"],
            max_nn=config["max_nn"],
            num_points=config["num_keypoints"],
        )
        save_pickle(os.path.join(outdir, f"train_features_{i}.pickle"), train_features)
        save_pickle(
            os.path.join(outdir, f"train_meshes_{i}.pickle"),
            train_meshes_subset,
        )
        save_pickle(
            os.path.join(outdir, f"train_pointclouds_{i}.pickle"),
            train_pointclouds,
        )
        print(f"Subset {i} Done")

    del train_meshes
    del train_features
    del train_pointclouds

    validation_pointclouds = get_py3d_pointclouds(
        validation_meshes, config["target_num_pts"], config["num_processors"]
    )
    test_pointclouds = get_py3d_pointclouds(
        test_meshes, config["target_num_pts"], config["num_processors"]
    )
    validation_features = get_fpfh_features(
        validation_pointclouds,
        num_processors=config["num_processors"],
        radius_feature=config["radius_feature"],
        max_nn=config["max_nn"],
        num_points=config["num_keypoints"],
    )
    test_features = get_fpfh_features(
        test_pointclouds,
        num_processors=config["num_processors"],
        radius_feature=config["radius_feature"],
        max_nn=config["max_nn"],
        num_points=config["num_keypoints"],
    )

    print("Saving Data")
    save_pickle(os.path.join(outdir, "validation_features.pickle"), validation_features)
    save_pickle(os.path.join(outdir, "test_features.pickle"), test_features)

    save_pickle(os.path.join(outdir, "validation_meshes.pickle"), validation_meshes)
    save_pickle(os.path.join(outdir, "test_meshes.pickle"), test_meshes)

    save_pickle(
        os.path.join(outdir, "validation_pointclouds.pickle"),
        validation_pointclouds,
    )
    save_pickle(os.path.join(outdir, "test_pointclouds.pickle"), test_pointclouds)
    print(f"Data is saved at {outdir}")


def load_data(data_dir):
    outdir = data_dir
    print(f"loading the data from {outdir}")
    train_features = []
    filenames = []
    for filename in os.listdir(outdir):
        if filename.startswith("train_features_"):
            filenames.append(filename)
        filenames = sorted(filenames, key=lambda x: x.split("_.pickle")[-1])

    for filename in filenames:
        train_features.extend(load_pickle(os.path.join(outdir, filename)))

    train_pointclouds = []
    filenames = []
    for filename in os.listdir(outdir):
        if filename.startswith("train_pointclouds_"):
            filenames.append(filename)
        filenames = sorted(filenames, key=lambda x: x.split("_.pickle")[-1])

    for filename in filenames:
        train_pointclouds.extend(load_pickle(os.path.join(outdir, filename)))

    train_meshes = []
    filenames = []
    for filename in os.listdir(outdir):
        if filename.startswith("train_meshes_"):
            filenames.append(filename)
        filenames = sorted(filenames, key=lambda x: x.split("_.pickle")[-1])

    for filename in filenames:
        train_meshes.extend(load_pickle(os.path.join(outdir, filename)))

    validation_features = load_pickle(
        os.path.join(outdir, "validation_features.pickle")
    )
    test_features = load_pickle(os.path.join(outdir, "test_features.pickle"))

    validation_meshes = load_pickle(os.path.join(outdir, "validation_meshes.pickle"))
    test_meshes = load_pickle(os.path.join(outdir, "test_meshes.pickle"))

    validation_pointclouds = load_pickle(
        os.path.join(outdir, "validation_pointclouds.pickle")
    )
    test_pointclouds = load_pickle(os.path.join(outdir, "test_pointclouds.pickle"))
    print("Done loading data")

    return {
        "train": (train_features, train_meshes, train_pointclouds),
        "validation": (
            validation_features,
            validation_meshes,
            validation_pointclouds,
        ),
        "test": (test_features, test_meshes, test_pointclouds),
    }


def get_data_dictionaries(config):
    data = load_data(config["data_dir"])
    train_features, train_meshes, train_pointclouds = data["train"]
    validation_features, validation_meshes, validation_pointclouds = data["validation"]
    test_features, test_meshes, test_pointclouds = data["test"]
    print("Creating Clusters")
    kmeans = create_clusters(train_features, codebook_size=config["codebook_size"])
    print("Transforming Features")
    train_features_transformed = featurize_data(
        train_features,
        kmeans,
        codebook_size=config["codebook_size"],
        use_sq=config["use_sq"],
    )
    validation_features_transformed = featurize_data(
        validation_features,
        kmeans,
        codebook_size=config["codebook_size"],
        use_sq=config["use_sq"],
    )
    test_features_transformed = featurize_data(
        test_features,
        kmeans,
        codebook_size=config["codebook_size"],
        use_sq=config["use_sq"],
    )

    train_data_dict = {
        "features": train_features_transformed,
        "targets": train_meshes,
        "point_clouds": train_pointclouds,
    }
    validation_data_dict = {
        "features": validation_features_transformed,
        "targets": validation_meshes,
        "point_clouds": validation_pointclouds,
    }
    test_data_dict = {
        "features": test_features_transformed,
        "targets": test_meshes,
        "point_clouds": test_pointclouds,
    }

    return train_data_dict, validation_data_dict, test_data_dict


def get_data_loader(data_dict, batch_size, num_workers, shuffle):
    features = data_dict["features"]
    targets = data_dict["targets"]
    point_clouds = data_dict["point_clouds"]
    dataset = BPADataSet(features, targets, point_clouds)
    dataloader = utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_function,
        drop_last=False,
    )
    return dataloader
