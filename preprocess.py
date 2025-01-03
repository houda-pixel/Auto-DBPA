import argparse

from utils import preprocess_data


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_folder",
        type=str,
        default="./data/ModelNet10",
        help="data directory",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data/ModelNet10/processed",
        help="directory tp store preprocessed data",
    )
    parser.add_argument(
        "--val_proportion",
        type=float,
        default=0.025,
        help="proportion of the training set for validation",
    )
    parser.add_argument(
        "--num_processors",
        type=int,
        default=50,
        help="number of processors",
    )
    parser.add_argument(
        "--radius_feature",
        type=float,
        default=0.5,
        help="radius_feature fpfh",
    )
    parser.add_argument(
        "--max_nn",
        type=int,
        default=100,
        help="max_nn fpfh",
    )
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=100,
        help="num_keypoints fpfh",
    )
    parser.add_argument(
        "--target_num_pts",
        type=int,
        default=10000,
        help="num_samples",
    )
    return parser


def main():
    import faulthandler

    faulthandler.enable()
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)
    preprocess_data(args)


if __name__ == "__main__":
    main()
