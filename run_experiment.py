import os
import argparse

from tqdm import tqdm
import jax
import optax
from jax import numpy as jnp
from catx.catx import CATX

from utils import (
    get_data_loader,
    evaluate,
    get_data_dictionaries,
    save_model,
)
from model import BPAEnvironment, MyCATXNetwork

from compute_metrics import compute_metrics


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/ModelNet10/processed",
        help="directory where preprocessed data is stored",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )
    parser.add_argument(
        "--num_of_steps",
        type=int,
        default=1000,
        help="total number of training steps",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=50,
        help="evaluation after x training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="dropout rate",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="epsilon",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=1 / 8,
        help="bandwidth",
    )
    parser.add_argument(
        "--discretization_parameter",
        type=int,
        default=8,
        help="discretization_parameter",
    )
    parser.add_argument(
        "--action_min",
        type=float,
        default=0.00001,
        help="action min",
    )
    parser.add_argument(
        "--action_max",
        type=float,
        default=0.1,
        help="action max",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="processor for dataloader",
    )
    parser.add_argument(
        "--num_processors",
        type=int,
        default=32,
        help="number of processors",
    )
    parser.add_argument(
        "--num_processors_eval",
        type=int,
        default=100,
        help="number of processors for evaluation",
    )
    parser.add_argument(
        "--train_shuffle",
        type=bool,
        default=False,
        help="shuffle the training set",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="number of points for chamfer loss",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=7,
        help="size of the code book",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Sample10K",
        help="experiment_name",
    )
    parser.add_argument(
        "--use_sq",
        action="store_true",
    )

    return parser


def train(train_data_dict, validation_data_dict, test_data_dict, config):
    epsilon = config["epsilon"]
    # JAX pseudo-random number generator
    rng_key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(rng_key)

    # Instantiate the environment
    environment = BPAEnvironment(
        train_data_dict,
        batch_size=config["batch_size"],
        shuffle=config["train_shuffle"],
        num_workers=config["num_workers"],
        num_processors=config["num_processors"],
        num_points=config["num_points"],
    )

    validation_dataloader = get_data_loader(
        validation_data_dict,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    test_dataloader = get_data_loader(
        test_data_dict,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )

    # Instantiate CATX
    catx = CATX(
        catx_network=MyCATXNetwork,
        optimizer=optax.adam(learning_rate=config["learning_rate"]),
        discretization_parameter=config["discretization_parameter"],
        bandwidth=config["bandwidth"],
        # radius min max
        action_min=config["action_min"],
        action_max=config["action_max"],
        # action_space=(1e-5, 0.1)
    )

    # Training loop
    costs_cumulative_train = []
    costs_cumulative_validation = []
    os.makedirs(
        os.path.join(config["data_dir"], config["experiment_name"]), exist_ok=True
    )
    model_save_path = os.path.join(
        config["data_dir"], config["experiment_name"], "model.pickel"
    )
    for i in tqdm(range(1, config["num_of_steps"] + 1)):
        obs = environment.get_new_observations()
        if obs is None:
            break

        if i == 1:
            network_extras = {"dropout_rate": 0.0}
            state = catx.init(
                obs=obs,
                epsilon=epsilon,
                key=key,
                network_extras=network_extras,
            )

        state.network_extras["dropout_rate"] = 0.0
        actions, probabilities, state = catx.sample(
            obs=obs, epsilon=epsilon, state=state
        )
        costs = environment.get_costs(actions=actions)
        state.network_extras["dropout_rate"] = config["dropout_rate"]
        state = catx.learn(
            obs=obs,
            actions=actions,
            probabilities=probabilities,
            costs=costs,
            state=state,
        )

        costs_cumulative_train.append(jnp.mean(costs).item())
        print({"steps": i, "train_cost": costs_cumulative_train[-1]})
        if i % config["evaluation_steps"] == 0:
            current_validation_cost = evaluate(
                validation_dataloader,
                state,
                catx,
                obs,
                config["num_processors_eval"],
                config["num_points"],
            )
            if len(costs_cumulative_validation) > 0:
                if current_validation_cost < min(costs_cumulative_validation):
                    save_model(state, model_save_path)
            else:
                save_model(state, model_save_path)

            costs_cumulative_validation.append(current_validation_cost)
            print({"validation_cost": costs_cumulative_validation[-1]})

    test_error = evaluate(
        test_dataloader,
        state,
        catx,
        obs,
        config["num_processors_eval"],
        config["num_points"],
        save=True,
        experiment_name=config["experiment_name"],
        data_dir=config["data_dir"],
        folder_name="test",
    )

    compute_metrics(
        os.path.join(config["data_dir"], config["experiment_name"], "test"),
        "test",
        config["num_processors_eval"],
    )

    train_dataloader = get_data_loader(
        train_data_dict,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )

    train_error = evaluate(
        train_dataloader,
        state,
        catx,
        obs,
        config["num_processors_eval"],
        config["num_points"],
        save=True,
        experiment_name=config["experiment_name"],
        data_dir=config["data_dir"],
        folder_name="train",
    )

    print({"train_cost_final": train_error})

    print({"test_cost": test_error})
    return (
        costs_cumulative_train,
        costs_cumulative_validation,
        test_error,
        state,
        catx,
        environment,
    )


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    config = vars(args)
    (train_data_dict, validation_data_dict, test_data_dict) = get_data_dictionaries(
        config
    )
    (
        costs_cumulative_train,
        costs_cumulative_validation,
        test_error,
        state,
        catx,
        environment,
    ) = train(train_data_dict, validation_data_dict, test_data_dict, config)


if __name__ == "__main__":
    main()
