import haiku as hk
from catx.network_module import CATXHaikuNetwork
from catx.type_defs import Observations, NetworkExtras, Logits

from utils import Globals, evaluate_predictions, get_data_loader


class MyCATXNetwork(CATXHaikuNetwork):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        layers = Globals.layers
        self.network = hk.nets.MLP(
            layers + [2 ** (self.depth + 1)], name=f"mlp_depth_{self.depth}"
        )

    def __call__(
        self,
        obs: Observations,
        network_extras: NetworkExtras,
    ) -> Logits:
        return self.network(
            obs,
            dropout_rate=network_extras["dropout_rate"],
            rng=hk.next_rng_key(),
        )


class BPAEnvironment:
    def __init__(
        self,
        data_dict,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        num_processors=2,
        num_points=2048,
    ) -> None:
        self.dataloader = get_data_loader(
            data_dict,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        self.data_iterator = iter(self.dataloader)
        self.num_processors = num_processors
        self.num_points = num_points

    def get_new_observations(self):
        try:
            batch = next(self.data_iterator)
        except StopIteration as e:
            self.data_iterator = iter(self.dataloader)
            batch = next(self.data_iterator)

        self.batch = batch
        return batch["features"]

    def get_costs(self, actions):
        actions = actions.tolist()
        return evaluate_predictions(
            self.batch["point_clouds"],
            actions,
            self.batch["targets"],
            self.num_processors,
            self.num_points,
        )
