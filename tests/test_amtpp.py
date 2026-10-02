"""Model and protocol invariants on artificial trip records."""
import unittest

import numpy as np
import pandas as pd
import torch

from amtpp import AMTPP, AMTPPConfig
from amtpp.data.strict import StrictMetroConfig, StrictMetroCorpus, collate_strict_metro
from amtpp.utils.common import set_seed


def example_frame():
    return pd.DataFrame([{"userID": user, "startTime": pd.Timestamp("2020-01-01") + pd.Timedelta(days=event, minutes=user), "origin": event % 4, "destination": (event + 1) % 4} for user in range(20) for event in range(6)])


class AMTPPTest(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.corpus = StrictMetroCorpus(example_frame(), StrictMetroConfig(city="synthetic", min_trips=4, history_end="2020-01-04 23:59:59", future_start="2020-01-05 00:00:00"))
        self.batch = collate_strict_metro(self.corpus.samples["validation"])

    def model(self, graph=False):
        config = AMTPPConfig(d_loc=4, d_dow=4, d_hour=4, n_heads=2, c_model=8, K=2, r=2, w_eos=0, topology_mode="graph" if graph else "none")
        kwargs = {}
        if graph:
            adjacency = np.zeros((self.corpus.S, self.corpus.S), dtype=np.float32)
            for station in range(1, self.corpus.S - 1):
                adjacency[station, station + 1] = adjacency[station + 1, station] = 1
            kwargs["topology_adjacency"] = adjacency
            index = np.arange(self.corpus.S, dtype=np.float32)
            distance = np.abs(index[:, None] - index[None, :])
            distance[0, :] = distance[:, 0] = 0
            kwargs["topology_distance"] = distance
            kwargs["od_logit_bias"] = -distance / distance.max()
        return AMTPP(self.corpus.S, [1], self.corpus.allowed_od_mask, config, **kwargs)

    def forward(self, model, batch=None):
        batch = self.batch if batch is None else batch
        return model(**{key: batch[key] for key in ("cond", "tau", "hour", "dow", "origin", "dest", "mask")})

    def test_disjoint_users_and_temporal_targets(self):
        users = [set(self.corpus.split_users[split]) for split in ("train", "validation", "test")]
        self.assertEqual([len(group) for group in users], [16, 2, 2])
        self.assertFalse(users[0] & users[1] or users[0] & users[2] or users[1] & users[2])
        for sample in self.corpus.samples["train"]:
            self.assertFalse(sample["future_mask"].any())
        for sample in self.corpus.samples["validation"]:
            self.assertTrue(np.all(~sample["target_mask"] | sample["future_mask"]))

    def test_future_station_does_not_expand_training_vocabulary(self):
        frame = example_frame()
        frame.loc[frame["startTime"] >= pd.Timestamp("2020-01-05"), "destination"] = 99
        with self.assertRaisesRegex(ValueError, "absent from train-history vocabulary"):
            StrictMetroCorpus(frame, self.corpus.cfg)

    def test_probabilities_normalize_and_loss_backpropagates(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                model = self.model(graph)
                output = self.forward(model)
                for attribute in ("o_prob", "d_prob"):
                    self.assertTrue(torch.allclose(output[attribute].sum(-1), torch.ones_like(output[attribute][..., 0]), atol=1e-6))
                    self.assertTrue(torch.all(output[attribute][..., 0] == 0))
                joint = output["od_prob"] * output["o_prob"][..., None, :]
                self.assertTrue(torch.allclose(joint.sum((-2, -1)), torch.ones_like(output["o_prob"][..., 0]), atol=1e-6))
                loss = model.nll(output, self.batch["tau"], self.batch["origin"], self.batch["dest"], torch.zeros_like(self.batch["tau"]), self.batch["target_mask"])["total"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_current_target_cannot_change_its_prediction(self):
        model = self.model().eval()
        original = self.forward(model)
        changed_batch = {key: value.clone() if torch.is_tensor(value) else value for key, value in self.batch.items()}
        changed_batch["origin"][:, -1] = 1
        changed_batch["dest"][:, -1] = 3
        changed_batch["tau"][:, -1] = 99
        changed_batch["hour"][:, -1] = 23
        changed_batch["dow"][:, -1] = 6
        changed = self.forward(model, changed_batch)
        for key in ("o_prob", "d_prob", "w", "beta_hat", "lambda_hat", "gamma_hat"):
            torch.testing.assert_close(original[key][:, -1], changed[key][:, -1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
