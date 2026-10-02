"""Prevent silent evaluation of a checkpoint on different inputs."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from amtpp.train import topology_arrays
from amtpp.utils.provenance import file_sha256, verify_checkpoint_inputs


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.state = {"metro_cfg": {"split_seed": 42}, "corpus_summary": {"users": 30}, "data_sha256": "data-a", "topology": {"path": "/old/network.npz", "sha256": "network-a", "mode": "graph"}}
        self.arguments = {"data_config": self.state["metro_cfg"], "corpus_summary": self.state["corpus_summary"], "data_sha256": "data-a", "topology": dict(self.state["topology"], path="/new/network.npz"), "check_topology": True}

    def test_same_inputs_may_be_relocated(self):
        verification = verify_checkpoint_inputs(self.state, **self.arguments)
        self.assertTrue(all(verification.values()))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            first.write_bytes(b"identical input")
            second.write_bytes(first.read_bytes())
            self.assertEqual(file_sha256(first), file_sha256(second))

    def test_changed_inputs_are_rejected(self):
        changes = {"data_config": {"split_seed": 43}, "corpus_summary": {"users": 31}, "data_sha256": "data-b", "topology": dict(self.arguments["topology"], sha256="network-b")}
        for name, value in changes.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                verify_checkpoint_inputs(self.state, **dict(self.arguments, **{name: value}))

    def test_old_checkpoint_reports_unverified_data_identity(self):
        state = copy.deepcopy(self.state)
        del state["data_sha256"]
        with self.assertWarnsRegex(RuntimeWarning, "exact data identity cannot be verified"):
            verification = verify_checkpoint_inputs(state, **self.arguments)
        self.assertFalse(verification["data_sha256_verified"])

    def test_invalid_topology_is_rejected_before_alignment(self):
        class Corpus:
            S = 3
            station_to_index = {10: 1, 11: 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.npz"
            asymmetric = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0]])
            np.savez(path, raw_station_id=np.array([10, 11]), shortest_hops=asymmetric)
            with self.assertRaisesRegex(ValueError, "symmetric"):
                topology_arrays(path, Corpus())
