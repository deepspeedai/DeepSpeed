# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The curriculum sampler must checkpoint the generator it actually draws from.

Every draw in DeepSpeedDataSampler goes through `self.np_rng`, a
`np.random.default_rng`. `state_dict` used to store `np.random.get_state()` --
the process-wide legacy RandomState, which the sampler never touches -- so the
saved value carried no information about how far the sampling stream had run,
and resume replayed it from the seed.
"""

import hashlib

import numpy as np

from deepspeed.runtime.data_pipeline.config import get_data_efficiency_config
from deepspeed.runtime.data_pipeline.constants import CURRICULUM_LEARNING_NP_RNG_STATE
from deepspeed.runtime.data_pipeline.data_sampling.data_sampler import DeepSpeedDataSampler

CLUSTER_SIZES = [10, 20, 30, 40]


def _config():
    metric = {
        "index_to_sample_path": "dummy",
        "index_to_metric_path": "dummy",
        "difficulty_type": "value",
        "clustering_type": "single_cluster",
        "min_difficulty": 8,
        "max_difficulty": 80,
        "schedule_type": "fixed_linear",
        "schedule_config": {
            "total_curriculum_step": 100,
            "difficulty_step": 8
        },
    }
    return get_data_efficiency_config({
        "data_efficiency": {
            "enabled": True,
            "seed": 1234,
            "data_sampling": {
                "enabled": True,
                "curriculum_learning": {
                    "enabled": True,
                    "data_cluster_path": "/tmp/clusters",
                    "curriculum_metrics": {
                        "dummy": metric
                    },
                },
            },
        }
    })


def _sampler():
    sampler = DeepSpeedDataSampler(_config(), 100, 8, 0, 1, None, 1, global_rank=0)
    sampler.data_clusters = [None] * len(CLUSTER_SIZES)
    sampler.data_cluster_sizes = list(CLUSTER_SIZES)
    return sampler


def _state_fingerprint(sampler):
    """A short, comparable digest of the checkpointed RNG state.

    Hashed rather than compared directly so this reads the same whether the
    checkpoint holds a bit-generator dict or the legacy RandomState tuple, and so a
    failure prints a digest instead of 624 words of Mersenne Twister.
    """
    return hashlib.sha256(repr(sampler.state_dict()[CURRICULUM_LEARNING_NP_RNG_STATE]).encode()).hexdigest()[:16]


def test_saved_state_moves_as_the_sampler_draws():
    sampler = _sampler()
    before = _state_fingerprint(sampler)
    for _ in range(3):
        sampler.sample_from_clusters()
    after = _state_fingerprint(sampler)

    assert before != after, "the saved RNG state must reflect the draws the sampler made"


def test_resume_continues_the_stream_rather_than_replaying_it():
    saved = _sampler()
    for _ in range(3):
        saved.sample_from_clusters()
    # data_cluster_paths empty so load_state_dict does no file I/O; everything else
    # goes through the real API.
    checkpoint = dict(saved.state_dict(), data_cluster_paths=[])
    expected = [saved.sample_from_clusters().tolist() for _ in range(3)]

    resumed = _sampler()
    resumed.load_state_dict(checkpoint)
    actual = [resumed.sample_from_clusters().tolist() for _ in range(3)]

    assert actual == expected, "resume must continue the sampling stream, not restart it"

    replayed = _sampler()
    assert [replayed.sample_from_clusters().tolist() for _ in range(3)] != expected, \
        "a sampler on a fresh seed must not already agree -- otherwise this proves nothing"


def test_the_global_numpy_rng_is_left_alone():
    sampler = _sampler()
    for _ in range(3):
        sampler.sample_from_clusters()

    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    sampler.load_state_dict(dict(sampler.state_dict(), data_cluster_paths=[]))
    after = np.random.get_state()[1]

    assert np.array_equal(before, after), "loading must not move the process-wide numpy RNG"


def test_a_legacy_checkpoint_still_loads():
    # Written before this was fixed: a tuple from np.random.get_state().
    sampler = _sampler()
    legacy = dict(sampler.state_dict(), data_cluster_paths=[])
    legacy[CURRICULUM_LEARNING_NP_RNG_STATE] = np.random.get_state()

    sampler.load_state_dict(legacy)  # must not raise
