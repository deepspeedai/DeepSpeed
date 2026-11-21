# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
from unittest import mock

import pytest

import deepspeed
import deepspeed.comm as dist
import deepspeed.runtime.engine as ds_engine
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def _tag_warnings(warning_mock):
    return [c for c in warning_mock.call_args_list if "already contains" in str(c.args[0])]


@pytest.mark.parametrize("tag_validation", ["Warn", "Ignore"])
class TestExistingTagWarning(DistributedTest):
    world_size = 2

    def test_warns_only_when_tag_dir_has_files(self, tmpdir, tag_validation):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                    "torch_adam": True
                }
            },
            "zero_optimization": {
                "stage": 1
            },
            "checkpoint": {
                "tag_validation": tag_validation
            },
        }
        model = SimpleModel(hidden_dim=4)
        engine, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        save_dir = str(tmpdir)
        expected = 1 if dist.get_rank() == 0 else 0

        if dist.get_rank() == 0:
            # Delay rank 0's check, so that if it ran after the point where other ranks start writing,
            # rank 1's ZeRO shard would already be in the fresh tag and the check would see it.
            check = engine._warn_if_checkpoint_tag_exists

            def delayed_check(*args, **kwargs):
                time.sleep(2)
                return check(*args, **kwargs)

            engine._warn_if_checkpoint_tag_exists = delayed_check

        # Every rank runs every save before asserting, so a failed check on one rank cannot leave
        # the other ranks waiting in a collective.
        counts = []
        with mock.patch.object(ds_engine.logger, "warning") as warning:
            # Fresh tag: nothing to overwrite, and other ranks' writes into it must not be seen as existing files.
            engine.save_checkpoint(save_dir, tag="step1")
            counts.append(len(_tag_warnings(warning)))

            # An empty tag directory (e.g. created by a save that crashed early) holds nothing to mix with.
            if dist.get_rank() == 0:
                os.makedirs(os.path.join(save_dir, "step2"))
            dist.barrier()
            engine.save_checkpoint(save_dir, tag="step2")
            counts.append(len(_tag_warnings(warning)))

            # Re-saving into a tag that already holds a checkpoint rewrites it in place; only rank 0 warns.
            engine.save_checkpoint(save_dir, tag="step1")
            counts.append(len(_tag_warnings(warning)))

            # Further re-saves under the same tag do not warn again.
            engine.save_checkpoint(save_dir, tag="step1")
            counts.append(len(_tag_warnings(warning)))

            calls = _tag_warnings(warning)

        assert counts == [0, 0, expected, expected]
        if expected:
            assert "'step1'" in calls[0].args[0]
