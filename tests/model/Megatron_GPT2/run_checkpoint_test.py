# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Note: please copy webtext data to "Megatron-LM" folder, before running this script.
"""

import unittest
import subprocess
import os
import re
from .test_common import BaseTestCase

LAYERS = 2
HIDDEN_SIZE = 128
ATTN_HEADS = 8


def remove_file(test_id, filename):
    cmd = f"if [ -f {filename} ] ; then rm -v {filename}; fi"
    print(f"{test_id} cmd: {cmd}")
    subprocess.run(cmd, shell=True, check=False, executable='/bin/bash')


def grep_loss_from_file(file_name):
    loss = 0.0

    with open(file_name, 'r') as f:
        lines = f.readlines()
        line_filter = "validation loss at the end of training for test data | LM loss:"
        match_number = re.compile(r'LM loss: ([-+]?[0-9]+\.?[0-9]*(?:[Ee][-+]?[0-9]+)?)')

        for line in lines:
            if line_filter in line:
                loss = re.findall(match_number, line)
                loss = float(loss[0])

    if loss == 0.0:
        print("no loss found in file ", file_name)

    return loss


class GPT2CheckpointTestCase(BaseTestCase):

    def __init__(self, methodName="DeepSpeed function test on GPT2 model"):
        super(GPT2CheckpointTestCase, self).__init__(methodName)

    def setUp(self):
        self.save_dir = os.getcwd()
        new_dir = os.path.dirname(__file__)
        if new_dir:
            os.chdir(new_dir)

    def tearDown(self):
        os.chdir(self.save_dir)

    def test_mp2_gpu4_node1_with_zero1(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero1",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu8_w_zero1",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero1.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_with_zero2(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu8_w_zero2",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_with_zero2_offload(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2_offload",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu8_w_zero2_offload",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_load_gpu1_node1_with_zero1(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "load_gpus": 1,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero1",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp1_gpu2_gpu1_w_zero1",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero1.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_load_gpu4_node1_with_zero1(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "load_gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero1",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp1_gpu2_gpu4_w_zero1",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero1.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_load_gpu1_node1_with_zero2(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "load_gpus": 1,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp1_gpu2_gpu1_w_zero2",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_load_gpu1_node1_with_zero2_offload(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "load_gpus": 1,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2_offload",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp1_gpu2_gpu1_w_zero2_offload",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_load_gpu4_node1_with_zero2(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "load_gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp1_gpu2_gpu4_w_zero2",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_load_gpu4_node1_with_zero2_offload(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "load_gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2_offload",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp1_gpu2_gpu4_w_zero2_offload",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_load_gpu2_node1_with_zero1(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "load_gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero1",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu4_gpu2_w_zero1",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero1.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu2_load_gpu4_node1_with_zero1(self):
        test_config = {
            "mp": 2,
            "gpus": 2,
            "load_gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero1",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu2_gpu4_w_zero1",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero1.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_load_gpu2_node1_with_zero2(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "load_gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu4_gpu2_w_zero2",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_load_gpu2_node1_with_zero2_offload(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "load_gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2_offload",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu4_gpu2_w_zero2_offload",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu2_load_gpu4_node1_with_zero2(self):
        test_config = {
            "mp": 2,
            "gpus": 2,
            "load_gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu2_gpu4_w_zero2",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu2_load_gpu4_node1_with_zero2_offload(self):
        test_config = {
            "mp": 2,
            "gpus": 2,
            "load_gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "tag": "ds_zero2_offload",
            "zero": True,
            "other_args": "",
            "checkpoint_name": "ckpt_mp2_gpu2_gpu4_w_zero2_offload",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_without_zero(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1100,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": 256,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "zero": False,
            "other_args": "",
            "tag": "ds_without_zero",
            "checkpoint_name": "ckpt_mp4_gpu16_wo_zero",
            "checkpoint_interval": 1000,
            "json": "ds_config_func_bs8_no_zero.json",
        }
        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def gen_name(self, test_config, prefix):
        save_dir = "checkpoint_test_logs"
        tag = test_config["tag"]
        checkpoint_name = test_config["checkpoint_name"]
        file_name = f"_{tag}_{checkpoint_name}.log"
        return os.path.join(save_dir, prefix + file_name)

    def run_test(self, test_config, r_tol):
        print("\n")

        print("{0}: starting......".format(self.id()))

        # Cache save and load gpu counts
        save_gpus = test_config["gpus"]
        if "load_gpus" in test_config:
            load_gpus = test_config["load_gpus"]
            del test_config["load_gpus"]
        else:
            load_gpus = test_config["gpus"]

        # save to current directory.
        checkpoint_folder = test_config["checkpoint_name"]
        checkpoint_interval = test_config["checkpoint_interval"]
        checkpoint_name = test_config["checkpoint_name"]
        #---------------remove old checkpoint---------------#
        try:
            cmd = f"rm -rf {checkpoint_name}"
            print(f"{self.id()} cmd: {cmd}")
            subprocess.run(cmd, shell=True, check=False, executable='/bin/bash')
        except:
            print("No old checkpoint")

        if "cpu_optimizer" in test_config and test_config["cpu_optimizer"]:
            cpu_optimizer_flag = " --cpu-optimizer"
        else:
            cpu_optimizer_flag = ""

        #-----------------Saving Checkpoint-----------------#
        # building checkpoint arguments
        test_config[
            "other_args"] = f"\"--save {checkpoint_folder} --save-interval {checkpoint_interval} {cpu_optimizer_flag}\""

        prefix = "gpt2_saving_checkpoint"

        # create checkpoint run...
        base_file = self.gen_name(test_config, prefix)

        # remove previous test log
        try:
            cmd = f"rm {base_file}"
            subprocess.run(cmd, shell=True, check=False, executable='/bin/bash')
        except:
            print(f"{self.id()} No old logs")

        print("{0}: Run for saving checkpoint".format(self.id()))
        self.run_gpt2_test(test_config, base_file)

        #-----------------Loading Checkpoint-----------------#

        # building checkpoint arguments
        test_config["other_args"] = f"\"--load {checkpoint_folder} {cpu_optimizer_flag} \""

        # set checkpoint load iteration
        try:
            cmd = f"echo {checkpoint_interval} > {checkpoint_name}/latest_checkpointed_iteration.txt"
            print(f"{self.id()} running cmd: {cmd}")
            subprocess.run(cmd, shell=True, check=False, executable='/bin/bash')
        except:
            print(f"{self.id()} Failed to update the checkpoint iteration file")
            return False

        prefix = "gpt2_loading_checkpoint"

        # set load gpus
        test_config["gpus"] = load_gpus

        print("{0}: Second run loading checkpoint and continuing.".format(self.id()))
        test_file = self.gen_name(test_config, prefix)

        # remove previous test log
        try:
            cmd = f"rm {test_file}"
            subprocess.run(cmd, shell=True, check=False, executable='/bin/bash')
        except:
            print(f"{self.id()} no previous logs for")
        self.run_gpt2_test(test_config, test_file)
        return self.check_parity(base_file, test_file, r_tol)

    def has_loss_data(self, file_name):
        has_loss = False
        if os.path.exists(file_name):
            loss = grep_loss_from_file(file_name)
            if loss != 0.0:
                has_loss = True

        return has_loss

    def check_parity(self, base_file, test_file, r_tol):
        base_loss = grep_loss_from_file(base_file)
        test_loss = grep_loss_from_file(test_file)

        print("baseline loss: {0}, test loss: {1}".format(base_loss, test_loss))

        if base_loss == 0.0 or test_loss == 0.0:
            return False

        if abs((base_loss - test_loss) / base_loss) > r_tol:
            return False

        return True


def checkpoint_suite():
    suite = unittest.TestSuite()

    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_node1_with_zero1'))
    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_node1_with_zero2'))
    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_node1_with_zero2_offload'))

    # Shrink DP
    suite.addTest(GPT2CheckpointTestCase('test_mp1_gpu2_load_gpu1_node1_with_zero1'))
    suite.addTest(GPT2CheckpointTestCase('test_mp1_gpu2_load_gpu1_node1_with_zero2'))
    suite.addTest(GPT2CheckpointTestCase('test_mp1_gpu2_load_gpu1_node1_with_zero2_offload'))

    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_load_gpu2_node1_with_zero1'))
    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_load_gpu2_node1_with_zero2'))
    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_load_gpu2_node1_with_zero2_offload'))

    # Expand DP
    suite.addTest(GPT2CheckpointTestCase('test_mp1_gpu2_load_gpu4_node1_with_zero1'))
    suite.addTest(GPT2CheckpointTestCase('test_mp1_gpu2_load_gpu4_node1_with_zero2'))
    suite.addTest(GPT2CheckpointTestCase('test_mp1_gpu2_load_gpu4_node1_with_zero2_offload'))

    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu2_load_gpu4_node1_with_zero1'))
    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu2_load_gpu4_node1_with_zero2'))
    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu2_load_gpu4_node1_with_zero2_offload'))

    suite.addTest(GPT2CheckpointTestCase('test_mp2_gpu4_node1_without_zero'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(checkpoint_suite())
