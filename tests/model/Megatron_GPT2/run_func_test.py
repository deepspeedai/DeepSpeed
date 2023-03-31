# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Note: please copy webtext data to "Megatron-LM" folder, before running this script.
"""

import unittest
import os
import re
from .test_common import BaseTestCase

LAYERS = 2
HIDDEN_SIZE = 128
ATTN_HEADS = 8
SEQ_LEN = 64
MASTER_PORT = 29700


def grep_loss_from_file(file_name):
    loss = 0.0
    print(f'grepping {file_name}')
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


class GPT2FuncTestCase(BaseTestCase):

    def __init__(self, methodName="DeepSpeed function test on GPT2 model"):
        super(GPT2FuncTestCase, self).__init__(methodName)

    def setUp(self):
        self.save_dir = os.getcwd()
        new_dir = os.path.dirname(__file__)
        if new_dir:
            os.chdir(new_dir)

    def tearDown(self):
        os.chdir(self.save_dir)

    def test_mp1_gpu2_node1_fp16(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_no_zero.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu1_node1_zero1(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs4_zero1.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_node1_zero1(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero1.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_zero1(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero1.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp4_gpu4_node1_zero1(self):
        test_config = {
            "mp": 4,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero1.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu1_node1_zero2(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs4_zero2.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_node1_zero2(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_zero2(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2.json",
        }

        basic_run_config = test_config
        succ = self.run_test(basic_run_config, 0.01)
        self.assertTrue(succ)

        partition_activation_config = test_config
        succ = self.run_partition_activations_test(partition_activation_config, 0.01)
        self.assertTrue(succ)

    def test_mp4_gpu4_node1_zero2(self):
        test_config = {
            "mp": 4,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2.json",
        }

        basic_run_config = test_config
        succ = self.run_test(basic_run_config, 0.01)
        self.assertTrue(succ)

        partition_activation_config = test_config
        succ = self.run_partition_activations_test(partition_activation_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu1_node1_zero2_ds_offload(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs4_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.02)
        self.assertTrue(succ)

    def test_mp1_gpu2_node1_zero2_ds_offload(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }
        succ = self.run_test(test_config, 0.02)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_zero2_gas(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": True,
            "json": "ds_config_func_bs8_zero2_gas3.json",
            "baseline": "ds_config_func_bs8_zero0_gas3.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

        succ = self.run_partition_activations_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_zero2_ds_offload(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }

        basic_run_config = test_config
        succ = self.run_test(basic_run_config, 0.02)
        self.assertTrue(succ)

        partition_activation_config = test_config
        succ = self.run_partition_activations_test(partition_activation_config, 0.02)
        self.assertTrue(succ)

    def test_mp4_gpu4_node1_zero2_ds_offload(self):
        test_config = {
            "mp": 4,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
        }

        basic_run_config = test_config
        succ = self.run_test(basic_run_config, 0.02)
        self.assertTrue(succ)

        partition_activation_config = test_config
        succ = self.run_partition_activations_test(partition_activation_config, 0.02)
        self.assertTrue(succ)

    def test_mp1_gpu1_node1_zero2_torch_offload(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs4_zero2_offload.json",
            "cpu_optimizer": True,
            "test_torch_offload": True,
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp1_gpu2_node1_zero2_torch_offload(self):
        test_config = {
            "mp": 1,
            "gpus": 2,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
            "test_torch_offload": True,
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp2_gpu4_node1_zero2_torch_offload(self):
        test_config = {
            "mp": 2,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
            "test_torch_offload": True,
        }

        basic_run_config = test_config
        succ = self.run_test(basic_run_config, 0.01)
        self.assertTrue(succ)

        partition_activation_config = test_config
        succ = self.run_partition_activations_test(partition_activation_config, 0.01)
        self.assertTrue(succ)

    def test_mp4_gpu4_node1_zero2_torch_offload(self):
        test_config = {
            "mp": 4,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2_offload.json",
            "cpu_optimizer": True,
            "test_torch_offload": True,
        }

        basic_run_config = test_config
        succ = self.run_test(basic_run_config, 0.01)
        self.assertTrue(succ)

        partition_activation_config = test_config
        succ = self.run_partition_activations_test(partition_activation_config, 0.01)

    def test_optimizer_scheduler(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 20,
            "layers": LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "seq_length": SEQ_LEN,
            "heads": ATTN_HEADS,
            "deepspeed": False,
            "json": "ds_config_func_scheduler.json",
        }

        succ = self.run_test(test_config, 0.01)
        # assure no crash.
        self.assertTrue(True)

    def run_partition_activations_test(self, test_config, r_tol):
        print("\n")
        print("{0}: starting......".format(self.id()))

        baseline_prefix = "gpt2_func_"
        prefix = "gpt2_partition_activation_"

        deepspeed_config = test_config["json"]
        baseline_deepspeed_config = False
        cpu_optimizer_flag = self.gen_cpu_optimizer_flag(test_config, True)

        # baseline run...
        # turnoff deepspeed if baseline deepspeed config
        # is not provided
        if not "baseline" in test_config:
            test_config["deepspeed"] = False
        else:
            test_config["json"] = test_config["baseline"]
            baseline_prefix += test_config["json"][0:-5]
            baseline_deepspeed_config = True

        test_config["other_args"] = f"\"{cpu_optimizer_flag}\""
        base_file = self.gen_output_name(test_config, baseline_prefix, baseline_config=baseline_deepspeed_config)

        # skip baseline run if it exists.
        if not self.has_loss_data(base_file):
            print("{0}: baseline run.".format(self.id()))
            self.run_gpt2_test(test_config, base_file)
        else:
            print("{0}: baseline exists.".format(self.id()))

        # DeepSpeed run...
        test_config["deepspeed"] = True
        cpu_optimizer_flag = self.gen_cpu_optimizer_flag(test_config, False)
        test_config["other_args"] = f"\"--deepspeed-activation-checkpointing {cpu_optimizer_flag}\""
        test_config["json"] = deepspeed_config

        print("{0}: DeepSpeed run.".format(self.id()))
        test_file = self.gen_output_name(test_config, prefix)
        self.run_gpt2_test(test_config, test_file)

        return self.check_parity(base_file, test_file, r_tol)

    def run_test(self, test_config, r_tol):
        print("\n")
        print("{0}: starting......".format(self.id()))

        prefix = "gpt2_func"
        baseline_prefix = prefix

        deepspeed_config = test_config["json"]
        baseline_deepspeed_config = False
        cpu_optimizer_flag = self.gen_cpu_optimizer_flag(test_config, True)

        # baseline run...
        # turn off deepspeed if a baseline deepspeed config
        # is not provided
        if not "baseline" in test_config:
            test_config["deepspeed"] = False
        else:
            test_config["json"] = test_config["baseline"]
            baseline_prefix = prefix + test_config["json"][0:-5]
            baseline_deepspeed_config = True

        test_config["other_args"] = f"\"{cpu_optimizer_flag}\""

        # baseline run...
        base_file = self.gen_output_name(test_config, baseline_prefix, baseline_config=baseline_deepspeed_config)

        # skip baseline run if it exists.
        if not self.has_loss_data(base_file):
            print("{0}: baseline run.".format(self.id()))
            self.run_gpt2_test(test_config, base_file)
        else:
            print("{0}: baseline exists.".format(self.id()))

        # DeepSpeed run...
        test_config["deepspeed"] = True
        cpu_optimizer_flag = self.gen_cpu_optimizer_flag(test_config, False)
        test_config["other_args"] = f"\"{cpu_optimizer_flag}\""

        print("{0}: DeepSpeed run.".format(self.id()))
        test_file = self.gen_output_name(test_config, prefix)
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

    def gen_cpu_optimizer_flag(self, test_config, is_baseline):
        if 'cpu_optimizer' in test_config and test_config['cpu_optimizer']:
            cpu_optimizer_flag = "--cpu-optimizer"
            if is_baseline:
                cpu_optimizer_flag += " --cpu_torch_adam"
                return cpu_optimizer_flag
            if 'test_torch_offload' in test_config and test_config['test_torch_offload']:
                cpu_optimizer_flag += " --cpu_torch_adam"
                return cpu_optimizer_flag
        else:
            cpu_optimizer_flag = ""

        return cpu_optimizer_flag


def suite():
    suite = unittest.TestSuite()

    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_fp16'))

    # Baseline = Megatron + Torch.Optim.Adam
    # Test = Megatron + Torch.Optim.Adam + ZeRO-Offload
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu1_node1_zero2_torch_offload'))
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_zero2_torch_offload'))
    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero2_torch_offload'))
    suite.addTest(GPT2FuncTestCase('test_mp4_gpu4_node1_zero2_torch_offload'))

    # Baseline = Megatron + Torch.Optim.Adam
    # Test = Megatron + DeepSpeedAdam + ZeRO-Offload
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu1_node1_zero2_ds_offload'))
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_zero2_ds_offload'))
    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero2_ds_offload'))
    suite.addTest(GPT2FuncTestCase('test_mp4_gpu4_node1_zero2_ds_offload'))

    suite.addTest(GPT2FuncTestCase('test_mp1_gpu1_node1_zero1'))
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_zero1'))
    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero1'))
    suite.addTest(GPT2FuncTestCase('test_mp4_gpu4_node1_zero1'))

    suite.addTest(GPT2FuncTestCase('test_mp1_gpu1_node1_zero2'))
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_zero2'))
    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero2'))
    suite.addTest(GPT2FuncTestCase('test_mp4_gpu4_node1_zero2'))

    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero2_gas'))

    suite.addTest(GPT2FuncTestCase('test_optimizer_scheduler'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
