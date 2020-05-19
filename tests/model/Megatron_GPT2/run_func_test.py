# coding=utf-8
# Copyright (c) 2019, The Microsoft DeepSpeed Team. All rights reserved.
#
# Note: please copy webtext data to "Megatron-LM" folder, before running this script.

import unittest
import subprocess
import os
import time
import re
from .test_common import BaseTestCase


def grep_loss_from_file(file_name):
    loss = 0.0

    with open(file_name, 'r') as f:
        lines = f.readlines()
        line_filter = "validation loss at the end of training for test data | LM loss:"
        match_number = re.compile('LM loss: ([-+]?[0-9]+\.?[0-9]*(?:[Ee][-+]?[0-9]+)?)')

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

    def test_mp1_gpu1_node1_zero1(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 1000,
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

        succ = self.run_partition_activations_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_mp4_gpu4_node1_zero2(self):
        test_config = {
            "mp": 4,
            "gpus": 4,
            "nodes": 1,
            "bs": 8,
            "steps": 1000,
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
            "deepspeed": False,
            "json": "ds_config_func_bs8_zero2.json",
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

        succ = self.run_partition_activations_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_optimizer_scheduler(self):
        test_config = {
            "mp": 1,
            "gpus": 1,
            "nodes": 1,
            "bs": 4,
            "steps": 20,
            "layers": 12,
            "hidden_size": 768,
            "seq_length": 256,
            "heads": 12,
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

        # baseline run...
        test_config["deepspeed"] = False
        base_file = self.gen_output_name(test_config, baseline_prefix)

        # skip baseline run if it exists.
        if not self.has_loss_data(base_file):
            print("{0}: baseline run.".format(self.id()))
            self.run_gpt2_test(test_config, base_file)
        else:
            print("{0}: baseline exists.".format(self.id()))

        # DeepSpeed run...
        test_config["deepspeed"] = True
        test_config["other_args"] = "--deepspeed-activation-checkpointing"
        print("{0}: DeepSpeed run.".format(self.id()))
        test_file = self.gen_output_name(test_config, prefix)
        self.run_gpt2_test(test_config, test_file)

        return self.check_parity(base_file, test_file, r_tol)

    def run_test(self, test_config, r_tol):
        print("\n")
        print("{0}: starting......".format(self.id()))

        prefix = "gpt2_func"

        # baseline run...
        test_config["deepspeed"] = False
        base_file = self.gen_output_name(test_config, prefix)

        # skip baseline run if it exists.
        if not self.has_loss_data(base_file):
            print("{0}: baseline run.".format(self.id()))
            self.run_gpt2_test(test_config, base_file)
        else:
            print("{0}: baseline exists.".format(self.id()))

        # DeepSpeed run...
        test_config["deepspeed"] = True
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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu1_node1_zero1'))
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_zero1'))
    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero1'))
    suite.addTest(GPT2FuncTestCase('test_mp4_gpu4_node1_zero1'))

    suite.addTest(GPT2FuncTestCase('test_mp1_gpu1_node1_zero2'))
    suite.addTest(GPT2FuncTestCase('test_mp1_gpu2_node1_zero2'))
    suite.addTest(GPT2FuncTestCase('test_mp2_gpu4_node1_zero2'))
    suite.addTest(GPT2FuncTestCase('test_mp4_gpu4_node1_zero2'))

    suite.addTest(GPT2FuncTestCase('test_optimizer_scheduler'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
