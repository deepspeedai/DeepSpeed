# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Note: please copy webtext data to "Megatron-LM" folder, before running this script.
"""

import unittest
import os
import re
from .BingBertSquad_test_common import BaseTestCase


def grep_loss_from_file(file_name):
    loss = 0.0

    with open(file_name, 'r') as f:
        lines = f.readlines()
        line_filter = "bert_squad_progress: step="
        match_number = re.compile(r'loss=([-+]?[0-9]+\.?[0-9]*(?:[Ee][-+]?[0-9]+)?)')

        for line in lines:
            if line_filter in line:
                loss = re.findall(match_number, line)
                loss = float(loss[0])

    if loss == 0.0:
        print("no loss found in file ", file_name)

    return loss


class BingBertSquadFuncTestCase(BaseTestCase):

    def __init__(self, methodName="DeepSpeed function test on BingBertSquad model"):
        super(BingBertSquadFuncTestCase, self).__init__(methodName)

    def setUp(self):
        self.save_dir = os.getcwd()
        new_dir = os.path.dirname(__file__)
        if new_dir:
            os.chdir(new_dir)

    def tearDown(self):
        os.chdir(self.save_dir)

    def test_gpu4_fp16(self):
        test_config = {
            "gpus": 4,
            "deepspeed": False,
            "json": "deepspeed_bsz24_fp16_config.json",
            "max_steps": 8,
            "max_epoch_steps": 4,
            "other_args": "--fp16 --print_steps 1"
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_gpu4_fp16_zero2(self):
        test_config = {
            "gpus": 4,
            "deepspeed": False,
            "json": "deepspeed_bsz24_fp16_zero2_config.json",
            "max_steps": 8,
            "max_epoch_steps": 4,
            "other_args": "--fp16 --print_steps 1"
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_gpu1_fp16(self):
        test_config = {
            "gpus": 1,
            "deepspeed": False,
            "json": "deepspeed_bsz24_fp16_config.json",
            "max_steps": 8,
            "max_epoch_steps": 4,
            "other_args": "--fp16 --print_steps 1"
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_gpu4_fp32(self):
        test_config = {
            "gpus": 4,
            "deepspeed": False,
            "json": "deepspeed_bsz24_fp32_config.json",
            "max_steps": 8,
            "max_epoch_steps": 4,
            "other_args": "--print_steps 1"
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def test_gpu1_fp32(self):
        test_config = {
            "gpus": 1,
            "deepspeed": False,
            "json": "deepspeed_bsz24_fp32_config.json",
            "max_steps": 8,
            "max_epoch_steps": 4,
            "other_args": "--print_steps 1"
        }

        succ = self.run_test(test_config, 0.01)
        self.assertTrue(succ)

    def run_test(self, test_config, r_tol):
        print("\n")
        print("{0}: starting......".format(self.id()))

        prefix = "BingBertSquad_func"

        test_config['other_args'] += f" --max_steps {test_config['max_steps']}"
        test_config['other_args'] += f" --max_steps_per_epoch {test_config['max_epoch_steps']}"

        # baseline run...
        test_config["deepspeed"] = False
        base_file = self.gen_output_name(test_config, prefix)

        # skip baseline run if it exists.
        if not self.has_loss_data(base_file):
            print("{0}: baseline run.".format(self.id()))
            self.run_BingBertSquad_test(test_config, base_file)
        else:
            print("{0}: baseline exists.".format(self.id()))

        # DeepSpeed run...
        test_config["deepspeed"] = True
        print("{0}: DeepSpeed run.".format(self.id()))
        test_file = self.gen_output_name(test_config, prefix)
        self.run_BingBertSquad_test(test_config, test_file)

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
    suite.addTest(BingBertSquadFuncTestCase('test_gpu4_fp16'))
    suite.addTest(BingBertSquadFuncTestCase('test_gpu4_fp16_zero2'))
    suite.addTest(BingBertSquadFuncTestCase('test_gpu1_fp16'))
    suite.addTest(BingBertSquadFuncTestCase('test_gpu4_fp32'))
    suite.addTest(BingBertSquadFuncTestCase('test_gpu1_fp32'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    runner.run(suite())
