# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import unittest
import subprocess
import os
import time


class BaseTestCase(unittest.TestCase):

    def __init__(self, methodName="DeepSpeed performance test"):
        super(BaseTestCase, self).__init__(methodName)
        self.test_dir = "./test"
        self.baseline_dir = "./baseline"
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

    def gen_output_name(self, test_config, prefix, baseline_config=False):
        other_args = test_config["other_args"] if "other_args" in test_config else ""
        zero_args = "_zero" if "zero" in test_config and test_config["zero"] else ""
        other_args = other_args.strip(' -\\').replace(" ", "").replace("\"", "")

        if other_args:
            other_args = "_" + other_args

        if test_config["deepspeed"] and not baseline_config:
            file_name = "_mp{0}_gpu{1}_node{2}_bs{3}_step{4}_layer{5}_hidden{6}_seq{7}_head{8}{9}_ds{10}-{11}.log".format(
                test_config["mp"], test_config["gpus"], test_config["nodes"], test_config["bs"], test_config["steps"],
                test_config["layers"], test_config["hidden_size"], test_config["seq_length"], test_config["heads"],
                other_args, zero_args, self.timestr)
            save_dir = self.test_dir
        else:
            file_name = "_mp{0}_gpu{1}_node{2}_bs{3}_step{4}_layer{5}_hidden{6}_seq{7}_head{8}{9}.log".format(
                test_config["mp"], test_config["gpus"], test_config["nodes"], test_config["bs"], test_config["steps"],
                test_config["layers"], test_config["hidden_size"], test_config["seq_length"], test_config["heads"],
                other_args)
            save_dir = self.baseline_dir

        return os.path.join(save_dir, prefix + file_name)

    def ensure_directory_exists(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def clean_test_env(self):
        cmd = "dlts_ssh pkill -9 -f /usr/bin/python"
        print(cmd)
        subprocess.run(cmd, shell=True, check=False, executable='/bin/bash')
        time.sleep(20)

    def run_gpt2_test(self, test_config, output):
        ds_flag = "-d " + test_config["json"] if test_config["deepspeed"] else ""
        ckpt_num = test_config["ckpt_num_layers"] if "ckpt_num_layers" in test_config else 1
        other_args = "-o " + test_config["other_args"] if "other_args" in test_config else ""

        cmd = "./ds_gpt2_test.sh -m {0} -g {1} -n {2} -b {3} -s {4} -l {5} -h {6} -q {7} -e {8} -c {9} {10} {11}".format(
            test_config["mp"], test_config["gpus"], test_config["nodes"], test_config["bs"], test_config["steps"],
            test_config["layers"], test_config["hidden_size"], test_config["seq_length"], test_config["heads"],
            ckpt_num, other_args, ds_flag)

        self.ensure_directory_exists(output)
        with open(output, "w") as f:
            print(cmd)
            subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', stdout=f, stderr=f)
