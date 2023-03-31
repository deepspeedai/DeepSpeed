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

    def gen_output_name(self, test_config, prefix):
        other_args = test_config["other_args"] if "other_args" in test_config else ""
        zero_args = "_zero" if "zero" in test_config and test_config["zero"] else ""
        other_args = other_args.strip(' -\\').replace(" ", "").replace("\"", "")

        if other_args:
            other_args = "_" + other_args

        if test_config["deepspeed"]:
            file_name = "_gpu{0}_{1}_ds{2}-{3}.log".format(test_config["gpus"], other_args, zero_args, self.timestr)
            save_dir = self.test_dir
        else:
            file_name = "_gpu{0}_{1}.log".format(test_config["gpus"], other_args)
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

    def run_BingBertSquad_test(self, test_config, output):
        ds_flag = " -d --deepspeed_config " + test_config["json"] if test_config["deepspeed"] else " "
        other_args = " " + test_config["other_args"] if "other_args" in test_config else " "

        cmd = "./run_BingBertSquad_sanity.sh -e 1 -g {0} {1} {2}".format(test_config["gpus"], other_args, ds_flag)

        self.ensure_directory_exists(output)
        with open(output, "w") as f:
            print(cmd)
            subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', stdout=f, stderr=f)
