# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Note: please copy webtext data to "Megatron-LM" folder, before running this script.
"""

import sys
import unittest

sys.path.append('../DeepSpeedExamples/Megatron_GPT2')
sys.path.append('../DeepSpeedExamples/BingBertSquad')

# Import the test cases here.
import Megatron_GPT2
import BingBertSquad


def pytest_hack(runner_result):
    '''This is an ugly hack to get the unittest suites to play nicely with
    pytest. Otherwise failed tests are not reported by pytest for some reason.

    Long-term, these model tests should be adapted to pytest.
    '''
    if not runner_result.wasSuccessful():
        print('SUITE UNSUCCESSFUL:', file=sys.stderr)
        for fails in runner_result.failures:
            print(fails, file=sys.stderr)
        assert runner_result.wasSuccessful()  # fail the test


def test_megatron():
    runner = unittest.TextTestRunner(failfast=True)
    pytest_hack(runner.run(Megatron_GPT2.suite()))


def test_megatron_checkpoint():
    runner = unittest.TextTestRunner(failfast=True)
    pytest_hack(runner.run(Megatron_GPT2.checkpoint_suite()))


def test_squad():
    runner = unittest.TextTestRunner(failfast=True)
    pytest_hack(runner.run(BingBertSquad.suite()))
