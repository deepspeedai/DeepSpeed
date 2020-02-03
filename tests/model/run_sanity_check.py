# coding=utf-8
# Copyright (c) 2019, The Microsoft DeepSpeed Team. All rights reserved.
#
# Note: please copy webtext data to "Megatron-LM" folder, before running this script.

import sys
import unittest

sys.path.append('../DeepSpeedExamples/Megatron_GPT2')

import os

# Import the test cases here.
import Megatron_GPT2


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


def test_run():
    runner = unittest.TextTestRunner(failfast=True)

    # Add test suites here.
    pytest_hack(runner.run(Megatron_GPT2.suite()))
    pytest_hack(runner.run(Megatron_GPT2.checkpoint_suite()))


if __name__ == '__main__':
    test_run()
