# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import pytest
import deepspeed
from deepspeed.utils.numa import parse_range_list


def basic_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int)
    return parser


def test_no_ds_arguments_no_ds_parser():
    parser = basic_parser()
    args = parser.parse_args(['--num_epochs', '2'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert not hasattr(args, 'deepspeed')
    assert not hasattr(args, 'deepspeed_config')


def test_no_ds_arguments():
    parser = basic_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepspeed')
    assert args.deepspeed == False

    assert hasattr(args, 'deepspeed_config')
    assert args.deepspeed_config == None


def test_no_ds_enable_argument():
    parser = basic_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2', '--deepspeed_config', 'foo.json'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepspeed')
    assert args.deepspeed == False

    assert hasattr(args, 'deepspeed_config')
    assert type(args.deepspeed_config) == str
    assert args.deepspeed_config == 'foo.json'


def test_no_ds_config_argument():
    parser = basic_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2', '--deepspeed'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepspeed')
    assert type(args.deepspeed) == bool
    assert args.deepspeed == True

    assert hasattr(args, 'deepspeed_config')
    assert args.deepspeed_config == None


def test_no_ds_parser():
    parser = basic_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args(['--num_epochs', '2', '--deepspeed'])


def test_core_deepscale_arguments():
    parser = basic_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args(['--num_epochs', '2', '--deepspeed', '--deepspeed_config', 'foo.json'])
    assert args

    assert hasattr(args, 'num_epochs')
    assert args.num_epochs == 2

    assert hasattr(args, 'deepspeed')
    assert type(args.deepspeed) == bool
    assert args.deepspeed == True

    assert hasattr(args, 'deepspeed_config')
    assert type(args.deepspeed_config) == str
    assert args.deepspeed_config == 'foo.json'


def test_core_binding_arguments():
    core_list = parse_range_list("0,2-4,6,8-9")
    assert core_list == [0, 2, 3, 4, 6, 8, 9]

    try:
        # negative case for range overlapping
        core_list = parse_range_list("0,2-6,5-9")
    except ValueError as e:
        pass
    else:
        # invalid core list must fail
        assert False

    try:
        # negative case for reverse order -- case 1
        core_list = parse_range_list("8,2-6")
    except ValueError as e:
        pass
    else:
        # invalid core list must fail
        assert False

    try:
        # negative case for reverse order -- case 2
        core_list = parse_range_list("1,6-2")
    except ValueError as e:
        pass
    else:
        # invalid core list must fail
        assert False
