# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import os
import pytest
import deepspeed
from deepspeed.utils.numa import get_numactl_cmd, parse_range_list


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
    assert args.deepspeed_config is None


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
    assert args.deepspeed_config is None


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


FAKE_NUMACTL = """#!/bin/sh
# Stand-in for numactl: reports one 8-core NUMA node and rejects the options listed in
# FAKE_NUMACTL_DENIED, the way a container without CAP_SYS_NICE rejects memory policies.
if [ "$1" = "--hardware" ]; then
    echo "available: 1 nodes (0)"
    echo "node 0 cpus: 0 1 2 3 4 5 6 7"
    exit 0
fi
for arg in "$@"; do
    case " $FAKE_NUMACTL_DENIED " in
        *" $arg "*)
            echo "set_mempolicy: Operation not permitted" >&2
            exit 1
            ;;
    esac
done
exit 0
"""


@pytest.mark.parametrize("denied_options, expected_cmd", [
    ("", ["numactl", "-m", "0", "-C", "0-3"]),
    ("-m", ["numactl", "-C", "0-3"]),
    ("-m -C", []),
])
def test_numactl_cmd_fallback(tmp_path, monkeypatch, denied_options, expected_cmd):
    # Launching a rank must not fail just because numactl rejects some binding options.
    fake_numactl = tmp_path / "numactl"
    fake_numactl.write_text(FAKE_NUMACTL)
    fake_numactl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_NUMACTL_DENIED", denied_options)
    monkeypatch.delenv("KMP_AFFINITY", raising=False)

    cores_per_rank, numactl_cmd = get_numactl_cmd("0-7", 2, 0)
    assert cores_per_rank == 4
    assert numactl_cmd == expected_cmd
