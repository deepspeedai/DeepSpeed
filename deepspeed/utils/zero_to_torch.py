#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import argparse

if __package__:
    from . import zero_to_fp32
else:
    import zero_to_fp32


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Convert a DeepSpeed ZeRO checkpoint to float32, float16, or bfloat16 PyTorch weights.")
    parser.add_argument("checkpoint_dir",
                        type=str,
                        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument("output_dir", type=str, help="directory for the converted PyTorch state_dict files")
    parser.add_argument("--dtype",
                        type=str,
                        choices=sorted(zero_to_fp32.OUTPUT_DTYPE_NAMES),
                        required=True,
                        help="output tensor dtype")
    parser.add_argument("--max_shard_size",
                        type=str,
                        default="5GB",
                        help="maximum size of each checkpoint shard, such as 5GB or 500MB")
    parser.add_argument("--safe_serialization",
                        default=False,
                        action='store_true',
                        help="save with safetensors instead of PyTorch pickle serialization")
    parser.add_argument("-t",
                        "--tag",
                        type=str,
                        default=None,
                        help="checkpoint tag used as a unique identifier, e.g., global_step1")
    parser.add_argument("--exclude_frozen_parameters", action='store_true', help="exclude frozen parameters")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug output")
    parsed_args = parser.parse_args(args)

    zero_to_fp32.debug = parsed_args.debug
    zero_to_fp32.convert_zero_checkpoint_to_state_dict(parsed_args.checkpoint_dir,
                                                       parsed_args.output_dir,
                                                       dtype=parsed_args.dtype,
                                                       max_shard_size=parsed_args.max_shard_size,
                                                       safe_serialization=parsed_args.safe_serialization,
                                                       tag=parsed_args.tag,
                                                       exclude_frozen_parameters=parsed_args.exclude_frozen_parameters)


if __name__ == "__main__":
    main()
