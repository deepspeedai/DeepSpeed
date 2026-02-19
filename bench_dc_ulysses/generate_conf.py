# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
from jinja2 import Template
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Config generation')

    parser.add_argument('--machine_rank', type=int, help='machine_rank')
    parser.add_argument('--num_machines', type=int, help='num_machines')
    parser.add_argument('--num_processes', type=int, help='num_processes')
    parser.add_argument('--zero_stage', type=int, choices=[0, 1, 2, 3], help='ZeRO stage')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--deepcompile', action='store_true', help='Use deepcompile')

    parser.add_argument('--template_file', type=Path, help='Template file')
    parser.add_argument('--output_file', type=Path, help='Output file')

    return parser.parse_args()


def main(args):
    with open(args.template_file, 'r') as f:
        template = Template(f.read())

    with open(args.output_file, 'w') as f:
        f.write(template.render(machine_rank=args.machine_rank,
                                num_machines=args.num_machines,
                                num_processes=args.num_processes,
                                zero_stage=args.zero_stage,
                                fp16=args.fp16,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                deepcompile=str(args.deepcompile).lower()))

if __name__ == '__main__':
    args = get_args()
    main(args)
