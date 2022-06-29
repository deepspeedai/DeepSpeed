import torch
from benchmarks.communication.utils import *
from benchmarks.communication.all_reduce import run_allreduce
from benchmarks.communication.all_gather import run_allgather
from benchmarks.communication.all_to_all import run_alltoall
from benchmarks.communication.pt2pt import run_pt2pt
from benchmarks.communication.constants import *

import time
import argparse
import os


# For importing
def main(args, rank):

    init_processes(local_rank=rank, args=args)

    for comm_op in ['allreduce', 'alltoall', 'allgather', 'pt2pt']:
        if comm_op == 'allreduce':
            run_allreduce(local_rank=rank, args=args)
        if comm_op == 'allgather':
            run_allgather(local_rank=rank, args=args)
        if comm_op == 'alltoall':
            run_alltoall(local_rank=rank, args=args)
        if comm_op == 'pt2pt':
            run_pt2pt(local_rank=rank, args=args)


# For directly calling benchmark
if __name__ == "__main__":
    args = benchmark_parser().parse_args()
    rank = args.local_rank
    main(args, rank)
