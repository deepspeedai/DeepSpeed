# The DeepSpeed Communication Benchmarking Suite

The intent of these benchmarks is to measure communication latency/bw of deepspeed and/or pytorch distributed communication operations at the Python layer. These benchmarks are complementary to C-level comms benchmarks like [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) and [NCCL Tests](https://github.com/NVIDIA/nccl-tests) in that users can:
- Easily debug which layer of the communication software stack hangs or performance degradations originate from.
- Measure the expected communication performance of either DeepSpeed comms or pure PyTorch distributed

To run benchmarks, there are two options:

1. Run a single communication operation:

For example, run with a single large message size (calculated to barely fit within GPU mem):
<pre>
deepspeed all_reduce.py
</pre>

Scan across message sizes:
<pre>
deepspeed all_reduce.py --scan
</pre>

Benchmark pure PyTorch distributed comms (without importing or using DeepSpeed) with MPI
<pre>
mpirun -np 16 --hostfile ${HOSTFILE} -x LD_LIBRARY_PATH -x PATH -x LD_PRELOAD python all_reduce.py --scan --dist="torch"
</pre>

or Slurm
<pre>
srun -n 16 python all_reduce.py --scan --dist="torch"
</pre>


2. Run all available communication benchmarks:

<pre>
deepspeed run_all.py
</pre>

Like the individual benchmarks, `run_all.py` supports scanning arguments for the max message size, bw-unit, etc. Simply pass the desired arguments to `run_all.py` and they'll be propagated to each comm op.

<pre>
usage: ds_bench [-h] [--local_rank LOCAL_RANK] [--trials TRIALS] [--warmups WARMUPS] [--maxsize MAXSIZE] [--async-op] [--bw-unit {Gbps,GBps}] [--backend {nccl}] [--dist {deepspeed,torch}] [--scan] [--raw] [--all-reduce] [--all-gather] [--all-to-all]
                [--pt2pt] [--broadcast] [--dtype DTYPE] [--mem-factor MEM_FACTOR] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
  --trials TRIALS       Number of timed iterations
  --warmups WARMUPS     Number of warmup (non-timed) iterations
  --maxsize MAXSIZE     Max message size as a power of 2
  --async-op            Enables non-blocking communication
  --bw-unit {Gbps,GBps}
  --backend {nccl}      Communication library to use
  --dist {deepspeed,torch}
                        Distributed DL framework to use
  --scan                Enables scanning all message sizes
  --raw                 Print the message size and latency without units
  --all-reduce          Run all_reduce
  --all-gather          Run all_gather
  --all-to-all          Run all_to_all
  --pt2pt               Run pt2pt
  --broadcast           Run broadcast
  --dtype DTYPE         PyTorch tensor dtype
  --mem-factor MEM_FACTOR
                        Proportion of max available GPU memory to use for single-size evals
  --debug               Enables all_to_all debug prints
</pre>

Note that `ds_bench` is a pre-packaged wrapper around `run_all.py`. Users can pass the same arguments as well:

<pre>
<path to deepspeed>/bin/ds_bench --scan --trials=10
</pre>

Finally, users can choose specific communication operations to run in `run_all.py` or `ds_bench` by passing them as arguments (all operations are run by default). For example:

<pre>
deepspeed run_all.py --scan --all-reduce --all-to-all --broadcast
</pre>


# Adding Communication Benchmarks

To add new communication benchmarks, follow this general procedure:

1. Copy a similar benchmark file (e.g. to add `reduce_scatter`, copy `all_reduce.py` as a template)
2. Add a new bw formula in `utils.get_bw`, a new maximum tensor element formula in `utils.max_numel`, and a new arg in `utils.benchmark_parser`
3. Replace comm op calls in new file with find-replace
4. Find a good default `mem_factor` for use in `run_<collective>_single()` function
5. Add new comm op to `run_all.py`
