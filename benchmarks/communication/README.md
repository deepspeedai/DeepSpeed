# Running Communication Benchmarks


To run benchmarks, there are two options:

1. Run a single communication operation:

For example, run with a single large message size:
<pre>
deepspeed all_reduce.py
</pre>

Scan across message sizes:
<pre>
deepspeed all_reduce.py --scan
</pre>

Each individual communication operation's benchmarks have separate benchmarking options. For `all_reduce.py`, for example:

<pre>
usage: ds_bench [-h] [--local_rank LOCAL_RANK] [--trials TRIALS] [--warmup WARMUP] [--maxsize MAXSIZE] [--async-op] [--bw-unit {Gbps,GBps}] [--backend {nccl}] [--dist {deepspeed,torch}] [--scan] [--dtype DTYPE] [--mem-factor MEM_FACTOR] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
  --trials TRIALS       Number of timed iterations
  --warmup WARMUP       Number of warmup (non-timed) iterations
  --maxsize MAXSIZE     Max message size as a power of 2
  --async-op            Enables non-blocking communication
  --bw-unit {Gbps,GBps}
  --backend {nccl}      Communication library to use
  --dist {deepspeed,torch}
                        Distributed DL framework to use
  --scan                Enables scanning all message sizes
  --dtype DTYPE         PyTorch tensor dtype
  --mem-factor MEM_FACTOR
                        Proportion of max available GPU memory to use for single-size evals
  --debug               Enables alltoall debug prints
</pre>

2. Run all available communication benchmarks:

<pre>
deepspeed run_all.py
</pre>

Like the individual benchmarks, `run_all.py` supports scanning arguments for the max message size, bw-unit, etc. Simply pass the desired arguments to `run_all.py` and they'll be propagated to each comm op.

Note that `ds_bench` is a pre-packaged wrapper around `run_all.py`. Users can pass the same arguments as well:

<pre>
<path to deepspeed>/bin/ds_bench --scan --trials=10
</pre>


# Adding Communication Benchmarks

To add new communication benchmarks, follow this general procedure:

1. Copy a similar benchmark file (e.g. to add `reduce_scatter`, copy `all_reduce.py` as a template)
2. Add a new bw formula in `utils.get_bw`
3. Add a new maximum tensor element formula in `utils.max_numel`
4. Replace comm op calls in new file with find-replace
5. Find a good default `mem_factor` for use in `run_<collective>_single()` function
6. Add new comm op to `run_all.py`
