# Benchmark for DeepCompile

## Setup

This experiment scripts require 1 node that has 2 A100/A40 GPUs
We tested the scripts with Python 3.10.12 and CUDA 12.3.

### Libraries

In addition, you need to install the following:

- PyTorch 2.5.1
- [modified version of DeepSpeed](https://github.com/tohtana/DeepSpeed-internal/tree/neeld2/debug-loss)

Here are an example of installation commands:

```bash
pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install datasets==3.1 accelerate

# Install DeepSpeed and DeepCompile
git clone -b neeld2/debug-loss https://github.com/tohtana/DeepSpeed-internal.git
cd DeepSpeed-internal
pip install -e transformers
cd ..
pip install -e DeepSpeed

# Clone this repository
git clone https://github.com/neeldani/bench_dc_ulysses.git
```

## Running the scripts

Test the setup by running the script:
```bash
bash run_ulysses.sh 6 [compile|deepcompile|eager|ringattn]
```

Here, 6 is the sequence length and is hardcoded because the input sequence inside run_acc_lm.py is hardcoded to easily verify the Q, K and V before and after the all-to-all. You may use pass `compile` to run compiled Ulysses (Ulysses with graph breaks) or `deepcompile` to run deepcompiled Ulysses (allwall inserted within the compiler pass)

We save the Q, K and V tensors before and after the all-toa-all:
For deepcompiled Ulysses, the tensors are saved here: https://github.com/tohtana/DeepSpeed-internal/blob/60feb352a6b0e22cf9a781b4e387d3919dc76833/deepspeed/compile/patch_aot_module.py#L243

For compiled Ulysses, the tensors are saved here: https://github.com/tohtana/DeepSpeed-internal/blob/60feb352a6b0e22cf9a781b4e387d3919dc76833/deepspeed/sequence/layer.py#L381

You can then run the script [check_qkv.py](https://github.com/neeldani/bench_dc_ulysses/blob/main/check_qkv.py) to compare the tensors at various stages i.e before all2all, after all2all, attention outputs etc 

## Code walkthrough
1. Script: [run_ulyssess.sh](https://github.com/neeldani/bench_dc_ulysses/blob/main/run_ulysses.sh)
2. The script calls: [run_acc_lm.py](https://github.com/neeldani/bench_dc_ulysses/blob/main/run_acc_lm.py). We have added support for another attention backend in HuggingFace called "ulysses" which uses DistributedAttention. The implementation can be found here: https://github.com/tohtana/DeepSpeed-internal/blob/60feb352a6b0e22cf9a781b4e387d3919dc76833/transformers/src/transformers/models/llama/modeling_llama.py#L306
3. If the `deepcompile` arg is passed to the config file, then a compiler pass will add the all2all's directy at the Torch IR level. The code for it can be found here: https://github.com/tohtana/DeepSpeed-internal/blob/neeld2/debug-loss/deepspeed/compile/patch_aot_module.py
