---
title: "Installation Details"
date: 2020-10-28
tags: getting-started
---

The quickest way to get started with DeepSpeed is via pip, this will install
the latest release of DeepSpeed which is not tied to specific PyTorch or CUDA
versions. DeepSpeed includes several C++/CUDA extensions that we commonly refer
to as our 'ops'.  By default, all of these extensions/ops will be built
just-in-time (JIT) using [torch's JIT C++ extension loader that relies on
ninja](https://pytorch.org/docs/stable/cpp_extension.html) to build and
dynamically link them at runtime.

```bash
pip install deepspeed
```

After installation, you can validate your installation and see which ops your machine
is compatible with via the DeepSpeed environment report with `ds_report` or
`python -m deepspeed.env_report`. We've found this report useful when debugging
DeepSpeed install or compatibility issues.

```bash
ds_report
```

## Pre-install DeepSpeed Ops

**Note:** [PyTorch](https://pytorch.org/) must be installed _before_ pre-compiling any DeepSpeed c++/cuda ops. However, this is not required if using the default mode of JIT compilation of ops.
{: .notice--info}

Sometimes we have found it useful to pre-install either some or all DeepSpeed
C++/CUDA ops instead of using the JIT compiled path. In order to support
pre-installation we introduce build environment flags to turn on/off building
specific ops.

You can indicate to our installer (either `install.sh` or `pip install`) that you
want to attempt to install all of our ops by setting the `DS_BUILD_OPS`
environment variable to `1`, for example:

```bash
DS_BUILD_OPS=1 pip install deepspeed
```

DeepSpeed will only install any ops that are compatible with your machine.
For more details on which ops are compatible with your system please try our
`ds_report` tool described above.

If you want to install only a specific op (e.g., `FusedLamb`), you can toggle
with `DS_BUILD` environment variables at installation time. For example, to
install DeepSpeed with only the `FusedLamb` op use:

```bash
DS_BUILD_FUSED_LAMB=1 pip install deepspeed
```

Available `DS_BUILD` options include:
* `DS_BUILD_OPS` toggles all ops
* `DS_BUILD_CPU_ADAM` builds the CPUAdam op
* `DS_BUILD_FUSED_ADAM` builds the FusedAdam op (from [apex](https://github.com/NVIDIA/apex))
* `DS_BUILD_FUSED_LAMB` builds the FusedLamb op
* `DS_BUILD_SPARSE_ATTN` builds the sparse attention op
* `DS_BUILD_TRANSFORMER` builds the transformer op
* `DS_BUILD_TRANSFORMER_INFERENCE` builds the transformer-inference op
* `DS_BUILD_STOCHASTIC_TRANSFORMER` builds the stochastic transformer op
* `DS_BUILD_UTILS` builds various optimized utilities
* `DS_BUILD_AIO` builds asynchronous (NVMe) I/O op

To speed up the build-all process, you can parallelize the compilation process with:

```bash
DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext" --global-option="-j8"
```

This should complete the full build 2-3 times faster. You can adjust `-j` to specify how many cpu-cores are to be used during the build. In the example it is set to 8 cores.

You can also build a binary wheel and install it on multiple machines that have the same type of GPUs and the same software environment (CUDA toolkit, pytorch, python, etc.)

```bash
DS_BUILD_OPS=1 python setup.py build_ext -j8 bdist_wheel
```

This will create a pypi binary wheel under `dist`, e.g., ``dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`` and then you can install it directly on multiple machines, in our example:

```bash
pip install dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
```


## Install DeepSpeed from source

After cloning the DeepSpeed repo from GitHub, you can install DeepSpeed in
JIT mode via pip (see below). This installation should complete
quickly since it is not compiling any C++/CUDA source files.

```bash
pip install .
```

For installs spanning multiple nodes we find it useful to install DeepSpeed
using the
[install.sh](https://github.com/microsoft/DeepSpeed/blob/master/install.sh)
script in the repo. This will build a python wheel locally and copy it to all
the nodes listed in your hostfile (either given via `--hostfile`, or defaults to
`/job/hostfile`).

When the code using DeepSpeed is used for the first time it'll automatically build only the CUDA
extensions, required for the run, and by default it'll place them under
`~/.cache/torch_extensions/`. The next time the same program is executed these now precompiled
extensions will be loaded form that directory.

If you use multiple virtual environments this could be a problem, since by default there is only one
`torch_extensions` directory, but different virtual environments may use different setups (e.g., different
python or cuda versions) and then the loading of a CUDA extension built by another environment will
fail. Therefore, if you need to you can override the default location with the help of the
 `TORCH_EXTENSIONS_DIR` environment variable. So in each virtual environment you can point it to a
 unique directory and DeepSpeed will use it to save and load CUDA extensions.

 You can also change it just for a specific run with:

```bash
 TORCH_EXTENSIONS_DIR=./torch-extensions deepspeed ...
```

## Building for the correct architectures

If you're getting the following error:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
when running deepspeed, that means that the cuda extensions weren't built for the card you're trying to use it for.

When building from source deepspeed will try to support a wide range of architectures, but under jit-mode it'll only
support the architectures visible at the time of building.

You can build specifically for a desired range of architectures by setting a `TORCH_CUDA_ARCH_LIST` env variable:

```bash
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ...
```

It will also make the build faster when you only build for a few architectures.

This is also recommended to ensure your exact architecture is used. Due to a variety of technical reasons, a distributed pytorch binary isn't built to fully support all architectures, skipping binary compatible ones, at a potential cost of underutilizing your full card's compute capabilities. To see which architectures get included during the deepspeed build from source - save the log and grep for `-gencode` arguments.

The full list of nvidia GPUs and their compute capabilities can be found [here](https://developer.nvidia.com/cuda-gpus).

## CUDA version mismatch

If you're getting the following error:

```
Exception: >- DeepSpeed Op Builder: Installed CUDA version {VERSION} does not match the version torch was compiled with {VERSION}, unable to compile cuda/cpp extensions without a matching cuda version.
```
You have a misaligned version of CUDA installed compared to the version of CUDA
used to compile torch. A mismatch in the major version is likely to result in
errors or unexpected behavior.

The easiest fix for this error is changing the CUDA version installed (check
with `nvcc --version`) or updating the torch version to match the installed
CUDA version (check with `python3 -c "import torch; print(torch.__version__)"`).

We only require that the major version matches (e.g., 11.1 and 11.8). However,
note that even a mismatch in the minor version _may still_ result in unexpected
behavior and errors, so it's recommended to match both major and minor versions.
When there's a minor version mismatch, DeepSpeed will log a warning.

If you want to skip this check and proceed with the mismatched CUDA versions,
use the following environment variable, but beware of unexpected behavior:

```bash
DS_SKIP_CUDA_CHECK=1
```

## Feature specific dependencies

Some DeepSpeed features require specific dependencies outside the general dependencies of DeepSpeed.

* Python package dependencies per feature/op please
see our [requirements directory](https://github.com/microsoft/DeepSpeed/tree/master/requirements).

* We attempt to keep the system level dependencies to a minimum, however some features do require special system-level
packages. Please see our `ds_report` tool output to see if you are missing any system-level packages for a given feature.

## Pre-compiled DeepSpeed builds from PyPI

Coming soon
