# One-Bit tests

In this folder, you can test the functionality and performance of different backend for doing compressed allreduce, which is the main algorithm in one-bit optimizers like [One-Bit Adam](https://www.deepspeed.ai/tutorials/onebit-adam/), [One-Bit Lamb](https://www.deepspeed.ai/tutorials/onebit-lamb/) and [Zero-One Adam](https://www.deepspeed.ai/tutorials/zero-one-adam/).

## How to run

### NCCL and MPI backend

Basically it requires your environment have relative communication backend installed, the NCCL backend of PyTorch distributed or Message Passing Interface (MPI) like MVAPICH2-GDR and OpenMPI. [Detailed Pre-requisites](https://www.deepspeed.ai/tutorials/zero-one-adam/#12-pre-requisites-for-01-adam).

To test accuracy and performance of NCCL backend:
```bash
python test_nccl_backend.py
python test_nccl_perf.py
```
Similarly, for MPI backend:
```bash
python test_mpi_backend.py
python test_mpi_perf.py
```

### Compressed backend

This backend provides an approach to abstract the generic part of one-bit optimizers and implements accelerator dependent part with DeepSpeed custom op builder. To use this `CompressedBackend` and test it, you should make sure that your current accelerator supports `PackbitsBuilder`, so that it could be loaded to do high performance packing and unpacking between float and Byte datatype.
An example can be found in `Deepspeed/op_builder/xpu/packbits.py`.

The test usage is same as others:
```bash
python test_compressed_backend.py
python test_compressed_perf.py
```
