<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.19.2] - 2026-06-16

### Added
- AutoEP (Automatic Expert Parallelism) ([#7938](https://github.com/deepspeedai/DeepSpeed/pull/7938))
- Biren SUPA accelerator support ([#8054](https://github.com/deepspeedai/DeepSpeed/pull/8054))
- `engine.coalesce_grad_reduction()` for ZeRO 1/2/3 multi-backward ([#7992](https://github.com/deepspeedai/DeepSpeed/pull/7992))
- torch.func transforms support for ZeRO 0/1/2 ([#8026](https://github.com/deepspeedai/DeepSpeed/pull/8026))
- Mixed-precision per-policy param/buffer dtype cast ([#8066](https://github.com/deepspeedai/DeepSpeed/pull/8066))

### Fixed
- FP16 optimizer flat buffer init now filters requires_grad ([#8029](https://github.com/deepspeedai/DeepSpeed/pull/8029))
- DeepCompile all-gather scheduler candidate selection ([#8033](https://github.com/deepspeedai/DeepSpeed/pull/8033))
- DeepCompile ZeRO-3 release parameter lifetime ([#8032](https://github.com/deepspeedai/DeepSpeed/pull/8032))
- DeepCompile ZeRO-1 grad target lifetime ([#8036](https://github.com/deepspeedai/DeepSpeed/pull/8036))
- ZeRO-3 DeepCompile grad dtype normalization before reduction ([#8038](https://github.com/deepspeedai/DeepSpeed/pull/8038))
- ZenFlow ZeRO-3 selective optimizer crash with parameter offload on nvme ([#8042](https://github.com/deepspeedai/DeepSpeed/pull/8042))
- ZenFlow Adam integration for updated PyTorch backward flow ([#7771](https://github.com/deepspeedai/DeepSpeed/pull/7771))
- Transformer kernel shared memory indexing to eliminate bank conflicts ([#8055](https://github.com/deepspeedai/DeepSpeed/pull/8055))
- ZeRO-3 coordinator trace invalidation on hook re-registration ([#8043](https://github.com/deepspeedai/DeepSpeed/pull/8043))
- Consistent fp32 grads flow ([#8056](https://github.com/deepspeedai/DeepSpeed/pull/8056))

### Changed
- Simplified module_inject.transpose ([#8028](https://github.com/deepspeedai/DeepSpeed/pull/8028))
- Removed AutoSP assertion against Transformers version ([#8044](https://github.com/deepspeedai/DeepSpeed/pull/8044))

## [0.19.1] - 2026-05-30

### Added
- SDMA allgather via mori for ZeRO-3 ([#7999](https://github.com/deepspeedai/DeepSpeed/pull/7999))
- Auto-detect CUTLASS for EvoformerAttention ([#8000](https://github.com/deepspeedai/DeepSpeed/pull/8000))
- bf16 optimizer states support with CPU offload ([#8010](https://github.com/deepspeedai/DeepSpeed/pull/8010))
- vmap support on LinearFunctionForZeroStage3 ([#8023](https://github.com/deepspeedai/DeepSpeed/pull/8023))
- flash-attn 2.7.0 support in FPDT attention ([#8022](https://github.com/deepspeedai/DeepSpeed/pull/8022))
- Configurable torch-latest dependency versions ([#8016](https://github.com/deepspeedai/DeepSpeed/pull/8016))

### Fixed
- FastFileWriter fd leak by closing aio_fd in _fini ([#8005](https://github.com/deepspeedai/DeepSpeed/pull/8005))
- ZeRO-3 forward crash on modules with plain dict _parameters ([#8009](https://github.com/deepspeedai/DeepSpeed/pull/8009))
- Gemma4 num attention head bugs ([#7990](https://github.com/deepspeedai/DeepSpeed/pull/7990))
- DeepCompile AOT kwargs patching for PyTorch >= v2.11 ([#8024](https://github.com/deepspeedai/DeepSpeed/pull/8024))
- test_zf.py hang bug ([#8012](https://github.com/deepspeedai/DeepSpeed/pull/8012))

### Changed
- Optimized singleton MoE collectives ([#7997](https://github.com/deepspeedai/DeepSpeed/pull/7997))
- Use subprocess instead of os.system in data_analyzer.py ([#7994](https://github.com/deepspeedai/DeepSpeed/pull/7994))
- Sort and dedupe -gencode flags in op_builder.builder ([#8021](https://github.com/deepspeedai/DeepSpeed/pull/8021))

## [0.19.0] - 2026-05-06

### Added
- ZeRO-3 defragment utility ([#7940](https://github.com/deepspeedai/DeepSpeed/pull/7940))
- Sequence Parallelism (AutoSP) support for Multimodal Models (ViT + LLM) ([#7984](https://github.com/deepspeedai/DeepSpeed/pull/7984))
- Gram Newton-Schulz orthogonalization for Muon optimizer ([#7953](https://github.com/deepspeedai/DeepSpeed/pull/7953))
- DeepSpeed NVTX domain support ([#7988](https://github.com/deepspeedai/DeepSpeed/pull/7988))
- SP deny list instead of allow list ([#7887](https://github.com/deepspeedai/DeepSpeed/pull/7887))

### Fixed
- Process hang in process-group shutdown ([#7941](https://github.com/deepspeedai/DeepSpeed/pull/7941))
- ZeRO 1 and 2 CPU-offloaded gradient norm ([#7967](https://github.com/deepspeedai/DeepSpeed/pull/7967))
- ZeRO-1/2 CPU-offloaded gradient loss with multiple backward() per step ([#7981](https://github.com/deepspeedai/DeepSpeed/pull/7981))
- Overlap-comm buffer lifetimes ([#7965](https://github.com/deepspeedai/DeepSpeed/pull/7965))
- DeepCompile+Z3 on PyTorch v2.9/2.10 ([#7951](https://github.com/deepspeedai/DeepSpeed/pull/7951))
- WarmupCosineLR multi-group initialization ([#7969](https://github.com/deepspeedai/DeepSpeed/pull/7969))
- FPQuantizer build ([#7963](https://github.com/deepspeedai/DeepSpeed/pull/7963))
- UB and negative shift warnings in fp_quantize_impl.cu ([#7973](https://github.com/deepspeedai/DeepSpeed/pull/7973))
- Duplicate/wrong -gencode flags in op_builder ([#7974](https://github.com/deepspeedai/DeepSpeed/pull/7974))
- Adam subgroup inconsistency ([#7982](https://github.com/deepspeedai/DeepSpeed/pull/7982))
- BF16_Optimizer last-microbatch grad leak under ZeRO-1 ([#7985](https://github.com/deepspeedai/DeepSpeed/pull/7985))
- topkgating major bug ([#7986](https://github.com/deepspeedai/DeepSpeed/pull/7986))
- DeepCompile backward graph recompilation due to unbalanced forward/backward visits ([#7980](https://github.com/deepspeedai/DeepSpeed/pull/7980))
- Autograd inplace error on CPT by detaching flat buffer ([#7948](https://github.com/deepspeedai/DeepSpeed/pull/7948))

### Changed
- Refactored consolidate transpose ([#7934](https://github.com/deepspeedai/DeepSpeed/pull/7934))
- Renamed dequantization template parameters ([#7976](https://github.com/deepspeedai/DeepSpeed/pull/7976))
- Dynamic offload now compatible with static optimizer offload ([#7979](https://github.com/deepspeedai/DeepSpeed/pull/7979))

[Unreleased]: https://github.com/deepspeedai/DeepSpeed/compare/v0.19.2...HEAD
[0.19.2]: https://github.com/deepspeedai/DeepSpeed/compare/v0.19.1...v0.19.2
[0.19.1]: https://github.com/deepspeedai/DeepSpeed/compare/v0.19.0...v0.19.1
[0.19.0]: https://github.com/deepspeedai/DeepSpeed/compare/v0.18.9...v0.19.0
