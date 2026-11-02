# Experiment: pr8602-gpu-regression

- **Status**: completed
- **Date**: 2026-10-05
- **Tmux pane**: 0:2.0（autodl-on/off 控制同 pane）
- **关机**: 18:16 autodl-off 成功（GPU 计费停止）

## Hardware

- Machine: root@connect.bjb1.seetacloud.com:15746 (autodl-pro-7838522e6e9b)
- GPU: NVIDIA GeForce RTX 4080 SUPER ×2, 32760 MiB each（本实验单卡）
- Driver/CUDA: 580.105.08 / torch cu130
- OS: AutoDL 容器, 系统盘 111G/127G used

## Environment

- Python: /root/miniconda3/bin/python 3.12.3（base env）
- torch 2.12.1+cu130 (cuda available), transformers 5.15.0.dev0
- Code: /root/pr8602-wt @ 42374ea（bundle 传输自本地 worktree pr-8602）

## 目标（GPU 回归清单）

1. CPU invariant + utils 测试（sanity）
2. JIT 编译 fused_glu op（验证 C14 builder 注册 + 移动后加载路径）
3. TestKernelReferenceConsistency 6 项（GPU kernel vs oracle）
4. TestRolloutTrainRollout 集成（Qwen3.5-0.8B；覆盖 C2/C5 no-grad 路由、C10 EOS pad、C11 3-D GLU kernel、C13 bf16 gate、decode_loop 删除后的 Python-loop fallback）
5. 专项脚本（如时间允许）：padded prompt 走 SDPA（C4）、fp16 模型不崩（C13）

## Commands

1. `cd /root/pr8602-wt && /root/miniconda3/bin/python -m pytest tests/unit/module_inject/test_segment_ki_kernels.py tests/unit/utils/test_static_cache.py -q`（pane 0:2.0）

## Timeline

- 23:17 autodl-on 开机成功
- 23:25 bundle (9311fd5..42374ea, 1.1M) scp → /root/DeepSpeed fetch → worktree /root/pr8602-wt
- 23:26 全量单测：**17 passed, 1 skipped**（3:39）；kernel 6 项全过（首项 68s 含 fused_glu JIT 编译）；1 skipped 为集成测试模型不可达（需 HF_HUB_OFFLINE=1 + 缓存模型 Qwen3.5-0.8B-Base）
- 23:5x fp16 回归首次运行：**发现新漏洞** — `_fused_gdn_forward` 的 `gdn_gates` 对 fp16 崩溃（"expected BFloat16 but found Half"）→ 本地修复 `32b404323`（bf16 gate）→ scp 重部署
- 00:0x 修复后重跑：**fp16 模型完整生成不崩**（脚本走完两部分）；segki report: fused_glu 18 segments replaced
- **C4 验证通过**：`PROMPT_PADDED_FLAG: True`，decode 走 SDPA 回退
- **C7 bug 实锤**：PADDED_MATCH=False，padded 输出含 8 个 `<|endoftext|>` 回声（prefill attend 了 padding，C7 未修所致）

## Results

| 项 | 结果 |
|---|---|
| 全量单测（GPU） | 17 passed, 1 skipped |
| fused_glu JIT 编译 + builder 注册（C14） | ✅ |
| fp16 模型生成（C13 + 32b404323） | ✅ 不崩 |
| padded prompt decode 回退（C4） | ✅ flag 生效 |
| hybrid continuous batching 显式拒绝（C6） | ✅ RAISED_OK |
| 最终态 bench 4B 短 prompt（5 tok） | segki+graph 66.7 / graph-only 66.2 / eager 21.9 tok/s |
| **最终态 bench 4B 长 prompt（2048 tok，减去 prefill 1.77s）** | **segki+graph 109.3 / graph-only 98.5 / eager 30.8 tok/s（segki +11.0%）** |
| 输出正确性 | graph-only vs eager **32 token 逐 token 一致**；segki 在 near-tie 位置（Δlogit 0.125/17.5）可分叉，进程内确定、跨进程可翻转；输出语义连贯 |

### 关键结论

1. 19 个 commit 全程零性能回归（4 次短 prompt 采样 66.7-67.5 tok/s，噪声带内）
2. **segki 增益随上下文长度增长**：5 tok → +0.8%，2048 tok → **+11.0%**（decode_attn kernel 替换慢速 SDPA mem_efficient 后端的价值在长 KV 才显现）
3. 4B decode 为权重带宽受限（~8GB @ ~0.9TB/s ≈ 9ms/tok），与 2K 实测 9.2ms/tok 吻合
4. token 级 parity：graph-only 与 eager bit-exact；segki 接受 near-tie 分叉（与 vLLM 同性质）


## Notes

- 远程需 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（缓存模型）
- 专项脚本：/root/pr8602-wt/test_fp16_padded.py（本地 experiments/test_fp16_padded.py）


## Notes

- 远程 /root/DeepSpeed 原工作区保持不动（旧 master + 本地修改）
- decode_loop 层已删除；DS_BUILD_DECODE_LOOP 不复存在
