# Experiment: qwen-he-kernel-inject-proto

- **Status**: 完成（rung1 接管 + rung2 结构验证 + 2 bug 修复 + OPT 二分结论；
  parity 适配为可选后续）
- **Date**: 2026-09-02（CPU 验证 / GPU 验证 2026-09-04~05）
- **Branch**: `gma/qwen-he-kernel-inject-proto` (worktree `.worktrees/qwen-inject`)

## Goal

回应 Minjia 对 kernel injection 的需求建议：用一个 prototype 验证
"hybrid engine 的 v1 kernel-injection 路径可以延伸到 Qwen 家族"这一概念，
并精确列出 Qwen3.x 需要的 kernel 侧工作。GPU 实例到位后跑数值与性能。

## Why a two-rung ladder

- **Rung 1 (exact): Qwen2/Qwen2.5 container** —— `Qwen2DecoderLayer` 与 llama
  同构（分离 q/k/v/o、gated SiLU MLP、RMSNorm、rotate_half rope），GQA 已被
  fused kernel 支持（`ds_attention.py` 的 `config.num_kv` + `repeat_kv`）。
  预期 GPU 上 greedy 与 golden **bit-exact**，可测注入收益。默认启用，
  `DS_QWEN2_INJECTION=0` 可关。
- **Rung 2 (structural): Qwen3.5/3.6/3.8 container** —— 三代 27B 共用
  `qwen3_5` 架构（见 he-rollout-ws2 journal 与 HF config 对比）。只注入
  `full_attention` block（`should_replace` 按 `block_type` 逐层筛选），
  GatedDeltaNet block 保持 native。**显式 opt-in**（`DS_QWEN35_INJECTION=1`），
  因为已知 kernel gaps（下节）未闭合前 forward 无 parity。

## Files

- `deepspeed/module_inject/containers/qwen2.py` (new) — DS_QWEN2Container + Qwen2LayerPolicy
- `deepspeed/module_inject/containers/qwen3_5.py` (new) — DS_QWEN3_5Container + Qwen3_5LayerPolicy
  （含 `should_replace` 静态方法：GDN 层回退 native）
- `deepspeed/module_inject/containers/__init__.py` / `replace_policy.py` — 注册
- `deepspeed/module_inject/utils.py` — `policy_to_ds_container` 的第二张
  policy→container 映射表同步注册（漏掉它 container 返回 None →
  `set_tensor_parallel_config` AttributeError，已修）
- `deepspeed/runtime/hybrid_engine.py` — `create_inference_containers` 增加
  per-instance opt-out hook（`hasattr(policy_cls, 'should_replace')` 守卫，
  对 nn.Linear 等泛型映射零影响）
- `experiments/test_qwen_inject_proto.py` (new) — baseline/zero3/check-filter
  三模式 + 注入结构 introspection（containers 数、injected 层的 block_type
  分布、generate_wrapped）

## CPU 结构验证结果 (2026-09-04)

| 检查 | 结果 |
|---|---|
| policy 解析（dscpu, tf 4.49） | Qwen2→Qwen2DecoderLayer ✓，Qwen3_5→None ✓（安全跳过） |
| policy 解析（py3.12, tf 5.16-dev） | Qwen2 ✓；`DS_QWEN35_INJECTION=1` 时 Qwen3_5→Qwen3_5DecoderLayer ✓ |
| 混合层筛选（check-filter, tiny 8 层） | `FILTER_VERDICT=PASS`：仅 full_attention 被选中（2/8，layer 3/7），GDN 全部回退 ✓ |
| zero3 world=2（Qwen2.5-0.5B） | 两 rank `MATCHED_BUT_NEEDS_GPU`：policy 匹配、进入 `create_module`，但 **容器构造期即加载 CUDA op**（`QKVGemmOp`→`builder.load()`→CPU no_impl）✓ 优雅降级 |
| golden baseline | 与 he-rollout-ws2 相同 prompt 输出一致 |

新发现（比预期更硬的边界）：**v1 推理栈的 op 在容器构造时就 load CUDA
module，CPU 上连容器都建不了**——不只是 forward 需 GPU。GPU 实例到位前，
CPU 只能验证到 policy 匹配 + 层筛选这一层（已全部通过）。

## Kernel gaps (Qwen3.5 full_attention, 实证于 transformers 5.16.0.dev0)

1. **head_dim ≠ hidden/heads**：0.8B 为 1024/8=128 vs head_dim=256；27B 为
   5120/24（不整除）。fused kernel 的 qkv 布局
   （`ds_attention.py`:`((heads + num_kv*2)//mp) * (hidden//heads)`）无法
   表达该族。**这是最根本的一条。**
2. **q_proj 2 倍宽**：`num_attention_heads * head_dim * 2`，后半是
   attn_output_gate 的源（`attn_output_gate: true`）。kernel 无 gate 输入，
   当前 mapping 只取前半（query）。
3. **逐头 q_norm/k_norm**：RMSNorm over head_dim，位于 projection 与 rope
   之间，kernel 路径无落点（无法静态 fold，输入相关）。
4. **GDN (linear_attention) block 无 kernel**：全仓 zero 支持；0.8B 为
   18/24 层，27B 为 48/64 层。chunked prefill + 递归 decode 状态管理是
   独立的 kernel 工程。
5. **partial rotary (0.25) + interleaved mrope**：`rotary_dim` 可表达部分，
   mrope section 文本 rollout 可先按标准 rope 处理，需验证。
6. MTP 层（`mtp_num_hidden_layers=1`）与 vision tower：不注入，留 native。

## Commands (CPU structural check, 全部已跑通)

```bash
WT=.worktrees/qwen-inject
# 混合层筛选（py3.12 + tok023 env）:
DS_SRC=$WT PYTHONPATH=/tmp/tok023 python3 $WT/experiments/test_qwen_inject_proto.py --mode check-filter
# rung 1: Qwen2.5-0.5B — expect MATCHED_BUT_NEEDS_GPU
$WT/../dscpu/python $WT/experiments/test_qwen_inject_proto.py --mode baseline --out /tmp/qi_golden.pt
DS_SRC=$WT dscpu/torchrun --standalone --nproc_per_node=2 \
    $WT/experiments/test_qwen_inject_proto.py --mode zero3 --ref /tmp/qi_golden.pt
# rung 2 (GPU 到位后): Qwen3.5-0.8B-Base + DS_QWEN35_INJECTION=1
# 期望 containers=6 (full_attention of 24)，GDN 18 层走 native
```

## GPU-phase 结果 (2026-09-04 晚, AutoDL 2xRTX4080-32G, torch 2.12.1+cu130, tf 5.14.0.dev0)

远端：`/root/DeepSpeed` 分支同名（自 master 9311fd5 + 本 prototype patch，scp 部署）。

### 已达成

| 项 | 结果 |
|---|---|
| JIT 编译 | inference CUDA op 一次通过（sm_89, cu130） |
| rung1 注入接管 | **containers=24, other_layers=2, generate_wrapped=True**（Qwen2.5-0.5B-Instruct, Z3, world=2）——hybrid engine 首次真正接管 Qwen generate |
| forward 全链路 | 打通，generate 完成（乱码→见下） |
| 4B 下载 | 实例关机前已完成（2/2 shards） |

### 修复的两个真 bug（均已在本地 worktree + 远端）

1. **GQA qkv 权重合并按 MHA 写死**：`ds_attention.py` `_merge_qkv` 与
   `_qkv_buffers` 分配固定 3×hidden_pp 行，Qwen2.5-0.5B (14q/2kv) 在
   `[896 vs 128]` 处崩溃。修复：num_kv>=0 时按 `hidden_pp + 2*num_kv_pp*head_dim`
   分配并按 q|k|v 实际行数切片（老模型路径 num_kv=-1 行为不变）。
   **独立成立的 upstream PR 素材。**
2. **transformers 5.x 层返回协议**：DS v1 层在 use_cache=True 时固定返回
   `(output, presents)`；5.x 层循环直接把返回值当 hidden_states 传给下一层
   → `'tuple' object has no attribute 'size'`。修复：`hybrid_engine.py`
   `_ds_forward_adapter` + `_zero3_forward` 内解包 `output[0]`。

### 关键发现：v1 注入栈对 transformers 5.x 系统性不兼容（数值层）

二分证据：**facebook/opt-1.3b**（HFOPTLayerPolicy 原生支持、无 rotary、
排除 theta 类配置因素）注入后同样退化为重复 token
（" Paris Paris Paris ..."，首 token 对、后续重复）→ 不是 Qwen 特有 gap。
症状指向 decode 步 KV/位置协议错位：HF 5.x 的 DynamicCache 期待层内
in-place update，DS 层不写回（自身 workspace KV 各自为政）→ HF 侧
cache_position/causal mask 与 DS 内部状态逐步错位。

工程含义：数值 parity 需要一个 **DS 层 ↔ HF Cache 协议适配层**
（预估 1-3 小时调试量级，含不确定性），这是 prototype 量化出的
v1 路径维护成本的核心组成部分。

## AutoTP 联合执行实验 (2026-09-05, Qwen2.5-0.5B-Instruct, world=2, GPU)

问题：container 注入与 AutoTP 能否联合执行？

| 路径 | 结果 |
|---|---|
| A: autotp_size=2 + 注入（默认开） | ❌ **构造层即不兼容**：AutoTP 先替换叶子 Linear，container policy 在 `get_hidden_heads` 读 `LinearLayer.in_features` 直接 AttributeError。联合需 (a) policy 适配 LinearLayer/ds_shape，(b) 解决 container inference_tp 与 AutoTP 分片的双重 TP 语义；且 KV cache 协议的数值问题仍在前 |
| B: autotp_size=2 + `DS_QWEN2_INJECTION=0`（native 回退） | ✅ **bit-exact PASS**（greedy 与 golden 一致，两 rank 一致），在当前分支 + tf 5.14 复现老 journal 结论 |

结论：**当前形态二选一**——AutoTP（native 回退）或 container 注入，不能同时。
顺带修复：测试脚本在 TP 语义下须全 rank 同 prompt（engine 的 TP 组输入一致性
检查会拦截 DP 式分 prompt）。

## 剩余步骤（2026-09-05 更新）

1. ~~rung1 接管~~ / ~~rung2 结构验证~~ / ~~AutoTP 联合实验~~ 均已完成
2. （可选）数值 parity 协议适配：DS 层写回 HF Cache，预估 1-3 小时含不确定性
3. （可选）性能对比：parity 前意义有限

## rung2 结构验证结果 (2026-09-05, Qwen3.5-4B-Base, DS_QWEN35_INJECTION=1)

4B config: 32 层 / **8 个 full_attention** / hidden 2560 / 16 q heads /
4 kv heads / head_dim 256。

| 检查 | 结果 |
|---|---|
| 4B golden baseline（native, GPU） | 正常输出（Qwen3.5 GDN 的 Triton kernel 需 GPU tensor，测试脚本 baseline 模式已修） |
| 注入筛选 | **containers=8, injected_layers=8 (full_attention=8, other=[])** —— 真实模型上精确命中全部 full_attention 层，GDN 24 层全部回退 native ✓ |
| 共存构造 | other_layers=194（GDN 层 + 其余模块走泛型 wrapper），generate_wrapped=True ✓ |
| forward | 如预登记失败于 `_merge_qkv`：`attn_qw` 8192 行（2x q_proj，gap #2，`set_q_k_v` 路径）vs kernel 期望 q 行数=hidden_pp 2560（head_dim≠hidden/heads，gap #1）。**结构验证目标全部达成；forward parity 需先闭合 kernel gap #1/#2（含 csrc 侧改动），与预登记清单一致。** |

过程修复（均已同步本地 worktree + 远端）：
- `Qwen3_5RMSNorm` 在 tf 5.x 暴露 `eps` 而非 `variance_epsilon`
  （qwen3_5.py get_hidden_heads 双兼容读取）

### 结论（给 Minjia 的材料骨架）

1. plumbing 可行：新家族 policy/container + 逐层筛选 hook 全部按设计工作，
   接管成功（containers>0, generate wrapped）。
2. 两个存量 bug（GQA merge、5.x tuple）+ 一个系统性协议不兼容（KV cache），
   均与 Qwen3.x 新特性无关——先于 Qwen3.5 特有 kernel gaps（head_dim≠h/h、
   2x q_proj、q_norm/k_norm、GDN）存在。
3. 因此 v1 注入路径对现代 transformers + 新模型族的实际启用成本 =
   协议适配 + 存量修复 + 家族 kernel gaps 三层叠加。

## Notes

- 主 checkout 有未提交的 shared-prefill 改动（也依赖 containers>0），
  本 worktree 从已提交 HEAD da4aa3513 分叉，二者正交但互补。
- `HybridEngineRollout.generate` 走 `module.generate` → containers>0 时
  自动被 hybrid engine 接管 —— 测试脚本即生产链路。
