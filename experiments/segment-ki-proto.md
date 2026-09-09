# Experiment: segment-ki-proto

- **Status**: CPU 阶段完成（正确性 + 跨家族复用全 PASS）；GPU 阶段
  （Triton kernel + profiler launch 计数）待实例
- **Date**: 2026-09-05
- **Branch**: `gma/segment-ki-proto` (worktree `.worktrees/segment-ki`)
- **前置**: `gma/qwen-he-kernel-inject-proto`（旧 KI 取证）与本仓库对话中
  的新 KI 架构设计（四条职权边界）

## Goal

验证段式 kernel injection（新 KI）的最小可行原型：AutoTP 独占分片与通信，
KI 只在**无通信段**内替换计算。验收 = 与 native 路径 greedy bit-exact。

## 设计契约（四条边界）

1. 通信边界：forward 内含 collective 的模块（`*Allreduce` 层）是硬边界，
   原样委托不触碰
2. 形状边界：kernel 消费实际分片权重（cat gate|up 沿输出维，分片内合法），
   不从 hidden/heads 推导任何布局
3. 协议边界：parent 模块返回契约不变，generate 全程 HF 驱动（无 tuple
   adapter 类补丁）
4. 命名边界：pattern 按**结构特征**（gate_proj/up_proj/down_proj 属性）检测，
   不按 HF 类名

## Files（零 core 侵入）

- `deepspeed/module_inject/segment_ki.py` (new, ~130 行) —
  `carries_collective` / `find_glu_segments`（结构检测）/ `apply_segment_ki`
  （显式调用，装在 deepspeed.initialize 之后）
- `experiments/test_segment_ki.py` (new) — baseline/autotp/autotp_ki 三模式
- **未修改任何 DeepSpeed core 文件**——与旧 KI prototype（5 个 core 文件
  +2 bug 修复）形成维护性对照

## CPU 结果 (2026-09-05, world=2, gloo, fp32, greedy 16 tokens)

| 测试 | segments | 输出 vs golden | verdict |
|---|---|---|---|
| Qwen2.5-0.5B baseline（native HF） | — | golden | — |
| Qwen2.5-0.5B autotp（无 KI 对照） | — | bit-exact | PASS |
| Qwen2.5-0.5B autotp+KI | found=24 replaced=24 | **bit-exact** | **PASS** |
| Qwen3.5-0.8B-Base autotp+KI（tf 5.16, 混合 GDN） | found=24 replaced=24 | **bit-exact** | **PASS** |

关键点：
1. **一次通过**：旧 KI 需 2 个 bug 修复 + 1 个协议适配器才跑通 forward 且
   从未达到 bit-exact（KV cache 协议问题）；新 KI 首跑即 bit-exact——
   四条边界把旧 KI 的四类 bug 在构造上排除。
2. **跨家族复用**：同一 `fused_glu` 零改动服务 Qwen2.5（tf 4.49）与
   Qwen3.5（tf 5.16、75% GDN 层）——"kernel 是 op 级复用资产"论点的直接
   实证；GDN 层的 MLP 段同样被覆盖（旧 KI 对 GDN 层覆盖为零）。
3. bit-exact 隐含验证：分片内 cat-单GEMM-chunk 与两个独立 GEMM 数值一致；
   绕过 column-parallel 层 forward 里的 `ColumnParallel.apply`（输入聚合）
   在 inference 路径无害。

## 过程发现：AutoTP 的通信点比 row/column 直觉更密

`LinearLayer.forward`（column-parallel）开头也有 `ColumnParallel.apply`
（对输入的 all_reduce / inference_all_reduce），即 gate/up 各做一次输入
聚合——当前场景下疑似 no-op（否则 bit-exact 不成立），但说明：
1. 段边界必须以 **forward 内实际执行的 collective** 为准，不能按并行类型
   想当然；生产版应由通信平面显式导出边界标注（marker），而非 KI 侧猜测
2. gate/up 的两次输入聚合若在训练路径真实执行，则段式融合天然把它们合并
   为零/一次——第一个"融合即省通信"的联合优化案例

## GPU-phase 结果 (2026-09-07, Qwen2.5-0.5B-Instruct, bf16, batch1, greedy 128 tok)

Native CUDA kernel `fused_glu.cu`（op_builder JIT，~100 行）：fp32 diff 2.4e-07；
bf16 diff ~1-2 ULP（单次舍入 vs composite 双次舍入），端到端文本正常。
性能矩阵（hf 基线 / old KI / segKI）：

| 路径 | 配置 | prefill_ms | decode tok/s | 数值 |
|---|---|---|---|---|
| hf.generate | 单卡 eager | 24.1 | 50.0 | ✅ |
| hf.generate + graph capture | 单卡，torch.compile reduce-overhead | 23.0 | 50.4 | ✅ |
| **hf + segKI（native kernel）** | 单卡，无 AutoTP | 22.3 | **51.6（+3.2%）** | ✅ ULP 级 |
| AutoTP native（对照） | TP=2 | 44.3 | 24.1 | ✅ |
| **AutoTP + segKI（native kernel）** | TP=2 | 35.8 | **33.2（+38%）** | ✅ |

old KI（container 注入）的性能数据**待数值修复后补录**——其 KV 协议
问题已修两处（DynamicCache 写回 + bool mask 中和，见
qwen-he-kernel-inject-proto 线），但输出仍为乱码，病灶定位到单层内部
（qkv 布局或 other_layers wrapper），修复前任何 old KI 性能数字均无效。

要点：
1. segKI 净贡献：单卡 +3.2%（MLP 段在 0.5B 占比小、attention+lm_head
   主导），**TP=2 下放大到 +38%**（融合省下的 launch/带宽在通信受限
   场景相对成本更高）。
2. graph capture（torch.compile reduce-overhead）在 hf.generate 上无收益
   （50.0→50.4）：capture 区只覆盖 forward，HF generate 每步 Python 循环
   开销在区外。
3. **两条线的合流论据**：segKI 证明可组合性与正确性 + 通信委托下仍有
   净收益；old KI 的 csrc megakernel 性能上限待数值修复后量化——若
   2.3× 级别成立，把 csrc megakernel 包进 segKI 的 op 签名层 = 正确性 +
   性能上限。

## 4B 目标家族实验结果 (2026-09-08, Qwen3.5-4B-Base, bf16, greedy 128 tok)

性能矩阵（替换 0.5B headline）：

| 路径 | b1 tok/s | b8 tok/s (合计) |
|---|---|---|
| hf eager 单卡 | 22.8 | 176.7 |
| AutoTP TP=2 | 12.3 | 99.0 |
| **AutoTP + segKI** | **13.3（+8%）** | **104.8（+5.9%）** |

正确性：GPU bf16 下 segKI 与 eager 前段完全一致，**首个分叉在第 68 个
生成 token**（token match rate 0.61 为分叉后词表重叠所致，文本各自连贯）
——bf16 单舍入 vs eager 双舍入的 ULP 级分歧长生成放大，非结构性错误；
CPU fp32 bit-exact 结论不变。

归因（轻量 profiler，短窗口噪声大，取定性）：TP=2 下 NCCL 占 device
time 37-58%——4B 通信主导。**+38%（0.5B，overhead 主导）→ +6-8%
（4B，通信/带宽受限）符合"融合收益 ∝ overhead 受限程度"的预测**：
MLP 融合省 launch/CPU，不省权重带宽。

## v1-on-4B 尝试 (csrc 修改授权首日, 推进两站)

1. **head_dim 解耦（gap #1）Python 侧修复**：config 增加 `head_dim=-1`
   参数（向后兼容），ds_attention 全部 `hidden//heads` 推导点改用
   attn_head_dim（qkv 布局/norm_factor/merge 的 q_rows=heads×head_dim）；
   qwen3_5 容器 set head_dim + set_q_k_v 切 2x q_proj（gap #2）。
   → **容器构造通过**（此前崩在 _merge_qkv 8192 vs 2560）
2. **混合 cache 交互修复**：HF Cache 的 GDN 槽（LinearAttentionLayer）
   与 attention 槽 API 不同 + container 枚举层号≠模型层号——改为从
   orig_module.self_attn.layer_idx 取真实层号 + 槽型防御守卫
   → **通过**
3. 当前障碍（队列下一项）：softmax_context CUDA 绑定的 host-pointer
   error（疑与 head_dim 解耦后的 kernel 分发路径有关，未及深挖）

教训：v1 对 qwen3_5 族的启用是"剥洋葱"——每修一层露出下一层；今日
证实构造层与协议层均可修，剩 kernel 分发层。

## GDN 段融合首战 (2026-09-09, 4080 SUPER, Qwen3.5-4B)

实现：`find_gdn_segments`（AutoTP 下安全降级——分片投影与全量 conv1d 布局
不匹配，实测确认）+ `_fused_gdn_forward`（in_proj_qkv/z/b/a 四合一 GEMM，
conv/FLA-scan/gated-norm/out_proj 全委托原模块）+ native `gdn_gates`
kernel（beta/g 融合，fp32 数学，softplus 大 x 稳定分支——初版没有，
relmax 0.46%→修复后 0.39% = 1 bf16 ULP）。

CPU 门禁：0.8B 纯 HF 18/18 GDN + 24/24 GLU **bit-exact 首跑通过**；
AutoTP world=2 GDN 安全跳过 + GLU 照常、输出一致。

GPU 4B 端到端：**MATCH_REF=True（greedy 与 hf eager 完全一致）**。
调试战果（4 个 kwargs 委托坑，全部记录在案）：GPU fused decode kernel
拒绝 cu_seqlens（CPU dispatcher 容忍）；层 kwargs（cache_params/
use_cache 等）经 **kwargs 泄漏进 scan——最终方案：scan 参数白名单。

性能：22.6 vs 22.9 tok/s（持平），prefill 183.8 vs 65.3ms（回归）。
归因（实测）：fused 输出切片的 non-contiguous 链（slice+transpose+
contiguous 单次 403μs @ prefill 尺寸）× 24 层，吞掉 4→1 GEMM 的收益；
FLA scan 主导 GDN 层时间，glue 占比小（4B 教训重演）。

Next（明确优化项，非架构问题）：fused 权重按目标布局预排（消除
slice/contiguous 链）；gates kernel 接受 strided 输入；之后重测。
