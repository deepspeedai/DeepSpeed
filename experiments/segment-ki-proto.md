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

## FLA 路由修复与 4B 正收益 (2026-09-10)

根因（profiler 三层证据不一致逼出的真凶）：transformers 5.14 在
`__init__` 用**实例属性**绑定 GDN kernel（`self.recurrent_gated_delta_rule
= fused_recurrent_gated_delta_rule or torch_...`），FLA 可用时绑 fused
kernel；segKI 委托误调模块级 `m5.torch_recurrent_gated_delta_rule`
（纯 torch 参考实现）→ 每层每步走慢速回退 → mul/add/sum 爆炸
（+604 elementwise/step）、gdn_core 减半。修复一行：委托改走实例属性，
自动跟随 transformers 的 kernel 选择。

最终数字（3 次均值，Qwen3.5-4B bf16 b1 单卡）：
| | tok/s |
|---|---|
| HF eager | 22.5 ± 0.2 |
| **segKI v4** | **24.8 ± 0.4（+10.3%）** |

收益分解（profiler 实证）：
- device time -3%（30.3→29.4ms/step）：GEMM 次数减半但权重带宽不变
  （mm+gemv device 时间 12.5 vs 12.7ms 持平），大头 FLA scan 委托未动
- 其余 ~7% 为 CPU dispatch 消除：每步少 ~250 次 aten/Python 穿越
  （4 个 in_proj forward → 1 matmul；门控 6-8 op → 1）
- 本质：batch 1 decode 是权重带宽受限，可收割的是 CPU 调度税——
  overhead 占比越高的场景收益越大
正确性：分叉位 [0,75] 与 GLU 时代 ULP 分叉完全同位（非结构性）。
layout 修复（strided gates + 显式 qkv repack + stride(-2) 越界修复）
同轮落地。

## Graph capture 实验 + GDN 静态化方案 (2026-09-10/11)

### 实验结果（Qwen2.5-0.5B-Instruct，HybridEngineRollout use_graph_capture）

| 配置 | tok/s | 说明 |
|---|---|---|
| eager | ~50 | golden 基线 |
| + graph capture | **182-193（3.7×）** | 文本正确 |
| + segKI + graph | 164-176（**-9%**） | ULP 分叉 |

三个发现：
1. **收益重叠关系实锤**：graph replay 吃掉 CPU dispatch 后，segKI 的
   CPU 收益（~7%）归零，其残余 device 开销（切片链）转为净负担——
   graph 与 segKI 收割同一池开销，是替代不是叠加。
2. **graph capture 对 GDN 混合架构崩**：`_generate_graph` 的 KV 拷贝循环
   假设每层 cache 槽都有 keys/values（`LinearAttentionLayer has no
   attribute 'keys'`）——DeepSpeedStaticCache 的 KV-only 世界观 vs GDN 的
   异构槽（recurrent_states + conv_states）。**非 GDN 固有缺陷**：GDN 状态
   天生固定形状（比增长 seq 的 KV 更 graph 友好）；vLLM/SGLang 已让
   hybrid GDN 模型（Qwen3-Next/3.5/3.6-27B）跑在 CUDA graph 上（独立
   状态 allocator + 原位更新）。
3. **存量 API 失效**：transformers 5.x cache 协议新增 `get_query_offset`，
   DeepSpeedStaticCache 未实现导致 graph 路径整体静默不可用（±segKI 均
   崩，模型无关）——已修复（转发 get_seq_length），0.5B graph 复活 3.7×。

### GDN graph capture 修复方案（预估 3-4 天）

Step 1 槽型感知（~1d）：DeepSpeedStaticCache 增加 DSStaticGDNLayer
（recurrent_states [b,vh,kd,vd] fp32 + conv_states [b,conv_dim,k-1] 环形），
拷贝循环按槽型分派；实现 HF Cache 的 GDN 协议方法
（has_previous_state/update_recurrent_state/update_conv_state——接口
清单复用 segKI GDN 委托的 kwargs 白名单战利品）。

Step 2 捕获安全（~1d，关键点）：HF 的 update_recurrent_state 是重绑定
（每次 replay 产生新 tensor，graph 必坏）→ 改为原位
`buffer.copy_(state)`；conv 已原位（causal_conv1d_update，vLLM 血统）；
FLA fused_recurrent 纯 GPU 固定形状，捕获兼容风险低（vLLM 先例）。

Step 3 捕获循环 + 门禁（~1d）：现有 static token/mask/position 骨架不动
（GDN 层递归路径无需 mask）；门禁沿用 greedy vs golden（ULP 容忍）+
跨序列状态 reset + 重放确定性。

风险：MTP 层（Qwen3.5 有 1 层，MtpCache 有 query offset 偏移语义）——
第一版排除于捕获外（rollout 不需要 MTP）。

收益账：0.5B graph=3.7×；4B 拿一半即 ~40+ tok/s 级，碾压一切融合手段；
且 DecodeGraphCache 是 engine 级，同时服务两条 KI 线。

### 修复后的重测项

graph 通了后 segKI 重定位：CPU 收益被 replay 吃掉，但 kernel 数影响
replay 时长——切片开销清零后的 segKI 在 graph 内应转正（少 ~250
kernel/replay）。这是 graph 修复后的第一个实验。

### 战略图景（最终版）

1. graph capture：3.7×（0.5B）——对 GDN 混合架构待修（本方案）
2. segKI 段融合：+10.3%（4B）——graph 不可用场景的过渡与补充
3. native scan kernel：device 侧 2× 台阶——与 1/2 正交（真打带宽）

## GDN graph capture Phase A (2026-09-11, CPU 验证通过)

实现（比 design note 预估更小——HF 的 LinearAttentionLayer 本身已
cudagraph-safe：copy_ 原位 + mark_static_address，注释明说为 cudagraphs
设计；DeepSpeed 侧只需透传而非重写）：
- `DSStaticGDNSlot`：GDN 槽透传 wrapper（bind 引用 prefill cache 的 HF 槽，
  __getattr__ 转发完整 HF 槽 API，零拷贝）
- `DeepSpeedStaticCache`：按 layer_types 混合建槽；协议方法族补齐
  （has_previous_state / update_recurrent_state / update_conv_state，
  各自槽型分派）；head_dim 解耦修复（cache 侧的 gap #1 翻版：
  early_init 曾用 hidden/heads=128，Qwen3.5 实际 256）
- `_generate_graph` 拷贝循环槽型分派（GDN 槽 bind，KV 槽照旧拷贝）

CPU 门禁（0.8B，复刻 _generate_graph 至 capture 前的全部逻辑 +
eager decode 走 DS 混合 cache）：**BIT_EXACT True**，槽型对齐
6 KV + 18 GDN 双向吻合。

Phase B（GPU 待做）：真 capture（FLA kernel 兼容性是主要剩余风险）、
4B 门禁、3.7× 级性能验证、graph 内 segKI 重测。

## GDN graph capture Phase B 进展 (2026-09-11, GPU)

已越过两站（每站都是 masking_utils × 混合槽的真实语义交互）：
1. **0 长 mask 崩溃修复**：transformers 的 mask 构建按 layer 采样
   get_mask_sizes。DSStaticGDNSlot 必须精确对齐 HF CacheLayer 基类公式
   ——GDN 槽无 seq 维 → `(query_length, 0)`。(0,0) 产生 0 长 mask 崩
   SDPA；(max_cache_len,0) 破坏 GDN 自身 padding 语义（第 1 token 就
   分叉）。`(query_length, 0)` 两边全对：CPU 0.8B **BIT_EXACT True**，
   GPU 越过崩溃。
2. 4B GPU 单步/3 步 eager decode 走 DS 混合 cache 全部通过
   （hooks 清空与否均过——排除 hook 假设）。

当前障碍（队列下一项）：warmup/capture 段 `(*bias): last dimension
must be contiguous`——性质未明（position_bias/conv bias 嫌疑），待
定位。附带发现：mask 尺寸的 layer 采样存在 max_len−2 类的偏移谜题
（[1,16,1,max_len−2] target），修复 mask 长度后不再触发，但机理
值得在修 bias 时一并厘清。

## Phase B 深挖检查点 (2026-09-11 深夜, bias contiguous 之谜解开)

`(*bias): last dimension must be contiguous` 是**误导性错误**——SDPA 内部
mask 形状不匹配走的检查路径。真实机制（in-source 插桩实证，SDPALOG）：

崩溃调用的实际形状：
  q=[1,16,1,256]  k=[1,16,69,256]（repeat 后）  mask=(1,1,1,1) stride 全 1
  → mask 长度 1（正是 GDN 槽 `(query_length,0)` 语义）被 full-attn 层
    拿到，与 kv=69 不匹配 → SDPA 报 bias contiguous。

即：**transformers 5.14 的 mask 构建共享 per layer_type，取样时可能拿到
GDN 槽语义**。HF 自己的约定（cache_utils ~597 行注释）：alternating cache
的容器级长度查询"must use attention layer idx"——但我们尝试的三种
GDN 槽/容器语义组合：(0,0) 崩 SDPA、(max,0) 第 1 token 分叉、
(attn_seq+q,0) mask=max−2 错位（21 vs 7 @ max 23——**max−2 代数**与
69=71−2 同模式，来源未定位，疑与 q_offset/MTP 相关）。

已知好状态（本 checkpoint 保留）：GDN 槽 `(query_length,0)` + 容器直转
——CPU BIT_EXACT True，GPU 越过 0 长 mask 崩溃，warmup 崩于此。

下个窗口的精确起点：
1. 读 masking_utils 的 mask 组装代数（kv_length/q_offset 如何组合出
   max−2），确定 GDN 槽被取样时该回答什么
2. 对照 HF DynamicCache 混合模型在 5.14 上为何正常（其容器
   get_seq_length/get_mask_sizes 的 alternating 特判）——这是权威模板
3. 注意：远端 transformers 的 sdpa_attention.py 曾插桩已恢复原样

## GDN graph capture 攻克 (2026-09-12, 4B capture 成功 2.3×, replay 数值待修)

mask 语义的完整解（多轮实证后）：
- HF StaticLayer.get_mask_sizes 返回**全宽 (max_cache_len, 0)**（非
  DynamicLayer 的 position 公式——早前对齐基类公式是误读）
- 容器对 GDN 槽查询重定向到第一个 attention 槽（HF alternating 约定）
- 容器补 is_compileable=True（HF StaticCache 属性）
- 教训两则：a) 插桩 print CUDA tensor 值 = host 同步 = capture 非法
  （自噬假线索）；b) prefill(HF cache) 答 69 与 decode(DS cache) 答 7
  的不一致即全宽 vs 位置公式的矛盾证据

结果：4B warmup ✓ → capture ✓ → **52.7-54.2 tok/s（vs eager 22.5，
2.3×）**。但输出退化（"Paris Paris..."，greedy 坍缩）——replay 数值
问题，嫌疑：KV 写入位置在 replay 未正确推进（write_position 静态
tensor 应被 replay 重读——需验证 DeepSpeedStaticLayer.update 的
arange+write_position 在 capture 内的行为）或 GDN 状态更新未进图。

下个窗口：对比 replay vs eager 单步 logits 定位（方法已验证）。

## Replay 退化定位：conv state 冻结 (2026-09-12)

反转证据链：
1. graph replay 本身无错：REPLAY0 vs EAGER0 maxdiff 2.09（bf16 级），
   argmax 一致——退化不是 capture/replay 问题
2. **eager DS-cache 路径在 GPU 上同样退化**（bf16/fp32 × sdpa/eager
   六配置全退化）——与 CPU bit-exact 的唯一差异 = GDN kernel 路径
   （FLA fused vs torch 回退）
3. 状态推进三分离：recurrent_state ✓ 推进、KV ✓ 推进、
   **conv_state 129.255→129.255 冻结** ← 真凶

机制：decode 步 GDN 走 `causal_conv1d_update`（FLA 原位 kernel，
不经过 cache.update_conv_state）——GPU 上该调用对 conv_state 的原位
写入静默失效（CPU torch 回退正常）。嫌疑：FLA kernel 对非连续输入
（in_proj 输出的 transpose 视图）的静默行为 / conv_state 布局与
kernel 期望不符 / HF 槽 state_idx 约定。

下个窗口（按性价比排序）：
1. 关键对照：纯 HF StaticCache（无 DS cache）GPU decode 是否也退化
   ——若退化则根因在 transformers 5.14 StaticCache×FLA，非我们的
   bind 设计（修复责任转移，材料价值反而更高）
2. 若 HF 原生正常：单层直调 causal_conv1d_update 复现冻结，查
   FLA kernel 的输入约束（contiguous/state 形状）

## conv 冻结根因追踪（三重反转, 2026-09-12/13）

1. HF 原生 generate（dynamic & static 均）完全正常；StaticCache 构造
   实证（spy_init）——问题不在 transformers、不在 StaticCache
2. **HF cache 本尊（pc）在我们 probe 的调用模式下同样退化**——问题
   在调用模式（2D/None/截断 mask 三种全试，全部退化）
3. **generate static 模式的真实 kwargs（forward pre-hook 实证）**：
   decode 步 `attention_mask = {'full_attention': (1,1,1,cur_len 增长),
   'linear_attention': None}`（**per-type 4D mask dict，GDN 得 None**），
   cache_position=None——与我 probe 的 2D mask 完全不同物种
   → 2D mask 的内部转换路径给 GDN 生成了非 None mask → 弄坏 conv 更新
   （与 conv_state sum 不变、首 token 对、逐步坍缩全部吻合）
4. 直喂 dict-mask（增长宽）→ sdpa 失配：mask(6) vs 全宽 K(29)——
   generate 内部在层前还有 K/mask 对齐处理未对齐

**下窗口精确起点**：sdpa 层入口抓 generate vs probe 的 K/mask 实际
形状各一组（一个 hook），补上最后这层对齐；随后静态全宽 mask 版进
graph。附带收获：GDN 的正确 decode 语义 = attention mask 为 None
（递归路径不吃 attention mask），2D→GDN mask 的内部转换是
transformers 5.15-dev 混合模型的疑似 bug（值得单独报 upstream）。

## conv 冻结：mask 理论否决，收窄到 GDN conv 分支 (2026-09-13)

本轮排除链（每项都有实验）：
- generate static 的 K=20 之谜解开：generation_config.max_length 默认
  20 = cache 全宽——generate 就是**全宽 K + 全宽 mask**（非增长截断）
- 全宽因果 dict-mask（GDN=None）+ 完全复刻 generate kwargs
  （cache_position=None）→ **cs_sum 仍 129.2547 一位不差冻结**
  → 2D-mask→GDN 转换理论、mask 形状理论、kwargs 理论全部否决
- 剩余唯一未检差异：GDN forward 的 conv 分支选择
  （use_precomputed_states/has_previous_state 容器语义）在 generate
  vs 手动 prefill+decode 下的不同——generate 的 prefill 细节
  （logits_to_keep 等）可能影响槽内状态初始化

下窗口终极一步（机械）：GDN forward conv 分支处插桩，打印
use_precomputed_states / 分支走向 / conv_state.data_ptr，
generate 与 probe 各跑一遍对照——一次钉死。

## conv 冻结根因钉死 + 残余单点 (2026-09-13 深夜)

终极插桩（GDN conv 分支对照）：
- generate：use_prev=True 分支，cs_sum 每步变化 ✓
- probe：**同分支同指针**，cs_sum 恒 129.255 ✗
- 关键洞察：cs_sum 恒定 = 每步写入相同值 = 症状可能是果不是因；
  两边 conv_state **初值**就不同（-354 vs 129）→ 分叉在 **prefill**

**根因实锤**：generate 的 prefill 传 `{'full_attention': None,
'linear_attention': None}`（dict 全 None）+ 无 cache_position；我传
2D ones mask + cache_position=arange → GDN prefill 的 conv_state
初始化错误。修复 prefill 为 generate 式 kwargs 后：
- **conv 每步更新 ✓，且前两步 cs_sum 与 generate 逐位一致**
  （-354.254 / -289.809）—— conv 问题关闭

残余单点：输出仍从第 2 步起分叉（我们 ' Paris'×N，generate
' Paris. \\n\\n The capital of...'）——第一步 token 正确（两边都是
Paris），第 2 步起不同。嫌疑收窄到 full-attn 侧：decode 循环我们仍
显式传 cache_position=[p]/position_ids=[[p]]（generate 传 None），
或 KV 写读位置。下一步：decode 也完全去掉显式位置参数对齐 generate，
若仍分叉则插桩 full-attn KV 读回。

## 追加（同夜最后）：位置参数也排除

decode 完全去掉 cache_position/position_ids（generate 式）→ 仍第 2 步
分叉。位置参数理论排除。残余单点更新：第 1 token 正确（prompt KV 读
取正确）→ 第 2 步错 = **step1 写入的 KV@pos5 被错误读回**。下窗口：
sdpa 层入口打印 K[:, :, 5, :4] 的值（generate vs probe 第 2 步），
直接对比写入内容与 rope 位置是否一致。conv 侧已完全关闭。

## 🎉 全绿达成 (2026-09-14, Qwen3.5-4B graph capture 端到端正确)

最后三个串联 bug（全部插桩实证）：
1. **prefill 2D mask 损坏 GDN 初始化**（前夜已修：dict 全 None）
2. **write_position off-by-one**：DS get_seq_length 曾返回 wp+1，模型据此
   推导 decode position → rope 全错位（KV5 maxdiff 2.52）；修为返回
   wp 本身（HF cumulative_length 语义：写入索引=已缓存计数）
3. **warmup 污染 GDN 状态**：capture 前 3 次 warmup 前向把 conv/
   recurrent 状态推进了 3 步 → 每个 replay 从污染态启动；修复：
   prefill 后快照 GDN 状态，warmup 后 capture 前 copy_ 恢复

最终数字（Qwen3.5-4B-Base, bf16, b1, 128 tok, 单卡）：
| 路径 | tok/s | 输出 |
|---|---|---|
| HF eager | 22.5 | ✅ golden |
| **graph capture（本修复）** | **30.4（+35%）** | **✅ 与 golden 逐 token 一致** |
| graph + segKI | 31.4（+3% vs graph） | ULP 分叉 @~68 tok（连贯，已知特性） |

64-token 口径 graph 曾达 54.8 tok/s（2.4×）——长上下文 decode 变慢是
GDN 状态与 KV 增长的正常代价。

修复总量回顾（GDN graph 支持从零到全绿）：6 处 mask/cache 协议对齐 +
prefill/位置/warmup 三 bug ≈ 80 行 Python，零 csrc 改动。
