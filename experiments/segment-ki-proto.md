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

## GPU-phase checklist（~1 小时实例时间）

1. 复跑 Qwen2.5/3.5 的 autotp_ki（bf16 + nccl）确认 GPU bit-exact
2. Triton fused silu·mul kernel（替换 `F.silu(g)*u` 两 op 为一），
   验证数值 + 计入 launch 统计
3. torch profiler 实测三条路径（native AutoTP / +KI composite / +KI triton）
   的 kernel 提交数，对照设计估算（native ~13-14/层 → KI ~9-10/层）

## Reproduction

```bash
WT=.worktrees/segment-ki
$WT/../dscpu/python $WT/experiments/test_segment_ki.py --mode baseline --out /tmp/ski_golden.pt
DS_SRC=$WT dscpu/torchrun --standalone --nproc_per_node=2 \
    $WT/experiments/test_segment_ki.py --mode autotp --ref /tmp/ski_golden.pt
DS_SRC=$WT dscpu/torchrun --standalone --nproc_per_node=2 \
    $WT/experiments/test_segment_ki.py --mode autotp_ki --ref /tmp/ski_golden.pt
# 跨家族（py3.12 + tok023）:
DS_SRC=$WT PYTHONPATH=/tmp/tok023 torchrun --standalone --nproc_per_node=2 \
    python3 $WT/experiments/test_segment_ki.py --mode autotp_ki \
    --ref /tmp/ski_q35_golden.pt --model Qwen/Qwen3.5-0.8B-Base
```
