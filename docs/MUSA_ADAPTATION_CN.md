# DeepSpeed 0.19.3 MUSA 适配修改说明（详细过程）

- **适配目标路径**：`/home/jd/wangmx/DeepSpeed_0.19.3`
- **对照基线（修改前）**：上游 DeepSpeed v0.19.3 源码（同树内已有 `mlu_*` 作为模板；干净基线亦对应 `/data/wangmx/DeepSpeed` 回滚后的 `real_accelerator.py`）
- **适配策略**：按 MLU Accelerator / `op_builder/mlu` 模式做最小移植，映射到 `torch.musa` + MCCL
- **原则**：只改该目录；不安装覆盖系统 `/home/DeepSpeed`（0.18.3）；不影响其他训练代码
- **文档位置**：
  - 本文件：`/home/jd/wangmx/DeepSpeed_0.19.3_MUSA适配修改说明.md`
  - 仓库摘要：`DeepSpeed_0.19.3/docs/MUSA_ADAPTATION.md`

---

## 0. 总体改动一览

| 类型 | 路径 | 动作 |
|------|------|------|
| 新增 | `accelerator/musa_accelerator.py` | 按 `mlu_accelerator.py` 移植 |
| 新增 | `deepspeed/accelerator/musa_accelerator.py` | 运行时镜像（内容一致） |
| 新增 | `op_builder/musa/` | 按 `op_builder/mlu/` 移植 |
| 新增 | `deepspeed/ops/op_builder/musa/` | 运行时镜像 |
| 修改 | `accelerator/real_accelerator.py` | 注册 / 校验 / 自动探测 / 工厂 |
| 修改 | `deepspeed/accelerator/real_accelerator.py` | 与上者对称修改 |

---

## 1. `real_accelerator.py`：支持列表注册

**涉及文件**（两处内容对称）：

- `accelerator/real_accelerator.py`
- `deepspeed/accelerator/real_accelerator.py`

### 修改前

```python
SUPPORTED_ACCELERATOR_LIST = ['cuda', 'cpu', 'xpu', 'npu', 'mps', 'hpu', 'mlu', 'sdaa', 'supa']
```

### 修改后

```python
SUPPORTED_ACCELERATOR_LIST = ['cuda', 'cpu', 'xpu', 'npu', 'mps', 'hpu', 'mlu', 'sdaa', 'supa', 'musa']
```

### 修改原因

`DS_ACCELERATOR` 环境变量与校验逻辑只允许列表内名称。不加入 `'musa'` 时，显式 `export DS_ACCELERATOR=musa` 会直接 `ValueError`，无法选用 MUSA 后端。

---

## 2. `real_accelerator.py`：显式 override 时校验依赖

位置：读取 `DS_ACCELERATOR` 后、对各加速器做 `import` 校验的分支（原 MLU 与 SUPA 之间）。

### 修改前

```python
        elif accelerator_name == "mlu":
            try:
                import torch_mlu  # noqa: F401
            except ImportError as e:
                raise ValueError("MLU_Accelerator requires torch_mlu, which is not installed on this system.")
        elif accelerator_name == "supa":
            try:
                import torch_supa  # noqa: F401 # type: ignore
            except ImportError as e:
                raise ValueError("SUPA_Accelerator requires torch_supa, which is not installed on this system.")
```

（中间无 `musa` 分支。）

### 修改后

```python
        elif accelerator_name == "mlu":
            try:
                import torch_mlu  # noqa: F401
            except ImportError as e:
                raise ValueError("MLU_Accelerator requires torch_mlu, which is not installed on this system.")
        elif accelerator_name == "musa":
            try:
                import torch_musa  # noqa: F401
            except ImportError as e:
                raise ValueError("MUSA_Accelerator requires torch_musa, which is not installed on this system.")
        elif accelerator_name == "supa":
            try:
                import torch_supa  # noqa: F401 # type: ignore
            except ImportError as e:
                raise ValueError("SUPA_Accelerator requires torch_supa, which is not installed on this system.")
```

### 修改原因

用户显式指定 `DS_ACCELERATOR=musa` 时，应尽早确认 `torch_musa` 已安装，避免落到工厂创建后再出现含糊错误；与 MLU / HPU / SUPA 的校验模式保持一致。

---

## 3. `real_accelerator.py`：自动探测顺序（关键）

位置：auto-detect 段，在 MLU 探测之后、SUPA/CUDA 探测之前插入 MUSA。

### 修改前

探测顺序大致为：XPU → NPU → SDAA → MPS → HPU → **MLU** → **SUPA** → **CUDA** → CPU。

```python
        if accelerator_name is None:
            try:
                import torch_mlu  # noqa: F401,F811
                accelerator_name = "mlu"
            except ImportError as e:
                pass
        if accelerator_name is None:
            try:
                # Detect Biren SUPA GPU. torch_supa spoofs torch.cuda ...
                import torch_supa
                ...
                    accelerator_name = "supa"
            except ImportError as e:
                pass
        if accelerator_name is None:
            try:
                import torch
                if torch.cuda.device_count() > 0 and torch.cuda.is_available():
                    accelerator_name = "cuda"
            except (RuntimeError, ImportError) as e:
                pass
```

（无 MUSA 探测。若仅有 MUSA 卡，常见情况是 `torch.cuda.is_available()` 因兼容层为真，从而 **误选 CUDA**。）

### 修改后

在 MLU 与 SUPA 之间增加：

```python
        if accelerator_name is None:
            try:
                import torch_musa  # noqa: F401,F811
                import torch

                # Detect MUSA before CUDA: torch_musa may expose CUDA-compatible APIs.
                if hasattr(torch, 'musa') and torch.musa.is_available():
                    accelerator_name = "musa"
            except ImportError as e:
                pass
```

即顺序变为：… → MLU → **MUSA** → SUPA → CUDA → CPU。

### 修改原因

1. **必须先于 CUDA**：`torch_musa` 常提供 CUDA 兼容 API，`torch.cuda.is_available()` 可能为真；若先判 CUDA，MUSA 机器会错误落到 `CUDA_Accelerator`。
2. **再检查 `torch.musa.is_available()`**：仅 `import torch_musa` 成功不够，需确认设备真正可用。
3. 放在 SUPA 之前：与「spoof CUDA 的加速器优先于 CUDA」同一思路（SUPA 注释已说明同类问题）。

---

## 4. `real_accelerator.py`：工厂创建实例

位置：`get_accelerator()` 按名称构造对象。

### 修改前

```python
    elif accelerator_name == 'mlu':
        from .mlu_accelerator import MLU_Accelerator
        ds_accelerator = MLU_Accelerator()
    elif accelerator_name == 'supa':
        from .supa_accelerator import SUPA_Accelerator
        ds_accelerator = SUPA_Accelerator()
```

### 修改后

```python
    elif accelerator_name == 'mlu':
        from .mlu_accelerator import MLU_Accelerator
        ds_accelerator = MLU_Accelerator()
    elif accelerator_name == 'musa':
        from .musa_accelerator import MUSA_Accelerator
        ds_accelerator = MUSA_Accelerator()
    elif accelerator_name == 'supa':
        from .supa_accelerator import SUPA_Accelerator
        ds_accelerator = SUPA_Accelerator()
```

### 修改原因

名称解析为 `musa` 后必须能实例化对应 Accelerator；否则 DeepSpeed 无法绑定设备、流、通信后端与 op_builder 目录。

---

## 5. 新增 `musa_accelerator.py`（相对 `mlu_accelerator.py` 的映射）

**修改前**：不存在 `musa_accelerator.py`；能力由 `mlu_accelerator.py` 提供给寒武纪 MLU。

**修改后**：新增

- `accelerator/musa_accelerator.py`
- `deepspeed/accelerator/musa_accelerator.py`（镜像）

以 MLU 为模板，主要差异如下。

### 5.1 类名 / 后端名 / 通信库

| 项 | 修改前（MLU 模板） | 修改后（MUSA） |
|----|-------------------|----------------|
| 类 | `MLU_Accelerator` | `MUSA_Accelerator` |
| `_name` | `'mlu'` | `'musa'` |
| `_communication_backend_name` | `'cncl'` | `'mccl'` |

**原因**：DeepSpeed 分布式初始化使用 `communication_backend_name()`；MUSA 栈通信为 **MCCL**，不是 CNCL。设备字符串与 `tensor.device` 前缀必须是 `musa:`。

### 5.2 torch API 命名空间

| 修改前 | 修改后 |
|--------|--------|
| `torch.mlu.*` | `torch.musa.*` |
| `device='mlu'` / `'mlu:N'` | `device='musa'` / `'musa:N'` |

覆盖：device / RNG / Stream / Event / memory / Tensor 工厂 / `on_accelerator()` 等。

**原因**：运行时绑定的是 `torch_musa` 暴露的 `torch.musa`，不能继续调用 `torch.mlu`。

### 5.3 内存 API 兼容（相对 MLU 的增强）

**修改前（MLU）**：直接调用 `torch.mlu.memory_cached` / `max_memory_cached` 等。

**修改后（MUSA）**：对部分 API 做 `hasattr` 回退，例如：

```python
def memory_cached(self, device_index=None):
    # torch_musa exposes memory_reserved; keep memory_cached alias for DeepSpeed callers.
    if hasattr(torch.musa, 'memory_cached'):
        return torch.musa.memory_cached(device_index)
    return torch.musa.memory_reserved(device_index)
```

`max_memory_cached` / `reset_max_memory_cached` 类似，回退到 `*_reserved` 或 `max_memory_allocated` / `reset_peak_memory_stats`。

**原因**：不同版本 `torch_musa` 内存统计 API 命名不完全与 CUDA/MLU 对齐；直接硬调用会在初始化或日志路径上 AttributeError。回退保证 DeepSpeed 调用方可运行。

### 5.4 `pin_memory` 签名

**修改前（MLU）**：`def pin_memory(self, tensor):`

**修改后（MUSA）**：`def pin_memory(self, tensor, align_bytes=1):`（仍调用 `tensor.pin_memory()`）

**原因**：与较新 DeepSpeed abstract / CUDA accelerator 签名对齐，避免因多余关键字参数报错。

### 5.5 op_builder 目录与环境变量

| 项 | 修改前（MLU） | 修改后（MUSA） |
|----|---------------|----------------|
| `op_builder_dir()` | `op_builder.mlu` / `deepspeed.ops.op_builder.mlu` | `op_builder.musa` / `deepspeed.ops.op_builder.musa` |
| `export_envs()` | `NEUWARE_HOME`, `CNCL`, … | `MUSA_HOME`, `MCCL`, `LD_LIBRARY_PATH`, `PATH` |
| `visible_devices_envs()` | `MLU_VISIBLE_DEVICES` | `MUSA_VISIBLE_DEVICES` |

**原因**：编译扩展与可见设备环境变量必须指向 MUSA 工具链；否则会找错 builder 或读错卡号。

### 5.6 torch 导入方式

**修改前（MLU）**：模块顶层 `import torch`（注释说明 setup 阶段可能无 torch）。

**修改后（MUSA）**：

```python
try:
    import torch
except ImportError:
    torch = None
```

**原因**：setup / 仅导入 op_builder 相关路径时允许无 torch，避免安装期硬失败（与部分上游 accelerator 写法一致）。

---

## 6. 新增 `op_builder/musa/`（相对 `op_builder/mlu/`）

**修改前**：无 `op_builder/musa/`（干净 0.19.3 无 MUSA builder）。

**修改后**：新增目录（并镜像到 `deepspeed/ops/op_builder/musa/`）：

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 `NotImplementedBuilder` / CPU Adam / CPU Adagrad / FusedAdam |
| `builder.py` | `MUSAOpBuilder` |
| `cpu_adam.py` | `CPUAdamBuilder` |
| `cpu_adagrad.py` | `CPUAdagradBuilder` |
| `fused_adam.py` | `FusedAdamBuilder` + `MUSAFusedAdam` |
| `no_impl.py` | 未实现 op 的占位 |

### 6.1 `builder.py`

**修改前（MLU）**：`class MLUOpBuilder(OpBuilder):`，使用 `CppExtension`。

**修改后（MUSA）**：`class MUSAOpBuilder(OpBuilder):`，逻辑同构（类名替换）。

**原因**：DeepSpeed 按 accelerator 的 `op_builder_dir()` 加载对应 Builder 基类；需独立 MUSA 命名空间，不能复用 `op_builder.mlu`。

### 6.2 `cpu_adam.py` / `cpu_adagrad.py`

**修改前**：继承 `MLUOpBuilder`。

**修改后**：继承 `MUSAOpBuilder`；源文件仍指向上游 CPU 实现：

```python
return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']
```

**原因**：CPU Adam 扩展与设备无关，可在 MUSA 训练中作为稳妥优化器路径（推荐 `torch_adam=True`，避免依赖 fused kernel）。

### 6.3 `fused_adam.py`（与 MLU 行为差异最大）

#### 修改前（MLU）

```python
class MLUFusedAdam:
    @staticmethod
    def multi_tensor_adam(...):
        torch.ops.torch_mlu.fused_adam(...)
```

假定 `torch.ops.torch_mlu.fused_adam` **一定存在**。

#### 修改后（MUSA）

```python
class MUSAFusedAdam:
    @staticmethod
    def multi_tensor_adam(...):
        if torch is None or not hasattr(torch.ops, 'torch_musa'):
            raise RuntimeError(
                "DeepSpeed FusedAdam on MUSA requires torch.ops.torch_musa.fused_adam; "
                "use DeepSpeedCPUAdam / torch optimizers, or install a torch_musa build that exports fused_adam.")
        musa_ops = torch.ops.torch_musa
        if hasattr(musa_ops, 'fused_adam'):
            return musa_ops.fused_adam(...)
        raise RuntimeError(
            "torch.ops.torch_musa has no fused_adam. ...")
```

#### 修改原因

当前集群 `torch_musa` **不一定**导出 `fused_adam`。若照搬 MLU 直接调用，会在优化器路径上出现难读的 AttributeError。改为：

1. 有 op 则走 fused；
2. 无 op 则给出可操作的明确错误（改用 CPU/torch Adam）。

推荐使用 `optimizer.params.torch_adam=True`，不依赖该 fused op。

---

## 7. 未修改但需知情的边界

| 项 | 说明 |
|----|------|
| 系统包 `/home/DeepSpeed`（0.18.3） | **未改**；可用 `PYTHONPATH` 隔离试用适配树 |
| `/data/wangmx/DeepSpeed` | 曾试改后已按要求回滚；**保留适配的是** `DeepSpeed_0.19.3` |
| 业务训练脚本 / LlamaFactory | **未改** |
| DeepSpeed 官方完整单测 | **未跑** |
| 多机 ZeRO-3 / 全量 fused kernel | **未覆盖** |
