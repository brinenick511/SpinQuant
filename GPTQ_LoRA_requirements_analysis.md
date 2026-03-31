# GPTQ 后按模块提取 H 与 W-Q 并合并 LoRA：需求与实现分析

## 1. 目标配置与需求
- 流程：`ptq.py` 路径，使用 `GPTQ`（非 `RTN`），且 `export_to_et=False`。
- 量化设定：weight-only（例如 `W4A16KV16` 或 `W2A16KV16`）。
- 新需求：在每一个 module 量化完成后，拿到该 module 的 `H` 与 `W-Q`，调用一个外部函数（在 `gptq_utils.py` 同级写一个 `rocob_utils.py` ）生成该 module 的 LoRA 更新，并把 LoRA 合并到该 module 权重中。
- 限制：先不关心 LoRA 生成函数内部实现细节，只关注接线与工程可行性。

## 2. 代码路径与责任划分
- `optimize_rotation.py`
  - 负责训练旋转矩阵（`R1` + per-layer `R2`），保存 `R.bin`。
  - 不执行 GPTQ/RTN 量化。
- `ptq.py`
  - PTQ 入口，调用 `ptq_model(...)`。
- `eval_utils/main.py::ptq_model`
  - 控制是否 rotate、是否 GPTQ/RTN、以及后续激活/KV量化配置。
  - 当 `args.w_bits < 16` 且 `not args.w_rtn` 时，走 `gptq_utils.gptq_fwrd(...)`。
- `eval_utils/gptq_utils.py::gptq_fwrd`
  - GPTQ 主体实现；按 layer、按 sequential group、按 module 执行统计与量化。

## 3. GPTQ（当前 PTQ 主路径）
- GPTQ 实际在 `model.model.layers` 内部线性层执行（`q/k/v/o + up/gate/down`）。
- `lm_head` 在该 GPTQ 路径中不量化（不在遍历集合内）。
- `embed_tokens` 也不在常规 GPTQ 路径中量化（`export_to_et` 特殊分支除外）。

## 4. 关于 sequential 分组与 H 生命周期
`gptq_fwrd` 的 `sequential` 分组（例如 `q_proj/k_proj/v_proj`）逻辑如下：
- 同组内每个 module 会创建一个独立 `GPTQ` 实例与独立 `H`。
- 通过同一次 forward + hooks 同步采样，因此 q/k/v 的输入统计来源相同，但实现上是三份独立 `H`。
- 删除时机是“按 module 逐个删除”，不是“整组统一删除”：
  - 每个 module `fasterquant(...)` 后会调用 `free()` 释放其 `H`。

## 5. 是否易于获取 `H` 与 `W-Q`
- `W-Q`：非常容易。
  - 量化前保存 `W_before = weight.clone()`。
  - 量化后读取 `Q_after = weight.clone()`（此时 weight 已被写成 fake-quant 后的浮点）。
  - 计算 `W_minus_Q = W_before - Q_after`。
- `H`：容易，但要在 `fasterquant` 内 `del self.H` 或外层 `free()` 前取走。

## 6. 关键 dtype/device 结论（当前实现）
### 6.1 `H`
- dtype：`torch.float32`
- device：`cuda`
- 原因：`H` 初始化为 `zeros(..., device=self.dev)` 且统计时使用 `inp.float()`。

### 6.2 `W-Q`
- 若直接用前后权重相减：
  - `W_before` dtype：模型权重 dtype（通常 `fp16` 或 `bf16`）
  - `Q_after` dtype：会被写回为层权重 dtype（通常 `fp16` 或 `bf16`）
  - `W_minus_Q` dtype：通常也是 `fp16`/`bf16`
  - device：`cuda`
- 若需要更稳定统计，可显式转 `float()` 后再相减。

## 7. 挂载兼容性评估（ActQuantWrapper/Linear）
最初方案是“挂 LoRA 模块”，兼容性风险点在于：
- `ActQuantWrapper` 内部持有 `self.module`（线性层）并缓存 `self.weight` 引用。
- 若替换整个模块类型，可能引入引用失配/state_dict/forward 签名问题。

当前已接受“LoRA 直接合并到权重”，则这些兼容性风险基本消失：
- 不替换模块类型，不改 forward。
- 仅对现有权重做增量合并（`weight.data += delta_lora`）。
- 与 `ActQuantWrapper` 保持天然兼容。

## 8. 实现难度与失败概率（最终版本）
前提：LoRA 生成函数内部视为黑盒，仅做调用与合并。

- 实现难度：低
- 失败概率：低（约 5%~12%）

主要剩余风险：
- dtype/device 对齐不当（`H` fp32，`W-Q` 可能 fp16/bf16）
- LoRA 增量幅度过大导致精度回退
- 短时显存峰值增加（取决于外部函数临时张量）

## 9. 推荐的最小改动实现策略（不涉及函数内部）
在 `eval_utils/gptq_utils.py` 的 per-module 量化循环中，按如下时序接线：
1. 量化前缓存 `W_before`
2. 在 `H` 被删除前获取 `H_current`
3. 调 `fasterquant` 完成该 module GPTQ
4. 读取 `Q_after` 并计算 `W_minus_Q`
5. 调外部函数：`delta_lora = f(H_current, W_minus_Q, module_meta)`
6. 直接合并到当前 module 权重：`weight.data += delta_lora`
7. 释放临时张量并继续下一个 module

该策略满足“每个 module 量化完立即处理”的需求，同时保持对现有代码结构的最小侵入。

## 10. 补充共识
- `optimize_rotation.py` 是第一阶段旋转学习阶段；GPTQ/RTN 在 `ptq.py` 的第二阶段；`ptq.py`顺带会测试，可能要把这部分独立出来。
