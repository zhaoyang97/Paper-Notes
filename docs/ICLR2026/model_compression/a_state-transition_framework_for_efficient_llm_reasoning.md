# A State-Transition Framework for Efficient LLM Reasoning

**会议**: ICLR 2026  
**arXiv**: [2602.01198](https://arxiv.org/abs/2602.01198)  
**代码**: 有  
**领域**: LLM效率  
**关键词**: efficient reasoning, linear attention, state transition, KV cache, long CoT  

## 一句话总结
提出将 LLM 推理过程建模为状态转移过程的高效推理框架，用 Linear Attention 将历史推理步骤的信息压缩为状态矩阵，使注意力复杂度从 $O(C^2)$ 降为 $O(C)$、KV cache 从 $O(C)$ 降为 $O(1)$，同时不缩短 CoT 序列，保持推理能力。额外的动量 momentum 策略缓解了噪声推理步导致的 overthinking 问题。

## 研究背景与动机
1. **领域现状**：Long CoT（如 o1、R1）显著提升了 LLM 的推理能力，但 Transformer 注意力的二次复杂度使得长 CoT 的计算和内存成本极高。
2. **现有痛点**：现有高效推理方法主要压缩 CoT 序列（缩短、删tokens、改写），但这与 test-time scaling 矛盾——压缩 CoT 会损害推理能力。
3. **核心矛盾**：高效性要求减少计算，但推理能力要求保留完整的推理链。压缩 CoT 内容 vs 压缩 CoT 的注意力计算是两回事。
4. **本文要解决什么？** 如何在不缩短 CoT 的情况下降低推理的计算和内存开销？
5. **切入角度**：每个推理步骤中，有用的推理信息（结论）远少于语言信息（语法、措辞）。用 linear attention 的状态矩阵只记录推理信息，扔掉语言信息。
6. **核心idea一句话**：让每个推理步骤的 token 通过线性注意力的状态矩阵高效访问历史推理信息，而非显式注意历史 token。

## 方法详解

### 整体框架
替换 LLM 的 softmax attention 为 Mixed Attention Module (MAM)：SA 子模块只注意当前步+query prompt 的 token，LA 子模块维护一个状态矩阵 $S_t$ 记录已完成推理步的信息。当前步的每个 token 通过 $\mathbf{o} = \mathbf{q} \cdot S_t$ 从状态矩阵中检索历史推理信息。

### 关键设计

1. **Mixed Attention Module (MAM)**:
   - 做什么：用 SA+LA 双路注意力替代原始 softmax attention
   - SA 子模块：保持原始 LLM 的 softmax attention，但每个 token 只注意 query prompt 和当前推理步的 token（完成步的 KV 被清除）
   - LA 子模块：用 linear attention 维护状态矩阵 $S_t = \sum_{i=1}^{t} k_i^T v_i$，每个 token 用查询向量 $q$ 从中检索历史信息。用 gating 机制控制历史信息的使用量
   - 设计动机：SA 确保当前步内的精确注意力不丢失，LA 提供对历史的高效访问。注意力从 $O(C^2)$ 降到 $O(C)$，KV cache 从 $O(C)$ 降到 $O(1)$

2. **State-based Reasoning Strategy（动量纠偏）**:
   - 做什么：用动量积累的全局推理方向纠正噪声推理步的偏差
   - 核心思路：将每步的状态变化 $\nabla_t = S_t - S_{t-1}$ 视为"梯度"，用动量积累全局方向 $\bar{\nabla}_{t-1} = \frac{1}{t-1}\sum \nabla_i$。完成第 $t$ 步后纠正：$\hat{\nabla}_t = (1-\alpha)\nabla_t + \alpha\bar{\nabla}_{t-1}$
   - 设计动机：基于 linear attention 的 TTT 视角——状态更新等价于梯度下降，动量方法是经典的梯度噪声缓解手段

3. **训练策略**:
   - 只训练 LA 子模块参数（LoRA）和思考模式特殊 token
   - 双损失：自回归损失 $\mathcal{L}_{AR}$ + 与 base model 的知识蒸馏损失 $\mathcal{L}_{KD}$
   - 高熵 token 分割 CoT 为推理步，每步用特殊 token 标注思考模式类型

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{AR} + \beta \mathcal{L}_{KD}$。用 95K 高质量数学 CoT 样本训练。基于 Qwen2.5 系列，1.5B 到 14B。

## 实验关键数据

### 主实验（vs 高效推理基线）

| 方法 | 类型 | GSM8K Acc | MATH-500 Acc | 推理延迟↓ |
|------|------|-----------|-------------|---------|
| Base model | 完整注意力 | 80.1 | 78.8 | 基线 |
| LightThinker | 压缩 CoT | 较低 | 较低 | 低 |
| INFTYTHINK | 摘要压缩 | 较低 | 较低 | 低 |
| H2O | KV cache 裁剪 | 低 | 低 | 低 |
| **Ours** | **状态转移** | **≥Base** | **≥Base** | **显著降低** |

### 关键发现
- 推理性能不降反升：状态转移框架在多个 benchmark 上等于甚至超过全注意力 base model
- 推理延迟大幅降低：CoT 越长优势越明显（因为状态矩阵大小恒定）
- 动量纠偏策略有效：消融显示它在 AIME 等难题上显著提升了正确率
- 在 1.5B, 7B, 14B 三个规模上一致有效
- 知识蒸馏损失对训练 LA 子模块至关重要

## 亮点与洞察
- **"不缩短 CoT，只缩短注意力"的思路**：完美绕开了高效推理与推理能力的矛盾——保留完整推理链但让注意力只在当前步内计算
- **TTT 视角下的动量纠偏**：利用 linear attention 等价于在线学习的理论性质，自然引入动量来抑制噪声步，理论优美且实验有效
- **对 test-time scaling 友好**：framework 允许 CoT 无限延长而计算和内存都线性增长

## 局限性 / 可改进方向
- Linear attention 的表达能力可能不如 softmax attention——复杂的跨步推理依赖可能在状态矩阵中丢失
- 高熵 token 分割推理步的方式可能不够精确，依赖训练数据的质量
- 仅在数学推理上验证，代码/科学推理等其他领域效果未知
- 需要从 DeepSeek-R1 蒸馏版本初始化，增加了训练 pipeline 复杂度

## 相关工作与启发
- **vs LightThinker（压缩 CoT）**: LightThinker 用特殊 token 压缩推理信息，但仍在 softmax attention 框架内；本文用 linear attention 直接替代，更彻底
- **vs TTT（Test-Time Training）**: 本文利用了 TTT 的 linear attention 理论视角，但将其应用于推理效率而非能力提升
- **vs KV cache 压缩（H2O, SapLLM）**: 这些方法在注意力分数上做选择性保留，可能丢失关键信息；本文通过双路设计（SA+LA）保证当前步内精确+历史步高效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 状态转移建模+混合注意力的设计非常新颖
- 实验充分度: ⭐⭐⭐⭐ 多规模模型、7 个 benchmark、多基线对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰，但符号和公式较多
- 价值: ⭐⭐⭐⭐⭐ 为长 CoT 推理的高效化提供了根本性解决方案
