# ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space

**会议**: ICLR 2026  
**arXiv**: [2603.04948](https://arxiv.org/abs/2603.04948)  
**代码**: [https://github.com/VITA-Group/Nabla-Reasoner](https://github.com/VITA-Group/Nabla-Reasoner)  
**领域**: LLM推理  
**关键词**: test-time scaling, gradient-based optimization, differentiable optimization, reward model, inference-time reasoning  

## 一句话总结
提出 ∇-Reasoner，将推理时的搜索从零阶（采样+评估）升级为一阶（梯度下降），在 token logits 空间上通过可微文本优化（DTO）结合 reward 梯度和 LLM 似然来迭代改进解码策略，在数学推理任务上提升 10-40% 准确率的同时减少 10-40% 的模型调用次数。

## 研究背景与动机
1. **领域现状**：推理时计算缩放（inference-time scaling）已成为提升 LLM 推理能力的重要途径。现有方法包括 Best-of-N、Self-Consistency、Tree-of-Thought、RAP 等，通过多次采样和评估来寻找高质量答案。
2. **现有痛点**：这些方法本质上都是**零阶搜索**——仅利用 reward 的标量值来筛选候选，没有利用 reward 的梯度方向信息。当搜索空间随序列长度指数增长时，无方向的搜索变得低效，性能随计算预算增加而饱和。
3. **核心矛盾**：reward model 本身是可微的（基于 transformer 的分类器），梯度信息唾手可得却被完全浪费。零阶方法无法有效利用 reward landscape 的结构信息。
4. **本文要解决什么？** 如何在推理时利用 reward 梯度来高效地引导 LLM 输出向高 reward 区域移动，同时保持生成的流畅性？
5. **切入角度**：将 LLM 推理重新表述为连续优化问题——在 token logits 空间上做梯度下降，用 straight-through estimator 桥接离散与连续空间。
6. **核心idea一句话**：用一阶梯度下降代替零阶搜索来做推理时策略优化，在 logits 空间上同时最大化 reward 和 LLM 似然。

## 方法详解

### 整体框架
∇-Reasoner 是一个迭代解码框架：给定 prompt，LLM 先生成一个完整的 rollout 及其 logits，然后通过 DTO 优化这些 logits，从优化后的 logits 中重新采样第一个 token，配合 rejection sampling 决定是否接受，再继续下一个 token 的生成。整个过程逐 token 推进，每个 token 都可能经过梯度优化。

### 关键设计

1. **Differentiable Textual Optimization (DTO)**:
   - 做什么：在 token logits 空间上进行梯度下降，同时优化 reward 和 LLM 似然
   - 核心思路：优化目标 $\mathcal{L}(\mathbf{y}) = -\lambda r(\mathbf{y}|\mathbf{x}) - \log \pi_{LLM}(\mathbf{y}|\mathbf{x})$，其中 reward 项提供方向引导，NLL 项防止偏离 LLM 分布（避免 reward hacking）。用 Gumbel-softmax straight-through estimator 将离散 token 参数化为连续 logits，使梯度可以流过
   - 设计动机：梯度同时**双向传播**——前缀 token 通过 NLL 正则化约束后续 token 保持一致性，后续 token 通过 attention 将 reward 信号反传给前面的 token，实现了类似 look-ahead 的全局优化效果

2. **迭代解码 + Rejection Sampling**:
   - 做什么：将 DTO 嵌入逐 token 解码循环，每步只采纳能提升 reward 的 token 修改
   - 核心思路：DTO 优化后从 $\text{softmax}(\tilde{\mathbf{z}}_1/\tau)$ 重新采样第一个 token $\tilde{y}_1$。若 $\tilde{y}_1 \neq y_1$，则生成新续写并比较 reward：仅当新续写的 reward 更高时才接受新 token
   - 设计动机：rejection sampling 保证每次修改都是有益的，避免梯度优化的噪声干扰。实验显示 DTO 将 rejection rate 从 ~66% 降到 ~29-40%

3. **加速策略（三项）**:
   - **梯度缓存**：one-hot token 在优化过程中不频繁变化，缓存 $\partial\mathcal{L}/\partial\mathbf{y}$ 并复用，仅在 token 翻转时重新计算
   - **Rollout 复用**：前一步被拒绝后，其 rollout 轨迹可直接作为下一步的 rollout
   - **置信度+梯度引导的 token 选择**：仅对高熵且高梯度的 token 运行 DTO，跳过高置信度或低梯度的 token

### 损失函数 / 训练策略
无需训练（纯推理时方法）。DTO 的优化目标：$\mathcal{L} = -\log \pi_{LLM}(\mathbf{y}|\mathbf{x}) - \lambda \cdot r(\mathbf{y}|\mathbf{x})$，其中 $\lambda$ 平衡 reward 和 NLL 正则化。理论证明 DTO 的 sample-space 梯度下降等价于 PPO 的 Wasserstein gradient flow（Theorem 4.1），统一了预训练缩放和推理时缩放的理论框架。

## 实验关键数据

### 主实验

| 模型 + 基准 | Greedy | SC (N=8) | BoN (N=8) | RAP | GRPO | ∇-Reasoner |
|---|---|---|---|---|---|---|
| Qwen-2.5-7B MATH-500 | 43.8 | 69.8 | 70.2 | 68.6 | 70.8 | **71.0** |
| Qwen-2.5-7B AMC | 33.0 | 49.4 | 50.1 | 50.1 | 52.8 | 51.5 |
| Qwen-2.5-7B-Inst MATH-500 | 71.2 | 76.6 | 77.8 | 80.2 | - | **80.4** |
| Qwen-2.5-7B-Inst AMC | 43.0 | 55.5 | 55.9 | 54.6 | - | **56.8** |
| Qwen-2.5-7B-Inst AIME24 | 5.3 | 25.0 | 22.5 | 1.6 | - | **26.6** |
| Llama-3.1-8B-Inst MATH-500 | 40.6 | 54.8 | 52.2 | 55.4 | - | **55.8** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| DTO rejection rate (Qwen-Inst) | 28.9% | 对比无 DTO baseline 的 66.5%，大幅降低 |
| DTO rejection rate (Llama-Inst) | 40.1% | 对比 baseline 66.9% |
| Reward model 4B vs 8B (MATH-500) | 80.4 vs 80.8 | 更大 reward model 仅提升 0.4% |
| 模型调用次数 | 减少 10-40% | 对比 BoN/SC |

### 关键发现
- DTO 将 rejection rate 从理论值 66% 降到约 30%，证明梯度优化确实有效改进了每一步的策略
- 计算效率优势：transformer 的并行执行使梯度计算接近一次前向传播的开销；confidence/gradient-guided selection 跳过大量不需要优化的 token
- 对 reward model 质量不敏感（4B vs 8B 差距 <1%）
- test-time scaling 曲线上，∇-Reasoner 的帕累托前沿始终优于 BoN 和 SC

## 亮点与洞察
- **从零阶到一阶的范式转换**：test-time scaling 领域的本质性改进，首次证明一阶梯度在推理时同样可用且更高效
- **理论优美**：证明了 DTO 的 sample-space 梯度下降等价于 PPO 的 Wasserstein gradient flow，统一了预训练缩放（参数空间优化）和推理时缩放（样本空间优化）
- **梯度缓存 trick 可复用**：由于 softmax 的硬化性质，one-hot token 不频繁变化的观察可以推广到其他需要对离散结构做梯度优化的场景

## 局限性 / 可改进方向
- 性能受限于 base model 和 reward model 的能力上限，无法超越两者的联合瓶颈
- base model 和 reward model 必须共享同一词表才能做 end-to-end 的 logit 优化，限制了模型组合的灵活性
- 当前只验证了数学推理任务，在代码生成、开放问答等场景的表现未知
- 与 serving engine（如 vLLM）的集成需要额外工程，需在解码循环中插入反向传播

## 相关工作与启发
- **vs Best-of-N / SC**: 纯零阶采样筛选，∇-Reasoner 用一阶梯度直接优化，用更少采样获得更好结果
- **vs ToT / RAP**: 同样是引导搜索，但依赖启发式树搜索和 Q 值估计，∇-Reasoner 用可微优化直接求解更高效
- **vs GRPO (训练时方法)**: 不修改模型权重达到接近 GRPO 的效果，且理论证明了两者的数学等价性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从零阶到一阶的范式转换，理论和方法都很新
- 实验充分度: ⭐⭐⭐⭐ 数学推理覆盖充分，但任务类型偏少
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，图示直观，叙事流畅
- 价值: ⭐⭐⭐⭐⭐ 为 test-time scaling 开辟了新方向，有望成为重要基线
