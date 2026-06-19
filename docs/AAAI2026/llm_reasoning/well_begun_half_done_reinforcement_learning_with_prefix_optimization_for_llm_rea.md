---
title: >-
  [论文解读] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning
description: >-
  [AAAI 2026][LLM推理][大语言模型推理] 发现 LLM 推理中的"起始锁定效应"（Beginning Lock-in Effect）——初始推理过程显著决定后续轨迹和最终结果，据此提出 PPPO 方法，仅优化前缀 token（约 26% 的 token）即可实现高达 18.02% 的准确率提升，同时减少输出 token 数量达 18.35%。
tags:
  - "AAAI 2026"
  - "LLM推理"
  - "大语言模型推理"
  - "RLVR"
  - "前缀优化"
  - "路径依赖"
  - "GRPO"
---

# Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning

**会议**: AAAI 2026  
**arXiv**: [2512.15274](https://arxiv.org/abs/2512.15274)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 大语言模型推理, RLVR, 前缀优化, 路径依赖, GRPO

## 一句话总结

发现 LLM 推理中的"起始锁定效应"（Beginning Lock-in Effect）——初始推理过程显著决定后续轨迹和最终结果，据此提出 PPPO 方法，仅优化前缀 token（约 26% 的 token）即可实现高达 18.02% 的准确率提升，同时减少输出 token 数量达 18.35%。

## 研究背景与动机

**RLVR 的训练效率问题**：基于可验证奖励的强化学习（RLVR，如 GRPO、DAPO）是提升 LLM 推理能力的关键技术（驱动了 DeepSeek R1、OpenAI o1 的成功）。然而现有方法存在根本性效率问题：

**全 token 均匀训练**：对所有生成的 token 不加区分地进行训练，忽略了不同 token 对最终性能的异质性贡献

**低影响 token 的拖累**：大量计算资源花费在优化低回报 token 上，反而阻碍了关键 token 的潜在提升

**现有解决方案的不足**：如 DAPO-FT 仅优化高熵 token（wait, however, rethink），虽鼓励探索但无法保证触发的推理路径质量

**核心洞察——路径依赖理论**：

论文从人类思维理论中的**路径依赖**（Path Dependence, David 1975）获得灵感：人类思维过程中，初始思维模式会实质性地约束后续的思维轨迹。高质量的初始思考引导走向理想结果（良性路径依赖），低质量的初始思考则导致次优结果（恶性路径依赖）。

**实验验证——BLE 现象**：论文通过控制实验验证了 LLM 推理中的类似现象：
- 从正确答案的前缀 token 开始推理：准确率提升高达 20.2%
- 从错误答案的前缀 token 开始推理：准确率下降高达 27.5%
- 即使在错误前缀后增加反思词（"wait"、"however"），最大恢复仅 9.2%

这说明 LLM 极难从低质量的初始推理中恢复，证实了 LLM 推理中的**起始锁定效应（BLE）**。

## 方法详解

### 整体框架

Progressive Prefix-token Policy Optimization (PPPO) 是一种新型 RLVR 方法，核心思想是：

**仅优化前缀 token 的梯度，屏蔽后续 token 的梯度更新**

通过教会 LLM"如何高质量地开始推理"，来正面影响整个推理轨迹：

$$P(\hat{y} = a | q, I_{high-quality}) > P(\hat{y} = a | q) > P(\hat{y} = a | q, I_{low-quality})$$

### 关键设计

#### 1. 前缀梯度屏蔽机制

对每个输出 $\mathbf{o}_i$，PPPO 仅保留前 $\eta$ 比例 token 的梯度，屏蔽其余 token：

$$H(j, \mathbf{o}_i) = \mathbb{1}(j \leq \lfloor \eta \cdot |\mathbf{o}_i| \rfloor)$$

其中 $\eta$ 为保留比例。PPPO 的优化目标在标准 GRPO/DAPO 的基础上引入 $H(\cdot, \cdot)$ 掩码：

$$\mathcal{J}_{PPPO}(\theta) = \mathbb{E}\left[\frac{1}{\sum_k |\mathbf{o}_k|} \sum_{i=1}^{N} \sum_{j=1}^{|\mathbf{o}_i|} H(j, \mathbf{o}_i) \cdot \min\left(r_{i,j}(\theta) \hat{A}_{i,j}, \text{clip}(\ldots)\hat{A}_{i,j}\right)\right]$$

设计动机：直接将训练资源集中到最具影响力的 token 上，而非分散在整个序列中。BLE 的实验发现证明前缀质量对整个推理轨迹有决定性影响。

#### 2. 渐进式前缀保留策略（Progressive Prefix Retention）

关键创新之一：$\eta$ 不是固定值，而是在训练过程中逐步增大：

$$\eta = \begin{cases} \eta, & \text{if } \Delta acc > 0 \\ \eta + \Delta\eta, & \text{if } \Delta acc \leq 0 \end{cases}$$

当验证集准确率不再提升时（连续多步无增长），增大 $\eta$。初始 $\eta = 15\%$，以 5% 步长递增至 35%。

设计动机：
- 从短前缀开始：降低学习复杂度，让 LLM 快速建立高质量推理起始的核心能力
- 逐步增长：基于已建立的扎实基础，扩展到更长序列时仍能保持高学习质量和稳定性
- 自适应触发：根据性能瓶颈自动调整，避免过早或过晚的切换

**为什么是 15%-35%？** 实验发现（Figure 2b）：
- 15% 时模型性能出现显著转折——BLE 在前 15% token 内显现
- 35% 时性能趋于稳定——BLE 在 35% 时基本确立

#### 3. 延续累积奖励（Continuation Accumulated Reward）

解决前缀 token 评估中的高随机性问题：

对每个输出 $\mathbf{o}_i$，提取前 $\eta$ 的 token 形成前缀 $\mathbf{b}_i$，然后从 $\mathbf{b}_i$ 出发生成 $G$ 个延续序列 $\{\mathbf{c}_j\}_{j=1}^G$，累积其分数作为前缀的奖励：

$$R_i = \sum_{j=1}^{G} \mathbb{1}(\hat{y}_{\mathbf{c}_j} = a) + \mathbb{1}(\hat{y}_{\mathbf{o}_i} = a)$$

设计动机：
- 单次采样评估前缀质量存在高度随机性（同一前缀可能导致正确或错误的延续）
- 通过多次延续采样取累积分数，降低随机性，提供更可靠的前缀质量信号
- 类似 Monte Carlo 思想：用多次模拟估计一个状态的价值

### 损失函数 / 训练策略

- **Backbone**：Qwen3 系列（1.7B、4B、8B），thinking mode
- **训练数据**：DAPO-Math-17K
- **采样**：每个问题采样 N=8 个输出，每个前缀采样 G=8 个延续（共 8×8=64 个采样）
- **学习率**：$1 \times 10^{-6}$
- **裁剪参数**：$\varepsilon_{low} = 0.2$，$\varepsilon_{high} = 0.28$
- **最大输出长度**：10240 tokens
- **基线对比均采样 64 个输出**，确保公平

## 实验关键数据

### 主实验

**5 个推理基准上的准确率对比 (avg@32)**:

| 方法 | AIME'24 | AIME'25 | MATH500 | AMC'23 | GPQA Diamond | 平均 |
|------|---------|---------|---------|--------|-------------|------|
| Qwen3-4B (基线) | 48.75 | 35.42 | 84.46 | 72.67 | 43.59 | 56.98 |
| + GRPO | 52.08 | 37.71 | 88.40 | 76.77 | 46.78 | 60.35 |
| + DAPO | 56.46 | 42.08 | 92.33 | 81.63 | 49.37 | 64.37 |
| + DAPO-FT | 56.25 | 42.08 | 92.38 | 82.00 | 49.21 | 64.38 |
| + **PPPO (本文)** | **63.54** | **53.44** | **94.60** | **83.06** | **52.07** | **69.34** |

| 方法 | AIME'24 | AIME'25 | MATH500 | AMC'23 | GPQA Diamond | 平均 |
|------|---------|---------|---------|--------|-------------|------|
| Qwen3-8B (基线) | 52.29 | 38.75 | 86.06 | 75.08 | 46.12 | 59.66 |
| + DAPO | 63.13 | 48.75 | 93.21 | 83.96 | 55.18 | 68.85 |
| + DAPO-FT | 63.75 | 49.38 | 93.65 | 84.11 | 54.77 | 69.13 |
| + **PPPO (本文)** | **72.19** | **59.69** | **94.73** | **86.75** | **58.13** | **74.30** |

### 消融实验

**训练效率对比**：

| 模型 | 方法 | 平均准确率提升 (AAI) ↑ | 优化 token 比例 (POT) ↓ | 学习效率 (LE=AAI/POT) ↑ |
|------|------|---------------------|----------------------|---------------------|
| Qwen3-4B | GRPO | 3.37 | 100% | 3.37 |
| Qwen3-4B | DAPO | 7.39 | 100% | 7.39 |
| Qwen3-4B | DAPO-FT | 7.39 | 20% | 37.02 |
| Qwen3-4B | **PPPO** | **12.36** | **26.17%** | **47.24** |
| Qwen3-8B | DAPO | 9.19 | 100% | 9.19 |
| Qwen3-8B | DAPO-FT | 9.47 | 20% | 47.36 |
| Qwen3-8B | **PPPO** | **14.64** | **24.83%** | **58.95** |

**延续采样次数消融**（Qwen3-4B）：

| 采样策略 | G | avg@4 ↑ | var@4 ↓ |
|---------|---|---------|---------|
| 单次 | 1 | 60.46 | 3.30 |
| 多次 | 4 | 66.11 | 1.47 |
| 多次 | 8 | 69.36 | 0.63 |
| 多次 | 16 | 69.53 | 0.56 |

### 关键发现

1. **PPPO 在 15 个设置中 14 个最优**（3 模型 × 5 基准），唯一例外仅差 0.14%
2. **训练效率提升巨大**：仅优化约 25% 的 token，准确率提升最高 14.64%，学习效率（LE）是 GRPO 的 17.5 倍
3. **推理效率同步提升**：PPPO 生成的 token 数量最少，最多减少 18.35%，说明高质量的推理起始导向了更简洁的推理路径
4. **前缀质量可迁移**：用 PPPO 训练的 Qwen3-4B 生成的前缀 token 可以指导 Qwen2.5-7B-Instruct 提升 9.04%
5. **累积奖励显著降低方差**：G=8 时方差从 3.30 降至 0.63，准确率提升 8.9 个百分点
6. **渐进式策略优于固定策略**：比固定 $\eta=35\%$ 提高 2.19% 准确率且减少 8.83% 训练 token

## 亮点与洞察

1. **BLE 发现具有普遍意义**：路径依赖在 LLM 推理中的验证为理解自回归生成特性提供了认知科学视角
2. **反直觉的高效率**：优化更少 token 获得更好效果，挑战了"更多优化=更好结果"的直觉
3. **"从简到繁"的课程学习**：渐进式前缀保留策略自然形成了课程学习效果
4. **前缀质量的跨模型可迁移性**：暗示高质量推理起始具有一定通用性，不局限于特定模型
5. **与人类认知的类比**：将认知科学中的路径依赖理论成功映射到 LLM 推理，连接了两个研究领域

## 局限与展望

1. **$\eta$ 阈值的选择**：15%-35% 基于实验观察，不同模型/任务可能需要不同范围
2. **计算开销**：每个前缀需要额外采样 G=8 个延续，总采样量为 8×8=64
3. **仅验证了数学/科学推理**：BLE 在代码生成、创意写作等其他任务中是否同样成立有待验证
4. **缺少更大模型实验**：最大仅用 Qwen3-8B，32B/70B 级别模型的效果未知
5. **前缀边界的"硬截断"**：token 粒度的截断可能切断推理的逻辑单元中间

## 相关工作与启发

- **GRPO** (Shao et al., 2024)：去掉 critic model 的 RL 优化框架
- **DAPO** (Yu et al., 2025)：解决 GRPO 的熵坍缩等四个问题
- **DAPO-FT** (Wang et al., 2025c)：优化高熵分叉 token，是最接近的对比方法
- **Path Dependence** (David, 1975)：人类思维中的路径依赖理论
- **DeepSeek R1**：RLVR 技术路线的成功案例
- 启发：**轨迹特定优化**优于全 token 均匀优化；认知科学理论可以有效指导 LLM 训练策略设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — BLE 的发现和 PPPO 方法都具有很强的原创性，从认知科学到 LLM 的跨学科洞察
- 实验充分度: ⭐⭐⭐⭐⭐ — 3 种模型规模、5 个基准、4 种基线方法、多维度消融
- 写作质量: ⭐⭐⭐⭐⭐ — 动机-发现-方法-验证的逻辑链完整流畅
- 价值: ⭐⭐⭐⭐⭐ — 对 RLVR 训练效率有实际指导意义，BLE 发现可影响后续大量研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](../../ACL2026/llm_reasoning/revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](../../ICLR2026/llm_reasoning/textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)
- [\[ICML 2026\] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling](../../ICML2026/llm_reasoning/blending_supervised_and_reinforcement_fine-tuning_with_prefix_sampling.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](../../ICML2026/llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)

</div>

<!-- RELATED:END -->
