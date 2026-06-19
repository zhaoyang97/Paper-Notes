---
title: >-
  [论文解读] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models
description: >-
  [NeurIPS 2025][预训练][LLM alignment] 提出 Residual Alignment Model (RAM)，将 LLM 对齐过程形式化为重要性采样，将大模型分解为冻结的 Proposal Module 和可训练的小型 Residual Aligner，以不到 1/8 参数实现可比甚至超越全参数 SFT/DPO 的对齐效果，同时解决了首 token 延迟问题。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "LLM alignment"
  - "importance sampling"
  - "residual alignment"
  - "modular alignment"
  - "token-level decoding"
  - "parameter-efficient"
---

# Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.19700](https://arxiv.org/abs/2505.19700)  
**代码**: 待确认  
**领域**: LLM预训练  
**关键词**: LLM alignment, importance sampling, residual alignment, modular alignment, token-level decoding, parameter-efficient

## 一句话总结

提出 Residual Alignment Model (RAM)，将 LLM 对齐过程形式化为重要性采样，将大模型分解为冻结的 Proposal Module 和可训练的小型 Residual Aligner，以不到 1/8 参数实现可比甚至超越全参数 SFT/DPO 的对齐效果，同时解决了首 token 延迟问题。

## 研究背景与动机

LLM 对齐（alignment）是确保模型输出符合领域需求和人类价值观的核心步骤。传统方法（SFT、RLHF、DPO）需要对整个大模型进行微调，存在以下问题：

**资源密集**：训练 8B+ 模型需要大量 GPU 资源，成本高昂

**部署碎片化**：不同领域需要部署独立模型，无法共享流量

**灵活性不足**：难以快速适配不同对齐需求

现有模块化方法（如 Aligner）通过训练适配器学习"纠正残差"来解耦对齐，但存在：
- **首 token 延迟**：需要等待上游模型完整响应后再纠正
- **OOD 风险**：Aligner 条件于参考响应 $\mathbf{y}'$，推理时参考来自 Proposal 而非真实分布，引入分布外问题

## 方法详解

### 整体框架

RAM 将目标对齐分布分解为两个模块的线性组合：

$$P_{\text{Aligned}}(\mathbf{y}|\mathbf{x}) \propto P_{\text{ProposalModule}}(\mathbf{y}|\mathbf{x}) \cdot P_{\text{ResidualAligner}}(\mathbf{y}|\mathbf{x})$$

**Proposal Module**（大模型，冻结）提供基础分布，**Residual Aligner**（小模型，可训练）作为重要性权重的估计器进行对齐补偿。

### 重要性采样分解

假设预训练模型 $P_M(\mathbf{y}|\mathbf{x})$ 估计通用分布 $P_\mathcal{D}$，对齐目标是逼近偏置子集分布 $P_\mathcal{S}$。利用重要性采样：

$$P_\mathcal{S}(\mathbf{y}|\mathbf{x}) = P_M(\mathbf{y}|\mathbf{x}) \cdot \frac{P_\mathcal{S}(\mathbf{y}|\mathbf{x})}{P_M(\mathbf{y}|\mathbf{x})}$$

引入自回归模型 $Q_\theta$ 估计重要性权重，归一化后得到 RAM：

$$P_\theta(\mathbf{y}|\mathbf{x}) = \frac{P_M(\mathbf{y}|\mathbf{x}) \cdot Q_\theta(\mathbf{y}|\mathbf{x})}{Z_\theta(\mathbf{x})}$$

其中 $Z_\theta(\mathbf{x}) = \sum_{\mathbf{y}} P_M(\mathbf{y}|\mathbf{x}) Q_\theta(\mathbf{y}|\mathbf{x})$。

### 句子级训练策略

从 SFT 目标出发，通过 Jensen 不等式和拉格朗日乘数法推导最终损失函数：

$$\mathcal{L}_{\text{SFT}}(P_\theta) = -\mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim \mathcal{S}}[\log Q_\theta(\mathbf{y}|\mathbf{x})] + \alpha \mathbb{E}_{\mathbf{x} \sim \mathcal{S}, \mathbf{y} \sim P_M}[\log Q_\theta(\mathbf{y}|\mathbf{x})]$$

- 第一项：在目标数据上最大化 Residual Aligner 的似然
- 第二项：控制 Proposal Module 分布的影响（$\alpha \in [0,1]$）

关键优势：$P_M$ 在训练全程冻结，仅需一次性生成采样数据。

### Token 级解码：Proposing-Aligning-Reducing (PAR) 采样

核心创新在于将序列级重要性采样转化为逐 token 的自回归过程：

$$P_\theta(y_l | y_{<l}, \mathbf{x}) = \frac{P_M(y_l | y_{<l}, \mathbf{x}) \cdot Q_\theta(y_l | y_{<l}, \mathbf{x})}{Z_\theta(y_{<l}, \mathbf{x})}$$

PAR 三步流程：
1. **Propose**：从 Proposal Module 用 nucleus sampling 生成 $n$ 个候选 token
2. **Align**：用 Residual Aligner 为每个候选计算重要性权重 $w(y_l^i) = \frac{Q_\theta(y_l^i | y_{<l}, \mathbf{x})}{Z_\theta(y_{<l}, \mathbf{x})}$
3. **Reduce**：归一化权重后做分类采样，选出最终 token

实现上利用 sparse Softmax：保留 Proposal 采样的 token，其余 logit 设为 $-\infty$，等价于标准 Softmax + 采样。

### 方差控制

- 训练阶段：$Q_\theta$ 学习补偿 $P_M$ 与 $P_\mathcal{S}$ 的差异，平滑极端权重
- 推理阶段：Top-P 区域采样 + 自归一化重要性采样

### KL 散度保护

当 $D_{KL}(P_M \| Q_\theta) > 0.1$ 时直接从 $P_M$ 采样，避免 Residual Aligner 退化。

## 实验关键数据

### 主实验：指令遵循与领域适应（SFT）

| 策略 | UltraChat LC% | TL;DR LC% |
|------|--------------|-----------|
| **Llama3.1-8B 家族** | | |
| W.Up 8B (基线) | 5.06 | 60.71 |
| SFT 1B (小模型独立) | 1.77 | 37.18 |
| W.Up 8B + Aligner 1B | 2.34 | 53.85 |
| **W.Up 8B + R.A. 1B** | **6.46** | **65.11** |
| SFT 8B (全参数微调) | 6.81 | 64.12 |
| **SFT 8B + R.A. 1B** | **7.32** | **66.11** |
| **Qwen2.5-14B 家族** | | |
| W.Up 14B | 10.42 | 53.11 |
| W.Up 14B + Aligner 3B | 8.08 | 53.85 |
| **W.Up 14B + R.A. 3B** | **12.32** | **57.76** |
| SFT 14B | 12.87 | 58.64 |
| **SFT 14B + R.A. 3B** | **12.88** | **64.91** |

1B Residual Aligner 配合 8B Proposal 可匹配甚至超过 8B 全参数 SFT。

### 偏好优化实验（Anthropic-HH）

| 策略 | Helpfulness GPT4-LC% | Harmlessness GPT4-LC% |
|------|---------------------|----------------------|
| SFT 8B | 58.59 | 65.31 |
| DPO 8B | 68.03 | 73.06 |
| DPO 8B + Aligner 1B | 55.31 | 70.12 |
| **DPO 8B + R.A. 1B** | **72.22** | **79.89** |
| DPO 14B | 74.53 | 71.41 |
| **DPO 14B + R.A. 3B** | **75.39** | **74.76** |

在 DPO 模型 win rate 已超 70% 的情况下，Residual Aligner 仍能提升 5–9%。Aligner 方法则因 OOD 问题显著劣于基线。

### 消融实验

**Residual Aligner 大小的影响**：
- 从 0.5B 到 8B，性能随大小增长但幅度不大（Llama3 平均 2.4%，Qwen2.5 平均 2.1%）
- 小模型即可获得大部分收益，性价比极高

**参数 α 的影响**：
- 在 1e-5 到 0.1 范围内，性能变化很小（CV 仅 1.67%–2.17%）
- 用户无需精调超参

**训练效率**：
- SFT 场景：相比全参数微调效率提升 **4 倍**
- DPO 场景：效率提升 **13.33 倍**

### 关键发现

1. RAM 的核心优势不在于最终性能的提升幅度，而在于用极少参数（1/8）逼近全参数效果
2. Aligner 在偏好优化任务上因 OOD 问题严重劣于 RAM
3. 小型 Residual Aligner（1B–3B）是最具性价比的选择
4. 多个 Residual Aligner 可共享同一 Proposal Module，实现跨域流量共享

## 亮点与洞察

1. **优雅的理论框架**：将对齐形式化为重要性采样，自然推导出模块分离，理论与实践高度统一
2. **首 token 延迟的彻底解决**：PAR 采样将序列级重采样转化为逐 token 操作，延迟与标准自回归解码相当
3. **训练时完全解耦**：Proposal Module 在训练阶段仅需一次性数据合成，之后完全不参与
4. **极佳的工程价值**：一个大模型 + 多个小 Aligner 的部署模式可大幅降低多领域对齐的成本
5. **通用性**：同一框架适用于 SFT、DPO 和领域适应三种场景

## 局限性

1. **同族要求**：Proposal Module 和 Residual Aligner 必须共享词表（同一模型家族），限制了组合灵活性
2. **KL 散度阈值**：0.1 的硬阈值缺乏理论支撑，可能在某些场景过于保守或激进
3. **评估局限**：主要依赖 AlpacaEval 2 框架，缺乏更多样的评估维度
4. **大型 Residual Aligner 收益有限**：增大 Aligner 参数的边际收益很小，暗示框架存在信息瓶颈
5. **未探索 RLHF**：仅验证了 SFT 和 DPO，未涉及在线 RLHF 场景

## 相关工作与启发

- 与 Residual EBM (Deng et al. 2020) 的关系：RAM 的形式与 Residual EBM 一致，但通过自回归分解实现了 token 级解码
- 与 Controlled Decoding 的区别：CD 学习前缀打分器，RAM 学习完整的重要性权重估计器
- 与 Aligner 的本质区别：RAM 直接建模 $P(\mathbf{y}|\mathbf{x})$ 而非 $P(\mathbf{y}|\mathbf{y}', \mathbf{x})$，避免 OOD
- 对 MoE 对齐的启发：RAM 的多 Aligner 共享 Proposal 模式可视为一种轻量级 Mixture-of-Experts 对齐方案

## 评分

- **创新性**: ⭐⭐⭐⭐ — 重要性采样框架理论优雅，PAR 解码策略新颖
- **实用性**: ⭐⭐⭐⭐⭐ — 训练效率提升显著，部署模式极具工程价值
- **实验严谨度**: ⭐⭐⭐⭐ — 双模型族 × 三任务 × 多基线，但评估维度偏单一
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，但符号较密集
- **推荐阅读指数**: ⭐⭐⭐⭐ — LLM 对齐方向的重要工作，模块化对齐研究者必读

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [\[ACL 2025\] FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](../../ACL2025/llm_pretraining/fr_spec_speculative_sampling.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)

</div>

<!-- RELATED:END -->
