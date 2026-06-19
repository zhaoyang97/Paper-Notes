---
title: >-
  [论文解读] Steering Information Utility in Key-Value Memory for Language Model Post-Training
description: >-
  [NeurIPS 2025][可解释性][后训练优化] 提出 InfoSteer，一种轻量级方法，将 Transformer 的 FFN 层视为关联键值记忆，通过前向传播干预（提升低活跃记忆向量的 key coefficient）和反向传播正则化（最大化 key 分布熵）来促进预训练知识在后训练阶段的充分利用。在 Qwen/LLaMA/Gemma 三个系列 6 个模型上，15 个 ID+OOD 任务一致提升，且被引导的 LM 展现出自适应信息分配行为。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "后训练优化"
  - "FFN键值记忆"
  - "信息引导"
  - "记忆向量激活"
  - "SFT增强"
---

# Steering Information Utility in Key-Value Memory for Language Model Post-Training

**会议**: NeurIPS 2025  
**arXiv**: [2507.05158](https://arxiv.org/abs/2507.05158)  
**代码**: [GitHub](https://github.com/chili-lab/InfoSteer)  
**领域**: 可解释性  
**关键词**: 后训练优化, FFN键值记忆, 信息引导, 记忆向量激活, SFT增强

## 一句话总结

提出 InfoSteer，一种轻量级方法，将 Transformer 的 FFN 层视为关联键值记忆，通过前向传播干预（提升低活跃记忆向量的 key coefficient）和反向传播正则化（最大化 key 分布熵）来促进预训练知识在后训练阶段的充分利用。在 Qwen/LLaMA/Gemma 三个系列 6 个模型上，15 个 ID+OOD 任务一致提升，且被引导的 LM 展现出自适应信息分配行为。

## 研究背景与动机

当前 LLM 训练流程标准化为预训练 + 后训练（SFT/RLHF）两阶段。大量研究表明模型的基础能力和知识主要在预训练阶段注入，后训练被认为是"精化和对齐"这些能力的过程。然而一个关键问题被忽视了：

**预训练知识未被充分利用**：标准 SFT 既不显式地训练也不激励模型去检索和应用预训练中存储的知识。例如 Gemma-2-9B 基础模型在 HellaSwag 上为 90.1%，通过 InfoSteer 可达 95.7%（+5.6%），说明大量知识未被 vanilla SFT 激活。

**FFN 作为键值记忆的解读**：Geva et al. (2021) 的开创性工作表明，FFN 的第一层产生 key coefficient，第二层的行向量是 value（记忆向量），FFN 输出是记忆向量的加权和 $\text{FFN}(h) = \sum_{i=1}^{d_m} k_i \mathbf{v}_i$。如果某个 $k_i$ 远大于其他 $k_j$，只有 $\mathbf{v}_i$ 被利用，其他记忆向量被忽略。

**缺乏知识利用引导**：后训练阶段没有任何机制来引导模型更好地使用这些存储的知识——即使相关知识已经编码在参数中，模型也可能因为 key 分布过于集中而无法访问。

InfoSteer 的核心目标是：在后训练中促进 FFN key coefficient 的高熵分布，使更多记忆向量被激活利用，从而解锁预训练中已学到但未被使用的知识。

## 方法详解

### 整体框架

InfoSteer 可以无缝集成到标准 SFT 流程中，提供两种互补的引导方式：(1) 前向传播中的干预——直接修改 key coefficient；(2) 反向传播中的正则化——在损失函数中加入熵项。两种方法的共同目标是使 key 分布更均匀，激活更多记忆向量。

### 关键设计

1. **干预方法（Intervention）**：在每层 FFN 的前向传播中，找到 key coefficient 最小的 $p\%$（默认 $p=0.01$），将其提升至该层 key coefficient 均值的 $\alpha$ 倍：

$$k_s^{(l)} \leftarrow \alpha \cdot \frac{1}{d_m}\sum_{i=1}^{d_m} k_i^{(l)}, \quad \text{for } s \in \mathcal{I}^{(l)}$$

其中 $\mathcal{I}^{(l)}$ 是第 $l$ 层最小 $p\%$ 元素的索引集合。设计动机：直接提升被忽略的记忆向量的贡献，使它们参与 FFN 输出的计算。$\alpha$ 控制干预强度——实验表明 $p=1, \alpha=2$ 效果最佳，过度干预（$p=2, \alpha=5$）反而降低性能。

2. **正则化方法（Regularization）**：在训练损失中加入 key 分布的熵正则项：

$$\mathcal{L} = \mathcal{L}_{\text{LM}} - \lambda \sum_{l=1}^{L} H(\hat{\mathbf{k}}^{(l)})$$

其中 $\hat{\mathbf{k}}^{(l)} = \mathbf{k}^{(l)} / \sum_i k_i^{(l)}$ 是归一化后的 key 分布，$\lambda=0.01$ 控制正则化强度。负号使损失最小化等价于熵最大化，鼓励 key 分布更均匀。设计动机：通过梯度信号间接引导模型学习更均匀的 key 激活模式，比干预方法更"柔和"。

3. **细粒度记忆向量分析（Information Surrogate）**：为理解每个记忆向量 $\mathbf{v}_i$ 编码的内容，将其通过语言模型解码头映射为词表上的 logits：

$$\phi_i = \mathbf{v}_i \cdot W_{\text{decode}} \in \mathbb{R}^{|V|}$$

对 $\phi_i$ 做 softmax 得到概率分布 $P_i$，计算其熵 $H(P_i)$：低熵表示高特异性向量（编码了特定主题知识，如 {'quantum', 'physics', 'superposition'}），高熵表示通用向量。可据此实现主题级定向引导——但实验发现细粒度引导仅带来边际改进，通用方法已足够好。

4. **SwiGLU 适配**：现代 LLM（Qwen/LLaMA/Gemma）使用 SwiGLU 变体。InfoSteer 将 key coefficient 定义为 $\mathbf{k} = \sigma(hW_{\text{gate}}) \odot (hW_{\text{up}})$——即 down-projection 之前的中间激活。核心思想是关注"与记忆向量关联之前的输入"，无论是标准 FFN 还是 gated FFN。

### 损失函数 / 训练策略

基础训练即标准 SFT（语言建模交叉熵损失），InfoSteer 仅在此基础上添加干预或正则化。默认超参数：干预 $p\%=0.01, \alpha=1$，正则化 $\lambda=0.01$。所有实验报告 3 次独立运行的平均分数。训练数据和超参数设置与 vanilla SFT 完全一致，仅增加一行代码的修改。

## 实验关键数据

### 主实验 — ID 评估（9 个任务平均精度）

| 模型 | 方法 | BoolQ | PIQA | HellaSwag | WinoG | ARC-c | 说明 |
|------|------|-------|------|-----------|-------|-------|------|
| Qwen-2.5-1.5B | base | 64.2 | 78.5 | 80.1 | 76.4 | 61.2 | — |
| | vanilla SFT | 68.5 | 82.9 | 84.8 | 80.8 | 65.8 | — |
| | **+intervention** | **69.3** | **84.4** | **93.1** | **84.2** | **68.2** | HellaSwag +8.3 |
| Gemma-2-9B | base | 71.6 | 86.3 | 90.1 | 82.5 | 77.8 | — |
| | vanilla SFT | 74.3 | 90.1 | 94.8 | 86.9 | 82.0 | — |
| | **+intervention** | **77.2** | **91.8** | **95.7** | **88.5** | **83.4** | 全面提升 |
| LLaMA-3-8B | base | 70.3 | 85.6 | 90.8 | 81.9 | 75.3 | — |
| | **+intervention** | **77.1** | **90.2** | **96.3** | **87.4** | **81.6** | HellaSwag +5.5 |

### OOD 评估（GSM8K 训练，5 个算术数据集测试）

| 方法 | ID Eval (GSM8K) | OOD Eval (5 datasets avg) | 说明 |
|------|----------------|--------------------------|------|
| Base Model | 63.7 | 85.3 | — |
| + Vanilla SFT | 65.7 (+2.0) | 83.7 (**-1.6**) | SFT 过拟合导致 OOD 下降 |
| + Steered SFT w/ intervention | **66.8 (+3.1)** | **86.6 (+1.3)** | ID 和 OOD 同时提升 |
| + Steered SFT w/ regularization | 66.1 (+2.4) | 86.0 (+0.7) | 正则化也有效 |

### 消融实验（Steering Magnitude）

| 配置 | 平均精度 | 说明 |
|------|---------|------|
| Base Model | 71.4 | — |
| + Vanilla SFT | 72.6 | — |
| + intervention (p=1, α=1) | 73.8 | 温和干预 |
| + intervention (p=1, α=2) | **75.5** | 最优干预强度 |
| + intervention (p=2, α=5) | 72.8 | 过度干预反而降低 |
| + regularization (λ=-0.01) | 72.3 | 负熵正则→降低性能（反面验证） |
| + regularization (λ=0.01) | 73.4 | — |
| + regularization (λ=0.05) | 74.7 | 较强正则 |

### 任务类型分析

| 任务类型 | Steered SFT 平均增益 | 排名 |
|---------|--------------------|----|
| 阅读理解 | +3.9 | 1 |
| 知识密集 | +3.3 | 2 |
| 常识推理 | +2.3 | 3 |
| 数学 | +1.1 | 4 |
| 语言学 | -0.3 | 5 |

### 关键发现

- **预训练知识严重未被利用**：所有三个模型系列（Gemma/LLaMA/Qwen）都从 InfoSteer 中获得了显著提升，说明这是一个普遍现象而非个例。Gemma-2-9B 在 HellaSwag 上从 90.1% 到 95.7%（+5.6%），无需额外预训练或增加参数。
- **Vanilla SFT 过拟合风险**：OOD 实验中，vanilla SFT 虽提升了 ID 性能（+2.0），却降低了 OOD 泛化（-1.6）。InfoSteer 同时提升 ID 和 OOD，表明它鼓励的是更通用的知识利用而非对特定分布的记忆。
- **自适应信息分配**：被引导的 LM 对语义丰富的 token（名词、动词）投入更多表征资源，对简单过渡词（','、'and'）投入更少——这种行为在 vanilla SFT 中不明显。
- **知识密集型任务获益最大**：阅读理解（+3.9）和知识密集任务（+3.3）从引导中获益远大于语言学任务（-0.3），符合直觉——语言学任务更依赖表面模式而非深层知识。

## 亮点与洞察

1. **"预训练知识未被充分利用"**是一个深刻且重要的发现，挑战了"SFT = 知识利用"的隐含假设。它意味着现有 SFT 管线可能存在系统性的效率损失。
2. **FFN 作为键值记忆的操控视角**极为优雅——不修改架构、不增加参数，仅通过改变 key 系数分布就能释放更多预训练知识。
3. **实现极简**：干预方法可以用一行代码实现（找最小 p% 改为均值的 α 倍），正则化也仅需在损失函数中加一项。
4. **负熵正则化的反面验证**（λ=-0.01 → 性能下降）是一个漂亮的控制实验，增强了因果论证力度。

## 局限与展望

- 细粒度引导（基于语义特异性分析选择性激活）仅带来边际改进，未充分发挥潜力。可能需要更精细的层级定位策略。
- 理论上 key 分布高熵不一定总是好的——某些任务可能需要高度稀疏的记忆激活模式（如精确事实检索）。目前缺少对这种情况的分析。
- 方法在语言学任务上略有负面影响（-0.3），说明"激活更多记忆向量"并非对所有任务类型都有益。
- 仅验证了 SFT 场景，未探索在 RLHF/DPO 等其他后训练方法中的适用性。

## 相关工作与启发

- 与 model steering 方向（activation steering、representation engineering）的区别：InfoSteer 不是在激活空间中施加方向性干预，而是在 FFN 的 key 分布上做熵引导，更接近"信息利用优化"而非"行为控制"。
- Geva et al. 的 FFN 键值记忆解读是本文的理论基础，InfoSteer 将这一解读从分析工具升级为训练干预手段。
- 与知识蒸馏的对比：蒸馏从外部教师获取知识，InfoSteer 从模型自身参数中释放已有知识。二者可能互补。

## 评分

- **新颖性**: ⭐⭐⭐⭐ FFN 键值记忆引导是新视角，将分析性解读转化为可操作的训练干预
- **实验充分度**: ⭐⭐⭐⭐⭐ 6 个模型（3 系列 × 2 尺度）、15+ 任务、ID+OOD、消融+任务类型分析
- **写作质量**: ⭐⭐⭐⭐ 方法直觉清晰，FFN-as-memory 的可视化很有说服力
- **价值**: ⭐⭐⭐⭐⭐ 即插即用的 SFT 改进方案，适用于所有使用 FFN 的 Transformer 模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)

</div>

<!-- RELATED:END -->
