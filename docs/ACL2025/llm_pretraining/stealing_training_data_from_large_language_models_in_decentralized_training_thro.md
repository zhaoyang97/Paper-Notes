---
title: >-
  [论文解读] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack
description: >-
  [ACL 2025][预训练][decentralized training] 提出 Activation Inversion Attack（AIA），首次系统揭示去中心化训练（流水线并行）中恶意阶段可通过截获中间激活值高效重构训练数据，在 Bloom-7B1 微调场景下可精确恢复 62% 的私人邮件和接近 100% 的生日信息。
tags:
  - "ACL 2025"
  - "预训练"
  - "decentralized training"
  - "activation inversion"
  - "privacy attack"
  - "pipeline parallelism"
  - "data leakage"
---

# Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack

**会议**: ACL 2025  
**arXiv**: [2502.16086](https://arxiv.org/abs/2502.16086)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: decentralized training, activation inversion, privacy attack, pipeline parallelism, data leakage

## 一句话总结

提出 Activation Inversion Attack（AIA），首次系统揭示去中心化训练（流水线并行）中恶意阶段可通过截获中间激活值高效重构训练数据，在 Bloom-7B1 微调场景下可精确恢复 62% 的私人邮件和接近 100% 的生日信息。

## 研究背景与动机

**去中心化训练**（Decentralized Training）基于流水线并行将模型层分配到多个异构设备上协同训练，是缓解 LLM 训练资源瓶颈的重要方案。例如 DeepSeek-V3（671B 参数）需要 2,664M H800 GPU hours，去中心化框架使更多参与者能够贡献算力。然而，在这种开放式协作中，各阶段之间传递的中间激活值和梯度可能暴露训练数据的隐私信息。

**现有安全研究的盲区**：当前去中心化训练的安全工作几乎只关注硬件故障容错（Thorpe et al. 2023、Jang et al. 2023）和恶意阶段对训练的干扰（Lu et al. 2024a），**隐私泄露风险几乎未被探索**。与联邦学习的梯度泄露攻击不同，去中心化训练中攻击者只能访问部分模型和局部激活/梯度，无法获取全局信息。传统方法（如 Zhu et al. 2019 的深度梯度泄露）因不可获取完整模型而失效。

**关键洞察**：作者通过预实验发现，同一数据在预训练模型和微调模型中的激活余弦相似度在早期层接近 100%，后期层仍高于 50%。这说明微调前后激活变化极小，意味着攻击者可以用预训练模型构建影子数据集来训练攻击模型。这个发现为"用激活值反推训练数据"提供了坚实的理论基础。

## 方法详解

### 整体框架

AIA 采用两步攻击策略：(1) 利用公开数据集和预训练模型构建影子数据集（text-activation 对），(2) 训练生成式攻击模型学习从激活到文本的逆映射，然后将该模型应用于实际训练中截获的激活值来重构受害者的训练数据。攻击者作为"诚实但好奇"（honest-but-curious）的阶段参与训练，不干扰训练过程因此难以被检测。

### 关键设计

1. **影子数据集构建（Shadow Dataset Construction）**:

    - 功能：在没有受害者训练数据的情况下，构建一个可用于训练攻击模型的代理数据集
    - 核心思路：直接使用 HuggingFace 上的预训练模型作为影子模型 $M_{sha}$，用公开数据集（WikiText）通过其前向传播生成对应的影子激活 $a_{sha} = M_{sha[1:i_{att}-1]}(d_{pub})$，构成 $(a_{sha}, d_{pub})$ 对。**无需任何额外训练**——预训练模型的泛化性保证了激活在微调前后保持稳定
    - 设计动机：降低攻击成本至最低。攻击者只需要知道目标模型的架构类型（如 GPT-2 / Bloom / LLaMA）并下载对应的预训练权重即可

2. **攻击模型训练（Attack Model Training）**:

    - 功能：学习从中间层激活值到原始文本的逆映射函数 $\phi \approx (f_{[1:i_{att}-1]})^{-1}$
    - 核心思路：攻击模型 $M_{att}$ 与受害者模型架构相同但**去掉了初始嵌入层**（因为输入是激活值而非 token），由解码器层和 lm_head 层组成，统一配置为 12 个解码器层。以影子激活为输入，以对应文本为目标进行训练
    - 设计动机：保持架构一致是关键——不同架构的模型对相同输入产生完全不同的激活分布（实验证明跨架构攻击 PPL 暴涨至 117~7400+）

### 损失函数 / 训练策略

攻击模型使用标准语言模型损失函数（teacher forcing）训练：

$$L = -\sum_{k=1}^{N} \log P(y_k | x_1, x_2, \ldots, x_{k-1})$$

其中 $y_k$ 是目标词，$x_i$ 代表输入的激活值序列。训练完成后，将实际训练中从前一阶段截获的激活 $a_{i_{att}-1}^{(t)}$ 输入攻击模型即可重构训练数据。

## 实验关键数据

### 主实验

**文本重构质量**（3 模型 × 4 数据集）：

| 受害者模型 | 数据集 | PPL↓ | ROUGE-1 | ROUGE-L | BLEU-1 | BLEU-4 | COS |
|-----------|--------|------|---------|---------|--------|--------|-----|
| GPT2-XL | PIIs | 3.73 | 0.84 | 0.84 | 0.77 | 0.59 | 0.89 |
| GPT2-XL | Pile | 1.65 | 0.98 | 0.98 | 0.95 | 0.89 | 0.97 |
| Bloom-7B1 | PIIs | 14.82 | 0.80 | 0.80 | 0.67 | 0.47 | 0.89 |
| Bloom-7B1 | OpenWebText | 4.64 | 0.95 | 0.95 | 0.89 | 0.80 | 0.95 |
| LLaMA3-8B | PIIs | 7.36 | 0.80 | 0.79 | 0.73 | 0.54 | 0.77 |
| LLaMA3-8B | Pile | 2.18 | 0.96 | 0.96 | 0.94 | 0.89 | 0.92 |

**隐私泄露攻击成功率（ASR）**：

| 受害者模型 | 方法 | 电话 ASR | 邮件 ASR |
|-----------|------|---------|---------|
| GPT2-XL | True-Prefix | 0.00 | 0.04 |
| GPT2-XL | SPT | 0.00 | 0.02 |
| GPT2-XL | **AIA** | **0.25** | **0.55** |
| Bloom-7B1 | True-Prefix | 0.01 | 0.18 |
| Bloom-7B1 | **AIA** | **0.42** | **0.62** |
| LLaMA3-8B | True-Prefix | 0.00 | 0.00 |
| LLaMA3-8B | **AIA** | **0.16** | **0.42** |

**各类 PII 精确恢复率**：

| PII 类型 | GPT2-XL | Bloom-7B1 | LLaMA3-8B |
|---------|---------|-----------|-----------|
| 生日 | **1.00** | **0.99** | **0.95** |
| 职业 | 0.97 | 0.98 | 0.89 |
| SSN | 0.76 | 0.57 | 0.38 |
| 地址 | 0.56 | 0.57 | 0.41 |
| 传真 | 0.25 | 0.48 | 0.20 |
| Bitcoin | 0.22 | 0.04 | 0.03 |
| UUID | 0.17 | 0.04 | 0.10 |

### 消融实验

| 消融维度 | 关键发现 | 说明 |
|---------|---------|------|
| 解码器层索引 | 层越靠近输入，攻击越好 | 余弦相似度降至 60% 以下时 PPL 仍 <40，仍可推断原始数据 |
| 模型规模 | 攻击效果与模型大小无关 | Bloom 560M→7B1 均表现良好，说明普适性 |
| 攻击模型架构 | 必须与受害者架构一致 | 跨架构攻击 PPL 暴涨：Mistral 攻击 GPT2-XL → PPL=117.45 vs 同架构 4.17 |

### 关键发现

- 去中心化训练中的激活值足以高精度重构训练数据，这是首次被系统证明的安全风险
- 结构化短数据（生日、职业）恢复率接近 100%，长随机序列（Bitcoin 地址、UUID）最难恢复（<20%）
- LLaMA3-8B 的对齐机制使传统基线攻击（True-Prefix、SPT）完全失效（ASR=0），但 AIA 仍能恢复 42% 的邮件——说明对齐对 AIA 防御不足
- 攻击在 honest-but-curious 模式下执行，不干扰训练过程，因此无法被常规检测手段发现

## 亮点与洞察

- 首次定义并系统验证了去中心化训练中的激活隐私泄露攻击面，是一个被忽视但影响严重的安全方向
- 攻击成本极低——仅需公开数据集和预训练模型权重，无需额外训练影子模型
- 微调前后激活的高相似度不仅是攻击可行的基础，对理解微调机制本身也有启发意义
- 架构敏感性的实验（跨架构攻击 PPL 暴涨 30~5000 倍）间接揭示了不同 Transformer 架构对内部表示空间的影响之大

## 局限与展望

- 假设攻击者已知受害者模型架构类型，现实中可能需要额外推断步骤
- 仅在微调场景下测试（5 epoch，模型过拟合），更温和的微调条件下攻击效果需进一步验证
- 从头预训练（from scratch）场景的攻击效果未知，此时激活相似度假设不再成立
- 未讨论针对 AIA 的防御方法（差分隐私、激活扰动、安全聚合等）
- 固定 6 阶段划分且第 3 阶段恶意的实验设置较为单一，更灵活的阶段配置未探索

## 相关工作与启发

- **vs 联邦学习梯度泄露** (Zhu et al. 2019): 梯度攻击需全局模型和全局梯度，AIA 仅需局部激活值，适用于流水线并行
- **vs 嵌入反演攻击** (Li et al. 2023, Morris et al. 2023): 嵌入反演假设攻击者可访问完整的静态模型，AIA 在动态训练过程中仅用部分模型
- **对后续研究的启发**: 去中心化训练框架需要引入激活噪声注入或差分隐私保护，同时需权衡安全性与训练效率的 trade-off

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示去中心化训练的激活隐私泄露风险，问题定义新颖重要
- 实验充分度: ⭐⭐⭐⭐ 3 模型 × 4 数据集 × 7 种 PII 类型 × 多维度消融，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，动机推导逻辑严谨，图示直观
- 价值: ⭐⭐⭐⭐ 对去中心化训练安全性有重要警示意义，推动该方向的防御研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)

</div>

<!-- RELATED:END -->
