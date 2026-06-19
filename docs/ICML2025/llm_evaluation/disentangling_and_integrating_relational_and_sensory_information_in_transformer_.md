---
title: >-
  [论文解读] Disentangling and Integrating Relational and Sensory Information in Transformer Architectures
description: >-
  [ICML 2025][LLM评测][注意力机制] 本文提出了 Dual Attention Transformer（DAT），通过在标准注意力机制中引入"关系注意力"头，将感知信息和关系信息解耦后并行处理再整合，在关系推理基准、数学问题求解、图像识别和语言建模等任务上均展现出显著的数据效率和参数效率提升。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "注意力机制"
  - "关系推理"
  - "Transformer"
  - "归纳偏置"
  - "关系注意力"
---

# Disentangling and Integrating Relational and Sensory Information in Transformer Architectures

**会议**: ICML 2025  
**arXiv**: [2405.16727](https://arxiv.org/abs/2405.16727)  
**代码**: [https://github.com/Awni00/dual-attention](https://github.com/Awni00/dual-attention)  
**领域**: LLM评测  
**关键词**: Dual Attention, 关系推理, Transformer架构, 归纳偏置, 关系注意力

## 一句话总结
本文提出了 Dual Attention Transformer（DAT），通过在标准注意力机制中引入"关系注意力"头，将感知信息和关系信息解耦后并行处理再整合，在关系推理基准、数学问题求解、图像识别和语言建模等任务上均展现出显著的数据效率和参数效率提升。

## 研究背景与动机
**领域现状**：Transformer 已成为最通用的神经网络架构，但其注意力机制本质上只路由"感知信息"（各个对象/token 的特征），缺乏对"关系信息"（对象间关系/比较）的显式处理。  

**现有痛点**：大量实验表明 Transformer 在需要关系推理的任务上表现不佳。虽然LLM通过大规模训练获得了一定的关系推理能力，但数据效率极低。  

**核心矛盾**：通用架构 vs. 归纳偏置的张力——纯通用架构在关系推理上数据效率低下，强归纳偏置的架构通用性受限。  

**本文目标**：在不破坏 Transformer 通用性的前提下，引入关系推理的归纳偏置。  

**切入角度**：区分感知信息（对象属性）和关系信息（对象间比较），为每类设计专用注意力机制。  

**核心 idea**：在多头注意力中用一部分头做标准感知注意力、另一部分头做关系注意力，让模型同时具备路由两种信息的能力。

## 方法详解

### 整体框架
DAT 保持 Transformer 结构（注意力 + MLP 交替堆叠），将多头注意力替换为"双注意力"（Dual Attention）：n_h^{sa} 个感知注意力头 + n_h^{ra} 个关系注意力头。两类头输出拼接后投影送入 MLP。兼容所有 Transformer 变体，支持因果掩码和 RoPE 等技术。

### 关键设计

1. **感知注意力（Sensory Attention = 标准 Self-Attention）**: 

    - 功能：路由各对象/token 的特征
    - 核心思路：标准 QKV 注意力，value 是源对象特征线性变换
    - 输出：Attention(x, y) = sum_i alpha_i * phi_v(y_i)

2. **关系注意力（Relational Attention）**: 

    - 功能：路由对象之间的关系信息
    - 核心思路：不检索源对象特征，而是计算目标与源之间的关系向量 r(x, y_i)，在 d_r 个特征子空间下做内积比较
    - 设计动机：每个维度对应一种"比较视角"，形成细粒度关系描述；引入符号标识符 s_i 标记来源
    - 关键公式：RelAttn(x, y) = sum_i alpha_i * (r(x, y_i) W_r + s_i W_s)
    - 和标准注意力的区别：标准注意力检索"源对象是什么"；关系注意力检索"源和目标之间有什么关系"

3. **符号分配机制（Symbol Assignment Mechanisms）**: 

    - 功能：为每个源对象分配抽象标识符，使接收方知道关系对应哪个对象
    - 三种变体：
        - **位置符号**：可学习位置嵌入
        - **相对位置符号**：相对位置嵌入 s_{j-i}，语言建模中更优
        - **符号注意力**：通过特征模板库匹配并分配有限符号库中的符号向量
    - 设计动机：通过有限符号库保持关系中心表示

4. **表达能力理论保证（Theorem 1）**: 

    - 证明关系注意力能以任意精度近似任意"选择后计算关系"函数 Rel(x, Select(x, y))

### 损失函数 / 训练策略
- 分类用交叉熵，语言建模用自回归 next-token prediction 损失
- 训练配置与标准 Transformer 基线一致，仅改变注意力头类型
- 语言建模在 FineWeb-Edu（100亿 token）训练，最大规模1.3B参数

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | DAT | Transformer | 提升 |
|------------|------|-----|------------|------|
| CIFAR-10 (ViDAT vs ViT) | 分类精度 | 89.7±0.1% (6.0M) | 86.4±0.1% (7.1M) | +3.3%, 参数更少 |
| CIFAR-100 (ViDAT vs ViT) | 分类精度 | 70.5±0.1% (6.1M) | 68.8±0.2% (7.2M) | +1.7%, 参数更少 |
| 语言建模 (1.3B) | scaling curve | 更优 | 基线 | 数据和参数效率更好 |
| Relational Games | 样本效率 | 显著更好 | 基线 | match_pattern差异巨大 |
| 数学推理 (多任务) | 字符级精度 | 各任务均优 | 基线 | 全面优势 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅关系注意力头 | 纯关系任务略优 | 差异很小，模型自动选择合适计算模式 |
| 对称 vs 非对称关系 | 视觉任务用对称更好 | 属性相似性关系天然对称 |
| 位置 vs 相对位置符号 | 相对位置在LM中更优 | 语言处理中相对位置更重要 |
| 可视化关系激活 | 编码语义关系 | model/state/machine之间激活值高 |

### 关键发现
- 在最难的 match_pattern 任务（需要二阶关系推理）上样本效率提升最显著
- 关系注意力在语言建模中学到人类可解读的语义关系（非句法关系）
- 计算复杂度与标准 Transformer 相同 O(n²)
- 关系注意力不仅提升性能，还提供新的可解释性视角

## 亮点与洞察
- 优雅地将"感知"与"关系"两种信息流在同一注意力框架内解耦和整合
- 1.3B 规模语言模型和权重已开源
- "符号"概念桥接联结主义和符号主义
- 多模态、多任务范式上全面验证

## 局限与展望
- 缺乏 Flash-Attention 等硬件级优化，实际训练速度慢
- 感知头和关系头数量比例需要调优，缺乏自适应机制
- 在10B+参数规模上的验证缺失
- mechanistic interpretability 留待未来工作

## 相关工作与启发
- Abstractor 架构（Altabaa et al., 2024）是关系注意力的直接灵感来源
- 启发：将关系信息作为一等公民在 Transformer 中处理可能是通向更高效推理的关键

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](../../NeurIPS2025/llm_evaluation/hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)
- [\[ICLR 2026\] Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents](../../ICLR2026/llm_evaluation/do_llm_agents_know_how_to_ground_recover_and_assess_evaluating_epistemic_compete.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICML 2025\] Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time](bounded_rationality_for_llms_satisficing_alignment_at_inference-time.md)

</div>

<!-- RELATED:END -->
