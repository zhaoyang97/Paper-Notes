---
title: >-
  [论文解读] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation
description: >-
  [CVPR 2025][视频生成][文本到视频] RAPO 提出一个检索增强的 Prompt 优化框架，通过从训练数据中构建关系图检索相关修饰语、微调 LLM 重构句式、以及判别器选取最优 prompt，将用户简短 prompt 转换为与训练数据分布对齐的优化 prompt，在 VBench 上将多物体生成从 37.71% 提升至 64.86%。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "文本到视频"
  - "提示学习"
  - "检索增强"
  - "关系图"
  - "LLM微调"
---

# The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation

**会议**: CVPR 2025  
**arXiv**: [2504.11739](https://arxiv.org/abs/2504.11739)  
**代码**: 无（项目主页提及 GitHub）  
**领域**: 图像生成 / 视频生成  
**关键词**: 文本到视频, Prompt优化, 检索增强, 关系图, LLM微调

## 一句话总结

RAPO 提出一个检索增强的 Prompt 优化框架，通过从训练数据中构建关系图检索相关修饰语、微调 LLM 重构句式、以及判别器选取最优 prompt，将用户简短 prompt 转换为与训练数据分布对齐的优化 prompt，在 VBench 上将多物体生成从 37.71% 提升至 64.86%。

## 研究背景与动机

**领域现状**：文本到视频（T2V）生成模型（LaVie、Latte 等）在大规模数据集训练后取得了显著进步。然而研究发现，使用详细、长格式的 prompt 通常比用户提供的简短描述能产生更高质量的视频。

**现有痛点**：用户提供的 prompt 通常简短且缺少必要细节。直接用 GPT-4 或 Open-Sora 的 prompt 扩展虽然增加了更多描述，但往往引入与训练数据不匹配的冗长复杂词汇和句式，反而误导模型，导致生成质量下降甚至不如原始简短 prompt。现有 T2I 的 prompt 优化方法对视频生成效果有限，特别是在时间维度（动作流畅度、时间一致性）上改善甚微。

**核心矛盾**：T2V 模型对输入 prompt 极度敏感，但"好 prompt"是模型特定的——需要与训练数据的词汇和句式分布对齐。用户不知道训练数据长什么样，通用 LLM 扩展的 prompt 也未必匹配。

**本文目标**：设计一个系统化的 prompt 优化框架，让优化后的 prompt 在保留用户语义的同时，尽可能贴合 T2V 模型训练数据的词汇、长度和格式分布。

**切入角度**：通过系统分析训练数据（Vimeo25M）发现，视频质量与 prompt 中动词-名词短语的选择和句式结构高度相关。因此需要从训练数据中提取统计规律来指导优化。

**核心 idea**：构建训练数据的关系图（场景→修饰语的图结构），通过检索增强为用户 prompt 添加相关修饰语，再用微调的 LLM 将句式重构为训练数据的格式，最后由判别器从两条优化路径中选择最优 prompt。

## 方法详解

### 整体框架

RAPO 包含三大模块：（1）词汇增强模块：从关系图中检索并通过 LLM 融合修饰语；（2）句式重构模块：微调 LLM 将增强后的 prompt 重构为训练数据格式；（3）Prompt 选择模块：判别器 LLM 在两条路径的输出中选择更优的一个。两条路径为——路径 A：词汇增强→句式重构；路径 B：直接用通用 LLM 改写。

### 关键设计

1. **关系图构建与检索**:

    - 功能：从训练数据中系统化提取场景-修饰语的结构化知识，为用户 prompt 提供高质量的增强候选
    - 核心思路：从 Vimeo25M 训练数据中用 Mistral LLM 提取约 210 万条有效句子中的场景和对应修饰语（主体、动作、氛围描述）。场景作为核心节点，修饰语作为子节点构建关系图 $\mathcal{G}$。对新的用户 prompt，用预训练句子编码器（all-MiniLM-L6-v2）提取特征，通过余弦相似度先检索 top-k 相关场景，再检索这些场景下的 top-k 相关修饰语
    - 设计动机：直接让 LLM 生成修饰语容易产生与训练数据不匹配的词汇；从训练数据本身检索确保增强内容来自模型"见过"的分布

2. **迭代融合+句式重构**:

    - 功能：将检索到的修饰语逐个融入 prompt 并重构为训练数据的统一格式
    - 核心思路：词汇增强采用迭代融合机制 $x_i^{m+1} = f(x_i^m, p_i^m)$，每次用冻结的 LLM $\mathcal{L}$（GPT-4）将一个修饰语合理地融入当前 prompt，逐步丰富语义。融合完成后，用指令微调的"重构 LLM"$L_r$（基于 LLaMA 3.1 LoRA 微调）将增强 prompt 重构为训练数据的长度和格式。$L_r$ 的训练数据为约 86k 对（格式不同但语义相同的 prompt 对），目标是让输出的句式分布匹配训练数据
    - 设计动机：一次性融入所有修饰语容易导致句子不自然；迭代融合保证每步的语义连贯性。单独的句式重构步骤确保最终 prompt 在长度和格式上与训练分布对齐——实验证明 RAPO 优化后的 prompt 长度分布最接近训练集

3. **Prompt 判别器选择**:

    - 功能：在两条优化路径（检索增强路径 vs 直接 LLM 改写路径）的输出中自动选择更适合 T2V 生成的 prompt
    - 核心思路：微调一个判别器 LLM $\mathcal{L}_d$（LLaMA 3.1 LoRA，3 epochs），输入为原始 prompt $x_i$ 和两个候选 $x_r, x_n$，输出选择标签。训练数据通过实际生成视频并用自动评估指标（根据 prompt 内容自动选择相关评估维度）来确定哪个候选更好。训练集包含 7k 由 Mistral 生成的覆盖 VBench 所有维度的文本
    - 设计动机：单一路径可能在某些 prompt 上失效，双路径+判别器提供了冗余和选择机制，提高了整体鲁棒性

### 损失函数 / 训练策略

重构模型 $L_r$ 和判别器 $L_d$ 都使用 LLaMA 3.1 的 LoRA 指令微调。$L_r$ 训练 8 epochs，$L_d$ 训练 3 epochs，batch size 32，LoRA rank 64，单张 A100 训练。关系图构建是离线一次性的。

## 实验关键数据

### 主实验

VBench 评估（LaVie 模型）：

| 方法 | 总分 | 成像质量 | 人体动作 | 物体类别 | 多物体 | 空间关系 |
|------|------|---------|---------|---------|--------|---------|
| LaVie 原始 | 80.89% | 69.00% | 95.80% | 92.09% | 37.71% | 37.27% |
| LaVie-GPT4 | 79.69% | 70.27% | 83.80% | 88.73% | 36.23% | 50.55% |
| LaVie-Open-sora | 79.75% | 70.42% | 87.00% | 91.29% | 36.52% | 54.37% |
| **LaVie-RAPO** | **82.38%** | **71.40%** | **96.80%** | **96.91%** | **64.86%** | **59.15%** |

EvalCrafter 评估：

| 方法 | 总分 | 文本-视频对齐 | 视觉质量 |
|------|------|-------------|---------|
| LaVie-RAPO | **256** | **74.38** | **66.62** |
| LaVie-Open-sora | 251 | 71.38 | 65.26 |
| LaVie 原始 | 248 | 69.60 | 64.81 |

### 消融实验

| 配置 | VBench 总分 |
|------|-----------|
| 词汇增强 only | 80.37% |
| 句式重构 only | 79.75% |
| 词汇增强+句式重构 | 81.58% |
| 句式重构+Prompt选择 | 81.75% |
| 词汇增强+Prompt选择 | 80.60% |
| **完整 RAPO** | **82.38%** |

不同 LLM 的鲁棒性（$\mathcal{L}$ 选择）：

| LLM | VBench 总分 |
|-----|-----------|
| GPT-4 | 82.38% |
| Mistral | 82.25% |
| LLaMA | 82.10% |

### 关键发现

- 多物体生成提升最显著：LaVie 从 37.71%→64.86%（+27.15pp），Latte 从 29.55%→52.78%（+23.23pp），说明训练数据匹配的描述能大幅改善多主体场景
- GPT-4 和 Open-Sora 的 prompt 优化有时反而降低性能（如 GPT-4 使 LaVie 人体动作从 95.80% 降至 83.80%），因为冗长复杂的描述会"迷惑"模型
- RAPO 优化后的 prompt 长度分布最接近训练集分布——这是性能提升的核心原因
- 三个模块各自贡献不同维度的提升，组合使用效果最佳（协同效应）
- RAPO 对 LLM 选择不敏感（GPT-4/Mistral/LLaMA 差异仅 ~0.3%），说明框架设计本身是关键
- 空间位置描述对多物体生成特别重要——attention map 可视化显示加入"相对空间位置"描述后物体分离更清晰

## 亮点与洞察

- **"让 prompt 适应模型"而非"让模型适应 prompt"的反直觉思路**：大多数工作试图让模型更好理解用户 prompt，本文反其道而行——分析训练数据的统计规律，将用户 prompt 改写为模型"最舒适"的格式。这个思路简单但深刻
- **关系图作为训练数据的结构化记忆**：将 210 万训练 prompt 压缩为场景→修饰语的图结构，实现高效检索。这种训练数据知识提取方法可推广到任何生成任务
- **对多物体生成的巨大提升**：27pp 的提升说明多物体失败很大程度上不是模型能力问题，而是 prompt engineering 问题。关键是加入明确的空间关系描述

## 局限与展望

- 关系图和重构模型需要针对每个 T2V 模型的训练数据重新构建，泛化到新模型时有迁移成本
- 仅在 LaVie 和 Latte 两个较早的 T2V 模型上验证，对 Sora、Kling 等最新模型的效果未知
- 判别器训练数据仅 7k，且依赖自动评估指标来标注偏好，可能引入偏差
- Prompt 优化是推理时的额外步骤，增加了延迟（需要多次 LLM 调用）

## 相关工作与启发

- **vs GPT-4 直接改写**: GPT-4 生成的 prompt 虽然信息更丰富但过长且词汇复杂，与训练分布不匹配，反而导致质量下降
- **vs Open-Sora prompt refiner**: 基于 LLaMA 3.1 微调但缺乏训练数据统计的细粒度指导，效果有限
- **vs Hao et al. (NeurIPS)**: 用强化学习优化 prompt 的美学评分，但针对 T2I 且不考虑训练数据分布对齐

## 评分

- 新颖性: ⭐⭐⭐⭐ 从训练数据构建关系图做检索增强 prompt 优化的思路新颖，双路径+判别器设计有工程巧思
- 实验充分度: ⭐⭐⭐⭐ 在 VBench/EvalCrafter/T2V-CompBench 三个 benchmark 上评估，消融全面，但只在两个较旧的模型上验证
- 写作质量: ⭐⭐⭐ 整体结构合理但部分描述冗长，公式符号略多
- 价值: ⭐⭐⭐⭐ 揭示了 prompt 与训练分布对齐的重要性，对实际应用中的 prompt engineering 有很好的指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[CVPR 2025\] Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](video-colbert_contextualized_late_interaction_for_text-to-video_retrieval.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)

</div>

<!-- RELATED:END -->
