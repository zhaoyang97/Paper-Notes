---
title: >-
  [论文解读] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][稀疏自编码器] 本文将稀疏自编码器（SAE）扩展到视觉-语言模型（如CLIP）上，提出了 MonoSemanticity score（MS）来定量评估神经元的单义性，并展示了通过操控 SAE 神经元可以直接引导多模态大模型（如 LLaVA）的输出，实现概念的插入与抑制。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "稀疏自编码器"
  - "单义性"
  - "CLIP"
  - "可解释性"
  - "模型引导"
---

# Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2504.02821](https://arxiv.org/abs/2504.02821)  
**代码**: [https://github.com/ExplainableML/sae-for-vlm](https://github.com/ExplainableML/sae-for-vlm)  
**领域**: 多模态VLM  
**关键词**: 稀疏自编码器, 单义性, CLIP, 可解释性, 模型引导

## 一句话总结

本文将稀疏自编码器（SAE）扩展到视觉-语言模型（如CLIP）上，提出了 MonoSemanticity score（MS）来定量评估神经元的单义性，并展示了通过操控 SAE 神经元可以直接引导多模态大模型（如 LLaVA）的输出，实现概念的插入与抑制。

## 研究背景与动机

**领域现状**：稀疏自编码器（SAE）近年在大语言模型（LLM）的可解释性研究中取得显著成功，通过将高维表征解耦为单义性的原子特征，帮助研究者理解和控制模型行为。然而在视觉-语言模型（VLM）领域，SAE 的应用仍局限于可解释分类或跨模型概念发现等初步探索。

**现有痛点**：VLM（如 CLIP）的神经元通常是"多义"的（polysemantic），即单个神经元会对多个不相关的概念（如手机和尺子）产生强响应。这种多义性严重阻碍了对模型内部运作的理解。更关键的问题是，目前缺少一个定量指标来系统评估 SAE 在视觉任务上的单义性效果。

**核心矛盾**：虽然 SAE 在理论上应能解耦多义神经元为单义表征，但由于没有合适的评估标准，研究者无法系统比较不同 SAE 架构的优劣，也无法验证解耦后的表征是否与人类感知对齐。

**本文目标** （1）如何定量度量 VLM 中 SAE 神经元的单义性？（2）哪些 SAE 设计因素对单义性贡献最大？（3）单义性神经元能否用于控制多模态大模型的输出？

**切入角度**：图像不同于文本——单张图像无需上下文即可直接激活神经元。因此可以通过衡量高激活图像之间的语义相似性来评估神经元的单义性。

**核心 idea**：提出基于激活加权图像相似度的 MonoSemanticity score，系统评估 VLM 上 SAE 的单义性，并利用单义神经元实现对多模态 LLM 的无监督概念级引导。

## 方法详解

### 整体框架

整体流程分三阶段：（1）在 CLIP 视觉编码器的中间层训练 SAE，将原始激活重构为更高维的稀疏表征；（2）用 MS 分数评估每个 SAE 神经元的单义性质量；（3）将训练好的 SAE 注入 LLaVA 的视觉编码器中，通过操控特定神经元激活值引导模型输出。

### 关键设计

1. **MonoSemanticity Score（MS）**:

    - 功能：定量度量单个 SAE 神经元是否只关注一个语义概念
    - 核心思路：对一组多样化图像集，用 DINOv2 提取图像间余弦相似度矩阵 $S$，然后收集每个神经元对所有图像的归一化激活值，计算激活加权的 pairwise 相似度：$\text{MS}^k = \frac{\sum_{n<m} r_{nm}^k s_{nm}}{\sum_{n<m} r_{nm}^k}$，其中 $r_{nm}^k = \tilde{a}_n^k \tilde{a}_m^k$ 是两张图像对神经元 $k$ 的联合激活强度。高激活图像越相似则 MS 越高
    - 设计动机：避免固定选择 Top-K 激活图像的问题（不同神经元专注程度差异大），通过连续加权覆盖不同激活模式

2. **Matryoshka SAE 架构**:

    - 功能：引入嵌套式多级重构目标，提升单义性
    - 核心思路：将 SAE 的 $\omega$ 个神经元分为嵌套组（如 $\{0.0625\omega, 0.1875\omega, 0.4375\omega, \omega\}$），每级只用前 $m$ 个神经元做重构，训练目标为所有级别重构损失之和。与 BatchTopK 激活函数结合，控制最多 $K$ 个神经元非零
    - 设计动机：Matryoshka 目标强制前面的神经元捕获更重要概念，形成从粗到细的层级结构，相同扩展因子下比普通 BatchTopK 有更高单义性

3. **视觉 SAE 引导多模态 LLM**:

    - 功能：通过操控 SAE 特定神经元的激活值来控制 LLaVA 生成内容
    - 核心思路：在 CLIP 视觉编码器第 22 层后注入 SAE，对所有 token 嵌入先编码为稀疏激活，然后将目标神经元 $k$ 的激活值强制设为 $\alpha$（正值插入概念、负值抑制概念），再用解码器还原为修改后的嵌入。全程不修改语言模型参数
    - 设计动机：利用 SAE 的单义性分解在视觉编码器层面实现概念级控制，保持引导的模块化

### 损失函数 / 训练策略

SAE 训练使用重构目标加稀疏正则：$\mathcal{L}(\mathbf{v}) = \mathcal{R}(\mathbf{v}) + \lambda \mathcal{S}(\mathbf{v})$。BatchTopK 变体通过 Top-K 激活函数直接控制稀疏度（$K=20$），Matryoshka 变体额外添加多级重构目标。在 ImageNet 上提取的 CLIP 激活向量上训练 $10^5$ 步，batch size 4096，Adam 优化器。

## 实验关键数据

### 主实验

| SAE 类型 | 扩展因子 | 最高 MS（CLIP 22层） | 无 SAE 的 MS | 提升 |
|----------|---------|---------------------|-------------|------|
| BatchTopK | ×1 | 0.66 | 0.01 | +0.65 |
| BatchTopK | ×16 | 0.92 | 0.01 | +0.91 |
| Matryoshka | ×1 | 0.83 | 0.01 | +0.82 |
| Matryoshka | ×8 | 0.94 | 0.01 | +0.93 |

| 引导方式 | 概念插入（双标准通过率） | 概念抑制（双标准通过率） |
|---------|----------------------|----------------------|
| SAE 引导（本文） | 42.4% | 52.5% |
| DiffMean 基线 | 35.8% | 33.3% |

### 消融实验

| 配置 | MS 趋势 | 说明 |
|------|---------|------|
| 扩展因子 ×1 vs 原始层 | 90% 神经元 MS 更高 | 稀疏分解本身提升单义性 |
| K=1（最大稀疏） | MS 最高但 R²=31.3% | 单义性与重构质量 trade-off |
| K=20 | MS 较高且 R²=66.8% | 平衡点 |
| K=50 | MS 较低但 R²=74.9% | 稀疏度不足 |

### 关键发现

- 即使 SAE 宽度与原始层相同（×1），约 90% 的 SAE 神经元也比原始神经元有更高 MS，证明稀疏字典学习本身就能提升概念分离
- Matryoshka SAE 在相同扩展因子下比 BatchTopK 取得更高 MS，但 R² 低 2-3 个点
- 引导实验中 SAE 方向在保持提示遵守方面远优于 DiffMean（85.8% vs 66.2%），说明单义神经元提供的引导方向更精确
- 大规模用户研究（1000 问题，71 用户）显示 MS 与人类判断一致率达 82.8%，当 MS 差异 >0.8 时一致率达 100%

## 亮点与洞察

- **将 NLP 中的 SAE 可解释性范式迁移到视觉领域**的工作很系统，从指标设计到人类验证到引导应用形成完整链路。MS 分数巧妙利用了图像不需要上下文这个特性，比 LLM 中的单义性评估更自然
- **概念引导的模块化设计**非常实用：只需在视觉编码器后插入 SAE，不修改语言模型，就能实现概念插入和抑制。为多模态模型安全控制提供了新路径
- 发现 ×1 扩展因子已显著提升单义性，说明稀疏重构目标本身的正则化效果比增加宽度更重要

## 局限与展望

- 高 MS 神经元并不总能产生精确引导效果，"金毛"神经元可能触发任何狗相关输出，源于 MLLM 缺乏与视觉编码器的细粒度对齐
- MS 分数仅适用于视觉表征，尚未扩展到文本模态
- 部分充当特征检测器的 SAE 神经元不产生任何明显引导效果，需要更好的神经元选择策略
- 用户研究规模（71 用户，1000 问题）相对有限

## 相关工作与启发

- **vs Anthropic Towards Monosemanticity**: Anthropic 在 LLM 上做 SAE 可解释性，本文迁移到 VLM。视觉模态评估更直观，且额外展示了跨模型引导应用
- **vs DiffMean 激活引导**: DiffMean 通过概念存在/不存在的均值差来引导。本文 SAE 方法在保持提示遵守方面远优（85.8% vs 66.2%），因为 SAE 方向更"干净"
- **vs CLIP-Dissect**: CLIP-Dissect 用文本描述解释神经元（外部标注），本文 SAE 是内在解耦方法，通过重构任务自动发现单义特征

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 SAE 可解释性从 LLM 系统扩展到 VLM 并提出配套评估指标，方向有价值但核心技术非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 跨多个视觉编码器、多层、多扩展因子的系统实验，加上大规模用户研究和引导应用
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，从指标定义到实验验证到应用展示层层递进
- 价值: ⭐⭐⭐⭐ 为 VLM 可解释性和安全控制提供了实用工具和基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] In-Context Compositional Learning via Sparse Coding Transformer](in-context_compositional_learning_via_sparse_coding_transformer.md)
- [\[CVPR 2026\] Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models](../../CVPR2026/multimodal_vlm/sparse-lavida_sparse_multimodal_discrete_diffusion_language_models.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)

</div>

<!-- RELATED:END -->
