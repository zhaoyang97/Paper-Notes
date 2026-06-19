---
title: >-
  [论文解读] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment
description: >-
  [AAAI 2026 Oral][图像生成][自回归图像生成] 提出 ARRA（Autoregressive Representation Alignment）训练框架，通过混合令牌 \<HYBNEXT\> 在训练时将外部视觉基础模型的全局表征蒸馏到自回归 LLM 的隐状态中，无需修改架构即可显著提升 LLM 的文本到图像生成质量。
tags:
  - "AAAI 2026 Oral"
  - "图像生成"
  - "自回归图像生成"
  - "表征对齐"
  - "大语言模型"
  - "文本到图像"
  - "全局一致性"
---

# Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment

**会议**: AAAI 2026 Oral  
**arXiv**: [2503.07334](https://arxiv.org/abs/2503.07334)  
**代码**: [https://github.com/HKU-HealthAI/ARRA](https://github.com/HKU-HealthAI/ARRA)  
**领域**: 医学影像 / 图像生成  
**关键词**: 自回归图像生成, 表征对齐, 大语言模型, 文本到图像, 全局一致性

## 一句话总结

提出 ARRA（Autoregressive Representation Alignment）训练框架，通过混合令牌 \<HYBNEXT\> 在训练时将外部视觉基础模型的全局表征蒸馏到自回归 LLM 的隐状态中，无需修改架构即可显著提升 LLM 的文本到图像生成质量。

## 研究背景与动机

1. **领域现状**：大语言模型（LLM）基于自回归（AR）"next-token prediction"范式在文本任务上取得了巨大成功。研究者尝试将该范式扩展到文本到图像（T2I）生成，如 DALL·E、LlamaGen 等。

2. **现有痛点**：next-token prediction 在语言任务中局部依赖自然与序列结构对齐，但在图像生成中，仅优化局部 token 级预测迫使模型关注孤立的 token 特征，忽略了空间结构化视觉内容所需的全局一致性。这导致生成图像出现碎片化部分（如 X 光中不对齐的肋骨）和语义不匹配。

3. **核心矛盾**：现有解决方案（如 Transfusion、Show-O）通过修改架构（双向注意力、嫁接扩散模块）注入全局约束，但偏离了标准 LLM 框架，限制了与预训练 LLM 的兼容性，也丧失了已有的 scaling laws 优势。

4. **本文目标**：能否在不改变 LLM 原始架构和推理机制的前提下，释放其在 T2I 生成中的全部潜力？

5. **切入角度**：全局一致性不需要架构复杂性，而是可以通过重新定义训练目标来实现。

6. **核心 idea**：设计混合令牌 \<HYBNEXT\>，让每个 token 在训练时同时受局部自回归损失和全局视觉对齐损失约束，将外部基础模型的语义信息蒸馏到自回归序列中。

## 方法详解

### 整体框架

ARRA 在标准自回归训练流程上增加一个全局视觉对齐（GVA）损失分支：(1) 文本和图像被分词为 token 序列，输入 transformer 做 next-token 预测；(2) 同时，外部预训练视觉编码器（如 BioMedCLIP）提取目标图像的全局视觉表征；(3) 自回归模型中每个 \<HYBNEXT\> token 的隐状态通过投影层与全局表征对齐。该对齐仅在训练时进行，推理时完全移除。

### 关键设计

1. **混合令牌 \<HYBNEXT\>**：
    - 功能：充当局部与全局学习之间的双向锚点
    - 核心思路：将自回归序列中每个待预测的 next token 定义为 \<HYBNEXT\>，它同时受两重约束——局部通过标准 codebook 匹配进行 next-token 预测，全局通过其隐状态与外部模型的压缩视觉特征对齐。这样每个 token 都被外部全局表征有效约束
    - 设计动机：相比在序列起始放置固定 \<REP\> 令牌，\<HYBNEXT\> 可遍历训练中的每个 token，避免长序列中注意力衰减（"attention sink"效应）导致约束失效

2. **全局视觉表征对齐（GVA Loss）**：
    - 功能：将外部基础模型的丰富语义（空间关系、目标一致性等）蒸馏到自回归模型中
    - 核心思路：通过预训练编码器 $\mathcal{E}_F$ 提取图像全局表征 $f_{GF} = \text{agg}(\mathcal{E}_F(I)) \in \mathbb{R}^{1\times D}$，用 2 层 MLP 投影层将 \<HYBNEXT\> 的隐状态 $f_L^i$ 映射为 $f_A$，最小化余弦相似度损失 $\mathcal{L}_{GVA} = \text{sim}(f_A, f_{GF})$
    - 设计动机：利用已经充分训练的视觉基础模型的知识，"免费"提供全局结构先验

3. **灵活的三种变体**：
    - 功能：适应不同场景的即插即用框架
    - 核心思路：ARRA-Base（从零训练）、ARRA（从纯文本 LLM 扩展为 T2I）、ARRA-Adapt（将通用生成模型适配到特定领域如医学影像）
    - 设计动机：展示 ARRA 作为训练框架的通用性，不依赖特定 LLM 架构

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L}_{ARRA}(\theta, \phi) = \mathcal{L}_{AR}(\theta) + \lambda \mathcal{L}_{GVA}(\theta, \phi)$$

- $\mathcal{L}_{AR}$：标准自回归交叉熵损失
- $\mathcal{L}_{GVA}$：全局视觉对齐余弦相似度损失
- $\lambda=1$，平衡两个目标

关键设计决策：
- 聚合策略：[CLS] token 优于 average pooling（全局信息更紧凑）
- 编码器选择：跨模态编码器（BioMedCLIP）适合缺乏图像生成能力的 LLM；领域特定编码器适合已有生成能力的 LLM 进行域适应

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | Baseline | 提升 |
|--------|------|------|----------|------|
| ImageNet (111M) | FID ↓ | 5.08 | 5.47 (LlamaGen) | -7.1% |
| ImageNet (343M) | FID ↓ | 3.61 | 4.33 (LlamaGen) | -16.6% |
| LAION-COCO (3.1B) | FID ↓ | 10.45 | 11.88 (LlamaGen) | -12.0% |
| MIMIC-CXR | FID ↓ | 5.30 | 7.11 (Chameleon) | -25.5% |
| MIMIC-CXR (Adapt) | FID ↓ | 4.15 | 5.10 (直接微调) | -18.6% |
| DeepEyeNet | FID ↓ | 35.01 | 38.37 (Chameleon) | -8.8% |

### 消融实验

| 配置 | FID (MIMIC-CXR) | 说明 |
|------|---------|------|
| w/o alignment | 5.10 | 无对齐基线（ARRA-Adapt） |
| \<REP\> 固定令牌 | 4.85 | 序列起始位置对齐 |
| \<HYBNEXT\> 混合令牌 | 4.15 | 逐 token 对齐，最优 |
| [CLS] 聚合 | 5.30 | 全局特征聚合最优（ARRA） |
| Average pooling | 6.56 | 局部特征混杂 |
| BioMedCLIP 编码器 | 4.15 | 领域特定跨模态，最优 |
| CLIP-L 通用编码器 | 4.63 | 通用跨模态 |
| Med-SAM 视觉编码器 | 4.08 | 结构先验强但语义弱 |

### 关键发现

- ARRA 在模型规模从 343M 到 3.1B 扩展时性能持续提升（FID: 11.67→10.45），保持了良好的 scaling 特性
- \<HYBNEXT\> 在所有场景中一致优于固定位置对齐 \<REP\>，证实了逐 token 全局约束的必要性
- 对于缺乏图像生成能力的 LLM，跨模态编码器（BioMedCLIP/CLIP）至关重要；对于已有生成能力的 LLM，领域特定编码器更有效
- 在不同分辨率（256×256、512×512）下 ARRA 都能稳定提升性能

## 亮点与洞察

- **极其简洁的方案**：不修改任何架构，不影响推理效率，仅通过重新定义训练目标即可解决跨模态全局一致性问题
- **三个核心 Takeaway**：(1) 逐步对齐优于固定位置；(2) [CLS] token 是最佳全局聚合策略；(3) 编码器选择应与 LLM 能力匹配
- **即插即用**：同一框架支持从零训练、文本模型扩展、领域适应三个场景，展示了极强的通用性
- 医学影像实验特别有说服力——胸部 X 光生成中肋骨对齐等细节的改善直观可见

## 局限与展望

- 训练时需要额外的外部编码器前向传播，增加了训练成本
- 当前仅使用全局 [CLS] 表征对齐，未探索 patch 级别更细粒度的对齐策略（与 AR 模型的兼容性挑战）
- 实验主要集中在医学影像数据集上，通用自然图像上的验证规模相对有限
- 对齐权重 $\lambda$ 未进行广泛的超参数搜索

## 相关工作与启发

- **vs Transfusion/Show-O**：需要修改架构（双向注意力/mask token 建模），与标准 LLM 不兼容；ARRA 保持架构不变
- **vs REPA (扩散模型对齐)**：REPA 使用 patch-wise 对齐，与 AR 模型不兼容（AR 训练时不同时输出所有 patch token）；ARRA 通过 \<HYBNEXT\> 桥接了对齐机制与 AR 架构
- **vs LlamaGen (纯自回归)**：仅有局部约束，缺乏全局结构一致性；ARRA 在不改架构前提下注入全局知识

## 评分

- 新颖性: ⭐⭐⭐⭐ 训练目标重设计的思路新颖，\<HYBNEXT\> 设计巧妙；但表征蒸馏本身不算全新概念
- 实验充分度: ⭐⭐⭐⭐⭐ 自然图像+医学影像、多尺度模型、三种变体、系统的组件分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机论证有力，实验分析总结出实用 Takeaway，结构清晰
- 价值: ⭐⭐⭐⭐ 即插即用的通用框架对 AR T2I 社区有实际价值，医学影像方面应用前景好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Curriculum Group Policy Optimization: Adaptive Sampling for Unleashing the Potential of Text-to-Image Generation](../../CVPR2026/image_generation/curriculum_group_policy_optimization_adaptive_sampling_for_unleashing_the_potent.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](../../ECCV2024/image_generation/textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ACL 2026\] Multimodal Large Language Models for Multi-Subject In-Context Image Generation](../../ACL2026/image_generation/multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[CVPR 2026\] VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation](../../CVPR2026/image_generation/va-p_variational_policy_alignment_for_pixel-aware_autoregressive_generation.md)

</div>

<!-- RELATED:END -->
