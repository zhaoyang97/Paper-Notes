---
title: >-
  [论文解读] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning
description: >-
  [ICCV 2025][多模态VLM][Dual-LoRA] 提出 Dual-LoRA 和 Visual Cue Enhancement (VCE) 两个模块，通过"从整体到局部"的范式解决高效视觉指令微调中的数据冲突问题，以仅 1.16× 推理时间开销超越 LoRA-MoE 方法。 高效视觉指令微调 (EVIT) 通过 L…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "Dual-LoRA"
  - "LoRA-MoE"
  - "视觉线索增强"
  - "数据冲突"
  - "高效指令微调"
---

# From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning

**会议**: ICCV 2025  
**arXiv**: [2411.12787](https://arxiv.org/abs/2411.12787)  
**代码**: [https://github.com/pengkun-jiao/Dual-LoRA](https://github.com/pengkun-jiao/Dual-LoRA)  
**领域**: 多模态VLM  
**关键词**: Dual-LoRA, LoRA-MoE, 视觉线索增强, 数据冲突, 高效指令微调

## 一句话总结

提出 Dual-LoRA 和 Visual Cue Enhancement (VCE) 两个模块，通过"从整体到局部"的范式解决高效视觉指令微调中的数据冲突问题，以仅 1.16× 推理时间开销超越 LoRA-MoE 方法。

## 研究背景与动机

高效视觉指令微调 (EVIT) 通过 LoRA 等适配器以较小计算开销将多模态大模型 (MLLM) 适配到下游任务。然而，随着任务多样性和复杂性增加，LoRA 微调面临严重的**数据冲突**问题：例如在食物相关多任务训练中，食材识别和食谱生成之间可能产生知识不一致的回答。

现有解决方案是 LoRA-MoE（将 LoRA 嵌入混合专家框架），通过专家激活实现局部化适应来缓解数据冲突。但 LoRA-MoE 存在两个问题：(1) 需要复杂设计来平衡激活策略、可训练参数和任务复杂性；(2) 多专家激活显著增加推理时间（4 专家 LoRA-MoE 需要 1.59× 基线延迟）。

作者的核心洞察来自人类认知：人类先获取整体知识（如烹饪），然后将特定部分（如食材识别）应用于特定任务——即"从整体到局部"的过程。

## 方法详解

### 整体框架

提出两个核心组件：(1) **Visual Cue Enhancement (VCE)** 模块增强视觉特征投影的局部细节；(2) **Dual Low-Rank Adaptation (Dual-LoRA)** 通过双子空间实现从整体知识到局部任务适配的转换。训练分两阶段：Stage 1 预训练 VCE 和视觉投影器，Stage 2 微调 VCE + 投影器 + Dual-LoRA。

### 关键设计

1. **Visual Cue Enhancement (VCE)**：典型 MLLM（如 LLaVA）仅使用 ViT 倒数第二层的高级特征图，容易忽略局部视觉细节。VCE 通过可变形注意力 (Deformable Attention) 从多个中间层特征图提取局部信息，具体选择 CLIP ViT-L 的第 2、8、14、20 层，以第 2 层作为锚点特征。对每个锚点 patch $p_q$，在每层通过可变形注意力聚合 $K$ 个邻域参考 patch 的信息，再通过多层级拼接和线性投影融合，最后与高级特征残差融合后送入视觉投影器。VCE 仅 5.53 MB。

2. **Dual-LoRA (技能空间 + 任务空间)**：基于理论分析（Proposition 1 & Corollary 1-2），证明单个 rank-$r$ 的 LoRA 理论上至少与 $K$ 个 LoRA 专家同样具有表达力。但 LoRA-MoE 实际表现更好是因为其"局部响应"能力。为此引入双空间：

    - **技能空间 (Skill Space)** $S$：低秩矩阵，用于稳定学习跨任务的整体知识
    - **任务空间 (Task Space)** $T$：rank 修正矩阵，通过非线性激活 $\sigma$（ReLU）动态调制技能空间

   输出公式为：$z = Wx + \frac{r}{\alpha} B(\text{Norm}(Sx) \odot \sigma(Tx))$

   通过 LayerNorm 平滑技能空间分布，ReLU 激活产生稀疏性实现局部响应。

3. **理论基础**：作者证明了 (a) rank-$K$ 的单 LoRA 包含 $K$ 个 rank-1 LoRA 的表示空间（Proposition 1）；(b) 通过元素乘配合非线性激活，可将单 LoRA 分解为任意 LoRA 组合（Corollary 2），这为 Dual-LoRA 的设计提供了理论支撑。

### 损失函数 / 训练策略

- 标准交叉熵损失用于指令微调
- Dual-LoRA 仅注入 LLM 的 query 和 value 投影层以提高效率
- $\alpha$ 设为 2× rank，LoRA dropout 0.05
- 基础模型：LLaVA-1.5-7B，CLIP ViT-L/14 视觉编码器

## 实验关键数据

### 主实验

| 方法 | UniFood IoU↑ | UniFood F1↑ | Recipe BLEU↑ | Recipe Rouge-L↑ | ScienceQA Acc↑ | Flickr30k BLEU↑ |
|------|------------|------------|-------------|----------------|---------------|----------------|
| Vanilla LoRA | 23.2 | 34.1 | 12.4 | 40.1 | 70.01 | 27.89 |
| LoRA-MoE (top-2) | 22.9 | 33.8 | 12.7 | 40.2 | 76.3 | 28.15 |
| LoRA-MoE (softmax) | 22.7 | 33.5 | 12.5 | 40.0 | 77.73 | 28.06 |
| RoDE | 23.6 | 34.6 | 13.8 | 41.4 | 78.39 | 28.17 |
| **Dual-LoRA** | **24.2** | **35.2** | **14.8** | **42.1** | **79.17** | **28.25** |
| **Dual-LoRA+VCE** | **24.5** | **35.5** | 14.7 | **42.2** | **80.01** | **28.71** |

通用 benchmark (LLaVA-1.5-7B 基础)：MMBench 65.6, POPE 87.2, MMVet 32.1（均为最优或次优）

### 消融实验

| VCE | Dual-LoRA | IoU↑ | F1↑ | BLEU↑ | Rouge-L↑ |
|-----|-----------|------|-----|-------|----------|
| ✗ | ✗ (LoRA) | 23.2 | 34.1 | 12.4 | 40.1 |
| ✗ | ✓ | **24.2** | **35.2** | **14.8** | **42.1** |
| ✓ | ✗ | 23.3 | 34.2 | 12.6 | 40.5 |
| ✓ | ✓ | **24.5** | **35.5** | 14.7 | **42.2** |

Dual-LoRA 内部设计消融：

| Skill @Norm | Task @Non-linear | IoU↑ | BLEU↑ |
|-------------|-----------------|------|-------|
| ✗ | ✗ | 21.4 | 13.1 |
| ✓ | ✗ | 22.6 | 14.1 |
| ✗ | ✓ | 21.9 | 13.4 |
| ✓ | ✓ | **24.2** | **14.8** |

### 关键发现

- Dual-LoRA **在所有下游任务上一致优于** vanilla LoRA 和 LoRA-MoE 方法
- 推理时间仅为 vanilla LoRA 的 1.16×（Dual-LoRA+VCE），而 4 专家 LoRA-MoE 为 1.59×
- VCE 模块的热力图可视化表明增强的视觉线索确实聚焦于图像中的关键区域
- 修正后的技能空间比原始技能空间具有更低的信息熵，说明更加任务特定化
- 参数越少时 Dual-LoRA 相对 vanilla LoRA 的优势越明显，证明其更好地解决了数据冲突
- Skill 空间的 LayerNorm 和 Task 空间的非线性激活均不可或缺

## 亮点与洞察

1. **理论驱动设计**：通过严格的数学证明（Proposition 1, Corollary 1-2）揭示单 LoRA 与 MoE 的等价性，进而设计出更高效的替代方案
2. **极致简洁**：Dual-LoRA 结构上远比 LoRA-MoE 简单，无需路由器和专家选择策略
3. **效率优势显著**：相同参数量下比所有 LoRA-MoE 方法更好，且推理时间开销微乎其微
4. **VCE 轻量有效**：仅 5.53 MB 即可从多级特征图聚合局部信息

## 局限与展望

- 实验主要基于 LLaVA-1.5-7B 单一 MLLM，未在更大模型（13B/70B）或更新架构（Qwen-VL-2、InternVL 等）上验证
- VCE 中间层的选择（2, 8, 14, 20）是手动设定的，可探索自适应选择
- Dual-LoRA 的 ReLU 激活下稀疏性对不同任务的影响程度未深入分析
- 未在视频理解等序列任务上验证效果

## 相关工作与启发

- 与 AdaMoLE、LLaVA-MoLE 等同期工作相比，Dual-LoRA 提供了结构更简洁的替代方案
- VCE 模块与 Cambrian-1 的 Spatial Vision Aggregator 理念类似，但更加轻量
- "技能空间+任务空间"的双空间设计可能推广到其他参数高效微调场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 理论驱动的双空间设计新颖，从"整体到局部"的认知启发有深度
- **实验充分度**: ⭐⭐⭐⭐ 多数据集、多 baseline、消融充分，但基础模型单一
- **写作质量**: ⭐⭐⭐⭐ 动机图示清晰，理论推导严谨
- **价值**: ⭐⭐⭐⭐ 对 MLLM 高效微调社区有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](../../ICML2025/multimodal_vlm/parrot_multilingual_visual_instruction_tuning.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](../../CVPR2025/multimodal_vlm/skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)

</div>

<!-- RELATED:END -->
