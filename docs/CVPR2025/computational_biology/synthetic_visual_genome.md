---
title: >-
  [论文解读] Synthetic Visual Genome
description: >-
  [CVPR 2025][计算生物][场景图] 提出**SVG**（Synthetic Visual Genome）数据引擎，通过GPT-4在已有人工标注基础上**补全缺失关系**（Stage 1）和**Robin自蒸馏+GPT-4编辑**（Stage 2/SG-Edit）两阶段管道，生成146K图像、2.6M物体、5.6M关系的密集场景图数据集，训练的**Robin-3B**模型仅用<3M实例即超越300M实例训练的同尺寸模型，在指代表达理解上达到88.9的SOTA。
tags:
  - "CVPR 2025"
  - "计算生物"
  - "场景图"
  - "关系推理"
  - "合成数据"
  - "自蒸馏"
  - "指代表达理解"
---

# Synthetic Visual Genome

**会议**: CVPR 2025  
**arXiv**: [2506.07643](https://arxiv.org/abs/2506.07643)  
**代码**: [https://synthetic-visual-genome.github.io/](https://synthetic-visual-genome.github.io/)  
**领域**: 计算生物  
**关键词**: 场景图, 关系推理, 合成数据, 自蒸馏, 指代表达理解

## 一句话总结

提出**SVG**（Synthetic Visual Genome）数据引擎，通过GPT-4在已有人工标注基础上**补全缺失关系**（Stage 1）和**Robin自蒸馏+GPT-4编辑**（Stage 2/SG-Edit）两阶段管道，生成146K图像、2.6M物体、5.6M关系的密集场景图数据集，训练的**Robin-3B**模型仅用<3M实例即超越300M实例训练的同尺寸模型，在指代表达理解上达到88.9的SOTA。

## 研究背景与动机

视觉关系推理——理解物体间的空间、功能、交互、社会和情感关系——被认为是人类认知的基础能力。然而，多模态语言模型（MLM）在精确表达关系方面仍然面临挑战。

**领域现状**：指令微调已被证明能有效为MLM注入特定推理能力，但关系推理的指令微调受限于缺乏大规模、密集标注的关系数据集。

**现有痛点**：
1. **Visual Genome标注稀疏**——每个物体平均仅1.5个关系标注，大量存在的关系未被标注（如"女人在婴儿前面"这样的显然关系被遗漏）
2. **关系类型单一**——VG主要包含空间关系，缺乏交互、情感、功能和社会关系
3. **人工标注不可扩展**——穷举标注所有物体的所有关系对人类标注者而言极其繁琐且成本高昂
4. **直接用GPT-4生成效果差**——从零开始让GPT-4V生成场景图会产生大量幻觉和定位错误

**核心矛盾**：密集场景图对关系推理至关重要，但人工标注无法扩展，直接AI生成又不可靠。

**切入角度**：不从零生成，而是在已有高质量人工标注的基础上**补全缺失关系**——让GPT-4看到已有的物体标注和关系后推理缺失的关系，大幅减少幻觉。然后通过自蒸馏管道迭代扩展到更多图像。

## 方法详解

### 整体框架

两阶段管道：**Stage 1（密集关系补全）**——从COCO子集的33K种子图像出发，利用多源标注（检测、区域描述、场景图、深度图）和SAM分割，提示GPT-4V对选定物体补全五类关系（空间、交互、情感、功能、社会），生成SVG-Relations数据集。**Stage 2（自蒸馏扩展/SG-Edit）**——先用SVG-Relations训练Robin-3B（Stage 1），然后用它对新图像（ADE20K、PSG、VG，共113K）生成场景图，再由GPT-4o编辑修正（删错加对），生成SVG-SG数据集。最终在完整数据上训练得到Robin-3B。

### 关键设计

1. **基于种子标注的关系补全（Dense Relationship Completion）**
    - **功能**：在保证准确性的前提下，将已有标注的关系密度提升4倍
    - **核心思路**：
     - 精选33K COCO图像，汇集来自COCO、LVIS（检测）、RefCOCO、VG（区域描述）、VG和GQA（场景图）、Depth-Anything（深度图）的多源标注
     - 使用SAM/Semantic-SAM生成分割mask，与标注框IoU>0.5的区域保留为"可靠区域"
     - 提示GPT-4V基于已有标注推断每个突出物体的缺失关系，按五个类别（空间、交互、功能、社会、情感）分别生成
     - 对空间关系用规则过滤，对其他关系用VQA模型过滤低质量标注
    - **设计动机**：GPT-4V从零生成场景图效果很差（大量幻觉），但在已有标注基础上**补全**效果好得多——已知物体位置和部分关系后，推理缺失关系是更可靠的任务

2. **SG-Edit自蒸馏管道（Self-Distillation with GPT-4 Editing）**
    - **功能**：将Stage 1的能力扩展到任意图像，实现大规模场景图数据生成
    - **核心思路**：
     - 用SVG-Relations训练Robin-3B (Stage 1)作为学生模型
     - Robin生成新图像的密集场景图（高效但可能有噪声）
     - GPT-4o作为编辑器对Robin的输出进行精修：删除错误关系（红→绿）、添加遗漏关系、补充物体属性描述
     - 将精修后的数据（SVG-SG，113K图像）用于Stage 2训练
    - **设计动机**：类似Segment Anything的迭代数据改进范式——先训模型→用模型生成→人/AI修正→再训模型。用GPT-4o替代人类修正大幅降低成本

3. **Robin-3B模型架构（像素级mask感知）**
    - **功能**：同时支持区域理解、关系推理和密集场景图生成
    - **核心思路**：三组件架构：
     - **视觉编码器**（ConvNext-Large）：编码全局图像为图像token
     - **像素级mask感知提取器**（ConvNext-Large）：将每个分割mask编码为mask token（最多99个区域）
     - **语言模型**（Qwen2.5-3B，8192 token上下文）：接收图像token、mask token和文本token，支持任意视觉指令和定地任务
    - **设计动机**：与仅用边界框文本坐标引用区域的模型不同，Robin用分割mask+文本双重表示，实现更精细的区域定位

### 训练策略

三阶段渐进训练：
- **Stage 0（对齐，1.28M样本）**：冻结视觉编码器，先训练图像投影器（LLaVA-Pretrain-558K），再解冻mask编码器学mask嵌入，最后微调LM
- **Stage 1（指令微调+场景图，1.73M样本）**：解冻视觉编码器，混合训练视觉指令、定地和场景图数据（含SVG-Relations）
- **Stage 2（蒸馏微调，1.23M样本）**：将SVG-Relations替换为SVG-SG，继续训练

## 实验关键数据

### 关系理解基准（≤4B模型对比）

| 模型 | 训练数据量 | GQA | VSR | MMBench | CRPE Rel | SugarCrepe | What's Up |
|------|-----------|-----|-----|---------|----------|------------|-----------|
| VILA1.5-3B | — | 61.5 | 61.0 | 63.4 | 67.8 | 86.3 | 50.6 |
| Phi-3-Vision-4B | 300M+ | — | 67.8 | 74.2 | 71.6 | 88.7 | 78.7 |
| BLIP-3-3B | 300M+ | — | 72.5 | 76.0 | 72.4 | 89.0 | 78.2 |
| **Robin-3B** | **<3M** | **61.6** | **76.4** | **77.6** | **68.2** | **90.1** | **86.2** |

### 指代表达理解（RefCOCO系列，R@1 IoU>0.5）

| 模型 | 参数量 | RefCOCO Val | Test-A | Test-B | Avg |
|------|--------|------------|--------|--------|-----|
| Ferret-13B | 13B | 89.5 | 92.4 | 84.4 | 85.6 |
| ASM-V2-13B | 13B | 90.6 | 94.2 | 86.2 | 87.4 |
| **Robin-3B** | **3B** | **91.6** | **94.3** | **88.6** | **88.9** |

### SVG数据集统计

| 数据集 | 图像数 | 标注者 | 物体/图 | 关系三元组/图 | 关系/物体 |
|--------|-------|--------|---------|-------------|----------|
| VG | 108K | 人类 | 35.2 | 21.4 | 0.6 |
| GQA | 85K | 人类 | 16.4 | 50.6 | 3.1 |
| SVG-Relations | 33K | GPT-4V | 13.2 | 25.5 | **1.9** |
| SVG-SG | 113K | Robin+GPT-4o | 19.8 | 42.3 | **2.4** |

### Stage 1 vs Stage 2提升

| 基准 | Robin-3B (Stage 1) | Robin-3B (最终) | 提升 |
|------|-------------------|-----------------|------|
| VSR | 73.7 | 76.4 | +2.7 |
| CRPE Rel | 65.9 | 68.2 | +2.3 |
| What's Up | 81.3 | 86.2 | +4.9 |
| RefCOCO Avg | 87.2 | 88.9 | +1.7 |

### 关键发现

1. Robin-3B **仅用<3M实例训练**即超越300M+实例训练的Phi-3-Vision和BLIP-3，在What's Up上领先7.5%（86.2 vs 78.7）
2. 指代表达理解达到**SOTA 88.9**，超越13B参数的ASM-V2（87.4），说明数据质量比模型规模更重要
3. GPT-4编辑蒸馏（Stage 2）在关系理解基准上带来一致提升，尤其在What's Up上+4.9%
4. SVG每个物体的关系密度是VG的4倍（2.4 vs 0.6），覆盖五类关系而非仅空间关系

## 亮点与洞察

1. **"补全"而非"从零生成"**——解决了GPT-4V直接生成场景图幻觉严重的问题。在已有标注基础上推理缺失关系比从零开始可靠得多，这一思路对其他数据增强任务有参考价值
2. **类SAM的迭代数据引擎**——训模型→用模型生成→AI修正→再训模型的闭环极具可扩展性，且用GPT-4o替代人类修正大幅降低了成本
3. **3B模型超越13B模型**——在关系理解和指代表达上，数据质量（密集多样的关系标注）比模型大小更关键
4. **五类关系的系统化**——将关系分为空间/交互/功能/社会/情感五类，比传统场景图仅关注空间关系更全面，更贴近人类的场景理解方式

## 局限性

1. Stage 1种子图像仅来自COCO子集（33K），可能存在域偏差
2. GPT-4V/4o参与数据生成的成本仍然不低，虽比人工便宜但不可忽略
3. SG-Edit管道中GPT-4o的编辑质量未经系统验证（可能引入新的偏差）
4. 当前仅扩展到113K图像（SVG-SG），未深入分析进一步扩展的效果

## 相关工作与启发

- **场景图数据集**: Visual Genome开创性但稀疏，GQA密度更高但仍有限。SVG通过AI辅助实现了密度和多样性的突破
- **数据引擎范式**: 类似Segment Anything的"模型→数据→模型"闭环，SVG将其应用到场景图领域，证明该范式的普适性
- **指令微调**: 高质量的关系数据是MLM关系推理能力的关键——Robin-3B的成功说明"给正确的数据比给更多数据更重要"
- **启发**: 在大模型时代，AI辅助数据标注（补全而非从零生成+AI编辑修正）正在成为构建高质量训练数据的主流范式

## 评分
- 新颖性: ⭐⭐⭐⭐ "补全而非从零生成"的数据引擎思路新颖，SG-Edit自蒸馏管道设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖关系理解、指代表达、区域识别、场景图生成四大任务，消融全面
- 写作质量: ⭐⭐⭐⭐ 系统清晰，图示直观，两阶段管道描述到位
- 价值: ⭐⭐⭐⭐ 数据引擎范式可推广，3B超13B证明数据质量比规模更重要

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[NeurIPS 2025\] Multimodal 3D Genome Pre-training](../../NeurIPS2025/computational_biology/multimodal_3d_genome_pre-training.md)
- [\[NeurIPS 2025\] MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs](../../NeurIPS2025/computational_biology/meicoder_decoding_visual_stimuli_from_neural_activity_by_leveraging_most_excitin.md)
- [\[ICLR 2026\] Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](../../ICLR2026/computational_biology/contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)
- [\[AAAI 2026\] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](../../AAAI2026/computational_biology/mergedna_context-aware_genome_modeling_with_dynamic_tokenization_through_token_m.md)

</div>

<!-- RELATED:END -->
