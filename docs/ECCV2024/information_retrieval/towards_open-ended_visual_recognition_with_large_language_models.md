---
title: >-
  [论文解读] Towards Open-Ended Visual Recognition with Large Language Model
description: >-
  [ECCV 2024][信息检索/RAG][开放式视觉识别] 提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。
tags:
  - "ECCV 2024"
  - "信息检索/RAG"
  - "开放式视觉识别"
  - "大语言模型"
  - "mask分类"
  - "生成式识别"
  - "多数据集训练"
---

# Towards Open-Ended Visual Recognition with Large Language Model

**会议**: ECCV 2024  
**arXiv**: [2311.08400](https://arxiv.org/abs/2311.08400)  
**代码**: [GitHub](https://github.com/bytedance/OmniScient-Model)  
**领域**: LLM/NLP  
**关键词**: 开放式视觉识别, 大语言模型, mask分类, 生成式识别, 多数据集训练

## 一句话总结

提出 OmniScient Model (OSM)——一个基于冻结 CLIP-ViT + 可训练 MaskQ-Former + 冻结 LLM (Vicuna-7B) 的生成式 mask 分类器，将视觉识别从"从预定义词表中选择类别"转变为"直接生成类别名称"，消除了训练和测试时对预定义词表的依赖，在 COCO 全景分割上超越 DaTaSeg +4.3 PQ。

## 研究背景与动机

开放世界中的目标定位与识别是计算机视觉的长期挑战。当前主流方法将问题分解为两个子任务：

**类别无关的 mask/box 提议**：SAM 等模型已展现强大的零样本泛化能力

**开放词表分类**：依赖 CLIP 等 VLM 的预提取文本嵌入进行分类

然而，现有开放词表识别方法存在两个根本局限：

**局限一：测试时必须提供类别名称**。用户需要预定义所有可能的语义类别，识别性能严重依赖这个预定义词表的质量。在真实场景中，预先枚举所有可能的类别名几乎不可能。

**局限二：多数据集训练时的标签冲突**。不同数据集的标签定义可能存在冲突（如 COCO 中的 "tv" 在语义上等同于 "monitor"），现有方法要么需要为每个数据集训练独立的解码器/分类器，要么需要人工合并标签空间，大大增加了工程复杂度。

作者提出了一个范式转换：**从判别式识别（从词表中选择）到生成式识别（直接生成类别名称）**，即开放式视觉识别（open-ended visual recognition）。

## 方法详解

### 整体框架

OSM 采用模块化架构，由三个核心组件构成：

1. **冻结 CLIP-ViT**：提取高分辨率视觉特征（滑动窗口方式）
2. **可训练 MaskQ-Former**：mask感知的视觉特征重采样模块
3. **冻结 LLM (Vicuna-7B)**：以生成方式预测类别名称

整体流程：输入图像 → CLIP-ViT 提取特征 → 分割模型（SAM/kMaX-DeepLab）生成 mask → MaskQ-Former 提取 mask 区域特征 → LLM 生成类别名称。

### 问题形式化：从分类到生成

传统分类：给定图像 $\mathbf{I}$ 和 $M$ 个分割 mask $\mathbf{M}$，预测每个 mask 的类别 $c_i \in C$，其中 $C$ 是预定义类别集。

**开放式识别**：假设词表 $C$ 在训练和测试时均未知。将识别重新定义为文本生成任务，最大化类名的条件似然：

$$p(c_i) = \prod_{j=0}^{N} p(c_{i,j} | c_{i,0}, \cdots, c_{i,j-1})$$

其中 $c_{i,j}$ 是类名的第 $j$ 个 token。模型直接生成类别名称，而非从候选集中选择。

### 高分辨率特征提取（滑动窗口策略）

**问题**：CLIP-ViT 预训练分辨率为 224×224，直接用于高分辨率输入时性能下降严重（frozen ViT 对分辨率变化的泛化能力差）。

**解决方案**：在输入层面采用滑动窗口策略：
- 将高分辨率图像（如 896×896）切分为多个与预训练尺寸匹配的窗口
- 每个窗口独立经过冻结 ViT 提取特征
- 添加全局位置编码补偿窗口间的位置信息缺失

实验表明，从 224→448 分辨率提升带来 **+12.8%** 的分类准确率提升，最优分辨率为 1120×1120。直接全局适配高分辨率（不用滑动窗口）性能下降 **-7.2%**。

### MaskQ-Former：mask感知的特征重采样

现有 Q-Former / Perceiver Resampler 使用全局注意力聚合图像特征，未考虑分割 mask 信息。OSM 提出 MaskQ-Former，包含两组可学习查询：

**Mask Queries（掩码查询）**：
- 通过 masked cross-attention 仅关注 mask 区域的像素特征
- 聚焦目标物体本身的视觉信息

**Context Queries（上下文查询）**：
- 关注 mask 的扩展区域（如 bounding box 区域）
- 提供目标物体周围的上下文信息（上下文对识别至关重要）

两组查询通过 self-attention 层进行信息交互，参数共享（仅查询初始化不同），几乎不增加额外开销。最终仅保留 Mask Queries 作为 LLM 的输入。

**上下文区域的消融**：bounding box 扩展 0.5× 效果最佳（相比全局上下文 **+2.6%**），过紧过松都不理想。

### Mode Query：词表特定 vs 词表无关的双模式

为兼顾开放式识别能力和与特定数据集词表的对齐能力，OSM 引入 **Mode Query** 机制：

- **Vocabulary-Specific Query**（词表特定查询）：每个训练数据集对应一个专属可学习查询。训练时激活对应数据集的 query，帮助模型"记忆"该数据集的标签空间
- **Vocabulary-Agnostic Query**（词表无关查询）：所有数据集共享的通用查询，训练时在任何数据集上都可激活，保持开放式识别能力

训练时每个 batch 中一半样本激活 vocab-specific query，一半激活 vocab-agnostic query。测试时可灵活选择：
- 需要与特定词表对齐 → 激活 vocab-specific query
- 需要开放式预测 → 激活 vocab-agnostic query

消融实验证明：没有 Mode Query 时，模型泛化能力更强但对特定数据集的对齐度下降；加入 Mode Query 后两方面均改善。

### 训练策略

**数据集**：联合6个公开分割数据集训练：
- COCO 全景分割、ADE20K 全景分割、Cityscapes 全景分割
- LVIS 实例分割、ADE-847 语义分割、PC-459 语义分割

**训练方式**：
- 采用指令微调（Instruction Tuning）策略
- 每次迭代随机选一张图和一个GT mask，从19个指令模板中随机选一个，插入真实类名
- 标准 next-token prediction 损失
- 初始化自 InstructBLIP 预训练权重（EVA-ViT-g/224 + Vicuna-7B）
- 32个 mask queries + 32个 context queries + 1个 mode query
- AdamW 优化器，学习率 4×10⁻⁵，权重衰减 0.05，cosine decay
- 各数据集 batch size 不同（COCO:32, LVIS:64, ADE20K:16 等）
- 共处理 6M 个 mask

**推理**：默认使用模板 "What is in the segmentation mask?"，贪心解码。

## 实验关键数据

### 主实验：GT Mask 分类

OSM 在6个数据集上评估 mask 分类准确率（Acc）和词表外预测比例（NIV）：

| 设置 | COCO Acc | LVIS Acc | ADE20K Acc | Cityscapes Acc | A847 Acc | PC459 Acc | 平均 Acc |
|------|----------|----------|------------|---------------|----------|----------|---------|
| 单数据集训练 | 85.5 | 68.3 | 82.3 | 79.4 | 76.9 | 80.9 | - |
| Learnable Embed | - | - | - | - | - | - | 78.9 |
| Text Embed | - | - | - | - | - | - | < 78.7 |
| OSM vocab-specific | - | - | - | - | - | - | 78.7 |
| OSM vocab-agnostic | - | - | - | - | - | - | 略低 |
| OSM† vocab-specific | - | - | - | - | - | - | 显著提升 |

关键发现：生成式模型（OSM）在判别任务上的性能与判别式模型（Learnable Embed）**持平**（78.7% vs 78.9%），且 NIV 极低，说明生成模型能有效约束在训练词表内。

### 主实验：与 Off-the-shelf 分割器结合

使用 kMaX-DeepLab 作为 mask 提议模型，与其他多数据集通用分割方法对比：

| 方法 | backbone | COCO PQ | ADE20K PQ | ADE20K mIoU | Cityscapes PQ | Cityscapes mIoU |
|------|----------|---------|-----------|-------------|---------------|-----------------|
| Mask2Former (专家模型) | ResNet50 | 51.9 | 39.7 | 46.1 | 62.1 | 77.5 |
| LMSeg | ResNet50 | 38.6 | 35.4 | 45.2 | 54.8 | 80.9 |
| DaTaSeg | ResNet50 | 49.0 | 29.8 | 48.1 | - | - |
| DaTaSeg | ViTDet-L | 53.5 | 33.4 | 54.0 | - | - |
| **OSM** | ResNet50 | **53.3** | **43.8** | **50.0** | 59.5 | 77.0 |
| **OSM** | ConvNeXt-L | **56.1** | **49.7** | **55.2** | **64.7** | 80.2 |

- OSM (ResNet50) 超越 LMSeg **+14.7 PQ（COCO）、+8.4 PQ（ADE20K）、+4.7 PQ（Cityscapes）**
- OSM 超越 DaTaSeg **+4.3 PQ（COCO, ResNet50）、+2.6 PQ（COCO, Large）**
- OSM 与专家模型 Mask2Former 性能可比

### 消融实验

**输入分辨率消融**：

| 分辨率 | 平均 Acc |
|--------|---------|
| 224×224 | 基线 |
| 448×448 | +12.8% |
| 896×896 | 继续提升 |
| 1120×1120 | 最优 |
| 更高 | 性能下降 |

**滑动窗口 vs 全局**：滑动窗口比直接全局处理高分辨率输入好 **+7.2%**。

**上下文区域扩展**：

| 上下文范围 | 相比全局的变化 |
|-----------|-------------|
| Global | 基线 |
| 0.0× (紧bbox) | +0.8% |
| 0.5× (扩展bbox) | +2.6% |

**开放词表评估**（仅用 COCO+LVIS 训练，零样本评估 ADE20K）：

| 方法 | PQ | AP | mIoU |
|------|----|----|------|
| MaskCLIP | 15.1 | 6.0 | 23.7 |
| FC-CLIP | 17.8 | 11.1 | 20.8 |
| ODISE | 19.5 | 10.8 | 23.8 |
| **OSM** | **21.4** | **12.4** | **26.9** |

### 关键发现

1. **生成式 ≈ 判别式**：OSM 在判别任务上与可学习嵌入分类器性能持平，打破了"生成式模型不适合判别任务"的偏见
2. **准确度-泛化权衡**：随训练 mask 数量增加（1M→9M），Acc 提升但 NIV 下降（泛化能力减弱），两者存在固有权衡
3. **涌现式部件识别**：OSM 从未在部件分割数据上训练，却能涌现出预测 "tail"、"ear" 等部件的能力（来自 LLM 的世界知识）
4. **对比 GPT-4V**：OSM 在 mask 分类上比 GPT-4V 更准确，但预测更保守（如预测 "person" 而非 "man in armor"）

## 亮点与洞察

1. **范式转换**：从判别式（从词表选择）到生成式（直接生成类名），彻底消除对预定义词表的依赖，是视觉识别范式的重要演进
2. **多数据集训练无需人工干预**：Mode Query 机制巧妙解决了跨数据集标签冲突问题，无需人工标签合并
3. **滑动窗口提取高分辨率特征**：看似简单的策略带来 12.8% 的巨大提升，证明冻结 ViT 的高分辨率适配不需要复杂设计
4. **MaskQ-Former 的 mask-context 双查询设计**：平衡了目标区域精确特征和上下文信息，参数共享设计高效优雅
5. **NIV 分析**：通过可视化 NIV 案例发现 OSM 的预测（如 "monitor"）常常比GT标注（如 COCO 中统一标注为 "tv"）更准确，揭示了固定词表标注的固有偏差

## 局限与展望

1. **准确度-泛化权衡**：训练越久模型越倾向保守预测（低 NIV），如何在保持准确性的同时维持开放生成能力是未解决的问题
2. **计算开销**：冻结 LLM (Vicuna-7B) 推理时需要自回归生成类名，速度远慢于简单的嵌入匹配
3. **预测保守性**：相比 GPT-4V，OSM 倾向生成更通用的类名（"person" vs "man in armor"），可能通过更强的基础 LLM（如 Llama-2）或更大规模数据缓解
4. **未处理图像级数据**：如 ImageNet，因单标签无法处理多目标场景。未来可探索图像级别的弱监督集成
5. **依赖外部分割器**：mask 质量直接影响识别性能，端到端训练分割+识别可能更优

## 相关工作与启发

- **Vocabulary-Free Classification**：并行工作通过检索/解析生成词表来消除预定义需求，OSM 更彻底地通过生成式方法解决
- **InstructBLIP / LLaVA**：模块化多模态 LLM 的代表，OSM 在此基础上加入 mask 感知能力
- **DaTaSeg / LMSeg**：多数据集判别式分割方法，需要独立分类器或人工标签合并，OSM 自然避免了这些问题
- **启发**：生成式识别范式可推广到检测、跟踪等其他视觉任务，Mode Query 机制可用于其他多任务/多数据集场景

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|:-----------:|------|
| 创新性 | 9 | 开放式识别范式 + MaskQ-Former + Mode Query，范式转换意义重大 |
| 技术深度 | 8 | 各模块设计精巧，滑动窗口/双查询/双模式query环环相扣 |
| 实验充分度 | 9 | 6个数据集、GT mask + off-the-shelf mask、多维消融、GPT-4V对比、开放词表评估 |
| 写作质量 | 8 | 问题动机清晰，判别式→生成式的对比图直观 |
| 实用价值 | 8 | 消除词表依赖，多数据集训练零干预，实际部署价值大 |
| **总分** | **8.4** | 范式创新突出，实验严谨全面，ByteDance出品质量可靠 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](../../ACL2025/information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](../../ICML2026/information_retrieval/understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](../../NeurIPS2025/information_retrieval/murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ECCV 2024\] Multi-Label Cluster Discrimination for Visual Representation Learning](multi-label_cluster_discrimination_for_visual_representation_learning.md)
- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)

</div>

<!-- RELATED:END -->
