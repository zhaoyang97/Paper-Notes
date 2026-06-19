---
title: >-
  [论文解读] Multi-Aspect Cross-modal Quantization for Generative Recommendation
description: >-
  [AAAI 2026 Oral][图像生成][生成式推荐] 提出 MACRec，在生成式推荐的语义 ID 学习和生成模型训练两个阶段引入多方面跨模态交互，通过跨模态量化（对比学习增强残差量化）和多方面对齐（隐式+显式），显著提升推荐性能并降低 ID 冲突率。 领域现状 生成式推荐（Generative Recommendat…
tags:
  - "AAAI 2026 Oral"
  - "图像生成"
  - "生成式推荐"
  - "跨模态量化"
  - "残差量化"
  - "对比学习"
  - "语义ID"
---

# Multi-Aspect Cross-modal Quantization for Generative Recommendation

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.15122](https://arxiv.org/abs/2511.15122)  
**代码**: [github.com/zhangfw123/MACRec](https://github.com/zhangfw123/MACRec)  
**领域**: 图像生成 / 推荐系统  
**关键词**: 生成式推荐, 跨模态量化, 残差量化, 对比学习, 语义ID

## 一句话总结

提出 MACRec，在生成式推荐的语义 ID 学习和生成模型训练两个阶段引入多方面跨模态交互，通过跨模态量化（对比学习增强残差量化）和多方面对齐（隐式+显式），显著提升推荐性能并降低 ID 冲突率。

## 研究背景与动机

### 领域现状

生成式推荐（Generative Recommendation, GR）是一种新兴推荐范式，将推荐任务重构为下一个 token 预测问题。具体流程：(1) 通过残差量化（RQ-VAE）将物品嵌入离散化为**语义标识符（Semantic ID）**序列；(2) 将用户历史交互表示为语义 ID 序列；(3) 用序列生成模型（如 T5）预测下一个物品的语义 ID。代表工作有 TIGER、LC-Rec、LETTER 等。

### 现有痛点

**单模态信息不足**：现有 GR 方法主要使用文本嵌入来构建语义 ID，但**单一模态的语义区分度有限**。例如，同一品牌不同功能的乐器在文本嵌入空间中距离很近（品牌信息主导），难以区分不同产品。

**量化过程中的层级语义损失**：RQ-VAE 在更深的量化层级中存在显著的语义损失，导致模型在分配 token 时缺乏清晰的语义引导，ID 分配趋于随机化。

**跨模态交互缺失**：现有多模态 GR 方法（如 MQL4GRec）对每种模态独立编码获取语义 ID，在量化过程中没有考虑跨模态交互，无法充分利用模态间的互补性。

### 核心矛盾

如何在语义 ID 的**学习阶段**和**使用阶段**（生成模型训练）充分利用多模态信息的互补性，构建层级语义清晰、冲突率低、且能有效训练生成模型的高质量语义 ID？

### 核心 Idea

**在两个阶段引入跨模态交互**：(1) **量化阶段**：通过跨模态对比学习增强 RQ-VAE 每一层的残差表示，利用视觉伪标签优化文本残差、文本伪标签优化视觉残差，减少各层的语义损失并降低 ID 冲突率。(2) **生成模型训练阶段**：通过隐式对齐（潜在空间对比学习）和显式对齐（跨模态生成任务）让模型学习不同模态语义 ID 之间的共同特征。

## 方法详解

### 整体框架

MACRec 包含两个主要模块：
1. **跨模态物品量化（Cross-modal Item Quantization）**：生成高质量的多模态语义 ID
2. **多方面对齐的生成推荐（Generative Recommendation with Multi-aspect Alignment）**：利用对齐策略训练 GR 模型

### 关键设计

#### 1. **双模态伪标签生成（Dual-modality Pseudo-label Generation）**

**功能**：为跨模态对比学习构造正样本对。

**核心思路**：分别对所有物品的文本嵌入 $\{\mathbf{t}_i\}$ 和视觉嵌入 $\{\mathbf{v}_i\}$ 进行 K-means 聚类（$K$=512），得到文本聚类标签 $\mathcal{C}_{text}$ 和视觉聚类标签 $\mathcal{C}_{vision}$。

**设计动机**：不同模态的聚类结果反映了不同方面的物品相似性——文本按品牌/描述聚类，视觉按外观/形状聚类。利用一种模态的聚类标签作为另一种模态的正样本引导，实现互补信息注入。

#### 2. **跨模态量化与对比学习（Cross-modal Quantization with Contrastive Learning）**

**功能**：在 RQ-VAE 的每一量化层引入跨模态对比学习，增强残差表示的判别性。

**核心思路**：

对文本和视觉嵌入分别进行 $L$ 层残差量化：
$$c_l^t = \arg\min_k \|\mathbf{r}_l^t - \mathbf{e}_{l,k}^t\|_2, \quad \mathbf{r}_{l+1}^t = \mathbf{r}_l^t - \mathbf{e}_{l,c_k^t}^t$$

在每层的残差表示上施加跨模态对比损失——用**视觉伪标签**构造文本残差的正样本，用**文本伪标签**构造视觉残差的正样本：

$$\mathcal{L}_{con}^{l,v\rightarrow t} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\langle\mathbf{r}_i^t, \mathbf{r}_{i,pos}^t\rangle/\tau)}{\sum_{j=1}^{B}\exp(\langle\mathbf{r}_i^t, \mathbf{r}_j^t\rangle/\tau)}$$

**层级起始策略**：对比损失从第 3 层开始施加（$\lambda_{con}^{0,1}=0$, $\lambda_{con}^{2,3}=0.1$），前两层保留模态特有信息。

**设计动机**：(1) 独立量化文本和视觉会导致码本坍缩（相似嵌入映射到同一码字）和利用率低下；(2) 跨模态对比学习让每层残差能捕捉不同模态的互补特征，减少层级语义损失。

#### 3. **跨模态重建对齐（Cross-modal Reconstruction Alignment）**

**功能**：在量化后的表示上施加跨模态对齐约束，进一步优化码本。

**核心思路**：将各层码字求和得到量化表示 $\hat{\mathbf{z}}^t = \sum_{l=0}^{L-1}\mathbf{e}_{l,c_k^t}^t$，对同一物品的文本和视觉量化表示进行双向对比对齐：

$$\mathcal{L}_{align}^{t\rightarrow v} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\langle\hat{\mathbf{z}}_i^t, \hat{\mathbf{z}}_i^v\rangle/\tau)}{\sum_{j=1}^{B}\exp(\langle\hat{\mathbf{z}}_i^t, \hat{\mathbf{z}}_j^v\rangle/\tau)}$$

**设计动机**：确保同一物品在不同模态的量化表示在潜在空间中保持一致，平衡码本利用率。

#### 4. **隐式对齐（Implicit Alignment）**

**功能**：在 GR 模型的潜在空间中对齐同一物品的文本和视觉语义 ID 表示。

**核心思路**：将文本和视觉语义 ID 分别通过 T5 Encoder 编码并进行 Mean Pooling，然后通过双向 InfoNCE 损失对齐：

$$\mathcal{L}_{implicit} = \mathcal{L}_{implicit}^{t\rightarrow v} + \mathcal{L}_{implicit}^{v\rightarrow t}$$

**设计动机**：让 GR 模型能识别不同模态语义 ID 之间的共性，为后续预测提供更好的特征基础。

#### 5. **显式对齐（Explicit Alignment）**

**功能**：通过跨模态生成任务在输出空间进行对齐。

**核心思路**：设计两类跨模态生成任务：
- **物品级对齐**：文本 ID → 视觉 ID，视觉 ID → 文本 ID
- **序列级对齐**：文本 ID 序列 → 下一物品视觉 ID，视觉 ID 序列 → 下一物品文本 ID

这些辅助任务与标准序列推荐任务联合训练。

**设计动机**：隐式对齐仅在编码端，显式对齐则在解码端进一步加强跨模态关联，让模型从不同角度学习模态间的共同特征。

### 损失函数 / 训练策略

**语义 ID 学习阶段**：
$$\mathcal{L}_{ID} = \mathcal{L}_{RQ-VAE} + \lambda_{con}^l \sum_{l=0}^{L-1}\mathcal{L}_{con}^l + \lambda_{align}\mathcal{L}_{align}$$

其中 $\mathcal{L}_{RQ-VAE}$ 包含重建损失和量化损失。

**GR 模型训练阶段**：
$$\mathcal{L}_{rec} = -\sum_{t=1}^{|y|}\log P_\theta(y_t | y_{<t}, x) + \lambda_{implicit}\mathcal{L}_{implicit}$$

推理时使用受限束搜索生成候选语义 ID，取两种模态得分平均作为最终结果。

超参设置：码本大小 $M$=256，4 层量化，AdamW 优化器，batch size 1024，学习率 0.001，$\lambda_{align}=0.001$，$\lambda_{implicit}=0.01$，$\tau=0.1$。

## 实验关键数据

### 主实验

三个 Amazon 数据集上的推荐性能对比：

| 数据集 | 指标 | TIGER | MQL4GRec | **MACRec** | 提升 |
|--------|------|-------|----------|---------|------|
| Instruments | HR@1 | 0.0754 | 0.0763 | **0.0819** | +7.3% |
| Instruments | NDCG@10 | 0.0950 | 0.0997 | **0.1046** | +4.9% |
| Arts | HR@1 | 0.0532 | 0.0626 | **0.0685** | +9.4% |
| Arts | NDCG@10 | 0.0806 | 0.0898 | **0.0953** | +6.1% |
| Games | HR@10 | 0.0857 | 0.1007 | **0.1078** | +7.1% |
| Games | NDCG@10 | 0.0453 | 0.0538 | **0.0565** | +5.0% |

MACRec 在三个数据集的所有指标上均取得最佳结果（p < 0.05），统计显著优于最强基线 MQL4GRec。

### 消融实验

各模块消融（HR@10）：

| 配置 | Instruments | Arts | Games | 说明 |
|------|------------|------|-------|------|
| **MACRec (完整)** | **0.1363** | **0.1329** | **0.1078** | 全部组件 |
| w/o $\mathcal{L}_{con}^l$ | 0.1289 | 0.1283 | 0.1018 | 移除跨模态量化对比 |
| w/o $\mathcal{L}_{align}$ | 0.1310 | 0.1301 | 0.1026 | 移除重建对齐 |
| w/o $\mathcal{L}_{implicit}$ | 0.1312 | 0.1296 | 0.1042 | 移除隐式对齐 |
| w/o Explicit Alignment | 0.1296 | 0.1299 | 0.1037 | 移除显式对齐 |

**关键观察**：$\mathcal{L}_{con}$ 的移除导致最大性能下降，说明量化阶段的跨模态对比学习是最关键的组件。

ID 冲突率对比（%）：

| 数据集 | 模态 | MQL4GRec | MACRec | 降幅 |
|--------|------|----------|--------|------|
| Instruments | Text | 2.76 | **2.38** | -14% |
| Instruments | Image | 3.71 | **3.23** | -13% |
| Arts | Text | 5.15 | **4.24** | -18% |
| Games | Image | 26.10 | **25.24** | -3.3% |

MACRec 有效降低了两种模态的 ID 冲突率。

### 关键发现

1. **跨模态量化最重要**：$\mathcal{L}_{con}$ 是影响最大的组件，说明在量化阶段引入跨模态交互对语义 ID 质量至关重要。
2. **模态互补性可视化验证**：文本嵌入擅长按品牌聚类，视觉嵌入擅长区分产品类别（如不同乐器形态），两者互补。
3. **码本利用率改善**：MACRec 的码字分配更均匀，避免了码本坍缩问题。
4. **对比损失起始层选择**：从第 3 层开始施加对比损失效果最佳，前两层保留模态特有信息、后两层利用跨模态信号补偿语义损失。

## 亮点与洞察

1. **量化阶段的跨模态交互是首创**：以往多模态 GR 在量化时独立处理各模态，MACRec 首次在 RQ-VAE 的每一层引入跨模态交互。
2. **多方面对齐策略全面**：涵盖量化层对比、重建对齐、编码端隐式对齐、解码端显式对齐四个层面。
3. **伪标签机制巧妙**：利用一种模态的聚类结果为另一种模态提供正样本，避免了跨模态标注成本。
4. **ID 冲突率分析深入**：不仅看推荐性能，还分析了量化质量（冲突率、码字分布），为理解方法有效性提供了多角度证据。

## 局限与展望

1. **仅使用文本+图像两种模态**：未考虑其他可能有用的模态（如价格、类别标签、用户评论）。
2. **聚类数 K=512 固定**：对不同数据集可能需要不同的聚类粒度设定。
3. **GR 模型骨干受限**：使用 T5-small（4层 encoder+decoder），在更大模型上的效果需要验证。
4. **MQL4GRec 公平性**：文中提到未使用 MQL4GRec 的百万级预训练数据以确保公平比较，但这也意味着 MQL4GRec 可能有更大潜力未被展示。
5. **推理效率**：双模态推理需要两条生成路径 + 分数融合，推理开销高于单模态 GR。

## 相关工作与启发

- **TIGER**：开创了 RQ-VAE 生成语义 ID 的范式，MACRec 在其基础上引入多模态。
- **MQL4GRec**：最直接的前序工作，使用多模态量化语言但缺乏量化阶段的跨模态交互。
- **CLIP/对比学习**：InfoNCE 损失在多个位置被使用，展示了对比学习在多模态推荐中的强大适用性。
- **对推荐系统的启发**：RQ-VAE 量化质量是 GR 性能的关键瓶颈，从量化阶段入手优化可能是更有效的提升路径。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 量化阶段跨模态交互是核心创新，多方面对齐策略全面
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多种消融、冲突率分析、码字分布可视化
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，方法描述详细
- **价值**: ⭐⭐⭐⭐ — 对 GR 领域贡献显著，跨模态量化思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](../../ICCV2025/image_generation/stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[CVPR 2026\] Aligning Multi-Character Narrative Image Generation with Multi-Aspect Human Preferences](../../CVPR2026/image_generation/aligning_multi-character_narrative_image_generation_with_multi-aspect_human_pref.md)
- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](../../CVPR2026/image_generation/cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](../../CVPR2025/image_generation/generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2026\] PhotoFramer: Multi-modal Image Composition Instruction](../../CVPR2026/image_generation/photoframer_multi-modal_image_composition_instruction.md)

</div>

<!-- RELATED:END -->
