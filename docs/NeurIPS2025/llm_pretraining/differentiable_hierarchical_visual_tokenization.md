---
title: >-
  [论文解读] Differentiable Hierarchical Visual Tokenization
description: >-
  [NeurIPS 2025 (Spotlight)][预训练][Visual Tokenizer] 提出一种端到端可微分的层次化视觉分词器，以像素级粒度自适应图像内容进行 token 划分，利用信息准则进行层次模型选择，可直接替换 ViT 的固定 patch 分词，并支持光栅-矢量转换。 Vision Transformer…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "预训练"
  - "Visual Tokenizer"
  - "层次化分词"
  - "可微分"
  - "超像素"
  - "信息准则"
---

# Differentiable Hierarchical Visual Tokenization

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2511.02652](https://arxiv.org/abs/2511.02652)  
**代码**: 有  
**领域**: 计算机视觉 / Vision Transformer  
**关键词**: Visual Tokenizer, 层次化分词, 可微分, 超像素, 信息准则

## 一句话总结

提出一种端到端可微分的层次化视觉分词器，以像素级粒度自适应图像内容进行 token 划分，利用信息准则进行层次模型选择，可直接替换 ViT 的固定 patch 分词，并支持光栅-矢量转换。

## 研究背景与动机

Vision Transformer (ViT) 及其变体已成为计算机视觉的主流架构，但其分词策略存在根本性的局限：

**固定 patch 分词的问题：**

**忽视空间结构**：无论图像内容如何，都将其划分为固定大小（如 16×16）的网格 patch，完全无视目标边界和语义结构

**语义不对齐**：一个 patch 可能同时包含前景和背景，导致 token 的语义不纯

**效率浪费**：纹理简单的区域（如天空）和复杂区域（如建筑细节）使用相同数量的 token，造成计算资源浪费

**信息损失**：固定网格的边界可能切割重要的视觉特征

**已有自适应分词方案的局限：**

- 大多不可微分，无法端到端训练
- 依赖预训练的分割模型，增加额外计算
- 难以与现有预训练 ViT 兼容

**DHVT 的核心思想**：设计一个完全可微分的视觉分词器，能够根据图像内容自适应地决定 token 的数量和位置，同时保持与现有架构的向后兼容性。

## 方法详解

### 整体框架

DHVT（Differentiable Hierarchical Visual Tokenization）包含三个核心组件：

1. **像素级特征提取**：提取每个像素的嵌入特征
2. **层次化分割/分组**：使用信息准则自上而下或自下而上地将像素分组为语义一致的 token
3. **Token 聚合**：将每组像素的特征聚合为单个 token 表示

### 关键设计

**1. 可微分的超像素生成**

DHVT 使用可微分的超像素（superpixel）方法将图像划分为语义一致的区域：

- 每个像素 $p_i$ 有一个特征向量 $\mathbf{f}_i$ 和空间位置 $(x_i, y_i)$
- 通过可微分的软分配（soft assignment），每个像素以概率分配到若干 token 组
- 分配概率基于特征相似度和空间邻近性

**2. 基于信息准则的层次模型选择**

DHVT 使用**贝叶斯信息准则（BIC）**来自动确定每个图像区域的最优 token 数量：

$$\text{BIC}(k) = -2\ln L + k \cdot \ln n$$

其中 $L$ 是给定 $k$ 个 token 时的似然，$n$ 是像素数。BIC 平衡了模型拟合度和复杂度：
- 纹理简单的区域：少量 token 即可充分表示（低 BIC）
- 纹理复杂的区域：需要更多 token（低 BIC 需更大 $k$）

层次化过程：
1. 初始从粗粒度（少量 token）开始
2. 递归地对每个 token 区域判断是否需要进一步细分
3. 当子分割不能显著降低 BIC 时停止
4. 最终得到自适应数量和大小的 token

**3. 向后兼容设计**

为了能**直接 retrofit 预训练 ViT**：
- 生成的 token 数量可变，但通过填充/截断与预训练模型的 token 数对齐
- token 特征通过可学习的投影层映射到与 patch embedding 相同的维度
- 位置编码根据 token 的空间位置动态生成，而非固定网格

**4. 光栅到矢量转换**

作为附加能力，DHVT 的层次化分割结果可以直接用于 raster-to-vector（光栅到矢量图）转换：每个 token 对应一个矢量图元素（多边形区域 + 均匀颜色/特征），无需额外训练。

### 损失函数 / 训练策略

DHVT 的分词器与下游任务联合端到端训练：

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{task}}$：下游任务损失（如分类的交叉熵、分割的 dice loss）
- $\mathcal{L}_{\text{reg}}$：正则化项，鼓励 token 边界与语义边界对齐，惩罚过多/过少的 token

可微分性保证梯度可以从下游损失传播回分词器参数，实现真正的端到端学习。

## 实验关键数据

### 主实验

**Table 1：ImageNet-1K 分类准确率**

| 方法 | Backbone | Top-1 Acc (%) ↑ | #Tokens (avg) |
|------|----------|-----------------|---------------|
| ViT-B/16 (fixed patch) | ViT-B | 81.8 | 196 (固定) |
| DynamicViT | ViT-B | 81.3 | ~130 |
| ToMe | ViT-B | 81.5 | ~150 |
| DHVT (Ours) | ViT-B | **82.3** | ~160 (自适应) |
| ViT-L/16 (fixed patch) | ViT-L | 85.2 | 196 (固定) |
| DHVT (Ours) | ViT-L | **85.7** | ~170 (自适应) |

DHVT 在使用更少平均 token 数的情况下，分类精度优于固定 patch 和 token pruning/merging 方法。

**Table 2：ADE20K 语义分割（mIoU）**

| 方法 | Backbone | mIoU (%) ↑ | #Tokens (avg) |
|------|----------|------------|---------------|
| ViT-B/16 + UperNet | ViT-B | 47.4 | 196 |
| SegFormer-B2 | MiT-B2 | 46.5 | 多尺度 |
| DHVT + UperNet | ViT-B | **48.8** | ~180 (自适应) |

在密集预测任务（语义分割）中，DHVT 的语义对齐 token 带来更显著的提升（+1.4 mIoU），因为 token 边界与语义边界的对齐直接有利于像素级预测。

### 消融实验

**层次化深度的影响**

| 最大层次深度 | Top-1 Acc (%) | 平均 #Tokens |
|-------------|---------------|-------------|
| 1 (无层次) | 81.6 | 196 |
| 2 | 82.0 | ~175 |
| 3 | **82.3** | ~160 |
| 4 | 82.2 | ~145 |

3 层层次深度达到最佳精度-效率平衡。过深的层次可能导致某些 token 过小而丢失信息。

**信息准则选择**

| 准则 | Top-1 Acc (%) |
|------|--------------|
| 固定数量 | 81.9 |
| AIC | 82.0 |
| BIC | **82.3** |
| MDL | 82.1 |

BIC 的惩罚项最好地平衡了 token 数量和表示质量。

### 关键发现

1. **语义对齐至关重要**：DHVT 的 token 边界与目标边界天然对齐，在分割等密集任务上优势更明显
2. **自适应 token 分配**：简单区域用少量 token，复杂区域用更多 token，计算资源分配更合理
3. **Retrofit 可行性**：可以直接在预训练 ViT 上微调分词器，无需从头训练
4. **矢量图转换**：层次化分割的副产品可直接用于 SVG 生成，展现了方法的通用性
5. **信息准则的有效性**：BIC 提供了无需调参的 token 数量自动选择机制

## 亮点与洞察

- **从固定到自适应的范式转换**：ViT 的 patch 分词是一个被广泛接受但从未被质疑的设计选择，DHVT 展示了更好的替代方案
- **可微分 + 信息准则**：将统计模型选择理论（BIC）融入端到端深度学习，是一个巧妙的结合
- **向后兼容**：不需要重新设计架构或从头预训练，可以直接增强现有模型
- **一石多鸟**：分类精度提升 + token 效率改善 + 免费矢量图转换
- **NeurIPS Spotlight** 体现了该方向的重要性和方法的完成度

## 局限与展望

1. **分词器额外计算**：可微分超像素生成本身有计算开销，需要在 token 减少带来的加速和分词开销间平衡
2. **可变 token 数处理**：batch 内 token 数不同需要特殊处理（填充或分桶），影响训练效率
3. **与大规模预训练集成**：是否能在 CLIP/DINOv2 级别的预训练中使用 DHVT 尚待验证
4. **视频扩展**：将层次化分词扩展到时序维度是自然的扩展方向
5. **超像素质量**：初始超像素的质量直接影响后续层次化构建

## 相关工作与启发

- **ViT (Dosovitskiy et al. 2021)**：固定 patch 分词的标准范式
- **DynamicViT**：动态 token 剪枝，但在分词后操作
- **ToMe (Token Merging)**：训练后合并冗余 token，正交方法
- **Superpixel 方法 (SLIC, etc.)**：经典的超像素分割方法
- **SegFormer**：用层次化特征的密集预测 Transformer

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 5 — 可微分层次化视觉分词是全新方向 |
| 技术质量 | 4 — 理论动机清晰，信息准则应用巧妙 |
| 实验充分性 | 4 — 分类 + 分割 + 矢量化验证 |
| 写作质量 | 4 — Spotlight 水平的清晰表达 |
| 影响力 | 5 — 可能重塑 ViT 分词范式 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Next Semantic Scale Prediction via Hierarchical Diffusion Language Models](next_semantic_scale_prediction_via_hierarchical_diffusion_language_models.md)
- [\[ACL 2025\] Adversarial Tokenization](../../ACL2025/llm_pretraining/adversarial_tokenization.md)
- [\[ACL 2025\] Tokenization is Sensitive to Language Variation](../../ACL2025/llm_pretraining/tokenization_is_sensitive_to_language_variation.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](../../ACL2025/llm_pretraining/incorporating_domain_knowledge_into_materials_tokenization.md)

</div>

<!-- RELATED:END -->
