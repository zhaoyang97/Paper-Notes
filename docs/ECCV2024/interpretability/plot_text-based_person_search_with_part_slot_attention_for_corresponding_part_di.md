---
title: >-
  [论文解读] PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery
description: >-
  [ECCV 2024][可解释性][text-based person search] 提出 PLOT 框架，利用基于 Slot Attention 的 Part Discovery Module 自动发现跨模态（图像-文本）对应的人体部件，结合 Text-based Dynamic Part Attention（TDPA）动态调整各部件重要性，无需部件级标注即可在三个 benchmark 上全面超越 SOTA。
tags:
  - "ECCV 2024"
  - "可解释性"
  - "text-based person search"
  - "注意力机制"
  - "part discovery"
  - "跨模态"
  - "对比学习"
---

# PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery

**会议**: ECCV 2024  
**arXiv**: [2409.13475](https://arxiv.org/abs/2409.13475)  
**代码**: [https://cvlab.postech.ac.kr/research/PLOT](https://cvlab.postech.ac.kr/research/PLOT)  
**领域**: 跨模态检索 / 行人搜索  
**关键词**: text-based person search, slot attention, part discovery, cross-modal retrieval, contrastive learning

## 一句话总结
提出 PLOT 框架，利用基于 Slot Attention 的 Part Discovery Module 自动发现跨模态（图像-文本）对应的人体部件，结合 Text-based Dynamic Part Attention（TDPA）动态调整各部件重要性，无需部件级标注即可在三个 benchmark 上全面超越 SOTA。

## 研究背景与动机
Text-based person search（基于文本的行人搜索）要求根据自然语言描述从大规模图像库中找到目标行人。这一任务的核心挑战在于建立文本与图像中人体部件的细粒度对应关系，因为不同行人的区分往往依赖于衣着、配饰等部件级细节。

现有方法存在三类问题：

**启发式部件提取**：将图像等距水平裁剪作为部件特征，对遮挡和姿态变化敏感，且包含大量无关背景信息

**学习式方法缺陷**：传统 cross-attention 方法容易生成冗余、缺乏区分性的部件特征；或依赖外部工具（关键点检测、属性分割）增加计算成本

**全局表示局限**：CLIP-based 方法（如 IRRA）主要关注全局特征，未针对人体部件设计

**核心 idea**：利用 Slot Attention 机制的竞争性聚合特性，让可学习的 "part slots" 自动发现并绑定到不同人体部件，通过跨模态共享 slots 自然建立部件对应关系，无需任何部件级监督。

## 方法详解

### 整体框架
基于预训练 CLIP（ViT-B/16 图像编码器 + CLIP-Xformer 文本编码器）为 backbone，每个模态提取两种表示：
- **Global embedding**：来自 [cls]/[EOS] token，捕获整体信息
- **Part embeddings**：来自 Part Discovery Module，捕获 $K$ 个部件的细粒度信息

推理时相似度 = 全局相似度 + TDPA 加权的部件相似度：$c(\mathbf{g}^{\mathcal{V}}, \mathbf{g}^{\mathcal{T}}) + c_{\text{agg}}(\mathbf{P}^{\mathcal{V}}, \mathbf{P}^{\mathcal{T}}; \mathbf{g}^{\mathcal{T}})$

### 关键设计

1. **Part Discovery Module（部件发现模块）**:

    - 功能：从图像 patch tokens 和文本 word tokens 中自动发现并提取 $K$ 个部件嵌入
    - 核心思路：基于 Slot Attention 机制。定义 $K$ 个可学习的 part slots $\mathbf{S}^0 \in \mathbb{R}^{K \times D}$，经过 $T$ 轮迭代的 Part Slot Attention（PSA）Block 精炼为部件嵌入
    - PSA Block 流程：
        - 计算 attention map：$A_{n,k} = \frac{e^{M_{n,k}}}{\sum_{i=1}^{K} e^{M_{n,i}}}$，其中 $M = \frac{k(\mathbf{x}^{\mathcal{V}}) q(\mathbf{S}^{t-1})^{\top}}{\sqrt{D_h}}$
        - **关键**：softmax 沿 slot 维度（$K$）归一化，而非输入维度，迫使 slots 之间竞争绑定不同的输入 token，确保部件互不重叠
        - 加权聚合：$\bar{A}_{n,k} = \frac{A_{n,k}}{\sum_{i=1}^{N} A_{i,k}}$，$\bar{\mathbf{S}}^t = \text{GRU}(\mathbf{S}^{t-1}, \bar{A}^{\top} v(\mathbf{x}^{\mathcal{V}}))$
        - MLP + 残差连接：$\mathbf{S}^t = \text{MLP}(\bar{\mathbf{S}}^{t-1}) + \bar{\mathbf{S}}^{t-1}$
    - 设计动机：Slot Attention 原用于 object-centric learning 中的无监督物体发现，本文将其首次应用于人体部件发现。相比传统 cross-attention（如 PAT），slot attention 的竞争机制天然鼓励部件分离
    - **跨模态对应**：两个模态的 Part Discovery Module 共享同一组初始 part slots $\mathbf{S}^0$，使得来自同一 slot 的视觉/文本部件嵌入天然对应

2. **Text-based Dynamic Part Attention (TDPA)**:

    - 功能：根据文本查询动态调整各部件相似度的权重
    - 核心思路：不同查询关注不同部件（如某查询只提到上衣和鞋子，则其他部件权重应降低）
    - 关键公式：
        - 权重预测：$\mathbf{a} = \sigma(\text{MLP}(\mathbf{g}^{\mathcal{T}})) \in \mathbb{R}^K$
        - 加权相似度：$c_{\text{agg}}(\mathbf{P}^{\mathcal{V}}, \mathbf{P}^{\mathcal{T}}; \mathbf{g}^{\mathcal{T}}) = \sum_{k=1}^{K} a_k \cdot c(\mathbf{p}_k^{\mathcal{V}}, \mathbf{p}_k^{\mathcal{T}})$
    - 与之前方法的区别：现有方法等权重聚合所有部件相似度，忽略了不同查询的焦点差异

3. **Cross-Modal Masked Language Modeling (CMLM)**:

    - 功能：辅助损失，促进跨模态交互学习
    - 核心思路：类似 BERT 的 MLM，随机 mask 15% 的文本 token，拼接图像 token 后用 transformer 恢复被 mask 的词
    - 公式：$\mathcal{L}_{\text{CMLM}} = -\frac{1}{L}\sum_{l=1}^{L} \mathbf{y}_l \log(\sigma(\mathbf{f}_l \mathbf{W}_{\text{CMLM}}))$

### 损失函数 / 训练策略
总损失：$\mathcal{L} = \mathcal{L}_{\text{Global}} + \mathcal{L}_{\text{Part}} + \mathcal{L}_{\text{CMLM}}$

- **Global Alignment Loss**：$\mathcal{L}_{\text{Global}} = \mathcal{L}_{\text{NCE}} + \mathcal{L}_{\text{ID}}$
    - $\mathcal{L}_{\text{NCE}}$：双向 InfoNCE 对比损失，对齐全局嵌入
    - $\mathcal{L}_{\text{ID}}$：身份分类损失（跨模态共享分类器），确保同一身份的嵌入接近

- **Part Alignment Loss**：$\mathcal{L}_{\text{Part}} = \mathcal{L}_{\text{PartNCE}} + \mathcal{L}_{\text{PartID}}$
    - $\mathcal{L}_{\text{PartNCE}}$：使用 TDPA 加权相似度 $c_{\text{agg}}$ 的 InfoNCE 损失
    - $\mathcal{L}_{\text{PartID}}$：将所有部件嵌入拼接后做身份分类

- 训练配置：Adam 优化器，60 epochs，batch size 128，CLIP 编码器学习率 $5 \times 10^{-6}$，其余参数学习率 ×20，cosine schedule + 5 epoch warm-up

## 实验关键数据

### 主实验
三个 benchmark 上的 R@1 对比（使用 CLIP-ViT-B/16 backbone）：

| 方法 | CUHK-PEDES R@1 | ICFG-PEDES R@1 | RSTPReid R@1 |
|------|----------------|----------------|--------------|
| TIPCB | 64.26 | - | - |
| IVT | 65.59 | 56.04 | 46.70 |
| CFine | 69.57 | 60.83 | 50.55 |
| IRRA (前 SOTA) | 73.38 | 63.46 | 60.20 |
| **PLOT (Ours)** | **75.28** | **65.76** | **61.80** |

在三个数据集上 R@1 分别超越 IRRA 1.9%、2.3%、1.6%，全面刷新 SOTA。

### 消融实验

**损失函数组合消融（CUHK-PEDES）**：

| 配置 | R@1 | R@5 | R@10 | 说明 |
|------|-----|-----|------|------|
| Global Only ($\mathcal{L}_{\text{NCE}}$) | 71.39 | 87.65 | 92.74 | 基线 |
| + $\mathcal{L}_{\text{ID}}$ | 71.83 | 88.06 | 92.58 | +0.44 |
| + $\mathcal{L}_{\text{CMLM}}$ | 72.65 | 88.58 | 92.93 | +1.26 |
| + Part Embeddings (无 $\mathcal{L}_{\text{PartID}}$) | 74.85 | 90.29 | 94.10 | +3.46 |
| **Full Model** | **75.28** | **90.42** | **94.12** | **+3.89** |

Part embeddings 贡献最大（R@1 +3.46%），Part ID loss 进一步提升 0.43%。

**部件发现方法对比**：

| 方法 | R@1 | R@5 | R@10 | 说明 |
|------|-----|-----|------|------|
| TIPCB (等距裁剪) | 73.23 | 89.10 | 94.04 | 启发式部件 |
| PAT (cross-attention) | 72.76 | 89.23 | 93.42 | 传统注意力 |
| **PLOT (slot attention)** | **75.28** | **90.42** | **94.12** | 竞争式 slot |

Slot attention 在 R@1 上超越 TIPCB 2.05%、PAT 2.52%，验证了竞争式部件发现的优势。

### 关键发现
- Part embeddings 是最关键的改进来源：引入后 R@1 提升 3.46%（从 72.65 到 74.85）
- Slot attention 的竞争归一化机制优于传统 cross-attention（PAT），后者的部件容易重叠、只关注显著区域
- TDPA 能根据查询文本自适应调整部件权重：如查询不包含"帽子"相关描述时，头部 slot 的权重自动降低
- 可视化显示各 slot 学到了语义一致的部件映射：slot 1 → 下装，slot 4 → 鞋子，slot 5 → 手持物品，slot 7 → 上衣，slot 8 → 头部
- 跨模态共享 slots 使得同一 slot 在图像和文本中关注语义相同的部件，提供了可解释的检索

## 亮点与洞察
- **Slot Attention → Part Discovery 的巧妙迁移**：将 object-centric learning 的 slot attention 创新应用于人体部件发现，竞争机制天然保证部件不重叠，无需外部监督
- **Slot 共享实现跨模态对应**：通过共享初始 part slots，图像和文本的部件嵌入自然对齐，无需显式部件对应标注
- **TDPA 查询自适应**：不同查询关注不同部件的思路简单有效，MLP 预测权重 + 端到端学习
- **可解释性**：每个 slot 的 attention map 可视化清晰展示了模型关注的人体区域，增强了检索的可解释性

## 局限与展望
- 作者坦诚：slots 会覆盖整个图像和文本，部分 slot 可能绑定到无关区域（背景）。TDPA 部分缓解此问题，但更显式的解决方案（如前景约束）会更好
- Slot 数量 $K=8$ 固定，未探索自适应 slot 数量
- 仅在 CLIP-ViT-B/16 上验证，未测试更大 backbone（ViT-L/14 等）
- 未与最新的基于 LLM 的方法（如用 GPT-4V 做 person search）对比

## 相关工作与启发
- **vs IRRA**：IRRA 只对齐全局特征，PLOT 在此基础上增加部件级对齐，R@1 提升 1.9-2.3%
- **vs TIPCB**：TIPCB 的等距裁剪是粗糙的部件近似，PLOT 通过 slot attention 学习到语义一致的部件
- **vs PAT (cross-attention)**：PAT 的传统 cross-attention 缺乏竞争机制，部件容易重叠，PLOT 的 slot attention 每个 slot 竞争绑定不同区域，部件更具区分性

## 评分
- 新颖性: ⭐⭐⭐⭐ Slot attention 在 person search 中的首次应用，部件发现+跨模态对应的统一框架设计优雅
- 实验充分度: ⭐⭐⭐⭐ 三个 benchmark 全面超 SOTA，多维度消融（损失组合、部件方法、TDPA），丰富的可视化分析
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，方法动机阐述充分，可视化效果好
- 价值: ⭐⭐⭐⭐ 部件级跨模态对齐的思路可广泛迁移到其他细粒度检索任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](../../CVPR2025/interpretability/interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[ICLR 2026\] Adaptive Concept Discovery for Interpretable Few-Shot Text Classification](../../ICLR2026/interpretability/adaptive_concept_discovery_for_interpretable_few-shot_text_classification.md)
- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](../../AAAI2026/interpretability/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](../../AAAI2026/interpretability/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)

</div>

<!-- RELATED:END -->
