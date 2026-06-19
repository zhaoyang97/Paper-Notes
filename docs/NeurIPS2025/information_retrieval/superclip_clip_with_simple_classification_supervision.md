---
title: >-
  [论文解读] SuperCLIP: CLIP with Simple Classification Supervision
description: >-
  [NeurIPS 2025][信息检索/RAG][CLIP] 在CLIP对比学习框架中引入一个超简单的分类损失（仅需添加一个轻量线性层，FLOPs增加仅0.077%），利用原始文本token的分类信号恢复CLIP未充分利用的细粒度文本监督，在零样本分类、图文检索和纯视觉任务上一致提升性能。 CLIP通过对比学习将图像和文本对…
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "CLIP"
  - "视觉-语言预训练"
  - "分类监督"
  - "细粒度对齐"
  - "对比学习"
---

# SuperCLIP: CLIP with Simple Classification Supervision

**会议**: NeurIPS 2025  
**arXiv**: [2512.14480](https://arxiv.org/abs/2512.14480)  
**代码**: [GitHub (hustvl/SuperCLIP)](https://github.com/hustvl/SuperCLIP)  
**领域**: 信息检索  
**关键词**: CLIP, 视觉-语言预训练, 分类监督, 细粒度对齐, 对比学习

## 一句话总结

在CLIP对比学习框架中引入一个超简单的分类损失（仅需添加一个轻量线性层，FLOPs增加仅0.077%），利用原始文本token的分类信号恢复CLIP未充分利用的细粒度文本监督，在零样本分类、图文检索和纯视觉任务上一致提升性能。

## 研究背景与动机

CLIP通过对比学习将图像和文本对齐到共享嵌入空间，在零样本分类和检索任务中表现优异。然而，最近的研究揭示了一个值得深思的现象：

**CLIP未能充分利用文本中的丰富监督信号**。这表现在三个方面：

**对比学习的固有局限**：CLIP仅优化全局图文相似度，忽略了文本中单词/短语级别的细粒度语义。例如，CLIP可能混淆雕像与真人（对象状态）、难以区分熊在河里还是河外（空间关系）。

**Web数据的稀疏性**：作者统计了DataComp-1B中1000万条caption，发现"man + newspaper"出现333次，但"man + newspaper + real/statue"仅6次，"bear + river + in/out"更是几乎为零。这些低频细粒度组合难以在同一batch中组成有效对比对。

**富文本描述反而降低CLIP性能**：使用LLaMA-3重新生成更详细的caption（Recap-DataComp）后，完全替换原始数据训练CLIP，性能反而下降。这说明对比学习范式无法有效利用更丰富的文本描述——添加的复杂性甚至会干扰学习。

**对batch size的强依赖**：CLIP需要大batch（通常16K+）才能在batch内形成多样化的正负对。小batch下性能急剧下降。

## 方法详解

### 整体框架

SuperCLIP在CLIP框架基础上仅增加一个轻量线性层，将视觉编码器的平均池化特征映射到文本分类目标。分类损失与对比损失联合优化，无需额外标注数据，训练数据、视觉编码器和文本编码器均直接复用CLIP。

### 关键设计

1. **文本token作为分类标签**：将每条caption通过CLIP的subword分词器得到token ID集合 $\mathcal{C}$，构造 $V$ 维K-hot向量 $\mathbf{y} \in \mathbb{R}^V$（$V$ 为词表大小）。与传统分类不同，这里的"类别"是原始文本token，无需任何人工过滤或词表构建。

2. **IDF加权**：直接使用K-hot标签会让高频停用词主导学习。引入逆文档频率（IDF）加权：

$$w_c = \log\left(\frac{|\mathcal{D}|}{1 + \text{df}(c)}\right)$$

归一化后的加权标签分布为：

$$\hat{y}_c = \frac{w_c y_c}{\sum_{c'=1}^V w_{c'} y_{c'}}$$

这使模型聚焦于信息密度高的词汇（如"zebra"、"skateboarding"），减少对"the"、"is"等功能词的过度关注。

3. **分类损失**：对视觉编码器的平均池化特征施加线性层得到logits $x_c$，用加权交叉熵：

$$\mathcal{L}_{\text{Class}} = -\sum_{c=1}^{V} \hat{y}_c \log\left(\frac{e^{x_c}}{\sum_{c'=1}^{V} e^{x_{c'}}}\right)$$

4. **总损失**：

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CLIP}} + \mathcal{L}_{\text{Class}}$$

分类损失不依赖batch内的负样本，因此天然对batch size不敏感，可缓解CLIP在小batch下的性能衰退。

### 损失函数 / 训练策略

- 在DataComp-1B子集（约13亿图文对）上预训练
- 图像分辨率224×224，AdamW优化器，余弦学习率调度
- 默认batch size 16K（与CLIP一致以公平比较）
- 线性层仅引入0.051 GFLOPs（L-size），占总计算量的0.077%
- 支持DualCaption模式：对比损失用短caption，分类损失用长caption

## 实验关键数据

### 主实验（不同模型尺寸）

| 模型 | ImageNet val(%) | ImageNet v2(%) | COCO图像检索(%) | Flickr图像检索(%) |
|------|----------------|----------------|----------------|-----------------|
| CLIP B-512M | 60.5 | 53.0 | 29.0 | 54.5 |
| **SuperCLIP B-512M** | **63.5** (+3.0) | **55.2** (+2.2) | **31.3** (+2.3) | **56.9** (+2.4) |
| CLIP L-512M | 66.1 | 57.4 | 32.7 | 57.0 |
| **SuperCLIP L-512M** | **70.1** (+4.0) | **62.5** (+5.1) | **35.9** (+3.2) | **62.4** (+5.4) |
| CLIP L-12.8B | 79.0 | 72.0 | 43.9 | 72.7 |
| **SuperCLIP L-12.8B** | **80.0** (+1.0) | **72.8** (+0.8) | **45.5** (+1.6) | **74.2** (+1.5) |

### 恢复富文本监督（Mixed Caption实验）

| 模型 | Caption比例 | 38数据集平均分类(%) | COCO图像检索(%) | Flickr文本检索(%) |
|------|-----------|-------------------|----------------|-----------------|
| CLIP-L (1.0/0.0) | 短100% | 45.7 | 32.7 | 76.4 |
| CLIP-L (0.0/1.0) | 长100% | 30.0 | 26.2 | 65.9 |
| CLIP-L (0.8/0.2) | 短80%/长20% | 46.8 | 37.0 | 78.8 |
| **SuperCLIP-L (Dual)** | 对比=短/分类=长 | **49.5** (+2.7) | **37.6** | **82.5** |

CLIP用100%长caption训练性能大幅下降（45.7→30.0），但SuperCLIP的DualCaption模式有效利用了长caption的丰富语义。

### 消融实验

| 配置 | ImageNet(%) | COCO图像检索(%) | Flickr文本检索(%) |
|------|------------|----------------|-----------------|
| λ=0.4 | 44.1 | 41.3 | 58.3 |
| λ=1.0 | **47.1** | **44.0** | 61.0 |
| λ=1.6 | 47.2 | 44.2 | **62.0** |
| 无IDF | 44.8 | (31.6, 51.7) | (48.0, 71.1) |
| **有IDF** | **47.1** | **(33.2, 54.7)** | **(48.9, 73.1)** |

### 泛化性验证

| 框架 | ImageNet val(%) | ImageNet v2(%) | COCO图像检索(%) | Flickr文本检索(%) |
|------|----------------|----------------|----------------|-----------------|
| SigLIP | 60.4 | 52.8 | 29.8 | 73.2 |
| **SuperSigLIP** | **64.1** (+3.7) | **55.9** (+3.1) | **32.5** (+2.7) | **75.9** (+2.7) |
| FLIP | 58.1 | 50.1 | 27.5 | 66.7 |
| **SuperFLIP** | **61.3** (+3.2) | **53.5** (+3.4) | **30.1** (+2.6) | **72.0** (+5.3) |

### 关键发现

- 词-图像相似度分析：CLIP的Top-20词全部是物体类别词（zebras, kites），而SuperCLIP成功将状态词（blurry）、空间词（inside）、动作词（stands）提升到高排名
- SuperCLIP在纯视觉任务上也有一致提升：线性探测+1.3~1.5%，语义分割+2.1~4.1%，深度估计也有改善
- 集成到LLaVA-1.5后，在VQAv2（+1.8）、MMBench（+6.8）等多模态任务上也优于CLIP编码器
- Batch size从32K降到1K时，SuperCLIP的性能衰退远小于CLIP

## 亮点与洞察

1. **极致简洁**：仅增加一个线性层和一个分类损失，就解决了CLIP的细粒度对齐短板，堪称"用最简单的方法解决实际问题"的典范
2. **深刻的问题洞察**：通过DataComp-1B的关键词共现统计，定量解释了为什么对比学习难以捕获细粒度语义
3. **DualCaption策略巧妙**：对比损失用短caption保持粗粒度对齐，分类损失用长caption提取细粒度语义，避免了需要精心调参的混合比例
4. **batch size鲁棒性**：分类损失天然不依赖batch size，为资源受限的训练场景提供了实用方案

## 局限与展望

- 分类监督仅从文本到视觉编码器方向，未探索从图像到文本编码器方向的增强
- IDF权重在训练前预计算，未能动态适应训练过程中的语义分布变化
- 线性分类头可能限制了更复杂语义关系的建模能力
- 在部分特定数据集（如DSprites、SmallNORB等合成数据集）上提升有限

## 相关工作与启发

- 与RegionCLIP（区域级监督）、Long-CLIP（长文本理解）等方法正交，可组合使用
- 分类监督的思路来自Classification-based Supervision (Huang et al. 2024)，本文将其规模化到CLIP预训练
- 为SigLIP、FLIP等其他对比学习变体提供了通用增强方案
- 启示：对比学习的信息瓶颈可能比想象的更严重，补充分类信号是简单有效的缓解手段

## 评分

- 新颖性: ⭐⭐⭐⭐ 方法极简但问题洞察深刻，IDF加权和DualCaption有巧思
- 实验充分度: ⭐⭐⭐⭐⭐ 多尺寸模型、多框架泛化、38数据集评估、MLLM集成、batch size分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机分析透彻，数据统计支撑充分
- 实用价值: ⭐⭐⭐⭐⭐ 零门槛可集成到任何CLIP训练流程，开源代码

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] External Knowledge Injection for CLIP-Based Class-Incremental Learning](../../ICCV2025/information_retrieval/external_knowledge_injection_for_clip-based_class-incremental_learning.md)
- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](scaling_language-centric_omnimodal_representation_learning.md)
- [\[CVPR 2025\] Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training](../../CVPR2025/information_retrieval/advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)

</div>

<!-- RELATED:END -->
