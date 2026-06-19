---
title: >-
  [论文解读] Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning
description: >-
  [AAAI 2026 Oral][多模态VLM][Composed Image Retrieval] 本文提出了HUG范式，通过细粒度高斯概率嵌入和异构不确定性估计（区分查询侧多模态协调不确定性与目标侧内容质量不确定性），结合动态加权融合和不确定性引导的对比学习，在Fashion-IQ和CIRR两个CIR基准上取得SOTA。
tags:
  - "AAAI 2026 Oral"
  - "多模态VLM"
  - "Composed Image Retrieval"
  - "不确定性建模"
  - "概率嵌入"
  - "细粒度匹配"
  - "高斯表示"
---

# Heterogeneous Uncertainty-Guided Composed Image Retrieval with Fine-Grained Probabilistic Learning

**会议**: AAAI 2026 Oral  
**arXiv**: [2601.11393](https://arxiv.org/abs/2601.11393)  
**代码**: [https://github.com/tanghme0w/AAAI26-HUG](https://github.com/tanghme0w/AAAI26-HUG)  
**领域**: 多模态VLM  
**关键词**: Composed Image Retrieval, 不确定性建模, 概率嵌入, 细粒度匹配, 高斯表示

## 一句话总结

本文提出了HUG范式，通过细粒度高斯概率嵌入和异构不确定性估计（区分查询侧多模态协调不确定性与目标侧内容质量不确定性），结合动态加权融合和不确定性引导的对比学习，在Fashion-IQ和CIRR两个CIR基准上取得SOTA。

## 研究背景与动机

### 领域现状
组合图像检索（Composed Image Retrieval, CIR）是一个新兴的多媒体检索方向，用户通过"参考图像+修改文本"的多模态查询来搜索目标图像。这在电商和社交媒体中具有重要价值，用户可以表达"找一件和这件类似但颜色不同的连衣裙"这样的复杂视觉偏好。

### 现有痛点
CIR任务中存在固有的**数据噪声**问题：

**内容质量不确定性**：模糊图像、信息量不足的文本在训练数据中不可避免

**多模态协调不确定性**：即使图像和文本单独来看质量很高，它们之间的对应关系也可能模糊或不匹配（如文本描述的修改方向与参考图像不太相关）

### 核心矛盾
现有的概率学习方法应用到CIR时存在两个主要缺陷：
- **实例级粗粒度建模**：无法捕获CIR中复杂的细粒度用户意图（比如"颜色换红色但保持款式"涉及多个属性层面的匹配）
- **同质策略处理查询和目标**：现有方法对查询侧（多模态）和目标侧（单模态）采用相同的不确定性估计，忽略了查询侧特有的跨模态协调问题

### 本文切入角度
提出**异构不确定性引导（HUG）**范式：用细粒度高斯嵌入表示每个查询/目标，并针对多模态查询和单模态目标分别设计不同的不确定性估计策略，通过可证明的动态加权机制融合不同来源的不确定性。

## 方法详解

### 整体框架
HUG基于BLIP-2的Q-Former架构。每个查询和目标图像被表示为K=32个高斯嵌入的序列，每个高斯 $z_q^k \sim \mathcal{N}(\mu_q^k, \sigma_q^{k2} \mathbf{I})$ 描述一个细粒度的概念（如颜色、款式、logo等），方差反映该概念上的不确定性。

- **查询侧**：Q-Former接收参考图像（视觉backbone提取特征注入cross-attention）和修改文本，通过32个可学习query token输出均值 $\mu_q \in \mathbb{R}^{32 \times D}$
- **目标侧**：Q-Former接收目标图像（文本输入留空），输出 $\mu_c \in \mathbb{R}^{32 \times D}$
- 查询和目标共享同一个Q-Former权重

### 关键设计

#### 1. **异构不确定性估计**
这是本文最核心的设计——对查询侧和目标侧采用不同的不确定性建模策略。

**目标侧**（单模态，只关注内容质量）：
- 使用1层轻量Transformer作为方差估计器 $g_V$：$\sigma_c^2 = g_V(\mu_c)$
- 只需建模视觉内容质量和信息丰富度

**查询侧**（多模态，需额外考虑跨模态协调）：
- **参考图像不确定性** $\sigma_r^2 = g_V(h(x_{[LQ]}, \emptyset, x_r))$：视觉内容质量，与目标侧共享 $g_V$
- **修改文本不确定性** $\sigma_t^2 = g_T(h(x_{[LQ]}, x_t, \emptyset))$：文本修改的清晰度和具体性
- **多模态协调不确定性** $\sigma_m^2 = g_M(\mu_q)$：参考图像与修改文本之间的协调程度，需要同时看到两个模态才能估计

设计动机：跨模态协调不确定性来自图文之间的内在交互，不能简单地由单模态不确定性组合得到。例如"把颜色改一下"对一件纯色衣服不确定性低，但对一件花纹复杂的衣服不确定性高。

#### 2. **多模态协调损失**
为了让 $g_M$ 学到有意义的协调不确定性，引入排序损失：

$$\mathcal{L}_{\text{Cord.}} = -\mathbb{E}_{(x_r,x_t,x_c) \neq (x_r',x_t',x_c')} \log \mathcal{S}(\bar{\sigma}_m^2(x_r,x_t) - \bar{\sigma}_m^2(x_r,x_t'))$$

直觉：同一三元组内的图-文对应关系应比不同三元组之间更好，前者的协调不确定性应更低。

#### 3. **动态加权融合**
通过动态权重将三种不确定性融合为综合查询不确定性：

$$w_x^k[i] = \frac{\exp(-\sigma_x^{k}[i]^2)}{\sum_{x' \in \{r,t,m\}} \exp(-\sigma_{x'}^{k}[i]^2)}$$

不确定性越大的分量权重越小，实现"自信的分量占比更高"。论文提供了理论证明：在合理假设下，动态融合比任何静态权重都能获得更紧的泛化误差上界（Proposition 1 + Corollary 1）。

#### 4. **不确定性引导的对比学习**
- **整体对比** $\mathcal{L}_{\text{HC}}$：使用Sigmoid对比损失，距离度量是两个高斯之间的期望欧氏距离 $d(z_q, z_c) = \|\mu_q - \mu_c\|_F^2 + \|\sigma_q\|_F^2 + \|\sigma_c\|_F^2$（闭式解，无需采样）
- **细粒度对比** $\mathcal{L}_{\text{FC}}$：促进32个细粒度不确定性分量的正交性和多样性，采用三种负采样策略：
    - 分量级：同一侧/实例的其他分量
    - 实例级：同一侧但不同实例的分量
    - 模态级：另一侧的任意分量

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{HUG}} = \mathcal{L}_{\text{HC}} + \lambda_{\text{FC}} \mathcal{L}_{\text{FC}} + \lambda_{\text{Cord.}} \mathcal{L}_{\text{Cord.}}$$

默认超参：$\lambda_{\text{FC}} = 0.5$，$\lambda_{\text{Cord.}} = 0.1$。使用AdamW优化器，batch size=32，学习率 $3 \times 10^{-5}$，单卡A100-80G训练。

## 实验关键数据

### 主实验

**Fashion-IQ数据集**（核心结果）：

| 方法 | Dress R@10 | Shirt R@10 | Top R@10 | Avg R@10 | Avg R@50 | 总平均 |
|------|-----------|-----------|---------|---------|---------|--------|
| CLIP4CIR | 33.81 | 39.99 | 41.41 | 38.40 | 61.74 | 50.07 |
| FAME-ViL | 42.19 | 47.64 | 50.69 | 46.84 | 69.75 | 58.29 |
| QuRe | 46.80 | 53.53 | 57.47 | 52.60 | 73.48 | 63.04 |
| **HUG** | **48.37** | **51.62** | **58.26** | **52.75** | **74.73** | **63.74** |

**CIRR数据集**：

| 方法 | R@5 | R@10 | R_s@1 | (R@5+R_s@1)/2 |
|------|-----|------|-------|--------------|
| QuRe | 82.53 | 90.31 | 78.51 | 80.52 |
| **HUG** | **83.20** | **92.03** | **80.65** | **81.93** |

值得注意的是，HUG超过了使用额外数据（★标注）和LLM（♠标注）的方法，说明在合理的不确定性建模下，模型无需额外策划的标注或LLM增强。

### 消融实验

| 配置 | Avg R@10 | Avg R@50 | 总平均 | 说明 |
|------|---------|---------|--------|------|
| (0) 点匹配基线 | 41.15 | 63.38 | 52.26 | InfoNCE，无不确定性 |
| (1) +概率嵌入 | 45.00 | 65.89 | 55.44 | GPO全局不确定性，+3.18 |
| (4) +三种细粒度对比 | 49.42 | 69.24 | 59.33 | 细粒度逐步提升 |
| (6) +多模态协调损失 | 52.26 | 73.95 | 63.11 | 关键跳跃，+3.78 |
| (7) +动态加权 | **52.75** | **74.73** | **63.74** | 最终模型 |

关键发现：
- (5)仅朴素引入跨模态不确定性反而性能下降，但加上协调损失(6)后大幅提升→证明必须用专门的损失来解耦跨模态与单模态不确定性
- 推理时间：21.35ms/query vs 基线7.51ms，增加约3倍但仍可接受

### 关键发现

- 学到的不确定性具有**可解释性**：不同细粒度分量对应不同的子概念（颜色、logo、袖长等），不确定性大小与这些属性的模糊程度正相关
- 动态加权在理论和实验上都优于静态加权
- 协调损失是将跨模态不确定性有效利用的关键

## 亮点与洞察

1. **异构设计思想**：查询（多模态）和目标（单模态）在结构上不对称，理应采用不同的不确定性建模策略——这个洞察简单但有效
2. **细粒度概率表示**：用32个高斯（Q-Former的query token）天然映射到属性级细粒度，比实例级粗粒度表示更能捕获CIR中的复杂意图
3. **理论保障**：动态加权的泛化误差上界证明，不仅是工程技巧，还有理论支撑
4. **可解释性**：学到的不确定性分量可以映射到人类可理解的视觉概念

## 局限与展望

- 推理时间增加约3倍（21ms vs 7ms per query），在大规模检索场景中可能成为瓶颈
- 32个高斯分量的数量是预设的（来自Q-Former的设计），是否有最优的分量数量值得探索
- 只在supervised CIR上验证，是否适用于zero-shot CIR值得研究
- 动态加权的理论证明依赖于损失函数的凸性假设，换用其他损失可能需要重新分析

## 相关工作与启发

- 概率嵌入学习在cross-modal retrieval中已有应用（PCME, PCME++），但本文首次在CIR中引入并考虑了异构性
- Q-Former的32个query token作为细粒度表示的载体是一个巧妙的设计选择
- 不确定性引导的对比学习策略可以推广到其他多模态匹配任务

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Self-guided Semantic Inspection for Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/self-guided_semantic_inspection_for_zero-shot_composed_image_retrieval.md)
- [\[CVPR 2026\] Adapting In-context Generation for Enhanced Composed Image Retrieval](../../CVPR2026/multimodal_vlm/adapting_in-context_generation_for_enhanced_composed_image_retrieval.md)
- [\[CVPR 2026\] Air-Know: Arbiter-Calibrated Knowledge-Internalizing Robust Network for Composed Image Retrieval](../../CVPR2026/multimodal_vlm/air-know_arbiter-calibrated_knowledge-internalizing_robust_network_for_composed_.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](../../CVPR2026/multimodal_vlm/eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](../../ACL2026/multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)

</div>

<!-- RELATED:END -->
