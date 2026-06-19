---
title: >-
  [论文解读] LEMoN: Label Error Detection using Multimodal Neighbors
description: >-
  [ICML 2025][多模态VLM][标签错误检测] 本文提出 LEMoN 方法，利用对比预训练多模态模型（如 CLIP）的嵌入空间中图像-文本对的多模态邻域结构，在分类和图像描述两个场景下自动检测标签错误，在训练无关的基线中 F1 提升 3-4%，过滤后的数据可改善下游分类和描述性能。 领域现状： 现代视觉-语言模型的训…
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "标签错误检测"
  - "多模态噪声标签"
  - "对比学习嵌入"
  - "最近邻方法"
  - "图像描述"
---

# LEMoN: Label Error Detection using Multimodal Neighbors

**会议**: ICML 2025  
**arXiv**: [2407.18941](https://arxiv.org/abs/2407.18941)  
**代码**: 有（未公开链接）  
**领域**: 多模态VLM  
**关键词**: 标签错误检测, 多模态噪声标签, 对比学习嵌入, 最近邻方法, 图像描述

## 一句话总结
本文提出 LEMoN 方法，利用对比预训练多模态模型（如 CLIP）的嵌入空间中图像-文本对的多模态邻域结构，在分类和图像描述两个场景下自动检测标签错误，在训练无关的基线中 F1 提升 3-4%，过滤后的数据可改善下游分类和描述性能。

## 研究背景与动机

**领域现状**: 现代视觉-语言模型的训练依赖海量的图像-文本对数据集（如 LAION-400M、CC-12M），这些数据大多从网络爬取，不可避免地包含大量**错误标签**——图像与描述不匹配。标签错误会降低下游模型的可靠性，在医疗等关键领域尤为严重。

**现有痛点**:
   - 大多数标签错误检测方法是**单模态的**：仅利用图像表示进行检测，忽略了文本信息
   - 部分高性能方法（如 AUM、Datamap）需要在下游任务上**训练分类器数个 epoch**，计算代价高
   - 现有方法假设标签是"k 选一"的离散类别，**无法处理自然语言标签**（如图像描述）
   - 最简单的 CLIP 相似度方法虽然免训练，但忽略了邻域结构中的丰富信息

**核心矛盾**: 数据集规模越大越难人工审核 → 需要自动检测 → 但现有自动方法要么需要昂贵训练，要么局限于单模态和离散标签。

**本文目标**: 提出一种**免训练、利用多模态邻域信息**的标签错误检测方法，同时适用于分类标签和自然语言描述。

**切入角度**: 在 CLIP 嵌入空间中，正确标签的图像-文本对应该有相似的邻居（邻居的图像对应的文本也应该和当前文本相似），而错误标签的对应关系会在邻域中暴露出不一致。

**核心 idea**: 结合图像-文本多模态距离和两个方向的跨模态邻域信息，构建错误标签检测分数。

## 方法详解

### 整体框架
给定数据集 $\mathcal{D} = \{(\mathbf{x}, \mathbf{y})_i\}_{i=1}^N$（图像-文本对），LEMoN 输出每个样本的"错误标签分数" $s$。核心流程：
1. 用预训练 CLIP 编码所有图像和文本
2. 对每个样本，计算三个分数分量并线性组合
3. 分数越高，越可能是错误标签

### 关键设计

1. **多模态距离 $d_{mm}$（基础分数）**:

    - 直接计算图像嵌入和文本嵌入的余弦距离：
    $d_{mm}(\mathbf{x}, \mathbf{y}) = d_{cos}(h_\theta^\mathcal{X}(\mathbf{x}), h_\theta^\mathcal{Y}(\mathbf{y}))$
    - 这就是 CLIP Similarity baseline——距离越大越可能是错误标签
    - **设计动机**: 这是最基本也最直接的信号，已被先前工作验证有效。LEMoN 以此为基础，在其上增加邻域信息

2. **图像空间邻域分数 $s_n$**:

    - 找到 $\mathbf{x}$ 在图像嵌入空间的 $k$ 个最近邻 $\{\mathbf{x}_{n1}, \ldots, \mathbf{x}_{nk}\}$
    - 计算当前文本 $\mathbf{y}$ 与这些邻居对应文本 $\mathbf{y}_{nj}$ 的距离，加权平均：
    $s_n(\mathbf{x}, \mathbf{y}, \mathcal{D}) = \frac{1}{k} \sum_{j=1}^k d_\mathcal{Y}(\mathbf{y}, \mathbf{y}_{nj}) \cdot e^{-\tau_{1,n} d_\mathcal{X}(\mathbf{x}, \mathbf{x}_{nj})} \cdot e^{-\tau_{2,n} d_{mm}(\mathbf{x}_{nj}, \mathbf{y}_{nj})}$
    - **直觉**: 如果我的图像和邻居的图像很像，但我的文本和邻居的文本差距很大——说明我的标签很可能是错的
    - **权重设计**:
        - $e^{-\tau_{1,n} d_\mathcal{X}}$: 降权距离远的邻居（自适应 $k$）
        - $e^{-\tau_{2,n} d_{mm}}$: 降权邻居本身可能也是错误标签的情况

3. **文本空间邻域分数 $s_m$**:

    - 找到 $\mathbf{y}$ 在文本嵌入空间的 $k$ 个最近邻 $\{\mathbf{y}_{m1}, \ldots, \mathbf{y}_{mk}\}$
    - 计算当前图像 $\mathbf{x}$ 与这些邻居对应图像 $\mathbf{x}_{mj}$ 的距离：
    $s_m(\mathbf{x}, \mathbf{y}, \mathcal{D}) = \frac{1}{k} \sum_{j=1}^k d_\mathcal{X}(\mathbf{x}, \mathbf{x}_{mj}) \cdot e^{-\tau_{1,m} d_\mathcal{Y}(\mathbf{y}, \mathbf{y}_{mj})} \cdot e^{-\tau_{2,m} d_{mm}(\mathbf{x}_{mj}, \mathbf{y}_{mj})}$
    - **直觉**: 如果与我文本描述相似的其他文本对应的图像和我的图像差距很大——也说明标签错误
    - **设计动机**: 与 $s_n$ 互补——$s_n$ 从图像邻域出发，$s_m$ 从文本邻域出发，两个方向的信号共同增强检测

4. **最终分数**:
    $s = f(\mathbf{x}, \mathbf{y}) = d_{mm}(\mathbf{x}, \mathbf{y}) + \beta \cdot s_n(\mathbf{x}, \mathbf{y}, \mathcal{D}) + \gamma \cdot s_m(\mathbf{x}, \mathbf{y}, \mathcal{D})$
    - $\beta, \gamma \geq 0$ 是超参数
    - **泛化性**: 当 $\beta = \gamma = 0$，退化为 CLIP Similarity；当 $\beta$ 大、$\gamma = 0$ 且使用离散距离，退化为 Deep k-NN

### 损失函数 / 训练策略
- **LEMoN 本身完全免训练**，仅需预训练 CLIP 模型
- 两种配置：
    - **LEMoN$_{opt}$**: 在标注验证集上搜索最优超参 $k, \beta, \gamma, \tau_{1,n}, \tau_{2,n}, \tau_{1,m}, \tau_{2,m}$
    - **LEMoN$_{fix}$**: 使用固定合理超参（$k=30, \beta=\gamma=5, \tau_1=0.1, \tau_2=5$），无需验证集
    - 两者差距仅 ~1.7% AUROC

## 实验关键数据

### 主实验（标签错误检测 - 分类场景）

| 数据集 | 方法 | 需训练? | AUROC (%) | AUPRC (%) | F1 (%) |
|--------|------|---------|-----------|-----------|--------|
| CIFAR-10 (human noise) | AUM | 是 | 98.3 | 97.9 | 94.0 |
| | Datamap | 是 | 98.2 | 97.6 | 93.4 |
| | CLIP Sim. | 否 | 93.8 | 92.4 | 86.9 |
| | Deep k-NN | 否 | 96.2 | 93.8 | 89.3 |
| | **LEMoN$_{opt}$** | **否** | **98.1** | **97.4** | **93.1** |
| CIFAR-100 (human noise) | AUM | 是 | 92.2 | 89.9 | 83.9 |
| | CLIP Sim. | 否 | 78.5 | 72.1 | 69.2 |
| | **LEMoN$_{opt}$** | **否** | **90.8** | **87.4** | **81.3** |

### 主实验（标签错误检测 - 描述场景）

| 数据集 | 方法 | AUROC (%) | AUPRC (%) | F1 (%) |
|--------|------|-----------|-----------|--------|
| MSCOCO | CLIP Sim. | 93.8 | 93.0 | 87.5 |
| | LLaVA | 80.3 | 63.4 | 74.9 |
| | **LEMoN$_{opt}$** | **95.6** | **94.6** | **89.3** |
| MIMIC-CXR | CLIP Sim. | 64.1 | 51.7 | 48.6 |
| | **LEMoN$_{opt}$** | **70.4** | **60.3** | **57.0** |

### 消融实验

| 配置 | mmimdb AUROC | mscoco AUROC | 说明 |
|------|-------------|-------------|------|
| 完整 LEMoN | 86.0% | 95.6% | 全部分量 |
| 去掉 $\tau_1, \tau_2$ | 85.3% (-0.7) | 94.9% (-0.7) | 自适应权重有贡献 |
| 去掉 $s_n$ (图像邻域) | 85.4% (-0.6) | 94.6% (-1.0) | 文本邻域更重要（mmimdb） |
| 去掉 $s_m$ (文本邻域) | 86.1% (-指) | 94.7% (-0.9) | 图像邻域更重要（mscoco） |
| 仅 $d_{mm}$ (CLIP Sim.) | 85.1% (-0.9) | 93.8% (-1.8) | 邻域整体贡献 ~1-2% |

### 下游过滤效果

| 数据集 | 方法 | BLEU-4 | CIDER | ROUGE |
|--------|------|--------|-------|-------|
| MSCOCO | 不过滤 (40% noise) | 27.5 | 54.3 | 36.5 |
| | CLIP Sim. 过滤 | 31.1 | 64.8 | 39.8 |
| | **LEMoN$_{opt}$ 过滤** | **31.4** | **65.4** | **40.1** |
| | 干净数据 (上限) | 32.0 | 66.3 | 40.4 |

### 关键发现
1. **免训练方法逼近需训练方法**: LEMoN$_{opt}$ 在 CIFAR-10 上 AUROC 98.1% 仅比 AUM 98.3% 低 0.2%，但完全不需要训练分类器
2. **在描述场景大幅领先**: 在 MSCOCO 上比 CLIP Sim. 提升 1.8% AUROC 和 1.8% F1
3. **LEMoN$_{fix}$ 无需验证集仍强**: 固定超参版本平均仅损失 1.7% AUROC
4. **每个数据集的模态依赖不同**: mmimdb 更依赖文本邻域（电影海报+情节摘要），mscoco 更依赖图像邻域
5. **过滤后几乎恢复到干净数据性能**: MSCOCO 上 LEMoN 过滤后 CIDER 65.4 vs 干净 66.3

## 亮点与洞察
- **方法优雅而通用**: 一个统一框架同时处理分类和描述场景，且泛化到多种数据集（自然图像/电影海报/胸片X光）
- **理论支撑扎实**: Proposition 4.1 证明 CLIP 损失对噪声标签具有 Lipschitz 鲁棒性；Proposition 4.2 证明对比学习嵌入天然可以区分正确/错误标签
- **实用性强**: LEMoN$_{fix}$ 版本完全不需要标注验证集，只需要预训练 CLIP 即可使用
- **Medical domain 验证**: 在 MIMIC-CXR（胸片X光+放射报告）上也有效，仅用 noisy data 从头训练 CLIP 即可替代域外预训练

## 局限与展望
- 在 MIMIC-CXR 等专业领域上效果提升相对有限（~6% AUROC 提升），说明域外 CLIP 的嵌入质量是瓶颈
- 未测试实例依赖（instance-dependent）噪声——一种更现实但更难的噪声类型
- 真实标签错误存在模糊性和主观性，二值化的"正确/错误"假设可能过于简化
- 超参搜索空间较大（7 个超参），即使 LEMoN$_{fix}$ 有效，其最优值的可迁移性有待更多验证
- 未探索将 LEMoN 分数与下游训练循环整合（如自适应过滤）

## 相关工作与启发
- LEMoN 是 Deep k-NN 和 CLIP Similarity 的自然推广，通过多模态邻域统一了两种方向
- 对图像描述质量控制具有直接应用价值——可在大规模数据集构建pipeline中用于自动过滤
- 启发：在其他多模态任务（如视频-文本、音频-文本）中，类似的邻域方法可能同样有效
- 域内 CLIP 从头训练（即使在 noisy data 上）竟优于大规模域外预训练，值得深思

## 评分
- 新颖性: ⭐⭐⭐⭐ 将多模态邻域信息引入标签错误检测，理论推导支撑设计
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、12个基线、理论分析、消融彻底、真实世界验证
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、理论+实验+消融+真实验证层层递进
- 价值: ⭐⭐⭐⭐ 方法实用、泛化性好，对数据质量控制有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](../../ACL2025/multimodal_vlm/error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ICML 2025\] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning](lada_scalable_label-specific_clip_adapter_for_continual_learning.md)
- [\[NeurIPS 2025\] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](../../NeurIPS2025/multimodal_vlm/unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](../../ACL2025/multimodal_vlm/unsolvable_problem_detection.md)

</div>

<!-- RELATED:END -->
