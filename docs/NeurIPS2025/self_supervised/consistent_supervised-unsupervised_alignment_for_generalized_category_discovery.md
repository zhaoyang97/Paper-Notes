---
title: >-
  [论文解读] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery
description: >-
  [NeurIPS 2025][自监督学习][Generalized Category Discovery] 提出 NC-GCD 框架，通过预分配固定的 Equiangular Tight Frame (ETF) 原型为已知类和新类建立统一优化目标，结合语义一致性匹配器 (SCM) 稳定跨迭代伪标签分配，在 6 个 GCD 基准上显著提升新类发现精度。
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "Generalized Category Discovery"
  - "Neural Collapse"
  - "ETF Prototype"
  - "聚类对齐"
  - "伪标签一致性"
---

# Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery

**会议**: NeurIPS 2025  
**arXiv**: [2507.04725](https://arxiv.org/abs/2507.04725)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: Generalized Category Discovery, Neural Collapse, ETF Prototype, 聚类对齐, 伪标签一致性

## 一句话总结
提出 NC-GCD 框架，通过预分配固定的 Equiangular Tight Frame (ETF) 原型为已知类和新类建立统一优化目标，结合语义一致性匹配器 (SCM) 稳定跨迭代伪标签分配，在 6 个 GCD 基准上显著提升新类发现精度。

## 研究背景与动机

**领域现状**：广义类别发现 (GCD) 旨在同时对已知类别分类并发现新类别。现有方法如 SimGCD、CMS 采用动态学习聚类原型或分类器权重的方式来组织特征空间。

**现有痛点**：动态优化原型导致两个关键问题——(a) 已知类和新类的优化目标不一致，模型倾向偏向有标签的已知类而忽略新类的决策边界；(b) 特征分布缺乏几何约束，新类完全依赖无监督优化，容易与特征相似的类别重叠（类别混淆）。

**核心矛盾**：新类和已知类在优化目标上的不等性，使模型无法在统一几何结构下同等地分离所有类别。

**本文目标** 能否预分配一个最优几何结构，使已知类和新类在特征空间中被等间距分离，并通过统一对齐损失实现一致学习？

**切入角度**：Neural Collapse 理论表明训练良好的分类网络其类别特征均值收敛到 Simplex ETF 结构——最大化类间分离、最小化类内方差。作者将此理论从"训练后的涌现现象"转为"训练前的先验约束"。

**核心 idea**：预先固定 ETF 原型作为所有类别的锚点，通过有监督和无监督的统一对齐损失为 GCD 提供一致的优化方向。

## 方法详解

### 整体框架
NC-GCD 包含四个核心组件：(1) 预训练视觉编码器 $f(\cdot)$（DINO ViT-B/16），(2) 周期性聚类模块 $g(\cdot)$，(3) 预分配 ETF 原型集 $P$，(4) 语义一致性匹配器 $\phi_{\text{SCM}}(\cdot)$。输入图像经编码器提取 embedding，周期性聚类将所有样本分组，高置信样本被拉向对应 ETF 原型。

### 关键设计

1. **预分配 ETF 原型**:

    - 功能：训练前生成固定的等角紧框架原型 $\mathbf{P} = \{p_1, \dots, p_K\}$
    - 核心思路：ETF 通过 $P = \sqrt{\frac{K}{K-1}} U (I_K - \frac{1}{K} \mathbf{1}_K \mathbf{1}_K^\top)$ 构造，满足 $p_k^\top p_j = \frac{K}{K-1}\delta_{k,j} - \frac{1}{K-1}$，保证所有类别最大均匀分离
    - 设计动机：固定 ETF 提供全局最优几何配置，消除已知/新类优化不一致

2. **无监督 ETF 对齐**:

    - 功能：每 $T$ 个 epoch 聚类，选每簇 top-$\alpha\%$ 高置信样本对齐 ETF
    - 核心思路：高置信样本通过 Dot-Regression Loss 向原型对齐：$\mathcal{L}_{\text{ETF}}^u = \frac{1}{|\tilde{D}_k|} \sum_{e_i \in \tilde{D}_k} \|e_i - p_k\|^2$
    - 设计动机：只用高置信样本避免噪声伪标签干扰

3. **有监督 ETF 对齐**:

    - 功能：将有标签样本特征对齐到 SCM 映射后的 ETF 原型
    - 核心思路：$\mathcal{L}_{\text{ETF}}^s = \frac{1}{|\mathcal{D}^l|} \sum \|e_i^l - p_a\|^2$，$a = \phi_{\text{SCM}}(y_i^l)$
    - 设计动机：需 SCM 保证真实标签到 ETF 的映射正确性

4. **语义一致性匹配器 (SCM)**:

    - 功能：保证跨聚类迭代的伪标签一致性
    - 核心思路：最优排列 $\sigma^* = \arg\max_{\sigma \in S_K} \sum_{k} \sum_{i} \mathbb{I}(\hat{y}_i^t = k)\mathbb{I}(\hat{y}_i^{t-1} = \sigma(k))$，用匈牙利算法实现一对一标签映射
    - 设计动机：周期性聚类不稳定，SCM 通过强制一对一匹配消除波动

### 损失函数 / 训练策略

统一 ETF 损失：$\mathcal{L}_{\text{ETF}} = (1-\gamma)\mathcal{L}_{\text{ETF}}^u + \gamma\mathcal{L}_{\text{ETF}}^s$

表征学习：$\mathcal{L}_{\text{REP}} = (1-\lambda)\mathcal{L}_{\text{REP}}^u + \lambda\mathcal{L}_{\text{REP}}^s$

最终：$\mathcal{L} = \beta\mathcal{L}_{\text{ETF}} + \mathcal{L}_{\text{REP}}$

## 实验关键数据

### 主实验（DINOv1, GT K 已知）

| 数据集 | NC-GCD (All/Old/New) | SPT (All/Old/New) | CMS (All/Old/New) |
|--------|---------------------|-------------------|-------------------|
| CUB-200 | 74.8/76.8/73.8 | 65.8/68.8/65.1 | 68.2/76.5/64.0 |
| Stanford Cars | 59.9/77.8/51.2 | 59.0/79.2/49.3 | 56.9/76.1/47.6 |
| FGVC Aircraft | 60.0/57.6/61.2 | 59.3/61.8/58.1 | 56.0/63.4/52.3 |
| ImageNet-100 | 88.4/94.1/85.5 | 85.4/93.2/81.4 | 84.7/95.6/79.2 |
| CIFAR-100 | 82.7/85.5/77.3 | 81.3/84.3/75.6 | 82.3/85.7/75.5 |

### 综合平均（GT K 已知）

| 方法 | 细粒度 All/New | 全部 All/New |
|------|---------------|-------------|
| SPT | 56.9/51.9 | 65.7/60.8 |
| CMS | 54.4/47.6 | 64.1/57.5 |
| **NC-GCD** | **60.3/56.7** | **68.7/64.9** |

### 关键发现
- 新类精度提升最显著：细粒度数据集新类准确率平均 +4.8%（vs SPT），固定 ETF 有效缓解新类欠分离
- 无 GT K 时仍鲁棒，ETF 框架对 K 估计误差有容忍度
- ImageNet-100 全类 88.4%，比次优 SPT 高 3.0%

## 亮点与洞察
- **Neural Collapse 从现象到先验**：将 NC 从训练后涌现转为训练前结构约束，可迁移到增量学习和联邦学习
- **SCM 匈牙利匹配**：用最优分配解决跨迭代伪标签漂移，简洁有效

## 局限与展望
- 需预估类别数 K，K 偏差影响 ETF 几何质量
- 仅 DINO ViT-B/16 验证，未探索 DINOv2 或更大模型
- 高置信阈值 $\alpha$ 需手动调节

## 相关工作与启发
- **vs SimGCD**: SimGCD 动态学习原型导致已知/新类目标不一致；NC-GCD 固定 ETF 消除此问题
- **vs TRAILER**: TRAILER 也用固定分类器但交叉熵 ETF 损失可能引入偏差；NC-GCD 分离监督/无监督对齐
- **vs CMS**: CMS 聚焦对比均值漂移，NC-GCD 同时优化特征几何

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 Neural Collapse 引入 GCD 是新视角，但 ETF 在其他领域已有应用
- 实验充分度: ⭐⭐⭐⭐ 6 个基准、两种 K 设定、10+ 方法对比
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，公式符号统一
- 价值: ⭐⭐⭐⭐ 为 GCD 提供结构化几何先验，新类精度提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](../../ICCV2025/self_supervised/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](../../CVPR2025/self_supervised/mos_modeling_object-scene_associations_in_generalized_category_discovery.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)

</div>

<!-- RELATED:END -->
