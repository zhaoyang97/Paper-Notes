---
title: >-
  [论文解读] Heavy Labels Out! Dataset Distillation with Label Space Lightening
description: >-
  [ICCV 2025][模型压缩][数据集蒸馏] 提出 HeLlO 框架，利用 CLIP 预训练模型和 LoRA-like 低秩知识迁移构建轻量级图像-标签投影器，将数据集蒸馏中软标签的存储需求降低至原来的 0.003%，同时保持甚至超越 SOTA 性能。 数据集蒸馏旨在将大规模训练集压缩为极小的合成集…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "数据集蒸馏"
  - "软标签压缩"
  - "CLIP"
  - "LoRA"
  - "标签空间轻量化"
---

# Heavy Labels Out! Dataset Distillation with Label Space Lightening

**会议**: ICCV 2025  
**arXiv**: [2408.08201](https://arxiv.org/abs/2408.08201)  
**代码**: 即将公开  
**领域**: 模型压缩 / Dataset Distillation  
**关键词**: 数据集蒸馏, 软标签压缩, CLIP, LoRA, 标签空间轻量化

## 一句话总结

提出 HeLlO 框架，利用 CLIP 预训练模型和 LoRA-like 低秩知识迁移构建轻量级图像-标签投影器，将数据集蒸馏中软标签的存储需求降低至原来的 0.003%，同时保持甚至超越 SOTA 性能。

## 研究背景与动机

数据集蒸馏旨在将大规模训练集压缩为极小的合成集。当前 SOTA 方法（SRe2L、G_VBSM、RDED）虽然大幅减少了图像数量，但严重依赖预训练教师模型生成的大量软标签来保持性能。

**核心问题**：软标签的存储开销巨大，甚至可与原始数据集相当。例如：
- ImageNet-1K, IPC=1：图像仅 ~15MB，但软标签超过 572MB（38 倍）
- ImageNet-1K, IPC=200：软标签达 110GB，与原始数据集相当
- 原因：每次数据增强都会生成独立的 C 维软标签（C=类别数），总量 = K（迭代数）× N_s（样本数）× C

这揭示了当前蒸馏方法的一个被忽视的瓶颈：**蒸馏了图像但没有蒸馏标签**。

## 方法详解

### 整体框架

HeLlO 框架用一个轻量级的在线投影器替代离线存储的海量软标签：
1. 基于 CLIP 图像编码器 + 线性变换构建投影器
2. 用文本嵌入初始化线性变换部分（零存储成本）
3. LoRA-like 低秩矩阵微调投影器至目标分布
4. 可选的图像更新以减少投影器误差
5. 下游训练时在线生成软标签

### 关键设计

1. **基于文本嵌入的投影器初始化**:

    - 利用 CLIP 的视觉-语言对齐能力，用各类别文本描述的归一化嵌入初始化线性变换 $W = (v_T)^T$
    - 数学等价性证明：文本嵌入初始化等价于预训练零样本分类（Proposition 1）
    - 无需额外存储（文本描述由固定 prompt 模板生成）
    - 设计动机：提供强起点，使投影器从预训练零样本能力出发进一步适配

2. **LoRA-Like 低秩知识迁移**:

    - 分解权重增量 $\Delta\theta = A \cdot B$，其中 $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, $r \ll d, k$
    - 同时对 CLIP 图像编码器的卷积层和线性变换部分应用 LoRA（不同 rank）
    - 训练目标结合多弱教师知识蒸馏和交叉熵：
    $\mathcal{L}(\mathcal{D};\theta) = MSE(f_\theta(X), Y') + \lambda CE(f_\theta(X), Y)$
    - 弱教师来自 ResNet-18 训练轨迹的不同阶段（9 个 checkpoint）
    - 设计动机：最小化微调成本的同时缩小预训练分布与目标分布的差距

3. **合成数据集初始化与更新**:

    - 初始化遵循 RDED：根据教师模型评估的难度选择最具代表性的图像 patch 并拼接
    - 额外的图像更新步骤：最小化原始分辨率与下采样再上采样版本在 CLIP 特征空间的差异
    $\mathcal{G}(\mathcal{E}_I, p) = MSE(\mathcal{E}_I(p), \mathcal{E}_I(\hat{p}))$
    - 设计动机：由于使用了替代投影器而非原始教师，需要更新图像以减少投影器上的信息损失

### 损失函数 / 训练策略

下游训练时的损失函数：
$$\phi^e = \phi^{e-1} - \alpha \nabla_\phi (MSE(f_\phi(\mathcal{A}(X_s)), Y^*) + \beta CE(f_\phi(\mathcal{A}(X_s)), Y_s))$$
- $Y^*$ 由投影器在线生成（而非预存储）
- 配置：ImageNet-100 rank=8/64，ImageNet-1K rank=8/128

## 实验关键数据

### 主实验（ResNet-18 Top-1 Accuracy, %）

| 数据集 | IPC | SRe2L | G_VBSM | RDED | **HeLlO** | 标签存储比 |
|--------|-----|-------|--------|------|-----------|-----------|
| IN-100 | 1 | 3.0 | - | 8.1 | **12.5 (+4.4)** | 0.1× |
| IN-100 | 10 | 9.5 | - | 36.0 | **48.9 (+12.9)** | 0.01× |
| IN-100 | 50 | 27.0 | - | 61.6 | **69.4 (+7.8)** | 0.002× |
| IN-1K | 1 | 0.1 | 1.7 | 6.6 | **12.9 (+6.3)** | 1e-4× |
| IN-1K | 10 | 21.3 | 31.4 | 42.0 | **43.7 (+1.7)** | 1e-5× |
| IN-1K | 50 | 46.8 | 51.8 | **56.5** | 52.2 (-) | 3e-6× |

教师模型参数量：RDED 10.7M vs HeLlO 仅 0.8M (0.07×)

### 消融实验

跨架构泛化（IN-1K, IPC=10）：

| 架构 | RDED | HeLlO | 提升 |
|------|------|-------|------|
| ShuffleNet-V2 | 23.3 | 26.5 | +3.2 |
| MobileNet-V2 | 34.4 | 38.1 | +3.7 |
| EfficientNet-B0 | 42.8 | 44.4 | +1.6 |
| Swin-V2-Tiny | 17.8 | **29.5** | **+11.7** |
| VGG-11 | 22.7 | 24.2 | +1.5 |

各组件增量消融（IN-1K, IPC=10）：

| 配置 | Acc. | #Params |
|------|------|---------|
| Probe Linear CLIP | 28.2 | 1.0M |
| + Multi-Weak-Teacher | 30.1 (+1.9) | 1.0M |
| + LoRA Knowledge Transfer | 43.5 (+13.4) | 1.5M |
| + Text-Embedding Init | 43.6 (+0.1) | 0.8M (↓0.7M) |
| + Image Update | 43.7 (+0.1) | 0.8M |

### 关键发现

- **LoRA 知识迁移是最关键组件**：带来 +13.4% 的巨大提升，将预训练嵌入有效适配到目标分布
- **文本嵌入初始化双重作用**：虽然精度提升仅 0.1%，但实际上减少了 0.7M 参数存储（不需要存储初始线性变换参数）
- **HeLlO 在小 IPC 和 Transformer 架构上优势最显著**：IN-100 IPC=10 超 RDED 12.9%；Swin-V2-Tiny 上超 RDED 11.7%
- **大规模场景有局限**：IN-1K IPC=50 时 HeLlO 低于 RDED 4.3%，说明投影器在极大标签空间中精度不足

## 亮点与洞察

- **问题定义精准**：首次聚焦数据集蒸馏中被忽视的"标签膨胀"问题，指出蒸馏了图像却没蒸馏标签
- **巧妙利用 CLIP 的视觉-语言对齐**：文本嵌入初始化实现了零额外存储成本的强起点
- **极致压缩比**：0.003% 的标签存储即可获得可比性能
- **LoRA 在新场景的应用**：将 LoRA 从 LLM 微调推广到数据集蒸馏中的投影器构建
- **跨架构泛化**：特别是在 Transformer 架构上的显著优势值得关注

## 局限与展望

- 大规模 + 大 IPC 场景（IN-1K IPC=50）性能仍不及 RDED，投影器精度有限
- 依赖 CLIP 预训练模型，对 CLIP 覆盖不好的领域（如医学影像）效果可能受限
- 弱教师的选择（训练轨迹中的 checkpoint 阶段）是超参数，需要针对不同 IPC 调整
- 图像更新步骤的增益较小（仅 +0.1%），成本效益比需评估
- 投影器实际推理时需要调用 CLIP 编码器，下游训练增加了在线计算开销

## 相关工作与启发

- RDED 的 patch 选择+拼接策略提供了有效的图像蒸馏基础
- LoRA 的低秩分解思想在知识压缩场景中的应用潜力
- 利用预训练模型的视觉-语言对齐能力进行标签空间重建是有前景的方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次关注软标签存储问题，解决方案巧妙但核心是已有技术的组合
- **实验充分度**: ⭐⭐⭐⭐ 多数据集、多架构、完整消融，但缺少更多大规模场景验证
- **写作质量**: ⭐⭐⭐⭐ 问题和方法阐述清晰，数学推导完善
- **价值**: ⭐⭐⭐⭐ 解决了数据集蒸馏中的实际瓶颈，对大规模蒸馏有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](dataset_distillation_via_the_wasserstein_metric.md)
- [\[CVPR 2026\] Rethinking Dataset Distillation: Hard Truths about Soft Labels](../../CVPR2026/model_compression/rethinking_dataset_distillation_hard_truths_about_soft_labels.md)
- [\[CVPR 2026\] Beyond Soft Label: Dataset Distillation via Orthogonal Gradient Matching](../../CVPR2026/model_compression/beyond_soft_label_dataset_distillation_via_orthogonal_gradient_matching.md)
- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](../../CVPR2025/model_compression/enhancing_dataset_distillation_via_non-critical_region_refinement.md)

</div>

<!-- RELATED:END -->
