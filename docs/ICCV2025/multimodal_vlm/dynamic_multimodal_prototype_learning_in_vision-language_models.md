---
title: >-
  [论文解读] Dynamic Multimodal Prototype Learning in Vision-Language Models
description: >-
  [ICCV 2025][多模态VLM][test-time adaptation] 提出 ProtoMM，一个 training-free 的多模态原型学习框架，通过将原型建模为文本描述和视觉粒子的离散分布，利用最优传输动态更新多模态原型，在 15 个 zero-shot 基准上达到 SOTA。 预训练视觉语言模型（如 CL…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "test-time adaptation"
  - "CLIP"
  - "多模态"
  - "optimal transport"
  - "Zero-Shot Classification"
---

# Dynamic Multimodal Prototype Learning in Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2507.03657](https://arxiv.org/abs/2507.03657)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: test-time adaptation, CLIP, Multimodal Prototype, optimal transport, Zero-Shot Classification

## 一句话总结

提出 ProtoMM，一个 training-free 的多模态原型学习框架，通过将原型建模为文本描述和视觉粒子的离散分布，利用最优传输动态更新多模态原型，在 15 个 zero-shot 基准上达到 SOTA。

## 研究背景与动机

预训练视觉语言模型（如 CLIP）在零样本分类中表现优异，但类别名称的歧义性限制了文本原型的判别力：

**词汇歧义**：如 "sword lily" 和 "blackberry lily" 都包含 "lily" 且余弦相似度高达 0.67，难以区分

**语义歧义**：如 "laptop" 和 "desktop computer" 无共同词但余弦相似度 0.69，因为都属于计算机

现有方法（TPT、TDA、AWT 等）仅从文本域构建原型，忽视了视觉信息能提供互补的判别线索。将测试图像的视觉特征融入原型可显著减少歧义——作者展示随着测试流推进，多模态原型与真实分布的 KL 散度从 18.7 下降到 9.5。

## 方法详解

### 整体框架

ProtoMM 由两个模块组成：
1. **分布式特征构建**（Distributed Feature Construction）：将图像和原型建模为离散分布
2. **多模态原型学习**（Multimodal Prototype Learning）：通过最优传输动态更新原型

### 关键设计

1. **分布式特征构建**：

    - 将测试图像增强为 $N$ 个视图（随机裁剪/翻转/缩放），分布表示为 $P_t = \sum_{n=1}^{N} a_t^n \delta_{\mathbf{x}_t^n}$
    - 通过 LLM（GPT-3.5）为每个类生成 $M$ 个文本描述，扩展为 $S$ 个视觉粒子
    - 多模态原型：$Q_c = \sum_{m=1}^{M} w_c^m \delta_{\mathbf{z}_c^m} + \sum_{s=1}^{S} w_c^{M+s} \delta_{\mathbf{e}_c^s}$
    - 视觉粒子初始化为文本描述特征的均值，随测试流动态更新
    - 重要性权重基于负熵计算：高置信度增强（与原型匹配好的）获得更高权重

2. **基于最优传输的预测**：

    - 构建代价矩阵 $\mathbf{C}_{tc}$ 为视觉增强与多模态原型间的余弦距离
    - 通过 Sinkhorn 算法求解带熵正则化的 OT 问题
    - 预测概率 $p(y_t=c|\mathbf{x}_t) \propto \exp(-d_{\text{OT}}(P_t, Q_c))$
    - OT 距离比点对点的余弦相似度更精确地衡量分布间距离

3. **动态原型更新**：

    - 对预测置信度 $\geq \tau$ 的高质量样本，利用传输计划 $\mathbf{T}_{tc}$ 计算增强得分 $\Theta_t = \mathbf{T}_{tc} \mathbf{w}_{y_t}$
    - 选取 top-$S$ 个高得分增强作为候选
    - 加权移动平均更新视觉缓存：$\mathbf{e}_c^s \leftarrow \frac{w_t^{M+s} \mathbf{e}_c^s + \theta_t^{(s)} \mathbf{x}_t^{(s)}}{w_t^{M+s} + \theta_t^{(s)}}$
    - 随着测试流推进，多模态原型持续积累更多视觉先验

### 损失函数 / 训练策略

此方法为 **training-free**，无需任何梯度更新或优化。核心参数：
- 增强数量 $N = M = 50$
- 视觉粒子数 $S = 25$
- 置信度阈值 $\tau = 0.8$

## 实验关键数据

### 主实验（11 数据集 cross-domain benchmark, ViT-B/16）

| 方法 | Aircraft | Caltech | Cars | DTD | EuroSAT | Flower | Food | Pets | SUN | UCF | ImageNet | Avg |
|------|----------|---------|------|-----|---------|--------|------|------|-----|-----|----------|-----|
| CLIP | 23.22 | 93.55 | 66.11 | 45.04 | 50.42 | 66.99 | 82.86 | 86.92 | 65.63 | 65.16 | 68.34 | 64.93 |
| TDA | 23.91 | 94.24 | 67.28 | 47.40 | 58.00 | 71.42 | 86.14 | 88.63 | 67.62 | 70.66 | 69.51 | 67.71 |
| AWT | 29.22 | 95.40 | 69.80 | 55.56 | 58.40 | 75.07 | 85.54 | 92.23 | 70.00 | 70.70 | 71.26 | 70.28 |
| **ProtoMM** | **31.02** | **95.70** | **69.92** | **56.38** | 56.11 | **77.40** | 85.89 | 91.90 | **70.78** | **71.76** | **72.01** | **70.70** |

### 消融实验

| 模块 | Eq.(5) 分布特征 | Eq.(10) 原型学习 | ImageNet (RN50) | ImageNet (ViT) | Caltech (ViT) |
|------|:---:|:---:|:---:|:---:|:---:|
| CLIP baseline | ✗ | ✗ | 59.81 | 68.34 | 93.55 |
| + 分布特征 | ✔ | ✗ | 60.82 | 68.98 | 94.65 |
| + 多模态原型 | ✔ | ✔ | **63.76** | **72.01** | **95.70** |

**OOD 泛化实验（ViT-B/16）**：

| 方法 | ImageNet | ImageNet-A | ImageNet-V2 | ImageNet-R | ImageNet-S | OOD Avg |
|------|----------|------------|-------------|------------|------------|---------|
| DOTA | 70.68 | 61.19 | 64.41 | 81.17 | 51.33 | 64.52 |
| DPE | 71.91 | 59.63 | 65.44 | 80.44 | 52.26 | 64.44 |
| **ProtoMM** | **72.01** | 64.02 | **65.93** | **80.87** | **51.97** | **65.69** |

### 关键发现

- 多模态原型带来的提升在细粒度数据集（Flowers +2.3%）和 OOD 场景上尤为显著
- 视觉增强数 40+ 和文本增强数 40+ 后性能趋于稳定
- 置信度阈值 $\tau = 0.8$ 是最优值，太高（缺少样本更新）或太低（引入噪声）都降低性能
- 传输计划热力图显示模型有效聚焦于目标物体区域
- 推理时间与 TDA 相当，远快于 TPT/DiffTPT 等需要梯度反传的方法

## 亮点与洞察

- **多模态原型的核心洞察**：文本原型天然存在歧义（语义相近的类名难区分），视觉特征可提供互补信息。随着测试流推进，多模态原型越来越准确——这是一种在线学习范式
- **最优传输的妙用**：传统方法将图像和原型视为单点计算相似度，OT 将它们视为分布间的距离度量，更好地处理多视角和多描述的情况
- **Training-free 设计**：无需反向传播，无需微调任何参数，推理效率高
- 分布特征构建和原型更新的贡献可分离验证，消融清晰

## 局限与展望

- 对 LLM 生成的文本描述质量有依赖，不同 LLM 生成的描述可能影响效果
- 视觉粒子通过移动平均更新，长期测试流中可能出现概念漂移
- 类别数非常多时（ImageNet 1000 类），OT 计算的开销可能增大
- 仅在分类任务上验证，未扩展到检测/分割等下游任务
- 阈值 $\tau$ 和 top-$S$ 等超参数需要预设

## 相关工作与启发

- **AWT**：最接近的工作，也将图像-文本距离建模为 OT 问题，但仅限于文本域原型对齐；ProtoMM 通过引入视觉粒子将其扩展为多模态
- **TDA / DOTA**：基于正负缓存的 training-free 方法，但需要 logit 融合且对融合超参数敏感
- **Sinkhorn 算法**：高效求解 OT 的核心工具，天然支持并行计算

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多模态原型+OT+动态更新的组合有新意，但各组件均非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 15 数据集+双 backbone+全面消融+OOD 评估+推理时间+可视化
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，图示直观（分布演化、传输热力图）
- **价值**: ⭐⭐⭐⭐ Training-free TTA 方向的有力推进，多模态原型思路有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[CVPR 2026\] Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting](../../CVPR2026/multimodal_vlm/enhancing_continual_learning_of_vision-language_models_via_dynamic_prefix_weight.md)
- [\[ICCV 2025\] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning](openvision_a_fully-open_cost-effective_family_of_advanced_vision_encoders_for_mu.md)
- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [\[ICCV 2025\] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph](dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)

</div>

<!-- RELATED:END -->
