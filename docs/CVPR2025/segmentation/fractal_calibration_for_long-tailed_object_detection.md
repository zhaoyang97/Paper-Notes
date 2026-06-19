---
title: >-
  [论文解读] Fractal Calibration for Long-Tailed Object Detection
description: >-
  [CVPR 2025][语义分割][长尾分布] 提出 FRACAL（FRActal CALibration），一种无需训练的后处理方法，首次将分形维数引入长尾目标检测的后校准中，通过对称校准频率轴（类别频率）和空间轴（类别位置均匀度），在 LVIS 数据集上将稀有类 mask AP 提升高达 8.6%，并在 COCO、V3Det、OpenImages 上展示泛化性。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "长尾分布"
  - "后校准"
  - "分形维数"
  - "logit 调整"
  - "空间感知"
---

# Fractal Calibration for Long-Tailed Object Detection

**会议**: CVPR 2025  
**arXiv**: [2410.11774](https://arxiv.org/abs/2410.11774)  
**代码**: [https://github.com/kostas1515/FRACAL](https://github.com/kostas1515/FRACAL)  
**领域**: 目标检测  
**关键词**: 长尾分布, 后校准, 分形维数, logit 调整, 空间感知

## 一句话总结

提出 FRACAL（FRActal CALibration），一种无需训练的后处理方法，首次将分形维数引入长尾目标检测的后校准中，通过对称校准频率轴（类别频率）和空间轴（类别位置均匀度），在 LVIS 数据集上将稀有类 mask AP 提升高达 8.6%，并在 COCO、V3Det、OpenImages 上展示泛化性。

## 研究背景与动机

**领域现状**：目标检测模型在长尾分布数据上训练时，频繁类表现良好但稀有类性能严重不足。现有方法主要通过重加权和重采样应对类别不平衡，但都需要修改训练流程。在图像分类领域，后校准的 Softmax 调整（PCSA）方法因其无需训练、兼容性强而备受关注。

**现有痛点**：目前的 PCSA 方法仅利用训练集的类别频率 $p_s(y)$ 来调整 logit，完全忽略了类别与其位置分布 $p_s(y, u)$ 之间的依赖关系。但在目标检测中，类别和位置高度相关（如"帽子"主要出现在图像上方，"鞋子"出现在下方），这一信息对后校准至关重要。

**核心矛盾**：直接建模类别-位置联合分布 $p(y, u)$ 面临网格粒度选择的困难。网格太粗（如 2x2）丢失位置信息；网格太细（如 64x64）稀有类的位置统计变得极其稀疏和噪声化。需要一种不依赖特定网格粒度的方式来编码空间分布信息。

**本文目标** (1) 如何在后校准中融入位置信息来补充频率信息？(2) 如何解决稀有类位置统计的稀疏性问题？

**切入角度**：分形维数是一种与具体网格大小无关的度量，衡量点集在空间中的"填充均匀度"。$\Phi=0$ 表示单点，$\Phi=2$ 表示完全均匀覆盖 2D 空间。通过 box-counting 方法在多个网格尺度上计算类别位置分布的分形维数，可以鲁棒地编码空间信息且不受稀疏性影响。

**核心 idea**：用分形维数捕捉目标类别在图像空间中的分布均匀度，与传统频率校准互补，双轴校准让检测器在频率和空间两个维度上都更均衡。

## 方法详解

### 整体框架

FRACAL 是一个推理时的后处理方法。输入为任意训练好的检测器的输出 logit $z_y$，输出为校准后的 logit $z_y'$。分两步校准：(1) 分类校准 C（基于频率）调整前景类 logit；(2) 空间校准 S（基于分形维数）进一步调整概率。两步合并后重归一化。FRACAL 的权重在训练集上预计算并存储在内存中，推理时无额外计算开销。

### 关键设计

1. **分类校准 C（频率校准）**:

    - 功能：通过类别频率调整 logit，减少频繁类偏见、增加稀有类检测
    - 核心思路：对前景类 logit 执行 $C(z_y) = z_y - \log_\beta(\frac{n_y}{\sum_i^C n_i}) + \log_\beta(\frac{1}{C})$，其中 $n_y$ 是类别 $y$ 的实例数，$\beta$ 是对数底数超参。设定目标分布为均匀分布 $p_t(y) = \frac{1}{C}$，因为 AP 指标独立评估每个类别、奖励均衡检测。背景 logit 不调整（假设目标和训练集的物体分布相同 $p_s(o,u) \approx p_t(o,u)$）
    - 设计动机：从 Bayes 最优分类的推导出发，频率校准等价于将训练分布偏移到均衡测试分布。此设计独立于后面的空间校准，可单独使用

2. **空间感知校准 S（分形维数校准）**:

    - 功能：利用分形维数衡量每类在空间中的均匀程度，降权均匀分布类别、升权稀疏分布类别
    - 核心思路：先用 box-counting 方法计算每类的分形维数 $\Phi(y)$：在多种网格粒度 $G$ 下统计有目标中心的网格数 $\nu_y$，对 $(\log G, \log \nu_y)$ 拟合直线的斜率即为 $\Phi(y)$。为处理稀有类稀疏问题，引入"二次规则"：只在 $G \leq \lfloor\sqrt{n_y}\rfloor$ 范围内计算（确保网格中至少有可能被填满）。然后在推理时：$S(z_y) = \frac{\sigma(z_y)}{\Phi(y)^\lambda}$，$\lambda$ 为超参。效果是均匀分布类（$\Phi$ 大）的概率被降权，非均匀类（$\Phi$ 小）被升权
    - 设计动机：分形维数与频率弱相关（Pearson 0.35-0.375），提供了互补信息。许多稀有类的 $\Phi \approx 2$ 说明二次规则对小样本集鲁棒。空间校准迫使检测器在所有位置均匀预测各类目标，消除空间偏见

3. **FRACAL 完整校准公式与 Sigmoid 扩展**:

    - 功能：组合频率和空间校准，并支持 Sigmoid 二元分类器
    - 核心思路：Softmax 检测器：$F(z_y) = \frac{S(C(z_y))}{\sum_{j=1}^{C+1} S(C(z_j))}$，先做频率校准再做空间校准，最后重归一化。Sigmoid 检测器：$F_b(z_i) = \eta(C(z_i) - \log_\beta(\frac{\Phi(y)^\lambda}{\sum_i^C \Phi(i)^\lambda}) + \log_\beta(\frac{1}{C})) \cdot \eta(z_i)$，将频率和空间校准都转到 logit 空间执行，$\eta(z_i)$ 作为背景过滤器
    - 设计动机：二元分类器同时做前景-背景分类和类间分类，需要先解耦再校准

### 损失函数 / 训练策略

FRACAL 无需训练。所有校准权重在训练集上一次性预计算。推理时在 NMS 之前应用校准，计算开销可忽略。超参数 $\beta$ 和 $\lambda$ 通过验证集搜索确定。

## 实验关键数据

### 主实验

| 方法 | Backbone | $AP^m$↑ | $AP^m_r$↑ | $AP^m_c$↑ | $AP^m_f$↑ |
|------|----------|---------|-----------|-----------|-----------|
| Baseline | R50 | 25.7 | 15.8 | 25.1 | 30.6 |
| NorCal | R50 | 25.2 | 19.3 | 24.2 | 29.0 |
| GOL | R50 | 27.7 | 21.4 | 27.7 | 30.4 |
| LogN | R50 | 27.5 | 21.8 | 27.1 | 30.4 |
| **FRACAL** | R50 | **28.6** | **23.0** | **28.0** | **31.5** |
| Seesaw | Swin-S | 32.4 | 25.6 | 32.8 | 34.9 |
| **FRACAL** | Swin-S | **34.4** | **27.8** | **34.5** | **36.4** |

使用 Swin-S 时稀有类相对提升达 8.6%（25.6→27.8），Swin-B+ImageNet22K 进一步达到 6.6pp 稀有类提升。

### 消融实验

| 配置 | $AP^m$ | $AP^m_r$ | 说明 |
|------|--------|----------|------|
| 仅频率校准 C | 27.8 | 21.2 | 频率信息已有效 |
| 仅空间校准 S | 26.1 | 16.3 | 单独空间校准不够 |
| 网格校准 $C_G$ (固定 G) | ≤27.5 | ≤21.3 | 稀有类受稀疏影响 |
| FRACAL (C + S) | **28.6** | **23.0** | 双轴互补最优 |

### 关键发现

- 分形维数与频率的 Pearson 相关性仅 0.35（LVIS），证实两者确实提供互补信息
- 空间校准不仅提升稀有类，也提升频繁类（31.5 vs 30.6 $AP^m_f$），因为某些频繁但空间分布非均匀的类也受益于位置去偏
- FRACAL 可无缝组合 MaskRCNN、GFLv2 等不同架构，以及 ResNet、Swin 等不同 backbone
- 在 COCO、V3Det、OpenImages 等不同不平衡程度的数据集上也有稳定提升，泛化性强

## 亮点与洞察

- **分形维数的引入非常巧妙**：它天然解决了空间统计中网格粒度选择的两难问题，而且对稀有类特别鲁棒（二次规则保证了小样本不被低估）。这种将分形几何引入检测后校准的思路很新颖
- **双轴校准的思想具有普适性**：不仅限于长尾检测，任何存在"频率偏见"和"空间偏见"的任务（如语义分割中的类别-像素位置不平衡）都可以借鉴
- **零训练成本是最大实用优势**：FRACAL 作为即插即用的推理后处理，可以和任何训练策略（数据增强、对比学习等）正交组合，不增加训练负担

## 局限与展望

- 超参数 $\beta$ 和 $\lambda$ 需要在验证集上搜索，不同数据集/模型可能需要不同值
- 分形维数假设类别位置分布可以量化其"均匀度"，对于真正位置无关的类别可能无额外增益
- 目前只在目标检测/实例分割上验证，未扩展到语义分割、全景分割等任务
- 空间校准是全局的（每类一个 $\Phi$），未考虑图像内容自适应的局部校准
- 对于极度稀有类（训练集出现 <4 次），分形维数无法计算只能赋默认值 $\Phi=1$，此时退化为纯频率校准

## 相关工作与启发

- **vs NorCal**：NorCal 也是后校准方法，但只用频率统计归一化前景概率，缺乏空间信息。FRACAL 在 R50 上比 NorCal 高 3.4pp $AP^m$、3.7pp $AP^m_r$
- **vs LogN**：LogN 用模型自身预测估计类别统计做标准化，需要前向遍历整个训练集来估算权重，比 FRACAL 更慢且模型依赖。FRACAL 只需数据集统计，模型无关
- **vs 训练时方法（Seesaw等）**：训练时方法需要修改训练流程且各方法难以叠加。FRACAL 可以在任何已训练模型上直接应用，且与训练方法正交叠加带来额外增益

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 分形维数在检测校准中的首次应用，角度新颖且理论推导清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多 backbone、多架构的全面验证，消融详尽
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，从理论到实践的过渡流畅
- 价值: ⭐⭐⭐⭐⭐ 即插即用的后处理方法，实际部署价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2025\] Towards Generalizable Scene Change Detection](towards_generalizable_scene_change_detection.md)

</div>

<!-- RELATED:END -->
