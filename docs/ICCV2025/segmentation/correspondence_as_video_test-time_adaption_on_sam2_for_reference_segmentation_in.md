---
title: >-
  [论文解读] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild
description: >-
  [ICCV 2025][语义分割][参考分割] CAV-SAM 将参考-目标图像对之间的对应关系表示为伪视频序列，通过基于扩散模型的语义过渡模块（DBST）桥接语义差异，以及测试时几何对齐模块（TTGA）对齐几何变化，使SAM2的视频分割能力零训练地适配参考分割任务，在跨域少样本分割基准上超越SOTA约5% mIoU。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "参考分割"
  - "SAM2"
  - "视频对象分割"
  - "扩散模型语义过渡"
  - "测试时自适应"
---

# Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild

**会议**: ICCV 2025  
**arXiv**: [2508.07759](https://arxiv.org/abs/2508.07759)  
**代码**: [https://github.com/wanghr64/cav-sam](https://github.com/wanghr64/cav-sam)  
**领域**: 图像分割 / 少样本分割  
**关键词**: 参考分割, SAM2, 视频对象分割, 扩散模型语义过渡, 测试时自适应

## 一句话总结

CAV-SAM 将参考-目标图像对之间的对应关系表示为伪视频序列，通过基于扩散模型的语义过渡模块（DBST）桥接语义差异，以及测试时几何对齐模块（TTGA）对齐几何变化，使SAM2的视频分割能力零训练地适配参考分割任务，在跨域少样本分割基准上超越SOTA约5% mIoU。

## 研究背景与动机

大视觉模型（如SAM）在下游任务中存在域差距和类别新颖性问题，参考分割（利用参考图像和掩码向模型传递新知识）是解决这一问题的有前景方向。然而：

- **现有方法依赖元学习**：Few-shot Segmentation (FSS) 和 Cross-Domain FSS (CD-FSS) 方法需要大量的元训练过程，带来巨大的数据和计算成本
- **离散图像对 vs 连续视频**：SAM2具备交互式视频对象分割（iVOS）能力，可以从标注帧传播分割到整个视频序列。但参考-目标图像对是离散的，与自然连续视频有两大差距：(a) 语义差异——iVOS追踪同一实例，FSS识别同一类别；(b) 几何变化——iVOS假设帧间平滑变换，参考分割中几何变化可能很大

关键观察：简单地将参考和目标图像拼接为伪视频，SAM2就能在CD-FSS基准上达到近SOTA水平（mIoU 60.68 vs SOTA ~61），说明iVOS模型具有巨大的参考分割潜力。

本文的核心idea：**将参考-目标图像之间的对应关系表示为视频**，利用扩散模型生成语义过渡序列弥合语义差异，通过测试时轻量微调对齐几何变化。

## 方法详解

### 整体框架

给定参考图像 $I_r$ 和掩码 $M_r$ 以及目标图像 $I_t$，CAV-SAM 包含两个模块：
1. **DBST**：用扩散模型在参考-目标之间生成语义过渡序列 $I_v^1, ..., I_v^{N_v}$
2. **TTGA**：通过原型向量和增强图像进行轻量测试时微调，生成伪标签作为SAM2的额外提示
最终SAM2以iVOS方式在伪视频序列上完成分割。

### 关键设计

1. **Diffusion-Based Semantic Transition (DBST，扩散语义过渡）**:

    - 功能：在参考和目标图像之间生成语义平滑过渡的伪视频帧
    - 核心思路：基于DiffMorpher，分别对参考和目标图像训练LoRA参数 $\Delta\theta_r$ 和 $\Delta\theta_t$，通过线性插值 $\Delta\theta_\alpha = (1-\alpha)\Delta\theta_r + \alpha\Delta\theta_t$ 融合语义；同时对DDIM反转得到的潜在噪声进行球面线性插值：$\mathbf{z}_{T\alpha}=\frac{\sin((1-\alpha)\phi)}{\sin\phi}\mathbf{z}_{Tr}+\frac{\sin(\alpha\phi)}{\sin\phi}\mathbf{z}_{Tt}$
    - 设计动机：iVOS模型追踪同一实例，但FSS要求识别同一类别的不同实例。扩散模型生成的语义过渡让SAM2能沿着连续变化路径追踪对象类别
    - 优化：移除了DiffMorpher中不必要的视觉细化模块，大幅降低推理成本

2. **Test-Time Geometric Alignment (TTGA, 测试时几何对齐）**:

    - 功能：让SAM2的图像编码器适应目标对象的几何变化，生成伪标签提示
    - 核心思路：仅微调SAM2图像编码器的FPN层。使用参考图像的原型向量 $\boldsymbol{p}_r = \text{MAP}(F_r, M_r)$ 和余弦相似度激活目标区域。训练损失为增强循环一致性（ACC）：$\mathcal{L} = \mathcal{L}_{aug} + \mathcal{L}_{cyc}$，其中 $\mathcal{L}_{aug}$ 监督增强图像的预测，$\mathcal{L}_{cyc}$ 通过增强图像的伪标签反向预测原图实现循环一致
    - 设计动机：直接从SAM2提取的特征无法有效激活语义过渡序列中的目标区域。仅用一张参考图像进行轻量微调（100步，仅FPN层），避免了元训练的高成本

3. **Augmentative Cyclic Consistency (ACC)**:

    - 功能：最大化利用有限标注数据的学习信号
    - 核心思路：$I_r \rightarrow \boldsymbol{p}_r \rightarrow \hat{M}_r^{aug}$（前向）→ $I_r^{aug} \rightarrow \hat{\boldsymbol{p}}_r^{aug} \rightarrow \hat{M}_r$（循环），使用预测伪标签而非GT来计算增强原型，提供更鲁棒的自监督信号
    - 设计动机：与使用GT掩码的ABC方法相比，ACC使用预测掩码作为中间步骤，提供了隐式的数据增强效果

### 损失函数 / 训练策略

- 无需meta-training，仅在测试时对每个参考图像进行100步轻量微调
- 使用SAM2 tiny版本，DBST中LoRA训练200步、rank=16、20步DDIM反转
- 语义过渡序列长度 $N_v=9$，$\alpha$ 从0.2到0.8均匀分布
- TTGA学习率 $1\times10^{-3}$，cosine退火

## 实验关键数据

### 主实验

**CD-FSS 4数据集 1-shot/5-shot mIoU：**

| 方法 | 类型 | Deepglobe | ISIC | Chest X-Ray | FSS-1000 | Average |
|------|------|-----------|------|-------------|----------|---------|
| IFA (CVPR24) | CD-FSS | 37.73 | 44.55 | 80.03 | 79.97 | 60.57 |
| DR-Adaptor (CVPR24) | CD-FSS | 41.29 | 40.77 | 82.35 | 79.05 | 60.86 |
| APSeg (CVPR24) | SAM-based | 35.94 | 45.43 | 84.10 | 79.71 | 61.30 |
| **CAV-SAM (Ours)** | **iVOS-based** | **39.11** | **50.36** | **86.97** | **79.78** | **64.06** |

5-shot: 本方法达到69.14 average mIoU，高于所有方法（APSeg 65.09, DR-Adaptor 65.42）。

在最具挑战性的Chest X-Ray医学数据集上提升最显著（1-shot: 86.97 vs 84.10 = +2.87）。

### 消融实验

| iVOS模型 | DBST | TTGA | mIoU |
|---------|------|------|------|
| SAM2 | ✗ | ✗ | 60.68 |
| SAM2 | ✓ | ✗ | 62.37 |
| SAM2 | ✗ | ✓ | 62.52 |
| SAM2 | ✓ | ✓ | **64.06** |

**ACC vs ABC对比：** ACC (Augmentative Cyclic Consistency) 显著优于使用GT掩码的ABC策略。

### 关键发现
- 简单拼接参考-目标图像作为伪视频即可达到近SOTA水平（60.68），证明iVOS模型天然适合参考分割
- DBST和TTGA模块各自贡献约+1.7和+1.8 mIoU，组合后效果叠加
- 在Deepglobe数据集上效果较差，推测是因为SAM面向物体分割，而Deepglobe是区域分割
- 虽然DBST生成的视觉效果不如原版DiffMorpher，但分割性能同样优秀——SAM2的鲁棒iVOS能力弥补了视觉质量的不足

## 亮点与洞察
- **视角转换精妙**：将参考分割问题重新定义为视频对象分割问题，巧妙利用了SAM2的iVOS能力
- **免meta-training**：完全跳过了传统FSS方法必需的大规模元训练过程，仅需测试时100步微调
- **语义一致性保障**：当参考和目标图像不含同一类别时，DBST生成无意义序列、TTGA原型不激活，自然避免误分割
- **通用框架**：可与任意iVOS模型和扩散模型搭配，未来模型进化可直接受益

## 局限与展望
- DBST模块仍需为每对图像训练LoRA（200步），推理开销较大
- 区域分割（如Deepglobe遥感场景）效果不理想，SAM/SAM2本身的局限
- 生成的伪视频序列质量依赖扩散模型能力，对风格差异极大的跨域可能效果有限
- 仅评估了1-shot和5-shot设置，未探索更多参考图像数量的收益规律
- SAM2 tiny可能限制性能上限，更大的SAM2模型可能带来进一步提升

## 相关工作与启发
- **SAM2**：本文的核心基座模型，提供iVOS能力来分割视频中的对象
- **DiffMorpher**：扩散模型图像变形方法，本文借鉴其LoRA和噪声插值生成语义过渡序列
- **APSeg / VRP-SAM**：SAM上的参考分割方法，但仍依赖meta-learning引入额外prompt encoder
- **PATNet / IFA / DR-Adaptor**：CD-FSS方法，需要完整的元训练流程

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将参考分割转化为视频分割是非常巧妙的视角转换，DBST+TTGA设计优雅
- 实验充分度: ⭐⭐⭐⭐ 4个数据集全面评估，消融充分，但缺少更多iVOS模型的对比
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，问题分析深入，图示直观易懂
- 价值: ⭐⭐⭐⭐ 提出了参考分割的新范式，对利用基础模型适配下游任务有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)
- [\[CVPR 2026\] Mixture of Prototypes for Test-time Adaptive Segmentation](../../CVPR2026/segmentation/mixture_of_prototypes_for_test-time_adaptive_segmentation.md)
- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)
- [\[ICML 2025\] IT³: Idempotent Test-Time Training](../../ICML2025/segmentation/it3_idempotent_test-time_training.md)

</div>

<!-- RELATED:END -->
