---
title: >-
  [论文解读] Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion
description: >-
  [CVPR 2025][模型压缩][图像压缩] 本文将Wasserstein Distortion（WD）作为优化目标应用于过拟合图像编解码器C3，结合公共随机性实现纹理再采样，在保持极低解码复杂度（<1% MACs of HiFiC）的同时达到与生成式压缩方法相当的视觉质量-码率权衡。 领域现状：近年来学习型图像压缩受生成…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "图像压缩"
  - "Wasserstein失真"
  - "感知质量"
  - "过拟合编解码器"
  - "纹理重建"
---

# Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion

**会议**: CVPR 2025  
**arXiv**: [2412.00505](https://arxiv.org/abs/2412.00505)  
**代码**: 无（提供了重建结果数据下载）  
**领域**: 模型压缩  
**关键词**: 图像压缩, Wasserstein失真, 感知质量, 过拟合编解码器, 纹理重建

## 一句话总结
本文将Wasserstein Distortion（WD）作为优化目标应用于过拟合图像编解码器C3，结合公共随机性实现纹理再采样，在保持极低解码复杂度（<1% MACs of HiFiC）的同时达到与生成式压缩方法相当的视觉质量-码率权衡。

## 研究背景与动机

**领域现状**：近年来学习型图像压缩受生成模型（GAN、扩散模型）启发，通过更好的自然图像分布建模获得了出色的图像质量。代表方法如HiFiC利用对抗损失，CDC基于扩散模型，都能生成视觉上令人信服的重建图像。

**现有痛点**：这些生成式压缩方法的计算复杂度比商用编解码器高出数个数量级，对移动设备等实际应用场景来说完全不可行。在"好（质量）、便宜（码率）、快（速度）"三者中似乎只能选其二。

**核心矛盾**：生成式方法追求的是精确建模自然图像分布来获得高质量，但精确建模分布本身就需要极高的复杂度。问题在于——高视觉质量是否真的需要精确的分布建模？

**本文目标**：证明通过建模人类视觉感知（而非数据分布），可以在低解码复杂度下实现与生成式压缩可比的视觉质量。

**切入角度**：人类视觉系统的外围视觉只能感知纹理统计特征而非精确像素值。Wasserstein Distortion（WD）正是建模了这一特性——在显著性低的区域允许纹理再采样，从而大幅节省比特。

**核心 idea**：用WD替代MSE作为C3过拟合编解码器的优化目标，并引入公共随机性（Common Randomness, CR）辅助纹理重建，实现"好、便宜又快"的图像压缩。

## 方法详解

### 整体框架
方法基于C3过拟合编解码器架构：每张图像独立优化一组多分辨率隐变量、合成网络和熵模型。输入为原始图像，隐变量经过双线性上采样和拼接后通过小型CNN合成网络输出RGB像素，熵网络为每个隐变量元素建模条件概率。本文只做两个改动：(1) 引入公共随机性噪声作为解码端额外输入；(2) 将MSE损失替换为Wasserstein Distortion。

### 关键设计

1. **Wasserstein Distortion (WD) 作为优化目标**:

    - 功能：度量原图与重建图在感知空间中的距离，同时考虑中央凹视觉和外围视觉的差异
    - 核心思路：从VGG网络提取多层特征图，在不同的空间位置根据 $\sigma$ 值决定池化区域大小，计算局部均值和标准差的2-Wasserstein散度。$\sigma$ 小时趋近于逐点比较（中央凹），$\sigma$ 大时允许纹理统计匹配（外围视觉）。作者提出了一种高效的近似计算方式：将 $\sigma$ 离散化为2的幂次，构建类高斯金字塔的多尺度级联来预计算局部统计量，再通过线性插值得到任意 $\sigma$ 对应的WD值
    - 设计动机：相比LPIPS等逐点特征距离，WD显式建模了视觉系统的注视-外围二元性，在外围区域容许纹理再采样，有效降低编码成本

2. **基于显著性的自适应 $\sigma$ 映射**:

    - 功能：根据图像区域的视觉重要性自适应调节WD的容许程度
    - 核心思路：利用EML-net预测显著性图 $s$，转换为密度 $p = p_{min} + (1-p_{min}) \cdot s/\bar{s}$，再映射为 $\sigma = \sigma_{max} \cdot p_{min}/p$。显著区域 $\sigma$ 小（要求精确重建），非显著区域 $\sigma$ 大（允许纹理再采样）
    - 设计动机：统一的常数 $\sigma$ 会导致语义重要内容（如文字）被当作可替换纹理处理。例如相机上的"ZENITAR-M"文字在常数 $\sigma$ 下变得不可读，但基于显著性的 $\sigma$ 能保护这些区域

3. **公共随机性 (Common Randomness, CR)**:

    - 功能：为解码端提供与编码端相同的伪随机噪声，使编解码器能协同重建随机纹理
    - 核心思路：在多个分辨率上生成i.i.d.标准高斯噪声（固定种子的伪随机数发生器），上采样后与隐变量拼接作为合成网络的额外输入。由于种子固定，不需要额外传输比特
    - 设计动机：没有CR时，编解码器只能用确定性结构（如直线）来近似随机纹理，效果有限。有了CR，编解码器可以对噪声进行"噪声整形"来高效重建随机纹理（如草地），从而将更多比特分配给结构性内容

### 损失函数 / 训练策略
优化目标为率失真损失，其中失真项为WD（替代原C3的MSE），率项为隐变量在熵模型下的交叉熵。两种WD变体：WD8使用常数 $\sigma=8$，WDs使用显著性驱动的 $\sigma$ 映射（$\sigma_{max}=16$）。编码复杂度因WD计算约增加6倍。

## 实验关键数据

### 主实验
在CLIC2020 professional验证集（41张图像）上，通过大规模人工评测研究（16659次成对比较）评估10种压缩方法。目标码率为0.075、0.15和0.3 bits/pixel。

| 方法 | 解码MACs/pixel | Elo评分趋势 | 说明 |
|------|---------------|------------|------|
| C3/WDs (本文) | ~10³ | 与HiFiC相当 | 公共随机性+显著性WD |
| C3/WD8 (本文) | ~10³ | 略低于WDs | 常数σ的WD |
| HiFiC | ~10⁵ | 最优 | 基于GAN的生成式压缩 |
| VVC | ~10² | 中等偏低 | 商用编解码器 |
| MLIC+ | ~10⁵ | 中等偏低 | MSE最优的学习型方法 |
| CDC | ~10⁶ | 中等偏低 | 基于扩散的压缩 |
| C3/MSE | ~10³ | 最低 | 原始C3 |

### 感知指标预测人类评分

| 指标 | 二元评分预测准确率 | Elo PCC | Elo SRCC |
|------|------------------|---------|----------|
| WDs (本文) | **72.87%** | **0.942** | **0.913** |
| LPIPS | ~67% | ~0.7 | ~0.7 |
| DISTS | ~67% | ~0.8 | ~0.8 |
| MS-SSIM | ~66% | ~0.7 | ~0.7 |
| PSNR | ~64% | ~0.5 | ~0.5 |

### 关键发现
- WD优化后，最高分辨率隐变量（编码细节）的比特分配显著减少，说明WD允许纹理再采样从而释放了比特预算给其他层
- CR进一步增强了这一效果：提供CR后，编解码器更少地"硬编码"纹理细节
- 作为感知指标，WD与人类Elo评分的Pearson相关性达到94%，远超LPIPS等现有指标
- 基于显著性的σ映射能有效保护语义重要区域（如文字），视觉改善显著

## 亮点与洞察
- **感知建模替代分布建模**的思路非常巧妙：不需要生成式模型的复杂度就能获得相当的感知质量，关键在于利用了人类视觉的"外围盲区"
- **公共随机性**是一个优雅的设计：零额外比特成本就能让编解码器协同重建随机纹理，本质上是把确定性编码问题转化为了随机编码问题
- **WD作为IQA指标**意外地表现极好（94% PCC），这个副产品可能比压缩方法本身更有价值，可迁移到任何需要感知质量评估的场景

## 局限与展望
- 编码复杂度增加约6倍（因WD比MSE计算更贵），对实时编码场景是瓶颈
- 特征空间和σ映射的选择完全是ad-hoc的，缺少系统性优化
- 人工评测规模有限（41张图×3个码率），需要更大规模验证
- 显著性模型的质量直接影响σ映射的效果，针对压缩任务的专用显著性模型可能更优

## 相关工作与启发
- **vs HiFiC**: HiFiC用GAN实现生成式压缩，视觉质量略优于本文，但解码复杂度高两个数量级以上。本文用感知损失+低复杂度架构实现了可比质量
- **vs CDC**: CDC基于扩散模型的压缩，解码复杂度最高（~10⁶ MACs），视觉质量却与MSE优化的方法相当，凸显了生成式方法在压缩中的效率问题
- **vs COOL-CHIC/C3系列**: 本文是C3的直接改进，仅改损失函数+加CR就获得了巨大质量提升，说明损失函数的选择比架构更关键

## 评分
- 新颖性: ⭐⭐⭐⭐ WD在压缩中的应用新颖，但核心组件（C3架构、WD指标）均为已有工作
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模人工评测研究是金标准，对比全面，分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 论文逻辑清晰，图表精美，motivation阐述非常到位
- 价值: ⭐⭐⭐⭐ 实际意义大（低复杂度高质量压缩），WD作为IQA指标的副发现也有独立价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)

</div>

<!-- RELATED:END -->
