---
title: >-
  [论文解读] HSI: A Holistic Style Injector for Arbitrary Style Transfer
description: >-
  [CVPR 2025][图像生成][风格迁移] HSI提出了一种基于全局风格统计特征和逐元素乘法的风格迁移模块，用线性复杂度替代自注意力的二次复杂度，同时通过双关系学习机制提升风格化质量，在效果和效率上均超越现有方法。 领域现状：任意风格迁移（AST）领域分为全局变换方法（如AdaIN、WCT）和局部匹配方法（如SANet、…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "风格迁移"
  - "注意力机制"
  - "线性复杂度"
  - "全局风格"
  - "逐元素乘法"
---

# HSI: A Holistic Style Injector for Arbitrary Style Transfer

**会议**: CVPR 2025  
**arXiv**: [2502.04369](https://arxiv.org/abs/2502.04369)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 风格迁移, 注意力机制, 线性复杂度, 全局风格, 逐元素乘法

## 一句话总结
HSI提出了一种基于全局风格统计特征和逐元素乘法的风格迁移模块，用线性复杂度替代自注意力的二次复杂度，同时通过双关系学习机制提升风格化质量，在效果和效率上均超越现有方法。

## 研究背景与动机

**领域现状**：任意风格迁移（AST）领域分为全局变换方法（如AdaIN、WCT）和局部匹配方法（如SANet、AdaAttN）。近年来基于注意力机制的方法通过建立内容-风格特征间的逐点语义对应取得了显著进展。

**现有痛点**：注意力机制在AST中存在三个关键问题：(1) 局部模式过度聚焦——逐点匹配导致风格图像中的显著局部区域（如人脸的眼睛）被重复复制到风格化结果中，产生不和谐的图案；(2) softmax指数运算产生偏置模式——过度关注某个突出的风格区域而忽略整体分布；(3) 矩阵乘法导致$O((H\times W)^2 \times C)$的二次复杂度——处理2K分辨率图像时直接GPU内存溢出。

**核心矛盾**：注意力机制虽然能建立语义对应但天然倾向局部匹配，与风格迁移需要的"全局风格整体性"诉求相矛盾。

**本文目标** 设计一个既能捕获全局风格特征又具有线性计算复杂度的风格变换模块。

**切入角度**：作者观察到用单点特征难以表达完整风格模式，因此应直接提取全局风格统计特征（均值、方差、偏度、峰度），并用逐元素乘法替代矩阵乘法来建立内容-风格关系。

**核心 idea**：用全局风格统计特征+逐元素乘法替代逐点注意力+矩阵乘法，实现线性复杂度的高质量风格迁移。

## 方法详解

### 整体框架
采用简单的encoder-decoder架构：encoder为固定参数的预训练VGG-19（到relu_4_1层），提取内容特征$F_c$和风格特征$F_s$后，通过一系列串联的HSI模块完成特征空间中的风格变换，生成合成特征$F_{cs}$，再由镜像结构的decoder转换回图像空间。Decoder中使用上采样替代池化层避免棋盘效应。

### 关键设计

1. **全局风格提取（Global Styles Extraction）**:

    - 功能：从风格特征中提取全面的全局统计表征，替代逐点特征匹配
    - 核心思路：计算风格key特征$K$的四种通道级统计量——均值$\mu$、方差$\sigma^2$、偏度$\gamma_1$和峰度$\gamma_2$。均值和方差描述风格的整体色调和变化程度，偏度和峰度描述风格特征分布的对称性和集中度。然后通过动态网络（深度可分离卷积+全局平均/最大池化）将$K$的平均值和最大值映射为权重$W$和偏置$b$，加权组合四种统计特征。
    - 设计动机：逐点特征无法表达完整风格模式，四种互补的统计特征能更全面描述风格分布特性，避免聚焦于局部显著区域导致的不和谐图案。

2. **动态双关系构建（Dynamic Dual Relations Construction）**:

    - 功能：建立两种互补的内容-风格关系，根据语义相似度动态调节权重
    - 核心思路：同时构建"局部内容-全局风格"关系（$Q \odot K_s$）和"全局内容-全局风格"关系（$Q_c \odot K_s$），通过相似度系数$\lambda_g$动态融合：$F_{qk} = \lambda_g \times (Q_c \odot K_s) \oplus (1-\lambda_g) \times (Q \odot K_s)$。$\lambda_g$基于$Q$和$K$的余弦相似度计算，当内容与风格语义相近时（如都是人脸），全局关系权重增大，增强整体和谐性。
    - 设计动机：局部关系保留内容细节，全局关系提升风格一致性。当两者语义相近时，应更多依赖全局关系来避免局部风格模式不和谐。

3. **线性复杂度变换过程（Linear-Complexity Transfer Process）**:

    - 功能：将计算复杂度从$O((H\times W)^2 \times C)$降至$O(H \times W \times C)$
    - 核心思路：HSI模块中所有特征交互都使用逐元素乘法$\odot$而非矩阵乘法$\otimes$。逐元素乘法只涉及对应位置的元素相乘，复杂度线性于特征图大小。整个HSI结构与自注意力相似（Q-K交互→归一化→加权V→残差连接），但将矩阵乘法全部替换为逐元素乘法。
    - 设计动机：这是首次将逐元素乘法应用于风格迁移中建立语义关系。线性复杂度使方法能处理任意分辨率图像（成功处理2K分辨率），而所有基于注意力的方法在高分辨率下均OOM。

### 损失函数 / 训练策略
总损失：$\mathcal{L} = 60\mathcal{L}_s + 5\mathcal{L}_c + 50\mathcal{L}_{adv}$。风格损失$\mathcal{L}_s$对齐VGG特征空间中风格化图像与风格图像的均值和标准差；内容损失$\mathcal{L}_c$同时对彩色图和灰度图计算VGG特征距离；对抗损失$\mathcal{L}_{adv}$使用多尺度判别器提升整体风格化效果。训练采用Adam优化器，batch size 4，学习率0.0001，在4090Ti上训练约4小时。

## 实验关键数据

### 主实验

| 方法 | Content Loss↓ | Style Loss↓ | LPIPS↓ | FID↓ |
|------|-------------|-------------|--------|------|
| AdaIN | 0.97 | 1.44 | 0.65 | 19.68 |
| SANet | 1.18 | 1.26 | 0.63 | 18.74 |
| AdaAttN | 1.21 | 1.52 | 0.57 | 19.34 |
| StyTr2 | 0.69 | 1.34 | 0.56 | 18.97 |
| StyA2K | **0.59** | 1.21 | 0.49 | 19.85 |
| **HSI (Ours)** | 0.62 | **0.95** | **0.46** | **18.46** |

### 效率对比（GPU显存/推理速度）

| 分辨率 | SANet | StyTr2 | StyA2K | HSI (Ours) |
|--------|-------|--------|--------|-----------|
| 1024×1024 | 2.48GB | OOM | 6.01GB | **2.11GB** |
| 2048×2048 | OOM | OOM | OOM | **8.12GB** |
| 1K推理速度 | - | OOM | - | **~50fps** |

只有HSI和AdaIN能在2K分辨率下运行，且HSI风格化质量远优于AdaIN。

### 关键发现
- 四种统计特征各自贡献不同的颜色和笔触风格，组合后能注入更丰富的风格元素
- 单独构建局部关系保留内容细节但风格不够一致，单独构建全局关系风格更整体但丢失细节，动态组合两者取得最佳平衡
- HSI模块作为即插即用组件替换SANet中的注意力模块后（SANet+HSI），重复风格图案和扭曲区域显著减少，验证了模块的通用性

## 亮点与洞察
- **逐元素乘法替代矩阵乘法的洞察**：首次发现在风格迁移中逐元素乘法足以建立有效的内容-风格语义关系，且天然具有线性复杂度。这一发现可能启发其他需要全局特征交互但对复杂度敏感的任务。
- **全局统计特征的多样性设计**：用四种统计量（均值、方差、偏度、峰度）全面描述风格分布，比仅用均值+方差的AdaIN更丰富，比逐点匹配的注意力更稳定。
- **即插即用的模块设计**：HSI结构与自注意力高度相似，可直接替换现有方法中的注意力模块提升效果，降低了使用门槛。

## 局限与展望
- Content Loss略高于StyA2K（0.62 vs 0.59），说明在极端情况下可能丢失少量内容细节
- 全局统计特征可能在内容与风格差异极大时效果受限（如自然风景+抽象画）
- 未探索视频风格迁移的时序一致性——线性复杂度的优势在视频场景下更有价值
- 改进方向：引入多尺度HSI处理不同频率的风格特征、结合扩散模型实现更精细的风格控制

## 相关工作与启发
- **vs SANet/AdaAttN**: 纯注意力方法过度关注局部导致重复图案（如人脸眼睛出现在多处），HSI通过全局风格提取彻底避免此问题
- **vs StyTr2**: 基于Transformer架构，1K就OOM；HSI线性复杂度可处理2K且效果更好
- **vs StyA2K**: StyA2K用all-to-key注意力降低了部分问题但复杂度仍为二次且容易欠风格化；HSI线性复杂度+更好的风格保真度
- **vs AdaIN**: 同为全局统计方法，但AdaIN仅用均值+方差，HSI增加偏度+峰度且有动态双关系机制，风格化质量显著更高

## 评分
- 新颖性: ⭐⭐⭐⭐ 逐元素乘法替代矩阵乘法的洞察新颖，但整体框架（encoder-decoder+VGG）较传统
- 实验充分度: ⭐⭐⭐⭐ 定性定量比较全面，消融实验验证了各组件，有即插即用实验和高分辨率测试
- 写作质量: ⭐⭐⭐⭐ 动机清晰，与注意力的对比分析深入，图示清晰
- 价值: ⭐⭐⭐⭐ 线性复杂度的风格迁移具有实用价值，即插即用设计降低了使用门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)
- [\[CVPR 2025\] SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer](scsa_a_plug-and-play_semantic_continuous-sparse_attention_for_arbitrary_semantic.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements](stylestudio_text-driven_style_transfer_with_selective_control_of_style_elements.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](../../ICCV2025/image_generation/domain_generalizable_portrait_style_transfer.md)

</div>

<!-- RELATED:END -->
