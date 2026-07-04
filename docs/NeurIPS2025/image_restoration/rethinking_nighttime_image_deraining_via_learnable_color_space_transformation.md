---
title: >-
  [论文解读] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation
description: >-
  [NeurIPS 2025][图像恢复][夜间图像去雨] 基于"夜间雨在YCbCr的Y通道（亮度）差异远大于RGB"的统计发现，提出可学习颜色空间转换器(CSC)在Y通道做去雨、隐式光照引导(IIG)编码夜间不均匀光照、以及光照感知的高质量数据集HQ-NightRain，三管齐下显著提升夜间去雨效果。
tags:
  - "NeurIPS 2025"
  - "图像恢复"
  - "夜间图像去雨"
  - "可学习颜色空间变换"
  - "YCbCr"
  - "隐式光照引导"
  - "HQ-NightRain数据集"
---

# Rethinking Nighttime Image Deraining via Learnable Color Space Transformation

**会议**: NeurIPS 2025  
**arXiv**: [2510.17440](https://arxiv.org/abs/2510.17440)  
**代码**: [guanqiyuan/CST-Net](https://github.com/guanqiyuan/CST-Net)  
**机构**: 大连工业大学 / 南京理工大学 / 大连海事大学
**领域**: 图像复原  
**关键词**: 夜间图像去雨, 可学习颜色空间变换, YCbCr, 隐式光照引导, HQ-NightRain数据集

## 一句话总结

基于"夜间雨在YCbCr的Y通道（亮度）差异远大于RGB"的统计发现，提出可学习颜色空间转换器(CSC)在Y通道做去雨、隐式光照引导(IIG)编码夜间不均匀光照、以及光照感知的高质量数据集HQ-NightRain，三管齐下显著提升夜间去雨效果。

## 研究背景与动机

**领域现状**：图像去雨已有大量深度学习方法（PReNet、Restormer、DRSformer等），但绝大多数聚焦白天场景。夜间场景因低光照、人造光源分布不均以及雨与光照的耦合效应，去雨难度远大于白天。

**现有痛点**：
1. **数据集不真实**：现有夜间雨数据集（GTAV-NightRain、RoadScene等）采用全局均匀的雨mask线性叠加到背景上，完全忽略了夜间雨仅在光源附近可见的物理特性，导致合成图像与真实场景存在明显域差距
2. **方法未定制化**：现有夜间去雨方法仍在RGB空间操作，没有利用夜间雨在特定颜色通道上的固有特性

**核心矛盾**：夜间雨退化的物理形成机制（光源附近可见、非均匀分布、亮度通道差异显著）与现有方法/数据集的"白天假设"（均匀雨分布、RGB空间处理）之间的错配。

**本文目标**：如何利用夜间雨的物理特性（亮度通道显著性 + 光照依赖的非均匀分布）来设计更有效的夜间去雨方法和数据集。

**切入角度**：从像素值统计分析入手——对比不同颜色空间的通道直方图，发现YCbCr的Y通道在雨/无雨图像之间差异最大（因为雨在低光条件下反射人造光源形成高对比亮度模式），由此驱动在Y通道做去雨的设计思路。

**核心 idea**：用可学习的颜色空间变换将图像从RGB转到YCbCr，在亮度差异最大的Y通道做退化去除，同时用隐式神经表示编码不均匀光照信息引导去雨。

## 方法详解

### 整体框架

CST-Net 是一个两阶段端到端网络：

1. **退化去除阶段(Stage 1)**：通过可学习CSC将RGB图像转到YCbCr空间，取出Y通道（亮度）送入Transformer编码器-解码器做雨退化去除，Cb/Cr通道保留传递
2. **颜色精炼阶段(Stage 2)**：将去雨后的Y通道与Cb/Cr拼接，经CSC转回RGB空间，与原始输入相加后送入第二阶段做颜色恢复

两个阶段均采用4层Transformer编码器-解码器结构（基于Restormer架构）。两阶段之间还有一个隐式光照引导(IIG)分支，提供光照感知的特征引导。

### 关键设计

1. **可学习颜色空间转换器 (CSC)**

    **功能**：实现RGB ↔ YCbCr双向颜色空间转换，替代传统固定矩阵变换。

    **核心思路**：标准RGB→YCbCr转换使用固定的3×3权重矩阵 $W$（如Y=0.299R+0.587G+0.114B），CSC将其替换为可学习参数矩阵 $\Phi = \{\varphi_{i,j}\}$，每个元素是一维可学习变量，通过MLP进行非线性变换后再做矩阵乘法：$[Y, Cb, Cr]^T = \text{MLP}(\Phi) \circ [R, G, B]^T$。

    **设计动机**：固定变换矩阵是为通用场景设计的标准范式，无法适应夜间场景中人造光源导致的复杂亮度分布。可学习参数能根据不同光照条件、场景类型自适应调整通道权重分配，使得在高光区域避免像素丢失，对复杂随机夜间雨场景更鲁棒。实验中固定变换的可视化结果显示高亮区域存在像素损失，而可学习CSC能动态调整权重避免此问题。

2. **隐式光照引导模块 (IIG)**

    **功能**：在两个阶段之间传递光照信息，引导模型关注光照区域的雨退化。

    **核心思路**：利用隐式神经表示(INR)编码光照信息——将夜间图像patch的像素坐标存入坐标集 $P \in \mathbb{R}^{H \times W \times 2}$，对每个像素施加与邻域中心距离相关的动态权重 $w(x', y')$（距离越远权重越小）提取局部光照信息 $P_{(x,y)}$，再将编码特征 $E$ 和位置光照信息拼接后送入MLP解码得到去雨后的Y通道值：$I_{\hat{Y}} = \text{MLP}(\text{cat}[E, P_{(x,y)}])$。

    **设计动机**：夜间光照分布极不均匀（光源附近亮、其他区域暗），不同光照区域的雨退化程度不同。相比显式光照估计（如Retinexformer），隐式表示更灵活——不同光照分布产生不同的MLP参数组，天然适应复杂夜间场景。特征可视化显示IIG能准确聚焦到光照区域来引导去雨。

3. **HQ-NightRain 数据集构建**

    **功能**：提供高真实度的夜间雨合成训练数据。

    **核心思路**：提出光照感知的雨合成管线——先从背景图的HSV空间V通道提取光照系数矩阵 $\mathbf{I}$，设定高低阈值 $\tau_1=0.2, \tau_2=0.8$ 将极暗和极亮区域的雨可见度降半，然后将雨mask与光照系数逐元素相乘 $\sigma(S) = S \odot \mathbf{I}$，生成非均匀分布的雨。此外对雨滴场景还加入散焦模糊模拟真实折射效果。新的雨模型为 $R_s = f[B, \sigma(S)]$，用3×3卷积融合替代简单线性加法。

    **设计动机**：根据照度第一定律，距光源越远照度越低，因此真实夜间的雨仅在光源附近可见。现有数据集全部做全局均匀雨叠加，违反了这一物理先验。t-SNE分析证实HQ-NightRain的特征分布与真实雨图的域差距显著小于现有数据集。数据集规模11,200对（训练10K + 验证900 + 测试300），分雨线(RS)/雨滴(RD)/混合(SD)三个子集，另附512张真实采集图像。

### 损失函数 / 训练策略

- 两阶段均采用4层Transformer编码器-解码器（Restormer架构）
- 优化器：Adam（默认参数），初始学习率 $2 \times 10^{-4}$，余弦退火至 $1 \times 10^{-6}$
- 训练500 epochs，patch size 128×128，batch size 4
- 单卡RTX 3090训练
- 光照阈值 $\tau_1 = 0.2$，$\tau_2 = 0.8$

## 实验关键数据

### 主实验

**HQ-NightRain + GTAV-NightRain 平均性能对比：**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| PReNet | 35.51 | 0.9669 | 0.0821 |
| Restormer | 38.50 | 0.9767 | 0.0526 |
| DRSformer | 38.74 | 0.9766 | 0.0535 |
| NeRD-Rain | 38.51 | 0.9754 | 0.0588 |
| **CST-Net** | **39.07** | **0.9778** | **0.0477** |

**真实数据集 RealRain-1k 性能：**

| 方法 | PSNR (1k-L)↑ | PSNR (1k-H)↑ |
|------|-------------|-------------|
| DRSformer | 27.21 | 23.73 |
| NeRD-Rain | 27.16 | 23.65 |
| **CST-Net** | **27.31** | **23.81** |

**多天气恢复 Multi-Weather6K（扩展实验）：**

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Restormer | 31.80 | 0.9228 |
| PromptIR | 31.69 | 0.9169 |
| **CST-Net** | **33.82** | **0.9642** |

### 消融实验

**模块消融（HQ-NightRain SD子集）：**

| 配置 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| 仅Stage1 + YCbCr + CSC | 35.09 | 0.9650 |
| 仅Stage2 + YCbCr + CSC | 36.44 | 0.9740 |
| Stage1+2 + RGB（无CSC） | 38.75 | 0.9838 |
| Stage1+2 + YCbCr + 固定变换 | 39.60 | 0.9857 |
| Stage1+2 + YCbCr + 可学习CSC | 39.88 | 0.9866 |
| **完整模型（+ IIG）** | **40.50** | **0.9881** |

**颜色空间对比（两阶段+固定变换）：**

| 颜色空间 | PSNR↑ | SSIM↑ |
|----------|-------|-------|
| RGB | 38.75 | 0.9838 |
| HSV | 39.03 | 0.9843 |
| HSL | 39.16 | 0.9844 |
| YUV | 39.19 | 0.9846 |
| **YCbCr** | **39.60** | **0.9857** |

### 关键发现

1. **Y通道优势明确**：YCbCr空间比RGB高0.85dB，比HSV高0.57dB，验证了"夜间雨在Y通道最显著"的统计观察
2. **可学习CSC的增益**：在YCbCr空间中，可学习CSC比固定变换高0.28dB，尤其在高亮区域避免像素损失
3. **IIG的贡献**：在可学习CSC基础上再加IIG，提升0.62dB（39.88→40.50），说明光照引导的必要性
4. **两阶段缺一不可**：单Stage1仅35.09dB，单Stage2为36.44dB，两阶段联合达39.88dB，架构设计合理
5. **数据集泛化性**：用HQ-NightRain训练IDT在RealRain-1k-L上达26.94dB，比GTAV-NightRain训练高0.47dB，证明光照感知合成有效缩小域差距
6. **多天气迁移**：CST-Net在Multi-Weather6K上比Restormer高2.02dB，说明Y通道策略不限于去雨

## 亮点与洞察

1. **统计驱动的物理洞察**：通过直方图对比发现夜间雨在Y通道差异最大，这个观察简洁、直觉、令人信服，且直接驱动了方法设计——是典型的"好观察→好方法"路线
2. **"可学习化"的极简设计**：将固定3×3变换矩阵替换为可学习参数+MLP，方法极其简单但有效。这种对常规操作做learnable化的思路值得借鉴
3. **完整的数据-方法协同设计**：不仅提出方法还提出数据集，且二者都围绕同一物理先验（雨的光照依赖性）展开，构成完整的解题闭环
4. **t-SNE域差距分析**：用ResNet50提取特征+t-SNE可视化来定量验证合成数据的真实性，方法论值得学习
5. **颜色空间选择的系统ablation**：对比了5种颜色空间（RGB/HSV/HSL/YUV/YCbCr），实验非常扎实

## 局限与展望

1. **CSC仍是全局变换**：可学习CSC是一个全局3×3矩阵，对所有像素施加相同变换。考虑到夜间场景光照分布的空间异质性极强，像素级或区域级自适应变换可能更优
2. **PSNR提升幅度有限**：在HQ-NightRain平均指标上，CST-Net对比DRSformer提升约0.33dB，优势不算压倒性
3. **合成数据的固有局限**：虽然t-SNE显示域差距减小，HQ-NightRain本质上仍是合成数据集，真实雨的复杂光学效应（散射、衍射、雾化）可能未完全模拟
4. **光照阈值的敏感性**：$\tau_1=0.2, \tau_2=0.8$ 是人工设定的，对不同类型光源（LED灯 vs 钠光灯）的适用性未讨论
5. **计算效率未报告**：两阶段Transformer + IIG的推理速度和模型大小未提供，对实时部署（如自动驾驶）的可行性不明

## 相关工作与启发

- **vs RLP (Zhang et al. 2023)**：RLP用循环残差网络学习雨的位置先验，但仍在RGB空间操作。CST-Net利用Y通道的亮度特性更有物理根据
- **vs NeRD-Rain (Chen et al. 2024)**：同样使用隐式神经表示，但NeRD-Rain用于RGB空间的双向去雨。CST-Net将INR限定在Y通道的光照编码上，定位更精准
- **vs Restormer/DRSformer**：通用去雨Transformer，不做颜色空间/光照的定制化。CST-Net验证了任务特定知识的价值
- **颜色空间启发**：YCbCr在去雨中的优势类似于HSV的V通道在低光增强中的作用（如RetinexFormer），暗示"选对通道比堆模型更重要"
- **可迁移思路**：可学习颜色空间变换可推广到去雾（雾在特定通道的特性）、水下图像恢复等任务；光照感知的数据合成管线也可用于其他夜间退化场景的数据生成

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Y通道统计洞察+可学习CSC+光照感知数据集三位一体，虽然单点技术不复杂，但问题分析到位、设计环环相扣
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5种颜色空间对比、完整模块消融、3类合成数据集+2个真实数据集+多天气扩展+下游检测应用+t-SNE域分析+特征可视化，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 动机从统计观察切入清晰有力，图表丰富（直方图、t-SNE、特征可视化、视觉对比），逻辑链完整
- **实用价值**: ⭐⭐⭐⭐ — 方法简单有效可复现，HQ-NightRain数据集对夜间去雨社区有开源价值，且在自动驾驶检测任务中验证了实用性
---
title: >-
  [论文解读] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation
description: >-
  [NEURIPS2025][图像恢复][夜间图像去雨] 提出CST-Net用于夜间图像去雨：基于夜间雨在Y通道（亮度）上比RGB更显著的观察，设计可学习颜色空间转换器(CSC)在YCbCr空间去雨，配合隐式光照引导模块(IIG)和新构建的光照感知合成数据集HQ-NightRain，在多个基准上达到SOTA。
tags:
  - NEURIPS2025
  - 图像恢复
  - 夜间图像去雨
  - 颜色空间变换
  - YCbCr
  - 光照引导
  - 数据集
---


<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](../../CVPR2025/image_restoration/hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[NeurIPS 2025\] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)

</div>

<!-- RELATED:END -->
