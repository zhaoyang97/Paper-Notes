---
title: >-
  [论文解读] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement
description: >-
  [ICCV 2025][图像恢复][图像退化建模] 提出首个通用图像退化模型，通过"压缩解纠缠"方法分离退化信息与图像内容，引入 IDEN 和 IDA 层处理非均匀退化，实现跨退化类型的编码、合成和迁移，可作为 plug-in 模块将非盲图像恢复方法转化为盲方法。 图像退化合成在图像恢复、艺术效果模拟等领域有重要应用…
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "图像退化建模"
  - "退化解纠缠"
  - "非均匀退化"
  - "盲图像恢复"
  - "胶片颗粒模拟"
---

# Towards a Universal Image Degradation Model via Content-Degradation Disentanglement

**会议**: ICCV 2025  
**arXiv**: [2505.12860](https://arxiv.org/abs/2505.12860)  
**代码**: 无（将在作者 GitHub 发布）  
**领域**: 图像复原  
**关键词**: 图像退化建模, 退化解纠缠, 非均匀退化, 盲图像恢复, 胶片颗粒模拟

## 一句话总结

提出首个通用图像退化模型，通过"压缩解纠缠"方法分离退化信息与图像内容，引入 IDEN 和 IDA 层处理非均匀退化，实现跨退化类型的编码、合成和迁移，可作为 plug-in 模块将非盲图像恢复方法转化为盲方法。

## 研究背景与动机

图像退化合成在图像恢复、艺术效果模拟等领域有重要应用。现有退化模型的根本局限在于：

**只针对特定退化类型**：噪声、降采样、雨、雾等各有专用模型，无法通用化

**需要用户提供退化参数**：如噪声级别、模糊核等，在盲恢复场景下不实际

**无法处理非均匀退化**：真实世界退化通常空间变化（如局部雾、雨滴等），现有模型假设退化是全局均匀的

**无法组合多种退化**：复杂退化（如同时有噪声+模糊+压缩+非均匀退化）难以建模

唯一尝试非退化特定建模的 Chen et al. 仍需为每种退化训练单独模型，且不支持非均匀退化和随机性。

## 方法详解

### 整体框架

系统包含两个退化编码网络（HDEN 和 IDEN）和一个退化合成网络。给定退化图像 $\mathbf{y}$，从中提取均匀退化嵌入 $\mathbf{e}_g = e_g(\mathbf{y})$ 和非均匀退化嵌入 $\mathbf{e}_l = e_l(\mathbf{y})$，然后将退化施加到干净图像 $\mathbf{x}$ 上：$\hat{\mathbf{y}} = \hat{f}(\mathbf{x}, \mathbf{e}_g, \mathbf{e}_l, \mathbf{n})$，其中 $\mathbf{n}$ 为随机状态。

### 关键设计

1. **均匀/非均匀退化编码（HDEN & IDEN）**：

    - **HDEN（均匀退化编码网络）**：双分支架构，短程分支在原始分辨率操作（捕获噪声等小感受野退化），长程分支在降采样分辨率操作（捕获模糊等大感受野退化）。输出经 MLP 融合为全局退化向量 $\mathbf{e}_g$
    - **IDEN（非均匀退化编码网络）**：关键创新——修改长程分支并用 CNN 替换 MLP 尾部，保留空间结构信息。输出 $\mathbf{e}_l$ 是一个保持空间维度的退化图，能表示空间变化的退化
    - 设计动机：不同退化需要不同感受野，双分支设计灵活适应；IDEN 保留空间信息是处理非均匀退化的前提

2. **IDA-SFT 退化合成层**：

    - **IDA（Inhomogeneous Degradation-Aware）层**：一种高效的空间变化卷积近似方案。理想情况下应使用空间变化核，但计算复杂度过高。IDA 通过深度卷积+下采样+逐元素乘法+反卷积实现：
    $\text{IDA}(\mathbf{F}_{in}, \mathbf{e}) = \text{DConv}(\text{DS}(\text{DConv}(\mathbf{e})) \odot \text{DS}(\mathbf{F}_{in}))$
    - 论文证明单个 IDA 层比四个深度可分离卷积层更有表达力
    - **IDA-SFT 复合层**：IDA 与 SFT（Spatial Feature Transform）并行组合：
    $\text{IDA-SFT}(\mathbf{F}_{in}, \mathbf{e}, \mathbf{n}) = \text{IDA}(\mathbf{F}_{in}, \mathbf{e}) + \alpha(\mathbf{e}) \odot \mathbf{F}_{in} + \beta(\mathbf{e}, \mathbf{n})$
   SFT 擅长引入随机状态和均匀退化，IDA 擅长非均匀退化，互补组合
    - 合成网络采用 U-Net 结构，包含多个 IDA-SFT block

3. **压缩解纠缠（Disentangle-by-Compression）**：

    - 核心创新：通过最小化退化嵌入的**边际熵之和**来实现三重解纠缠：
    $\mathcal{L}_{rate\_g} = \sum_i H(e_g^{(i)}), \quad \mathcal{L}_{rate\_l} = \sum_{i,j} H(e_l^{(i,j)})$
    - 信息论证明：$\sum_i H(e^{(i)}) = H(\mathbf{e}) + D_{KL}(p(\mathbf{e}) \| q(\mathbf{e}))$
        - 最小化 $H(\mathbf{e})$：由于 $H(\mathbf{e}) = I(\mathbf{e}; \mathbf{x}) + H(\mathbf{d})$，最小化嵌入熵等价于减少嵌入与图像内容的互信息 → **分离退化与内容**
        - 最小化 $D_{KL}$：使嵌入各维度独立 → **各退化分量解耦**
    - 均匀/非均匀同理分别约束 → **分离均匀与非均匀退化**
    - 概率密度用学习方法估计（分别用不同的密度估计器处理 $e_g$ 和 $e_l$）

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L} = \mathcal{L}_{sim} + \lambda_g \mathcal{L}_{rate\_g} + \lambda_l \mathcal{L}_{rate\_l} + \lambda_c \mathcal{L}_{contra} + \lambda_r \mathcal{L}_{color} + \lambda_d \mathcal{L}_{diver} + \lambda_g \mathcal{L}_{gan}$$

- $\mathcal{L}_{sim}$：DISTS 感知距离（对噪声随机状态最不敏感）
- $\mathcal{L}_{diver}$：多样性损失 $= -\text{SSIM}(\hat{\mathbf{y}}, \hat{\mathbf{y}}')$，鼓励不同随机状态产生不同输出
- $\mathcal{L}_{gan}$：对抗损失，增强生成真实感
- $\mathcal{L}_{contra}$：对比损失；$\mathcal{L}_{color}$：颜色保持损失
- 训练数据：300K Wikipedia Quality Images，筛选 40K 训练对，退化按典型图像处理流水线随机组合

## 实验关键数据

### 主实验——退化复现与迁移（WQI 测试集）

| 任务 | MS-SSIM↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|------|----------|-------|--------|--------|
| 退化复现 | 0.879 | 0.860 | 0.295 | 0.141 |
| 退化迁移 | 0.875 | 0.856 | 0.306 | 0.147 |

迁移性能接近复现，证明退化信息有效从内容中分离。

### 消融实验

压缩解纠缠的效果（LPIPS↓）：

| 模型 | Direct 迁移 | Mixed 迁移 |
|------|-------------|------------|
| 完整模型 | 0.271 | 0.286 |
| 无熵正则（无解纠缠） | 0.290 (+0.018) | 0.334 (+0.048) |

IDA 和 IDEN 的效果（LPIPS↓）：

| 模型 | Global-only | Direct | Mixed |
|------|-------------|--------|-------|
| 完整模型 | 0.388 | 0.271 | 0.286 |
| 无 IDA | 0.404 | 0.298 | 0.356 |
| 无 IDA + 无 IDEN | 0.394 | 0.577 | 0.588 |

移除 IDEN 后 Direct/Mixed 迁移性能断崖式下降（+0.3 LPIPS），说明非均匀退化建模至关重要。

### 盲图像恢复（RSG + 我们的退化模型 vs 原始盲 RSG）

| 退化组合 | 准确度(LPIPS↓) w/o→w/ | 保真度(LPIPS↓) w/o→w/ | 真实度(pFID↓) w/o→w/ |
|----------|---------------------|---------------------|-------------------|
| NA（噪声+伪影） | 0.680→**0.513** | 0.424→**0.334** | 228.8→**28.1** |
| NP（噪声+修补） | 0.713→**0.485** | 0.187→**0.081** | 221.9→**20.6** |
| UNAP（全部4种） | 0.666→**0.560** | 0.251→**0.141** | 147.4→**31.7** |

作为 plug-in 后，盲恢复质量大幅提升，尤其是 pFID 从 200+ 降到 20-60。

### 关键发现

- 模型自动学习到 5 个有意义的退化嵌入维度，每个维度控制一组相关退化
- 迁移性能与复现性能接近，验证了退化与内容的有效解纠缠
- 胶片颗粒模拟中，通用模型的复现分数甚至优于专用模型
- 非均匀退化（如雨滴）的迁移成功证明了 IDEN 和 IDA 的必要性

## 亮点与洞察

- **理论支撑扎实**：压缩解纠缠有信息论的严格证明，不是靠直觉的启发式设计
- **首创性强**：首个同时处理均匀和非均匀退化组合的通用模型
- **应用价值大**：作为 plug-in 将非盲恢复转为盲恢复，无需改变恢复方法本身
- **IDA 层设计精妙**：以远小于空间变化卷积的计算代价实现了更强的表达能力

## 局限与展望

- 训练数据的退化类型和组合仍是有限的合成数据，对更极端的真实退化泛化能力待验证
- 恢复实验仅在人脸图像（FFHQ）上做，因底层 GAN 限制；DPS 实验因算力受限只有定性结果
- 退化嵌入维度数量由模型自动决定（通过方差分析确定活跃维度），但对维度数量的控制有限
- 未探索视频退化场景的时域一致性

## 相关工作与启发

- 与 Style Transfer 的关键区别：退化具有随机性（噪声模式各异）、需要保持图像内容（不改变肤色/纹理）、需要处理非均匀分布
- 压缩解纠缠方法受神经图像压缩文献启发，将率失真优化中的熵约束用于退化分离
- 启发：disentangle-by-compression 的思想可推广到其他需要分离不同信息源的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个通用退化模型，理论方法和架构设计均有创新
- **实验充分度**: ⭐⭐⭐⭐ 退化迁移/复现/胶片颗粒/盲恢复多任务验证，消融详细
- **写作质量**: ⭐⭐⭐⭐ 理论推导严谨，但正文受限于篇幅、大量内容在补充材料中
- **价值**: ⭐⭐⭐⭐⭐ 解决了一个长期存在的基础性问题，有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] UniRes: Universal Image Restoration for Complex Degradations](unires_universal_image_restoration_for_complex_degradations.md)
- [\[CVPR 2026\] Degradation-Robust Fusion: An Efficient Degradation-Aware Diffusion Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios](../../CVPR2026/image_restoration/degradation-robust_fusion_an_efficient_degradation-aware_diffusion_framework_for.md)
- [\[ECCV 2024\] Spatially-Variant Degradation Model for Dataset-free Super-resolution](../../ECCV2024/image_restoration/spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](../../CVPR2025/image_restoration/visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)

</div>

<!-- RELATED:END -->
