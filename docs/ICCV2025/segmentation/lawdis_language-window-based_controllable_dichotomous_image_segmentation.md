---
title: >-
  [论文解读] LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation
description: >-
  [ICCV 2025][语义分割][二分图像分割] 提出 LawDIS，一个基于潜在扩散模型的可控二分图像分割框架，通过宏观语言控制（LS）和微观窗口细化（WR）两种模式的协同，实现高质量前景目标掩码生成，在 DIS5K 基准上全面超越 11 种 SOTA 方法。 二分图像分割（DIS）旨在从高分辨率图像中精确分割前景目标…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "二分图像分割"
  - "潜在扩散模型"
  - "语言控制"
  - "窗口细化"
  - "高精度分割"
---

# LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation

**会议**: ICCV 2025  
**arXiv**: [2508.01152](https://arxiv.org/abs/2508.01152)  
**代码**: [GitHub](https://github.com/XinyuYanTJU/LawDIS)  
**领域**: 图像分割  
**关键词**: 二分图像分割, 潜在扩散模型, 语言控制, 窗口细化, 高精度分割

## 一句话总结

提出 LawDIS，一个基于潜在扩散模型的可控二分图像分割框架，通过宏观语言控制（LS）和微观窗口细化（WR）两种模式的协同，实现高质量前景目标掩码生成，在 DIS5K 基准上全面超越 11 种 SOTA 方法。

## 研究背景与动机

二分图像分割（DIS）旨在从高分辨率图像中精确分割前景目标，要求像素级高精度的轮廓描绘。随着高质量相机设备的普及，分割任务已从粗糙定位演进到高精度边界刻画。DIS 在 3D 重建、图像编辑、增强现实和医学图像分割等领域有广泛应用。

现有 DIS 方法面临两个核心挑战：

**语义歧义问题**：当图像包含多个前景实体时，判别式学习范式（逐像素分类）无法灵活指定目标对象，缺乏用户交互能力

**几何细节捕获瓶颈**：为捕获高分辨率目标的几何细节，现有方法通常引入额外高分辨率数据流或将图像分割为固定大小 patch，但无法适应可变 patch 尺寸。如 MVANet 在非训练尺寸的局部 patch 上表现显著退化

本文核心动机：将 DIS 重新定义为**图像条件掩码生成任务**，利用潜在扩散模型的生成能力，无缝集成用户控制，解决上述两类问题。

## 方法详解

### 整体框架

LawDIS 基于预训练 Stable Diffusion v2，将 DIS 重构为条件去噪扩散过程。框架包含三个核心组件：
- **模式切换器（Mode Switcher）**：一维向量经位置编码后加到扩散模型的时间嵌入上，切换宏观/微观模式
- **宏观模式（Macro Mode）**：语言控制分割策略（LS），根据用户语言提示生成初始掩码
- **微观模式（Micro Mode）**：窗口控制细化策略（WR），在用户指定的可变尺寸窗口内精细化掩码

### 关键设计一：生成式 DIS 范式

将 DIS 建模为条件概率分布 $D(s|x)$，其中 $s$ 为分割掩码、$x$ 为 RGB 图像。使用 VAE 编码器 $\phi$ 将分割掩码和图像映射到低维潜在空间，在潜在空间中执行扩散过程：

- **前向过程**：从 $\mathbf{z}_0^{(s)}$ 起逐步添加高斯噪声，构建离散马尔科夫链
- **反向过程**：U-Net $f_\theta$ 在每个时间步预测噪声，结合图像条件 $\mathbf{z}^{(x)}$ 逐步去噪
- **架构修改**：复制 U-Net 输入层以匹配拼接的图像特征和噪声掩码特征，复制权重后减半以避免激活值膨胀

### 关键设计二：双模式联合训练

**宏观模式训练**：激活 $\psi_a$，模型接收完整图像 $x$、分割掩码 $s$ 和 VLM 生成的语言提示 $\mathcal{T}$。使用 CLIP 编码提示为控制嵌入 $c_\mathcal{T}$，通过交叉注意力注入 U-Net：
$$\mathcal{L}_{macro} = \|\boldsymbol{\epsilon} - f_\theta(\mathbf{z}_t^{(s)}, \mathbf{z}^{(x)}, c_\mathcal{T}, t, \psi_a)\|_2^2$$

**微观模式训练**：激活 $\psi_b$，选取前景目标的最小外接矩形作为局部窗口，裁剪得到局部 patch $x_p$ 和局部掩码 $s_p$，使用空提示 $c_\varnothing$ 避免语义不匹配：
$$\mathcal{L}_{micro} = \|\boldsymbol{\epsilon}_p - f_\theta(\mathbf{z}_t^{(s_p)}, \mathbf{z}^{(x_p)}, c_\varnothing, t, \psi_b)\|_2^2$$

**联合训练**：$\mathcal{L}_u = \mathcal{L}_{macro} + \mathcal{L}_{micro}$，两种模式共享同一 U-Net，实现协同增强。

### 关键设计三：VAE 解码器微调

训练完 U-Net 后，冻结编码器和 U-Net，仅微调 VAE 解码器 $\varphi$：
- 添加编码器到解码器的快捷连接
- 输出通道从 3 降为 1（掩码单通道），权重通过通道平均初始化
- 引入 **TCD（Trajectory Consistency Distillation）** 调度器，将采样过程简化为单步，既节省显存又提升推理效率

### 损失函数

VAE 解码器微调使用结构损失：
$$\mathcal{L}_d = \mathcal{L}_{wbce}(\hat{s}, s) + \mathcal{L}_{wiou}(\hat{s}, s)$$
分别为加权二元交叉熵损失和加权 IoU 损失。

### 推理流程

1. **语言控制分割**（宏观模式）：输入完整图像 + 语言提示 → 单步 TCD 去噪 → 解码得到初始分割图
2. **窗口控制细化**（微观模式，可选）：用户选定不满意区域 → 裁剪局部 patch → 以初始分割结果（而非纯噪声）作为扩散起点 → 单步去噪 → 精细化掩码替换原始区域。可无限重复直到满意

## 实验关键数据

### 主实验：DIS5K 基准（DIS-TE 2000 张）

| 方法 | $F_\beta^\omega$ ↑ | $F_\beta^{mx}$ ↑ | $\mathcal{M}$ ↓ | $\mathcal{S}_\alpha$ ↑ | $E_\phi^{mn}$ ↑ |
|------|-----|-----|------|------|------|
| IS-Net (2022) | 0.726 | 0.799 | 0.070 | 0.819 | 0.858 |
| InSPyReNet (2022) | 0.838 | 0.891 | 0.039 | 0.900 | 0.923 |
| BiRefNet (2024) | 0.858 | 0.896 | 0.035 | 0.901 | 0.934 |
| GenPercept (2024) | 0.816 | 0.868 | 0.043 | 0.880 | 0.923 |
| MVANet (2024) | 0.862 | 0.907 | 0.034 | 0.909 | 0.938 |
| **Ours-S（仅 LS）** | **0.898** | **0.929** | **0.027** | **0.925** | **0.955** |
| **Ours-R（LS+WR）** | **0.908** | **0.932** | **0.024** | **0.926** | **0.959** |

- Ours-S 在 DIS-TE1 上 $F_\beta^\omega$ 超越 MVANet 6.6%，Ours-R 进一步在 DIS-TE4 提升 2.0%
- 整合双控制后，在 DIS-TE1 上相比 MVANet 提升 7.0%

### 消融实验

| 设置 | $F_\beta^{mx}$ ↑ | $\mathcal{M}$ ↓ | $\mathcal{S}_\alpha$ ↑ | $E_\phi^{mn}$ ↑ |
|------|------|------|------|------|
| Baseline（无模式切换/提示/VAE微调） | 0.904 | 0.047 | 0.904 | 0.916 |
| 无微观模式训练 | 0.912 | 0.037 | 0.909 | 0.943 |
| 无 VAE 解码器微调 | 0.919 | 0.040 | 0.915 | 0.933 |
| **完整 Ours-S** | **0.926** | **0.032** | **0.920** | **0.955** |

**宏观控制消融**：

| 设置 | $F_\beta^{mx}$ ↑ | $\mathcal{M}$ ↓ | $\mathcal{S}_\alpha$ ↑ |
|------|------|------|------|
| 无提示（训练+测试） | 0.912 | 0.036 | 0.908 |
| 无提示（仅测试） | 0.915 | 0.036 | 0.909 |
| **有提示（训练+测试）** | **0.926** | **0.032** | **0.920** |

**微观控制消融**：

| 设置 | $F_\beta^\omega$ | $\mathcal{M}$ | $BIoU^m$ ↑ | $HCE_\gamma$ ↓ |
|------|------|------|------|------|
| 基础 Ours-S | 0.890 | 0.032 | 0.795 | 2481 |
| 从高斯噪声初始化 | -4.7% | +1.9% | -7.1% | -863 |
| 自动窗口选择 | +1.7% | -0.5% | +2.9% | -767 |
| 半自动窗口选择 | +2.0% | -0.6% | +3.2% | -871 |

### 关键发现

- 从初始分割结果（而非高斯噪声）启动微观模式的扩散过程至关重要，可间接传递全局上下文信息
- 即使无用户干预的全自动窗口选择，WR 策略仍能有效提升边界精度
- 双模式联合训练提供了可扩展的几何表示能力，使模型适应不同输入尺寸
- VAE 解码器微调对高分辨率分割至关重要，能用细粒度细节补充去噪掩码特征

## 亮点与洞察

1. **范式转换**：首次将 DIS 从判别式逐像素分类重新定义为生成式掩码生成问题，利用扩散模型的百科全书式视觉-语言理解能力
2. **可控性设计**：宏观语言控制解决"分割哪个目标"的语义歧义，微观窗口细化解决"边界不够精细"的几何精度问题，两者通过模式切换器统一在一个扩散模型中
3. **高效推理**：引入 TCD 调度器实现单步去噪，使扩散模型在分割任务中具有实际可用的推理速度
4. **初始化策略的巧妙设计**：微观模式使用宏观模式的分割结果（而非纯噪声）作为扩散起点，间接在两种模式之间传递上下文信息，这是性能提升的关键
5. **灵活的用户交互**：窗口细化可无限重复，支持渐进式精化，适合高精度个性化应用场景

## 局限性

- 基于 Stable Diffusion v2，模型参数量较大，部署成本高于传统判别式方法
- 语言提示依赖 VLM 自动生成或用户手动提供，在完全自动化场景中需要额外的提示生成模块
- 微观模式的窗口选择在全自动模式下仍依赖边缘检测启发式，效果略弱于用户手动选择
- 所有输入统一 resize 到 $1024^2$，对极端长宽比图像可能引入形变
- 仅在 DIS5K 单一基准上评估，对其他高精度分割任务（如人像抠图、医学分割）的泛化性未验证

## 相关工作

- **二分图像分割**：IS-Net 引入中间监督；FP-DIS 利用频率先验；BiRefNet/MVANet 通过多分辨率 patch 增强细节；但这些判别式方法缺乏灵活的语义控制和局部窗口自适应能力
- **扩散模型用于分割**：GenPercept 将生成模型转为确定性单步范式；Wang et al. 提出扩散细化模型增强掩码质量；本文首次将单个 Stable Diffusion 扩展为宏观+微观双模式
- **高分辨率分割**：InSPyReNet 和 BiRefNet 通过额外分辨率数据流增强细节，MVANet 使用固定大小 patch 分割，但均缺乏对可变 patch 尺寸的自适应能力

## 评分

- 新颖性：⭐⭐⭐⭐ — 将 DIS 重构为可控生成任务是有创意的范式转换；双模式切换器的设计简洁有效
- 技术深度：⭐⭐⭐⭐ — 扩散模型的适配（输入层复制、TCD 单步推理、VAE 微调策略）考虑周全，微观模式的初始化策略设计巧妙
- 实验充分度：⭐⭐⭐⭐ — DIS5K 上全指标超越 11 种方法，消融覆盖各组件；但缺少其他数据集和效率对比
- 写作质量：⭐⭐⭐⭐ — 结构清晰，宏观-微观的叙述逻辑自然
- 推荐度：⭐⭐⭐⭐ — 为高精度分割引入了新的生成式范式，实验结果具有说服力

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching](../../CVPR2026/segmentation/flowdis_language-guided_dichotomous_image_segmentation_with_flow_matching.md)
- [\[ICCV 2025\] Refer to Any Segmentation Mask Group With Vision-Language Prompts](refer_to_any_segmentation_mask_group_with_vision-language_prompts.md)
- [\[CVPR 2026\] High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy](../../CVPR2026/segmentation/high-precision_dichotomous_image_segmentation_via_depth_integrity-prior_and_fine.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)

</div>

<!-- RELATED:END -->
