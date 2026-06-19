---
title: >-
  [论文解读] Visual Prompting for One-Shot Controllable Video Editing Without Inversion
description: >-
  [CVPR 2025][视频生成][视频编辑] 本文从视觉提示（Visual Prompting）的全新视角解决一次性可控视频编辑（OCVE）问题，通过图像修复扩散模型完成编辑传播，并提出内容一致性采样（CCS）和时序-内容一致性采样（TCS）两种采样策略，无需 DDIM 反演即可实现高质量可控视频编辑。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频编辑"
  - "视觉提示"
  - "DDIM反演"
  - "一致性模型"
  - "Stein变分梯度下降"
---

# Visual Prompting for One-Shot Controllable Video Editing Without Inversion

**会议**: CVPR 2025  
**arXiv**: [2504.14335](https://arxiv.org/abs/2504.14335)  
**代码**: [https://vp4video-editing.github.io/](https://vp4video-editing.github.io/)  
**领域**: 图像/视频生成 / 视频编辑  
**关键词**: 视频编辑, 视觉提示, DDIM反演, 一致性模型, Stein变分梯度下降

## 一句话总结

本文从视觉提示（Visual Prompting）的全新视角解决一次性可控视频编辑（OCVE）问题，通过图像修复扩散模型完成编辑传播，并提出内容一致性采样（CCS）和时序-内容一致性采样（TCS）两种采样策略，无需 DDIM 反演即可实现高质量可控视频编辑。

## 研究背景与动机

**领域现状**：一次性可控视频编辑（OCVE）是一种高效的视频编辑范式——用户只需用任意图像编辑工具编辑视频的第一帧，系统自动将编辑传播到后续帧。现有 OCVE 方法（如 AnyV2V、Videoshop）普遍依赖 DDIM 反演将源视频转换为噪声潜变量，再注入编辑引导生成编辑后的视频。

**现有痛点**：(1) DDIM 反演在每一步引入近似误差，累积后导致重建质量退化，进而削弱编辑后视频的内容一致性。具体来说，反演过程中用 $\epsilon_\theta(z_{t-1}, t)$ 近似 $\epsilon_\theta(z_t, t)$ 是不精确的。(2) 为保持时序一致性，一些方法引入视频扩散模型提供时序先验，但开源视频数据集质量有限，且视频扩散模型计算量极大。

**核心矛盾**：DDIM 反演是现有 OCVE 方法的基础但也是瓶颈——它是重建源视频信息的关键手段，但其累积误差又是内容不一致的根源。

**本文目标**：完全绕过 DDIM 反演，找到一种无需将源帧反演为噪声的视频编辑框架，同时保证内容一致性和时序一致性。

**切入角度**：作者发现 OCVE 和视觉提示（Visual Prompting）本质上都在做同一件事——将某种修改跨图像传播。在视觉提示中，示例图像对展示了变换规则，查询图像被施加相同变换；在 OCVE 中，第一帧的编辑需要传播到后续帧。

**核心 idea**：将 OCVE 重新定义为视觉提示任务，利用预训练图像修复扩散模型的视觉推理能力完成编辑传播，用一致性模型的多步一致性采样保证内容一致性，用 Stein 变分梯度下降保证时序一致性。

## 方法详解

### 整体框架

整体流程分三步：(1) 将第一帧源图和编辑图作为示例对、当前源帧作为查询、空白区域作为待生成编辑帧，组织成 2×2 网格输入图像修复扩散模型；(2) 通过 CCS 生成与源帧内容一致的编辑帧；(3) 通过 TCS 对所有编辑帧进行时序一致性调整。输入为源视频 + 第一帧编辑结果，输出为完整编辑视频。

### 关键设计

1. **视觉提示与修复扩散模型的结合**:

    - 功能：将 OCVE 任务转化为修复扩散模型可处理的视觉提示格式，绕过 DDIM 反演
    - 核心思路：将输入信息 $G(i)$ 组织为 2×2 网格——左上放第一帧源图，右上放第一帧编辑图（作为视觉提示示例），左下放第 $i$ 帧源图（查询），右下留空由模型生成（编辑输出）。掩码 $M$ 对应标记右下区域为需生成区域。文本提示 $p$ 用 CLIP 编码的编辑方向差异向量代替：$p = \lambda_1 \cdot \{E_{CLIP}(I^e(1)) - E_{CLIP}(I^s(1))\}$。关键在于源帧通过编码器直接编码为特征输入（而非反演为噪声），从根源避免了反演误差
    - 设计动机：修复扩散模型天然擅长在保持上下文一致性的同时补全缺失区域，这与 OCVE"根据示例推断编辑"的需求高度吻合。将编辑方向用 CLIP 嵌入差异表示，比文本描述更精确地捕获编辑意图

2. **内容一致性采样（CCS）**:

    - 功能：基于一致性模型的多步一致性采样，确保生成的编辑帧与源帧保持内容一致性
    - 核心思路：首先修改修复扩散模型的采样方程——设 $\sigma_t = \sqrt{1-\alpha_{t-1}}$ 消除 adjustment 项，变为非马尔科夫过程。引入一致性噪声 $\epsilon^c$ 替代参数化噪声，构建一致性模型 $\hat{f}(z_t, t, \epsilon^c(t)) = (z_t - \sqrt{1-\alpha_t}\epsilon^c(t))/\sqrt{\alpha_t}$。CCS 的关键操作：(a) 在第一个时间步强制生成源帧，通过计算对应的一致性噪声 $\epsilon^c(t; z_0^s) = (\hat{z}_t - \sqrt{\alpha_t}z_0^s)/\sqrt{1-\alpha_t}$ 实现；(b) 利用噪声校准机制引导生成逐步从源帧过渡到目标编辑帧，使用去噪差异 $\Delta\epsilon_t = \epsilon_\theta(z_t(I^e), t) - \epsilon_\theta(z_t(I^s), t)$ 作为编辑方向信号
    - 设计动机：普通扩散采样的马尔科夫性质使得各时间步生成的图像相互独立，无法保证内容一致性。一致性模型的核心特性是"同一轨迹上的所有点映射到相同的初始状态"，借此保证生成的编辑帧与源帧保持一致

3. **时序-内容一致性采样（TCS）**:

    - 功能：将所有编辑帧作为分布样本，通过 SVGD 优化使其逼近源帧的时序分布，保证帧间时序一致性
    - 核心思路：将源帧视为从某分布中的 $N$ 个样本 $\{z(i)\}_{i=1}^N$，CCS 生成的编辑帧 $\{\hat{z}^{(0)}(i)\}_{i=1}^N$ 需要调整以逼近该分布。使用 SVGD 进行确定性更新：$\hat{z}_{\ell-1}^{(0)}(i) = \hat{z}_\ell^{(0)}(i) - \eta \cdot \hat{\phi}(\hat{z}_\ell^{(0)}(i))$，其中 $\hat{\phi}$ 包含两项——基于所有样本平均梯度的驱动项（保证优化稳定性）和基于 RBF 核的排斥项（防止模式坍缩）
    - 设计动机：CCS 独立处理每一帧，未显式建模帧间时序关系。TCS 通过 SVGD 的确定性迭代，比视频扩散模型更快且不依赖视频训练数据。将视频帧视为分布样本并用 SVGD 优化的思路非常巧妙

### 损失函数 / 训练策略

CCS 和 TCS 都是推理时的采样策略，无需额外训练。CCS 使用 30 个时间步，TCS 使用 50 个时间步。超参数 $\lambda_1 = 0.7$、$\lambda_2 = 1.2$、$\eta = 2.0$。使用 Stable Diffusion Inpainting 1.5 作为基础模型。

## 实验关键数据

### 主实验

在 MagicBrush 衍生数据集上的定量对比（10388 个编辑元组）：

| 方法 | CLIPtar↑ | TIFA↑ | CLIPsrc↑ | Flow↓ | FVD↓ | CLIPTC↑ | 时间(s)↓ |
|------|---------|-------|---------|------|-----|--------|---------|
| AnyV2V | 87.1 | 67.0 | 91.3 | 24.6 | 17.1 | 93.9 | 149 |
| Videoshop | 88.8 | 64.4 | 91.0 | 19.0 | 14.8 | 95.2 | 32 |
| **Ours** | **90.1** | **69.1** | **93.2** | **21.9** | **15.2** | **97.1** | **19** |

### 消融实验

| 配置 | CLIPtar↑ | CLIPsrc↑ | CLIPTC↑ | 时间(s)↓ |
|------|---------|---------|--------|---------|
| w/o CCS | 80.3 | 81.3 | 95.8 | 19 |
| w/o TCS | 89.8 | 92.8 | 89.8 | 18 |
| Full model | **90.1** | **93.2** | **97.1** | 19 |

### 关键发现

- 去掉 CCS 后，源帧保真度（CLIPsrc）从 93.2 大幅降至 81.3，证明 CCS 对内容一致性至关重要
- 去掉 TCS 后，时序一致性（CLIPTC）从 97.1 降至 89.8（下降 7.3 个百分点），证明 TCS 有效保证了时序平滑性
- 本方法仅需 19 秒处理一段视频，比 AnyV2V (149s) 快约 8 倍，比 Videoshop (32s) 快约 1.7 倍
- 使用轻量图像扩散模型替代视频扩散模型是速度优势的关键来源

## 亮点与洞察

- **视角转换的巧妙性**：将 OCVE 重新定义为视觉提示任务是本文最核心的创新。这一视角转换直接消除了 DDIM 反演的需求，从根本上解决了累积误差问题
- **无需训练的采样策略**：CCS 和 TCS 都是推理时的采样方法修改，不需要额外训练。这意味着可以即插即用到任何兼容的扩散模型中
- **SVGD 用于时序一致性**：将视频帧视为分布样本、用 SVGD 确保分布匹配的思路非常新颖。相比视频扩散模型，SVGD 的确定性更新既高效又不依赖视频数据
- **CLIP 嵌入差异作为编辑方向**：比文本描述更准确地捕获编辑意图，且天然兼容基于 CLIP 文本编码器的扩散模型

## 局限与展望

- 基于 SD Inpainting 1.5，分辨率受限，未适配更新的扩散模型架构
- 2×2 网格布局将有效分辨率压缩为原来的 1/4，可能影响细节保真度
- TCS 的 SVGD 更新需要同时处理所有帧，对长视频的内存和计算需求较大
- 编辑范围受限于修复扩散模型的视觉推理能力，对于大幅度的结构性编辑可能力不从心
- 未探索与视频生成模型（如 Sora）的结合潜力

## 相关工作与启发

- **vs AnyV2V**: AnyV2V 使用 DDIM 反演 + 视频扩散模型，计算量大（149s）且受反演误差影响。本文绕过反演、使用图像扩散模型，速度快 8 倍且质量更优
- **vs Videoshop**: Videoshop 同样使用 DDIM 反演但优化了特征注入方式，本文从根本上消除了反演需求，在编辑保真度和源帧保真度上均超越
- **vs 一致性模型**: 本文创造性地将一致性模型的多步一致性采样性质嫁接到修复扩散模型上，无需训练一致性模型，只需修改采样方程

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 视觉提示视角解决OCVE的思路非常新颖，CCS和TCS的设计都有扎实的理论支撑
- 实验充分度: ⭐⭐⭐⭐ 大规模数据集评估全面，消融设计合理，但缺少人工评估的详细报告
- 写作质量: ⭐⭐⭐⭐ 动机论述清晰，方法推导完整但数学符号较密集
- 价值: ⭐⭐⭐⭐⭐ 绕过DDIM反演的新范式，速度和质量的双重提升使其极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VACE: All-in-One Video Creation and Editing](../../ICCV2025/video_generation/vace_all-in-one_video_creation_and_editing.md)
- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](sketchvideo_sketch-based_video_generation_and_editing.md)

</div>

<!-- RELATED:END -->
