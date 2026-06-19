---
title: >-
  [论文解读] Balanced Image Stylization with Style Matching Score
description: >-
  [ICCV 2025][图像生成][Style Transfer] 提出 Style Matching Score（SMS），将图像风格化重铸为风格分布匹配问题，通过渐进频谱正则化和语义感知梯度精炼，在风格对齐与内容保持之间取得卓越平衡，并可蒸馏到轻量前馈网络实现一步风格化。 图像风格化的核心挑战是在有效转移目标风格与保持源…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Style Transfer"
  - "Score Distillation"
  - "扩散模型"
  - "LoRA"
  - "Frequency Domain Regularization"
---

# Balanced Image Stylization with Style Matching Score

**会议**: ICCV 2025  
**arXiv**: [2503.07601](https://arxiv.org/abs/2503.07601)  
**代码**: [https://github.com/showlab/SMS](https://github.com/showlab/SMS)  
**领域**: Image Generation / Style Transfer  
**关键词**: Style Transfer, Score Distillation, diffusion model, LoRA, Frequency Domain Regularization

## 一句话总结

提出 Style Matching Score（SMS），将图像风格化重铸为风格分布匹配问题，通过渐进频谱正则化和语义感知梯度精炼，在风格对齐与内容保持之间取得卓越平衡，并可蒸馏到轻量前馈网络实现一步风格化。

## 研究背景与动机

图像风格化的核心挑战是在有效转移目标风格与保持源图内容之间取得平衡。现有方法各有短板：

**零样本文本驱动**（如 FreeStyle）：文本难以精确描述复杂风格，"一图胜千言"

**单样本示例驱动**（如 StyleID）：过度依赖单张风格图，容易产生纹理叠加而非真正风格迁移

**集合微调方法**（如 Style-LoRA + ControlNet）：虽能捕捉风格分布，但 ControlNet 的边缘条件限制了内容保持

**SDS 类编辑方法**（如 DDS）：缺乏显式身份保持机制，容易偏离源图

SMS 的核心 idea：利用风格 LoRA 整合的扩散模型估计目标风格分布的分数函数，通过最小化生成分布与风格分布的 KL 散度来实现风格匹配，同时用频域正则化和语义梯度修正来保持内容。

## 方法详解

### 整体框架

SMS 包含三个核心组件：
1. **Style Matching Objective**：基于分数蒸馏的风格分布匹配
2. **Progressive Spectrum Regularization**：频域渐进正则化保护内容
3. **Semantic-Aware Gradient Refinement**：语义引导的梯度修正

### 关键设计

1. **风格匹配目标（Style Matching Objective）**

    - 参数化生成器 $G_\theta$（可以是图像本身或前馈网络的参数）
    - 最小化生成分布 $p_{G_\theta}$ 与目标风格分布 $p_{style}$ 的 KL 散度
    - 使用两个噪声预测模型估计分数函数：
        - $\epsilon_{style}$：集成了风格 LoRA 的预训练扩散模型（固定），估计 $p_{style}$
        - $\epsilon_{fake}^\phi$：动态学习，估计当前 $p_{G_\theta}$
    - KL 散度梯度：
    $\nabla_\theta D_{KL} \approx \mathbb{E}_{t,\epsilon}[w_t(\epsilon_{style}(z_t^{tgt}; y_{src}, t) - \epsilon_{fake}^\phi(z_t^{tgt}; y_{src}, t))\frac{\partial G_\theta}{\partial \theta}]$
    - 与 DMD 的区别：DMD 用通用预训练模型 $\epsilon_{real}$，SMS 换成风格 LoRA 模型 $\epsilon_{style}$

2. **渐进频谱正则化（Progressive Spectrum Regularization）**

    - 观察：实图与风格化图的主要差异在高频成分
    - 核心思想：在高噪声步（大 $t$）保护更多低频成分保持结构，在低噪声步（小 $t$）允许高频修改添加风格细节
    - 损失：$L_{freq} = \|\mathcal{F}_{low}(z_0^{tgt}, t) - \mathcal{F}_{low}(z_0^{src}, t)\|_2^2$
    - 其中 $\mathcal{F}_{low}(z, t) = \text{LPF}(\text{DCT}(z), \text{thld}(t))$
    - 截止频率 $\text{thld}(t)$ 随 $t$ 递减：大 $t$ 时保护范围广（严格），小 $t$ 时保护范围窄（宽松）
    - 比空间域正则化更优雅：直接空间域正则要么保持不足，要么限制过严

3. **语义感知梯度精炼（Semantic-Aware Gradient Refinement）**

    - 动机：不同像素需要不同程度的风格化（前景主体需要更强、背景需要更少）
    - 利用扩散模型的语义先验计算 relevance map：
    $\mathcal{R}(z_t^{src}, t) = \text{Norm}(|\epsilon_{real}(z_t^{src}; y_{edit}, t) - \epsilon_{real}(z_t^{src}; y_\emptyset, t)|)$
    - $y_{edit}$ = "Turn it into {target style}"，与空条件的差异高亮语义重要区域
    - 作为逐元素权重调制梯度：语义重要区域强调风格、非关键区域抑制变化
    - $\mathcal{R}$ 是自适应、时间步相关的，随扩散过程自然调整

### 损失函数 / 训练策略

总体目标：$L_{SMS} = L_{style} + \lambda \cdot L_{freq}$

其中 $L_{style}$ 包含语义权重修正：
$$L_{style} = \mathbb{E}_{t,\epsilon}[\|\mathcal{R}(z_t^{src}, t) \odot w_t (\epsilon_{style} - \epsilon_{fake}^\phi)\|_2^2]$$

**自适应窄化采样策略**：
- 随迭代进行，逐步缩小时间步采样范围上界
- $t \sim \mathcal{U}(t_{min}, t_{upper})$，$t_{upper} = (1 - \frac{iter_{cur}}{iter_{total}}) \cdot t_{max}$
- 避免均匀采样的不一致正则强度和线性退火的内容偏离

**前馈扩展**：
- 用轻量网络 $G_\theta$（~43MB）替代逐图优化
- 重建预热 + 每批次变时间步采样 + SMS 损失训练

## 实验关键数据

### 主实验（单图风格化 - Ghibli 风格）

| 方法 | LPIPS ↓ | FID ↓ | ArtFID ↓ | PickScore ↑ |
|------|---------|-------|----------|-------------|
| FreeStyle | 0.690 | 12.361 | 22.582 | 0.683 |
| StyleID | 0.608 | 19.007 | 32.169 | 0.405 |
| InstantStyle+ | 0.538 | 14.949 | 24.532 | 1.019 |
| Style-LoRA | 0.438 | 12.267 | 19.077 | 2.067 |
| DDS | 0.513 | 15.233 | 24.554 | 0.537 |
| **SMS** | **0.326** | **13.089** | **18.686** | 1.487 |

用户偏好（300 次比较）：

| 指标 | FreeStyle | StyleID | InstantStyle+ | Style-LoRA | DDS | **SMS** |
|------|-----------|---------|---------------|-----------|-----|---------|
| 风格 | 0.060 | 0.147 | 0.083 | 0.100 | 0.033 | **0.577** |
| 内容 | 0.003 | 0.127 | 0.136 | 0.090 | 0.017 | **0.627** |
| 整体 | 0.013 | 0.110 | 0.127 | 0.077 | 0.020 | **0.653** |

### 消融实验

| 配置 | LPIPS ↓ | ArtFID ↓ | 说明 |
|------|---------|----------|------|
| 仅风格匹配（无 $L_f$, 无 $\mathcal{R}$） | 0.703 | 26.403 | 引入噪声和虚假细节 |
| +频谱正则（无 $\mathcal{R}$） | 0.505 | 24.132 | 结构增强但有高频伪影 |
| +语义精炼（无 $L_f$） | 0.536 | 27.514 | 选择性风格化但缺直接约束 |
| **完整 SMS** | **0.326** | **18.686** | 最优平衡 |
| 随机 $t$ 采样 | 0.389 | 32.936 | 图像模糊 |
| 线性退火 $t$ | 0.408 | 23.524 | 局部身份偏移 |

前馈风格化（~43MB 模型，实时推理）：

| 方法 | LPIPS ↓ | ArtFID ↓ |
|------|---------|----------|
| Scenimefy | 0.422 | 18.561 |
| DDS | 0.321 | 18.338 |
| PDS | 0.427 | 24.590 |
| **SMS** | **0.268** | **17.079** |

### 关键发现

- SMS 在内容保持（LPIPS）上大幅领先，同时风格对齐（FID）保持竞争力
- ArtFID = (LPIPS+1)·(FID+1) 综合指标上全面最优，证明平衡能力
- 频谱正则和语义精炼各自有效，组合使用效果最佳
- 自适应窄化采样优于随机采样和线性退火
- 前馈版本同样有效，验证了 SMS 从像素空间到参数空间的可扩展性

## 亮点与洞察

- **问题重新定义**：将风格化问题提升为分布匹配问题，而非简单的图像转换
- **频域正则的精妙设计**：利用 DCT 和时间步感知的低通滤波，比空间域正则更灵活
- **语义先验的创新用法**：用无条件与有条件扩散预测的差异作为语义重要性图
- **像素→参数空间的统一**：同一框架支持单图优化和批量训练前馈模型

## 局限与展望

- 依赖现成的 Style-LoRA，LoRA 的质量直接影响风格化效果
- 单图优化仍需 500 步迭代（~数分钟），速度有待提升
- 基于 SD 1.5，未探索 SDXL 等更强模型
- 未涉及视频风格化和 3D 风格化（作者提到 NeRF/3DGS 是未来方向）

## 相关工作与启发

- 与 SDS/VSD/DMD 共享高层动机，但专注于风格蒸馏而非 3D 生成
- 与 DDS 的对比：DDS 没有显式身份保持，且减少的噪声去噪方向不基于当前优化图像
- 频域正则思路可推广到其他需要内容-风格解耦的任务（如视频编辑、3D 纹理生成）
- Style-LoRA 作为风格表示的有效性得到验证，可启发更多基于 LoRA 的应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 风格分布匹配 + 频谱正则 + 语义梯度精炼的组合很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 定量+定性+用户研究+消融+前馈扩展全面覆盖
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，可视化丰富
- 价值: ⭐⭐⭐⭐ 实用性强，开源代码，适用于多种风格

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[CVPR 2026\] Mixture of Style Experts for Diverse Image Stylization](../../CVPR2026/image_generation/mixture_of_style_experts_for_diverse_image_stylization.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)

</div>

<!-- RELATED:END -->
