---
title: >-
  [论文解读] Zero-Shot Head Swapping in Real-World Scenarios
description: >-
  [CVPR 2025][头部替换] 提出HID（Head Injection Diffusion），一种零样本头部替换方法，通过IOMask自动生成上下文感知的编辑掩码实现无缝头身融合，并引入hair injection模块精确迁移发型细节，在包含上半身和多角度面部的真实场景中实现SOTA性能。 与仅替换面部身份(ID)的f…
tags:
  - "CVPR 2025"
  - "头部替换"
  - "零样本"
  - "扩散模型"
  - "自动掩码生成"
  - "发型注入"
---

# Zero-Shot Head Swapping in Real-World Scenarios

**会议**: CVPR 2025  
**arXiv**: [2503.00861](https://arxiv.org/abs/2503.00861)  
**代码**: 无  
**领域**: 其他  
**关键词**: 头部替换, 零样本, 扩散模型, 自动掩码生成, 发型注入

## 一句话总结

提出HID（Head Injection Diffusion），一种零样本头部替换方法，通过IOMask自动生成上下文感知的编辑掩码实现无缝头身融合，并引入hair injection模块精确迁移发型细节，在包含上半身和多角度面部的真实场景中实现SOTA性能。

## 研究背景与动机

与仅替换面部身份(ID)的face swapping不同，head swapping需要将整个头部（包括面部ID、脸型和发型）从头部图像无缝融合到身体图像上，复杂度显著更高。

现有head swapping方法面临三个关键限制：(1) **依赖面部裁剪数据** — 大多方法（FaceX、REFace）仅在面部居中裁剪的图像上操作，当头部替换后需要粘回完整身体时，容易出现不协调（如原图长发残留、颜色不一致）；(2) **掩码方法不够智能** — 现有掩码针对裁剪数据优化，无法处理复杂场景（如长发超出裁剪区域）；(3) **视角鲁棒性差** — 大多方法仅处理正面视角，缺少对侧脸等多角度的支持。

本文的核心贡献是设计了一种零样本方法，直接在包含上半身的完整图像上进行head swapping，自动生成适应上下文的掩码。

## 方法详解

### 整体框架

HID基于PhotoMaker V2构建，分两阶段：(1) 左阶段——通过ID Fusion模型和Hair Fusion模型分别提取身份和发型嵌入，替换文本嵌入中对应部分；(2) 右阶段——对身体图像进行DDIM反转获取潜在表示，利用IOMask确定需编辑的头部区域，在ControlNet（OpenPose）约束下通过条件去噪生成最终结果。

### 关键设计1：IOMask（反转正交掩码）

- **功能**: 自动生成上下文感知的头部编辑掩码，无需人工标注
- **核心思路**: 对身体图像做DDIM反转到特定时间步$t$，分别计算身体条件噪声$\epsilon_\theta(C_b)$和头部条件噪声$\epsilon_\theta(C_h, \varnothing)$。提取后者相对于前者的正交分量：$\epsilon_\theta^{orth} = \epsilon_\theta(C_h,\varnothing) - \frac{\langle\epsilon_\theta(C_b), \epsilon_\theta(C_h,\varnothing)\rangle}{\|\epsilon_\theta(C_b)\|^2}\epsilon_\theta(C_b)$。经高斯滤波和阈值化得到二值掩码
- **设计动机**: 直接差分$\epsilon_\theta(C_h)-\epsilon_\theta(C_b)$会产生随机噪声。正交分量计算确保：方向一致的区域（身体部分）值小，方向不同的区域（头部需替换区域）值大，是更精确的编辑区域指示器

### 关键设计2：Hair Injection模块

- **功能**: 精确迁移头部图像的发型特征到生成结果
- **核心思路**: 受PhotoMaker V2启发，训练Hair Fusion模型——用Q-former和MLP将CLIP图像编码的发型特征与预训练Hair Encoder提取的发型嵌入融合。融合后的嵌入替换文本嵌入中"hairstyle"对应位置。训练时用SCHP生成的人体解析掩码只重建人物区域
- **设计动机**: PhotoMaker V2专注于面部ID保持，无法保证发型准确迁移。通过专门的发型注入模块，发型信息得到独立且精确的表达

### 关键设计3：头部注入扩散过程

- **功能**: 在去噪过程中无缝融合新头部与原始身体
- **核心思路**: 从DDIM反转得到的潜在$\hat{z}_T$开始去噪，每步用IOMask混合：$z_{t-1} = \tilde{z}_{t-1} \odot \mathcal{M} + \hat{z}_{t-1} \odot (1-\mathcal{M})$。掩码外区域始终保持原始身体的反转潜在，掩码内由带头部条件的去噪生成
- **设计动机**: 从反转潜在而非纯噪声开始确保肤色、衣物等细节一致性；逐步混合而非一次性粘贴确保边界自然过渡

### 损失函数

训练阶段使用标准扩散模型损失，Hair Fusion模块训练时增加基于SCHP掩码的masked重建损失，确保只重建人物区域。

## 实验关键数据

### 主实验：定量对比（SHHQ数据集）

| 方法 | FID↓ | Head LPIPS↓ | Head CLIP-I↑ | Hair LPIPS↓ | Hair CLIP-I↑ |
|------|------|-------------|-------------|-------------|-------------|
| REFace | 40.72 | 0.0770 | 0.7867 | 0.0658 | 0.8563 |
| **HID (Ours)** | **37.19** | **0.0721** | **0.8512** | **0.0596** | **0.8707** |

### 消融实验

| 配置 | 效果 |
|------|------|
| Full HID | 最优头部替换，发型精确迁移，身体完整保留 |
| w/o Hair Injection | 发型细节丢失（如长发变短发） |
| w/o IOMask | 身体图像信息几乎完全丢失 |

### IOMask变体对比

| IO Map变体 | 效果 |
|------------|------|
| Naive IO map | 大量随机噪声覆盖无关区域 |
| w/o orthogonal | 较好但区域不够精确 |
| **Full IO map** | 精确聚焦于需替换的头部区域 |

### 关键发现

- IOMask的正交分量计算相比简单差分大幅减少了噪声伪影
- 从反转潜在开始生成是保持身体一致性的关键
- 在包含上半身的完整图像上操作避免了裁剪-粘回的不协调问题
- HID在所有指标上均超越唯一可比的零样本方法REFace

## 亮点与洞察

1. **IOMask的正交分量思路**: 利用噪声预测的几何关系自动推断编辑区域，比注意力图方法更高效
2. **问题重新定义**: 将head swapping放在包含上半身的完整图像上处理，比面部裁剪方式更贴合实际需求
3. **模块化设计**: ID注入和发型注入独立处理，各自专注于自身任务

## 局限与展望

- 仅与REFace做了定量对比（唯一开源的零样本方法），对比基线单薄
- IOMask的阈值$\tau$需要调节，可能对不同场景敏感
- 未探索极端场景（如佩戴帽子、头巾等遮挡情况）

## 相关工作与启发

- IOMask的正交分量掩码思路可推广到其他扩散模型编辑任务（如衣物替换、饰品编辑）
- Hair Injection模块的设计可用于其他需要精确发型控制的生成任务

## 评分

⭐⭐⭐⭐ — 问题动机清晰，IOMask设计巧妙利用了扩散模型噪声预测的几何性质。将head swapping推进到真实场景（非裁剪图像）是有实际价值的贡献。但对比实验偏少。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](../../ACL2025/others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[CVPR 2026\] NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering](../../CVPR2026/others/naf_zero-shot_feature_upsampling_via_neighborhood_attention_filtering.md)
- [\[ECCV 2024\] AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes](../../ECCV2024/others/addme_zero-shot_group-photo_synthesis_by_inserting_people_into_scenes.md)
- [\[CVPR 2026\] Zero-shot Detection of AI-Generated Image via RAW-RGB Alignment](../../CVPR2026/others/zero-shot_detection_of_ai-generated_image_via_raw-rgb_alignment.md)
- [\[ICML 2025\] Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Settings](../../ICML2025/others/suitability_filter_a_statistical_framework_for_classifier_evaluation_in_real-wor.md)

</div>

<!-- RELATED:END -->
