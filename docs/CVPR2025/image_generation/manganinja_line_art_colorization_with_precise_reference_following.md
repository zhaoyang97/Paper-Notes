---
title: >-
  [论文解读] MangaNinja: Line Art Colorization with Precise Reference Following
description: >-
  [CVPR 2025][图像生成][线稿上色] MangaNinja 是一个基于扩散模型的参考图引导线稿上色方法，通过渐进式 Patch Shuffling 策略训练模型学会局部语义匹配能力，并引入 PointNet 驱动的点控制机制实现精细颜色对应，在大姿态差异、多参考图、跨角色上色等挑战场景中显著超越现有方法。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "线稿上色"
  - "扩散模型"
  - "参考图引导"
  - "点控制"
  - "Patch Shuffling"
---

# MangaNinja: Line Art Colorization with Precise Reference Following

**会议**: CVPR 2025  
**arXiv**: [2501.08332](https://arxiv.org/abs/2501.08332)  
**代码**: 暂未公开  
**领域**: 图像生成 / 动漫上色  
**关键词**: 线稿上色, 扩散模型, 参考图引导, 点控制, Patch Shuffling

## 一句话总结
MangaNinja 是一个基于扩散模型的参考图引导线稿上色方法，通过渐进式 Patch Shuffling 策略训练模型学会局部语义匹配能力，并引入 PointNet 驱动的点控制机制实现精细颜色对应，在大姿态差异、多参考图、跨角色上色等挑战场景中显著超越现有方法。

## 研究背景与动机

1. **领域现状**：线稿上色（line art colorization）是漫画和动画制作中的核心环节。参考图引导的上色方法（给定一张彩色参考图和一张线稿，将颜色迁移到线稿上）比文本/笔触引导方法能更好保持角色一致性。
2. **现有痛点**：(1) 现有方法（如 BasicPBC、AnimeDiffusion）在参考图和线稿差异较大（不同姿势、不同视角）时容易产生语义错配和颜色混淆; (2) 通常要求参考图和线稿高度相似，实际应用中不现实; (3) 缺乏精细控制能力——无法指定"线稿的这个部位应该对应参考图的那个颜色"
3. **核心矛盾**：模型倾向于学习全局风格迁移而非局部语义匹配——当参考图整体结构与线稿相似时，简单的全局匹配就够了；但实际场景中姿势/视角/细节经常大不相同。
4. **本文目标**：(1) 让模型学到精准的局部语义匹配能力，即使参考图和线稿差异很大也能正确对应; (2) 提供交互式精细控制，用户可通过点对应来指导困难区域的上色
5. **切入角度**：利用动漫视频帧对作为训练数据（天然提供同一角色不同姿态的对应），通过渐进式 Patch Shuffling 打破参考图的整体结构，强迫模型学习局部匹配而非全局copy。
6. **核心 idea**：将参考图打散为 patches 后随机排列作为训练输入，迫使模型学会 patch 级局部匹配能力，辅以点控制实现精细上色引导。

## 方法详解

### 整体框架
双分支 U-Net 结构：Reference U-Net 编码参考图特征，Denoising U-Net 以线稿+噪声为输入进行去噪生成。两个 U-Net 通过 self-attention 层的 K/V 拼接实现特征注入（$\text{Attn} = \text{softmax}(\frac{Q_{tar}[K_{tar}, K_{ref}]^T}{\sqrt{d}})[V_{tar}, V_{ref}]$）。训练数据来自动漫视频——随机采样两帧，一帧做参考，另一帧提取线稿做目标。

### 关键设计

1. **渐进式 Patch Shuffling（核心创新）**:
    - 功能：强制模型学习局部语义匹配能力而非全局风格迁移
    - 核心思路：训练时将参考图切割为网格 patch（从 2×2 到 32×32），随机打乱位置后送入 Reference U-Net。由于参考图的整体结构被破坏，模型无法依赖全局空间布局，必须学习根据局部语义内容（颜色、纹理、形状）来匹配线稿的对应部位。训练过程中渐进增加 patch 数量（从粗到细），实现从全局到局部的匹配能力递进
    - 设计动机：不做 patch shuffling 时模型倾向于简单的全局 style transfer（因为训练帧对通常结构相似），DINO 相似度从 64.13 提升到 67.78（+3.65），且只有学会局部匹配后点控制才真正生效

2. **PointNet 驱动的点控制机制**:
    - 功能：让用户通过标注匹配点对实现精细颜色控制
    - 核心思路：用户在参考图和线稿上分别标注对应点，每对匹配点在各自的单通道 point map 上赋予相同唯一整数值。通过 PointNet（多层 Conv + SiLU）编码为多尺度 embedding $E_{tar}$ 和 $E_{ref}$，注入到 cross-attention 的 Q 和 K 中：$Q'_{tar} = Q_{tar} + E_{tar}$, $K'_{ref} = K_{ref} + E_{ref}$。训练时随机选 0-24 对匹配点（使用 LightGlue 自动提取），推理时用户可选择不用点或手动标注
    - 设计动机：纯自动匹配在困难场景（如鼻子阴影、小面积配件花纹、多角色构图）中仍有歧义，点控制提供了用户介入的精确通道

3. **多 Classifier-Free Guidance（Multi-CFG）**:
    - 功能：分别控制参考图和点控制的引导强度
    - 核心思路：定义三个条件组合的噪声预测：无条件 $\epsilon(\emptyset, \emptyset)$、仅参考图 $\epsilon(c_{ref}, \emptyset)$、参考图+点 $\epsilon(c_{ref}, c_{points})$，通过两个权重 $\omega_{ref}$ 和 $\omega_{points}$ 独立控制两种引导的强度。增大 $\omega_{ref}$ 增强自动匹配能力，增大 $\omega_{points}$ 增强点控制精度
    - 设计动机：不同任务场景需要不同的平衡——简单场景用高 $\omega_{ref}$ 自动匹配即可，复杂场景（跨角色上色）需要高 $\omega_{points}$ 精确控制

### 损失函数 / 训练策略
两阶段训练：第一阶段（180k steps）联合训练 Reference U-Net + Denoising U-Net + PointNet，随机 drop 参考图和点信号作为 unconditional generation 训练（condition dropping）; 第二阶段（20k steps）仅训练 PointNet 增强点控制。使用 SD 1.5 预训练权重初始化两个 U-Net。训练数据：sakuga-42m 数据集过滤后保留 30 万视频 clip。8×A100-80G 一天完成训练。

## 实验关键数据

### 主实验

| 方法 | DINO ↑ | CLIP ↑ | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ |
|------|--------|--------|--------|----------|---------|
| BasicPBC | 42.64 | 79.64 | 17.58 | 0.894 | 0.33 |
| IP-Adapter | 55.42 | 82.39 | 16.19 | 0.845 | 0.30 |
| AnyDoor* (w/ mask) | 63.79 | 83.91 | 16.24 | 0.874 | 0.27 |
| MangaNinja (无点) | 68.23 | 88.34 | 20.37 | 0.962 | 0.22 |
| MangaNinja (有点) | 69.91 | 90.02 | 21.34 | 0.972 | 0.21 |

MangaNinja 在所有指标上大幅领先，尤其 DINO 相似度（+6.12 vs AnyDoor*）和 PSNR（+5.10 vs AnyDoor*）。

### 消融实验

| 配置 | DINO ↑ | CLIP ↑ | PSNR ↑ | MSE ↓ |
|------|--------|--------|--------|------|
| Base model | 64.13 | 85.05 | 18.12 | 0.0151 |
| + condition dropping | 64.92 | 85.44 | 19.02 | 0.0125 |
| + progressive patch shuffle | 67.78 | 87.42 | 20.18 | 0.0091 |
| + multi CFG | 64.63 | 86.02 | 18.74 | 0.0133 |
| + two-stage training | 64.32 | 86.34 | 19.36 | 0.0113 |
| Full model | 69.91 | 90.02 | 21.34 | 0.0072 |

### 关键发现
- **Progressive Patch Shuffling 贡献最大**：DINO +3.65、PSNR +2.06、MSE 减半（0.0151→0.0091），是最关键的设计
- **点控制只有在学会局部匹配后才有效**：base model 加点控制提升很小，但 patch shuffle 后加点控制效果显著
- **Condition dropping 对自动匹配有独立贡献**：即使不用点引导也能提升 DINO +0.79
- **多参考图融合**和**跨角色上色**是 MangaNinja 独有能力，现有方法均无法实现

## 亮点与洞察
- **Patch Shuffling 的"逼出舒适区"训练哲学**：通过故意破坏输入的全局结构来迫使模型学习更鲁棒的局部匹配能力，这个思路可迁移到任何需要学习语义对应的任务（如 image editing、virtual try-on、pose transfer）
- **点控制的层级设计**：PointNet embedding 加到 attention 的 Q/K 上而非直接作为额外 token，既不破坏原有 attention 的信息流，又能精确引导对应关系。这种 attention 调制方式比 cross-attention 更优雅
- **视频帧对作为自监督对应数据**：动漫视频天然提供同角色不同帧的对应关系，免去人工标注。这种思路可迁移到时尚（outfit 视频帧对）、自动驾驶（连续帧对应）等领域

## 局限与展望
- 基于 SD 1.5 的分辨率受限（512×512），无法处理高分辨率漫画稿
- 训练数据完全来自动漫视频，对真实照片的线稿上色可能泛化性不足
- 点控制虽然精确但仍需手动标注，后续可结合自动匹配（如 DINOv2 特征匹配）实现自动精细对应
- 论文未评估复杂背景场景的上色效果（测试集均为前景角色分割后的结果）
- 多参考图融合时如何自动解决冲突（如两张参考图同一部位颜色不同）未深入讨论

## 相关工作与启发
- **vs BasicPBC**: BasicPBC 是非生成式方法，从参考图邻近区域采样颜色，在参考与线稿差异大时效果差; MangaNinja 的生成式匹配能力远超之
- **vs IP-Adapter/AnyDoor**: 都是基于扩散模型的图像条件生成，但缺乏精细匹配能力，只能做粗粒度风格迁移，颜色混淆严重
- **vs AnimeDiffusion**: 同为动漫上色方法但仅支持简单场景，MangaNinja 的 patch shuffle + 点控制使其在复杂场景中远超之
- **与 visual correspondence 的交叉**：Patch shuffling 实质上在训练中强化了扩散模型的隐式视觉对应能力，与近年 diffusion features for correspondence 的研究方向一致

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Patch shuffling 训练策略和点控制机制设计巧妙，组合起来解决了长期痛点
- 实验充分度: ⭐⭐⭐⭐ 构建了专门的评测基准，消融实验充分，多种挑战场景展示完整
- 写作质量: ⭐⭐⭐⭐ 图示丰富直观，方法讲解清晰
- 价值: ⭐⭐⭐⭐⭐ 对动漫/漫画产业有直接实用价值，多参考图和跨角色上色功能开拓了新应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[CVPR 2026\] Towards High-resolution and Disentangled Reference-based Sketch Colorization](../../CVPR2026/image_generation/towards_high-resolution_and_disentangled_reference-based_sketch_colorization.md)
- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)

</div>

<!-- RELATED:END -->
