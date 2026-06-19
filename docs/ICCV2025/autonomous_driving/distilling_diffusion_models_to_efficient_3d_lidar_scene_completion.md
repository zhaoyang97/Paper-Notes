---
title: >-
  [论文解读] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion
description: >-
  [ICCV 2025 (Oral)][自动驾驶][LiDAR场景补全] 提出 ScoreLiDAR，一种针对 3D LiDAR 场景补全的扩散模型蒸馏方法，通过场景级和点级结构损失引导蒸馏，将补全时间从 30.55 秒压缩到 5.37 秒（>5x 加速），同时在 SemanticKITTI 上超越所有 SOTA 方法。
tags:
  - "ICCV 2025 (Oral)"
  - "自动驾驶"
  - "LiDAR场景补全"
  - "扩散模型蒸馏"
  - "结构损失"
  - "点云重建"
  - "自动驾驶感知"
---

# Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion

**会议**: ICCV 2025 (Oral)  
**arXiv**: [2412.03515](https://arxiv.org/abs/2412.03515)  
**代码**: [https://github.com/happyw1nd/ScoreLiDAR](https://github.com/happyw1nd/ScoreLiDAR)  
**领域**: 自动驾驶 / 3D视觉  
**关键词**: LiDAR场景补全, 扩散模型蒸馏, 结构损失, 点云重建, 自动驾驶感知

## 一句话总结

提出 ScoreLiDAR，一种针对 3D LiDAR 场景补全的扩散模型蒸馏方法，通过场景级和点级结构损失引导蒸馏，将补全时间从 30.55 秒压缩到 5.37 秒（>5x 加速），同时在 SemanticKITTI 上超越所有 SOTA 方法。

## 研究背景与动机

**领域现状**：3D LiDAR 场景补全（Scene Completion）的目标是从稀疏的 LiDAR 扫描恢复出完整的 3D 场景。扩散模型因其训练稳定性和高补全质量，已被成功应用于该任务（如 LiDiff）。具体来说，扩散模型在 3D 点云的潜在空间中进行去噪，逐步从随机噪声中恢复出完整场景。

**现有痛点**：扩散模型需要大量采样步骤（通常 50-1000 步）才能生成高质量结果，这导致每帧补全耗时超过 30 秒。对于自动驾驶等实时应用场景，这种速度完全不可接受——车辆需要在毫秒级时间内感知周围环境。

**核心矛盾**：扩散模型的质量来源于其多步迭代去噪过程，但计算成本也正比于步数。简单减少步数会严重损害补全质量，尤其是 3D 场景的几何结构（如建筑物轮廓、道路平面）在少步采样时容易变形。

**本文目标**：设计一种蒸馏方法，让学生模型在极少的采样步数（如 2-8 步）内达到甚至超过教师模型（50+ 步）的补全质量，同时保持 3D 场景的几何结构完整性。

**切入角度**：作者观察到 3D LiDAR 场景具有强烈的结构先验——建筑物成平面、道路近似水平、树木有特定形态。如果在蒸馏过程中显式约束学生模型保持这些结构特征，就能在少步采样下维持几何质量。

**核心 idea**：将分数蒸馏（Score Distillation）与专门设计的结构损失（Structural Loss）结合——结构损失同时在全局场景层面和局部关键点层面约束学生模型的输出几何，使蒸馏后的模型在少步采样下仍能准确重建 3D 结构。

## 方法详解

### 整体框架

ScoreLiDAR 采用"教师-学生"蒸馏范式。教师模型是训练好的 LiDiff（一个基于 3D 点云的扩散模型），使用 50 步 DDIM 采样。学生模型共享教师的网络架构，但目标是在 2-8 步内完成补全。蒸馏分为两阶段：首先通过分数匹配蒸馏（Score Matching Distillation）让学生学习教师的去噪分布；然后用结构损失微调让学生输出的 3D 几何更加精确。输入是稀疏 LiDAR 点云，输出是补全后的稠密 3D 场景。

### 关键设计

1. **分数匹配蒸馏（Score Matching Distillation）**:

    - 功能：将教师模型的多步去噪知识压缩到学生模型的少步推理中
    - 核心思路：训练学生模型使其在每个时间步的预测分数（score function）匹配教师模型。具体来说，给定噪声点云 $x_t$，教师模型预测 $\epsilon_\text{teacher}(x_t, t)$，学生模型的损失为 $\mathcal{L}_\text{score} = \|\epsilon_\text{student}(x_t, t) - \epsilon_\text{teacher}(x_t, t)\|^2$。通过渐进蒸馏（progressive distillation），将步数逐步减半：50→25→12→8→4→2
    - 设计动机：直接在输出空间蒸馏（如让学生 2 步的输出匹配教师 50 步的输出）会导致梯度不稳定。分数匹配在每个时间步都提供监督信号，训练更加稳定

2. **场景级结构损失（Scene-wise Structural Loss）**:

    - 功能：约束学生模型补全结果的整体几何结构
    - 核心思路：对补全结果进行体素化（voxelization），然后计算学生输出和 ground truth 在体素占用网格（occupancy grid）上的差异。使用 Chamfer Distance 的体素版本：$\mathcal{L}_\text{scene} = \text{CD}_\text{voxel}(\text{Vox}(\hat{x}_0), \text{Vox}(x_0^*))$。体素化使得损失对点的微小位置偏移更加鲁棒，更关注整体结构（如"这里应该有一面墙"）而非精确点位
    - 设计动机：传统的逐点 Chamfer Distance 对异常点敏感且不能捕捉全局几何结构。体素化后的结构损失更好地约束了场景的拓扑和宏观形状

3. **点级关键点损失（Point-wise Landmark Loss）**:

    - 功能：约束学生模型在关键几何位置的精确性
    - 核心思路：从 ground truth 场景中提取关键点（landmarks），这些点位于几何显著的位置（如角点、边缘交汇处、平面交线等）。通过 FPS（Farthest Point Sampling）在曲率较高的区域采样关键点，然后计算学生输出中对应位置的偏差：$\mathcal{L}_\text{point} = \sum_{p \in \text{landmarks}} \min_{q \in \hat{x}_0} \|p - q\|^2$，同时约束关键点之间的相对配置不变
    - 设计动机：场景级损失保证了整体形状，但在精细几何（如窗户边角、柱子边缘）上可能不够精确。关键点损失是对场景级损失的互补，确保重要几何细节被保留

### 损失函数 / 训练策略

总损失为三项之和：$\mathcal{L} = \mathcal{L}_\text{score} + \alpha \mathcal{L}_\text{scene} + \beta \mathcal{L}_\text{point}$。训练分两阶段：第一阶段只用 $\mathcal{L}_\text{score}$ 进行渐进蒸馏（50→2 步）；第二阶段加入结构损失进行微调。使用 SemanticKITTI 训练，Adam 优化器，学习率 1e-4。蒸馏后还用一个轻量级 refine_net 做最终精修。

## 实验关键数据

### 主实验

| 方法 | 步数 | CD↓ (×10⁻³) | IoU↑ | 时间 (s/frame) |
|------|------|-------------|------|---------------|
| ScoreLiDAR (Ours) | 8 | **3.12** | **0.84** | **5.37** |
| LiDiff (Teacher) | 50 | 3.45 | 0.82 | 30.55 |
| S2CFormer | 1 | 4.23 | 0.78 | 0.12 |
| SCPNet | 1 | 4.56 | 0.76 | 0.09 |
| JS3C-Net | 1 | 5.12 | 0.73 | 0.15 |
| LMSCNet | 1 | 5.89 | 0.70 | 0.08 |

### 消融实验

| 配置 | CD↓ (×10⁻³) | IoU↑ | 说明 |
|------|-------------|------|------|
| Full ScoreLiDAR (8步) | **3.12** | **0.84** | 完整模型 |
| w/o 场景级结构损失 | 3.58 | 0.81 | 整体结构变差 |
| w/o 点级关键点损失 | 3.41 | 0.82 | 精细几何退化 |
| w/o 所有结构损失 | 3.89 | 0.79 | 仅分数蒸馏，几何质量明显下降 |
| 2 步采样 | 3.67 | 0.80 | 步数过少，质量下降但仍优于传统方法 |
| 4 步采样 | 3.28 | 0.83 | 质量-速度的良好折中 |

### 关键发现

- ScoreLiDAR 以 8 步推理不仅比教师模型（50 步）快 5.7 倍，补全质量还更好（CD 降低 9.6%），这是因为结构损失提供了额外的几何先验
- 结构损失的贡献：去掉场景级损失 CD 增加 14.7%，去掉点级损失 CD 增加 9.3%，去掉全部结构损失 CD 增加 24.7%——两个层级的结构约束互补且都很重要
- 即使压缩到 2 步，ScoreLiDAR 仍优于所有非扩散方法（S2CFormer 等），展示了扩散蒸馏的强大能力
- ICCV 2025 Oral，66 stars on GitHub，说明领域认可度高

## 亮点与洞察

- **结构损失的双层设计非常巧妙**：场景级（体素化 CD）关注全局拓扑，点级（关键点损失）关注局部细节，两者互补。这种"宏观+微观"的损失设计思路可以迁移到任何涉及 3D 结构保持的蒸馏/压缩任务
- **"蒸馏后超越教师"的有趣现象**：结构损失作为额外的归纳偏置，使学生模型的几何质量甚至优于教师。这提示蒸馏不只是压缩，还可以是"加入新知识"的过程
- **面向实际应用的设计**：5.37 秒虽然还不够实时，但将这种蒸馏+结构损失的思路应用到更快的骨干网络上，有望达到实时补全

## 局限与展望

- 5.37 秒/帧对于自动驾驶实时应用仍然偏慢（需要 <100ms），需要进一步加速
- 蒸馏依赖于教师模型的质量上限，如果教师模型本身存在系统性偏差，学生也会继承
- 目前只在 SemanticKITTI 上评估，对其他数据集（如 nuScenes、Waymo）的泛化性需验证
- 结构损失中的关键点提取依赖几何分析，可能在极度稀疏的场景中不稳定
- 未来方向：将蒸馏技术与更快的 3D 骨干（如稀疏卷积）结合；扩展到动态场景补全

## 相关工作与启发

- **vs LiDiff（教师模型）**: LiDiff 是首个将扩散模型用于 LiDAR 场景补全的工作，取得了高质量结果但速度慢。ScoreLiDAR 在保持质量的同时实现了 5x+ 加速
- **vs S2CFormer/SCPNet（传统方法）**: 这些方法虽然实时但质量远不如扩散方法。ScoreLiDAR 证明了通过蒸馏可以在速度和质量之间找到更好的平衡
- **vs 图像领域蒸馏（如 LCM, SDXL-Turbo）**: 2D 图像蒸馏不需要考虑 3D 几何结构，而 ScoreLiDAR 的结构损失是专门为 3D 场景设计的核心创新

## 评分

- 新颖性: ⭐⭐⭐⭐ 场景级+点级结构损失是新颖的设计，首次将蒸馏引入 3D LiDAR 补全
- 实验充分度: ⭐⭐⭐⭐⭐ 多种基线对比、详细消融、不同步数分析，ICCV Oral 级别
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 对自动驾驶感知有实际意义，代码公开且已获广泛关注

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures](../../AAAI2026/autonomous_driving/linext_revisiting_lidar_completion_with_efficient_non-diffusion_architectures.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](../../CVPR2025/autonomous_driving/distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[NeurIPS 2025\] Neurosymbolic Diffusion Models](../../NeurIPS2025/autonomous_driving/neurosymbolic_diffusion_models.md)
- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](../../NeurIPS2025/autonomous_driving/towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](decoupled_diffusion_sparks_adaptive_scene_generation.md)

</div>

<!-- RELATED:END -->
