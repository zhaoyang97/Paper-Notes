---
title: >-
  [论文解读] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization
description: >-
  [ICCV 2025][视频生成][视频定制生成] DualReal 首次提出身份与运动的自适应联合训练框架，通过 Dual-aware Adaptation 和 StageBlender Controller 实现两个维度的无损融合，在 CLIP-I 和 DINO-I 指标上平均提升 21.7% 和 31.8%。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "视频定制生成"
  - "身份-运动融合"
  - "联合训练"
  - "Transformer"
  - "自适应控制"
---

# DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization

**会议**: ICCV 2025  
**arXiv**: [2505.02192](https://arxiv.org/abs/2505.02192)  
**代码**: [https://wenc-k.github.io/dualreal-customization](https://wenc-k.github.io/dualreal-customization) (项目页)  
**领域**: 视频生成  
**关键词**: 视频定制生成, 身份-运动融合, 联合训练, 扩散 Transformer, 自适应控制

## 一句话总结

DualReal 首次提出身份与运动的自适应联合训练框架，通过 Dual-aware Adaptation 和 StageBlender Controller 实现两个维度的无损融合，在 CLIP-I 和 DINO-I 指标上平均提升 21.7% 和 31.8%。

## 研究背景与动机

视频定制生成（Video Customized Generation）是当前视频生成领域的重要前沿，目标是从用户提供的参考图像和运动视频中，生成同时保持身份一致性和运动一致性的视频。现有方法如 VideoBooth、AnimateDiff、DreamVideo 等多采用**孤立定制范式（isolated customized paradigm）**：分别训练身份适配器和运动适配器，在推理时直接融合参数。

然而，这种孤立范式完全忽视了身份与运动之间**内在的相互约束和协同依赖关系**。具体表现为：

**身份-运动冲突**：添加运动先验后身份保真度不可逆地下降。论文图 2 展示了固定身份训练步数、逐渐增加运动训练步数后，身份质量持续恶化

**最优步数不一致**：对不同身份，最小化身份退化的运动训练步数各不相同，不存在通用的最优步数

**扩散过程的自然规律被打破**：去噪过程本身会在不同步骤动态调整对空间（身份）和时间（运动）的关注重点，而孤立训练强行在所有步骤上均匀采样，导致优化方向冲突

核心矛盾：**两个维度的参数在不同训练数据下分布差异极大，无约束的联合更新会导致跨维度的知识破坏性干扰**。例如在运动训练阶段，用静态图片微调运动适配器会不可逆地损害其动态生成能力。

DualReal 的核心 idea：不做孤立训练再融合，而是在训练过程中就构建两个维度的相互依赖关系，通过**冻结的对方维度先验引导当前训练**，并利用**去噪阶段和网络深度的分工特性**自适应分配两个维度的权重。

## 方法详解

### 整体框架

DualReal 基于 DiT（Diffusion Transformer）架构，在每个 DiT block 中插入双适配器模块（DA-Block），包含身份适配器和运动适配器。整体流程：

1. 每个训练步动态切换训练模式 $Z \in \{0, 1\}$（身份或运动）
2. 对应数据注入噪声，结合文本嵌入送入 DiT
3. StageBlender Controller 根据当前去噪时间步和融合特征，计算各 DA-Block 中两个适配器的权重系数 $\omega_i$
4. DA-Block 中，训练维度的适配器参数更新，冻结维度的适配器保持固定提供先验引导
5. 梯度掩码阻止非训练维度的参数更新，防止知识泄漏

### 关键设计

1. **Dual-aware Adaptation（双感知适应）**：

    - 功能：在训练过程中动态切换身份/运动训练步骤，利用冻结维度的先验引导当前维度的学习
    - 核心思路：双适配器采用 bottleneck 结构，身份适配器直接映射输入特征，运动适配器额外通过条件线性层映射参考图像嵌入。输出加权融合：
        - $\hat{f}_{out}^i = \omega_i \cdot f_{mo}^i + (1 - \omega_i) \cdot f_{id}^i + f_{dit}^i$
    - 正则化策略：通过梯度掩码 $M$ 实现参数隔离。当训练运动时 $Z=1$，只有运动适配器参数 $\theta_m$ 被更新；训练身份时 $Z=0$，只有 $\theta_i$ 被更新：
        - $\theta^{(t+1)} = \theta^{(t)} - M \odot \nabla_\theta \mathcal{L}$
        - $M = Z \cdot M_m + (1-Z) \cdot M_i$
    - 设计动机：冻结维度的适配器在前向传播中提供内在正则化，约束维度过拟合；梯度掩码完全避免跨维度干扰。这种"参考学习"机制让两个维度可以在共享参数空间中协作

2. **StageBlender Controller（阶段混合控制器）**：

    - 功能：根据去噪时间步和 DiT 层深度，自适应生成各 DA-Block 中身份/运动适配器的权重系数
    - 核心思路：
        - 输入特征经 pooling 后，通过 Adaptive LayerNorm（注入时间步嵌入）调制：$f'' = \text{MLP}(\text{LN}(f')) * \alpha + \beta$
        - 门控融合：$f_g = f'' + \gamma \cdot f'$
        - 下传 MLP 生成分组权重：$\omega^{(1)}, ..., \omega^{(n)} = \text{softmax}(\Gamma \cdot \text{MLP}(f_g))$
        - 其中 $\Gamma: \mathbb{R}^L \to \mathbb{R}^n$ 将 DiT 层深度映射到 $n$ 个权重组
    - 视觉分析（图 7）：浅层 block 随去噪推进逐渐增大身份权重（保护身份细节），最深 block 则持续增大运动权重。这验证了 DiT 不同深度对身份/运动的分工
    - 设计动机：去噪早期以粗粒度布局为主（运动主导），后期精细化（身份主导），不同深度的 block 也有不同的功能偏好。固定权重无法捕捉这种动态变化

3. **权重分组机制（Weight Groups, n=7）**：

    - 功能：将 42 层 DiT block 分为 7 组，每组共享一组权重系数
    - 核心思路：既避免 n=1 时上下文不足，也避免 n=42 时细节被稀释
    - 消融结果：n=7 在 CLIP-I 和 DINO-I 上显著优于 n=1, n=2, n=42

### 损失函数 / 训练策略

- 使用标准的视频扩散重建损失 $\mathcal{L}$
- 身份/运动训练步骤的切换比例由预定义超参数控制
- 以 CogVideoX-5B 为基础 DiT 模型
- 50 个身份主体（每个 3-10 张图），21 个运动序列，每个 case 50 种 prompt

## 实验关键数据

### 主实验

| 方法 | CLIP-T↑ | CLIP-I↑ | DINO-I↑ | T.Flickering↑ | T.Cons↑ | Motion Smooth↑ | DD偏差 |
|------|---------|---------|---------|---------------|---------|----------------|--------|
| MotionBooth | 0.317 | 0.566 | 0.459 | 0.962 | 0.972 | 0.973 | -1.07 |
| LoRA | 0.323 | 0.425 | 0.286 | 0.956 | 0.976 | 0.973 | +13.32 |
| CogVideoX-5B | 0.336 | 0.521 | 0.424 | 0.947 | 0.973 | 0.965 | +14.49 |
| DreamVideo | 0.278 | 0.458 | 0.334 | 0.949 | 0.963 | 0.968 | -3.18 |
| **DualReal** | **0.323** | **0.629** | **0.551** | **0.965** | **0.983** | **0.978** | +2.94 |

DualReal 在 CLIP-I 和 DINO-I 上分别比次优方法提升 11.1% 和 20.0%，在三个运动质量指标（T.Cons、Motion Smoothness、T.Flickering）上均排名第一。

### 消融实验

| 配置 | CLIP-T | CLIP-I | DINO-I | DD偏差 | 说明 |
|------|--------|--------|--------|--------|------|
| w/o Dual-aware Adapt. | 0.334 | 0.616 | 0.647 | -5.53 | 分离训练+推理融合 |
| w/o StageBlender | 0.346 | 0.619 | 0.652 | -3.31 | 固定权重，无时间步感知 |
| w/o Weight Groups | 0.335 | 0.662 | 0.766 | -3.12 | 所有block统一调制 |
| **DualReal (full)** | **0.333** | **0.674** | **0.771** | **-2.70** | 完整方法 |

分组数消融结果：n=1 (0.662/0.766), n=2 (0.632/0.660), n=42 (0.631/0.706), **n=7 (0.674/0.771)**。

### 关键发现

- 去掉 Dual-aware Adaptation（即回退到孤立训练+推理融合）后，DD 偏差从 -2.70 恶化到 -5.53，运动几乎崩塌，证实联合训练的必要性
- StageBlender Controller 视觉分析验证了两个重要规律：(1) 随去噪进行，模型对身份的关注单调递增；(2) 最深层 block 专注于运动建模
- 固定权重（无 StageBlender）导致手部等细节过度适配运动模式

## 亮点与洞察

1. **首次系统性地揭示了孤立定制范式的根本缺陷**：不是技术不够好，而是范式本身忽略了维度间的内在依赖。图 2 的可视化分析非常有说服力
2. **梯度掩码正则化**思想简单但有效，彻底切断了跨维度的梯度干扰，比 soft regularization 更干净
3. **StageBlender 的视觉分析**（图 7）为理解 DiT 内部的时空分工提供了新视角：浅层管身份、深层管运动，这一发现对后续定制生成工作有重要参考价值
4. 分组机制 n=7 的最优性暗示了 DiT 内部存在约 7 个功能相近的 block 集群

## 局限与展望

- 评估数据集规模偏小（50 主体 × 21 运动 × 50 prompt），泛化性需更大规模验证
- Dynamic Degree 指标显示运动强度偏低（DD=14.96 vs 参考 12.02），可能对高动态场景的支持不足
- 身份/运动切换比例是固定超参数，未来可探索自适应切换策略
- 未与最近的 IP-Adapter、Animate-Anyone 等方法对比

## 相关工作与启发

- DreamVideo 的"分离训练+推理融合"是本文主要对标的孤立范式基线
- CogVideoX 的 expert transformer 架构为本文提供了强大的基础模型
- 梯度掩码策略借鉴了多任务学习中防止"灾难性遗忘"的经典方法
- 对于视频/图像生成中需要同时控制多个维度的任务（如文本+姿态、风格+结构），本文的联合训练框架具有通用参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](dreamrelation_relation-centric_video_customization.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)

</div>

<!-- RELATED:END -->
