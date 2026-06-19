---
title: >-
  [论文解读] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding
description: >-
  [CVPR 2025][人体理解][3D注视估计] 提出 GA3CE 方法，通过将主体 3D 姿态和场景物体位置编码到以主体为中心的自我中心空间中，并设计方向-距离分解的 D3 位置编码，在 Transformer 中学习 3D 注视方向与场景上下文的空间关系，在无约束设置下将 3D 注视角度误差降低 13%–37%。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "3D注视估计"
  - "自我中心变换"
  - "位置编码"
  - "场景理解"
  - "Transformer"
---

# GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding

**会议**: CVPR 2025  
**arXiv**: [2505.10671](https://arxiv.org/abs/2505.10671)  
**代码**: [https://woven-visionai.github.io/ga3ce-project](https://woven-visionai.github.io/ga3ce-project) (项目页)  
**领域**: 人体理解  
**关键词**: 3D注视估计, 自我中心变换, 位置编码, 场景理解, Transformer

## 一句话总结
提出 GA3CE 方法，通过将主体 3D 姿态和场景物体位置编码到以主体为中心的自我中心空间中，并设计方向-距离分解的 D3 位置编码，在 Transformer 中学习 3D 注视方向与场景上下文的空间关系，在无约束设置下将 3D 注视角度误差降低 13%–37%。

## 研究背景与动机

1. **领域现状**：3D 注视估计在监控、零售分析等场景中有重要应用。现有方法通常依赖 2D 外观特征或在后处理中有限地利用深度信息来约束注视方向。

2. **现有痛点**：当主体远离摄像机或背对时，无法获得清晰的眼部特征。现有方法要么忽略场景空间关系（如 GAFA），要么仅在不可学习的后处理步骤使用深度（如 GFIE），无法端到端学习主体-场景-注视的空间关系。

3. **核心矛盾**：从 2D 观测估计 3D 注视方向本质上是病态问题——相同的 3D 场景在不同相机位姿下产生截然不同的 2D 外观和 3D 注视方向，增加了学习难度。此前没有方法系统地解决这种相机位姿变化带来的复杂性。

4. **本文目标** 三个子问题：(i) 什么是有效的主体和场景表示？(ii) 如何消除不同相机位姿带来的变化？(iii) 如何建模主体、场景和注视之间的空间关系？

5. **切入角度**：人类视觉研究表明，主体视野中物体的方向和距离强烈影响注视行为。作者从这一启发出发，将 3D 上下文对齐到以主体为中心的自我中心空间，并将方向和距离分解编码。

6. **核心 idea**：用 3D 姿态和物体位置作为中间表示，通过自我中心变换消除相机位姿变差，再用方向-距离分解的位置编码在 Transformer 中学习空间关系来预测 3D 注视方向。

## 方法详解

### 整体框架
输入为 RGB 图像、深度图和相机内参，输出为主体的 3D 注视方向（单位向量）。方法分三步：(1) 使用预训练模型提取 3D 人体姿态关键点和场景物体 3D 位置作为中间表示；(2) 通过 GA3CE（注视感知 3D 上下文编码）将这些 3D 上下文变换到自我中心空间并编码；(3) Transformer 编码器-解码器学习物体间空间关系并输出残差注视方向，最终反变换得到注视方向。

### 关键设计

1. **3D 上下文表示（Subject & Object Representation）**:

    - 功能：将主体和场景从 2D 图像抽象为 3D 中间表示
    - 核心思路：主体表示使用预训练 3D 姿态估计器（MotionBERT）从裁剪的全身图像提取 $N_{pose}=15$ 个 3D 关键点 $P_{pose}$，加上头部外观估计器预测的视线方向 $\mathbf{v}$ 作为先验。场景表示使用 MobileSAM 的 segment-everything 方式获取所有物体实例掩码，通过深度图和相机内参反投影到 3D 空间取中位数得到物体 3D 位置 $P_{object}$。
    - 设计动机：用 3D 表示替代 2D 外观，避免了不同相机角度下 2D 外观剧变的问题，是后续几何归一化的基础。使用 SAM 实现了类别无关的物体检测，不需要预定义物体类别。

2. **自我中心变换（Egocentric Transformation）**:

    - 功能：将所有 3D 上下文归一化到以主体头部为原点、视线方向为 z 轴的统一坐标系
    - 核心思路：先将姿态和物体位置平移到以头部位置为原点，再旋转使视线方向 $\mathbf{v}$ 对齐到 $\mathbf{z}=[0,0,1]$。为保持旋转一致性，提出 cyclotorsion rotation——受眼球反转运动启发，约束旋转矩阵 $R=\text{Euler}(\theta,\phi,0)$ 使水平轴保持水平，避免简单轴角旋转带来的 z 轴不一致问题。变换后：$P'_{pose}=sR(P_{pose}-\mathbf{t}_{pose})$, $P'_{object}=R(P_{object}-\mathbf{t}_{object})$。
    - 设计动机：几何归一化消除了相机位姿变化带来的 2D 表示多样性，让网络只需学习自我中心空间中更简洁的空间关系，大幅降低学习难度。

3. **D3 位置编码（Direction-Distance-Decomposed PE）**:

    - 功能：将 3D 点编码为同时捕获方向和距离相似性的高维特征
    - 核心思路：对 3D 点 $\mathbf{p}$，将其分解为方向分量 $\mathbf{p}/\|\mathbf{p}\|$ 和距离分量 $\|\mathbf{p}\|$，分别施加正弦位置编码后拼接：$\tilde{\gamma}(\mathbf{p})=\gamma(\mathbf{p}/\|\mathbf{p}\|)\oplus\gamma(\|\mathbf{p}\|)$。与标准位置编码只在参考点附近相似度高不同，D3 编码在从原点到参考点的方向上逐渐增加相似度（形如径向注视模式），更符合注视行为特征。
    - 设计动机：人类注视固定通常集中在视线中心附近，方向和距离是影响注视的关键因素。D3 编码显式捕获这两个维度的信息，比标准 PE 更适合注视估计任务。

### 损失函数 / 训练策略
损失函数为预测注视方向与真值之间的角度误差：$\mathcal{L}=\arccos(\mathbf{g}^T\mathbf{g}_{GT})$。使用 AdamW 优化器，学习率 0.0014，在单块 A10G GPU 上训练 20 个 epoch。SAM、姿态估计器和视线方向估计器的权重冻结不参与训练。Transformer 编解码器各 3 层。

## 实验关键数据

### 主实验

| 数据集 | 指标(3D MAE↓) | GA3CE | GFIE (之前SOTA) | 提升 |
|--------|--------------|-------|-----------------|------|
| GFIE | 3D MAE (°) | **11.1** | 17.7 | **37%** |
| GFIE (零样本深度) | 3D MAE (°) | 12.3 | 17.7 | 31% |
| GFIE + GFM | 3D MAE (°) | **10.6** | 16.4 | 35% |
| CAD-120 (跨域) | 3D MAE (°) | **25.2** | 27.3 | 8% |
| CAD-120 + GFM | 3D MAE (°) | **15.8** | 19.8 | 20% |
| GAFA (全场景平均) | 3D MAE (°) | **19.9** | 22.9 | **13%** |

### 消融实验

| 配置 | GFIE 3D MAE | GAFA 3D MAE | 说明 |
|------|-------------|-------------|------|
| Appearance only | 19.4 | 22.9 | 仅外观（基线） |
| + Pose | 13.1 | 20.3 | 加入 3D 姿态 |
| + Pose + Object | **11.1** | **19.9** | 完整模型 |
| w/o ECT (无自我中心变换) | 15.3 | - | 变换贡献显著 |
| 标准 PE (代替 D3 PE) | 12.5 | - | D3 PE 有效 |

### 关键发现
- 3D 姿态是最重要的上下文，从 19.4° 降到 13.1°（GFIE），贡献最大
- 物体位置在 GFIE 数据集上贡献显著（13.1→11.1），因其场景中主体与物体交互频繁；在 GAFA 上贡献较小（20.3→19.9），因 GAFA 中主体与物体交互较少
- 零样本深度估计器（Depth Anything）替代真实深度传感器后性能下降仅 1.2°，说明方法对深度质量鲁棒
- Transformer 解码器注意力可视化表明，靠近真实注视目标的物体获得了最高注意力权重

## 亮点与洞察
- **自我中心变换的几何归一化**：用几何先验而非学习的方式消除了相机位姿变化的影响，简单高效，可迁移到任何需要处理多视角输入的 3D 理解任务
- **D3 位置编码**：方向-距离分解的设计巧妙地将人类视觉特性融入特征编码中，产生的径向注视模式天然适合注视任务。这种分解思想可推广到其他需要建模空间关系的任务（如指向估计、注意力预测）
- **无需 3DMM 或额外标注**：通过 SAM 和预训练姿态估计器实现了完全自动的 3D 上下文提取，降低了部署门槛
- **Cyclotorsion rotation** 是个精巧的工程设计，解决了视线对齐时旋转不一致的几何问题

## 局限与展望
- 当前仅使用物体位置，未利用物体外观特征（如 CLIP 语义），加入语义信息可能进一步提升
- 依赖预训练模型（SAM、姿态估计器）的质量，极端遮挡或罕见姿态下可能退化
- 单帧设置，未利用时序信息。GAFA 的时序版本已表明时序有帮助（21.7° vs 22.9°）
- GAFA 数据集上 3D 姿态估计可能不够准确（远距离主体），限制了性能上限
- 未探索多人场景中的注视估计

## 相关工作与启发
- **vs GFIE**: GFIE 先估计 2D 注视点再用深度约束 3D 注视方向，是解耦的两步法；GA3CE 直接端到端学习 3D 注视，无需 2D gaze following 模块
- **vs GAFA**: GAFA 利用时序 RGB 和 2D 体流估计注视；GA3CE 用 3D 姿态和物体位置作为更鲁棒的中间表示，单帧即超越 GAFA
- 自我中心变换的思想可用于其他社交信号理解任务（如手势方向估计、交互意图预测）

## 评分
- 新颖性: ⭐⭐⭐⭐ 自我中心变换和 D3 PE 是新颖的设计，但整体框架仍是成熟组件的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、完整消融、可视化分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰、动机充分，方法描述详细
- 价值: ⭐⭐⭐⭐ 无约束场景注视估计的显著进步，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](../../ECCV2024/human_understanding/occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](../../ECCV2024/human_understanding/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)

</div>

<!-- RELATED:END -->
