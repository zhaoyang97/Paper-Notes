---
title: >-
  [论文解读] ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping
description: >-
  [CVPR 2025][机器人][零样本抓取] ZeroGrasp 提出了一个基于八叉树条件变分自编码器（CVAE）的统一框架，从单张 RGB-D 图像同时完成高分辨率 3D 物体重建和 6D 抓取姿态预测，通过多物体编码器和 3D 遮挡场建模物体间关系，在 GraspNet-1B 基准上达到 SOTA，并在真实机器人上验证了泛化能力。
tags:
  - "CVPR 2025"
  - "机器人"
  - "零样本抓取"
  - "3D重建"
  - "八叉树"
  - "遮挡推理"
  - "抓取姿态预测"
---

# ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping

**会议**: CVPR 2025  
**arXiv**: [2504.10857](https://arxiv.org/abs/2504.10857)  
**代码**: [https://sh8.io/#/zerograsp](https://sh8.io/#/zerograsp)  
**领域**: 3D视觉 / 机器人抓取  
**关键词**: 零样本抓取, 3D重建, 八叉树, 遮挡推理, 抓取姿态预测

## 一句话总结

ZeroGrasp 提出了一个基于八叉树条件变分自编码器（CVAE）的统一框架，从单张 RGB-D 图像同时完成高分辨率 3D 物体重建和 6D 抓取姿态预测，通过多物体编码器和 3D 遮挡场建模物体间关系，在 GraspNet-1B 基准上达到 SOTA，并在真实机器人上验证了泛化能力。

## 研究背景与动机

**领域现状**：机器人抓取需要对目标物体的几何形状有准确理解。当前主流方法大多直接从部分观测信息（如点云或深度图）回归抓取姿态，而不显式建模物体的完整几何形状。少数利用多视角重建的方法则需要额外拍摄多张图像，增加了系统复杂性和计算开销。

**现有痛点**：不建模几何的方法容易导致意外碰撞和不稳定的抓取接触；多视角重建在密闭空间（如货架、箱子）中不可行；现有数据集在 3D 形状标注和物理有效抓取标注方面严重不足，限制了从单视角进行零样本抓取的能力。

**核心矛盾**：精确的抓取需要高质量的 3D 重建来进行物理约束和碰撞检测，但单视角重建本身存在巨大的不确定性，尤其在多物体堆叠场景中遮挡严重，重建和抓取这两个任务之前是分离的。

**本文目标**：(1) 从单张 RGB-D 图像实现近实时的高分辨率 3D 重建 + 6D 抓取姿态预测；(2) 处理多物体场景中的遮挡和碰撞；(3) 仅用合成数据训练即可泛化到真实世界的新物体。

**切入角度**：作者观察到稀疏体素表示（如八叉树）在单视角 3D 重建中兼具速度和精度优势，且遮挡推理和物体间空间关系的建模对重建和抓取都有益处。

**核心 idea**：用八叉树 CVAE 统一重建与抓取两个任务，引入多物体编码器建模碰撞关系，引入 3D 遮挡场编码可见性信息，并利用重建结果通过接触约束优化抓取姿态。

## 方法详解

### 整体框架

输入是单张 RGB-D 图像。首先用 SAM2 生成 2D 实例分割掩码，结合图像特征和深度图将每个物体反投影到 3D 空间构建输入八叉树。输入八叉树经过 CVAE 的 Prior 网络提取潜在特征，再通过多物体编码器（Transformer）建模物体间关系，同时 3D 遮挡场编码遮挡信息。最终 Decoder 预测每个物体的完整八叉树（含 SDF、法线和抓取姿态）。预测的抓取姿态还可利用重建结果进行基于接触约束的精修。整个推理过程约 5 FPS。

### 关键设计

1. **八叉树条件变分自编码器（Octree-based CVAE）**:

    - 功能：对单视角 3D 重建中的不确定性进行概率建模，同时预测 3D 形状和抓取姿态
    - 核心思路：Encoder 接收输入和目标八叉树预测潜在分布，Prior 仅从输入八叉树预测先验分布，训练时最小化两者的 KL 散度。Decoder 逐层预测占用概率，在最终层预测 SDF、法线和抓取参数（graspness、quality、view、angle、width、depth）。采用经济监督策略只在有效抓取点学习抓取预测
    - 设计动机：八叉树的层级结构使得高分辨率重建既省内存又快速；CVAE 的概率建模能处理单视角重建的固有歧义性，相比确定性方法更能生成合理的完整形状

2. **多物体编码器（Multi-Object Encoder）**:

    - 功能：建模场景中多个物体之间的空间关系，实现无碰撞的重建和抓取预测
    - 核心思路：在潜在空间中使用 K 层标准 Transformer block，以所有物体的体素中心和特征作为输入 token，通过 self-attention（使用 RoPE 位置编码）让不同物体的特征互相交互。输入格式为 $[\ell_1, \ldots, \ell_L]$ 同时包含所有物体的潜在特征
    - 设计动机：Per-object 的 Prior 网络缺乏全局空间感知，无法避免物体间碰撞或重叠。在潜在空间而非原始空间做 attention 大幅减少了计算量

3. **3D 遮挡场（3D Occlusion Fields）**:

    - 功能：将可见性/遮挡信息局部化到体素级别，增强被遮挡区域的重建质量
    - 核心思路：将每个潜在空间的体素细分为 $B^3$ 个小块，投影到图像平面，通过深度测试判断每个小块是被目标自身遮挡（self-occlusion flag $o_{\text{self}}$）还是被其他物体遮挡（inter-occlusion flag $o_{\text{inter}}$）。两个 flag 拼接后通过 3 层 3D CNN 编码为遮挡特征，与潜在特征 concat
    - 设计动机：多物体编码器主要学习避免碰撞（局部上下文），而遮挡建模需要理解全局可见性关系——遮挡者和被遮挡者可能距离很远。3D 遮挡场通过简化的体渲染将全局信息局部化到每个体素，降低了学习难度

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{grasp}} + \mathcal{L}_{\text{KL}}$：

- **重建损失** $\mathcal{L}_{\text{rec}}$：每层占用 BCE + 最终层 SDF 和法线的 L1
- **抓取损失** $\mathcal{L}_{\text{grasp}}$：graspness 的 L1 + quality/angle/width/depth 的交叉熵
- **KL 散度** $\mathcal{L}_{\text{KL}}$：Encoder 和 Prior 分布匹配

另外还创建了 ZeroGrasp-11B 合成数据集（100 万 RGB-D 图像、12K 物体、113 亿物理有效抓取标注），基于 Objaverse-LVIS。

## 实验关键数据

### 主实验

| 方法 | GraspNet-1B CD↓ | F1↑ | NC↑ | ReOcS-Easy CD↓ | ReOcS-Hard CD↓ |
|------|-----------------|-----|-----|-----------------|-----------------|
| Minkowski | 6.84 | 81.45 | 77.89 | 5.59 | 9.11 |
| OCNN | 7.23 | 82.22 | 78.44 | 5.26 | 8.69 |
| OctMAE | 7.57 | 78.38 | 75.19 | 5.53 | 6.76 |
| **ZeroGrasp** | **6.05** | **84.08** | **78.46** | **4.76** | **6.73** |

### 消融实验

| 配置 | 说明 |
|------|------|
| Full model | 在 GraspNet-1B 和所有遮挡难度上均为最佳 |
| w/o Multi-Object Encoder | 物体间碰撞增加，重建质量下降 |
| w/o 3D Occlusion Fields | 遮挡区域重建明显变差，尤其在 ReOcS Hard 场景 |
| w/o Grasp Refinement | 抓取成功率下降，碰撞检测失效 |

### 关键发现

- 3D 遮挡场对困难遮挡场景贡献最大，在 ReOcS Hard 上提升显著
- 抓取姿态精修算法（基于接触约束 + 碰撞检测）利用重建结果可有效提升抓取准确率——验证了高质量重建确实能反哺抓取
- 仅用合成数据训练即可在真实机器人上泛化，成功抓取新物体
- 推理速度约 5 FPS，满足近实时需求

## 亮点与洞察

- **统一框架的设计哲学**：将重建和抓取耦合而不是分离，使得两个任务互相增强——重建提供碰撞检测和接触约束，抓取任务的监督信号也间接提升了表面质量。这种"共享表征、多任务互利"的思路可迁移到其他机器人操作任务
- **3D 遮挡场的简洁有效**：用简单的 ray casting + 二值 flag 就将复杂的全局可见性问题局部化，避免了复杂的体渲染。这个 trick 可以用于任何需要遮挡感知的 3D 任务
- **合成到真实的零样本迁移**：ZeroGrasp-11B 数据集的规模和多样性（12K 物体、113 亿抓取）是成功零样本泛化的关键

## 局限与展望

- 依赖 SAM2 的实例分割质量——分割失败会级联影响重建和抓取
- 目前仅支持平行夹爪，未覆盖灵巧手等更复杂的夹持器
- 八叉树表示虽然高效但对非常薄的结构（如纸张边缘）仍有分辨率限制
- 合成到真实的 domain gap 在复杂光照和反光材质下可能更明显

## 相关工作与启发

- **vs OctMAE**：OctMAE 做场景级重建但不分割也不预测抓取，ZeroGrasp 做实例级重建+抓取，通过分割后的 per-object 处理和多物体编码器在密集场景中表现更好
- **vs GraspNet / GSNet**：传统抓取方法不显式重建 3D 形状，碰撞检测只能用部分点云，ZeroGrasp 利用完整重建做更精确的碰撞检测，从根本上减少碰撞抓取
- **vs FoundationPose**：虽同属零样本 3D 方法但目标不同——FoundationPose 做位姿估计，ZeroGrasp 做形状重建+抓取，可互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一重建与抓取的框架设计有新意，3D 遮挡场是较新的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖基准测试、消融、真实机器人实验，数据集也自建了两个
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 对机器人抓取和 3D 重建社区都有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[CVPR 2025\] DRAWER: Digital Reconstruction and Articulation with Environment Realism](drawer_digital_reconstruction_and_articulation_with_environment_realism.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](../../NeurIPS2025/robotics/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[ECCV 2024\] Prioritized Semantic Learning for Zero-shot Instance Navigation](../../ECCV2024/robotics/prioritized_semantic_learning_for_zero-shot_instance_navigation.md)
- [\[CVPR 2026\] Obstruction Reasoning for Robotic Grasping](../../CVPR2026/robotics/obstruction_reasoning_for_robotic_grasping.md)

</div>

<!-- RELATED:END -->
