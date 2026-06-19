---
title: >-
  [论文解读] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching
description: >-
  [ICCV 2025][图像生成][机器人操作] EC-Flow 提出了"具身中心光流"范式，从无动作标注的 RGB 视频中预测机器人本体的像素级运动轨迹，结合 URDF 运动学约束将视觉预测转化为可执行动作，在可变形物体、遮挡和非位移操作等场景中大幅超越物体中心方法。 当前语言引导的机器人操作系统主要依赖模仿学习（VLA…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "机器人操作"
  - "无动作标注视频学习"
  - "光流预测"
  - "URDF运动学"
  - "扩散模型"
---

# EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching

**会议**: ICCV 2025  
**arXiv**: [2507.06224](https://arxiv.org/abs/2507.06224)  
**代码**: [ec-flow1.github.io](https://ec-flow1.github.io)  
**领域**: 图像生成  
**关键词**: 机器人操作, 无动作标注视频学习, 光流预测, URDF运动学, 扩散模型

## 一句话总结

EC-Flow 提出了"具身中心光流"范式，从无动作标注的 RGB 视频中预测机器人本体的像素级运动轨迹，结合 URDF 运动学约束将视觉预测转化为可执行动作，在可变形物体、遮挡和非位移操作等场景中大幅超越物体中心方法。

## 研究背景与动机

当前语言引导的机器人操作系统主要依赖模仿学习（VLA 模型），需要大规模低层动作标注数据集。虽然互联网上存在海量无动作标注的操作视频，但如何从中学习操作策略仍是未解难题。

现有从视频学习操作的方法分为两类：

**视频预测 + 模仿学习**：先预测未来帧，再通过逆动力学模型推断动作。但仍需动作标注数据训练逆动力学模型

**物体中心光流方法**（如 AVDC、Track2Act）：预测物体的光流轨迹并通过物体变换推断动作。无需动作标注，但存在三大根本缺陷：
   - **刚体假设**：假设物体各部分做统一刚性变换，无法处理可变形物体（如衣物折叠）
   - **遮挡脆弱性**：动作完全依赖物体状态变化推断，物体被遮挡时失效
   - **非位移操作失败**：无法捕捉旋转（如转动开关）或微小运动（如按下鼠标按钮）

核心洞察：**将预测目标从物体光流转向机器人本体（embodiment）光流，可以从根本上规避上述三个限制。** 机器人本体的运动：(a) 不依赖物体属性→适用于任何物体，(b) 在大多数场景中可见→对遮挡鲁棒，(c) 直接反映执行动作→可处理非位移任务。

## 方法详解

### 整体框架

EC-Flow 包含两个核心模块：
1. **具身中心光流预测**（Sec 3.2）：从初始帧和语言指令出发，预测机器人本体上随机采样点在未来 T 步的 2D 轨迹
2. **运动学感知动作计算**（Sec 3.3）：利用机器人 URDF 文件的运动学约束，将 2D 光流转化为末端执行器的 6-DoF 位姿变换

### 关键设计

1. **带扩散模型的光流预测**：

    - 数据构建：用 Grounded SAM 分割机器人本体掩码 → 在掩码内随机采样 $N_p=400$ 个点 → 用 CoTracker 跟踪其在整个视频中的 2D 轨迹
    - 采用条件扩散模型预测光流，条件信号 $\mathbf{c} = [\tilde{\mathbf{v}}, \tilde{\mathbf{l}}, \tilde{\mathbf{s}}]$ 包含：视觉上下文（ResNet-50 编码初始帧）、语言引导（CLIP 文本编码器）、起始状态（初始点坐标）
    - 训练目标：$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \mathbf{z}_0, \epsilon}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{c})\|_2^2]$
    - 推理时使用 DDIM 采样生成完整 T=8 步轨迹

2. **目标图像预测 — 对齐物体交互和语言指令**：

    - 挑战：纯具身光流预测可能生成不与目标物体有效交互的动作
    - 解决方案：引入辅助目标图像预测分支，与光流预测分支共享扩散时间步
    - 目标图像生成器接收增强条件 $\mathbf{c_t}^{\text{img}} = [\tilde{\mathbf{v}}, \tilde{\mathbf{l}}, \tilde{\mathbf{s}}, f_t^{\text{flow}}]$，其中 $f_t^{\text{flow}}$ 为光流预测输出
    - 总损失：$L = L_{\text{flow}} + \lambda L_{\text{image}}$（$\lambda=0.4$）
    - 设计动机：通过联合优化，建立隐式约束——生成的动作必须导致物理合理的物体状态变化，且与语言指令对齐

3. **URDF 感知的动作计算**：

    - 步骤 1（点分配）：利用 URDF 文件的几何属性计算各关节的 2D 包围盒，将采样点分配到对应关节（仅保留唯一归属于单个关节的点）
    - 步骤 2（动作优化）：通过最小化重投影误差来优化末端执行器位姿：
   
    $\mathbf{T}_{ee}^* = \arg\min_{\mathbf{T}_{ee}} \sum_{j=1}^M \sum_{i=1}^{N_j} \|\pi(\mathbf{T}_{ee} \cdot {}_j^{ee}\mathbf{T} \cdot \mathbf{P}_{ji}^{3D}) - \mathbf{P}_{ji}^{(t+1)_{2D}}\|_2$
   
    - 设计动机：不同关节有不同运动学约束，不能简单计算统一变换；URDF 是机器人系统的标准配置文件，无需额外标注

### 损失函数 / 训练策略

- 光流预测：扩散噪声预测 MSE 损失
- 目标图像：扩散噪声预测 MSE 损失（权重 0.4）
- 训练配置：8 张 4090 GPU，batch size 56，flow lr=5e-5，image lr=1e-4，DDIM 250 步采样

## 实验关键数据

### 主实验 — Meta-World 基准

| 方法 | 数据需求 | 平均成功率 ↑ |
|------|---------|-------------|
| BC-Scratch | 动作标注 | 0.204 |
| BC-R3M | 动作标注 | 0.360 |
| Diffusion Policy | 动作标注 | 0.298 |
| UniPi | 视频+动作标注 | 0.093 |
| AVDC (物体中心) | 仅视频 | 0.489 |
| Track2Act (物体中心) | 仅视频 | 0.556 |
| **EC-Flow (具身中心)** | **仅视频** | **0.720** |

EC-Flow 以 16.4% 的优势超越最强基线。特别在 btn-top-press（1.00 vs 0.40）和 hammer-strike（0.88 vs 0.24）等遮挡/非位移任务中提升显著。

### 消融实验 — 各组件贡献

| # | 光流预测方式 | 目标图像 | 点过滤 | 仅末端 | 平均成功率 |
|---|-----------|---------|-------|-------|-----------|
| 1 | 端到端 EC-Flow | ✓ | ✓ | ✗ | **0.720** |
| 2 | 视频+GT光流 | ✗ | ✓ | ✗ | 0.636 |
| 3 | 端到端 EC-Flow | ✗ | ✗ | ✗ | 0.582 |
| 4 | 端到端 EC-Flow | ✓ | ✓ | ✓ | 0.604 |
| 5 | 视频+GT光流 | ✓ | ✓ | ✗ | 0.667 |

目标图像预测（#3 vs #1: +13.8%）和全身点建模（#4 vs #1: +11.6%）贡献最大。

### 关键发现

1. **遮挡鲁棒性**：物体中心方法在物体被遮挡时完全失败，而 EC-Flow 通过机器人其他可见关节的运动推断动作
2. **视频预测的幻觉问题**：视频预测模型可能生成多个机械臂的幻觉，导致光流跟踪错误；端到端光流预测避免了这一问题
3. **真实世界验证**（7 个任务）：EC-Flow 在可变形物体操作上提升 45%，非位移操作提升 80%（相对 Track2Act）
4. **跨具身数据可用**：初步实验表明，加入 50 个人类视频可将 2-demo 场景的成功率从 46% 提升至 70%

## 亮点与洞察

- "从物体中心到具身中心"的范式转换是本文最核心的贡献，极为简洁但效果显著
- 目标图像预测作为辅助任务来对齐语言-物体-动作的设计很巧妙，避免了直接预测物体光流的困难
- 整个系统仅需 RGB 视频 + URDF 文件，没有动作标注、没有 3D 点云、没有特殊传感器，部署成本极低

## 局限与展望

- 当前需要手动设置操作起始位姿，未集成自动抓取位姿生成
- 夹爪状态（开/关）从物体中心方法借鉴，仍依赖物体分割
- DDIM 250 步采样的推理速度较慢（光流预测 4.37s），可用 flow matching 加速
- 深度感知依赖 D435i 摄像头，对复杂形变任务（如折叠衣物）的精度受限

## 相关工作与启发

- 与 General Flow 的区别：General Flow 需要 RGBD 输入预测 3D 流场，EC-Flow 仅用 RGB
- 与 MT-π 的区别：MT-π 依赖 5 个预定义夹爪关键点且必须始终可见，EC-Flow 对部分遮挡鲁棒
- 启发：具身中心的思路可能推广到多臂协作、灵巧手操作等更复杂的具身场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （具身中心光流的范式转换非常优雅）
- 实验充分度: ⭐⭐⭐⭐ （仿真+真实，充分消融，但真实实验规模有限）
- 写作质量: ⭐⭐⭐⭐⭐ （动机清晰，方法描述详尽）
- 价值: ⭐⭐⭐⭐⭐ （显著降低了从视频学习操作的门槛）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](a0_affordance_aware_hierarchical_model_robotic_manipulation.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[NeurIPS 2025\] Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems](../../NeurIPS2025/image_generation/equivariant_flow_matching_for_symmetry-breaking_bifurcation_problems.md)
- [\[CVPR 2025\] VideoWorld: Exploring Knowledge Learning from Unlabeled Videos](../../CVPR2025/image_generation/videoworld_exploring_knowledge_learning_from_unlabeled_videos.md)
- [\[CVPR 2026\] PoseD-Flow: Versatile and Guided Flow Matching Model of Human Pose](../../CVPR2026/image_generation/posed-flow_versatile_and_guided_flow_matching_model_of_human_pose.md)

</div>

<!-- RELATED:END -->
