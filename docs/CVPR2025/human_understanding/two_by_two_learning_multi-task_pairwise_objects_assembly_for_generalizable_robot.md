---
title: >-
  [论文解读] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation
description: >-
  [CVPR 2025][人体理解][3D组装] 本文提出了 2BY2 数据集——首个大规模日常成对物体组装数据集（18类任务、517对物体），并设计了一种两步式 SE(3) 位姿估计网络，利用等变特征实现多任务物体配对组装，在所有任务上达到 SOTA，并通过真实机器人实验验证了泛化能力。 领域现状：3D 组装任务（如家具组装…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "3D组装"
  - "成对物体装配"
  - "SE(3)位姿估计"
  - "等变特征"
  - "机器人操作"
---

# Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation

**会议**: CVPR 2025  
**arXiv**: [2504.06961](https://arxiv.org/abs/2504.06961)  
**代码**: [https://tea-lab.github.io/TwoByTwo/](https://tea-lab.github.io/TwoByTwo/)  
**领域**: 人体/物体理解  
**关键词**: 3D组装, 成对物体装配, SE(3)位姿估计, 等变特征, 机器人操作

## 一句话总结

本文提出了 2BY2 数据集——首个大规模日常成对物体组装数据集（18类任务、517对物体），并设计了一种两步式 SE(3) 位姿估计网络，利用等变特征实现多任务物体配对组装，在所有任务上达到 SOTA，并通过真实机器人实验验证了泛化能力。

## 研究背景与动机

**领域现状**：3D 组装任务（如家具组装、零件配合）在日常生活中无处不在，也是未来家庭机器人的核心能力之一。现有基准和数据集主要集中在几何碎片重组（如 Breaking Bad）或工厂零件装配（如 Factory），对日常物体交互场景覆盖不足。

**现有痛点**：现有方法在几何碎片匹配方面表现不错，但面对日常组装场景（如将花插入花瓶、将面包放入面包机）时表现不佳。这类任务不仅需要几何对齐，还需要理解物体之间的功能关系和空间语义。此外，现有数据集规模有限，任务种类单一，缺少对称性标注，无法满足日常组装的多样性需求。

**核心矛盾**：几何匹配方法只关注局部形状对齐，忽略了日常组装中的语义约束和功能性关系（如"瓶盖需要盖在瓶子上"而非随机拼接），导致场景迁移能力差。

**本文目标**：(1) 构建首个覆盖 18 种日常组装任务的大规模成对物体数据集；(2) 设计能同时处理多种组装任务且具有泛化能力的 SE(3) 位姿估计方法。

**切入角度**：作者观察到人类在组装时通常是分步进行的——先放好底座/容器（如花瓶），再放置配件（如花），因此提出模仿这种两步组装逻辑的网络架构。

**核心 idea**：用两步式网络架构分别预测底座和配件的 SE(3) 位姿，结合 SE(3) 等变特征提取和跨物体特征融合，实现精确的成对物体装配。

## 方法详解

### 整体框架

输入为两个点云 $\mathcal{P}_A$（配件，如花）和 $\mathcal{P}_B$（底座，如花瓶），各包含 1024 个 3D 点。两个点云经过随机 SO(3) 旋转和质心平移增强。网络分两步预测：Branch B 先预测底座的规范位姿（旋转 + 平移），将其变换到标准空间；然后 Branch A 在已变换的底座基础上，结合配件信息，预测配件的规范位姿。最终输出两个 SE(3) 变换，将物体组装到预定义的规范空间。

### 关键设计

1. **两步式成对网络架构（Two-step Pairwise Network）**:

    - 功能：模仿人类组装逻辑，先定位底座再定位配件，分步预测各物体的 6DoF 位姿
    - 核心思路：Branch B 使用 Two-scale VN DGCNN 编码器提取 $\mathcal{P}_B$ 的 SE(3) 等变特征 $\mathcal{E}_B$，通过 MLP 预测头输出旋转和平移。变换后的 $\mathcal{P}_B$ 与原始 $\mathcal{P}_A$ 一起送入 Branch A，提取 $\mathcal{P}_A$ 的等变特征 $\mathcal{E}_A$ 和 $\mathcal{P}_B$ 的 SO(3) 不变特征 $\mathcal{I}_B$，通过特征融合后预测配件位姿
    - 设计动机：联合预测两个物体位姿时，误差会相互干扰；分步预测可以隔离误差传播，先确保底座位姿准确，再在此基础上预测配件

2. **双尺度 SE(3) 等变编码器（Two-scale VN DGCNN）**:

    - 功能：从点云中提取既保持 SE(3) 等变性又能捕捉多尺度几何信息的特征
    - 核心思路：基于 Vector Neuron DGCNN 扩展，使用两个不同 K 值的 KNN 分支进行特征提取，分别捕获局部细节和全局形状信息，然后拼接并通过额外的 VN 卷积层融合。通过将点云减去质心实现 T(3) 平移等变性，即 $f(\mathcal{P} + \mathcal{T}) = f(\mathcal{P}) + \mathcal{T}$
    - 设计动机：单一尺度的 KNN 图无法同时捕捉精细几何细节和整体形状；金字塔结构可以同时获取全局和局部信息，提升特征表达能力

3. **跨物体特征融合模块（Cross Object Fusion）**:

    - 功能：将底座的几何信息注入配件的特征表示中，使 Branch A 能综合两个物体的信息进行位姿预测
    - 核心思路：通过逐点乘法（element-wise multiplication）将 $\mathcal{I}_B$（SO(3) 不变特征）和 $\mathcal{E}_A$（SE(3) 等变特征）融合。这样保证了融合后特征对 $\mathcal{P}_A$ 旋转的等变性，即 $f(R \cdot (\mathcal{I}_B * \mathcal{E}_A)) = R \cdot f(\mathcal{I}_B * \mathcal{E}_A)$
    - 设计动机：使用不变特征 × 等变特征的方式，既引入了底座的形状信息作为上下文，又不破坏配件特征的旋转等变性

### 损失函数 / 训练策略

损失函数为旋转损失和平移损失的加权和：$\mathcal{L} = \lambda_{rot}\mathcal{L}_{rot} + \lambda_{trans}\mathcal{L}_{trans}$。平移使用 L1 损失，旋转使用测地距离（Geodesic Distance）：$\mathcal{L}_{rot} = \arccos\left(\frac{\text{tr}(\mathcal{R}_{gt}\mathcal{R}_{pred}^T) - 1}{2}\right)$。测地距离度量旋转流形上两个旋转之间的最短路径，提供平滑有界的角度误差和稳定梯度。

训练策略上，Branch A 和 Branch B 独立训练。Branch A 训练时使用真值（canonical pose 下的 $\mathcal{P}_B$），测试时才使用 Branch B 的预测结果进行级联推理。旋转和平移使用分离的 MLP 预测头，避免不同收敛速度的干扰。

## 实验关键数据

### 主实验

| 方法 | 平移 RMSE(T) ↓ | 旋转 RMSE(R) ↓ |
|------|--------------|--------------|
| Jigsaw | 0.360 | 53.34 |
| Puzzlefusion++ | 0.342 | 58.23 |
| NSM | 0.284 | 70.30 |
| SE(3)-Assembly | 0.233 | 52.34 |
| **Ours** | **0.110** | **41.44** |

在全部 18 个细分任务上均超越 baseline，平均平移 RMSE 降低 0.046，旋转 RMSE 降低 8.97。

### 消融实验

| 编码器 | 平移 RMSE(T) | 旋转 RMSE(R) |
|--------|------------|------------|
| PointNet | 0.264 | 75.38 |
| DGCNN | 0.277 | 72.46 |
| VN DGCNN（单尺度）| 0.123 | 44.67 |
| w/o Two-step | 0.139 | 45.20 |
| **Ours (Two-scale VN DGCNN + Two-step)** | **0.110** | **41.44** |

### 关键发现

- 双尺度 VN DGCNN 相比单尺度 VN DGCNN 在所有任务上均有提升，验证了多尺度特征提取的有效性
- 两步式网络设计相比端到端联合预测（w/o Two-step）有明显提升，证实分步预测可有效隔离误差
- 真实机器人实验中，方法总体成功率达 77.5%（baseline SE(3) Assembly 仅 22.5%），在插花任务上成功率达 100%

## 亮点与洞察

- **数据集贡献突出**：2BY2 是首个面向日常场景的大规模成对组装数据集，涵盖 18 种任务（从插花到插 USB），填补了该领域的空白
- **仿人组装逻辑**：两步式架构巧妙地模仿了人类的自然组装顺序，既有直觉合理性又有工程效果
- **等变性设计精细**：在特征融合时刻意保持等变/不变性，确保数学性质不被破坏，体现了对对称性的深入理解

## 局限与展望

- 当前方法假设可以获得干净的点云输入，在噪声点云或部分遮挡场景下表现未知
- 数据集中的对称性标注需要人工完成，扩展新任务的人工成本较高
- 真实机器人实验仅在 4 个任务上测试，更多复杂场景（如多步组装序列）的泛化能力有待验证
- 未来可结合视觉基础模型自动获取语义信息，或引入基于扩散模型的位姿细化策略

## 相关工作与启发

- Breaking Bad、Neural Shape Mating 等碎片重组方法为形状匹配提供了基础，但 2BY2 将问题推广到日常功能性组装
- SE(3) 等变网络（如 VNN）在这类需要任意位姿泛化的任务中展现了强大的优势，值得在更多机器人操作任务中推广
- 两步式预测思路可推广到更复杂的多体组装序列

## 评分

- **新颖性**: 8/10 — 数据集和两步式等变网络架构的结合是新颖的贡献
- **实验充分度**: 8/10 — 18 个任务的全面评测 + 真实机器人实验，但消融分析可更细致
- **写作质量**: 7/10 — 结构清晰，但部分细节描述可以更精炼
- **价值**: 8/10 — 数据集和方法对机器人操作社区都有较高价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)
- [\[ICCV 2025\] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning](../../ICCV2025/human_understanding/ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)
- [\[CVPR 2025\] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)
- [\[CVPR 2025\] FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly](freeuv_ground-truth-free_realistic_facial_uv_texture_recovery_via_cross-assembly.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)

</div>

<!-- RELATED:END -->
