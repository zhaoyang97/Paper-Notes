---
title: >-
  [论文解读] Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving
description: >-
  [ECCV 2024][人体理解][情感识别] 本文针对辅助驾驶场景提出 UbH-GCN，利用上半身骨骼序列构建层次化图结构（UbH-Graph）动态建模关节运动与情感的关系，并引入类别特定变化机制平衡不均衡数据分布，在 AIDE 辅助驾驶数据集上超越现有多模态方法。 领域现状：情感识别在辅助驾驶中至关重要——理解驾驶员的情…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "情感识别"
  - "骨骼序列"
  - "图卷积网络"
  - "辅助驾驶"
  - "层次图结构"
---

# Upper-Body Hierarchical Graph for Skeleton Based Emotion Recognition in Assistive Driving

**会议**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03697.pdf)
**代码**: [https://github.com/jerry-wjh/UbH-GCN](https://github.com/jerry-wjh/UbH-GCN)  
**领域**: 人体理解  
**关键词**: 情感识别, 骨骼序列, 图卷积网络, 辅助驾驶, 层次图结构

## 一句话总结

本文针对辅助驾驶场景提出 UbH-GCN，利用上半身骨骼序列构建层次化图结构（UbH-Graph）动态建模关节运动与情感的关系，并引入类别特定变化机制平衡不均衡数据分布，在 AIDE 辅助驾驶数据集上超越现有多模态方法。

## 研究背景与动机

**领域现状**：情感识别在辅助驾驶中至关重要——理解驾驶员的情绪状态（如愤怒、疲倦、焦虑）有助于提升行车安全和人机交互体验。当前情感识别研究主要依赖面部表情、语音信号和生理信号（如 EEG、心率），而基于身体姿态的方法相对较少。

**现有痛点**：(1) 面部和语音方法在驾驶场景中受限——驾驶员可能戴口罩/墨镜、车内噪声大、光照变化剧烈；(2) 现有基于骨骼的方法大多使用全身姿态，但驾驶场景中下半身被遮挡，只能获取上半身骨骼；(3) 传统 GCN 方法使用预定义的固定邻接矩阵，无法自适应地建模不同情感状态下关节间的动态关系；(4) 驾驶场景中不同情感类别的数据严重不均衡，影响模型泛化能力。

**核心矛盾**：驾驶场景中可用的身体信息有限（仅上半身），且动作幅度受限（坐姿状态），导致不同情感之间的骨骼运动差异微妙。固定的图结构和标准的分类训练策略难以捕获这些细微差异。

**本文目标** (1) 如何仅从上半身骨骼中有效提取情感特征？(2) 如何动态建模关节与情感之间的关系？(3) 如何处理驾驶数据中的类别不均衡问题？

**切入角度**：作者观察到上半身运动中不同层次的关节组具有不同的情感表达作用——例如手部动作表达焦虑，头部姿态表达疲倦。通过层次化的图结构将单个关节、关节组、全身动作分层建模，可以更好地捕获多尺度的情感线索。

**核心 idea**：通过上半身层次图结构分层捕获多尺度情感运动特征，配合类别特定变化机制平衡训练数据，实现驾驶场景下的精准情感识别。

## 方法详解

### 整体框架

UbH-GCN 接受上半身骨骼关节序列作为输入（包括头部、肩部、手臂、手部等关节的 2D/3D 坐标随时间的变化）。骨骼数据首先通过 UbH-Graph 模块构建多层次图表示，然后经过多层图卷积提取时空特征，结合类别特定变化增强模块进行特征调制，最后通过分类头输出情感类别。支持多模态（关节坐标、骨骼向量、关节运动）的后融合。

### 关键设计

1. **上半身层次图结构（UbH-Graph）**:

    - 功能：动态建模上半身关节间的层次化拓扑关系，捕获多尺度情感运动模式
    - 核心思路：将上半身关节组织为三层层次结构。底层是每个单独关节节点（如左手腕、右肘等），中间层是功能性关节组（如左臂组、头部组、躯干组），顶层是全局上半身表示。层次间通过可学习的聚合连接。在每一层，邻接矩阵由两部分动态构成：$A = A_{phys} + A_{learn}$，其中 $A_{phys}$ 是基于物理连接的固定拓扑，$A_{learn}$ 是通过注意力机制学习的动态关系
    - 设计动机：不同情感涉及不同的关节组合——紧张时手部和肩部紧张，疲劳时头部下垂。层次化的图结构能分别捕获局部（单关节）、中观（关节组）和全局（整体姿态）的情感线索

2. **动态图卷积模块（Dynamic Graph Convolution）**:

    - 功能：在时序维度上动态调整图结构，捕获情感表达的时序动态
    - 核心思路：在每个时间步，根据当前帧的关节特征计算注意力权重，更新邻接矩阵的可学习部分。具体地，对任意两个关节 $i, j$，其动态连接权重为 $a_{ij}^t = \text{softmax}(\phi(h_i^t)^T \psi(h_j^t) / \sqrt{d})$，其中 $\phi, \psi$ 是线性变换。时序卷积采用多尺度 TCN 提取不同时间跨度的运动模式
    - 设计动机：情感表达是时变的——例如从平静到愤怒的过程中，关节间的协调关系会发生变化。静态邻接矩阵无法捕获这种动态变化

3. **类别特定变化增强（Class-Specific Variation Enhancement）**:

    - 功能：平衡不同情感类别的特征分布，缓解数据不均衡问题
    - 核心思路：在训练过程中，维护每个情感类别的特征均值和协方差。对少数类样本，通过在特征空间中沿类别特定方向施加高斯扰动来增加变异性：$h_{aug} = h + \epsilon \cdot \sigma_c$，其中 $\sigma_c$ 是类别 $c$ 的特征标准差，$\epsilon \sim \mathcal{N}(0, 1)$。这使得少数类在特征空间中的覆盖范围更广，降低过拟合风险
    - 设计动机：驾驶数据中某些情感（如愤怒）的样本远少于其他类别（如平静）。传统的过采样或类别加权策略只在数据或损失层面操作，而特征层面的增强更直接且不引入噪声样本

### 损失函数 / 训练策略

使用交叉熵损失配合类别权重平衡：$L = -\sum_c w_c \cdot y_c \log(\hat{y}_c)$。四模态后融合策略：分别训练关节坐标（joint）、骨骼向量（bone）、关节运动（joint motion）、骨骼运动（bone motion）四个模型，预测时对 softmax 输出加权求和。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 准确率 | F1 分数 | 模态 |
|--------|------|--------|---------|------|
| AIDE | UbH-GCN (4-way) | **62.7%** | **0.584** | 骨骼 |
| AIDE | AIDE-baseline (multimodal) | 58.3% | 0.541 | 面部+骨骼+语音 |
| AIDE | CTR-GCN | 56.8% | 0.519 | 骨骼 |
| AIDE | HD-GCN | 57.2% | 0.525 | 骨骼 |
| AIDE | 2s-AGCN | 55.1% | 0.503 | 骨骼 |
| Emliya | UbH-GCN (4-way) | **78.4%** | **0.762** | 骨骼 |

### 消融实验

| 配置 | 准确率 | 说明 |
|------|--------|------|
| Full UbH-GCN | **62.7%** | 完整模型 |
| w/o 层次图 (单层图) | 59.1% | 层次结构贡献约 3.6% |
| w/o 动态邻接矩阵 (固定拓扑) | 58.5% | 动态图贡献约 4.2% |
| w/o 类别特定变化 | 60.3% | 变化增强贡献约 2.4% |
| 全身骨骼 (含下半身估计) | 61.2% | 上半身更适合驾驶场景 |
| 单模态 (joint only) | 58.9% | 多模态融合贡献约 3.8% |

### 关键发现
- UbH-GCN 仅使用骨骼信息就超越了使用面部+骨骼+语音的多模态基线方法，说明骨骼在驾驶场景中的情感表达潜力被严重低估
- 动态邻接矩阵是最关键的组件，说明情感识别中关节间的动态协调关系比固定拓扑更重要
- 在 Emliya 日常动作数据集上也表现良好，证明方法具有跨场景泛化能力

## 亮点与洞察
- **上半身专用设计**非常精准地匹配了驾驶场景的实际约束。这种"根据应用场景裁剪输入"的思路可以迁移到其他受限场景，如轮椅用户识别、手术室医生姿态分析等
- **类别特定变化增强**在特征空间而非数据空间进行数据增强，既避免了传统过采样产生的重复样本问题，又比 SMOTE 等方法更好地保持了特征分布的合理性
- 仅骨骼模态超越多模态方法的结果具有启发性，说明在特定场景下，精心设计的单模态方法可能比粗暴的多模态融合更有效

## 局限与展望
- 情感识别的绝对准确率仍然不高（62.7%），说明仅从上半身骨骼识别情感仍然非常困难
- AIDE 数据集规模较小，且仅包含特定人群和驾驶场景，泛化性有待验证
- 未引入时序建模的先进架构（如 Transformer），可能遗漏了长程时序依赖
- 可以考虑将上半身骨骼与车辆行为数据（如方向盘角度、刹车频率）融合，提供更丰富的情感线索

## 相关工作与启发
- **vs CTR-GCN**: CTR-GCN 使用通道级拓扑细化但不考虑层次结构。UbH-GCN 引入层次化图结构，从多尺度建模关节关系，在驾驶场景效果更好
- **vs HD-GCN**: HD-GCN 使用层次化分解但基于预定义的层次结构且针对动作识别设计。UbH-GCN 专门为上半身情感识别设计了层次，且邻接矩阵是动态学习的
- **vs ST-GCN**: ST-GCN 是 GCN 动作识别的开创性工作，使用固定邻接矩阵。UbH-GCN 在其基础上引入了动态性、层次性和上半身特化设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 将层次图结构和类别特定增强应用于辅助驾驶情感识别的切入角度新颖
- 实验充分度: ⭐⭐⭐ 数据集较少（仅两个），且 AIDE 数据集规模不大
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 对辅助驾驶中的情感识别有实用价值，骨骼超越多模态的发现有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Modeling and Driving Human Body Soundfields through Acoustic Primitives](modeling_and_driving_human_body_soundfields_through_acoustic_primitives.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](../../CVPR2026/human_understanding/cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)
- [\[CVPR 2026\] Action Motifs: Self-Supervised Hierarchical Representation of Human Body Movements](../../CVPR2026/human_understanding/action_motifs_self-supervised_hierarchical_representation_of_human_body_movement.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)

</div>

<!-- RELATED:END -->
