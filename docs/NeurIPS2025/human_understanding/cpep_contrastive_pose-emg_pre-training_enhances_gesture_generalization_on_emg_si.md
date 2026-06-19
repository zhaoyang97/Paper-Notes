---
title: >-
  [论文解读] CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals
description: >-
  [NeurIPS 2025][人体理解][肌电信号] 提出 CPEP 框架，通过对比学习将低质量 EMG 信号表征与高质量手部姿态表征对齐，使 EMG 编码器获得姿态感知能力，首次实现从 EMG 信号零样本识别未见手势，分布内手势分类提升 21%、未见手势分类提升 72%。 领域现状：基于视觉的手势识别已非常成熟…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "肌电信号"
  - "EMG"
  - "手势识别"
  - "对比学习"
  - "零样本分类"
  - "跨模态对齐"
---

# CPEP: Contrastive Pose-EMG Pre-training Enhances Gesture Generalization on EMG Signals

**会议**: NeurIPS 2025  
**arXiv**: [2509.04699](https://arxiv.org/abs/2509.04699)  
**代码**: 无  
**领域**: 人体理解 / 手势识别  
**关键词**: 肌电信号, EMG, 手势识别, 对比学习, 零样本分类, 跨模态对齐

## 一句话总结
提出 CPEP 框架，通过对比学习将低质量 EMG 信号表征与高质量手部姿态表征对齐，使 EMG 编码器获得姿态感知能力，首次实现从 EMG 信号零样本识别未见手势，分布内手势分类提升 21%、未见手势分类提升 72%。

## 研究背景与动机

**领域现状**：基于视觉的手势识别已非常成熟，但在可穿戴设备上受限于功耗和隐私。表面肌电信号 (sEMG) 低功耗、易集成，适合可穿戴实时手势预测。

**现有痛点**：(a) EMG 信号信噪比低、变异性大，传统自监督预训练效果不佳；(b) 监督方法（如 emg2pose 姿态回归）泛化能力差，无法识别未见手势或新用户；(c) 大规模 EMG 数据采集困难成本高。

**核心矛盾**：EMG 是"弱模态"，单独学习难以产生高质量表征；手部姿态是"强模态"，含丰富结构和语义信息。如何利用强模态先验提升弱模态表征？

**本文目标** 让 EMG 编码器习得"姿态感知"表征，在嵌入空间通过检索姿态实现零样本分类。

**切入角度**：借鉴 CLIP 跨模态对比预训练，但针对 EMG-姿态特殊性设计——预训练单模态编码器减少配对数据需求，冻结强模态编码器只训练弱模态。

**核心 idea**：用对比学习将 EMG 表征拉向配对姿态表征，无需任务特定训练即可零样本手势识别。

## 方法详解

### 整体框架
CPEP 分三阶段：(1) MAE 自监督预训练 EMG 和姿态编码器；(2) 对比预训练冻结姿态编码器，InfoNCE 对齐两模态 [CLS] 表征；(3) 下游评估通过线性探测或零样本最近邻检索。

### 关键设计

1. **单模态编码器预训练 (MAE)**:

    - 功能：分别为 EMG 和姿态预训练 Transformer 编码器
    - 核心思路：标准 MAE，沿时间分 patch，mask 比率 $r=50\%$，仅编码未 mask token，解码器重建全序列。$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \|\psi(\phi(\{\mathbf{z}_j\}_{j\notin\mathcal{M}}))_i - \mathbf{z}_i\|_2^2$
    - 设计动机：先学好各自基本表征，减少对比阶段配对数据需求

2. **对比姿态-EMG 预训练 (CPEP)**:

    - 功能：冻结姿态编码器 $\mathcal{E}_p$，训练 EMG 编码器 + 投影头 $h$
    - 核心思路：EMG 嵌入 $u_i = h(\mathcal{E}_x(x_i))_{[\text{CLS}]}$，姿态嵌入 $v_i = (\mathcal{E}_p(p_i))_{[\text{CLS}]}$，$\ell_2$ 归一化后对称 InfoNCE：$\mathcal{L}_{\text{CPEP}} = \frac{1}{2N}\sum_{i} [-\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})} - \log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ji})}]$，$s_{ij} = \tilde{u}_i^\top\tilde{v}_j / \tau$
    - 设计动机：冻结姿态编码器是关键——双方同时更新会破坏姿态表征质量（实验证实无法收敛）

3. **零样本分类协议**:

    - 功能：嵌入空间 k-最近邻投票
    - 核心思路：预计算姿态嵌入，每个 EMG 查询检索 top-$k$（$k=10$）最近姿态，多数投票：$\hat{y}_j = \text{mode}\{y(p) | p \in \mathcal{R}_j\}$
    - 设计动机：零样本能力验证 EMG 表征学会了姿态结构信息

### 损失函数 / 训练策略

三阶段训练：EMG/Pose-MAE 各 100 epoch -> CPEP 对比 100 epoch（batch=256, 可学习 $\tau$ 初始 0.02）-> 线性探测。4x V100，训练约 4.5 小时/模型。

## 实验关键数据

### 主实验（手势分类准确率）

| 方法 | LP 分布内 | LP 未见 | ZS 分布内 | ZS 未见 |
|------|----------|--------|----------|--------|
| emg2pose (基准) | 0.647 | 0.312 | - | - |
| EMG-MAE | ~0.55 | ~0.30 | - | - |
| PoseT (监督) | ~0.60 | ~0.35 | - | - |
| **CPEP** | **0.782** | **0.536** | **0.757** | **0.481** |
| Pose-MAE (上界) | ~0.85 | ~0.65 | - | - |

### 消融实验

| 配置 | LP 分布内 | ZS 分布内 | LP 未见 | ZS 未见 |
|------|----------|----------|--------|--------|
| EMG encoder Frozen | 0.372 | 0.344 | 0.326 | 0.298 |
| EMG encoder RandInit | 0.748 | 0.701 | 0.479 | 0.454 |
| AvgPool | 0.761 | 0.711 | 0.518 | 0.454 |
| **CPEP (full)** | **0.782** | **0.757** | **0.536** | **0.481** |

### 关键发现
- MAE 预训练初始化至关重要：随机初始化收敛慢精度低，双编码器同时训练无法收敛
- [CLS] 比 AvgPool 更好，全局上下文对手势识别更有效
- 冻结 EMG 编码器效果极差（0.372 vs 0.782），必须微调
- EMG patch 越长性能越差，需细粒度时序建模

## 亮点与洞察
- **首个 EMG 零样本手势识别框架**：零样本超过基准线性探测（0.481 vs 0.312 未见手势），对比预训练学到的表征有真正泛化能力
- **强模态锚定弱模态**范式可迁移到 IMU-视频对齐、EEG-行为对齐等场景

## 局限与展望
- 仅 emg2pose 单一数据集验证，泛化到其他 EMG 采集设备和协议未知
- 手势类别少（4+4 类），实际应用需要数十甚至上百种手势
- 未与 SigLIP、CLAP 等进阶对比学习方法比较
- Workshop 论文，实验规模有限，统计显著性未报告
- 未探索在线适应或少样本微调场景
- 对不同用户间 EMG 信号差异的鲁棒性分析不足

## 相关工作与启发
- **vs emg2pose**: 监督姿态回归泛化有限；CPEP 对比对齐获得结构化嵌入支持零样本
- **vs CLIP**: 借鉴跨模态对比思路但做了关键适配——预训练编码器减少数据需求、冻结强模态
- **vs NeuroPose/Vemg2pose**: 这些基准模型也用了 Transformer 架构但以监督回归为目标，嵌入质量不足以支持检索式分类

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 CLIP 式对比预训练应用于 EMG-姿态对齐实现零样本
- 实验充分度: ⭐⭐⭐ Workshop 论文，单数据集，手势类别少
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述简洁
- 价值: ⭐⭐⭐⭐ 开创 EMG 零样本手势识别新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[CVPR 2025\] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](../../CVPR2025/human_understanding/posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](raptr_radar-based_3d_pose_estimation_using_transformer.md)

</div>

<!-- RELATED:END -->
