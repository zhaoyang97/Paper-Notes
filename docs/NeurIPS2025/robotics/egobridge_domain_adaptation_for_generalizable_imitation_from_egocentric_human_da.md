---
title: >-
  [论文解读] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data
description: >-
  [NeurIPS 2025 Oral][机器人][cross-embodiment] 提出 EgoBridge 框架，利用最优传输(OT)在策略潜在空间中对齐人类和机器人数据的联合分布（特征+动作），结合动态时间规整(DTW)构建伪配对，实现从第一人称人类数据到机器人的跨具身知识迁移，在真实世界任务中绝对成功率提升达 44%。
tags:
  - "NeurIPS 2025 Oral"
  - "机器人"
  - "cross-embodiment"
  - "域适应"
  - "最优传输"
  - "egocentric"
  - "模仿学习"
---

# EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2509.19626](https://arxiv.org/abs/2509.19626)  
**代码**: [ego-bridge.github.io](https://ego-bridge.github.io/)  
**领域**: 强化学习  
**关键词**: cross-embodiment, 域适应, 最优传输, egocentric, 模仿学习

## 一句话总结
提出 EgoBridge 框架，利用最优传输(OT)在策略潜在空间中对齐人类和机器人数据的联合分布（特征+动作），结合动态时间规整(DTW)构建伪配对，实现从第一人称人类数据到机器人的跨具身知识迁移，在真实世界任务中绝对成功率提升达 44%。

## 研究背景与动机

**领域现状**：行为克隆(BC)结合大规模遥操作数据在机器人操作中取得显著进展。但遥操作数据收集成本高、难以覆盖多样场景。穿戴设备（如 AR 眼镜）可低成本采集大量第一人称人类操作数据，包含观测和动作信息。

**现有痛点**：人类数据和机器人数据之间存在多重域差距——(a) 视觉外观差异（人手 vs 机械臂）；(b) 运动学差异（同一动作空间下行为分布不同）；(c) 传感模态差异（机器人有腕部相机，人类没有）。直接混合训练(co-training)不能自动产生有效迁移。

**核心矛盾**：简单 co-training 假设共享潜在空间自然涌现，但实际上人类和机器人的潜在特征形成分离的聚类（latent covariate shift），$\mu_H \neq \mu_R$，导致从人类数据学到的行为无法迁移到机器人。

**本文目标**：(a) 显式对齐人类和机器人域的潜在表示；(b) 在对齐过程中保留动作相关信息；(c) 使机器人能执行仅在人类数据中出现的新行为。

**切入角度**：将跨具身学习形式化为域适应问题。不同于全局分布对齐（如对抗训练、MMD），利用 OT 的几何结构保留局部动作对应关系。

**核心 idea**：用 DTW 引导 OT 的代价函数，在对齐潜在特征时自动发现行为相似的人类-机器人伪配对，实现动作感知的联合分布对齐。

## 方法详解

### 整体框架
EgoBridge 是一个 co-training 框架。编码器 $f_\phi$ 将人类/机器人观测映射到共享潜在空间 $\mathcal{Z}$，Transformer 解码器 $\pi_\theta$ 从潜在特征生成动作。总损失为 $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) + \alpha\mathcal{L}_{\text{OT-joint}}$。BC 损失端到端优化整个网络，OT 损失仅优化编码器。

### 关键设计

1. **联合分布最优传输 (Joint OT)**:

    - 功能：对齐人类和机器人数据在潜在空间中的联合分布 $P(f_\phi(O), A)$
    - 核心思路：给定人类样本 $\{(o_i^H, a_i^H)\}$ 和机器人样本 $\{(o_j^R, a_j^R)\}$，用 Sinkhorn 算法求解带熵正则化的最优传输计划 $T_\epsilon^*$：$\mathcal{L}_{\text{OT-joint}} = \sum_{i,j}(T_\epsilon^*)_{ij} \cdot \mathcal{C}((f_\phi(o_i^H), a_i^H), (f_\phi(o_j^R), a_j^R))$
    - 设计动机：与仅对齐边际分布 $P(f_\phi(O))$ 的标准域适应不同，联合对齐同时考虑特征和动作，防止对齐破坏动作相关信息。梯度鼓励编码器将行为相似的跨域样本映射到附近
    - 与标准 OT 的区别：标准 OT 用欧氏距离作为代价函数可能将外观相似但行为不同的样本配对，联合 OT 确保配对的同时行为也相似

2. **DTW 引导的代价函数设计**:

    - 功能：用动态时间规整识别行为相似的跨域伪配对
    - 核心思路：对 mini-batch 中的每对人类-机器人动作轨迹计算 DTW 距离 $\text{DTW}(\mathbf{a}^H, \mathbf{a}^R) = \min_\pi \sum_{(i,j)\in\pi}\|a_i^H - a_j^R\|^2$。找到每个机器人样本的最佳人类匹配 $i^*(j) = \arg\min_i A_{ij}$，然后修改代价矩阵：
    $\tilde{C}_{ij} = \begin{cases} D_{ij} \cdot \lambda & \text{if } i = i^*(j) \\ D_{ij} & \text{otherwise} \end{cases}$
   其中 $D_{ij} = \|f_\phi(o_i^H) - f_\phi(o_j^R)\|^2$，$\lambda \ll 1$ 大幅降低伪配对的传输代价
    - 设计动机：(a) DTW 天然处理时间对齐差异（人类执行通常比遥操作快 2-3 倍）；(b) 使用"软监督"——不直接用 DTW 距离作为损失，而是用它识别配对后降低 OT 中的传输代价；(c) 比 MSE 配对更鲁棒（消融实验证实）

3. **共享策略架构**:

    - 功能：统一处理人类和机器人两种数据源
    - 核心思路：编码器 $f_\phi$ 包含模态特定 stem（共享 vision stem 处理第一人称 RGB，独立 stem 处理机器人腕部相机）和共享 Transformer encoder trunk。解码器 $\pi_\theta$ 为多层 Transformer decoder，通过交替 self/cross-attention 生成动作。$M$ 个可学习 context token 用于计算 OT 损失
    - 设计动机：共享 vision stem 强制视觉对齐；分离腕部相机 stem 因为人类数据没有对应模态；DETR 风格架构支持灵活的多模态输入

4. **数据收集系统**:

    - 人类数据：Meta Project Aria 智能眼镜，采集第一人称 RGB 和双手 SE(3) 笛卡尔位姿
    - 机器人数据：Eve 机器人 + 同款 Aria 眼镜模拟人类手眼配置，消除相机设备差异
    - 动作空间统一：双臂末端执行器 SE(3) 位姿 + 轨迹 chunk

### 损失函数 / 训练策略
$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{BC-cotrain}}(\phi,\theta) + \alpha\mathcal{L}_{\text{OT-joint}}(\phi)$。BC 损失对人类+机器人数据均匀采样，OT 损失仅更新编码器参数。动作和本体感觉做具身特定的高斯归一化。

## 实验关键数据

### 真实世界主实验

| 方法 | Scoop Coffee In-Dist. | Scoop Obj. Gen. | Scoop Scene+Obj | Drawer (SR) | Drawer Beh. Gen. | Laundry (SR) |
|------|------|------|------|------|------|------|
| Robot-only BC | 33% | 40% | 7% | 9% | 0% | 28% |
| Co-train | 53% | 46% | 0% | 22% | 0% | 33% |
| EgoMimic | 60% | 53% | 0% | 14% | 0% | 33% |
| MimicPlay | 33% | 27% | 0% | 14% | 0% | 28% |
| ATM | 47% | 33% | 0% | 6% | 8% | 28% |
| **EgoBridge** | **67%** | **60%** | **27%** | **47%** | **33%** | **72%** |

### 消融实验 (Drawer 任务)

| 配置 | Drawer SR | Beh. Gen. SR | 说明 |
|------|----------|-------------|------|
| **EgoBridge (full)** | **47%** | **33%** | 完整模型 |
| MSE 替代 DTW | 14% | 17% | DTW 配对是关键，去掉后掉 33% |
| Standard OT (边际对齐) | 33% | 17% | 联合 OT 优于边际 OT |
| Co-train (无对齐) | 22% | 0% | 无对齐完全无法行为泛化 |

### 关键发现
- **DTW 配对贡献最大**：将 DTW 替换为 MSE 后性能暴跌（47%→14%），说明时间对齐和运动学鲁棒的配对是核心
- **联合 OT 优于边际 OT**：标准 OT 仅对齐特征边际分布但忽略动作对应关系，泛化能力显著下降
- **行为泛化是独特能力**：其他所有基线在仅人类数据覆盖的新抽屉位置上完全失败(0%)，唯独 EgoBridge 实现 33% 成功率
- **潜在空间可视化**：t-SNE 显示 EgoBridge 的人类-机器人特征重叠度最高（Wasserstein-2 距离最小），KNN 配对语义最相似

## 亮点与洞察
- **OT + DTW 的组合设计**：OT 提供可微的分布对齐框架，DTW 提供时间鲁棒的行为匹配度量。两者组合使得对齐是"行为感知"的——这个 idea 可迁移到任何跨域模仿学习场景
- **从人类学新行为**：这是最有价值的贡献。大多数方法只能增强机器人已有行为的鲁棒性，EgoBridge 能让机器人执行从未遥操作过的新行为，真正实现了人类数据的"增值"
- **软监督优于硬约束**：DTW 不直接作为损失函数，而是通过降低 OT 代价来引导对齐。这种软监督方式更鲁棒，避免了 DTW 本身的噪声影响

## 局限与展望
- **单任务 DTW**：DTW 基于动作轨迹距离，在多任务联合训练时可能无法区分不同任务中相似的局部运动。作者提到未来可用 VLM 的语言嵌入距离替代
- **仍需少量机器人数据**：不是纯粹的人类到机器人迁移，需要目标域的种子数据。能否进一步减少/消除？
- **SE(3) 动作空间假设**：要求人类和机器人共享末端执行器位姿空间，对灵巧操作等不同运动学的任务可能受限
- **评估规模**：每个任务的测试 rollout 数量较少（15-48 次），统计置信度有限

## 相关工作与启发
- **vs EgoMimic**：EgoMimic 使用视觉遮挡和数据归一化等启发式方法桥接域差距，缺乏显式对齐。EgoBridge 用 OT 显式对齐，在所有任务上表现更好
- **vs MimicPlay**：MimicPlay 使用层次策略（高层规划co-train + 低层解码fine-tune），用 KL 散度对齐边际分布。EgoBridge 的联合 OT 对齐更保留动作信息
- **vs ATM**：ATM 通过点轨迹追踪提取运动信息然后冻结高层训练低层。这种两阶段方法可能丢失细粒度对应关系

## 评分
- 新颖性: ⭐⭐⭐⭐ OT + DTW 的组合在跨具身模仿学习中是新颖的，联合分布对齐的形式化也很清晰
- 实验充分度: ⭐⭐⭐⭐ 三个真实世界任务（含双臂），仿真消融，潜在空间可视化，充分验证假设
- 写作质量: ⭐⭐⭐⭐ 问题形式化清晰，方法动机充分，实验条理分明
- 价值: ⭐⭐⭐⭐⭐ 行为泛化能力是真正的突破——让人类数据不再只是"锦上添花"而是真的能教机器人新技能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI Coordination](inner_speech_as_behavior_guides_steerable_imitation_of_diverse_behaviors_for_hum.md)
- [\[CVPR 2026\] NIL: No-data Imitation Learning](../../CVPR2026/robotics/nil_no-data_imitation_learning.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[NeurIPS 2025\] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)

</div>

<!-- RELATED:END -->
