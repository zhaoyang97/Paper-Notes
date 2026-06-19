---
title: >-
  [论文解读] Diffusion Model is a Good Pose Estimator from 3D RF-Vision
description: >-
  [ECCV 2024][人体理解][人体姿态估计] 提出mmDiff，一种基于扩散模型的毫米波雷达人体姿态估计框架，通过全局-局部雷达上下文提取和结构-运动一致性约束，有效应对雷达点云稀疏、噪声大和信号不一致的挑战，显著超越现有SOTA。 领域现状：人体姿态估计（HPE）是计算机视觉的核心任务，主流方案依赖RGB(D)相机…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "人体姿态估计"
  - "毫米波雷达"
  - "扩散模型"
  - "点云"
  - "RF-Vision"
---

# Diffusion Model is a Good Pose Estimator from 3D RF-Vision

**会议**: ECCV 2024  
**arXiv**: [2403.16198](https://arxiv.org/abs/2403.16198)  
**代码**: [https://fanjunqiao.github.io/mmDiff-site/](https://fanjunqiao.github.io/mmDiff-site/)  
**领域**: 人体理解  
**关键词**: 人体姿态估计, 毫米波雷达, 扩散模型, 点云, RF-Vision

## 一句话总结

提出mmDiff，一种基于扩散模型的毫米波雷达人体姿态估计框架，通过全局-局部雷达上下文提取和结构-运动一致性约束，有效应对雷达点云稀疏、噪声大和信号不一致的挑战，显著超越现有SOTA。

## 研究背景与动机

**领域现状**：人体姿态估计（HPE）是计算机视觉的核心任务，主流方案依赖RGB(D)相机。毫米波雷达因能穿透障碍物、保护隐私、不受光照影响等优势，正成为HPE的新兴传感器。现有毫米波HPE方法主要通过多帧聚合增强点云密度，并直接借用视觉领域的LSTM或Transformer架构。

**现有痛点**：(1) 毫米波雷达空间分辨率有限，点云极度稀疏（通常<100个点），几何信息不足；(2) 多径效应导致幻影点，人体部分会被漏检（miss-detection）；(3) 信号镜面反射和环境干扰导致感知数据时间不一致，引起姿态抖动和尺寸变化。

**核心矛盾**：视觉领域的Transformer/LSTM特征编码器是为稠密、一致的视觉数据设计的，无法有效处理毫米波雷达的稀疏性和不一致性。直接使用MLP投影PC特征到关节特征时，漏检关节的噪声会污染整体特征。

**本文目标** 设计专门针对噪声雷达点云的HPE方法，同时解决(1)部分人体漏检导致的特征提取不可靠和(2)信号不一致导致的姿态不稳定两大问题。

**切入角度**：扩散模型天然擅长去噪，可从噪声分布中恢复目标分布。漏检的关节可通过已检测关节的信息推断，扭曲的人体结构可被逐步矫正。

**核心 idea**：用条件扩散模型进行姿态生成，通过四个专用模块从噪声雷达数据中提取干净、一致的条件特征（全局/局部雷达上下文 + 肢体长度/时序运动一致性），引导扩散过程生成准确且稳定的人体姿态。

## 方法详解

### 整体框架

mmDiff采用两阶段训练的条件扩散模型：
- **阶段一**：训练点云编码器 + GRC模块 + 粗姿态解码器，获得全局关节特征 $F^j$ 和粗估计姿态 $\tilde{H}$
- **阶段二**：冻结阶段一参数，训练LRC、SLC、TMC三个条件模块和扩散模型。粗姿态 $\tilde{H}$ 用于初始化扩散逆过程的起点 $\hat{H}_K$

输入为6维雷达点云 $R_t \in \mathbb{R}^{N \times 6}$（包含x,y,z坐标和速度、能量、幅度），输出为17个关节的3D坐标 $H_t \in \mathbb{R}^{17 \times 3}$。

### 关键设计

1. **条件扩散模型**:

    - 功能：以雷达特征为条件，从噪声姿态中逐步去噪生成最终姿态
    - 核心思路：前向过程逐步加噪 $q(H_k | H_{k-1}) = \mathcal{N}(H_k | \sqrt{1-\beta_k} H_{k-1}, \beta_k I)$；逆过程用条件噪声估计器 $\hat{\varepsilon}_k = \hat{\varepsilon}_\theta(\hat{H}_k, C, k)$ 预测并移除噪声。骨架用GCN编码为 $Z_k \in \mathbb{R}^{17 \times 96}$ 的潜在表示，条件特征在每个GCN block前通过加法注入。
    - 设计动机：相比直接回归，扩散模型可建模姿态的概率分布，对漏检关节可从已检测关节推断，对异常姿态可逐步矫正。

2. **全局雷达上下文 (GRC)**:

    - 功能：从整体点云特征中提取关节级别的鲁棒特征，应对漏检问题
    - 核心思路：借鉴ViT的cls token设计，随机初始化可训练的关节特征模板 $\bar{F}^j \in \mathbb{R}^{17 \times 1024}$，与PC特征拼接送入Global-Transformer：$F^j, F'^r = \Phi^g(\bar{F}^j, F^r)$。每个关节独立地从PC特征中提取与自身相关的信息。
    - 设计动机：传统MLP投影无法隔离漏检关节的影响；Transformer的注意力机制使每个关节token可选择性关注相关的点云区域，漏检部分不会污染已检测关节的特征。

3. **局部雷达上下文 (LRC)**:

    - 功能：在关节邻域提取高分辨率的点级特征
    - 核心思路：使用扩散过程中间状态的姿态 $\hat{H}_k$ 作为**动态锚点**，KNN选取每个关节附近 $\bar{K}=50$ 个最近邻点，通过局部Transformer进行点对点自注意力：$C^{loc} = \bigcup_{i} \Phi^l \circ g^l(\bar{R}_k^i)$
    - 设计动机：全局特征分辨率不足，局部点云可提供更精细的信息。使用动态锚点（而非静态粗估计锚点）可在不同扩散步骤反映关节的误差变化。

4. **结构肢体长度一致性 (SLC)**:

    - 功能：提取人体16段肢体长度 $\hat{L} \in \mathbb{R}^{16}$ 作为结构约束
    - 核心思路：从关节特征 $F^j$ 用MLP解码肢体长度，再投影到条件嵌入空间：$C^{lim} = g_2^{lim}(g_1^{lim}(F^j))$。肢体长度loss确保准确估计。
    - 设计动机：同一人的肢体长度应保持恒定，这一物理约束可减少姿态尺寸变化和结构扭曲。

5. **时序运动一致性 (TMC)**:

    - 功能：从历史 $\Delta t = 8$ 帧的粗估计姿态中提取运动模式
    - 核心思路：用共享GCN编码器将历史姿态序列编码为特征序列，再用1D卷积沿时间维度提取运动信息：$C^{tem} = g_2^{tem}(\bigcup_{i} g_1^{tem}(\tilde{H}_{t-i}))$
    - 设计动机：人类运动是连续平滑的，1D卷积可通过对历史帧特征平均来提取平滑运动模式，同时可捕获动作趋势（如"举手"时手部z值递增），避免突变帧和姿态抖动。

### 损失函数 / 训练策略

- **阶段一**：$\mathcal{L}_{joint} = E_{i \sim [1,17]} \|h^i - \tilde{h}^i\|_2^2$（关节回归L2损失）
- **阶段二**：$\mathcal{L}_{diff} = \mathbb{E}_{k,\varepsilon_k} \|\varepsilon_k - \hat{\varepsilon}_\theta(H_k, k, C)\|_2^2 + \lambda \cdot \mathbb{E}_{i \sim [1,16]} |l^i - \hat{l}^i|_1$
    - 扩散去噪损失 + 肢体长度L1回归损失（$\lambda = 5$）
- 训练100个epoch，batch size 1024，Adam优化器，学习率 $2 \times 10^{-5}$
- 扩散步数 $K = 25$，常数噪声调度 $\beta = 0.001$
- 推理时取5个假设的平均

## 实验关键数据

### 主实验（mmBody数据集，MPJPE/PA-MPJPE mm）

| 方法 | 基础场景平均 | 恶劣环境平均 | 总平均 |
|------|-------------|-------------|--------|
| P4Transformer | ~72/~59 | ~82/~62 | 78.10/60.56 |
| PoseFormer | ~68/~55 | ~76/~57 | 73.52/56.07 |
| DiffPose | ~68/~56 | ~77/~59 | 73.31/58.23 |
| **mmDiff(G,L,T,S)** | ~**65**/**49** | ~**70**/**53** | **68.08/53.71** |

mmDiff相比P4Transformer：MPJPE降低12.8%，PA-MPJPE降低11.3%；恶劣环境下降幅更大（14.7%/12.0%）。

### mm-Fi数据集

| 方法 | Random MPJPE | Cross-Subject MPJPE | Cross-Env MPJPE |
|------|-------------|---------------------|-----------------|
| PointTransformer | 73.09 | 75.96 | 88.28 |
| DiffPose | 73.44 | 70.31 | 86.35 |
| **mmDiff(G,S,T)** | **65.26** | **65.62** | **82.73** |

### 消融实验

| 配置 | 平均MPJPE | 平均PA-MPJPE | 说明 |
|------|----------|-------------|------|
| 无扩散（P4Trans基线） | 78.10 | 60.56 | 基线 |
| + 扩散（无条件） | 73.31 | 58.23 | 扩散本身可建模姿态分布 |
| + GRC | 70.99 | 55.80 | 全局上下文显著提升 |
| + GRC + LRC | 69.79 | 55.04 | 局部细节进一步改善 |
| + GRC + LRC + TMC | 69.16 | 53.96 | 时序约束减少抖动 |
| + GRC + LRC + TMC + SLC | **68.08** | **53.71** | 完整模型最优 |
| 去掉SLC | 69.45 | 54.53 | 肢体长度约束重要 |
| 去掉TMC | 69.16 | 53.96 | 时序一致性对稳定性关键 |

### 模型效率

| 模块 | 延迟 | 参数量 | GFLOPs |
|------|------|--------|--------|
| P4Transformer（基线） | 40.48ms | 128M | 43.50 |
| 扩散模型 + 全部条件模块 | 11.85ms | 18.51M | 0.62 |

### 关键发现

- 每个条件模块都有独立贡献，SLC和TMC的消除影响最大
- mmDiff在恶劣环境（烟雾、黑暗、遮挡）下的优势更显著
- 动态锚点比静态锚点的LRC效果更好（MPJPE 70.43 vs 71.05）
- 关节特征模板比直接PC特征引导效果更好（MPJPE 70.99 vs 72.50）
- 条件模块计算开销极小（延迟<2ms/模块），适合边缘部署

## 亮点与洞察

- **扩散模型首次用于毫米波雷达HPE**：巧妙地将雷达噪声去除类比为扩散去噪过程，构建直觉合理的方法论框架
- **四模块各有明确分工**：GRC处理漏检（关节隔离特征提取）、LRC提升分辨率（局部点级注意力）、SLC保持结构一致（肢体长度约束）、TMC保持运动平滑（时序特征融合），设计逻辑清晰
- **高效的模块设计**：四个条件模块总计仅18.5M参数和0.62 GFLOPs，比基线还小，具有实际部署价值
- **跨域泛化能力**：cross-subject实验表现接近random splitting，说明学到的人体结构和运动模式可泛化到未见主体

## 局限与展望

- 在多目标场景下性能下降，因为雷达点云中多人信号会互相干扰
- 相比RGB-D方法在标准场景下仍有差距（但在恶劣环境下反超）
- mm-Fi数据集因点云太稀疏（N<100）无法使用LRC模块
- TMC模块依赖历史帧的粗估计质量，如果前几帧都很差，可能形成误差累积
- 扩散推理需要25步迭代，虽然每步很快但总延迟比直接回归方法高

## 相关工作与启发

- **vs P4Transformer**: 毫米波HPE基线方法，直接用PC编码器+Transformer，无法有效处理噪声和漏检，mmDiff在MPJPE上降低12.8%
- **vs DiffPose**: 基于RGB 2D姿态到3D姿态的扩散方法，直接用于雷达效果不佳，因为没有针对雷达噪声特性的条件设计
- **vs PoseFormer**: 时空Transformer方法，从粗姿态做refinement，但对雷达的信号不一致性缺乏专门处理

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个将扩散模型应用于毫米波雷达HPE的工作，四个条件模块设计针对性强
- 实验充分度: ⭐⭐⭐⭐⭐ 两个公开数据集、完整消融、效率分析、稳定性分析（AKV）、肢体长度分布统计，非常充分
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述完整，图表精美
- 价值: ⭐⭐⭐⭐ 为隐私保护的非视觉HPE提供了强基线，在恶劣环境下的表现证明了RF-vision的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] SCAPE: A Simple and Strong Category-Agnostic Pose Estimator](scape_a_simple_and_strong_category-agnostic_pose_estimator.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] 3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
