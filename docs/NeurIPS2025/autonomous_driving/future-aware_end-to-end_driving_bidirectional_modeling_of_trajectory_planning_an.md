---
title: >-
  [论文解读] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution
description: >-
  [NEURIPS2025][自动驾驶][end-to-end driving] 提出 SeerDrive，通过双向建模场景演化与轨迹规划（未来感知规划 + 迭代交互），在 NAVSIM 和 nuScenes 上取得 SOTA。 现有端到端自动驾驶方法大多采用"一步式"范式：仅基于当前帧的传感器观测直接预测未来数秒的轨迹…
tags:
  - "NEURIPS2025"
  - "自动驾驶"
  - "end-to-end driving"
  - "world model"
  - "BEV"
  - "trajectory planning"
  - "iterative refinement"
---

# Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution

**会议**: NEURIPS2025  
**arXiv**: [2510.11092](https://arxiv.org/abs/2510.11092)  
**代码**: [LogosRoboticsGroup/SeerDrive](https://github.com/LogosRoboticsGroup/SeerDrive)  
**领域**: 自动驾驶  
**关键词**: end-to-end driving, world model, BEV, trajectory planning, iterative refinement

## 一句话总结
提出 SeerDrive，通过双向建模场景演化与轨迹规划（未来感知规划 + 迭代交互），在 NAVSIM 和 nuScenes 上取得 SOTA。

## 背景与动机
现有端到端自动驾驶方法大多采用"一步式"范式：仅基于当前帧的传感器观测直接预测未来数秒的轨迹。这种做法存在两个核心局限：

1. **忽视场景动态演化**：当前帧快照无法充分反映交通场景未来的变化（如前方车辆减速、行人横穿），导致规划缺乏前瞻性。
2. **忽略双向耦合关系**：自车的未来行为会反过来影响周围场景的演化（如变道后后车的反应），但这种双向依赖在已有方法中鲜被显式建模。

作者从 world model 的研究趋势中获得灵感——如果能预测未来场景并将其与规划过程深度耦合，就能实现更具适应性的决策。

## 核心问题
如何在端到端驾驶框架中显式建模**未来场景演化**与**轨迹规划**之间的双向关系，使规划器既能利用对未来场景的预见，又能将自车的规划意图反馈给场景预测模型？

## 方法详解

### 整体框架
SeerDrive 包含两个核心模块在闭环中迭代协作：
- **BEV World Modeling Network**：预测未来 BEV 语义地图
- **End-to-End Planning Network**：基于当前与未来 BEV 特征生成轨迹

### 1. 特征编码
- 多视角图像与 LiDAR 经 TransFuser 融合为当前 BEV 特征 $F_{\rm bev}^{\rm curr} \in \mathbb{R}^{H \times W \times C}$
- 锚定多模态轨迹与自车状态经 MLP 编码为当前自车特征 $F_{\rm ego}^{\rm curr} \in \mathbb{R}^{M \times C}$（$M$ 为轨迹模态数）
- 轻量 BEV 解码器生成当前 BEV 语义地图 $\mathcal{B}_{\rm curr}$ 用于监督

### 2. 未来 BEV 世界建模
- 将 $F_{\rm bev}^{\rm curr}$ 展平并沿模态维度重复，与 $F_{\rm ego}^{\rm curr}$ 拼接得到场景特征 $F_{\rm scene}^{\rm curr}$
- 经 Transformer Encoder（即 BEV World Model）预测未来场景特征 $F_{\rm scene}^{\rm fut}$
- 从中提取未来 BEV 特征 $F_{\rm bev}^{\rm fut}$，再经 BEV 解码器生成未来语义地图 $\mathcal{B}_{\rm fut}$ 用于监督
- **仅预测最终规划步的 BEV**（如 4 秒后），而非中间帧序列——消融实验表明这已足够且更高效

### 3. 未来感知规划（Future-Aware Planning）
核心挑战：如何让规划器同时利用当前和未来 BEV 特征而不产生表征纠缠。解决方案是**解耦策略**：

- **当前分支**：$F_{\rm ego}^{\rm curr}$ 通过 Transformer Decoder 与 $F_{\rm bev}^{\rm curr}$ 交互 → MLP 解码得轨迹 $\mathcal{T}_a$
- **未来分支**：用锚定轨迹的终点初始化未来自车特征 $F_{\rm ego}^{\rm fut}$，通过 Transformer Decoder 与 $F_{\rm bev}^{\rm fut}$ 交互 → MLP 解码得轨迹 $\mathcal{T}_b$
- **融合**：采用 Motion-aware Layer Normalization (MLN) 将 $F_{\rm ego}^{\rm fut}$ 注入 $F_{\rm ego}^{\rm curr}$，得到未来感知的自车表征 → 解码最终轨迹 $\mathcal{T}_{\rm final}$

### 4. 迭代场景建模与车辆规划（Iterative Scene Modeling and Vehicle Planning）
- 将规划网络输出的更新后自车特征 $F_{\rm ego}^{\rm curr}$ 反馈给 BEV World Model，生成更新的未来 BEV
- 如此迭代 $N$ 次（实验中 $N=2$ 为最佳），每次产生一组语义地图和轨迹，均参与训练监督
- 这一设计体现了双向耦合的核心思想：场景预测指导规划，规划结果又反过来修正场景预测

### 5. 端到端训练
总损失 = BEV 语义地图损失 + 轨迹规划损失，涵盖所有迭代轮次的输出。NAVSIM 上权重设置为 $\lambda_1=10, \lambda_2=0.1, \lambda_3=1$。

## 实验关键数据

### NAVSIM（navtest，闭环评估）

| 方法 | PDMS ↑ | NC ↑ | DAC ↑ | EP ↑ |
|------|--------|------|-------|------|
| TransFuser | 84.0 | 97.7 | 92.8 | 79.2 |
| DiffusionDrive | 88.1 | 98.2 | 96.2 | 82.2 |
| WoTE | 88.3 | 98.5 | 96.8 | 81.9 |
| Hydra-NeXt | 88.6 | 98.1 | 97.7 | 81.8 |
| **SeerDrive** | **88.9** | 98.4 | 97.0 | **83.2** |
| SeerDrive (V2-99) | **90.7** | 98.8 | 98.6 | 84.2 |

### nuScenes（开环，L2 位移误差 / 碰撞率）

| 方法 | Avg L2 ↓ | Avg Col. ↓ |
|------|----------|------------|
| SparseDrive | 0.61 | 0.08 |
| BridgeAD | 0.59 | 0.09 |
| MomAD | 0.60 | 0.09 |
| **SeerDrive** | **0.43** | **0.06** |

### 关键消融结果（NAVSIM PDMS）
- 去掉 Future-Aware Planning 和 Iterative：87.1（-1.8）
- 仅去掉 Future BEV 注入：87.9（-1.0）
- 仅去掉迭代：88.1（-0.8）
- 完整 SeerDrive：**88.9**
- 迭代次数：1 次 → 88.1，2 次 → **88.9**，3 次 → 88.7
- MLN 优于 Concat（88.3）和 Add（88.5）

## 亮点
1. **范式创新**：首次在端到端驾驶中显式建模场景演化与轨迹规划的双向闭环交互，超越传统一步式范式
2. **解耦设计精巧**：当前/未来 BEV 特征分别与自车特征独立交互后再通过 MLN 融合，避免表征纠缠——消融证明直接联合学习反而掉点
3. **迭代收敛快**：仅需 2 次迭代即可达到最佳，3 次反而略降，说明设计紧凑高效
4. **训练代价低**：8 张 RTX 3090 仅需 ~5 小时（NAVSIM），具有较好的可复现性
5. **仅预测终帧 BEV**：消融表明预测中间帧序列（1s-2s-3s-4s）并不比仅预测最终帧更好，设计简洁高效

## 局限与展望
1. **仅在非反应式/开环评估**：NAVSIM 是非反应式仿真，nuScenes 是开环回放，缺少 CARLA 等全闭环验证
2. **BEV 语义地图的表达能力有限**：仅预测 BEV 语义地图，无法建模 3D 高度信息和遮挡关系
3. **迭代次数有上限**：3 次迭代时性能已开始下降，说明当前迭代机制可能存在信息退化问题
4. **未来 BEV 仅预测终帧**：虽然消融显示中间帧收益不大，但可能是因为融合方式（简单拼接）不够精细
5. **缺少对长尾场景的专门分析**：如极端天气、复杂交叉口等场景下的表现未单独讨论

## 与相关工作的对比
- **vs DiffusionDrive / GoalFlow**：这些方法用扩散/流匹配改进轨迹生成过程，但不建模未来场景；SeerDrive 从场景预测角度入手，提供互补视角
- **vs WoTE**：WoTE 用世界模型在线评估候选轨迹（选最优解），SeerDrive 则通过迭代交互让世界模型直接参与规划优化
- **vs OccWorld / Drive-OccWorld**：这两者用 occupancy 做场景预测并联合预测动作，但是自回归逐帧生成；SeerDrive 直接预测终帧 BEV 并通过迭代细化
- **vs LAW / SSR**：这些方法将世界模型仅作为训练时的辅助监督信号，推理时不参与；SeerDrive 在推理时也使用世界模型进行迭代交互

## 启发与关联
1. 双向建模思路可扩展到其他决策任务（如机器人操作中的环境预测与动作规划耦合）
2. 迭代交互范式类似 Diffusion 的去噪过程——能否用 diffusion/flow matching 替代当前的确定性迭代？
3. 将 BEV 世界模型替换为更强的生成式模型（如 diffusion-based BEV 生成）可能进一步提升未来场景预测质量
4. 与 DriveTransformer 的 scaling law 研究结合，探索迭代次数和模型规模的协同效应

## 评分
- 新颖性: ⭐⭐⭐⭐ — 双向闭环交互范式是有意义的新视角
- 实验充分度: ⭐⭐⭐⭐ — 两个数据集 + 丰富消融，但缺闭环仿真
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表规范
- 价值: ⭐⭐⭐⭐ — 对端到端驾驶中世界模型的使用方式提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](../../AAAI2026/autonomous_driving/diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[CVPR 2026\] ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/resad_normalized_residual_trajectory_modeling_for_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](../../AAAI2026/autonomous_driving/drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)

</div>

<!-- RELATED:END -->
