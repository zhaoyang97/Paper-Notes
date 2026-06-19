---
title: >-
  [论文解读] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving
description: >-
  [AAAI 2026][自动驾驶][潜在世界模型] 提出面向规划的潜在世界模型框架WorldRFT，通过VGGT空间编码、分层规划分解+局部感知迭代精炼、基于GRPO的碰撞感知强化微调，在nuScenes上将碰撞率降低83%（0.30%→0.05%），在NavSim上仅用相机即逼近LiDAR SOTA（87.8 vs 88.1 PDMS）。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "潜在世界模型"
  - "强化学习微调"
  - "GRPO"
  - "分层规划"
  - "VGGT"
  - "碰撞感知"
---

# WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving

**会议**: AAAI 2026  
**arXiv**: [2512.19133](https://arxiv.org/abs/2512.19133)  
**代码**: [pengxuanyang/WorldRFT](https://github.com/pengxuanyang/WorldRFT)  
**领域**: 自动驾驶 / 端到端规划 / 世界模型  
**关键词**: 潜在世界模型, 强化学习微调, GRPO, 分层规划, VGGT, 碰撞感知

## 一句话总结
提出面向规划的潜在世界模型框架WorldRFT，通过VGGT空间编码、分层规划分解+局部感知迭代精炼、基于GRPO的碰撞感知强化微调，在nuScenes上将碰撞率降低83%（0.30%→0.05%），在NavSim上仅用相机即逼近LiDAR SOTA（87.8 vs 88.1 PDMS）。

## 研究背景与动机

**领域现状**：端到端自动驾驶正从多任务感知架构（UniAD、VAD、SparseDrive等，依赖3D标注）转向基于潜在世界模型的自监督范式（LAW、SSR等，无需感知标注）。

**痛点一：空间感知弱**：当前重建导向的潜在表征缺乏3D空间理解力。World4Drive的单目深度估计存在跨视角不一致。

**痛点二：规划交互低效**：单一全局规划query从整个特征图生成轨迹，注意力分散，无法捕获局部关键结构。

**痛点三：安全意识缺失**：纯模仿学习仅最小化与专家轨迹偏差，不区分安全/非安全偏差，缺乏主动碰撞回避能力。

**核心idea**：三模块对齐"场景理解→规划决策→安全优化"全链路——VGGT几何先验增强空间感知、分层规划+局部感知精炼提取规划相关特征、GRPO强化微调实现主动避碰。

## 方法详解

### 整体框架
环视RGB → **SWE**（ResNet+冻结VGGT融合，输出空间感知潜在表征 $W^t_{\text{latent}}$）→ **HPR**（三并行子任务query交互 + 局部感知迭代精炼K=3轮）→ **RFT**（轨迹高斯化 + 碰撞奖励 + GRPO策略优化）

### 关键设计一：Spatial-aware World Encoder (SWE)
- ResNet-50提取多视图特征 $F_t \in \mathbb{R}^{M \times h \times w \times D}$
- 冻结VGGT从环视图提取多视角一致3D token $t_{3D}$
- 轻量跨注意力融合：$W^t_{\text{latent}} = \text{Cross-Attn}(F_t, t_{3D})$
- Grounded-SAM生成伪语义标签，交叉熵损失 $\mathcal{L}_{sem}$ 辅助语义学习

### 关键设计二：Hierarchical Planning Refinement (HPR)
**三并行子任务**：
- **目标区域定位**：Laplace分布建模（中心 $\mu$ + 尺度 $b$），NLL损失训练。尺度 $b$ 反映场景复杂度，作为后续精炼的条件信号
- **空间路径规划**：每隔2m采样N=30/50个空间均匀路径点（非时间均匀），解耦空间/时间学习
- **时序轨迹预测**：每0.5s共T=6/8个轨迹点，L1损失对齐专家轨迹

**局部感知迭代精炼（K=3轮）**：
1. 将三子任务输出编码为统一状态 $F_s$
2. 轨迹点投影到特征图 → 可变形卷积采样局部特征 $F_{\text{local}}$
3. 以目标区域尺度 $b$ 为条件，融合局部+全局特征
4. 残差更新：$T^{(k+1)} = T^{(k)} + 0.1 \cdot \Delta T^{(k)}$

### 关键设计三：Safety-aware RFT（GRPO碰撞感知微调）
1. **轨迹高斯化**：将确定性轨迹转为高斯分布（均值 $\mu_\theta$ + 辅助方差网络 $\Sigma_\theta$），使RL探索成为可能
2. **碰撞感知奖励**：$r=-1$（碰撞）/$0$（安全），基于自车与周围agent bbox距离
3. **GRPO优化**：采样G=10条轨迹 → 逐点归一化奖励 → 累积优势函数 $Adv_j=\sum_{t\geq j}\tilde{r}_t$ → PPO-clip目标 + KL散度约束
4. 附加参考损失（$l_2$回归到预训练输出）+ 最大熵损失（防过早收敛）

### 损失函数
- **预训练**：$\mathcal{L} = 0.2\mathcal{L}_{sem} + 0.2\mathcal{L}_{rec} + 0.001\mathcal{L}_{target} + 1.0\mathcal{L}_{traj}$
- **RFT**：$\mathcal{L}_{RL} = -J(\theta) + 0.1 D_{KL} + 0.12\mathcal{L}_{ref} + 0.1\mathcal{L}_{entropy}$

## 实验

### 主实验一：nuScenes开环规划

| 方法 | 训练方式 | L2 Avg↓ | CR 1s↓ | CR 2s↓ | CR 3s↓ | CR Avg↓ |
|------|---------|---------|--------|--------|--------|---------|
| UniAD | P-IL | 1.03 | 0.05 | 0.17 | 0.71 | 0.31 |
| VAD | P-IL | 0.72 | 0.07 | 0.18 | 0.43 | 0.23 |
| PARA-Drive | P-IL | 0.48 | 0.14 | 0.23 | 0.39 | 0.25 |
| DiffusionDrive | P-IL | 0.57 | 0.03 | 0.05 | 0.16 | 0.08 |
| LAW (无标注) | SS-L | 0.61 | 0.14 | 0.21 | 0.54 | 0.30 |
| World4Drive | SS-L | 0.50 | 0.02 | 0.12 | 0.33 | 0.16 |
| **WorldRFT (无RFT)** | SS-L | **0.47** | 0.10 | 0.11 | 0.23 | 0.15 |
| **WorldRFT (含RFT)** | SS-L & RL | 0.48 | **0.00** | **0.00** | **0.16** | **0.05** |

### 主实验二：NavSim闭环规划

| 方法 | 输入 | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|------|-----|-----|------|------|--------|-----|-------|
| UniAD | C | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| DiffusionDrive | C&L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| LAW | C | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| World4Drive | C | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| **WorldRFT (含RFT)** | **C** | **97.8** | **96.8** | **94.0** | **100.0** | **81.7** | **87.8** |

### 消融实验（nuScenes）

| ID | VGGT | Target | Path | Refine | L2↓ | CR↓ |
|----|------|--------|------|--------|-----|-----|
| 1 | — | — | — | — | 0.59 | 0.16 |
| 4 | — | ✓ | ✓ | ✓ | 0.52 | 0.08 |
| 7 | ✓ | ✓ | ✓ | — | 0.50 | 0.06 |
| 8 | ✓ | ✓ | ✓ | ✓ | **0.48** | **0.05** |

### 关键发现
1. **RFT对安全性提升83%**：碰撞率0.30→0.05%，1s和2s碰撞率降至0.00%，L2仅略增0.01m——RFT学会了主动让行
2. **VGGT是关键组件**：加入后L2降7.7%、CR降37.5%，DAC 96.8全场最高（含LiDAR方法）
3. **分层规划+精炼递进有效**：Target+Path使CR从0.16降至0.08（-50%），精炼再降至0.05
4. **概率目标优于确定性**：Laplace分布建模使CR降50%（0.10→0.05）
5. **仅相机逼近LiDAR SOTA**：NavSim 87.8仅落后LiDAR方法DiffusionDrive 0.3分

## 亮点
- 首个将GRPO（源自DeepSeek-Math）引入自动驾驶轨迹规划，实现从行为克隆到主动避碰的范式转变
- "无标注"范式下碰撞率超越所有感知标注方法（0.05% vs DiffusionDrive 0.08%）
- VGGT冻结+轻量cross-attention融合，优雅引入3D几何先验
- 分层规划分解精巧：目标区域（方向意图）+ 空间路径（几何形状）+ 时序轨迹（动力学），各用最合理监督

## 局限性
1. **安全-精度trade-off**：RFT后L2从0.47升至0.48，存在轻微精度牺牲
2. **仅NavSim闭环**：未在CARLA等更复杂仿真环境验证
3. **碰撞奖励过于简单**：仅二值（-1/0），未考虑距离连续奖励或舒适性惩罚
4. **导航指令仍需标注**：左/右/直行指令依赖人工标注
5. **ResNet-50主干**：未探索ViT等更强主干的性能上界

## 相关工作
- **端到端驾驶**：UniAD/VAD（级联BEV多任务）、PARA-Drive（并行架构）、SparseDrive（稀疏感知）、DiffusionDrive（扩散规划）
- **世界模型**：GAIA-1（生成式场景）、DriveWorld（占用预测）、LAW/SSR（自监督潜在世界模型）
- **RL驾驶**：RAD（3DGS虚拟环境+RL）、AlphaDrive（VLM+RL）

## 评分
⭐⭐⭐⭐⭐ — 工作完整度极高，三模块各解决一个核心问题且消融充分。GRPO引入驾驶规划是一大亮点，碰撞率降低83%的结果有说服力。仅相机在NavSim逼近LiDAR SOTA，展现无标注范式巨大潜力。消融覆盖backbone、精炼迭代数、路径配置、目标建模、数据量等各维度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning](../../CVPR2026/autonomous_driving/rlftsim_realistic_and_controllable_multi-agent_traffic_simulation_via_reinforcem.md)
- [\[ECCV 2024\] Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving](../../ECCV2024/autonomous_driving/improving_agent_behaviors_with_rl_fine-tuning_for_autonomous_driving.md)
- [\[CVPR 2026\] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving](../../CVPR2026/autonomous_driving/wam-flow_parallel_coarse-to-fine_motion_planning_via_discrete_flow_matching_for_.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)

</div>

<!-- RELATED:END -->
