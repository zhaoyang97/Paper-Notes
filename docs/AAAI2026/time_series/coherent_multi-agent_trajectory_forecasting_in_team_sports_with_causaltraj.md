---
title: >-
  [论文解读] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj
description: >-
  [AAAI 2026][时间序列][多智能体轨迹预测] 提出CausalTraj——一种时序因果、基于似然的多智能体轨迹预测模型，通过逐步自回归建模智能体间时空交互，在NBA、篮球和橄榄球数据集上实现了联合指标（minJADE/minJFDE）的最优结果，同时保持有竞争力的单智能体精度。 问题背景 多智能体轨迹预测是体育分析…
tags:
  - "AAAI 2026"
  - "时间序列"
  - "多智能体轨迹预测"
  - "因果自回归模型"
  - "团队运动分析"
  - "联合指标"
  - "混合高斯分布"
---

# Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj

**会议**: AAAI 2026  
**arXiv**: [2511.18248](https://arxiv.org/abs/2511.18248)  
**作者**: Wei Zhen Teoh
**代码**: [wezteoh/causaltraj](https://github.com/wezteoh/causaltraj)  
**领域**: 时间序列  
**关键词**: 多智能体轨迹预测, 因果自回归模型, 团队运动分析, 联合指标, 混合高斯分布

## 一句话总结

提出CausalTraj——一种时序因果、基于似然的多智能体轨迹预测模型，通过逐步自回归建模智能体间时空交互，在NBA、篮球和橄榄球数据集上实现了联合指标（minJADE/minJFDE）的最优结果，同时保持有竞争力的单智能体精度。

## 研究背景与动机

### 问题背景
多智能体轨迹预测是体育分析的核心任务，在自动驾驶和人群导航等领域同样关键。该任务要求根据历史观测预测未来多个智能体的联合运动轨迹。其挑战在于：未来具有随机性和多模态性，且每个智能体的运动依赖于所有其他智能体的集体配置。

### 现有方法的问题
当前主流方法（如GroupNet、LED、MoFlow）主要以**单智能体指标**（minADE、minFDE）作为评估和优化目标。这些指标独立评估每个智能体，允许不同智能体选取来自不同预测场景的最优轨迹。这带来两个根本问题：

**缺乏联合建模**：训练损失独立地为每个智能体监督多假设轨迹选择，不建模哪些轨迹能组合成合理的联合未来。模型可能在单智能体指标上表现优异，但联合预测质量差。

**场景不连贯**：生成的轨迹单独看合理，但无法形成连贯的多智能体演化——比如球的运动与球员运动不一致，球员间缺乏协调的站位阵型。

### 设计思路
作者认为**学习真实联合分布**才是核心目标，场景连贯性和单智能体精度应自然地作为副产品出现。受因果自回归架构在语言模型和3D环境生成中成功的启发，CausalTraj采用时序因果框架：逐步演化空间和智能体间动态，而非将整个未来压缩到一个全局潜变量中。相比并行预测全部未来时步的方法（要求所有输出在给定潜变量下条件独立），逐步因果建模降低了对潜变量容量的需求，更适合捕捉复杂的联合依赖关系。

## 方法详解

### 问题建模
考虑$N$个交互智能体在2D坐标空间中运动。给定历史轨迹$X_{1:P} \in \mathbb{R}^{N \times P \times 2}$（$P$个历史帧），目标是预测联合未来轨迹$\hat{X}_{P+1:T} \in \mathbb{R}^{N \times F \times 2}$（$F = T - P$个未来帧）。一个联合预测称为一个"场景"（scenario），模型需要估计条件分布$p(X_{P+1:T} \mid X_{1:P})$。

### 因果似然建模框架
CausalTraj将联合条件分布分解为时步上的因果乘积：

$$p(X_{P+1:T} \mid X_{1:P}) = \prod_{t=P}^{T-1} p(X_{t+1} \mid X_{1:t})$$

模型预测每步位移$\Delta X_{t+1} = X_{t+1} - X_t$的条件分布，而非绝对位置。推理时从预测分布中采样位移并递归更新位置；训练时采用teacher forcing并行计算所有时步。

### 混合高斯输出
每步位移分布建模为$M=8$个高斯成分的混合：

$$p(\Delta X_{t+1} \mid X_{1:t}) = \sum_{m=1}^{M} \pi_{t+1,m} \mathcal{N}(\Delta X_{t+1}; \mu_{t+1,m}, \Sigma_{t+1,m})$$

每个时步网络输出：$M$个混合权重logits、$M \times N \times 2$个均值、$M \times N \times 3$个Cholesky参数。协方差矩阵假设为块对角形式（智能体间条件独立），但共享的混合权重仍能耦合不同智能体的结果，表达联合依赖结构。

训练损失为负对数似然加上混合权重的熵正则化，防止成分坍缩：

$$\mathcal{L} = \mathcal{L}_{\text{NLL}} - \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}, \quad \lambda_{\text{ent}} = 0.05$$

### 模型架构

**智能体历史编码器**：提供两种变体：
- **因果PointNet编码器**：将PointNet的全局max-pooling替换为lookback max-pooling，仅对$t' \leq t$的时步做聚合，保持时序因果性。通过零填充和滑动窗口池化实现，兼顾并行训练效率和层次特征聚合。
- **Mamba2编码器**：用2层MLP投影初始特征后，独立通过Mamba2层处理每个智能体序列。Mamba2结合紧凑状态表示和注意力式上下文建模，在多个数据集上表现更优。

**智能体嵌入**：使用学习的嵌入区分仅3种角色（两支球队 + 球），拼接到编码特征后通过MLP融合。

**智能体间关系编码器**：核心创新是**空间关系Transformer编码器（SRTE）**。在标准self-attention基础上，显式编码成对空间几何信息。具体做法是构建成对"mesh"张量$M_t[q,k] = [x_{q,t} - x_{k,t}; z_{q,t}; z_{k,t}]$，包含相对位移和编码特征。该张量用于计算key/value投影，使注意力机制直接利用精确的欧氏位移信息。

**场景聚合与预测头**：将所有智能体特征与位置/速度信息拼接后通过MLP压缩维度，再按时步拼接所有智能体形成场景级表示，最后通过3层MLP输出混合高斯参数。

### 联合评估指标
论文强调使用联合指标（minJADE、minJFDE），与传统单智能体指标minADE/minFDE的关键区别在于：单智能体指标允许每个智能体从不同场景样本中选最优；而联合指标要求从$k$个完整场景中选出整体最优的一个来评估。这直接衡量模型生成连贯联合配置的能力。

## 实验关键数据

### 表1：NBA SportVU数据集结果（米，minADE₂₀/minFDE₂₀ 和 minJADE₂₀/minJFDE₂₀）

| 时间 | 指标类型 | GroupNet | LED | MoFlow(joint) | MoFlow | CausalTraj(C-PN) | CausalTraj(Mamba2) |
|------|---------|----------|-----|---------------|--------|-----------------|-------------------|
| 1.0s | per-agent | 0.25/0.32 | 0.21/0.27 | 0.28/0.39 | 0.18/0.25 | 0.15/0.21 | **0.14/0.20** |
| 2.0s | per-agent | 0.47/0.68 | 0.44/0.56 | 0.48/0.71 | **0.34/0.47** | 0.34/0.50 | 0.33/0.49 |
| 4.0s | per-agent | 0.95/1.22 | 0.81/1.10 | 0.89/1.32 | **0.71/0.87** | 0.77/1.01 | 0.77/1.02 |
| 1.0s | joint | 0.50/0.77 | 0.34/0.64 | 0.40/0.67 | 0.37/0.68 | 0.28/0.50 | **0.27/0.49** |
| 2.0s | joint | 1.04/1.91 | 0.78/1.55 | 0.81/1.61 | 0.80/1.61 | **0.62/1.18** | 0.62/1.21 |
| 4.0s | joint | 2.12/3.72 | 1.63/2.99 | 1.72/3.33 | 1.69/3.31 | **1.34/2.47** | 1.38/2.57 |

CausalTraj在短时间域（≤2s）单智能体指标上优于所有基线，在联合指标上**全面大幅领先**。以4s预测为例，minJADE从最优基线的1.63（LED）降至1.34，降幅18%。

### 表2：消融实验（Basketball-U，20帧，联合指标）

| 配置 | minJADE₂₀ | minJFDE₂₀ |
|------|-----------|-----------|
| CausalTraj (Mamba2) 完整版 | **0.97** | **1.77** |
| 去掉SRTE（用标准Transformer） | 0.99 | 1.81 |
| 单高斯（去掉混合） | 1.03 | 1.86 |
| 仅用成分均值采样 | 1.05 | 2.13 |

消融结果表明：空间关系Transformer编码器（SRTE）提供了可测量的改进；多模态混合建模至关重要——从8个高斯成分退化到单高斯使minJADE上升6%；从分布中采样而非仅取均值对minJFDE影响尤其显著（从1.77升至2.13，+20%）。

## 亮点与洞察

- **联合指标视角的贡献**：首次系统性地将联合指标（minJADE/minJFDE）引入体育轨迹预测领域，揭示了传统per-agent指标的盲区——许多SOTA模型在联合预测上表现实际远不如预期
- **因果分解的优势**：时序因果分解天然支持逐步建模智能体间交互演化，避免了并行预测方法对全局潜变量容量的高要求。简洁的MoG似然训练无需近似推断
- **定性证据有说服力**：可视化结果清晰展示了CausalTraj能生成协调的方向变换和逼真的球传递，而基线模型倾向于生成平滑但缺乏协调性的运动
- **MoFlow(joint obj.)的教训**：仅修改损失函数为联合目标而不改变模型结构，在min-based联合指标上几乎无提升，说明联合建模需要架构层面的支持

## 局限与展望

- **球-球员交互建模不足**：定性结果仍存在球员持球时球与球员间距离不合理等现象，作者推测与模型中有限的球员-球协方差学习能力有关
- **块对角协方差的局限**：在每个混合成分内假设智能体间条件独立，虽然共享混合权重提供了一定的耦合，但无法精细建模智能体间的空间协方差
- **缺乏物理约束**：球偶尔会与球场边界碰撞等不合理行为，模型未引入物理规则或边界约束
- **自回归推理效率**：逐步采样的推理方式比并行预测方法更慢，在需要实时预测的场景中可能受限
- **数据集规模有限**：仅在三个体育数据集上验证，未扩展到自动驾驶或人群导航等其他多智能体域

## 与相关工作的对比

- **GroupNet (CVPR'22)**: 基于图和超图的空间交互建模，采用CVAE框架。在联合指标上CausalTraj大幅领先（NBA 4s: 1.34 vs 2.12 minJADE）
- **LED (CVPR'23)**: 基于潜扩散的轨迹预测。LED的per-agent指标优于GroupNet但弱于MoFlow，联合指标同样显著落后于CausalTraj
- **MoFlow (CVPR'25)**: 基于flow matching的单步去噪模型，在长时域per-agent指标上仍为最优。但其联合预测能力有限——即使使用联合损失训练的变体在min-based联合指标上也几乎无改善
- **SportsTraj (ICLR'25)**: 在Basketball-U/Football-U上此前最优，使用联合训练目标。但其架构基于Mamba+图网络，CausalTraj的Mamba2变体在所有联合指标上均超越

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将因果自回归框架与MoG似然结合用于联合多智能体轨迹预测、SRTE设计有巧思，但整体是已有组件的有效组合
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集、多基线对比、消融实验完整，定性可视化有说服力。但缺少推理速度对比和更大规模验证
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，联合指标的重要性论证有力。开源代码和项目页面增强了可复现性
- 价值: ⭐⭐⭐⭐ — 联合指标视角对体育轨迹预测社区有引导意义，模型在实际应用（战术模拟、比赛分析）中有明确价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[ICCV 2025\] V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction](../../ICCV2025/time_series/v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[ICLR 2026\] A Unified Federated Framework for Trajectory Data Preparation via LLMs](../../ICLR2026/time_series/a_unified_federated_framework_for_trajectory_data_preparation_via_llms.md)
- [\[AAAI 2026\] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints](mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)

</div>

<!-- RELATED:END -->
