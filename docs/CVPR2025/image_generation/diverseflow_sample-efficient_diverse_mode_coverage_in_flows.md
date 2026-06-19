---
title: >-
  [论文解读] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows
description: >-
  [CVPR 2025][图像生成][Flow Matching] 本文提出DiverseFlow，一种无需训练的推理时方法，通过行列式点过程（DPP）在flow模型的ODE求解过程中引入样本间耦合梯度约束，在固定采样预算下显著提高生成样本的多样性和模式覆盖率。 领域现状：Flow Matching和扩散模型等连续时间生成模型…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "Flow Matching"
  - "多样性采样"
  - "行列式点过程"
  - "多义词生成"
  - "模式覆盖"
---

# DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows

**会议**: CVPR 2025  
**arXiv**: [2504.07894](https://arxiv.org/abs/2504.07894)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: Flow Matching, 多样性采样, 行列式点过程, 多义词生成, 模式覆盖

## 一句话总结
本文提出DiverseFlow，一种无需训练的推理时方法，通过行列式点过程（DPP）在flow模型的ODE求解过程中引入样本间耦合梯度约束，在固定采样预算下显著提高生成样本的多样性和模式覆盖率。

## 研究背景与动机

**领域现状**：Flow Matching和扩散模型等连续时间生成模型已成为主流，在文本到图像生成等任务中取得了卓越效果。目前大量工作聚焦于提升保真度和采样效率。

**现有痛点**：标准的IID采样方式在有限采样预算下，可能反复生成相似的结果而遗漏分布中的其他模式。例如prompt "A famous boxer"可能只生成狗（拳师犬），而忽略运动员这一含义。用户不得不反复采样直到覆盖所需模式。

**核心矛盾**：源分布中距离远的样本，经过flow映射后不一定在目标分布中也远——flow映射不保持距离结构。因此"选择多样的source样本"这种直觉方法不奏效。同时，优化source样本需要多次完整ODE仿真和反向传播，计算成本过高。

**本文目标**：在不增加采样次数的前提下，让K个样本覆盖尽可能多的模式。

**切入角度**：利用DPP的"repulsion"特性——DPP天然给"更多样"的集合赋予更高概率，且可微分。

**核心 idea**：在ODE求解的每一步，用当前样本估计目标样本，构建DPP似然度量样本集的多样性，将其梯度注入ODE速度场，形成一组耦合ODE使样本互斥。

## 方法详解

### 整体框架
给定K个源样本 $\{x_0^{(i)}\}$，在标准flow ODE求解过程的每一步，先通过Euler步估计各样本的目标位置 $\hat{x}_1^{(i)}$，然后在特征空间构建DPP核矩阵评估集合多样性，对DPP对数似然求梯度并注入ODE速度场。最终K条轨迹从独立ODE变为耦合ODE系统。

### 关键设计

1. **DPP多样性目标**:

    - 功能：度量一组样本的多样性并提供梯度信号
    - 核心思路：构建核矩阵 $L^{(ij)} = \exp(-h \|F(\hat{x}_1^{(i)}) - F(\hat{x}_1^{(j)})\|^2 / \text{med}(D))$，其中 $F$ 为特征提取器（如ViT）。DPP似然 $\mathcal{L} = \det(L) / \det(L+I)$，样本越多样则行列式越大。对数似然的梯度 $\nabla_{x_t^{(i)}} \log \mathcal{L}$ 作为排斥力注入ODE
    - 设计动机：DPP对重复样本赋予零概率（行列式含相同行），是最严格的多样性度量。相比SVGD的核和方法，DPP基于体积的度量更适合发现新模式

2. **质量约束（Quality Constraint）**:

    - 功能：防止多样性梯度将样本推离合理区域
    - 核心思路：通过反向估计 $\hat{x}_0^{(i)}$ 检查其是否仍在源分布的高概率区（用χ²分位数判断），若偏离过远则降低该样本的DPP权重 $q^{(i)}$。修正核为 $L_q = L \odot q q^T$
    - 设计动机：纯排斥力可能将样本推到低密度区产生低质量输出，质量项实现diversity与quality的平衡

3. **耦合ODE系统**:

    - 功能：将独立的K条ODE轨迹耦合为一个多样性驱动的系统
    - 核心思路：修改第i个粒子的速度为 $\tilde{v}_t^{(i)} = v_t^{(i)} - \gamma(t) \nabla_{x_t^{(i)}} \log \mathcal{L}$，其中 $\gamma(t)$ 为时变缩放因子。$\gamma=0$ 退化为标准IID采样。使用Euler方法就求解
    - 设计动机：在ODE求解过程中逐步优化，避免了需要多次完整ODE仿真+反向传播的高计算成本

### 损失函数 / 训练策略
完全无需训练，是推理时的样本优化方法。

## 实验关键数据

### 主实验（ImageNet-256类条件生成，Precision/Recall）

| 方法 | CFG | Precision↑ | Recall↑ |
|------|-----|-----------|---------|
| LFM | 1.5 | 0.69 | 0.44 |
| LFM + DiverseFlow | 1.5 | 0.69 | **0.47** |
| LFM | 2.0 | 0.77 | 0.41 |
| LFM + DiverseFlow | 2.0 | 0.76 | **0.46** |
| LFM | 4.0 | 0.69 | 0.26 |
| LFM + DiverseFlow | 4.0 | 0.70 | **0.38** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 多样源样本直接映射 | 不保证目标多样性（Fig.2验证） |
| IID采样 vs DiverseFlow | K=5覆盖3/10模式 vs 5/10模式 |
| 不同FM formulation | CFM和MB-OT均受益于DiverseFlow |
| 高CFG更受益 | CFG=4时Recall提升+0.12（从0.26到0.38） |

### 关键发现
- DiverseFlow在不降低Precision的前提下显著提升Recall，尤其在高CFG（低多样性）场景下提升最大
- 在2D合成实验中，5个样本可覆盖5个模式（IID只能覆盖3个）
- 在多义词文本生成中能发现多种语义（如"boxer"同时生成拳师犬和运动员）
- 在人脸修复任务中能生成更多样的面部表情和特征

## 亮点与洞察
- 将DPP引入flow模型采样是新颖的组合。DPP的行列式天然适合度量"集合体积"即多样性
- "源空间多样 ≠ 目标空间多样"的观察虽然直觉可得但首次系统验证，为多样性采样研究提供了重要基准
- 质量约束通过源分布的概率密度来防止退化，设计简洁有效

## 局限与展望
- 需要计算样本间的DPP核矩阵和梯度，计算开销与样本数K²成正比
- 当K很大时（远超模式数），排斥力可能导致部分样本质量下降
- 特征提取器F的选择对结果有较大影响，不同任务可能需要不同的F
- 可考虑扩展到SDE采样器和更高效的DPP近似方法

## 相关工作与启发
- **vs Particle Guidance (Corso et al.)**: 基于SVGD的行求和作为多样性度量，容忍重复样本；DiverseFlow基于行列式（体积），对重复赋零概率，更严格
- **vs CFG调节**: 降低CFG可增加多样性但牺牲质量；DiverseFlow在高CFG下也能保持多样性
- **vs 多次独立采样**: DiverseFlow在相同预算下覆盖更多模式，效率更高

## 评分
- 新颖性: ⭐⭐⭐⭐ DPP+Flow的组合新颖，但推理时guidance是已有范式
- 实验充分度: ⭐⭐⭐⭐ 多任务验证（文本生成、修复、类条件），合成数据消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，理论推导严谨，可视化优秀
- 价值: ⭐⭐⭐⭐ 通用推理时多样性方法，应用场景广泛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [\[NeurIPS 2025\] HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing](../../NeurIPS2025/image_generation/hollowflow_efficient_sample_likelihood_evaluation_using_hollow_message_passing.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2026\] Learning Straight Flows: Variational Flow Matching for Efficient Generation](../../CVPR2026/image_generation/learning_straight_flows_variational_flow_matching_for_efficient_generation.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](../../ICLR2026/image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)

</div>

<!-- RELATED:END -->
