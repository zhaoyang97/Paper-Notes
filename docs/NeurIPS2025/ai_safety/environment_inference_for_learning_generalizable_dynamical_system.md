---
title: >-
  [论文解读] Environment Inference for Learning Generalizable Dynamical System
description: >-
  [NeurIPS 2025 Spotlight][AI安全][dynamical systems] 提出 DynaInfer 框架，通过分析固定神经网络的预测误差来推断未标注轨迹的环境标签，实现无环境标签条件下的动态系统泛化学习，在 ODE/PDE 系统上性能匹配甚至超越 Oracle（已知标签）。
tags:
  - "NeurIPS 2025 Spotlight"
  - "AI安全"
  - "dynamical systems"
  - "environment inference"
  - "OOD generalization"
  - "multi-environment learning"
  - "K-means analogy"
---

# Environment Inference for Learning Generalizable Dynamical System

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.19784](https://arxiv.org/abs/2510.19784)  
**代码**: [GitHub](https://github.com/shixuanliu-andy/DynaInfer)  
**领域**: AI Safety / 动态系统泛化  
**关键词**: dynamical systems, environment inference, OOD generalization, multi-environment learning, K-means analogy

## 一句话总结
提出 DynaInfer 框架，通过分析固定神经网络的预测误差来推断未标注轨迹的环境标签，实现无环境标签条件下的动态系统泛化学习，在 ODE/PDE 系统上性能匹配甚至超越 Oracle（已知标签）。

## 研究背景与动机

**领域现状**：数据驱动方法（如 Neural ODE、FNO）在复杂动态系统建模中取得成功，但通常假设训练数据独立同分布。近期工作（LEADS、CoDA）引入多环境设置——将动态分解为全局共享和环境特定两部分——以提升泛化能力。

**现有痛点**：这些泛化方法**依赖环境标签**，即需要知道每条轨迹属于哪个环境。但在实际场景中环境标签往往不可用：科学实验中数据在未知条件下收集；不同来源聚合数据时标签丢失；隐私敏感领域（医疗、金融）环境信息受限。

**核心矛盾**：多环境泛化方法需要环境标签来区分共性和差异，但数据获取过程中环境标签很难获得。如何在**无标签**条件下推断出有意义的环境划分？

**本文目标**：在训练时环境标签完全缺失（甚至环境数目 $M$ 未知）的条件下，为混合轨迹推断环境分配，使下游泛化算法正常工作。

**切入角度**：同一环境的轨迹遵循相同的动态方程，因此在相同神经网络下应产生相似的预测误差。可以利用预测误差的一致性来推断环境归属——这与 K-means 聚类中数据点到质心的距离最小化类似。

**核心 idea**：把神经网络类比为 K-means 的"质心"，把轨迹的预测损失类比为"距离"，通过交替更新环境分配和网络参数来推断潜在环境标签。

## 方法详解

### 整体框架
DynaInfer 是一个迭代式框架：
- **输入**：$N$ 条混合轨迹（无环境标签），假设环境数 $M$
- **输出**：环境分配 $\hat{e}$、全局参数 $\theta$、环境特定参数 $\phi$
- **流程**：随机初始化 → 交替执行（环境推断步 + 参数优化步）→ 收敛

### 问题形式化

动态系统由微分方程 $dx_t/dt = f(x_t)$ 描述。多环境下，每条轨迹 $x^i$ 的动态为 $dx^i_t/dt = h(x^i_t; \theta, \phi_{e_i})$，其中 $\theta$ 为全局参数，$\phi_{e_i}$ 为环境特定参数。优化目标为：

$$\hat{e}^*, \theta^*, \phi^* = \arg\min_{\hat{e}, \theta, \phi} R_{\hat{e}}(\theta, \phi) = \sum_{i=1}^N \int_{t \in I} \left\|\frac{dx^i_t}{dt} - h(x^i_t; \theta, \phi_{\hat{e}_i})\right\|_2^2 dt + \lambda \sum_{e=1}^M \Omega(\phi_e)$$

### 关键设计

1. **Bias-aware Environment Assignment（偏差感知环境分配）**:

    - 功能：在第 $r$ 轮，利用上一轮固定的网络参数推断每条轨迹的最优环境标签
    - 核心思路：$\hat{e}_i^{(r)} = \arg\min_{e \in [M]} \int_{t \in I} \|dx^i_t/dt - h(x^i_t; \theta^{(r-1)}, \phi_e^{(r-1)})\|_2^2 dt$
    - 每条轨迹被分配到使其预测损失最小的环境——类似 K-means 中将数据点分配到最近质心
    - 设计动机：同一环境的轨迹共享动态方程，用相同网络预测时误差模式一致

2. **Assignment-driven Optimization（分配驱动优化）**:

    - 功能：基于当前环境分配，优化全局和环境特定参数
    - 核心思路：$\theta^{(r)}, \phi^{(r)} = \arg\min_{\theta, \phi} R_{\hat{e}^{(r)}}(\theta, \phi)$
    - 类似 K-means 中更新质心为簇内均值
    - 设计动机：以无偏方式更新参数，让每条轨迹等权贡献

3. **理论保证（Proposition 3.1）**:

    - 每轮损失单调不增：$R_{\hat{e}^{(r+1)}}(\theta^{(r+1)}, \phi^{(r+1)}) \leq R_{\hat{e}^{(r)}}(\theta^{(r)}, \phi^{(r)})$
    - 若损失严格下降，则每步至少降低常数 $C > 0$，保证有限步内收敛
    - 在 $h$ 关于 $\theta, \phi_e$ 线性、$\Omega$ 严格凸的假设下，$\arg\min_{\theta,\phi} R_{\hat{e}}(\theta, \phi)$ 解空间有限，满足收敛条件

### 测试时环境推断
- In-domain 测试时同样无标签：用轨迹前 $< 2\Delta t$ 的片段在所有环境特定网络上评估预测偏差，选择最佳匹配的环境
- Domain adaptation 时：在目标域提供标签进行微调

## 实验关键数据

### In-domain 泛化结果（测试 MSE / MAPE）

| 数据集 | 分配策略 | LEADS MSE | CoDA-$l_1$ MSE | CoDA-$l_2$ MSE |
|--------|---------|-----------|---------------|---------------|
| LV | All in One | 7.41E-2 | 7.40E-2 | 7.41E-2 |
| LV | Random | 7.38E-2 | 7.39E-2 | 7.39E-2 |
| LV | One per Env | 4.91E-4 | 9.14E-4 | 8.43E-4 |
| LV | **DynaInfer** | **7.93E-5** | **1.83E-4** | **1.82E-4** |
| LV | Oracle | 7.02E-5 | 3.19E-5 | 2.72E-5 |
| GS | DynaInfer | **4.14E-5** | 1.23E-4 | 7.25E-5 |
| GS | Oracle | 1.34E-4 | 9.60E-5 | 7.04E-5 |
| NS | DynaInfer | **7.05E-3** | 1.62E-2 | 1.19E-2 |
| NS | Oracle | 6.55E-3 | 1.73E-2 | 9.46E-3 |

### Domain Adaptation 结果

| 数据集 | 分配策略 | LEADS MAPE | CoDA-$l_1$ MAPE | CoDA-$l_2$ MAPE |
|--------|---------|-----------|----------------|----------------|
| LV | All in One | 9.92 | 26.90 | 27.80 |
| LV | DynaInfer | **2.84** | **10.16** | **10.30** |
| LV | Oracle | 3.16 | 8.27 | 10.61 |
| NS | DynaInfer | **77.29** | 108.17 | **96.57** |
| NS | Oracle | 67.58 | 124.22 | 91.06 |

### 关键发现
- DynaInfer 在所有数据集和基础模型上**一致性大幅超越**其他无标签策略（All in One、Random、One per Env）
- 在 GS 和 NS 数据集上 DynaInfer 甚至**超过 Oracle**（LEADS base），说明偏差感知的环境推断可以补偿人工标签的局限
- "All in One" 和 "Random" 几乎完全失效（MSE 高出 3 个数量级），证明环境划分对泛化至关重要
- 环境标签快速收敛至真实标签，验证了 K-means 类比的有效性

## 亮点与洞察
- **K-means 类比**非常简洁有力：神经网络≈质心，预测损失≈距离，交替优化≈EM 算法。将聚类思路从欧氏空间迁移到函数空间
- **超越 Oracle** 的现象令人惊讶：表明人工标注的环境边界可能不是学习最优的，数据驱动的划分可能更好
- 框架**模型无关**：可与 LEADS、CoDA 等任意基础泛化模型组合，是一个上游工具
- 理论保证（单调收敛 + 常数步长下降）虽简单但实用

## 局限与展望
- 需要预设环境数 $M$，虽然论文声称对 $M$ 选择鲁棒，但未提供自动确定 $M$ 的方法
- 收敛性分析依赖 $h$ 关于参数线性和 $\Omega$ 严格凸的假设，对深度非线性网络不完全成立
- 环境间差异很小时（如物理参数仅微调），预测损失的区分度可能不足
- 测试时环境推断依赖轨迹的早期片段，对初始条件敏感的系统可能不稳定

## 相关工作与启发
- **vs LEADS**: LEADS 首次提出多环境动态系统泛化，但假设环境标签已知；DynaInfer 解除此假设
- **vs CoDA**: CoDA 提供功能和参数两种分解方式，同样需要标签；DynaInfer 作为上游工具为其提供标签
- **vs 传统聚类（K-means on Euclidean space）**: DynaInfer 将聚类思想扩展到函数空间，用预测损失替代欧氏距离

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次解决动态系统泛化中的环境标签缺失问题，K-means 类比优雅
- 实验充分度: ⭐⭐⭐⭐ ODE+PDE 系统覆盖全面，in-domain+adaptation 双设置，3种基础模型×5种策略
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，算法简洁，理论和实验对应良好
- 价值: ⭐⭐⭐⭐ 解决了实际场景中普遍存在的标签缺失问题，作为插件模块实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding](../../ICML2025/ai_safety/secemb_sparsity-aware_secure_federated_learning_of_on-device_recommender_system_.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](../../ICCV2025/ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[CVPR 2026\] DFD-HR: Generalizable Deepfake Detection via Hierarchical Routing Learning](../../CVPR2026/ai_safety/dfd-hr_generalizable_deepfake_detection_via_hierarchical_routing_learning.md)
- [\[CVPR 2026\] Meta-FC: Meta-Learning with Feature Consistency for Robust and Generalizable Watermarking](../../CVPR2026/ai_safety/meta-fc_meta-learning_with_feature_consistency_for_robust_and_generalizable_wate.md)

</div>

<!-- RELATED:END -->
