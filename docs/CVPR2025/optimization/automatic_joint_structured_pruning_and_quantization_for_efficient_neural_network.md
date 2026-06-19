---
title: >-
  [论文解读] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression
description: >-
  [CVPR 2025][优化/理论][剪枝] 提出 GETA 框架实现自动联合结构化剪枝和量化感知训练：量化感知依赖图（QADG）构建通用剪枝搜索空间 + 部分投影 SGD 保证逐层比特约束 + 可解释的联合学习策略，在 CNN 和 Transformer 上均达到竞争力或领先的压缩性能。 领域现状：结构化剪枝和量化是两种基…
tags:
  - "CVPR 2025"
  - "优化/理论"
  - "剪枝"
  - "量化"
  - "joint optimization"
  - "dependency graph"
  - "QADG"
---

# Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression

**会议**: CVPR 2025  
**arXiv**: [2502.16638](https://arxiv.org/abs/2502.16638)  
**代码**: —  
**领域**: 优化 / 模型压缩  
**关键词**: structured pruning, quantization-aware training, joint optimization, dependency graph, QADG

## 一句话总结
提出 GETA 框架实现自动联合结构化剪枝和量化感知训练：量化感知依赖图（QADG）构建通用剪枝搜索空间 + 部分投影 SGD 保证逐层比特约束 + 可解释的联合学习策略，在 CNN 和 Transformer 上均达到竞争力或领先的压缩性能。

## 研究背景与动机

**领域现状**：结构化剪枝和量化是两种基础的 DNN 压缩技术，通常独立应用。联合优化（co-optimization）有潜力产生更小、更高质量的模型。

**现有痛点**：
   - **工程困难**：现有联合方案流程复杂，涉及多阶段（先剪枝再量化、交替优化等）
   - **黑盒优化**：需要大量超参数调节来控制整体压缩率（如每层剪枝率和比特宽度的搜索）
   - **架构泛化不足**：大多数方法仅适用于特定网络架构（如仅 CNN），无法自动处理任意 DNN

**核心矛盾**：剪枝改变网络结构（通道数），量化改变数值精度（比特宽度），两者交互复杂，独立优化各自的超参数已是 NP-hard，联合更具挑战。

**切入角度**：
   - 用 QADG 自动构建任意量化感知网络的剪枝搜索空间
   - 用部分投影 SGD 将离散比特约束转化为连续优化问题
   - 用白盒优化替代黑盒搜索

**核心 idea**：QADG 统一搜索空间 + 投影 SGD 约束满足 + 可解释剪枝-量化关系 = 一键联合压缩。

## 方法详解

### 整体框架
GETA 接受任意 DNN 和目标压缩率作为输入，输出联合剪枝+量化后的模型：
1. QADG 分析网络结构，构建剪枝搜索空间
2. 在搜索空间中联合优化每层的剪枝率和比特宽度
3. 部分投影 SGD 确保约束满足
4. 一次训练，无需后处理

### 关键设计

1. **量化感知依赖图 (QADG)**

    - 功能：为任意量化感知 DNN 自动构建结构化剪枝搜索空间
    - 核心思路：
        - 扩展传统依赖图以考虑量化操作（如伪量化节点）
        - 自动识别可剪枝的通道组和它们的依赖关系
        - 处理跳连、多分支等复杂拓扑
    - 优势：**架构无关**，可处理 CNN、Transformer、混合架构等任意结构
    - 实现：基于计算图的静态分析

2. **部分投影随机梯度法 (Partially Projected SGD)**

    - 功能：保证逐层比特宽度约束在训练过程中始终满足
    - 核心思路：
        - 将离散比特约束 $b_l \in \{2, 4, 8, ...\}$ 松弛为连续变量
        - 每步梯度更新后投影到约束集上
        - 交替更新权重参数和比特宽度/剪枝率
    - 数学保证：收敛到约束可行域内的驻点
    - 优势：无需外层搜索（如 NAS、强化学习），白盒可解释

3. **联合学习策略**

    - 功能：建立剪枝和量化之间的可解释关系
    - 核心思路：
        - 剪枝后通道减少 → 同一层可用更高精度量化
        - 量化后精度降低 → 需保留更多通道补偿
        - 通过拉格朗日乘子法自动平衡两者
    - 关键洞察：剪枝率和比特宽度之间存在**互补关系**
    - 实现：联合优化目标函数同时包含精度损失和压缩率约束

### 训练策略
- 端到端一次性训练，无需多阶段
- 不需要预训练-剪枝-微调的传统流程
- 支持从头训练和从预训练模型出发

## 实验关键数据

### ResNet-18 / ImageNet

| 方法 | Top-1 Acc↑ | FLOPs↓ | 描述 |
|------|-----------|--------|------|
| 仅剪枝 baseline | 参考 | 参考 | 独立剪枝 |
| 仅量化 baseline | 参考 | 参考 | 独立量化 |
| 联合 baseline | 参考 | 参考 | 两阶段 |
| **GETA** | **竞争力/最优** | **更高压缩率** | 一阶段联合 |

### Transformer 架构（ViT / DeiT）

| 方法 | Top-1 Acc↑ | 压缩率 | 特点 |
|------|-----------|--------|------|
| 独立剪枝 | 基线 | 中 | 仅剪枝 |
| 独立量化 | 基线 | 中 | 仅量化 |
| **GETA** | **更高** | **更高** | 联合优化 |

### 消融实验

| 组件 | 对精度的影响 |
|------|-----------|
| w/o QADG（手动搜索空间） | 精度下降，且不泛化 |
| w/o 投影SGD（无约束） | 约束违反，比特不可控 |
| w/o 联合策略（独立优化） | 压缩率-精度 trade-off 变差 |
| **完整 GETA** | **最优 trade-off** |

### 关键发现
- 联合优化始终优于独立优化后简单组合
- QADG 的自动化消除了手工设计搜索空间的需求
- 投影 SGD 确保训练过程中约束从不违反
- 在 CNN 和 Transformer 上均有效，验证了架构无关性

## 亮点与洞察
- **完全自动化**：无需手动设计每层的剪枝率/比特宽度
- **白盒优化**：相比 NAS 类黑盒搜索，可解释性强
- **一次训练**：消除多阶段流程的工程复杂度
- **架构通用**：QADG 自动处理任意网络拓扑

## 局限与展望
- 极端压缩率下精度下降仍较大
- QADG 构建需要静态图分析，对动态图支持有限
- 目前仅验证了分类任务，检测/分割等下游任务待验证

## 评分
- 新颖性: ⭐⭐⭐⭐ QADG+投影SGD+联合策略组合新颖
- 实验充分度: ⭐⭐⭐⭐ CNN+Transformer双架构验证
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 对模型部署有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](../../ICML2025/optimization/the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [\[NeurIPS 2025\] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](../../NeurIPS2025/optimization/dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)
- [\[ICML 2025\] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](../../ICML2025/optimization/sdp-crown_efficient_bound_propagation_for_neural_network_verification_with_tight.md)
- [\[ICML 2025\] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](../../ICML2025/optimization/interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](../../NeurIPS2025/optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)

</div>

<!-- RELATED:END -->
