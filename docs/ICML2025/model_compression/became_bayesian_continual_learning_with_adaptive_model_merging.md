---
title: >-
  [论文解读] BECAME: BayEsian Continual Learning with Adaptive Model MErging
description: >-
  [ICML 2025][模型压缩][持续学习] 提出 BECAME——基于贝叶斯持续学习原则重新建模模型融合机制，利用 Laplace 近似推导出最优融合系数的闭式解，结合梯度投影（稳定性）和无约束训练（可塑性）的两阶段框架，在多个持续学习基准上显著超越 SOTA。 领域现状：持续学习旨在让模型在新任务上增量学习而不灾难性遗…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "持续学习"
  - "模型融合"
  - "贝叶斯学习"
  - "梯度投影"
  - "稳定性-可塑性平衡"
---

# BECAME: BayEsian Continual Learning with Adaptive Model MErging

**会议**: ICML 2025  
**arXiv**: [2504.02666](https://arxiv.org/abs/2504.02666)  
**代码**: [https://github.com/limei0818/BECAME](https://github.com/limei0818/BECAME)  
**领域**: 持续学习  
**关键词**: 持续学习, 模型融合, 贝叶斯学习, 梯度投影, 稳定性-可塑性平衡

## 一句话总结
提出 BECAME——基于贝叶斯持续学习原则重新建模模型融合机制，利用 Laplace 近似推导出最优融合系数的闭式解，结合梯度投影（稳定性）和无约束训练（可塑性）的两阶段框架，在多个持续学习基准上显著超越 SOTA。

## 研究背景与动机

**领域现状**：持续学习旨在让模型在新任务上增量学习而不灾难性遗忘旧任务。核心挑战是稳定性（保留旧知识）和可塑性（学习新任务）的平衡。

**现有痛点**：
   - 梯度投影方法（OGD、GPM 等）通过约束梯度到旧特征空间的正交补空间来保证稳定性——但严格约束限制了可塑性
   - 模型融合方法（IMM、CoMA）将旧模型和新模型的参数做加权平均——但融合系数要么手动调、要么用简单启发式，缺乏理论基础
   - 现有融合方法假设各任务独立（独立高斯后验）——忽略了任务间的序贯依赖

**核心矛盾**：梯度投影牺牲可塑性换稳定性，无约束训练牺牲稳定性换可塑性——两者都是次优的。模型融合有望结合两者优点，但最优融合点在哪里？

**本文目标**：理论推导最优融合系数+实践验证。

**切入角度**：贝叶斯持续学习中，旧模型提供先验分布，新数据提供似然——后验的 MAP 估计自然定义了最优融合点。

**核心 idea**：在梯度投影解 $\theta^{GP}$（高稳定性）和无约束解 $\hat{\theta}$（高可塑性）之间的线性路径上，存在最优融合点 $\theta^* = (1-\alpha)\theta^{GP} + \alpha \hat{\theta}$，且 $\alpha$ 有基于 Laplace 近似的闭式解。

## 方法详解

### 整体框架
BECAME 两阶段流程（对每个新任务）：
1. **Stage 1 - 梯度投影训练**：得到稳定但可塑性不足的 $\theta^{GP}$
2. **Stage 1.5 - 无约束继续训练**：从 $\theta^{GP}$ 出发不加约束训练得到 $\hat{\theta}$（高可塑性但遗忘旧任务）
3. **Stage 2 - 自适应融合**：计算最优 $\alpha^*$ 并融合 $\theta^* = (1-\alpha^*)\theta^{GP} + \alpha^* \hat{\theta}$

### 关键设计

1. **模型融合的贝叶斯重建**:

    - 功能：将线性模型融合重新表述为贝叶斯后验的 MAP 估计
    - 核心思路：
        - 旧后验 $p(\theta | D_{\text{old}}) \approx \mathcal{N}(\theta^{GP}, H_{\text{old}}^{-1})$（Laplace 近似）
        - 新数据似然 $p(D_{\text{new}} | \theta)$
        - 融合后验 $p(\theta | D_{\text{all}}) \propto p(D_{\text{new}} | \theta) \cdot p(\theta | D_{\text{old}})$
    - 与前人的区别：IMM/CoMA 假设各任务后验独立→融合系数缺乏理论依据；BECAME 基于序贯贝叶斯更新→融合系数从后验 MAP 自然导出
    - 设计动机：贝叶斯框架提供了最优性保证——融合点最大化所有任务的联合后验

2. **最优融合系数的闭式解**:

    - 功能：在 $\theta^{GP}$ 和 $\hat{\theta}$ 的线性路径上找最优 $\alpha^*$
    - 核心思路：
        - 目标：$\alpha^* = \arg\min_\alpha \mathcal{L}_{\text{all}}((1-\alpha)\theta^{GP} + \alpha \hat{\theta})$
        - 关键定理：累积损失沿线性路径是 $\alpha$ 的凸函数（在 Laplace 近似下）
        - 闭式解：$\alpha^* = \frac{\delta^T H_{\text{old}} \delta + \nabla \mathcal{L}_{\text{new}}(\theta^{GP})^T \delta}{\delta^T (H_{\text{old}} + H_{\text{new}}) \delta}$
       其中 $\delta = \hat{\theta} - \theta^{GP}$，$H$ 为 Hessian
    - 设计动机：闭式解消除了超参数搜索——不同任务、不同损失景观自动产生不同的 $\alpha^*$

3. **可塑性存在性定理**:

    - 功能：证明融合点一定优于两个端点（理论保证）
    - 核心思路：沿线性路径，累积损失在两个端点的梯度方向相反→中间必有极值点
    - 设计动机：理论保证了 BECAME 不会比单独使用梯度投影或无约束训练更差

### 损失函数 / 训练策略
- Stage 1：标准交叉熵 + 梯度投影约束
- Stage 1.5：标准交叉熵（无约束）
- Stage 2：计算 $\alpha^*$（闭式解，无需训练）
- Hessian 用 Fisher 信息矩阵对角近似（计算高效）
- 即插即用——可叠加到任何梯度投影方法上

## 实验关键数据

### 主实验
10-split CIFAR-100（Class-IL 设置）：

| 方法 | 平均准确率 ↑ | 遗忘率 ↓ | 最终任务可塑性 ↑ |
|------|-----------|---------|---------------|
| OGD | 68.2 | 12.3 | 76.5 |
| GPM | 71.4 | 9.8 | 78.2 |
| NSCL | 73.1 | 8.5 | 79.8 |
| IMM (等权融合) | 72.8 | 9.1 | 81.3 |
| CoMA (手动调权) | 74.5 | 8.2 | 82.1 |
| **BECAME (闭式最优融合)** | **77.3** | **6.9** | **85.7** |

### 多基准对比

| 基准 | BECAME 提升 (vs 最佳基线) |
|------|----------------------|
| 10-split CIFAR-100 | +2.8% |
| 20-split CIFAR-100 | +3.5% |
| 5-split miniImageNet | +2.1% |
| 10-split TinyImageNet | +4.2% |

### 消融实验

| 配置 | 准确率 (CIFAR-100) | 说明 |
|------|-------------------|------|
| 仅梯度投影（$\alpha=0$） | 73.1 | 高稳定性低可塑性 |
| 仅无约束（$\alpha=1$） | 65.8 | 高可塑性低稳定性 |
| 等权融合（$\alpha=0.5$） | 74.2 | 手动但合理 |
| **闭式最优 $\alpha^*$** | **77.3** | 自适应最优 |
| $\alpha^*$ 用网格搜索验证 | 77.1 | 确认闭式解接近最优 |

### 关键发现
- BECAME 的最优 $\alpha^*$ 因任务而异——简单任务 $\alpha^*$ 较大（更多可塑性），困难任务 $\alpha^*$ 较小（更多稳定性）
- 可塑性指标提升最显著（+5.9%）——说明融合主要释放了被梯度投影抑制的可塑性
- 闭式解与网格搜索的差异 <0.2%——验证了理论推导的准确性
- 在所有 4 个基准上一致超越 SOTA——方法的通用性强
- Laplace 近似虽然是近似，但闭式解足够准确——实际中 Hessian 的对角近似就够用

## 亮点与洞察
- **贝叶斯推导 → 闭式融合系数**——将模型融合从"凭经验调参"提升到"有理论保证的最优解"
- 两阶段框架巧妙：先用梯度投影"划定安全范围"，再用无约束训练"探索极限"，最后用最优融合"找到最佳平衡点"
- 可塑性存在性定理的直觉极佳：两个端点的梯度方向相反 → 中间必有更好的点
- 即插即用设计：任何现有梯度投影方法（OGD/GPM/NSCL）都可以立即受益
- 对角 Fisher 近似使方法在计算上与标准训练几乎无差异

## 局限与展望
- Laplace 近似假设后验为高斯——对复杂多模态后验可能不准确
- 对角 Hessian 近似忽略了参数间的相关性
- 线性融合路径假设可能不最优——非线性路径（如曲线融合）可能找到更好的融合点
- 仅在分类任务上验证——生成任务、NLP 任务待探索
- 任务边界需要已知——任务增量设置可能不适用于任务无关场景

## 相关工作与启发
- **vs IMM**: 等权或手动调权融合，无理论保证；BECAME 有贝叶斯最优保证+闭式解
- **vs CoMA**: 手动超参搜索找融合系数；BECAME 自动计算
- **vs EWC**: 正则化方法直接约束参数变化；BECAME 先不约束再融合，更灵活
- **vs PackNet/HAT**: 架构方法为每个任务分配专属参数；BECAME 共享参数但自适应融合
- **启发**：贝叶斯推导在模型融合中的应用可推广到任何多模型组合场景（如联邦学习、模型集成）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 贝叶斯推导闭式融合系数具有理论深度和实用价值
- 实验充分度: ⭐⭐⭐⭐⭐ 4基准×多方法×充分消融×理论验证
- 写作质量: ⭐⭐⭐⭐⭐ 损失景观可视化极其直观
- 价值: ⭐⭐⭐⭐⭐ 为持续学习中的模型融合提供了理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)
- [\[ICML 2025\] TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](treelora_efficient_continual_learning_via_layer-wise_loras_guided_by_a_hierarchi.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](../../NeurIPS2025/model_compression/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[ICML 2025\] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective](rethinking_the_stability-plasticity_trade-off_in_continual_learning_from_an_arch.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/model_compression/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)

</div>

<!-- RELATED:END -->
