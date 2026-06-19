---
title: >-
  [论文解读] Modeling Neural Activity with Conditionally Linear Dynamical Systems
description: >-
  [NeurIPS 2025][线性动力系统] 提出条件线性动力系统（CLDS），通过高斯过程先验让线性动力系统参数随观测到的实验协变量非线性变化，在保留线性模型可解释性和高效推断的同时建模神经回路的非线性动态。 神经群体活动在时间、试次和实验条件之间表现出复杂的非线性动态。传统的线性动力系统（LDS）凭借卡尔曼滤波和EM算法…
tags:
  - "NeurIPS 2025"
  - "线性动力系统"
  - "高斯过程"
  - "神经活动建模"
  - "贝叶斯推断"
  - "环形吸引子"
---

# Modeling Neural Activity with Conditionally Linear Dynamical Systems

**会议**: NeurIPS 2025  
**arXiv**: [2502.18347](https://arxiv.org/abs/2502.18347)  
**代码**: [GitHub](https://github.com/neurostatslab/clds)  
**领域**: 计算神经科学 / 状态空间模型  
**关键词**: 线性动力系统, 高斯过程, 神经活动建模, 贝叶斯推断, 环形吸引子

## 一句话总结

提出条件线性动力系统（CLDS），通过高斯过程先验让线性动力系统参数随观测到的实验协变量非线性变化，在保留线性模型可解释性和高效推断的同时建模神经回路的非线性动态。

## 研究背景与动机

神经群体活动在时间、试次和实验条件之间表现出复杂的非线性动态。传统的线性动力系统（LDS）凭借卡尔曼滤波和EM算法在推断上高效且可解释，但受限于时不变线性假设，无法捕获神经回路中广泛存在的非线性结构（如环形吸引子）。

近年来，机器学习社区提出了大量非线性方法（RNN、Transformer、扩散模型等），虽然预测精度有所提升，但面临几大痛点：（1）模型拟合困难，推断方法各异且缺乏统一框架；（2）模型难以科学解释——分析一个训练好的RNN远比分析一个线性系统要困难；（3）在数据稀缺场景下（如每个条件只有一次试次）表现不佳。

CLDS的核心洞察是：在许多神经科学实验中，外部协变量（如头朝向、运动方向）是已知的，神经回路的非线性往往来源于这些协变量对动态参数的调制。CLDS将这种依赖性显式建模：**条件于协变量后动态是线性的**，但参数在协变量空间上可以非线性地平滑变化。这恰好平衡了经典方法和现代非线性方法的优势。

## 方法详解

### 整体框架

CLDS定义了一族随实验条件 $\bm{u}_t$ 变化的LDS模型。给定 $N$ 个神经元在 $K$ 次试次、长度 $T$ 上的记录：

$$\mathbf{x}_{t+1} = \mathbf{A}(\bm{u}_t)\mathbf{x}_t + \mathbf{b}(\bm{u}_t) + \epsilon_t$$

$$\mathbf{y}_t = \mathbf{C}(\bm{u}_t)\mathbf{x}_t + \mathbf{d}(\bm{u}_t) + \omega_t$$

其中 $\mathbf{x}_t \in \mathbb{R}^D$ 是潜在状态，$\mathbf{y}_t \in \mathbb{R}^N$ 是观测到的神经活动。关键在于系统矩阵 $\{\mathbf{A}, \mathbf{b}, \mathbf{C}, \mathbf{d}, \mathbf{m}\}$ 都是条件变量 $\bm{u}_t$ 的函数，且这种映射是非线性可学习的。

### 关键设计

1. **高斯过程先验（GP Prior）**: 对每个参数矩阵的每个元素施加近似GP先验，使用有限基函数展开 $\mathbf{M}_{ij}(\bm{u}) = \sum_{\ell=1}^{L} w_\ell^{(ij)} \phi_\ell(\bm{u})$，权重 $w_\ell^{(ij)} \sim \mathcal{N}(0,1)$。基函数选用正则傅里叶特征来近似平方指数核。设计动机：GP先验在条件空间上编码了平滑性假设，允许在相邻条件之间共享统计力量（statistical power sharing）。长度尺度 $\kappa$ 控制了模型的表达力——$\kappa \to \infty$ 退化为时不变LDS，$\kappa \to 0$ 则各条件独立。

2. **条件线性回归与闭式MAP推断**: 通过Kronecker积将条件化的回归问题转化为扩展特征空间中的贝叶斯线性回归 $\mathbf{y}_n = \mathbf{W}^\top \mathbf{z}_n + \epsilon_n$，其中 $\mathbf{z}_n = \phi(\bm{u}_n) \otimes \mathbf{x}_n$。MAP估计归结为求解Sylvester方程 $\mathbf{Z}^\top\mathbf{Z}\mathbf{W} + \mathbf{W}\Sigma = \mathbf{Z}^\top\mathbf{Y}$，可以解析求解。设计动机：保留线性回归的闭式解，避免随机梯度下降等迭代优化。

3. **EM推断框架**: E步通过卡尔曼平滑获取潜在状态后验矩（精确计算），M步利用上述条件线性回归的闭式解更新参数。所有E步和M步都是解析的，保证边际对数似然单调递增。设计动机：充分利用条件线性结构，保留经典LDS的所有推断优势。

4. **复合动态（Composite Dynamics）可视化**: 通过对 $\bm{u}_t$ 在条件 $\mathbf{x}_t$ 下取期望来边际化条件变量 $\mathbf{x}_{t+1} = \mathbb{E}_{p(\bm{u}|\mathbf{x}_t)}[\mathbf{A}(\bm{u})\mathbf{x}_t + \mathbf{b}(\bm{u})]$，将条件依赖的线性动态"拼接"成全局非线性流场，便于可视化和解释。

### 损失函数 / 训练策略

目标函数为MAP估计的后验对数概率。使用EM算法迭代优化，从GP先验中采样初始化。超参数 $\{L, \kappa, \sigma\}$ 和潜在维度 $D$ 通过80/20试次划分的验证集选择。支持扩展到非高斯似然（如Poisson）——此时后验仍为对数凹的，可用标准优化例程。

## 实验关键数据

### 主实验

| 实验场景 | 指标 | CLDS | LDS | gpSLDS | LFADS |
|---------|------|------|-----|--------|-------|
| 合成HD环形吸引子 | Co-smoothing $R^2$ | **0.86** | - | - | - |
| 真实小鼠HD（ADn） | 调谐曲线恢复 | 近乎完美 | - | - | - |
| 猕猴中心外伸实验（1试次/条件） | Co-smoothing $R^2$ | **最高** | 最低 | 接近CLDS | 最低 |
| 猕猴中心外伸实验（多试次/条件） | Co-smoothing $R^2$ | **最高** | - | 接近CLDS | 持续改善但仍低 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 固定 $\mathbf{C}$ vs 学习 $\mathbf{C}$ | 参数恢复质量 | 固定 $\mathbf{C}$ 避免不可辨识性，学习 $\mathbf{C}$ 恢复质量依然良好 |
| 不同信噪比 | 参数恢复 $R^2$ | 信噪比下降时恢复质量仍然可靠 |
| 不同训练试次数 | Co-smoothing $R^2$ | CLDS在1个试次/条件时优势最大（数据效率出色） |
| GP长度尺度 $\kappa$ | 模型表达力 | $\kappa$ 控制了从时不变LDS到完全非参的连续谱 |

### 关键发现

- CLDS在极低数据场景（1试次/条件）大幅超越基线模型，体现了贝叶斯框架和条件间统计力量共享的优势
- 在合成环形吸引子实验中，CLDS几乎完美恢复了真实的动态矩阵 $\mathbf{A}(\theta)$ 和偏置 $\mathbf{b}(\theta)$
- 在真实小鼠HD数据中，CLDS恢复了环形吸引子结构，且经验调谐曲线与模型预测高度一致
- 即使底层数据并非CLDS生成（模型失配），CLDS仍然能捕获核心非线性结构

## 亮点与洞察

- **优雅的设计哲学**：不是简单地叠加非线性，而是在观测协变量和潜在动态之间做了精巧的分工——非线性交给协变量空间，线性保留在潜在空间。这种分解既有物理意义又有计算优势。
- **与Wishart过程模型的深层联系**：CLDS可视为Wishart过程的动态扩展，为跨时间噪声相关性估计提供了新框架。
- **数据效率**：在神经科学中试次数据稀缺是常态，CLDS通过贝叶斯先验和条件间插值巧妙解决了这一实际问题。

## 局限与展望

- 当前仅实现了高斯观测模型，尚未扩展到Poisson等更真实的尖峰计数似然
- 假设了条件线性动态，当潜在状态与协变量之间的关联较弱时（如认知任务中的长期内部推理），近似误差可能增大
- 依赖观测到的协变量时间序列 $\bm{u}_{1:T}$，在预测或协变量部分缺失时性能下降
- 未充分利用参数上的GP先验来传递完整的参数后验分布

## 相关工作与启发

- **与gpSLDS的对比**：CLDS用观测协变量而非离散潜在过程来驱动动态转换，推断更直接但代价是非全无监督
- **与LFADS的对比**：LFADS是完全非线性替代方案，在大数据下可能更强，但在小数据下不如CLDS
- **启发**：条件线性的思想可推广到其他序列建模问题——任何有丰富外部协变量的时序数据都可能受益于这类"局部线性+全局非线性"的建模范式

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 条件线性的思路虽有先例，但GP先验+闭式EM的完整框架推进显著
- **实验充分度**: ⭐⭐⭐⭐ — 合成与真实数据结合，含模型失配测试和低数据分析，但缺少非高斯似然验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 行文清晰，数学推导严谨，图表设计优秀
- **价值**: ⭐⭐⭐⭐ — 为计算神经科学提供了实用且有原则的建模工具，在数据稀缺场景尤其有价值

## 补充笔记

- CLDS与SLDS的核心区别：SLDS的"切换"由离散潜在过程驱动（推断困难），CLDS的"切换"由观测协变量驱动（推断简单）
- 在猕猴reaching实验中使用了二维条件（角度+延迟/运动指示），展示了CLDS处理混合连续-离散条件的能力
- 代码已开源，基于JAX实现，适合GPU加速

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](../../ICLR2026/others/a_new_approach_to_controlling_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[ICML 2025\] Position: Solve Layerwise Linear Models First to Understand Neural Dynamical Phenomena](../../ICML2025/others/position_solve_layerwise_linear_models_first_to_understand_neural_dynamical_phen.md)
- [\[NeurIPS 2025\] Model Context Protocol for Vision Systems: Audit, Security, and Protocol Extensions](model_context_protocol_for_vision_systems_audit_security_and_protocol_extensions.md)

</div>

<!-- RELATED:END -->
