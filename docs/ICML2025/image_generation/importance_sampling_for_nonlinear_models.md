---
title: >-
  [论文解读] Importance Sampling for Nonlinear Models
description: >-
  [ICML2025][图像生成][importance sampling] 通过引入非线性映射的伴随算子（adjoint operator），将线性模型中经典的范数采样和杠杆分数采样系统性地推广到非线性模型，首次为神经网络等非线性模型的重要性采样提供了理论近似保证。 机器学习训练的核心是求解优化问题 $\min_{\thet…
tags:
  - "ICML2025"
  - "图像生成"
  - "importance sampling"
  - "nonlinear adjoint"
  - "leverage scores"
  - "active learning"
  - "subspace embedding"
---

# Importance Sampling for Nonlinear Models

**会议**: ICML2025  
**arXiv**: [2505.12353](https://arxiv.org/abs/2505.12353)  
**代码**: 未开源  
**领域**: 重要性采样 / 非线性模型 / 随机数值线性代数  
**关键词**: importance sampling, nonlinear adjoint, leverage scores, active learning, subspace embedding

## 一句话总结

通过引入非线性映射的伴随算子（adjoint operator），将线性模型中经典的范数采样和杠杆分数采样系统性地推广到非线性模型，首次为神经网络等非线性模型的重要性采样提供了理论近似保证。

## 研究背景与动机

机器学习训练的核心是求解优化问题 $\min_{\theta} \mathcal{L}(\theta) = \sum_{i=1}^{n} \ell(f_i(\theta))$，其中 $f_i$ 可以是非线性映射。随着数据规模爆炸性增长，如何高效选取"重要"的数据子集进行训练成为关键问题。

在**线性模型**中，随机数值线性代数（RandNLA）已经发展出成熟的重要性采样理论：
- **行范数采样**：按 $q_i = \|\mathbf{x}_i\|_2^2$ 分配权重
- **杠杆分数采样**：按 $q_i = \langle \mathbf{e}_i, \mathbf{X}\mathbf{X}^\dagger \mathbf{e}_i \rangle$ 分配权重
- 这些方法在样本量 $s \in \mathcal{O}(d \ln d / \varepsilon^2)$ 时，能保证线性子空间嵌入性质

然而，**将这些工具推广到非线性模型一直是一个重大挑战**。现有工作（如 Gajjar et al., 2023, 2024）虽然对单神经元模型做了探索，但存在明显不足：近似保证中出现了一个远大于1的常数 $C$（超过1000），形式为 $\mathcal{L}(\theta_\mathcal{S}^\star) \leq C \cdot \mathcal{L}(\theta^\star) + \mathcal{O}(\varepsilon)$，严重削弱了实用价值。

本文的核心动机是：**能否建立一个系统化框架，使得非线性模型也能获得 $C=1$ 的高质量近似保证？**

## 方法详解

### 1. 非线性伴随算子（Nonlinear Adjoint Operator）

本文的核心创新在于引入非线性映射 $f: \mathbb{R}^p \to \mathbb{R}$ 的伴随算子。利用积分形式的中值定理：

$$f(\theta) = f(\mathbf{0}) + \int_0^1 \left\langle \frac{\partial}{\partial \theta} f(t\theta), \theta \right\rangle \mathrm{d}t$$

由此定义**伴随算子**为：

$$\mathbf{f}^\star(\theta) \triangleq \int_0^1 \frac{\partial}{\partial \theta} f(t\theta) \, \mathrm{d}t$$

这使得任意可微非线性函数都可以写成"类内积"形式：$f(\theta) = \langle \hat{\theta}, \hat{\mathbf{f}}^\star(\theta) \rangle$，其中 $\hat{\theta} = [\theta; 1]$，$\hat{\mathbf{f}}^\star(\theta) = [\mathbf{f}^\star(\theta); f(\mathbf{0})]$。

**关键意义**：这个表示将非线性函数的结构"线性化"，使得线性子空间嵌入理论可以直接应用。

### 2. 特殊模型的闭式伴随

当 $f = g \circ h$，且 $h$ 具有 $\alpha$ 阶正齐次性（$h(t\theta) = t^\alpha h(\theta)$）时，伴随算子有闭式解（Proposition 3.1）：

$$\mathbf{f}^\star(\theta) = \frac{g(h(\theta)) - g(0)}{\alpha \cdot h(\theta)} \cdot \frac{\partial h(\theta)}{\partial \theta}$$

**广义线性预测器**（$f(\theta) = \phi(\langle \theta, \mathbf{x} \rangle)$）：

$$\mathbf{f}^\star(\theta) = \frac{\phi(\langle \theta, \mathbf{x} \rangle) - \phi(0)}{\langle \theta, \mathbf{x} \rangle} \cdot \mathbf{x}$$

**ReLU 两层神经网络**（$f(\theta) = \sum_{j=1}^m \phi(a_j \cdot \max\{\langle \mathbf{b}_j, \mathbf{x} \rangle, 0\})$）：伴随算子按每个神经元分解，齐次度 $\alpha=2$。

### 3. 非线性重要性采样分数

定义**非线性对偶矩阵** $\hat{\mathbf{F}}^\star(\theta) \in \mathbb{R}^{n \times (p+1)}$，每一行是第 $i$ 个数据点对应的伴随算子。

- **非线性杠杆分数**：$\tau_i(\theta) = \frac{\langle \mathbf{e}_i, \hat{\mathbf{F}}^\star(\theta) [\hat{\mathbf{F}}^\star(\theta)]^\dagger \mathbf{e}_i \rangle}{\text{Rank}(\hat{\mathbf{F}}^\star(\theta))}$

- **非线性范数分数**：$\tau_i(\theta) = \frac{\|\hat{\mathbf{f}}_i^\star(\theta)\|_2^2}{\|\hat{\mathbf{F}}^\star(\theta)\|_F^2}$

### 4. 理论保证

**参数依赖近似**：对固定 $\theta$，采样 $s \in \mathcal{O}(p \log(p/\delta) / \varepsilon^2)$ 个点后，以概率 $\geq 1-\delta$：

$$(1-\varepsilon) \mathcal{L}(\theta) \leq \mathcal{L}_\mathcal{S}(\theta) \leq (1+\varepsilon) \mathcal{L}(\theta)$$

**参数无关近似**：通过两个关键技术解决参数依赖问题：
1. 利用激活函数的有界性条件（如 $l \leq \phi^2(t)/t^2 \leq u$），将非线性分数用线性分数上界近似
2. 利用 $\varepsilon$-网构造和 union bound 将单点保证扩展到全参数空间

最终获得 $C=1$ 的保证：$\mathcal{L}(\theta_\mathcal{S}^\star) \leq \mathcal{L}(\theta^\star) + \mathcal{O}(\varepsilon)$。

### 5. 计算流程

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| 1 | 计算线性杠杆/范数分数作为非线性分数的近似 | $\mathcal{O}(nd^2)$ |
| 2 | 按近似分数非均匀采样 $s$ 个数据点 | $\mathcal{O}(s)$ |
| 3 | 在采样子集上训练模型 | 取决于模型 |
| 4 | （可选）后验诊断：识别重要样本、异常检测 | $\mathcal{O}(n)$ |

## 实验关键数据

实验覆盖多种监督学习场景，验证了理论结果：

| 实验场景 | 采样方式 | 主要发现 |
|----------|----------|----------|
| 广义线性预测器（Swish 激活） | 杠杆分数 vs 均匀采样 | 非线性杠杆分数显著优于均匀采样，接近全数据性能 |
| ReLU 两层神经网络 | 范数分数 vs 均匀采样 | 用 10-20% 数据即可达到近似全数据的训练损失 |
| 异常检测 | 非线性分数排序 | 高分数数据点与已知异常高度吻合 |
| 模型可解释性 | 分数可视化 | 非线性分数比线性分数更精确地识别"关键"样本 |

核心实验结论：
- 非线性重要性分数在采样效率上优于线性近似分数和均匀采样
- 随着采样量增加，子集训练损失快速收敛到全数据损失
- 该框架可作为后验诊断工具识别重要样本和异常点

## 亮点与洞察

1. **理论突破**：首次将 $C=1$ 的近似保证（$\mathcal{L}(\theta_\mathcal{S}^\star) \leq \mathcal{L}(\theta^\star) + \mathcal{O}(\varepsilon)$）扩展到神经网络等非线性模型，前人工作中 $C > 1000$
2. **优雅的数学工具**：伴随算子的引入类似于 Riesz 表示定理的非线性推广，将非线性函数表示为"类内积"形式，从而桥接了线性子空间嵌入理论
3. **闭式计算**：对于具有正齐次内层的复合函数（覆盖大量常见模型），伴随算子可直接计算而无需数值积分
4. **双重应用**：既可用于训练加速（前向采样），也可用于后验诊断（模型可解释性、异常检测）
5. **统一框架**：范数分数和杠杆分数在统一的伴随算子框架下自然推导，且在线性模型时退化为经典定义

## 局限与展望

1. **欠参数化假设**：理论要求 $n \geq p$（数据量 ≥ 参数量），不适用于当前主流的过参数化深度学习场景
2. **激活函数限制**：有界性条件 $l \leq \phi^2(t)/t^2 \leq u$ 对 ReLU 不直接满足（$l=0$），需要额外假设有界参数域
3. **多层网络扩展困难**：目前主要分析两层网络，深层网络的伴随算子结构和理论保证仍待探索
4. **计算开销**：杠杆分数计算本身需要 $\mathcal{O}(nd^2)$，对超大规模数据仍有负担；论文未给出快速近似方案
5. **Lipschitz 常数未知**：参数无关保证中的误差项依赖于 Lipschitz 常数 $L(f, \mathbf{X}, R)$，实际中难以估计
6. **实验规模偏小**：缺少大规模深度学习任务（如 ImageNet、LLM 训练）的验证

## 相关工作与启发

- **线性重要性采样**: Drineas et al. (2006), Woodruff et al. (2014) — 本文的理论基础
- **非线性采样先驱**: Gajjar et al. (2023, 2024) — 单神经元模型的杠杆分数采样（$C \gg 1$）
- **Coreset**: Langberg & Schulman (2010), Feldman (2020) — 灵敏度采样框架
- **数据选择与课程学习**: 本框架可能启发更理论驱动的数据选择策略
- **模型剪枝/蒸馏**: 非线性重要性分数可指导识别"信息丰富"的样本子集

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 伴随算子概念是全新的理论工具，将线性与非线性重要性采样统一
- 实验充分度: ⭐⭐⭐ — 理论验证充分但缺少大规模实际任务的实验
- 写作质量: ⭐⭐⭐⭐ — 数学推导严谨清晰，记号系统完整
- 价值: ⭐⭐⭐⭐ — 为非线性模型的数据采样提供了坚实的理论基础，有广泛启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Tree-Sliced Wasserstein Distance with Nonlinear Projection](tree-sliced_wasserstein_distance_with_nonlinear_projection.md)
- [\[ICML 2025\] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models](morse_dual-sampling_for_lossless_acceleration_of_diffusion_models.md)
- [\[ICML 2026\] Efficient Learning of Deep State Space Models via Importance Smoothing](../../ICML2026/image_generation/efficient_learning_of_deep_state_space_models_via_importance_smoothing.md)
- [\[ICML 2025\] Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions](annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod.md)
- [\[CVPR 2026\] Nonlinear Color Transfer via Learnable Bezier Flows](../../CVPR2026/image_generation/nonlinear_color_transfer_via_learnable_bezier_flows.md)

</div>

<!-- RELATED:END -->
