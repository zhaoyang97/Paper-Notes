---
title: >-
  [论文解读] Improving the Straight-Through Estimator with Zeroth-Order Information
description: >-
  [NeurIPS 2025][优化/理论][量化] 本文提出 FOGZO（First-Order-Guided Zeroth-Order Gradient Descent），将 STE 梯度作为偏置源注入零阶梯度估计中，在保留 STE 的计算效率的同时利用零阶信息纠正 STE 的偶发错误方向，仅多 2 次前向传播即在 DeiT、ResNet、LLaMA 上实现 1-22 点的精度/困惑度改善。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "量化"
  - "straight-through estimator"
  - "zeroth-order optimization"
  - "FOGZO"
  - "gradient estimation"
---

# Improving the Straight-Through Estimator with Zeroth-Order Information

**会议**: NeurIPS 2025  
**arXiv**: [2510.23926](https://arxiv.org/abs/2510.23926)  
**代码**: [GitHub](https://github.com/1733116199/fogzo)  
**领域**: 优化  
**关键词**: quantization-aware training, straight-through estimator, zeroth-order optimization, FOGZO, gradient estimation

## 一句话总结
本文提出 FOGZO（First-Order-Guided Zeroth-Order Gradient Descent），将 STE 梯度作为偏置源注入零阶梯度估计中，在保留 STE 的计算效率的同时利用零阶信息纠正 STE 的偶发错误方向，仅多 2 次前向传播即在 DeiT、ResNet、LLaMA 上实现 1-22 点的精度/困惑度改善。

## 研究背景与动机
**领域现状**：量化感知训练（QAT）是获得低比特模型的有效途径，其核心挑战是 round/sign 函数几乎处处梯度为零。Straight-Through Estimator（STE）用平滑函数的 Jacobian 替代不可微运算的 Jacobian，是 QAT 的事实标准方法。

**STE 的问题**：STE 在高精度时效果好，但在低精度（1-2 bit）时引入参数振荡，偶尔产生错误方向的梯度。尽管 STE 理论根基薄弱，13 年来一直是最主流的方法。

**零阶方法的局限**：n-SPSA 等零阶方法虽然理论上更sound（基于随机平滑），但需要 $2n$ 次前向传播，对深度网络极不实际。小 $n$ 时梯度方差爆炸，导致收敛很慢。

**核心思路**：STE 是一个"足够好但偶尔犯错"的梯度估计器。如果能用零阶信息纠正 STE 的这些错误，就能以接近 STE 的计算量超越 STE 的精度。

## 方法详解

### FOGZO 算法

**核心公式**：构造混合扰动向量：

$$v_i = \sqrt{\beta} \cdot s_i \hat{g} + \sqrt{1-\beta} \cdot u_i$$

其中 $\hat{g} = g/\|g\|$ 是归一化 STE 梯度，$s_i \sim 2 \cdot \text{Ber}(0.5) - 1$ 保证零均值对称性，$u_i \sim p(u)$ 是无偏随机扰动，$\beta$ 控制对 STE 的信任度。

**梯度估计**：

$$G = \frac{1}{n} \sum_{i=1}^n \frac{\hat{L}(\theta + \epsilon v_i) - \hat{L}(\theta - \epsilon v_i)}{2\epsilon} v_i$$

### 启发式推导
通过一阶 Taylor 展开和零均值性质，$\mathbb{E}[G]$ 近似为：

$$\mathbb{E}[G] \approx \beta \underbrace{\hat{g}\hat{g}^\top \nabla \hat{L}_{\text{smooth}}}_{\text{biased}} + (1-\beta) \underbrace{\nabla \hat{L}_{\text{smooth}}}_{\text{unbiased}}$$

- 当 STE 正确时（$\hat{g}$ 与 $\nabla \hat{L}_{\text{smooth}}$ 对齐），$\hat{g}^\top \nabla \hat{L}_{\text{smooth}}$ 大，biased 项贡献显著
- 当 STE 错误时（$\hat{g}$ 与 $\nabla \hat{L}_{\text{smooth}}$ 正交），biased 项被自然抑制（标量 $\hat{g}^\top \nabla \hat{L}_{\text{smooth}} \approx 0$）

### 超参数选择：从 STE 到隐式平滑

**关键洞察**：每个 STE 隐式定义了一种平滑。STE 将不可微算子 $h(x)$ 的 Jacobian 替换为平滑代理 $h_{\text{smooth}}(x)$ 的 Jacobian，而代理可视为原算子在某扰动下的期望：

$$h_{\text{smooth}}(x) = \mathbb{E}_{u \sim \bar{p}(u)}[h(x + \bar{\epsilon} u)]$$

通过反解该方程获得 $(\bar{\epsilon}, \bar{p}(u))$：

| STE 类型 | $\bar{\epsilon}$ | $\bar{p}(u)$ |
|----------|------------------|---------------|
| Identity (round) | $1/(2\sqrt{3})$ | $U(-\sqrt{3}, \sqrt{3})$ |
| Hardtanh (sign) | $1/\sqrt{3}$ | $U(-\sqrt{3}, \sqrt{3})$ |
| Tanh (sign) | $\pi/\sqrt{12}$ | $\bar{\epsilon}(1-\tanh^2(\bar{\epsilon}u))/2$ |
| ApproxSign (sign) | $1/\sqrt{6}$ | $\text{tri}(u/\sqrt{6})/\sqrt{6}$ |

实际设定中 $\epsilon = \alpha \bar{\epsilon}$（$\alpha$ 为量化尺度）。

### $\beta$ 调度策略
- 训练初期 $\beta = 1$（完全信任 STE）
- 线性衰减到 $\beta_{\min}$（逐步引入零阶修正）
- 后期学习率小，能容忍更大的梯度方差

## 实验关键数据

### 浅层网络实验（2-layer MLP, MNIST, 2-bit）

| 方法 | $n$ | 相对计算量 | 训练损失 |
|------|-----|-----------|---------|
| Identity STE | - | 1× | baseline |
| n-SPSA ($n=1$) | 1 | 3× | 显著差于 STE |
| n-SPSA ($n=7960$) | 7960 | 15920× | 略优于 STE |
| FOGZO ($\beta=0.999, n=1$) | 1 | 3× | 优于 STE |

**核心发现**：FOGZO 在 $n=1$ 时即可超越 STE，相比 n-SPSA 达到同等性能节省 **796×** 计算量。

### 深度网络实验（固定 $\alpha$, 各种 STE）

| 模型 | 数据集 | Identity-STE | Identity-FOGZO | tanh-STE | tanh-FOGZO |
|------|--------|-------------|---------------|---------|-----------|
| DeiT-Tiny | ImageNet-100 | 62.72% | **70.06%** (+7.3%) | 41.98% | **46.8%** |
| LLaMA-9m | C4 (ppl) | 109.95 | **105.64** | 123.97 | **121.51** |
| ResNet-18 | ImageNet-100 | 79.92% | **80.42%** | 74.68% | **75.02%** |

### 集成 SOTA 方法（LSQ + FOGZO, 2-bit 权值）

| 模型 | 数据集 | LSQ+STE (loss/acc) | LSQ+FOGZO (loss/acc) |
|------|--------|-------------------|---------------------|
| DeiT-Small | ImageNet-100 | 2.62 / 79.55% | **2.57 / 80.06%** |
| LLaMA-20m | 13B C4 tokens (ppl) | 50.85 | **50.61** |
| ResNet-50 | ImageNet-100 | 0.43 / 82.81% | **0.39 / 83.67%** |

### 权值-激活量化（QuEST/LSQ + FOGZO, 2-bit W+A）

| 模型规模 | QuEST (ppl) | QuEST-FOGZO (ppl) | LSQ (ppl) | LSQ-FOGZO (ppl) |
|----------|------------|-------------------|----------|----------------|
| 95M | 37.75 | **37.37** | 39.06 | **37.38** |
| 200M | 26.63 | **26.45** | - | - |
| 300M | 22.90 | **22.72** | - | - |

### 训练时间对比（LLaMA-30M, RTX 5090）

| 方法 | C4 tokens | 困惑度 | 训练时间 |
|------|-----------|-------|---------|
| STE | 3.522B | 38.25 | 3.7h |
| 70% STE + 30% FOGZO | 3.0B | **37.93** | 3.7h |

**注**：相同训练时间下 FOGZO 用更少数据达到更低困惑度，说明数据效率更高。

## 亮点与洞察
- **极简有效**：仅额外 2 次前向传播（$n=1$），实现上就是在标准 backward 后加一步有限差分，无需修改优化器
- **STE 隐式平滑的优美联系**：将 STE 代理函数反解为随机平滑，为零阶方法的 $\epsilon$ 和 $p(u)$ 选择提供了原则性指导
- **自适应抑制机制**：当 STE 给出错误方向时，有限差分自然将其贡献压缩为零；正确时则保留——无需额外检测机制
- **广泛适用**：与 Identity/tanh/ApproxSign 等各种 STE、LSQ/QuEST 等 SOTA 量化方法均可组合

## 局限与展望
- **理论保证较弱**：主要依赖启发式推导（一阶 Taylor + 零均值假设），缺乏收敛性严格证明
- **$\beta_{\min}$ 需要调**：虽然接近 1 的范围较窄，但仍需要少量搜索
- **仅测试到 300M 参数**：是否在 billion 级模型上仍有收益需验证
- **额外的内存/计算开销**：虽然 $n=1$ 时开销可控（约多 60-70% 训练时间），但在极大规模训练中可能仍不可忽略
- **作者用"r% STE + (100-r)% FOGZO"缓解开销**：但最优 $r$ 值也是超参数

## 评分
- 新颖性: ⭐⭐⭐⭐ STE + 零阶混合的思路新颖，STE 隐式平滑解析是加分项
- 理论深度: ⭐⭐⭐ 主要是启发式推导，缺乏严格收敛性分析
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 MLP/CNN/ViT/LLM，多种量化方法，多种 STE，训练时间对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，推导过程易跟随，实验组织良好
- 实用价值: ⭐⭐⭐⭐⭐ 即插即用，对低比特量化训练有直接的实际意义

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](private_zeroth-order_optimization_with_public_data.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](../../ICML2026/optimization/learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](../../ICML2026/optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)

</div>

<!-- RELATED:END -->
