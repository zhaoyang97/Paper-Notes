---
title: >-
  [论文解读] Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions
description: >-
  [NeurIPS 2025][时间序列][签名核] 提出 PowerSig，通过自适应截断的局部 Neumann 级数展开高效计算签名核（signature kernel），将内存从 $O(\ell^2)$ 降到 $O(\ell P)$，使签名核可扩展到单GPU上百万级长度的时间序列。 领域现状： 签名核（Signature…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "签名核"
  - "Neumann级数"
  - "长时间序列"
  - "偏微分方程"
  - "核方法"
---

# Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions

**会议**: NeurIPS 2025  
**arXiv**: [2502.20392](https://arxiv.org/abs/2502.20392)  
**代码**: [https://github.com/geekbeast/powersig](https://github.com/geekbeast/powersig)  
**领域**: 时间序列  
**关键词**: 签名核, Neumann级数, 长时间序列, Goursat PDE, 核方法

## 一句话总结

提出 PowerSig，通过自适应截断的局部 Neumann 级数展开高效计算签名核（signature kernel），将内存从 $O(\ell^2)$ 降到 $O(\ell P)$，使签名核可扩展到单GPU上百万级长度的时间序列。

## 研究背景与动机

**领域现状**: 签名核（Signature Kernel）是分析高维序列数据的前沿工具，根植于粗糙路径理论，具有重参数化不变性、全局特征性和噪声鲁棒性等优良性质。已被广泛应用于金融建模、信号处理等领域。

**现有痛点**: 
   - 现有计算方法要么通过动态规划计算截断签名核（$O(\ell^2)$ 内存），要么通过有限差分求解全局 Goursat PDE（同样 $O(\ell^2)$ 内存）
   - KSig 库在 RTX 4090 上最多处理 ~$16 \times 10^4$ 步的序列，sigkernel 约 $10^3$ 步
   - 对于金融高频数据、传感器长时间监测等场景，序列长度轻松超过百万，现有方法完全不可用

**核心矛盾**: 签名核理论性质极好 → 但计算上无法扩展到长序列和粗糙时间序列。

**本文目标**: 在保持精度的前提下，将签名核计算扩展到 $10^6$+ 长度的时间序列。

**切入角度**: 利用分段线性时间序列的几何结构，将 Goursat PDE 的全局求解转化为逐tile的局部 Neumann 级数展开，只需存储局部级数系数而非全局 $\ell \times \ell$ 网格。

**核心 idea**: 将签名核定义 PDE 重写为 Volterra 积分方程，在每个 tile 上做 Neumann 级数展开，用递归边界传播连接相邻 tiles，实现内存线性于序列长度的计算。

## 方法详解

### 整体框架

签名核定义为 Goursat PDE 的解：

$$\frac{\partial^2 K(s,t)}{\partial s \partial t} = \rho_{\boldsymbol{x},\boldsymbol{y}}(s,t) K(s,t), \quad K(0,\cdot) = K(\cdot,0) = 1$$

其中 $\rho_{\boldsymbol{x},\boldsymbol{y}}(s,t) = \langle \hat{\boldsymbol{x}}'(s), \hat{\boldsymbol{y}}'(t) \rangle$ 是分段常数的。

等价 Volterra 积分方程：$K(s,t) = 1 + \int_0^t \int_0^s \rho(u,v) K(u,v) \, du \, dv$

### 关键设计

#### 1. Tile 分解与局部解

**功能**: 将 $[0,1]^2$ 按时间序列数据点分为 $(\ell-1)^2$ 个 tiles $T_{k,l}$，在每个 tile 上局部求解。

**核心思路**: 在 tile $T_{k,l}$ 上，$\rho$ 是常数 $\rho_{k,l} = \Delta_k \boldsymbol{x} \cdot \Delta_l \boldsymbol{y}$。利用 Volterra 形式的 Neumann 级数：

$$\kappa_{k,l} = \sum_{n=0}^{\infty} \boldsymbol{T}_{k,l}^n \big(\kappa_{k-1,l}(\sigma_k, \cdot) + \kappa_{k,l-1}(\cdot, \tau_l) - \kappa_{k-1,l-1}(\sigma_k, \tau_l)\big)$$

每个 tile 的解只依赖左侧和下方邻居的边界值。

**设计动机**: 积分算子 $\boldsymbol{T}_\rho$ 的谱半径为零（Lemma 2.3），保证 Neumann 级数无论 $\rho$ 大小总收敛。

#### 2. Tile 中心幂级数展开（Proposition 2.8）

**功能**: 将每个 tile 的解表示为以 tile 角点为中心的幂级数。

**核心公式**: 

$$\kappa_{k,l}(s,t) = \sum_{i,j=0}^{\infty} \tilde{c}_{i,j}^{(k,l)} (s - \sigma_k)^i (t - \tau_l)^j$$

系数矩阵 $\tilde{C}^{k,l} = A_{k,l} \odot B_{k,l} \odot W$，其中：
- $A_{k,l}$ 编码局部粗糙度 $\rho_{k,l}$
- $B_{k,l}$ 编码边界条件（从相邻 tiles 递归传播）
- $W$ 是组合权重矩阵

**设计动机**: Tile 中心展开避免了原点展开中 $L_\sigma, R_\tau$ 矩阵的计算开销，显著简化递推。

#### 3. 自适应截断策略

**功能**: 根据每个 tile 上 $|\rho_{k,l}|$ 的大小自适应决定截断阶数。

**核心思路**: 截断误差衰减为 $O((n!)^{-2})$，$|\rho_{k,l}|$ 小的 tile 用低阶即可，$|\rho_{k,l}|$ 大的 tile 需更高阶。默认截断阶数 7 即可覆盖绝大多数情况。

#### 4. 并行化与 DAG 调度

同一反对角线上的 tiles（$k + l = \text{const}$）彼此独立，可并行计算。整个计算沿有向无环图（DAG）的拓扑序依次推进。

### 复杂度分析

| 方面 | 现有方法 (KSig/sigkernel) | PowerSig |
|------|-------------------------|----------|
| 时间 | $O(\ell^2 d)$ | $O(\ell^2 d)$（相同） |
| 空间 | $O(\ell^2)$ | $O(\ell P)$，$P \ll \ell$ |
| 最大序列长度 (RTX 4090) | ~$1.6 \times 10^5$ | $> 10^6$ |

## 实验关键数据

### 精度对比

| 序列长度 $\ell$ | PowerSig MAPE | KSig PDE MAPE |
|----------------|--------------|---------------|
| 9 | ~$10^{-12}$ | ~$10^{-8}$ |
| 17 | ~$10^{-11}$ | ~$10^{-5}$ |
| 65 | ~$10^{-10}$ | ~$10^{-3}$ |
| 129 | ~$10^{-10}$ | ~$10^{-2}$ |
| 513 | ~$10^{-9}$ | ~$10^{-1}$ |

PowerSig 精度高出 KSig-PDE 数个数量级。

### 粗糙序列（低 Hurst 指数）精度

| Hurst 指数 | PowerSig MAPE | KSig PDE MAPE |
|-----------|--------------|---------------|
| 0.4 | ~$10^{-10}$ | ~$10^{-2}$ |
| 0.1 | ~$10^{-9}$ | ~$10^{0}$ |
| 0.005 | ~$10^{-8}$ | ~$10^{2}$ |

Hurst 指数越低（越粗糙），PowerSig 的优势越大。

### 内存与运行时间

| 序列长度 | PowerSig 内存 | KSig 内存 |
|---------|-------------|----------|
| 513 | ~10 MB | ~400 MB |
| 4097 | ~30 MB | KSig OOM |
| 524,289 | ~720 MB | 不可行 |

### 下游任务

| 任务 | PowerSig | KSig-PDE |
|------|----------|----------|
| Bitcoin 回归 MAPE (test) | **2.81%** | 3.23% |
| Eigenworms 分类 (L=1024) | **61.1%** | OOM |

### 关键发现

1. PowerSig 在所有序列长度和粗糙度上精度均优于 PDE 和 DP 方法
2. 内存从 $O(\ell^2)$ 降到 $O(\ell P)$，使 $10^6$ 级序列在单 GPU 上可行
3. 对低 Hurst 指数（高粗糙度）序列数值稳定性显著提升
4. Bitcoin 回归任务上，更高精度的签名核带来了更好的预测性能
5. 高维扩展性好：$d$ 从 2 到 8192，运行时间近乎线性

## 亮点与洞察

- **空间复杂度突破**: $O(\ell^2) \to O(\ell P)$ 是关键贡献，使签名核走向实用
- **数学优雅**: Volterra 积分方程 → Neumann 级数 → 递归 tile 传播的推导链非常清晰
- **自适应截断**: 利用 $\rho_{k,l}$ 大小控制每个 tile 的展开深度，平衡精度与计算
- **数值稳定**: PDE 方法在粗糙路径上误差爆炸，PowerSig 利用局部展开避免了这一问题
- **谱半径为零保证**: Lemma 2.3 确保 Neumann 级数对任意 $\rho$ 值都收敛，不需要收缩映射

## 局限与展望

1. 保持了 $O(\ell^2 d)$ 的时间复杂度，未改善运行时间
2. 目前仅支持分段线性插值，高阶插值或可学习插值尚未探索
3. 核矩阵计算的 $O(N^2)$ 复杂度（$N$ 为序列数量）是核方法的通病，未解决
4. tile 间边界传播是串行的（沿反对角线并行），未充分利用硬件并行能力
5. 未提供后验误差控制（a posteriori error bound）

## 相关工作与启发

- Salvi et al. (2021) 首次将签名核表述为 Goursat PDE，PowerSig 在其基础上实现了 tile 局部化
- 与 Cass et al. (2025) 的并发工作基于不同数学前提（但也用了递归幂级数）
- 可扩展到金融高频数据分析、长期传感器监测、天文射电信号处理等需要处理极长序列的应用

## 评分

⭐⭐⭐⭐

数学推导精美，解决了签名核的关键可扩展性瓶颈。内存改善是实质性的（$O(\ell^2) \to O(\ell P)$），但时间复杂度未改善，且核方法本身的 $O(N^2)$ 问题依然存在。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Are Global Dependencies Necessary? Scalable Time Series Forecasting via Local Cross-Variate Modeling](../../ICLR2026/time_series/are_global_dependencies_necessary_scalable_time_series_forecasting_via_local_cro.md)
- [\[CVPR 2025\] L2GTX: From Local to Global Time Series Explanations](../../CVPR2025/time_series/l2gtx_from_local_to_global_time_series_explanations.md)
- [\[NeurIPS 2025\] WaLRUS: Wavelets for Long-range Representation Using SSMs](walrus_wavelets_for_long-range_representation_using_ssms.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
