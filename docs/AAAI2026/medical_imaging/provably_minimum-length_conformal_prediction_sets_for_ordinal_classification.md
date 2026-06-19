---
title: >-
  [论文解读] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification
description: >-
  [AAAI 2026][医学图像][保形预测] 提出 min-CPS 及其正则化变体 min-RCPS，一种模型无关的序数保形预测方法，通过线性时间滑动窗口算法求解每个样本的最小长度预测区间，在保证覆盖率的同时平均减少 15% 的预测集大小，且提供了实例级最优性的理论保证。 领域现状：序数分类（Ordinal Classif…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "保形预测"
  - "序数分类"
  - "预测集大小"
  - "不确定性量化"
  - "滑动窗口算法"
  - "模型无关"
---

# Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification

**会议**: AAAI 2026  
**arXiv**: [2511.16845](https://arxiv.org/abs/2511.16845)  
**代码**: [github.com/xrty/OCP](https://github.com/xrty/OCP)  
**领域**: 医学图像 / 不确定性量化 / 序数分类  
**关键词**: 保形预测, 序数分类, 预测集大小, 不确定性量化, 滑动窗口算法, 模型无关

## 一句话总结
提出 min-CPS 及其正则化变体 min-RCPS，一种模型无关的序数保形预测方法，通过线性时间滑动窗口算法求解每个样本的最小长度预测区间，在保证覆盖率的同时平均减少 15% 的预测集大小，且提供了实例级最优性的理论保证。

## 研究背景与动机

**领域现状**：序数分类（Ordinal Classification）在高风险应用中广泛使用（医学诊断分级、信用评估、年龄估计），标签之间有天然顺序但间距未知。保形预测（Conformal Prediction, CP）是通用的不确定性量化框架，提供分布无关的覆盖率保证。

**现有序数 CP 方法的问题**：
   - **Ordinal APS**（Lu et al. 2022）：从最高概率标签出发贪心扩展到相邻标签，但这种贪心搜索是**启发式的**，没有最优性保证，可能产生不必要大的预测区间
   - **COPOC**（Dey et al. 2023）：强制要求模型输出单峰分布，通过学习辅助模块实现，破坏了 CP 的**模型无关性**
   - 都没有从预测效率（prediction efficiency）的角度提供理论分析

**核心矛盾**：已有方法要么是启发式（无最优性保证），要么依赖模型假设（非模型无关），且都未深入分析覆盖率-效率权衡。

**切入角度**：将序数 CP 形式化为**实例级最小长度覆盖问题**，设计可证明最优的算法。

## 方法详解

### 问题设定

- 输入 $X \in \mathcal{X}$，序数标签 $Y \in \{1, 2, \ldots, K\}$
- 模型 $f(X)$ 输出各类概率分布
- 目标：构造连续区间 $\hat{C}_\tau(X) = \{y: l(X;\tau) \le y \le u(X;\tau)\}$ 满足边际覆盖率 $\mathbb{P}\{Y \in \hat{C}_\tau(X)\} \ge 1-\alpha$

### 实例级最小长度覆盖（Instance-Level Minimum-Length Covering）

对任意输入 $X$ 和阈值 $\tau$，求解：

$$\min_{(l,u) \in \mathcal{U}(X;\tau)} \ell(l,u) \triangleq u - l$$

约束：
1. **覆盖概率**：$\sum_{k=l}^{u} f(X)_k \ge \tau$
2. **包含锚点**：$l \le \hat{y}^*(X) \le u$（$\hat{y}^*$ 是模型预测的最高概率标签/众数）

### 滑动窗口算法（Algorithm 1）

**关键设计**：
1. 预计算前缀和 $P_k = \sum_{i=1}^k f(X)_i$，使任意区间概率查询 $O(1)$
2. 外层循环：$u$ 从 $\hat{y}^*$ 单调增到 $K$
3. 内层循环：$l$ 从小到大移动，只要覆盖概率 $P_u - P_{l-1} \ge \tau$ 且 $l \le \hat{y}^*$
4. 记录满足约束的最短区间

**理论保证（Theorem 1）**：
- Algorithm 1 精确求解最小长度覆盖问题
- 时间复杂度 $O(K)$（线性！vs 暴力搜索 $O(K^2)$）
- 模型无关：对任意预测分布成立（无需单峰/任何分布假设）

### min-CPS：校准阈值 $\tau$

通过二分搜索确定 $\tau$ 使经验覆盖率 $F(\tau) \ge 1-\alpha$：

$$F(\tau) = \frac{1}{n} \sum_{i=1}^n \mathbb{1}[l^*(X_i;\tau) \le Y_i \le u^*(X_i;\tau)]$$

**Lemma 1**：如果 $f(X)$ 满足**径向单调性**（radial monotonicity，即概率从众数向两侧递减），则 $F(\tau)$ 关于 $\tau$ 单调非递减，保证二分搜索收敛。

**Theorem 2**：在径向单调性条件下，min-CPS 的校准阈值提供 $(1-\alpha)$ 边际覆盖率保证。

总时间复杂度：$O(\log(1/\epsilon) \cdot n \cdot K)$。

### min-RCPS：长度正则化变体

动机：预测区间越大的样本不确定性越高，应通过惩罚减少跳变。

修改可行集约束：
$$\sum_{k=l}^u f(X)_k - \lambda \cdot \ell(l,u) \ge \tau$$

引入 $\lambda$ 对区间长度的线性惩罚，使大区间需要更高的概率质量才能满足约束。当 $\lambda=0$ 退化为 min-CPS。

**Corollary 1**：min-RCPS 保持可交换性，仍享有标准 CP 覆盖率保证。

## 实验

### 数据集（4个，跨多种模态）

| 数据集 | 类型 | 类别数 | 描述 |
|--------|------|--------|------|
| UTKFace | 图像 | ~100 | 面部年龄估计 |
| Avocado Price | 表格 | ~50 | 牛油果价格分级 |
| Electric Motor Temp. | 时序 | ~80 | 电机温度监测 |
| IMDB | 图像 | ~100 | 面部年龄估计（大规模） |

### 基线
- Naive CDF：直接从累积概率构造区间
- Ordinal APS：序数分类的自适应预测集
- WCRC（Weighted CRC）：加权保形风险控制

### 主实验结果（$\alpha=0.1$）

| 数据集 | min-CPS 集大小↓ | min-RCPS 集大小↓ | vs Ordinal APS |
|--------|----------------|-----------------|----------------|
| Temperature | 5.09 | 4.99 | -5.11% / -6.96% |
| UTKFace | 29.77 | 29.77 | -7.00% / -7.01% |
| Avocado Price | 9.12 | 8.92 | **-39.93%** / **-41.21%** |
| IMDB | 28.18 | 28.26 | -3.49% / -3.16% |
| **平均** | — | — | **-14%** / **-15%** |

所有方法的覆盖率均 $\ge 1-\alpha$，验证了理论保证。

### 不同 $\alpha$ 值的敏感性分析（Avocado Price）

| $\alpha$ | min-CPS 减小 | min-RCPS 减小 |
|----------|--------------|---------------|
| 0.10 | -39.93% | -41.21% |
| 0.05 | -36.22% | -35.58% |
| 0.01 | -24.80% | -26.38% |

平均跨所有 $\alpha$：min-CPS -33.65%，min-RCPS -34.39%。

### 关键发现

1. **在 Avocado Price 上提升最大**（~40%），因为该数据集标签分布不规则，贪心方法生成大量不必要的大区间
2. **empirical monotonicity 成立**：在所有实验中验证了 $F(\tau)$ 的单调性（Figure 2），支持理论假设
3. **min-RCPS 在大部分场景下略优于 min-CPS**，$\lambda$ 的正则化对不确定样本的区间压缩有效

## 亮点与洞察

1. **从启发式到可证明最优**：首次为序数 CP 的预测效率提供实例级最优性保证
2. **线性时间算法**：滑动窗口将 $O(K^2)$ 降到 $O(K)$，对类别数多的任务（如年龄 K=100+）至关重要
3. **完全模型无关**：不要求单峰/任何分布假设（不同于 COPOC），真正保持了 CP 的分布无关优势
4. **正则化变体 min-RCPS 的直觉优雅**：大区间=高不确定性→需要更高概率才能满足覆盖→自然压缩
5. **理论+实验闭环**：Theorem 1（最优性）→ Lemma 1（单调性条件）→ Theorem 2（覆盖保证）→ 实验验证单调性

## 局限性

1. 覆盖率保证依赖**径向单调性假设**（Lemma 1），虽然实验中均成立，但对严重多峰分布可能不满足
2. 仅保证边际覆盖率，不保证条件覆盖率（class-conditional coverage）
3. min-RCPS 的 $\lambda$ 需要在验证集上调参，增加了超参敏感性
4. 实验中的基础模型训练细节未详述，无法判断 min-CPS 对弱模型的表现
5. 未在真实医学诊断应用中验证（尽管标题提到医学影像，实验主要用面部年龄和商品价格）

## 相关工作

- **保形预测**：标准 CP（Vovk 2005）、APS（Romano 2020）、RAPS、SAPS、RC3P
- **序数 CP**：Ordinal APS（Lu 2022，贪心启发式）、COPOC（Dey 2023，要求单峰）、WCRC（Xu 2023，加权风险控制）
- **序数分类**：面部年龄估计、糖尿病视网膜分级、美学评估

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐⭐ — 将序数 CP 形式化为最小长度覆盖问题并给出可证明最优解，理论贡献突出
- **实验**：⭐⭐⭐⭐ — 4个数据集、多个 $\alpha$ 值，提升显著且一致
- **写作**：⭐⭐⭐⭐⭐ — 定理-引理-推论的逻辑链清晰，算法伪代码完整
- **实用性**：⭐⭐⭐⭐ — 线性时间即插即用，适用于任何序数分类模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](../../ICML2026/medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[AAAI 2026\] Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction](cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)
- [\[AAAI 2026\] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)
- [\[CVPR 2026\] LATA: Laplacian-Assisted Transductive Adaptation for Conformal Uncertainty in Medical VLMs](../../CVPR2026/medical_imaging/lata_laplacian-assisted_transductive_adaptation_for_conformal_uncertainty_in_med.md)

</div>

<!-- RELATED:END -->
