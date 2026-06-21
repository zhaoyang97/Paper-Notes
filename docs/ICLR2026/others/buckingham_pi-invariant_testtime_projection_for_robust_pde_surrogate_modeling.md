---
title: >-
  [论文解读] Buckingham $\pi$-Invariant Test-Time Projection for Robust PDE Surrogate Modeling
description: >-
  [ICLR 2026][偏微分方程] 利用 Buckingham π 定理把"不同单位/尺度造成的 OOD 偏移"识别为物理等价的尺度变换，提出一种**免训练、模型无关**的测试时投影：在 log 空间内沿保 π 等价类把测试样本平移到最近的训练等价类，使 FNO/U-Net 等代理模型在极端 OOD 下 MAE 最多降低约 91%。
tags:
  - "ICLR 2026"
  - "偏微分方程"
  - "Buckingham π 定理"
  - "量纲分析"
  - "测试时投影"
  - "OOD 泛化"
  - "神经算子"
---

# Buckingham $\pi$-Invariant Test-Time Projection for Robust PDE Surrogate Modeling

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2FFhwssQda](https://openreview.net/forum?id=2FFhwssQda)  
**代码**: 待确认  
**领域**: AI for Science / PDE 代理模型  
**关键词**: PDE 代理模型, Buckingham π 定理, 量纲分析, 测试时投影, OOD 泛化, 神经算子  

## 一句话总结
利用 Buckingham π 定理把"不同单位/尺度造成的 OOD 偏移"识别为物理等价的尺度变换，提出一种**免训练、模型无关**的测试时投影：在 log 空间内沿保 π 等价类把测试样本平移到最近的训练等价类，使 FNO/U-Net 等代理模型在极端 OOD 下 MAE 最多降低约 91%。

## 研究背景与动机
- **领域现状**：FNO、PINN 等 PDE 代理模型在分布内插值上很强，但一旦测试输入的物理单位和数值尺度与训练集差异巨大（如热源 $q$ 跨 5 个数量级），预测精度就崩溃，OOD 泛化是物理机器学习的核心难题。
- **现有痛点**：①**降维学习量纲群**（dimensionless learning）多停留在标量场景，对 2D/3D 空间场直接做逐像素 Buckingham-π 缩放会失败——分子为零会把大量输入塌缩到 0，分母近零会让 π 发散，数值不稳定；②**测试时训练 / 自适应**（TTT/TTA）需要在推理时反向更新模型，带来优化开销和延迟，且很少针对回归与空间场设计。
- **核心矛盾**：很多被当作"分布漂移"的 OOD，**本质上只是保持 π 群不变的尺度变化**（窄管与大河只要雷诺数相同就动力学等价），却被代理模型当成全新分布而失效——是表示问题，不是真分布问题。
- **本文目标**：在不重训模型、不改架构的前提下，把 OOD 测试输入"对齐"到训练分布附近，同时严格保持其 Buckingham π 不变量，从而恢复 OOD 精度并控制开销。
- **核心 idea**：**【保 π 测试时投影】** 把量纲缩放看成 log 空间的平移，输入的物理内容只活在 π 值里；于是在样本自身的 π 等价类（仿射子空间）内移动，投影到最近的训练 π 等价类上，用一个微小的 log 空间最小二乘问题完成对齐。

## 方法详解

### 整体框架
方法是接在任意已训练代理模型前的**纯推理预处理步骤**，分三阶段：①**Domain Profile Reduction**——把高维空间场用算术均值压成少量代表性变量，规避逐像素 π 退化；②**保 π 测试时投影**——在 log 空间把测试样本沿保 π 方向投到最近训练类，求出缩放系数 $\exp(v^*)$；③**离线 π-uniform + 质心约简**——用主导参数把训练集 π 分布均匀化并做 K-means，把投影复杂度从 $O(MN)$ 降到 $O(KN)$。预测后再按缩放系数做逆变换还原物理量。

```mermaid
flowchart LR
    A[测试场 X̃] --> B[Domain Profile Reduction<br/>场→均值代表变量 x̃]
    B --> C[log 空间分解<br/>保π分量∥ / 改π分量⊥]
    C --> D[找最近训练等价类 i*<br/>O(KN) 质心约简]
    D --> E[缩放系数 exp v*<br/>逐通道缩放 X̃*=X̃⊙exp v*]
    E --> F[代理模型<br/>CNN/U-Net/FNO]
    F --> G[逆缩放还原 → 预测解]
    H[(训练集)] -.π-uniform采样+K-means.-> D
```

### 关键设计

**1. Domain Profile Reduction：用全局统计量替代逐像素 π，根治退化。** 对空间场直接做 Buckingham-π 会遇到致命问题——若热源场 $q$ 或体力场 $f$ 在某些位置为零或近零，逐像素 π 值会塌缩或发散，作者称之为"π-information loss"。解法是引入特征提取器 $\psi: X \mapsto x$，把每个离散场映射成有限个特征变量，并选用**场值的算术均值** $\bar k, \bar q, \bar E, \bar f$ 作为代表标量。这一全局统计量天然对局部零值与离群点鲁棒，保证代表尺度非零，从而后续的 log-线性投影数值稳定；非空间输入则直接取用。这是把 π 方法从标量推广到 2D 空间场的关键前置。

**2. log 空间保 π 投影：把对齐拆成"保 π 平移 + 改 π 物理差"两个正交分量。** 由 Buckingham π 定理的 log 形式，$\log \Pi(x)=\Phi^\top \log x$，其中 $\Phi$ 张成量纲矩阵的零空间。componentwise 缩放 $x\mapsto x\odot\exp(v)$ 在 log 空间就是平移 $z\mapsto z+v$；当 $v\in\ker(\Phi^\top)$ 时 π 值不变，于是每个输入 $z$ 生成一个仿射 π 等价类 $z+\ker(\Phi^\top)$。优化目标即在测试等价类内找到离训练集最近的点：

$$\tilde x^* = \arg\min_{\tilde x' \in [\psi(\tilde X)]_\pi}\ \mathrm{dist}\big(\tilde x', \{\psi(X_i)\}_{i=1}^M\big)$$

记 $v_i^t = z_i - \tilde z$，用投影算子 $P_\parallel = I - \Phi(\Phi^\top\Phi)^{-1}\Phi^\top$ 把它分解为保 π 的 intra-class 分量 $P_\parallel v_i^t$（尺度变化）与改 π 的 inter-class 分量 $(I-P_\parallel)v_i^t$（真实物理差异）。取 $v_i^*=P_\parallel v_i^t$ 即得测试类与第 $i$ 个训练类之间的商距离，最优训练类为 $i^*=\arg\min_i \|v_i^t - v_i^*\|^2$，最终缩放后的输入 $\tilde x^* = \tilde x\odot\exp(v^*)$。整个过程只是一个 log 空间的带约束最小二乘，开销极小。

**3. π-uniform 策略：调主导参数把训练 π 分布均匀化，扩大覆盖。** 投影只有在训练集 π 覆盖足够宽时才有效，但原始训练 π 分布往往单侧偏斜。作者用 SHAP 分析找出**主导参数**——其尺度最直接控制 π 群（热问题中 $q$ 贡献度高达 48.7%）；然后固定其他输入，只调主导参数使每个训练样本的 π 值匹配一个目标均匀分布。注意均匀化的是 $\log\pi$ 而非 $\pi$，以最大化覆盖范围。配合保 π 缩放约束（热：$\log\beta+2\log\delta-\log\alpha-\log\gamma=0$），训练空间得以覆盖更宽的 π 区间。

**4. 质心约简：K-means 代表点把投影从 $O(MN)$ 降到 $O(KN)$。** 朴素地把 $N$ 个测试样本与全部 $M$ 个训练样本两两比较需 $O(MN)$。作者在均匀化后的 log 特征上跑 K-means，只保留 $K\in\{1,\dots,10\}$ 个质心作为投影代表点，复杂度降为 $O(KN)$。实验显示聚类投影的 MAE 在统计噪声内与全量基线持平，而投影时间缩短约 $100\times$；当 π 病态（如 $q\approx0$ 或 $\Delta T\approx0$）时退化为在非退化通道上做同样的 log 对齐，保留空间异质性。

## 实验关键数据

### 主实验表格
2D 稳态热传导（Thermal）与线弹性（Stress）OOD 测试（训练/测试 π 范围不相交，如 $\log_{10}q$ 训练 $[0,7.5]$、测试 $[7.5,12]$）：

| Method | Thermal MAE | Thermal RMSE | Time | Stress MAE | Stress RMSE | Time |
|---|---|---|---|---|---|---|
| CNN | 8.43 | 9.99 | - | 0.96 | 1.17 | - |
| CNN + Pairwise Proj. | 2.63 | 3.24 | 100.3 | 0.53 | 0.71 | 73.6 |
| CNN + π-uniform + 10-Centroids | **1.79** | **2.23** | 1.80 | **0.60** | **0.84** | 1.36 |
| U-Net | 13.60 | 15.29 | - | 0.81 | 0.99 | - |
| U-Net + Pairwise Proj. | 1.75 | 2.31 | 99.1 | 0.17 | 0.28 | 94.9 |
| U-Net + π-uniform + 10-Centroids | **1.18** | **1.53** | 2.31 | **0.22** | **0.39** | 1.58 |
| FNO | 9.88 | 11.43 | - | 3.20 | 4.19 | - |
| FNO + Pairwise Proj. | 1.38 | 1.74 | 151.4 | 0.28 | 0.42 | 94.0 |
| FNO + π-uniform + 10-Centroids | **1.25** | **1.60** | 2.31 | **0.33** | **0.53** | 1.54 |

U-Net 热问题 MAE 从 13.60 降到 1.18（约 **91%** 降幅），FNO 热问题从 9.88 降到 1.25。

### 消融实验表格
质心约简的候选数对比（Baseline=全量最近 / Clustered=K 质心 / Random=K 随机样本）：

| 投影方式 | MAE | 投影时间 |
|---|---|---|
| Pairwise（全量基线） | 最低参考 | $O(MN)$，~100–150s |
| K-Centroids（聚类） | 与基线在统计噪声内持平 | ~$100\times$ 加速 |
| K-Randoms（随机） | 明显高于聚类 | 同量级 |

随候选数增加，聚类投影 MAE 收敛到全量基线，而随机投影始终留有差距，验证质心代表性。

### 关键发现
- **改善集中在最差 OOD 案例**：Top-3 worst 案例提升最大，正是原始代理模型完全无法外推的区域。
- **几乎零额外训练**：方法免训练、模型无关，对 CNN/U-Net/FNO 三种架构一致有效。
- **精度-速度兼得**：质心约简把投影时间从约 100s 压到 ~1.8s，精度几乎无损。

## 亮点与洞察
- **重新定义 OOD**：把"单位/尺度差异导致的 OOD"重新诠释为保 π 的尺度等价，OOD 从"需要模型适应的难题"变成"可在输入端解析校正的表示问题"，视角新颖且物理可解释。
- **几何化的优雅**：intra-class（保 π 尺度）⊥ inter-class（改 π 物理差）的正交分解，把抽象的量纲商空间落地为一个可计算的投影算子，理论与工程衔接干净。
- **真正即插即用**：作为推理前置步骤接在任意代理模型前，不碰训练、不改架构，落地成本极低。

## 局限与展望
- **假设强**：要求系统由 PDE 主导，含经验/混合成分的系统可能失效。
- **代表统计量损信息**：用算术均值概括场会模糊高度不规则的空间模式。
- **极端 OOD 仍退化**：超出训练 π 范围的极端样本，即便 π-uniform 扩容也会掉点。
- **任务范围有限**：仅验证了稳态热传导与线弹性两类静态 PDE，作者展望扩展到瞬态/对流 PDE（对流-扩散、Navier-Stokes）及不确定性感知投影。

## 相关工作与启发
- **量纲分析学习**（Bakarji 2022、Xie 2022、SINDy 系列）：本文把标量 π 学习推广到 2D 空间场，并系统处理了零集塌缩与分母爆炸两大失败模式。
- **神经算子**（FNO 及变体、PINO）：本文不与之竞争，而是作为其 OOD 鲁棒性的免训练增强层。
- **测试时训练/自适应**（TTT、Tent 等）：本文提供了一条无需反向优化的样本级、保 π 对齐替代路径，速度更优。
- **启发**：当 OOD 的"漂移"其实带有已知物理对称性（量纲、伸缩、平移群）时，与其让模型去学习不变性，不如在输入端用解析变换显式消除——这一思路可迁移到任何具备已知群作用的科学计算任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把 Buckingham-π 不变量与测试时投影结合，并解决空间场 π 退化问题，视角与切入点都新。
- **实验充分度**: ⭐⭐⭐ 三种架构、两类 PDE、含质心约简消融，论证扎实；但 PDE 类型仅静态两例，缺瞬态/对流验证。
- **写作质量**: ⭐⭐⭐⭐ 理论推导（log 形式、正交分解）清晰，图示与流程图到位。
- **价值**: ⭐⭐⭐⭐ 免训练、模型无关、即插即用，对 AI4Science 代理模型的 OOD 部署有实际意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](prior-free_tabular_test-time_adaptation.md)
- [\[ICLR 2026\] PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference](priorguide_test-time_prior_adaptation_for_simulation-based_inference.md)
- [\[ICLR 2026\] Exposing Mixture and Annotating Confusion for Active Universal Test-Time Adaptation](exposing_mixture_and_annotating_confusion_for_active_universal_test-time_adaptat.md)
- [\[ECCV 2024\] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory](../../ECCV2024/others/membn_robust_test-time_adaptation_via_batch_norm_with_statistics_memory.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
