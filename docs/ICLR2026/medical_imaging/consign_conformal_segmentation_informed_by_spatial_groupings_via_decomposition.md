---
title: >-
  [论文解读] CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition
description: >-
  [ICLR 2026][医学图像][Conformal Prediction] CONSIGN 用 SVD 从分割模型的多次采样里提取空间相关的"主不确定性方向"，构造出能联合改变的空间感知共形预测集，在保持统计覆盖保证的同时把预测集体积压到比逐像素方法小几个数量级。 领域现状：分割模型给出的逐像素 softmax 置信度只…
tags:
  - "ICLR 2026"
  - "医学图像"
  - "Conformal Prediction"
  - "Conformal Risk Control"
  - "图像分割"
  - "Uncertainty Quantification"
  - "SVD/PCA"
---

# CONSIGN: Conformal Segmentation Informed by Spatial Groupings via Decomposition

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pHF5CXB0YH](https://openreview.net/forum?id=pHF5CXB0YH)  
**代码**: 待确认  
**领域**: 医学图像 / 不确定性量化 / 共形预测  
**关键词**: Conformal Prediction, Conformal Risk Control, Image Segmentation, Uncertainty Quantification, SVD/PCA, Medical Imaging  

## 一句话总结
CONSIGN 用 SVD 从分割模型的多次采样里提取空间相关的"主不确定性方向"，构造出能联合改变的空间感知共形预测集，在保持统计覆盖保证的同时把预测集体积压到比逐像素方法小几个数量级。

## 研究背景与动机
**领域现状**：分割模型给出的逐像素 softmax 置信度只是启发式分数，没有统计意义上的可靠性保证；在医学影像这类高风险场景里，"模型说 90% 确定"并不等于"真有 90% 覆盖"。共形预测（Conformal Prediction, CP）提供了把启发式不确定性转成带有限样本覆盖保证的严格框架，本文用的是其中的共形风险控制（Conformal Risk Control, CRC），目标是让预测集 $C(X)$ 满足 $\mathbb{E}[\ell(C(X_{test}), Y_{test})] \le \alpha$。

**现有痛点**：把 CP 直接搬到分割上时，绝大多数方法（RAPS、Mossina 等）都是**逐像素独立**地构造预测集 $C_\lambda(X_{ij}) = \{l : f(X_{ij})_l \ge 1-\lambda\}$。但图像像素之间存在强空间相关——相邻像素的标签往往一起变。逐像素独立处理虽然覆盖保证仍然成立，却会把大量"实际不可能同时出现"的不一致组合塞进预测集里（比如一只羊和一头牛在边界处胡乱拼接），导致预测集**体积爆炸**、过度保守、可解释性差。

**核心矛盾**：覆盖保证要求"宁可放大也别漏"，而逐像素独立放大的方式忽略了空间结构，放大得毫无必要地大。如何在**保住统计保证**的同时，让预测集只沿着**真正有意义的空间变化方向**张开？

**本文目标**：构造一种空间感知的共形预测集，里面的样本沿着协调的空间结构联合变化，从而显著缩小预测集体积，且兼容任何能产生多次采样输出的预训练分割模型（dropout / Bayesian / ensemble 都行）。

**核心 idea**：**[空间分解]** 借鉴图像复原里用 SVD 提取主不确定性方向的思路（Belhasin 2023, Nehme 2023），从模型采样矩阵里抽出少数几个主成分方向作为"不确定性基"，预测集只在这几个方向的系数区间内张开；**[非线性量化]** 利用分割是离散分类任务这一性质，引入 argmax 投影 $P(\sigma)$，使得即便只用截断的 $K\in\{2,5\}$ 个主成分也能覆盖真值，无需像回归版那样补特殊流程。

## 方法详解

### 整体框架
CONSIGN 分两步：**构造空间感知预测集** + **校准**。给定一张图，先用预训练模型 $f$ 多次采样得到 $N_s$ 个 softmax 输出，组成样本矩阵后做 SVD，取前 $K$ 个主成分方向 $\{u_k\}$ 作为不确定性基；预测集定义为"均值加上这几个基方向的线性组合、再经 argmax 投影后能得到的所有分割图"。校准阶段在校准集上逐步增大尺度参数 $\lambda$ 来放大系数区间，直到经验风险降到阈值以下，从而拿到满足覆盖保证的 $\hat\lambda$。

```mermaid
flowchart TD
    A[预训练模型 f 多次采样<br/>得到 N_s 个 softmax 输出] --> B[构造样本矩阵 Ŝ-μ<br/>做 SVD: UΣVᵀ]
    B --> C[取前 K∈{2,5} 个主成分 u_k<br/>提取空间不确定性方向]
    C --> D[对每个 u_k 算系数分位区间<br/>a_k, b_k]
    D --> E["预测集 C*_λ: 均值 + Σ c_k u_k<br/>经 argmax 投影 P(·), c_k∈[A_k,B_k]"]
    E --> F[校准: 逐步增大 λ<br/>解约束优化判断 Y 是否在集合内]
    F --> G[得到 λ̂ 满足覆盖保证<br/>E[ℓ] ≤ α]
```

### 关键设计

**1. 用 SVD 提取空间相关的主不确定性方向：把"哪里不确定、怎么一起变"压缩成几个基向量。** CONSIGN 不要求模型有特殊结构，只要能产生采样 $\hat s_1, \dots, \hat s_{N_s} \in \mathbb{R}^{WHL}$（来自 dropout、概率 U-Net 或 ensemble 都行）。它把采样减去均值后组成矩阵做约简 SVD：$\hat S - \mu(X)\cdot \mathbf{1}^T = U\Sigma V^T$，其中 $U$ 的每一列 $u_k$ 就是采样空间里方差最大的主方向，天然编码了"哪些像素区域不确定、这些区域如何协同变化"。关键在于只需算前 $K < \min\{WHL, N_s\}$ 个奇异值，实验里 $K\in\{2,5\}$ 就够——相比全基表示 $K=WHL$ 计算量骤降。这一步是整个方法"空间感知"和"低维高效"两个优点的来源。

**2. 在主成分系数空间里定义预测集：让预测沿有意义的方向联合张开，而非逐像素乱炸。** 对每个主方向先算系数的经验分位区间 $a_k = Q_{\alpha/2}\{\langle u_k, \hat s_n - \mu\rangle\}$、$b_k = Q_{1-\alpha/2}\{\cdots\}$，再围绕分位中点构造随 $\lambda$ 线性缩放的对称区间：

$$A_k = \frac{a_k+b_k}{2} - \lambda \Sigma_{k,k}\frac{b_k-a_k}{2}, \quad B_k = \frac{a_k+b_k}{2} + \lambda \Sigma_{k,k}\frac{b_k-a_k}{2}.$$

用奇异值 $\Sigma_{k,k}$ 加权意味着方差越大的主成分允许越宽的系数范围——模型越不确定的方向给越多自由度。预测集随之定义为

$$C^*_\lambda(X) = \Big\{ Y : \exists c \in \prod_{k=1}^K [A_k, B_k],\ Y \overset{\beta}{=} P\big(\mu(X) + \sum_{k=1}^K c_k u_k(X)\big) \Big\}.$$

每个落在系数盒子里的 $c$ 经 argmax 投影 $P(\cdot)$ 得到一张完整、空间协调的分割图，这正是它比逐像素方法"realistic 且 coherent"的根本原因。

**3. 非线性 argmax 投影让截断 PCA 也能覆盖真值：把分类任务的离散性变成优势。** 与回归版（Belhasin 2023）不同，CONSIGN 在系数到标签之间插入了非线性量化步骤 $P(\sigma)$（沿标签维取 argmax）。正因为这一步把连续 softmax 量化成离散标签，即使强行把 $u_{K+1},\dots,u_{WHL}$ 的系数全设为零，只用前 $K$ 个主成分，重构出的分割图在 $\lambda$ 足够大时通常仍能命中真值——回归里做不到，所以那边需要额外补流程保覆盖。文中还把回归的精度率改成**逐标签精度率** $\beta$：$Y_1 \overset{\beta}{=} Y_2$ 当 $\frac{1}{L}\sum_l \frac{\sum_{ij}\mathbb{I}(Y_1^{ij}=l \wedge Y_2^{ij}=l)}{\sum_{ij}\mathbb{I}(Y_1^{ij}=l)} > \beta$，避免偏向高频标签。

**4. 约束优化式校准 + 终止机制：在无法显式判断"真值是否在集合内"时仍保住保证。** 由于预测集形式复杂，无法穷举所有 $c$ 来检查 $Y \in C^*_\lambda(X)$，CONSIGN 改为求解约束最小化 $c^* = \arg\min_{c\in B} L(Y, P(\mu(X) + \sum_k c_k u_k))$（损失见式 11），若数值解满足 $Y \overset{\beta}{=} P(\sigma)$ 就判定命中。校准算法逐步加 $\lambda$ 直到经验风险 $\hat R(\lambda) \le \alpha - \frac{1-\alpha}{N_{cal}}$。即使数值求解器漏掉了本应存在的 $c$（导致 $\lambda$ 被多抬一点、区间偏保守），由 Angelopoulos 定理可证覆盖保证 $\mathbb{P}[Y_{test}\in C^*_{\hat\lambda}] \ge 1-\alpha$ 仍成立（Lemma 1）。对那些主方向退化、$\lambda\to\infty$ 也不收敛的病态情形，设 $\lambda_{max}$ 强制终止，**显式告知用户**当前参数下拿不到有意义的预测集，而不是悄悄给个失效的集合。

## 实验关键数据

### 主实验设置
- **数据集（5 个）**：医学三个 —— M&Ms-2、MS-CMR19（心脏，dropout U-Net 采样）、LIDC（肺结节，概率 U-Net 采样）；COCO 两个子集 —— animals、vehicles（DeepLabV3+ 不同 backbone 组成 ensemble 采样）。
- **基线**：逐像素 RAPS（PW）、空间感知 SACP（Liu 2025）。
- **CONSIGN 变体**：$K=2$ 与 $K=5$ 两种主成分数。
- **评测指标**：Chao 估计器（估预测集里唯一分割图数量，越小越好）、sEC（采样估计覆盖率，应收敛到 $1-\alpha$）、平均 Pearson 相关 $\hat\rho$（衡量样本是否被限制在低维流形）。

| 维度 | CONSIGN | PW (RAPS) / SACP |
|------|---------|------------------|
| Chao 估计器（预测集体积） | 始终被两个基线**包住**（更小）；LIDC 上差距最大，相差数个数量级 | 显著更大 |
| sEC 收敛速度 | 更少采样即达到 $1-\alpha$ 覆盖；COCO-vehicle 仅 10 个样本即满足 | 收敛更慢 |
| 采样间相关 $\hat\rho$ | 样本相关性强 → 被限制在低维子空间 | 近独立 → 内在维度更高 |

### 关键发现
- **空间结构带来的增益取决于采样质量**：用专门设计来产多样样本的概率 U-Net（LIDC）时，体积比基线小**几个数量级**；用 dropout 采样（心脏数据）时增益相对温和。
- **$K=2$ vs $K=5$ 的权衡**：$K=5$ 给系数更多自由度，可能预测范围更广、Chao 估计更高；$K=2$ 更紧凑。多数情况下少量主成分已足够。
- **COCO 高不确定场景**：物体大、内在不确定性高时，CONSIGN 的 Chao 估计也会偏高，说明增益与任务的空间结构强弱相关。
- **参数选择经验**：医学等安全攸关场景用小 $\alpha$（如 0.05）严格控信、细小结构用高 $\beta$（>0.8）；大物体分割可放宽 $\beta$；强模型允许更激进的参数。

## 亮点与洞察
- **把"空间相关"从口号落成可计算的低维基**：SVD 主成分既定位了不确定区域，又编码了像素如何协同变化，几个方向就能撑起整个预测集，这是它体积优势的根源。
- **抓住分类任务的离散性做文章**：argmax 量化让截断 PCA 仍能覆盖真值，免去了回归版的补丁流程——一个"任务特性反而成了方法优势"的漂亮设计。
- **模型无关 + 保证不打折**：只要模型能采样就能用，且即使数值求解器不完美，统计覆盖保证也由理论兜底，工程上很实用。
- **失败时显式报警**：$\lambda_{max}$ 终止机制把"拿不到有效预测集"明确暴露给用户，而非给出一个表面合理实则失效的结果，对高风险场景是负责任的设计。

## 局限与展望
- **预测集体积只能采样估计**：$C^*$ 的真实体积无法解析计算，只能靠 Chao 估计器/采样近似，评测精度受采样数限制。
- **依赖采样质量**：增益高度依赖预训练模型能否产出有意义的多样样本；dropout 这类弱采样下优势明显缩水。
- **校准含数值优化**：约束最小化无法保证全局最优，可能让 $\lambda$ 偏保守（虽不破坏保证但牺牲紧致度）；且每个校准样本都要解优化，计算成本高于逐像素直接判断。
- **病态情形需人工设 $\lambda_{max}$**：主方向退化时方法会终止，参数 $\alpha, \beta, \lambda_{max}$ 的选择需要领域经验。
- **展望**：把空间分解推广到更强的生成式采样器、自动化参数选择、以及降低校准阶段约束优化的开销，都是自然的后续方向。

## 相关工作与启发
- **共形预测 / CRC**：Vovk、Angelopoulos 等奠定 CP 与共形风险控制框架，本文在 Angelopoulos (2024) 的校准定理上证明覆盖保证。
- **分割 CP 的逐像素路线**：Wundram (2024)、Mossina (2024) 把 CRC 扩到多类分割但仍逐像素；Blot (2025) 做组条件风险控制；本文指出它们共同的体积爆炸问题。
- **空间感知 CP**：SACP（Liu 2025）在邻域聚合分数，是本文最直接的对照；Brunekreef (2024) 按相似区域聚合非一致性分数但依赖定制校准；Davenport (2024) 让分数依赖到边界距离。
- **SVD 提取不确定性**：Belhasin (2023)、Nehme (2023) 在图像复原/回归里用 SVD 取主方向，本文把这套迁移到分割并用 argmax 投影改造，是核心方法来源。
- **启发**：在结构化输出任务里，"先把输出空间的相关性压成低维基、再让不确定性沿基张开"是一个通用范式，可迁移到深度估计、光流、视频分割等任何能多次采样的稠密预测任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把 SVD 主成分不确定性从回归迁移到分割，并用 argmax 量化巧妙解决截断 PCA 的覆盖问题，是清晰且有理论支撑的创新组合。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 3 个医学 + 2 个 COCO 数据集、三种采样模型、两个基线、多指标多随机划分，但缺少与更多近期空间感知 CP 方法的横向对比、体积只能采样估计。
- **写作质量**: ⭐⭐⭐⭐ — 问题动机和方法推导清晰，公式与算法伪代码完整，覆盖保证有 Lemma 支撑；部分记号偏密集。
- **价值**: ⭐⭐⭐⭐ — 模型无关、带统计保证、显著压缩预测集体积，对医学影像等高风险分割的可信 UQ 有实际落地价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation](hfsti-net_hierarchical_frequency-spatial-temporal_interactions_for_video_polyp_s.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[CVPR 2026\] PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation](../../CVPR2026/medical_imaging/pmrnet_physics-informed_multi-scale_refinement_network_for_medical_image_segment.md)
- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](../../AAAI2026/medical_imaging/unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)

</div>

<!-- RELATED:END -->
