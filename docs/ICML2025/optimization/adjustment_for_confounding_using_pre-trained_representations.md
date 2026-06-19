---
title: >-
  [论文解读] Adjustment for Confounding using Pre-Trained Representations
description: >-
  [ICML 2025][优化/理论][因果推断] 本文研究如何利用预训练神经网络的隐表示来调整非表格数据（如图像、文本）中的混杂因素，形式化了表示充分性条件，证明了稀疏性/可加性假设在可逆线性变换（ILT）下不成立，并基于低内在维度和层次组合模型建立了深度网络的收敛速率理论，从而保证 DML 框架下 ATE 估计的有效推断。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "因果推断"
  - "平均处理效应"
  - "预训练表示"
  - "双机器学习"
  - "内在维度"
---

# Adjustment for Confounding using Pre-Trained Representations

**会议**: ICML 2025  
**arXiv**: [2506.14329](https://arxiv.org/abs/2506.14329)  
**代码**: 无  
**领域**: 优化  
**关键词**: 因果推断, 平均处理效应, 预训练表示, 双机器学习, 内在维度

## 一句话总结

本文研究如何利用预训练神经网络的隐表示来调整非表格数据（如图像、文本）中的混杂因素，形式化了表示充分性条件，证明了稀疏性/可加性假设在可逆线性变换（ILT）下不成立，并基于低内在维度和层次组合模型建立了深度网络的收敛速率理论，从而保证 DML 框架下 ATE 估计的有效推断。

## 研究背景与动机

### 问题设定

因果推断中，**平均处理效应（ATE）**的估计是核心任务。在观测数据中，混杂因素同时影响处理变量 $T$ 和结果变量 $Y$，导致朴素估计产生偏差。传统方法通常处理表格型混杂变量，但在医学等领域，混杂信息往往隐藏在**非表格数据**中——例如 CT 扫描图像中的疾病严重程度同时影响治疗选择和预后。

### 现有方法的局限

- **双机器学习（DML）框架**（Chernozhukov et al., 2017）允许使用 ML 方法调整非线性混杂效应，但最初针对表格数据设计
- 直接将非表格数据送入 DML 面临**维度灾难**：图像维度极高，而医学场景样本量有限
- 使用预训练模型提取隐表示 $Z = \varphi(W)$ 是自然的解决方案，但**缺乏理论保证**
- 关键理论障碍：表示 $Z$ 仅在可逆线性变换（ILT）下可识别，即 $Z$ 和 $QZ$（$Q$ 为可逆矩阵）在信息论上等价

### 核心动机

论文旨在回答：**在什么条件下，预训练表示可以替代原始非表格数据用于 ATE 的混杂调整？** 这需要解决表示非可识别性带来的结构假设失效问题，并建立保证有效统计推断的收敛速率理论。

## 方法详解

### 整体框架

论文的理论框架分为三个层次：

1. **表示充分性**（Section 3.1）：预训练表示何时包含足够的混杂信息
2. **收敛速率分析**（Section 4-5）：在 ILT 非可识别性下，什么假设能保证快速收敛
3. **DML 推断有效性**（Section 5.3）：将以上结果整合到 DML 估计器中

给定 $n$ 个 i.i.d. 观测 $(T, W, Y)$，目标是估计：

$$\text{ATE} \coloneqq \mathbb{E}[\mathbb{E}[Y|T=1,W] - \mathbb{E}[Y|T=0,W]]$$

其中 $W$ 是非表格混杂数据，$T \in \{0,1\}$ 是二值处理变量。

### 关键设计

#### 1. 表示充分性条件

论文形式化了三种递进的充分性条件（Definition 3.1）：

- **$P$-valid**：最弱条件，保证 $Z$ 是有效的调整集，即 $\mathbb{E}_P[\mathbb{E}_P[Y|T=t,\sigma(Z)]] = \mathbb{E}_P[\mathbb{E}_P[Y|T=t,W]]$
- **$P$-OMS（结果均值充分）**：几乎处处相等，即 $\mathbb{E}[Y|T=t,Z] = \mathbb{E}[Y|T=t,W]$
- **$P$-ODS（结果分布充分）**：$Y \perp W | T, Z$，最强条件

关键洞察：**$P$-valid 是保证 ATE 正确估计的充分且必要条件**，无需最强的 ODS 条件。

#### 2. ILT 下的不变性分析

这是本文最核心的理论贡献。论文系统分析了不同结构假设在 ILT 下的不变性：

| 结构假设 | ILT 不变性 | 对应收敛速率 | 适用性 |
|---------|-----------|------------|--------|
| 光滑性（$s$-smooth） | ✅ 保持 | $n^{-s/(2s+d)}$ | 合理但不足 |
| 可加性（Additive） | ❌ 不保持 | $n^{-s/(2s+1)}$ | 不合理 |
| 稀疏性（Sparse） | ❌ 不保持 | $\sqrt{p\log(d/p)/n}$ | 不合理 |
| 内在维度（$d_\mathcal{M}$） | ✅ 保持 | $n^{-s/(2s+d_\mathcal{M})}$ | 合理 |

**Lemma 4.2** 证明了对于 Haar 测度下几乎所有的 ILT $Q$：若 $f$ 可加且含非线性分量，则 $f \circ Q^{-1}$ 不可加；若 $f$ 是稀疏线性的，则 $f \circ Q^{-1}$ 不稀疏。

这意味着 Lasso、随机森林、基于坐标轴分裂的树方法等依赖稀疏性/可加性假设的方法，**在预训练表示空间中理论上不能保证快速收敛**。

#### 3. 层次组合模型 + 流形结构（Assumption 5.2）

论文提出将**流形假设**与**层次组合模型（HCM）**结合的新假设：

目标函数可分解为 $f_0 = f \circ \psi$，其中：
- $\mathcal{M}$ 是紧致的 $d_\mathcal{M}$ 维光滑流形
- $\psi: \mathcal{M} \to \mathbb{R}^p$ 是 $s_\psi$-光滑映射（流形到欧氏空间的嵌入）
- $f$ 是 HCM（层次组合模型）的 $k$ 层结构

HCM 的定义（Definition 5.1）是递归的：
- **Level 0**：$f(x) = x_j$（选取某个坐标）
- **Level $k$**：$f(x) = h(h_1(x), \ldots, h_p(x))$，其中 $h$ 是 $s$-光滑函数，$h_i$ 是 level $k-1$ 的 HCM

**Lemma 5.3** 证明了 Assumption 5.2 在 ILT 下不变——这是关键的理论保证。HCM 的层次结构天然契合深度神经网络的逐层计算，使得 DNN 能高效利用此结构。

#### 4. DNN 收敛速率（Theorem 5.5）

在 Assumption 5.2 下，存在前馈 DNN 架构使得：

$$\|\hat{f} - f_0\|_{L_2(P_Z)} = O_p\left(\max_{(s,p) \in \mathcal{P} \cup (s_\psi, d_\mathcal{M})} n^{-s/(2s+p)}\right)$$

速率仅取决于约束集 $\mathcal{P}$ 和流形嵌入参数 $(s_\psi, d_\mathcal{M})$ 中的**最差情况对**。利用 Whitney 嵌入定理，$p$ 可取 $2d_\mathcal{M}$，$s_\psi = \infty$，从而速率主要由流形内在维度控制。

### 损失函数 / 训练策略

#### DML 估计器

采用交叉拟合（cross-fitting）策略：将样本分为 $K$ 折，每折的 nuisance 函数在其余折上训练。最终 ATE 估计器使用正交化得分函数（orthogonalized score）：

$$\rho(T_i, Y_i, Z_i; g, m) = g(1, Z_i) - g(0, Z_i) + \frac{T_i(Y_i - g(1,Z_i))}{m(Z_i)} + \frac{(1-T_i)(Y_i - g(0,Z_i))}{1 - m(Z_i)}$$

其中 $g(t,z) = \mathbb{E}[Y|T=t, Z=z]$ 是结果回归函数，$m(z) = \mathbb{P}[T=1|Z=z]$ 是倾向性得分。

#### DML 推断有效性（Theorem 5.7）

当 $g$ 和 $m$ 满足 Assumption 5.2，且满足正则条件：

$$\min_{(s,p) \in \mathcal{P}_g \cup (s_\psi, d_\mathcal{M})} \frac{s}{p} \times \min_{(s',p') \in \mathcal{P}_m \cup (s'_\psi, d_\mathcal{M})} \frac{s'}{p'} > \frac{1}{4}$$

则 DML 估计器满足渐近正态：$\sqrt{n}(\widehat{\text{ATE}} - \text{ATE}) \to \mathcal{N}(0, \sigma^2)$。该条件刻画了光滑度与维度之间的权衡——每个组合函数的输入维度需小于其光滑度的两倍。

## 实验关键数据

### 主实验

实验使用两种非表格数据模拟混杂场景：

| 数据集 | 数据类型 | 预训练模型 | 表示维度 $d$ | 样本量 |
|-------|---------|-----------|------------|-------|
| IMDb Movie Reviews | 文本 | BERT (bert-base-uncased) | 768 | 50,000 |
| Chest X-ray (Kermany) | 图像 | DenseNet-121 (TorchXRayVision) | 1,024 | 5,863 |

**Label Confounding 实验结果**（IMDb，5 次模拟）：

| 估计器 | 方法类型 | ATE 偏差 | 95% CI 覆盖 | 说明 |
|-------|---------|---------|------------|-----|
| Naive | 不调整 | 强负偏差 | 不覆盖 | 未考虑混杂 |
| DML (Lasso) | 稀疏假设 | 偏差显著 | 不覆盖 | ILT 下稀疏性失效 |
| DML (RF) | 树方法 | 偏差显著 | 不覆盖 | 坐标轴分裂不适用 |
| DML (Linear) | 无惩罚线性 | **无偏** | **覆盖** | ILT 不变的估计器 |
| S-Learner | 单一回归 | 偏差显著 | 不覆盖 | 缺少双稳健性 |
| Oracle | 真标签调整 | 无偏 | 覆盖 | 基准上界 |

### 消融实验

**Complex Confounding 实验**（X-ray，自编码器构造混杂）：

| 配置 | ATE 偏差 | CI 覆盖 | 说明 |
|-----|---------|--------|-----|
| DML + 神经网络 | 小 | 高覆盖 | HCM 结构下 DNN 适应低维流形 |
| DML + 随机森林 | 大 | 低覆盖 | 稀疏假设在 ILT 下失效 |
| S-Learner + 神经网络 | 小 | CI 过于乐观 | 缺少双稳健性 |
| DML (预训练) vs DML (CNN) | 预训练无偏 / CNN 偏差大 | 仅预训练覆盖 | 500 样本下预训练优势明显 |

### 关键发现

1. **ILT 非不变方法失败**：Lasso 和随机森林在预训练表示上表现不佳，因为稀疏性/可加性假设在 ILT 下不成立——这正是理论预测的
2. **内在维度远低于环境维度**：X-ray 表示的内在维度约 $d_\mathcal{M} \approx 12$，而环境维度 $d = 1024$，支持流形假说
3. **预训练至关重要**：在有限样本（500 张图）下，使用预训练表示的 DML 无偏，而从头训练 CNN 的 DML 有偏
4. **DML 的双稳健性重要**：S-Learner 即使用同样的神经网络也不能保证有效推断，只有 DML 的正交化得分能正确覆盖

## 亮点与洞察

1. **理论与实践的紧密对应**：Lemma 4.2 预测 Lasso/RF 会失败 → 实验完美验证；Theorem 5.5 预测 DNN 适应低维结构 → 实验确认
2. **从"什么不行"到"什么行"的完整论证**：先证明稀疏/可加性不合理（负面结果），再建立基于流形+HCM 的替代理论（正面结果）
3. **实际指导意义**：在预训练表示上做因果推断时，应优先使用神经网络（而非 Lasso/树方法）作为 nuisance 估计器
4. **HCM + 流形假设**的提出具有普适性，不仅适用于 ATE 估计，可推广到其他半参数推断任务

## 局限与展望

1. **单一模态假设**：仅考虑单一非表格数据源的混杂，未讨论多模态融合场景（如图像+文本同时作为混杂）
2. **仅限 ATE**：未涉及 ATT（处理组平均处理效应）或 CATE（条件平均处理效应）的推断
3. **预训练质量假设**：理论依赖于预训练表示满足 $P$-valid 条件，但实践中难以验证此条件是否成立
4. **HCM 结构的验证**：难以直接检验 nuisance 函数是否具有 HCM 结构，约束集 $\mathcal{P}$ 的参数选取依赖领域知识
5. **计算成本**：未讨论不同 nuisance 估计器的计算效率差异

## 相关工作与启发

- **DML 框架**（Chernozhukov et al., 2017, 2018）：本文的理论基础，扩展到非表格数据场景
- **表示学习 + 因果推断**（Veitch et al., 2019, 2020）：直接使用非表格数据但缺乏收敛速率保证
- **流形假说**（Fefferman et al., 2016）：高维数据集中在低维流形上，本文利用此假说建立收敛理论
- **DNN 适应内在维度**（Chen et al., 2019; Schmidt-Hieber, 2019）：DNN 能自动适应流形维度，本文将此结果与 HCM 结合
- **启发**：该理论框架可推广到强化学习中的 off-policy 评估、多模态因果发现等领域

## 评分

| 维度 | 分数 (1-10) | 说明 |
|-----|-----------|-----|
| 创新性 | 8 | 首次系统分析预训练表示在因果推断中的理论保证 |
| 理论深度 | 9 | 从不变性分析到收敛速率到推断有效性，层次清晰 |
| 实验验证 | 7 | 实验设计精巧但场景偏合成，缺乏真实因果任务 |
| 实用价值 | 7 | 提供了明确的方法选择指导，但条件验证仍困难 |
| 写作质量 | 8 | 逻辑链条清晰，理论动机和实验对应良好 |
| **总分** | **7.8** | 理论贡献扎实的因果推断与表示学习交叉工作 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](../../ICLR2026/optimization/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)
- [\[NeurIPS 2025\] Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](../../NeurIPS2025/optimization/contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](../../NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](../../ICML2026/optimization/probing_neural_tsp_representations_for_prescriptive_decision_support.md)

</div>

<!-- RELATED:END -->
