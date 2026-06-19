---
title: >-
  [论文解读] Efficient Network Automatic Relevance Determination
description: >-
  [ICML2025][ARD] 将自动相关性确定（ARD）从单输出扩展到多输出回归场景，提出 NARD 框架联合估计稀疏回归系数和输出精度矩阵，并设计 Sequential/Surrogate/Hybrid 三种加速算法将复杂度从 $\mathcal{O}(d^3)$ 降至 $\mathcal{O}(p^2)$。
tags:
  - "ICML2025"
  - "ARD"
  - "稀疏贝叶斯"
  - "精度矩阵估计"
  - "Graphical Lasso"
  - "多输出回归"
  - "特征选择"
---

# Efficient Network Automatic Relevance Determination

**会议**: ICML2025  
**arXiv**: [2506.12352](https://arxiv.org/abs/2506.12352)  
**代码**: 未公开  
**领域**: 稀疏贝叶斯学习 / 特征选择 / 多输出回归  
**关键词**: ARD, 稀疏贝叶斯, 精度矩阵估计, Graphical Lasso, 多输出回归, 特征选择

## 一句话总结
将自动相关性确定（ARD）从单输出扩展到多输出回归场景，提出 NARD 框架联合估计稀疏回归系数和输出精度矩阵，并设计 Sequential/Surrogate/Hybrid 三种加速算法将复杂度从 $\mathcal{O}(d^3)$ 降至 $\mathcal{O}(p^2)$。

## 研究背景与动机
- **多输出回归**广泛应用于基因组学、蛋白质组学等领域，需要同时建模输入-输出关系和输出间的依赖结构
- 生物数据通常具有**超高维特征**（数千基因/蛋白），但只有少量特征真正影响表型，需要**稀疏特征选择**
- 现有方法的不足：
    - MRCE 等频率方法使用 $\ell_1$ 惩罚，但对超参数敏感
    - HS-GHS、JRNS 等贝叶斯 MCMC 方法精度高但计算极其昂贵（>3000秒）
    - 经典 ARD/SBL 仅处理单输出，无法捕捉输出间相关性
- **核心动机**：设计一个既能联合估计稀疏回归系数与输出精度矩阵，又能高效处理高维数据的贝叶斯框架

## 方法详解

### NARD 框架
对回归系数矩阵 $W$ 施加**矩阵正态分布先验**：

$$W \sim \mathcal{MN}(0, V, K^{-1})$$

其中 $K = \text{diag}(\alpha_1, \cdots, \alpha_d)$ 为列精度矩阵，$V$ 为行协方差矩阵。通过 ARD 机制，当 $\alpha_i \to \infty$ 时，相应特征被自动剪除。

对精度矩阵 $\Omega = V^{-1}$ 施加 $\ell_1$ 惩罚以鼓励稀疏，使用 **Graphical Lasso** 更新：

$$\hat{V}, \hat{V}^{-1} \leftarrow \textbf{glasso}(V, \lambda)$$

整体采用**块坐标下降（BCD）**交替优化 $W$、$\alpha$、$V$，每轮迭代复杂度为 $\mathcal{O}(m^3 + d^3)$。

### Sequential NARD
受 Tipping 贪心方法启发，**逐个评估特征的贡献**：
- 将边际似然分解为与单个 $\alpha_i$ 相关/无关的两部分
- 定义 $\eta_i = \text{Tr}(q_i q_i^\top V^{-1}) - m s_i$，最优 $\alpha_i$ 为：

$$\alpha_i = \begin{cases} \frac{m s_i^2}{\eta_i}, & \eta_i > 0 \\ \infty, & \eta_i \leq 0 \end{cases}$$

- 从近空模型开始，逐步添加/删除特征，仅在 MLF 增大时保留更新
- 复杂度降至 $\mathcal{O}(m^3 + p^3)$，其中 $p \ll d$ 为最终保留特征数

### Surrogate NARD
引入**代理函数**逼近边际似然的下界：
- 利用 Lipschitz 连续梯度构造下界 $R(W, W')$，使 $\text{Tr}(g(W)) \leq \text{Tr}(R(W, W'))$
- 关键优势：$S_{xx} = K + \rho I$ 为对角矩阵，求逆仅需 $\mathcal{O}(d)$
- 通过 BCD 交替更新 $W, W', V, K$，复杂度降至 $\mathcal{O}(m^3 + d^2)$

### Hybrid NARD
结合 Sequential 的特征选择与 Surrogate 的高效计算：
- 用 Sequential 方式评估特征是否纳入模型
- 纳入后用 Surrogate 方式更新矩阵
- 复杂度进一步降至 $\mathcal{O}(m^3 + p^2)$

## 实验关键数据

### 合成数据 ($d=5000, m=1500, N=1500$)

| 方法 | TPR | FPR | 总时间(秒) |
|------|-----|-----|-----------|
| MRCE | 0.9083 | 0.0072 | 53 |
| CAPME | 0.8972 | 0.0124 | 52 |
| HS-GHS | 0.9463 | 0.0033 | >3000 |
| JRNS | 0.9485 | 0.0037 | >3000 |
| NARD | 0.9483 | 0.0062 | 49 |
| Sequential NARD | 0.9459 | 0.0067 | 35 |
| Surrogate NARD | 0.9462 | 0.0072 | 31 |
| Hybrid NARD | 0.9471 | 0.0068 | 23 |

### 可扩展性 ($N=20000, m=2000$, 单步时间/秒)

| $d$ | MRCE | NARD | Surrogate NARD |
|-----|------|------|----------------|
| 5000 | 12.2 | 10.7 | 3.7 |
| 10000 | 33.4 | 30.8 | 8.9 |
| 20000 | 201.9 | 168.6 | 33.7 |
| 30000 | 421.3 | 376.7 | 64.7 |

### 衰老表型数据
- 1022 名健康个体，5641 个表型（1522 宏观 + 4119 分子表型）
- NARD 各变体与基线的 Jaccard 指数 >98.5%，表明特征选择高度一致
- NARD 约 24 分钟，Sequential NARD 约 14 分钟

### TCGA 癌症数据
- 7 种肿瘤类型，10 条关键信号通路
- NARD 成功识别出 COAD 中 PI3K/AKT 通路的 GSK3-AKT 调控关系，与已知生物学知识吻合

## 亮点与洞察
- **统一框架**：将 ARD 特征选择与 Graphical Lasso 精度矩阵估计有机结合，一个框架同时解决两个问题
- **理论优雅**：三种加速方案均有严格的数学推导支撑（代理下界证明、复杂度分析）
- **实用性强**：Hybrid NARD 在 $d=5000$ 时加速约 2×，$d=30000$ 时加速约 6×，且精度几乎不损失
- **生物可解释性**：在 TCGA 数据上发现的蛋白质交互网络与已知生物学通路一致
- 可通过核方法自然扩展到非线性场景

## 局限与展望
- 模型核心假设是**线性关系**，虽提及核扩展但未深入实验验证
- Graphical Lasso 中 $\lambda$ 的选择依赖 5 折交叉验证，计算成本未计入总时间对比
- 未与近年的深度特征选择方法（如门控网络、注意力机制）对比
- 理论分析缺少收敛速率的严格界
- 代码**未公开**，可复现性存在障碍

## 相关工作与启发
- **MRCE** (Rothman et al., 2010)：$\ell_1$ 联合惩罚 $W$ 和 $\Omega$，频率方法代表
- **HS-GHS** (Li et al., 2021)：Horseshoe + Graphical Horseshoe 先验，贝叶斯方法精度高但极慢
- **Tipping 快速 ARD** (Tipping & Faul, 2003)：Sequential 更新的灵感来源
- 对研究高维多输出稀疏学习、贝叶斯特征选择、生物网络推断的方向有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ — ARD 到多输出的扩展自然且有意义，三种加速方案设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 合成+衰老表型+TCGA 多场景验证，缺少与深度方法的对比
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰严谨，算法伪代码完整
- 价值: ⭐⭐⭐⭐ — 在高维生物统计领域有实际应用价值，代码未开源略有遗憾

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[AAAI 2026\] Autonomous Concept Drift Threshold Determination](../../AAAI2026/others/autonomous_concept_drift_threshold_determination.md)
- [\[ACL 2025\] AutoMixer: Checkpoint Artifacts as Automatic Data Mixers](../../ACL2025/others/automixer_checkpoint_artifacts_as_automatic_data_mixers.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](../../ACL2025/others/improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)

</div>

<!-- RELATED:END -->
