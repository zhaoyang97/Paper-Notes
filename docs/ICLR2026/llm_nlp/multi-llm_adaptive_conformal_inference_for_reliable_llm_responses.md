# Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses

**会议**: ICLR2026  
**arXiv**: [2602.01285](https://arxiv.org/abs/2602.01285)  
**代码**: [GitHub](https://github.com/MLAI-Yonsei/MACI)  
**领域**: llm_nlp  
**关键词**: Conformal Inference, LLM Factuality, Multi-LLM Ensemble, False-Claim Filtering, Distribution-Free Guarantee  
**作者**: Kangjun Noh, Seongchan Lee, Ilmun Kim, Kyungwoo Song（延世大学 & KAIST）

## 一句话总结

提出 MACI（Multi-LLM Adaptive Conformal Inference），通过**累积乘积型 conformity score** + **多 LLM 集成**的 factuality 评分 + **组条件校准**，在严格保证用户指定错误率的同时，显著提升 LLM 回复中事实性声明的保留率。

## 研究背景与动机

1. **LLM 幻觉问题**：LLM 在医疗、法律等高风险领域被广泛使用，但回复中可能包含虚假信息（hallucination），亟需提供统计保证。
2. **Conformal Inference (CI) 的引入**：CI 提供无分布假设的有限样本保证，已有工作（BCI, Mohri & Hashimoto 2024）将其用于 LLM 回复的虚假声明过滤——将回复分解为原子声明，基于 factuality score 设阈值过滤。
3. **BCI 过于保守**：BCI 使用单一全局阈值，仅提供边际覆盖（marginal coverage），在子群体间可能出现严重的过覆盖/欠覆盖；其 conformity score 仅依赖单个最差声明的分数，对估计误差极其敏感，导致大量真实声明被误删。
4. **CCI 保证松弛**：CCI（Cherian et al., 2024）引入自适应阈值函数以实现条件保证，但依赖自适应错误率 $\alpha$，在高风险场景中不适用；其线性特征空间难以捕捉 LLM 回复的复杂语义分组结构。
5. **Conformity score 设计缺陷**：既有方法均基于单个极端声明分数构造 conformity score，忽视了其余声明的集体置信信息。
6. **核心目标**：在严格控制组条件覆盖率（group-conditional coverage）的前提下，最大化真实声明的保留率（retention ratio）。

## 方法详解

### 整体框架

MACI 的整体流程：
1. **声明分解**：将 LLM 回复 $D = (P, C, Y)$ 分解为原子声明集合 $C = \{c_1, \dots, c_{|C|}\}$
2. **多 LLM 评分**：使用 $M$ 个黑盒 LLM 对每个 (prompt, claim) 对生成 verbalized factuality score $p_m(P, c) \in [0, 1]$
3. **集成优化**：通过优化权重 $w$ 得到集成评分 $p_{\text{ens}}(P, c; w) = \sum_{m=1}^{M} w_m p_m(P, c)$
4. **累积乘积过滤**：按 factuality score 降序排列声明，保留累积乘积 $\ge \tau$ 的前 $K$ 个声明
5. **组条件校准**：在校准集上对每个组 $k$ 独立计算分位数阈值 $\hat{Q}_{1-\alpha}^{(k)}$

### 关键设计 1：乘积型 Conformity Score

**Oracle 过滤规则**：给定排列 $\pi_i$ 使 $p_i^*(c_{i,\pi_i(1)}) \ge \cdots \ge p_i^*(c_{i,\pi_i(N_i)})$，定义截断索引：

$$K_i^*(\tau) = \max\left\{k \in [N_i] : \prod_{j=1}^{k} p_i^*(c_{i,\pi_i(j)}) \ge \tau \right\}$$

与 BCI/CCI 仅使用单个极端分数不同，MACI 的 conformity score 是**所有保留声明 factuality score 的累积乘积**：

$$E_i = \inf\{\tau \in [0,1] : F(\hat{p}, \tau, U_i; P_i, C_i) \subseteq A_i\}$$

这种乘积聚合方式直接反映"保留集整体为事实"的联合可信度，对单个声明的估计误差更鲁棒。

### 关键设计 2：组条件校准（Mondrian Framework）

对于分组函数 $g: \mathcal{P} \times \mathcal{C} \to \{1, \dots, K\}$，在校准集 $\mathcal{I}_k = \{i : g(P_i, C_i) = k\}$ 上独立计算阈值：

$$\hat{Q}_{1-\alpha}^{(k)} = \text{Quantile}(\{E_i : i \in \mathcal{I}_k\}, 1-\alpha)$$

**Theorem 2** 证明：在可交换性假设下，对任意组 $k$ 均满足：

$$\mathbb{P}\big(F_{n,\alpha}^{(k)}(P_{n+1}, C_{n+1}) \subseteq A_{n+1} \mid g(P_{n+1}, C_{n+1}) = k\big) \ge 1 - \alpha$$

### 关键设计 3：多 LLM 集成优化

**动机**：Theorem 3 证明保留率差距 $\Delta$ 受控于估计误差的多项式速率：

$$\Delta \le \mathfrak{C}' \big(\mathbb{E}[(\hat{p} - p^*)^2]\big)^{\frac{\beta}{\beta+2}}$$

即 factuality score 的 MSE 越小，保留率越接近 oracle。

**优化目标**：由于 oracle $p^*$ 不可观测，采用代理目标——在保持 $\text{TPR} \ge 1-\delta$ 的约束下最小化 FPR：

$$p^\star = \arg\min_{p} \mathbb{E}[\text{FPR}(p, \tau_{p,\delta})]$$

使用 $M=3$ 个模型（Llama-3.3-70B-Instruct、Qwen-2.5-72B-Instruct、DeepSeek-V3）的加权集成实现。

## 实验

### 实验设置

- **数据集**：MedLFQA（医疗 QA）、WikiBio（维基百科传记）、ExpertQA（专家级 QA）
- **基线**：BCI（Basic CI, Mohri & Hashimoto 2024）、CCI（Conditional CI, Cherian et al. 2024）
- **分组标准**：每个数据集定义语义分组（如医疗内容类型、浏览量、问题领域）+ False-Claim Risk 通用分组
- **目标覆盖率**：$1-\alpha \in \{0.80, 0.90, 0.95\}$，30 次重复实验取均值

### 主实验：覆盖率 & 保留率（Table 1 精选）

| 数据集 | 方法 | $1{-}\alpha{=}0.80$ Cov. | Ret. | $1{-}\alpha{=}0.90$ Cov. | Ret. | $1{-}\alpha{=}0.95$ Cov. | Ret. |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MedLFQA | BCI | 0.80 ✅ | 0.06 | 0.90 ✅ | 0.02 | 0.95 ✅ | 0.01 |
| | CCI | 0.81 ✅ | 0.56 | 0.90 ✅ | 0.31 | 0.95 ✅ | 0.18 |
| | **MACI** | **0.80 ✅** | **0.71** | **0.90 ✅** | **0.50** | **0.95 ✅** | **0.30** |
| WikiBio | BCI | 0.81 ✅ | 0.02 | 0.90 ✅ | 0.01 | 0.95 ✅ | 0.01 |
| | CCI | 0.79 ✅ | 0.19 | 0.89 ✅ | 0.11 | 0.93 ❌ | 0.06 |
| | **MACI** | **0.81 ✅** | **0.43** | **0.90 ✅** | **0.25** | **0.95 ✅** | **0.13** |
| ExpertQA | BCI | 0.91 ❌ | 0.13 | 0.91 ✅ | 0.13 | 0.91 ❌ | 0.13 |
| | CCI | 0.85 ❌ | 0.18 | 0.85 ❌ | 0.17 | 0.85 ❌ | 0.17 |
| | **MACI** | **0.80 ✅** | **0.45** | **0.90 ✅** | **0.15** | **0.95 ✅** | **0.10** |

**核心发现**：
- MACI 在几乎所有组上达到目标覆盖率，同时保留率远超基线
- BCI 保留率极低（MedLFQA 仅 1%~6%），过于保守
- CCI 在 WikiBio ($\alpha$=0.05) 和 ExpertQA 上出现欠覆盖，组条件保证失效

### 消融与分析

#### 多 LLM 集成效果（Figure 3）

| 配置 | FPR ↓ | MSE ↓ | 保留率 ↑ |
|------|:-----:|:-----:|:--------:|
| 单 LLM | 高 | 高 | 低 |
| 算术均值集成 | 中 | 中 | 中 |
| **MACI（优化集成）** | **最低** | **最低** | **最高** |

- 不同 LLM 在虚假声明检测上的 Jaccard 距离很大（模式互补），验证了集成的合理性
- FPR 的改善与 MSE 的改善一致，证明代理目标与 oracle 目标对齐

#### 时间成本（Table 3，WikiBio 500 样本）

| 阶段 | SelfCheck | FSC-KG | CCI | **MACI** |
|------|:---------:|:------:|:---:|:--------:|
| 评分（s/样本） | 3.25 | 19.30 | 3.25 | **1.20** |
| 校准（s） | — | — | 10.33 | **3.24** |
| 总时间（s） | — | — | 1643.91 | **598.98** |

MACI 单次评分 + 轻量校准，总时间仅为 CCI 的 **36%**。

#### 协变量偏移（Table 2，MACI-DRE）

在 MedLFQA 上构造校准/测试分布不一致的 covariate shift 场景，MACI-DRE 通过密度比估计重采样校准集，有效缓解偏移带来的组覆盖率偏差，同时保持相近的保留率。

## 亮点

- **乘积型 conformity score**：首次将文档级过滤建模为声明分数的累积乘积，比极端值方法更鲁棒，是本文最核心的设计贡献
- **首个保留率理论分析**：Theorem 3 建立了 oracle-estimator 偏差与真实声明保留之间的定量关系，为集成设计提供理论动机
- **即插即用**：MACI 仅需要 per-claim 标量分数，可作为任意 LLM 生成器的后处理过滤器
- **实际效率**：总时间成本最低，适合实时部署

## 局限性

- **组定义依赖先验知识**：分组函数 $g$ 需要手动定义（如医疗内容类型），对于未知领域可能不容易设计
- **校准集规模要求**：组条件校准要求每个组有足够的校准样本（$n_k$），小组样本不足时阈值偏保守
- **ExpertQA 上保留率偏低**：当数据集噪声大、假声明比例高时（如 ExpertQA），保留率仍然有限（$\alpha=0.05$ 时仅 10%）
- **Covariate shift 处理是可选后处理**：MACI-DRE 需要额外的密度比估计步骤，增加了系统复杂度
- **对 factuality scorer 质量的依赖**：理论上保留率受限于 $\hat{p}$ 与 $p^*$ 的 MSE，若所有 base LLM 在同方向出错则集成增益有限

## 相关工作

- **BCI**（Mohri & Hashimoto, 2024）：首个将 CI 用于 LLM 事实性过滤的工作，但仅提供边际覆盖且保留率极低
- **CCI**（Cherian et al., 2024）：引入条件 CI + 自适应 $\alpha$ 提升保留率，但线性阈值函数难以捕捉复杂语义分组，自适应 $\alpha$ 不适用于高风险场景
- **多校准/多有效 CP**（Jung et al., 2023; Liu & Wu, 2025）：提供多组/多有效覆盖保证，但往往倾向保守，保留率低
- **RAG 增强 CI**（Feng et al., 2025）：将 CI 转移到外部检索组件，本质改变了保证对象
- **采样一致性方法**（SelfCheck, FSC-KG）：无严格统计保证，且时间成本高

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 累积乘积 conformity score + 保留率理论分析 + 多 LLM 集成优化的组合具有原创性
- **实验充分度**: ⭐⭐⭐⭐ — 3 个数据集、多种分组标准、消融、时间成本、协变量偏移，实验全面扎实
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，动机充足，结构完整
- **价值**: ⭐⭐⭐⭐ — 为 LLM 在高风险领域的可靠部署提供了实用且有理论保证的方案
