---
title: >-
  [论文解读] Private Frequency Estimation via Residue Number Systems
description: >-
  [AAAI 2026][本地差分隐私] 提出 ModularSubsetSelection (MSS)，一种基于剩余数系统（RNS）的本地差分隐私频率估计协议，在保持与 SubsetSelection 和 PGR 相当的估计精度的同时，显著降低通信开销（比 SS 减少达一半）、大幅加速服务器解码（比 PGR 快 11-448 倍）、并实现最低的数据重建攻击成功率。
tags:
  - "AAAI 2026"
  - "本地差分隐私"
  - "频率估计"
  - "剩余数系统"
  - "通信效率"
  - "数据重建攻击"
---

# Private Frequency Estimation via Residue Number Systems

**会议**: AAAI 2026  
**arXiv**: [2511.11569](https://arxiv.org/abs/2511.11569)  
**代码**: [GitHub](https://github.com/hharcolezi/private-frequency-oracle-rns)  
**领域**: 其他  
**关键词**: 本地差分隐私, 频率估计, 剩余数系统, 通信效率, 数据重建攻击

## 一句话总结

提出 ModularSubsetSelection (MSS)，一种基于剩余数系统（RNS）的本地差分隐私频率估计协议，在保持与 SubsetSelection 和 PGR 相当的估计精度的同时，显著降低通信开销（比 SS 减少达一半）、大幅加速服务器解码（比 PGR 快 11-448 倍）、并实现最低的数据重建攻击成功率。

## 研究背景与动机

本地差分隐私（LDP）是联邦分析中保护用户隐私的核心范式，广泛应用于 Apple 键盘预测、Google Chrome RAPPOR、Microsoft 遥测系统等十亿级设备场景。LDP 频率估计的核心任务是：每个用户持有私有输入 $x_i \in [k]$，通过本地随机化机制 $\mathcal{M}$ 发送扰动消息给不受信任的服务器，服务器恢复群体直方图 $\hat{\mathbf{f}}$。

评价 LDP 协议的四个关键维度构成一个**多约束体制**：

**效用（Utility）**：估计精度，以 MSE 衡量

**通信（Communication）**：每用户传输比特数

**服务器运行时间**：解码复杂度

**可攻击性（Attackability）**：对抗者从单条消息重建原始输入的成功率

现有协议在这四个维度上各有取舍：
- **RandomizedResponse (GRR)**：通信低（$\lceil\log_2 k\rceil$ 比特），但 MSE 有 $\Theta(k/e^\varepsilon)$ 的差距，且可攻击性最高
- **SubsetSelection (SS)**：达到最优 MSE，但通信开销 $\Theta(\omega \log_2(k/\omega))$ 较高
- **RAPPOR/OUE**：MSE 最优，但消息长度 $O(k)$，服务器代价 $O(nk)$
- **ProjectiveGeometryResponse (PGR)**：MSE 最优、通信仅 $\lceil\log_2 k\rceil$，但要求域大小匹配射影几何约束、依赖有限域算术、动态规划解码开销 $O(n + ke^\varepsilon \log k)$

核心问题：**能否设计一种协议同时在精度、带宽、计算和抗攻击四个维度取得良好平衡？**

## 方法详解

### 整体框架

MSS 采用**"分而治之"（divide & conquer）** 设计。用户端将输入通过 RNS 分解为多个小域上的残差后随机发送其一，服务器端通过加权最小二乘求解器重建完整直方图。

### 关键设计

#### 1. 用户端 "Divide" 混淆（Algorithm 1）

**剩余数系统编码**：给定 $\ell$ 个两两互素的模 $m_0, \ldots, m_{\ell-1}$，将输入 $x$ 映射为残差向量：

$$\mathbf{r}(x) = (x \bmod m_0, \ldots, x \bmod m_{\ell-1})$$

由**中国剩余定理（CRT）**，当 $\prod_j m_j \geq k$ 时该映射为单射。

**模块化采样**：用户不发送所有残差（这会分割隐私预算），而是：
1. 均匀随机选取一个模索引 $J \sim \text{Uniform}([\ell])$
2. 计算残差 $r = x \bmod m_J$
3. 用完整隐私预算 $\varepsilon$ 对 $r$ 在域 $[m_J]$ 上执行 SubsetSelection
4. 发送 $(J, Z)$，其中 $Z \subseteq [m_J]$ 是大小为 $\omega_J$ 的扰动子集

通信开销仅 $\lceil\log_2 \ell\rceil + \lceil\log_2 \binom{m_J}{\omega_J}\rceil$ 比特，远小于标准 SS。

**隐私保证**：由于 $J$ 独立于 $x$ 且均匀分布，隐私损失完全由单个 SS 机制的 $\varepsilon$-LDP 决定，因此 MSS 满足 $\varepsilon$-LDP（Theorem 1）。

#### 2. 服务器端 "Conquer" 估计

**设计矩阵构造**：对每个模 $m_j$，构造映射矩阵 $A_j \in \{0,1\}^{m_j \times k}$，其中 $A_j[r,x] = \mathbf{1}\{x \bmod m_j = r\}$。堆叠所有 $A_j$ 得到稀疏设计矩阵 $A \in \{0,1\}^{T \times k}$，$T = \sum_j m_j$。

**方差最优行权重**：基于 SS 的真假包含概率 $p_j, q_j$，计算最优 GLS 权重矩阵 $W^{1/2}$，得到加权设计矩阵 $A_w = W^{1/2}A$。

**去偏与最小二乘求解**：对每个块的残差计数去偏后，求解正则化最小二乘问题：

$$\hat{\mathbf{f}} = \arg\min_{\mathbf{z}} \|A_w \mathbf{z} - \tilde{\mathbf{s}}\|_2^2 + \lambda\|\mathbf{z}\|_2^2$$

使用 LSMR 迭代算法高效求解，利用 $A_w$ 的结构化稀疏性，每次迭代仅需 $O(k\ell)$。

#### 3. 模选择优化

通过理论 $\kappa$ 界限（Theorem 4）和随机采样相结合：
1. 由 Gershgorin 圆盘定理推导条件数上界 $\kappa \leq \alpha \cdot \frac{\ell + T^\star}{\ell - T^\star}$
2. 在素数带 $[k/(\beta\ell), \beta k/\ell]$ 中随机抽样 $\ell$ 个互素素数
3. 验证 CRT 覆盖、满秩和 $\kappa \leq \kappa_{\max}$ 条件
4. 从所有合格候选中选择精确 MSE 最小的组合

### 损失函数 / 训练策略

本文为隐私机制而非学习算法，核心优化目标为最小化 MSE。关键理论结果：

- **无偏性（Theorem 2）**：当 $\lambda=0$ 时，$\mathbb{E}[\hat{\mathbf{f}}] = \mathbf{f}$
- **渐近无偏性（Corollary 1）**：当 $\lambda > 0$ 时，偏差为 $O(\lambda)$
- **MSE 上界（Theorem 3）**：$\text{MSE}_{\text{MSS}} \leq \frac{4\kappa e^\varepsilon}{n(e^\varepsilon - 1)^2}$，即标准 SS 最优 MSE 的 $\kappa$ 倍
- **DRA 上界（Eq. 7）**：$\widehat{\mathbb{E}}[\text{DRA}]_{\text{MSS}} = \frac{1}{\ell}\sum_{j=0}^{\ell-1} \frac{p_j}{\omega_j \lceil k/m_j \rceil}$

## 实验关键数据

### 主实验

实验设置：$n=10000$ 用户，$k \in \{1024, 22000\}$，$\varepsilon \in \{0.5, \ldots, 5.0\}$，300 次独立试验。

| 协议 | MSE (相对SS) | 通信 (bits) | 服务器时间 | DRA |
|------|-------------|-------------|-----------|-----|
| GRR | 最高 (差 $\Theta(k/e^\varepsilon)$) | $\lceil\log_2 k\rceil$ | $O(n+k)$ | 最高 |
| SS | 最优 | 高 | 中等 | 中等 |
| OUE/RAPPOR | 最优 | $O(k)$ | $O(nk)$ | 中等 |
| PGR | 最优 | $\lceil\log_2 k\rceil$ | 高（见下） | 中等偏高 |
| **MSS** | **≤1.3× SS** | **比SS低达50%** | **最快** | **最低** |

### 服务器运行时间对比（$k=22000, n=10000$）

| $\varepsilon$ | MSS (秒) | PGR (秒) | MSS 加速比 |
|---|---|---|---|
| 2.0 | 0.160±0.027 | 2.897±0.220 | 18.1× |
| 3.0 | 0.272±0.086 | 9.618±0.679 | 35.4× |
| 4.0 | 0.168±0.056 | 11.461±0.702 | 68.3× |
| 4.5 | 0.127±0.047 | 56.906±3.570 | **447.8×** |
| 5.0 | 0.152±0.054 | 3.208±0.198 | 21.1× |

MSS 在所有隐私预算下均大幅快于 PGR，加速比从 11× 到 448×。

### 关键发现

1. **效用接近最优**：MSS 的经验 MSE 与 SS/PGR 之比始终 $\leq 1.3$，在 Zipf 和 Spike 两种分布下结论一致
2. **通信大幅降低**：尤其在高隐私（小 $\varepsilon$）regime 下，MSS 通信开销可为 SS 的一半
3. **抗攻击性最强**：MSS 在所有 $\varepsilon$ 值下 DRA 均最低，因为模块化随机化将概率质量分散到多个残差类上
4. **PGR 的 DRA 异常**：当域大小 $k$ 小于射影几何要求的自然大小 $K$ 时，PGR 的截断不匹配打破组合对称性，导致 DRA 急剧增加
5. **条件数 $\kappa$ 可控**：实践中 $\kappa$ 从未超过约 1.3，远低于理论上界 $\kappa_{\max}=10$

## 亮点与洞察

1. **数论与隐私的优雅结合**：利用中国剩余定理将大域问题分解为多个小域问题，是一种非常自然且巧妙的"分而治之"策略
2. **四维度统一优化**：不同于以往只关注效用-通信或效用-隐私的二元权衡，MSS 同时优化精度、带宽、计算和抗攻击四个维度
3. **无代数先决条件**：相比 PGR 需要有限域算术和射影几何约束，MSS 仅需整数运算，接受任意 $k$ 和 $\varepsilon$，实际部署门槛极低
4. **可调参数 $\ell$**：允许实践者在通信-精度谱上灵活导航，PGR 不具备此灵活性
5. **理论分析完整**：提供了精确 MSE 闭式解、条件数界限、DRA 上界等完整理论体系

## 局限与展望

1. MSE 的 $\kappa$ 倍常数因子虽然实际很小（$\leq 1.3$），但理论上仍然不是信息论最优的
2. 模选择过程（Algorithm 2-3）在极大域上可能需要较多采样试验，虽然可离线预计算并缓存
3. 仅讨论了频率估计这一基本任务，作者提到未来将扩展到 heavy hitters 和多维估计
4. 实验仅使用合成分布（Zipf 和 Spike），缺少真实联邦分析部署场景的验证
5. 正则化参数 $\lambda$ 的选择（如 $1/\varepsilon^2$）对强隐私（小 $\varepsilon$）场景可能引入不可忽略的偏差

## 相关工作与启发

- **SubsetSelection (SS)**：统计最优但通信开销高的经典方法，MSS 的"分块 SS"可视为其模块化扩展
- **PGR**：代数编码方法，MSE 最优但部署复杂度高，MSS 以略微次优的理论精度换取了极大的实用性提升
- **RAPPOR**：Google 的工业级 LDP 系统，消息长度 $O(k)$ 限制了大域场景
- 启发：数论工具（CRT、互素模）在隐私计算中有很大潜力，分而治之思想可广泛应用于大域统计估计问题

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 5 | RNS+LDP 的结合非常新颖，开辟了新方向 |
| 技术深度 | 5 | 完整的理论分析框架：无偏性、方差界、条件数界、DRA 界 |
| 实验充分性 | 4 | 四个维度全面对比，但仅合成数据 |
| 实用价值 | 4 | 部署简单、无代数约束，适合工业级联邦系统 |
| 写作质量 | 5 | 结构清晰，表1的四维对比一目了然 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](reward_redistribution_via_gaussian_process_likelihood_estimation.md)
- [\[AAAI 2026\] A New Strategy for Verifying Reach-Avoid Specifications in Neural Feedback Systems](a_new_strategy_for_verifying_reach-avoid_specifications_in_neural_feedback_syste.md)
- [\[NeurIPS 2025\] Private Evolution Converges](../../NeurIPS2025/others/private_evolution_converges.md)

</div>

<!-- RELATED:END -->
