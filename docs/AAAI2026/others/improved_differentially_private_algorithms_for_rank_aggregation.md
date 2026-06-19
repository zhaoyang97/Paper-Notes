---
title: >-
  [论文解读] Improved Differentially Private Algorithms for Rank Aggregation
description: >-
  [AAAI 2026][差分隐私] 针对差分隐私下的排名聚合问题，提出了改进的近似算法：首次研究footrule排名聚合问题并给出近最优算法（可推导出Kemeny问题的2-近似），同时通过结合二路边际查询和无偏估计技术改进了Kemeny排名聚合的PTAS加性误差（指数从3降至65/22）。 排名聚合是将多个用户对候选项的排名…
tags:
  - "AAAI 2026"
  - "差分隐私"
  - "排名聚合"
  - "Kemeny排名"
  - "footrule距离"
  - "近似算法"
---

# Improved Differentially Private Algorithms for Rank Aggregation

**会议**: AAAI 2026  
**arXiv**: [2511.11319](https://arxiv.org/abs/2511.11319)  
**代码**: 无  
**领域**: 其他  
**关键词**: 差分隐私, 排名聚合, Kemeny排名, footrule距离, 近似算法

## 一句话总结

针对差分隐私下的排名聚合问题，提出了改进的近似算法：首次研究footrule排名聚合问题并给出近最优算法（可推导出Kemeny问题的2-近似），同时通过结合二路边际查询和无偏估计技术改进了Kemeny排名聚合的PTAS加性误差（指数从3降至65/22）。

## 研究背景与动机

排名聚合是将多个用户对候选项的排名组合成一个最优共识排名的基本问题，在选举投票、搜索结果排序等场景中有广泛应用。该问题有两个经典的最优性准则：基于Kendall tau距离的Kemeny排名（NP-hard）和基于位置偏差的footrule排名（多项式时间可解）。

由于输入排名通常包含用户敏感信息（如投票偏好），在保护隐私的前提下进行排名聚合至关重要。差分隐私（DP）作为隐私保护的金标准，为此提供了严格的数学保障。Alabi等人（AAAI'22）提出了DP下Kemeny排名聚合的PTAS和5-近似算法，但存在两个核心矛盾：

- PTAS虽能达到 $\alpha = 1+\xi$ 的乘法因子，但加性误差 $\beta = \tilde{O}(m^3/\varepsilon n)$ 较大
- 5-近似算法虽有较小的加性误差 $\beta = \tilde{O}(m^{2.5}/\varepsilon n)$，但乘法因子 $\alpha = 5+\xi$ 过大

同时，footrule排名聚合在DP下从未被研究过。本文的核心idea是：通过首先解决footrule问题（利用改进的二叉树机制），既能得到footrule的近最优DP算法，又能通过footrule与Kendall tau距离的关系推导出Kemeny问题的2-近似，同时利用分桶+二路边际查询技术改进PTAS。

## 方法详解

### 整体框架

本文提出两个主要算法：(1) 基于近似中位数（ApxMed）的footrule排名聚合算法，可衍生为Kemeny问题的 $(2, \beta)$-近似算法；(2) 改进的Kemeny排名聚合PTAS，分"大$n$"和"小$n$"两种情况处理。

### 关键设计

1. **改进的二叉树机制（Modified Binary Tree Mechanism）**:

    - 功能：高效回答差分隐私下的近似中位数（ApxMed）查询，即对每个候选位置 $j$ 估计 $\gamma_j(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n |x_i - j|$
    - 核心思路：在经典二叉树机制的基础上进行扩展。原始机制用于回答区间频率查询，本文将每个节点 $t$ 存储两个聚合值：$v_t^{agg} = \frac{1}{n}\sum_{x_i \in I(t)}|x_i - r(t)|$（位置偏差之和）和 $u_t^{agg} = \frac{1}{n}\sum_{x_i \in I(t)} 2^{\ell(t)}$（计数缩放量）。通过兄弟节点的信息可以重建 $\gamma_j$ 的估计
    - 设计动机：朴素方法中每个数据点 $\pi_i(q)$ 影响最多 $m$ 个输出值，导致隐私预算被过度分割、噪声过大。二叉树结构确保每个数据点仅出现在 $O(\log m)$ 个区间中，大幅降低噪声。此外引入权重因子 $\kappa^{d-\ell(t)}$ 进一步减少对数因子的误差
    - 精度保证：$\varepsilon$-DP下 $\beta = O(m\log m / \varepsilon n)$，$(ε,δ)$-DP下 $\beta = O(m\sqrt{\log m \log(1/\delta)} / \varepsilon n)$

2. **从ApxMed到Footrule排名聚合（Algorithm 2）**:

    - 功能：利用并行ApxMed解决footrule排名聚合
    - 核心思路：将footrule排名聚合建模为最小权二部匹配。对每对候选 $q$ 和位置 $j$，分配代价 $\gamma_q^j = \frac{1}{n}\sum_{i=1}^n |\pi_i(q) - j|$。先用 $m$-并行ApxMed私有地估计所有权重，再求解最小权二部匹配
    - 设计动机：由于Spearman footrule距离介于Kendall tau距离的1倍到2倍之间，footrule的 $(1,\beta)$-近似直接给出Kemeny的 $(2,\beta)$-近似，将乘法因子从5降至2

3. **改进的Kemeny PTAS——大$n$情况（Leveraging Unbiasedness）**:

    - 功能：当 $n = \Omega_{\varepsilon,\delta}(m\log m)$ 时，通过无偏高斯噪声扰动权重矩阵
    - 核心思路：对权重矩阵 $\mathbf{w}^\Pi$ 加独立高斯噪声 $\mathcal{N}(0, \sigma^2)$ 得到私有化矩阵 $\tilde{\mathbf{w}}$。关键发现是：对于 $w_{uv} < w_{vu}$ 的情况，可以将 $w_{uv}$ 替换为0、$w_{vu}$ 替换为 $w_{vu} - w_{uv}$，只对后者加噪声即可避免截断偏差，同时保证高概率下的非负性
    - 设计动机：直接加噪声可能产生负值，非私有PTAS无法处理；截断修复会引入 $O(m^3)$ 的偏差。本文的无偏化处理在 $\sigma = O(1/\log m)$ 下即可保证非负性

4. **改进的Kemeny PTAS——小$n$情况（Reduction to 2-Way Marginals, Algorithm 3）**:

    - 功能：当 $n$ 较小时，通过分桶将Kemeny排名编码为二路边际查询
    - 核心思路：将排名值域 $[m]$ 分为 $B$ 个桶。对同桶内的比较用标准Laplace/Gaussian机制处理（灵敏度降低），对跨桶比较编码为二路边际查询 $t_{uv} = \frac{1}{n}\sum_{i} \mathbf{1}[\iota(\pi_i(u)) < \iota(\pi_i(v))]$，利用DP二路边际算法的优越精度保证
    - 设计动机：DP二路边际算法在小 $n$ 时的误差 $O_{\varepsilon,\delta}(m^{1/4}/\sqrt{n})$ 远优于标准噪声添加。通过优化桶数 $B$，最终在 $(ε,δ)$-DP下将PTAS加性误差从 $m^3$ 降至 $m^{65/22} \approx m^{2.955}$

### 理论结果汇总

本文的核心理论贡献可概括为以下定理：

- **Footrule排名聚合**（Theorem 4.1）：$(1, \beta)$-近似，$\varepsilon$-DP下 $\beta = O(m^3 \log m / \varepsilon n)$，$(ε,δ)$-DP下 $\beta = O(m^{2.5}\sqrt{\log m \log(1/\delta)} / \varepsilon n)$，均近最优
- **Kemeny 2-近似**（Theorem 4.2）：$(2, \beta)$-近似，加性误差与footrule相同
- **Kemeny PTAS**（Theorem 5.1）：$(1+\xi, \beta)$-近似，$(ε,δ)$-DP下 $\beta = \tilde{O}_\xi(m^{65/22}\sqrt{\log(1/\delta)} / \varepsilon n)$

## 实验关键数据

### 主结果（理论界对比）

本文为纯理论工作，无实验数据。以下以定理形式总结核心结果：

| 隐私模型 | 近似比 $\alpha$ | 加性误差 $\beta$（$m$ 的指数） | 来源 |
|----------|----------------|-------------------------------|------|
| $(\varepsilon,\delta)$-DP | $1+\xi$ | $m^3$ | Alabi et al. |
| $(\varepsilon,\delta)$-DP | $5+\xi$ | $m^{2.5}$ | Alabi et al. |
| $(\varepsilon,\delta)$-DP | $1+\xi$ | $m^{65/22} \approx m^{2.955}$ | **本文** |
| $(\varepsilon,\delta)$-DP | $2$ | $m^{2.5}$ | **本文** |
| $(\varepsilon,\delta)$-DP | 任意 $\alpha \geq 1$ | $m^{2.5}$（下界） | Alabi et al. |

### 不同DP模型下的Footrule结果

| DP模型 | 近似比 | 加性误差 | 最优性 |
|--------|--------|---------|--------|
| $\varepsilon$-DP | 1 | $O(m^3 \log m / \varepsilon n)$ | 近最优 |
| $(\varepsilon,\delta)$-DP | 1 | $O(m^{2.5}\sqrt{\log m \log(1/\delta)} / \varepsilon n)$ | 近最优 |
| $\varepsilon$-LDP | 1 | $O(m^{2.5}\sqrt{\log m} / \varepsilon\sqrt{n})$ | — |

### 关键发现

- Footrule排名聚合在 $\varepsilon$-DP 和 $(ε,δ)$-DP 下的加性误差已接近理论下界（仅差对数因子）
- 2-近似算法在LDP模型下是非交互式的（Alabi et al.的算法需要 $O(\log m)$ 轮通信）
- PTAS的加性误差指数从3降到65/22，朝着下界2.5迈出了重要一步

## 亮点与洞察

- 首次研究footrule排名聚合的差分隐私问题，并通过footrule-Kendall tau距离关系巧妙地改进了Kemeny问题的乘法近似比
- 改进的二叉树机制设计精巧：通过存储位置偏差聚合值（而非仅频率），将经典范围查询机制扩展到了中位数估计，具有独立研究价值
- 小 $n$ 情况下的分桶+二路边际查询归约是本文最具技术含量的贡献，巧妙利用了DP文献中对二路边际查询的深入研究成果
- 无偏化技巧（用 $w_{vu} - w_{uv}$ 替代 $w_{uv}$）简洁而有效地解决了高斯噪声引入负值的问题

## 局限与展望

- PTAS的加性误差指数65/22 ≈ 2.955与下界2.5之间仍有差距，尤其在 $\varepsilon$-DP下差距更大（27/7 ≈ 3.857 vs 下界3）
- 纯理论工作，缺少实验验证算法在实际数据集上的表现（如真实投票数据、推荐系统排名等）
- 改进的PTAS需要分大 $n$ / 小 $n$ 两种情况分别处理，算法的实现复杂度较高
- LDP模型下的PTAS结果未给出，仅有2-近似

## 相关工作与启发

- Alabi et al. (AAAI'22) 奠定了DP排名聚合的理论基础，本文在此基础上系统性改进
- 二叉树机制（Dwork et al., Chan et al.）是DP中处理连续查询的经典工具，本文展示了其向新问题的扩展方式
- 二路边际查询的DP算法（Nikolov, Dwork et al.）在小样本量时的优异表现，被本文巧妙地用于排名聚合场景
- 启发：DP问题的算法改进往往来自对问题结构的深入挖掘（如分桶、无偏化），而非简单地套用通用DP机制

## 评分

- 新颖性: ⭐⭐⭐⭐ （理论贡献扎实但属于改进型工作）
- 实验充分度: ⭐⭐ （纯理论，无实验验证）
- 写作质量: ⭐⭐⭐⭐⭐ （证明严谨，结构清晰）
- 价值: ⭐⭐⭐⭐ （缩小了上下界差距，首次研究footrule DP问题）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](../../ICLR2026/others/missing_mass_for_differentially_private_domain_discovery.md)
- [\[CVPR 2026\] DP-FedAdamW: An Efficient Optimizer for Differentially Private Federated Large Models](../../CVPR2026/others/dp-fedadamw_an_efficient_optimizer_for_differentially_private_federated_large_mo.md)
- [\[AAAI 2026\] Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)

</div>

<!-- RELATED:END -->
