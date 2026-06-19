---
title: >-
  [论文解读] Soft Quality-Diversity Optimization
description: >-
  提出 Soft QD Score 作为无需行为空间离散化的质量多样性优化新目标，并据此推导出可微分算法 SQUAD，在高维行为空间中具有更好的可扩展性，且在标准基准上与 SOTA 竞争力相当。 - Quality-Diversity (QD) 优化：寻找一组既高质量又行为多样的解，应用于 RL 策略多样化、红队测试、内容生…
tags:

---

# Soft Quality-Diversity Optimization

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2512.00810](https://arxiv.org/abs/2512.00810)
- **代码**: [https://github.com/conflictednerd/soft-qd](https://github.com/conflictednerd/soft-qd)
- **领域**: LLM评测
- **关键词**: quality diversity, optimization, differentiable, evolutionary computation, soft objectives

## 一句话总结
提出 Soft QD Score 作为无需行为空间离散化的质量多样性优化新目标，并据此推导出可微分算法 SQUAD，在高维行为空间中具有更好的可扩展性，且在标准基准上与 SOTA 竞争力相当。

## 研究背景与动机
- **Quality-Diversity (QD) 优化**：寻找一组既高质量又行为多样的解，应用于 RL 策略多样化、红队测试、内容生成等。
- **传统方法的局限**：
  1. 将行为空间离散化为网格/CVT 单元格，遭受维度灾难——单元格数指数增长或单元格体积指数膨胀
  2. 不可微的离散化阻碍梯度优化
- **核心问题**：能否设计无需离散化的 QD 目标，直接在连续行为空间上进行可微优化？

## 方法详解

### 整体框架
SQUAD 把质量-多样性优化从"在离散单元格里填充档案"重写成"在连续行为空间上最大化一个软目标"。它先用高斯核把每个解的质量摊成一片随距离衰减的"亮度场"，把整片空间的总亮度定义为 Soft QD Score；再推导出这个目标的一个可微下界，让 $N$ 个解通过梯度同时优化质量并互相排斥，从而绕开网格离散化带来的维度灾难。

### 关键设计

**1. 行为值与 Soft QD Score：把质量摊成连续的"亮度场"**

传统 QD 把行为空间切成网格，再往每个单元格塞最优解，单元格数随维度指数膨胀，这正是整体框架要绕开的第一个痛点。SQUAD 改为把每个解 $\theta_n$ 看成一盏照亮行为空间的光源：亮度正比于它的质量 $f_n = f(\theta_n)$，影响以高斯方式随到它行为描述符 $\mathbf{b}_n = \text{desc}(\theta_n)$ 的距离衰减，于是空间中任一点 $\mathbf{b}$ 的行为值（behavior value）取所有光源的最大亮度

$$v_{\bm{\theta}}(\mathbf{b}) = \max_{1 \leq n \leq N} f_n \exp\!\left(-\|\mathbf{b} - \mathbf{b}_n\|^2 / 2\sigma^2\right).$$

把它在整个行为空间上积分，就得到无需离散化的标量目标 Soft QD Score $S(\bm{\theta}) = \int_{\mathcal{B}} v_{\bm{\theta}}(\mathbf{b})\, d\mathbf{b}$。要让这个积分变大，就必须让高质量的解尽量分散地铺满整片空间——质量与多样性被自然地编码进同一个连续、处处定义的目标，再不依赖单元格划分。

**2. 理论性质：证明软目标行为合理且兼容旧定义**

一个新目标若想取代成熟的 QD Score，得先证明它不会退化成奇怪的东西。定理 1 给出三条性质，把"亮度场"这个直觉钉成可信的目标。单调性：新增一个解、或提升任一现有解的质量，$S(\bm{\theta})$ 都不减，符合"解越多越好、质量越高越好"的直觉。次模性（submodularity）：后加入的解边际贡献递减，意味着往已经密集的区域再塞解回报变低，优化会自然把解推向空白区域——多样性不是额外加的正则项，而是目标本身的结构使然。极限等价：当核宽 $\sigma \to 0$ 时高斯核收缩成尖峰，Soft QD Score 收敛到传统 QD Score（相差一个常数倍）。第三条尤其关键，它说明 Soft QD 不是另起炉灶，而是旧目标的连续光滑推广，$\sigma$ 就是控制"软硬程度"的旋钮。

**3. SQUAD 可微下界：把不可解的积分换成成对吸引—排斥力**

设计 1 的目标虽然连续，却卡在那个积分上——它没有闭式解，无法直接对 $\bm{\theta}$ 求梯度。SQUAD（Soft QD Using Approximated Diversity）的做法是转而最大化一个可计算且可微的下界，作为优化的实际目标：

$$\tilde{S}(\bm{\theta}) = \sum_{n=1}^N f_n - \sum_{1 \leq i < j \leq N} \sqrt{f_i f_j}\, \exp\!\left(-\|\mathbf{b}_i - \mathbf{b}_j\|^2 / \gamma^2\right).$$

第一项是质量项，像吸引力一样驱动每个解各自提升 $f_n$；第二项是多样性项，对行为接近的解对施加惩罚，像排斥力一样把它们推开，于是优化就成了在吸引与排斥之间找均衡。精髓在排斥强度用两个解质量的几何均值 $\sqrt{f_i f_j}$ 加权：只有当两个解都已经较优时排斥才显著，对低质量解之间的相似几乎不罚。这意味着差解会先沿质量项把 $f_n$ 做上去，等质量上来后排斥才生效、开始与同伴分散——一条不靠人工调度、天然涌现的"先学好、再分开"课程。超参 $\gamma$ 控制排斥的作用半径，从而直接拧动质量与多样性的权衡。

**4. 高效实现：把 $O(N^2)$ 排斥压到线性并处理有界空间**

下界目标的多样性项朴素算需要遍历所有 $O(N^2)$ 对解，群体一大就算不动。由于指数衰减让远处解的排斥力迅速可忽略，SQUAD 对每个解只计算它在行为空间里 $k$ 个最近邻的排斥，把复杂度降到 $O(Nk)$，并配合 mini-batch 分批更新压低显存；消融显示结果对 $k$ 和批大小都不敏感。另一处工程关键是下界推导假设行为空间无界（$\mathcal{B}=\mathbb{R}^d$），可现实问题（如图像坐标）的描述符常被限制在有界区间，高斯核在边界附近会失真，于是 SQUAD 先用 logit 变换把有界空间映射到无界空间再优化——这一步对有界问题至关重要，缺了它性能会明显掉。

### 一个完整示例
以同时优化 $N$ 个解为例，看一轮迭代里它们的状态如何变化。初始化 $N$ 个解后，外层循环跑 $T$ 步；每步内按 mini-batch 处理：先为批内每个解找出 $K$ 近邻，据此算出下界目标 $\tilde{S}$ 及其对参数的梯度，用 Adam 更新参数，再重新评估更新后解的质量 $f_n$ 与行为描述符 $\mathbf{b}_n$ 供下一轮使用。早期低质量解的排斥项被几何均值压低，它们主要沿质量项的梯度爬升；随着质量上来，排斥项变强、相互推开，整群解逐渐自组织成"高质量且铺满空间"的均衡分布。

## 实验关键数据

### 主实验 1：高维行为空间可扩展性（Linear Projection）

LP 基准在 4、8、16 维行为空间上对比可扩展性：

| 行为空间维度 | CMA-MAEGA / CMA-MEGA | Sep-CMA-MAE / GA-ME / DNS | **SQUAD** |
|-----------|-----------|-------------|---------|
| 4D | 略占优 | 明显落后 | 竞争力强、追平 |
| 8D | 优势缩小 | 退化 | **追平并反超** |
| 16D | 被超越 | 严重退化 | **两项指标均最优** |

> 利用描述符梯度的方法（SQUAD、CMA-M(A)EGA）远好于不用梯度的（Sep-CMA-MAE、GA-ME、DNS）；而 SQUAD 不离散化行为空间，维度越高优势越明显，且方差最低、最稳定。

### 主实验 2：图像合成 (IC) 与潜空间照亮 (LSI)

| 方法 | IC QD Score | IC Vendi Score | LSI QD Score | LSI Vendi Score |
|------|-----------|---------------|-------------|----------------|
| CMA-MAEGA | 最优级 | 较高 | 最优级 | 较高 |
| Sep-CMA-MAE | 高 | 中 | 高 | 中 |
| DNS-G | 中 | 较高 | 中 | 较高 |
| **SQUAD** | 竞争力强 | **最高** | 竞争力强 | **最高** |

> SQUAD 在多样性指标上一致最优，在 QD Score 上与 SOTA 竞争。

### 消融实验：$\gamma$ 的影响

| $\gamma$ 值 | 平均质量 | Vendi Score | 说明 |
|-----------|---------|-------------|------|
| 小 | 最高 | 最低 | 弱排斥→偏向质量 |
| 中 | 高 | 高 | 平衡 |
| 大 | 较低 | 最高 | 强排斥→偏向多样性 |

> $\gamma$ 直观控制质量-多样性权衡。

### 关键发现
1. 基于单元格的方法在高维行为空间中因维度灾难而失败
2. SQUAD 的连续目标自然避免了离散化问题
3. 排斥力中的几何均值项使低质量解先提升质量——形成自然的"先质量后多样性"课程
4. Logit 变换对有界行为空间至关重要
5. K 近邻近似对最终结果几乎无影响（因为指数衰减使远处解影响可忽略）

## 亮点与洞察
- **范式转换**：从离散单元格到连续 Soft 目标，避免维度灾难
- **优雅的物理类比**：吸引力（质量提升）+ 排斥力（多样性分散）= 自组织均衡
- **理论完备**：单调性、次模性、极限等价性提供坚实基础
- **几何均值的巧妙效果**：自动实现"低质量→先提升质量；高质量→开始分散"的课程

## 局限性
- 下界近似忽略了三阶及以上的多体交互，可能在非常紧密的群体中不准确
- $\gamma$ 超参需要调优，且与问题规模和行为空间结构相关
- 比基于 archive 的方法缺少显式的解存储，不便直接查询特定行为
- 在非可微目标函数（如需要模拟器的 RL 问题）中不直接适用

## 相关工作
- **QD 算法**: MAP-Elites (Cully et al., 2015), CMA-MEGA (Fontaine & Nikolaidis, 2021/2023)
- **可微 QD**: DQD (Fontaine & Nikolaidis, 2021), PGA-ME (Nilsson & Cully, 2021)
- **新颖性搜索**: Lehman & Stanley (2011), DNS (Bahlous-Boldi et al., 2025)
- **多样性度量**: Vendi Score (Friedman & Dieng, 2023)

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — Soft QD 重新定义了 QD 优化的目标，范式性贡献
- 理论深度: ⭐⭐⭐⭐ — 单调性、次模性、极限等价性、下界推导
- 实验充分性: ⭐⭐⭐⭐ — 三个基准域、多维度可扩展性、消融分析
- 实用价值: ⭐⭐⭐⭐ — 高维 QD 的可行方案，但限于可微目标

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](../../ICML2025/others/diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](../../AAAI2026/others/hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](../../ACL2025/others/a_multi-persona_framework_for_argument_quality_assessment.md)

</div>

<!-- RELATED:END -->
