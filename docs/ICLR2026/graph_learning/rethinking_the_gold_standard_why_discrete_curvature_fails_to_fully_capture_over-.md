---
title: >-
  [论文解读] Rethinking the Gold Standard: Why Discrete Curvature Fails to Fully Capture Over-squashing in GNNs?
description: >-
  [ICLR 2026][图学习][over-squashing] 这篇论文系统反驳了"高负曲率 = 过度挤压"这个图学习领域的金标准：通过构造反例图族证明高负曲率只是过度挤压的**充分非必要**条件，提出 MOSR 指标量化曲率漏检了 30%~40% 的挤压边，并给出新的加权曲率 WAF3 及其线性时间 MinHash 近似算法（500 万边图上 23.6 秒，比现有最快曲率快 133.7 倍）。
tags:
  - "ICLR 2026"
  - "图学习"
  - "over-squashing"
  - "离散曲率"
  - "Forman 曲率"
  - "图重连"
  - "MinHash"
---

# Rethinking the Gold Standard: Why Discrete Curvature Fails to Fully Capture Over-squashing in GNNs?

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QYtmqCoilk](https://openreview.net/forum?id=QYtmqCoilk)  
**代码**: 论文附录提供匿名链接（待确认）  
**领域**: 图学习 / GNN 理论 / 离散曲率 / 过度挤压  
**关键词**: over-squashing、离散曲率、Forman 曲率、图重连、MinHash

## 一句话总结
这篇论文系统反驳了"高负曲率 = 过度挤压"这个图学习领域的金标准：通过构造反例图族证明高负曲率只是过度挤压的**充分非必要**条件，提出 MOSR 指标量化曲率漏检了 30%~40% 的挤压边，并给出新的加权曲率 WAF3 及其线性时间 MinHash 近似算法（500 万边图上 23.6 秒，比现有最快曲率快 133.7 倍）。

## 研究背景与动机

**领域现状**：过度挤压（over-squashing）是消息传递图神经网络（MPNN）的核心顽疾——当信息要跨越图中的"瓶颈"结构从源节点流向远处目标节点时，会被压缩进狭窄通道而严重失真。Topping 等人（2021）提出了一个被广泛接受的判据：**高负离散曲率的边就是制造瓶颈、导致过度挤压的边**。这一观点催生了大量后续工作，包括基于曲率的图重连（rewiring）、曲率启发的 GNN 架构等，离散曲率俨然成了检测过度挤压的"金标准"。

**现有痛点**：但 Topping 等人其实只证明了"高负曲率 ⟹ 过度挤压"这一个方向（充分性），从未证明反方向（必要性）。也就是说，没有人验证过"所有被严重挤压的边是否都表现出高负曲率"。这个看似细微却至关重要的缺口长期被整个社区忽视——如果必要性不成立，那么大量基于曲率筛选挤压边的方法会系统性地漏掉一批真正被挤压的边。

**核心矛盾**：离散曲率（无论是 Ollivier-Ricci、Balanced Forman 还是 Augmented Forman）本质上只刻画一条边 $(u,v)$ 的**一阶邻域**有多紧密相连，这种"紧密度"主要由边参与的三角形数量决定。而过度挤压的真实度量是雅可比范数 $\|\partial h_t^{(L)}/\partial h_s^{(0)}\|$，它依赖于**多跳的信息传播路径**。一阶局部量去近似多跳全局现象，天然存在盲区。

**本文目标**：拆成四个环环相扣的子问题——（1）必要性到底成不成立？（理论）（2）实际图上曲率漏检了多少挤压边、漏在哪里？（实证）（3）能不能设计一个漏检更少的新曲率？（方法）（4）新曲率能不能在亿级边的大图上算得动？（算法）

**切入角度**：作者从一个反直觉的观察出发——只要让边 $(u,v)$ 的一阶邻居各自再连出大量"扇出"节点，信息每经过一个一阶邻居就被这些扇出节点"稀释"一次，多跳传播下挤压会非常严重；但 $(u,v)$ 自身的三角形数量却拉满，曲率反而高度为正。这个观察精准地撕开了"局部紧密 ≠ 全局畅通"的裂缝。

**核心 idea**：用"加权"修正曲率对高度数节点的错误计数（高度数节点其实对信息流贡献很小），把基于三角形计数的 Augmented Forman-3 升级为 WAF3，并把它重写成加权 Jaccard 相似度从而用 MinHash 线性加速。

## 方法详解

### 整体框架

这是一篇"先破后立"的理论+方法论文，整条线索是：**反例证伪金标准 → 指标量化漏检 → 新曲率修补 → 近似算法落地**。

第一步（破），作者构造一个反例图族 $G^c_{n,m}$，理论证明其中存在被严重挤压、但 8 种主流离散曲率都判为高度正值的边，从而否定"高负曲率"作为过度挤压的必要条件。第二步（量化），定义 Missed Over-Squashing Ratio（MOSR）指标，以雅可比范数为真值，统计有多少真正被挤压的边被曲率漏掉，在 21 个数据集上实测 Ollivier-Ricci 漏检可超 30%，并用边介数（edge betweenness）揭示漏检边的位置规律——曲率只擅长抓"簇间桥边"，而簇内部的挤压边被系统性忽略。第三步（立），把 Augmented Forman-3 改写成"三角形内节点数 − 三角形外节点数"的等价形式，加上按度数加权的修正函数 $f$，得到 WAF3，并证明只要 $f(+\infty)=0^+$ 就能消除前述反例。第四步（落地），把 WAF3 重写为加权 Jaccard 相似度，用 MinHash 把每条边的计算降到常数复杂度 $O(H|E|)$，达到理论下界。

由于这是机制分析与指标/算法设计，并非多模块串行的可视化 pipeline，这里不强行画框架图，下面的关键设计按"反例 → 指标 → 新曲率 → 近似"四个环节依次展开。

### 关键设计

**1. 反例图族 $G^c_{n,m}$：用"扇出稀释"证伪曲率必要性**

针对"高负曲率是过度挤压必要条件"这一未经验证的假设，作者构造了一个可控的反例图族。$G^c_{n,m}$ 含源节点 $s$、目标节点 $t$、$n$ 个一阶邻居 $N_1=\{u_i\}$（每个 $u_i$ 同时连 $s$ 和 $t$）以及 $n\times m$ 个二阶邻居 $N_2=\{v_{ij}\}$（每个 $u_i$ 再扇出 $m$ 个 $v_{ij}$）。它的巧妙之处在于把两个量反向拉扯：一方面 $(s,t)$ 恰好凑满 $n$ 个三角形 $s-u_i-t$，达到任意边三角形数的上限 $\min\{d_s-1,d_t-1\}$，于是几乎所有曲率定义下 $\mathrm{Curv}(s,t)$ 都高度为正；另一方面信息从 $s$ 流向 $t$ 必须经过 $N_1$，而每个 $u_i$ 都连着 $m$ 个 $v_{ij}$，信息每过一次 $u_i$ 就被 $m$ 个节点稀释。

Theorem 4 严格地证明了这种背离：当 $m\to+\infty$ 时，雅可比范数以 $O(m^{-1})$ 的速度趋向其理论下界（即挤压趋于最严重）

$$\phi_L(a,b) := \prod_{l=0}^{L-1}\|W^{(l)}\|\left(\tfrac{1}{a+1}+\tfrac{1}{b+1}\right)^{L-1}\frac{\rho}{\sqrt{(a+1)(b+1)}}$$

但 $\alpha$-Ollivier-Ricci、Lin-Lu-Yau、Balanced Forman（含/不含 4-cycle）、Jost-Liu、Augmented Forman-3/4 这 8 种曲率全都满足 $\mathrm{Curv}(s,t)>c>0$。根本原因被作者一句话点透：离散曲率只看一阶邻居，而 GNN 至少要两轮消息传递，一阶量注定捕捉不到多跳路径上的稀释效应。

**2. MOSR：把"漏检了多少挤压边"变成可量化指标**

反例只说明"存在"漏检，但实际图上漏检有多严重需要一把尺子。作者定义 Missed Over-Squashing Ratio。先以雅可比范数 $\mathrm{JacoNorm}(u,v):=\|\partial h_v^{(L)}/\partial h_u^{(0)}\|_F$ 作为过度挤压的真值度量；再把曲率筛出的高负曲率边集合记为 $E_q := \{e\in E \mid \mathrm{Curv}(e)\le \mathrm{Percentile}(C^-,q)\}$（$C^-$ 是所有负曲率值的集合，$q$ 越小判据越严），令 $J_q:=\max_{(u,v)\in E_q}\mathrm{JacoNorm}(u,v)$ 作为"被曲率认可的挤压程度阈值"。MOSR 定义为：

$$\mathrm{MOSR}_q := \frac{\sum_{(u,v)\in E}\mathbf{1}_{\mathrm{Curv}(u,v)\ge 0}\cdot \mathbf{1}_{\mathrm{JacoNorm}(u,v)\le J_q}}{\sum_{(u',v')\in E}\mathbf{1}_{\mathrm{JacoNorm}(u',v')\le J_q}}$$

分子数的是"曲率非负、却同样被挤压"的边（即被曲率漏掉的），分母数的是所有真正被挤压的边，比值就是漏检率。这个指标的关键价值在于它直接以模型真值（雅可比范数）为基准，而非曲率自己的内部一致性，因此能客观暴露曲率的盲区。配套地，作者还用边介数 $\mathrm{Between}(e)=\sum_{u\ne v}\sigma_{uv}(e)/\sigma_{uv}$ 定义 BetwIden / BetwAll / BetwIgno 三个统计量，刻画"被识别 / 全部 / 被漏检"三类边的平均介数，从而定位漏检边的拓扑位置。

**3. WAF3：用度数加权修正"高度数节点被误计"**

实验发现 Augmented Forman-3（AF3）在所有曲率里复杂度最低、实际漏检最少，于是作者选它做改进。先把 AF3 改写成一个直观的等价形式：

$$AF3(u,v) = 4 - d_u - d_v + 3\triangle(u,v) = \underbrace{|B(u)\cap B(v)|}_{\text{三角形内节点数}} - \underbrace{\big(|N(u)/B(v)| + |N(v)/B(u)|\big)}_{\text{三角形外节点数}}$$

其中 $N(v)$ 是 $v$ 的一阶邻居，$B(v)=N(v)\cup\{v\}$ 是 $v$ 的闭邻域。这说明 AF3 本质是"构成三角形的节点数 减去 剩余一阶邻居数"。但正如反例揭示的，三角形计数把每个节点同等对待，而高度数节点其实对从源到目标的信息流贡献很小（它把信息稀释了）。WAF3 因此给每个节点按度数加一个修正权重 $f:\mathbb{R}\to\mathbb{R}$：

$$\mathrm{WAF3}_f(u,v) := \sum_{i\in B(u)\cap B(v)} f(d_i) - \Big(\sum_{i\in N(u)/B(v)} f(d_i) + \sum_{i\in N(v)/B(u)} f(d_i)\Big)$$

AF3 是 $f\equiv 1$ 的特例。Theorem 5 证明只要 $f(+\infty)=0^+$（高度数节点权重趋零），WAF3 就能摆脱第 1 点构造的反例——这正是它与所有旧曲率的本质区别。论文用 GCN 风格的权重 $f(x)=1/(1+x)$，既满足条件又保持 $O(1)$ 的单点复杂度，使 WAF3 的总复杂度仍与最快的 AF3 持平。

**4. MinHash 近似：把交集运算变成 Jaccard 相似度，复杂度压到线性**

WAF3 复杂度中的 $d_{\max}$ 因子来自集合交集 $B(u)\cap B(v)$，而大图上 $d_{\max}$ 会随节点数增长，这是曲率工具难以推广到大规模图学习的瓶颈。作者的破解办法是 Theorem 6：把 WAF3 等价改写成基于**加权 Jaccard 相似度**的形式

$$\mathrm{Jaccard}_f(N(u),N(v)) := \frac{\sum_{i\in N(u)\cap N(v)} f(d_i)}{\sum_{i\in N(u)\cup N(v)} f(d_i)}$$

代入后 WAF3 只剩下加权 Jaccard 项与各自邻居加权和，不再需要显式求交集。而加权 Jaccard 相似度恰好有成熟的加速手段——加权 MinHash 通过采样 $H$ 个哈希函数把单边相似度计算降到常数 $O(H)$，$H$ 越大近似误差越小。于是整图复杂度降到 $O(H|E|)$，达到"每条边常数复杂度"的理论下界。这还催生了一个新范式：先用近似算法快速筛出候选边，再对候选集做精确计算确定最终的高负曲率边。

### 损失函数 / 训练策略
本文不涉及新的训练目标——它是对"曲率作为过度挤压判据"的理论分析与曲率/算法设计。雅可比范数作为过度挤压真值时基于 Assumption 1（计算图中所有路径以相同概率 $\rho$ 被激活）做解析，曲率计算本身无需训练。MinHash 的哈希数 $H$ 是唯一关键超参（$H\in\{100,1000,10000\}$）。

## 实验关键数据

实验覆盖 3 种主流曲率（Ollivier-Ricci、Augmented Forman-3、Balanced Forman）× 3 种 GNN（GCN / GAT / GraphSAGE）× 21 个数据集，共 350 个 MOSR 数值。

### 主实验：曲率漏检率 MOSR（GCN 列，$q=25$ 部分摘录）

| 数据集 | Ollivier-Ricci | Augmented Forman-3 | WAF3（本文，GCN） |
|--------|---------------|--------------------|------------------|
| Cora | 0.103 | 0.027 | 0.014 |
| Citeseer | 0.151 | 0.034 | 0.040 |
| Photo | 0.503 | 0.043 | 0.024 |
| Chameleon | 0.643 | 0.116 | 0.066 |
| Squirrel | 0.723 | 0.137 | 0.039 |
| **平均（GCN, MOSR₁₀/₂₅）** | .271/.307 | .067/.079 | **.036/.045** |

- Observation 1：在 $q=10$ 下离散曲率系统性漏检 6.7%~38.6% 的挤压边，$q=25$ 时升至 7.9%~39.8%，证明曲率无法完美识别过度挤压。
- Observation 3：复杂度最高的 Ollivier-Ricci（$O(|E|d_{\max}^3)$）漏检最严重，复杂度最低的 AF3 反而平均最好，作者据此推荐在 GNN 中优先用 AF3 系。
- WAF3（$f=1/(1+x)$）把 GCN 平均 MOSR 进一步从 AF3 的 6.7%/7.9% 降到 **3.6%/4.5%**，相比已是最佳的 AF3 再降 3% 以上（Observation 6）。

### 漏检边位置分析（边介数，GCN, $q=25$）

| 统计量 | 含义 | 典型结论 |
|--------|------|---------|
| BetwIden | 被曲率识别的边平均介数 | 多数情况 BetwIden > BetwAll |
| BetwAll | 全部边平均介数 | 基准 |
| BetwIgno | 被曲率漏检的边平均介数 | 多数情况 BetwAll > BetwIgno |

- Observation 4 & 5：BetwIden > BetwAll > BetwIgno 说明曲率只擅长识别充当"簇间桥梁"的高介数边，而**簇内部**的低介数挤压边被系统性忽略。作者据此重新定性：曲率不是"识别过度挤压"的金标准，而只是"识别簇间桥边"的金标准。

### 近似算法效率与精度（ER 随机图，$p=0.0005$，24GB 显存上限）

- Observation 7：当随机图规模达 $10^5$ 节点 / $5\times10^6$ 边时，ORC、BFC 已无法在可忍受时间内算完，精确 WAF3 也需 3000+ 秒；MinHash 近似在 $H=100$ 时仅 23.6 秒，**加速 133.7 倍**。
- Observation 8：$H=100$ 时近似值与精确值的 Kendall Tau-b 排序相似度约 95%（仅 2.5% 边对错序），$H=1000$ 升至 98%+，$H=10000$ 的边际提升不足 1%。由于重连方法只关心曲率最小的那批边的相对排序，95% 排序一致已足够实用。

### 关键发现
- 三角形计数的盲区是核心病灶：高度数节点贡献了三角形数却几乎不传递信息，导致 $G^c_{n,m}$ 这类"扇出稀释"结构被曲率误判为高正曲率。
- 模型架构显著影响识别效果：曲率在 GAT 上优于 GraphSAGE，而 GCN 上最优（Observation 2），说明过度挤压的真值本身与模型耦合。
- 排序一致性比绝对值更重要：这一观察让 MinHash 近似在实践中可用——重连只需挑出曲率最小的边。

## 亮点与洞察
- **证伪而非新建**：论文最"啊哈"的地方是把社区默认的"曲率=过度挤压判据"拆成充分性与必要性两面，指出大家只证了一半就当全部用了，并用一个极简可控的反例图族 $G^c_{n,m}$ 把必要性证伪——这种"先破后立"的批判性贡献在以堆 SOTA 为主的图学习圈很稀缺。
- **MOSR 把哲学问题变成可测量指标**：以雅可比范数为真值、用百分位数定义高负曲率边集，让"曲率到底漏了多少"从口头争论变成 350 个可复现数字，这套测量框架本身可迁移到评估任何"代理判据 vs 真值"的场景。
- **等价改写解锁加速**：把 WAF3 重写成加权 Jaccard 相似度（Theorem 6）是点睛之笔——它把一个看似只能暴力求交集的曲率，接到了成熟的 MinHash 加速生态上，复杂度直接打到理论下界。这种"换个等价形式就能复用已有高效算法"的思路值得借鉴。
- **加权修正的最小充分条件**：Theorem 5 只要求 $f(+\infty)=0^+$ 就能消除反例，给出了设计权重函数的明确边界，而非拍脑袋选 $1/(1+x)$。

## 局限与展望
- WAF3 仍只看一阶邻域加权，并未真正引入多跳信息——它修补的是"高度数节点误计"这个盲区，但 Theorem 4 揭示的"一阶量无法捕捉多跳稀释"的根本张力，加权能缓解到什么程度仍有边界，某些异配数据集（如 Amazon-ratings）上 GAT/SAGE 的 MOSR 仍偏高。
- 雅可比范数作为真值依赖 Assumption 1（所有路径等概率 $\rho$ 激活）这一简化假设，真实训练后的模型路径激活并不均匀，真值本身可能有偏。
- 论文以"检测准确性（MOSR）"为主轴，对"用 WAF3 做图重连后下游任务精度提升多少"只在附录涉及，缺少与现有曲率重连方法在端到端任务上的正面对比。
- MinHash 近似的 95% 排序一致虽够用，但 2.5% 错序边在严格筛选最小曲率边的场景下仍可能引入噪声，对极端稀疏/稠密图的鲁棒性未充分讨论。

## 相关工作与启发
- **vs Topping et al. (2021, Balanced Forman + SDRF)**：他们确立了"高负曲率边导致瓶颈"并据此重连，本文证明这只是充分非必要条件，曲率系统性漏掉簇内挤压边，因此把曲率重新定性为"簇间桥边检测器"而非"过度挤压检测器"。
- **vs Di Giovanni et al. (2023)**：他们从宽度/深度/拓扑角度分析过度挤压并用雅可比范数刻画，本文沿用雅可比范数作真值，但把焦点放在"曲率作为代理量的漏检率"这一前人未量化的问题上。
- **vs Augmented Forman 系（Forman 2003 / Tori et al. 2024）**：本文以 AF3 为基座，通过度数加权 $f$ 升级为 WAF3，在保持最低复杂度的同时把漏检率再降 3%+，并补上 MinHash 近似让它能跑大图。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次证伪"曲率=过度挤压"的必要性，批判性与建设性兼具
- 实验充分度: ⭐⭐⭐⭐ 21 数据集 × 3 曲率 × 3 GNN 共 350 个 MOSR，但下游任务对比偏弱
- 写作质量: ⭐⭐⭐⭐⭐ 四个子问题环环相扣，理论—实证—方法—算法叙事清晰
- 价值: ⭐⭐⭐⭐⭐ 纠正了社区一个被默认多年的隐含假设，并给出可落地的替代曲率与加速算法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity](glstm_mitigating_over-squashing_by_increasing_storage_capacity.md)
- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](cords_-_continuous_representations_of_discrete_structures.md)
- [\[ICLR 2026\] Discrete Bayesian Sample Inference for Graph Generation](discrete_bayesian_sample_inference_for_graph_generation.md)
- [\[ICLR 2026\] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)

</div>

<!-- RELATED:END -->
