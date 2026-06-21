---
title: >-
  [论文解读] Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data
description: >-
  [ICLR2026][因果推理][后处理选择偏差] 本文指出干预因果发现中一个长期被忽视的难题——**后处理选择**（intervention 后才按质控标准筛样本，如单细胞实验只保留高活性细胞），它会伪装成因果响应使现有方法把"有无直接因果边"误判为同一等价类；作者用增广 DAG 显式建模选择变量，提出比传统等价类更细的 **FI-Markov 等价**与新图表示 **F-PAG**，并给出可证明 sound & complete 的 **F-FCI** 算法，能从观测+干预数据中同时辨认因果关系、潜在混杂与后处理选择。
tags:
  - "ICLR2026"
  - "因果推理"
  - "后处理选择偏差"
  - "潜在混杂"
  - "干预因果发现"
  - "Markov 等价类"
  - "F-PAG"
---

# Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=qclNnbjxNJ](https://openreview.net/forum?id=qclNnbjxNJ)  
**代码**: https://github.com/GongxuLuo/F-FCI  
**领域**: 因果发现 / 干预因果学习  
**关键词**: 后处理选择偏差、潜在混杂、干预因果发现、Markov 等价类、F-PAG  

## 一句话总结
本文指出干预因果发现中一个长期被忽视的难题——**后处理选择**（intervention 后才按质控标准筛样本，如单细胞实验只保留高活性细胞），它会伪装成因果响应使现有方法把"有无直接因果边"误判为同一等价类；作者用增广 DAG 显式建模选择变量，提出比传统等价类更细的 **FI-Markov 等价**与新图表示 **F-PAG**，并给出可证明 sound & complete 的 **F-FCI** 算法，能从观测+干预数据中同时辨认因果关系、潜在混杂与后处理选择。

## 研究背景与动机
**领域现状**：干预因果发现（interventional causal discovery）的主流套路是利用"干预带来的分布变化"来定向因果边。经典的判据是一组跨干预模式：干预了"因"后，"果"的边缘分布 $p(\text{effect})$ 会变，但条件分布 $p(\text{effect}\mid\text{cause})$ 不变；反过来干预"果"则 $p(\text{cause})$ 不变、$p(\text{cause}\mid\text{effect})$ 变。GIES、IGSP、FCI-with-intervention 等方法都建立在这套不变性分析之上，并已被扩展到含潜在混杂（latent confounder）的场景。

**现有痛点**：作者强调一个在生物实验里普遍存在、却被现有框架完全忽略的问题——**后处理选择（post-treatment selection）**，即样本是在干预**之后**才被选择性纳入数据集的。典型例子：基因扰动实验里只有通过质控（如高活性）的扰动细胞才被测序；临床试验 per-protocol 分析里只保留完成 80% 以上随访的受试者。问题在于，后处理选择产生的统计指纹——$p(\text{effect})$ 变而 $p(\text{effect}\mid\text{cause})$ 不变——**和真因果关系的指纹一模一样**。

**核心矛盾**：因为指纹相同，现有干预因果框架把"$X_1$ 和 $X_2$ 之间确有直接因果边"与"两者只是被同一选择变量牵连、并无直接边"放进了**同一个等价类**（论文 Figure 1(a) vs (b)），从而既无法区分因果关系与后处理选择，也检测不出选择究竟发生在哪。这是一个**表示能力的缺口**：现有 DAG/MAG/PAG 的等价类粒度太粗，根本表达不了这种区别。

**本文目标**：在同时含潜在混杂 $L$ 与选择 $S$ 的一般设定下，(1) 建立能显式表达后处理选择的因果形式化；(2) 刻画其 Markov 性质并定义一个更细的干预等价类；(3) 给出一个可证明正确、完备的算法把它从数据里学出来。

**切入角度**：作者的关键观察是——虽然在"因—果"这一对端点上后处理选择和因果关系不可分，但只要把视线放宽到**路径上的中间变量**，二者会在更高阶的对称性与干预反应上露出马脚。例如对中间节点 $X_3$ 做硬干预，能"打开"被选择效应堵住的路径，从而用 $\psi_3 \not\perp\!\!\!\perp X_2$ 这类信号去判定 $X_1$、$X_2$ 间到底有没有直接因果边。

**核心 idea**：用"增广 DAG + 选择变量 + 中间诱导节点上的额外硬干预"，把传统等价类细化为 **FI-Markov 等价**，并配一套带新边型的图表示 **F-PAG** 与算法 **F-FCI**，越过粗等价类、直达 DAG 级别的真结构。

## 方法详解

### 整体框架
本文要解决的是"后处理选择伪装成因果"这一不可识别性。整体思路分三层递进：**先建模**——把选择变量 $S$ 和干预指示子 $\psi$ 一起塞进增广 DAG，让观测数据与各次干预数据统一在一张图里；**再刻画**——分析这张图的 Markov 性质，弄清两张不同的增广 DAG 在什么条件下会产生相同的条件独立（CI）模式，由此定义一个比传统等价类更细的 FI-Markov 等价，并设计 F-PAG 图来唯一表示它；**最后学习**——给出约束式算法 F-FCI，从观测+干预数据里恢复 F-PAG，并证明它 sound（输出的标记都正确）且 complete（能辨认的结构都辨认出来）。

数据生成被写成一个在 $S=1$（即"已被选中"）条件下的因子分解：第 $k$ 次干预下的联合分布

$$p^{(k)}_s(X) = \prod_{i\in I^{(k)}} p^{(k)}\!\big(X_i \mid \hat X_{\mathrm{pa}_G(i)}, S{=}1\big)\;\prod_{j\notin I^{(k)}} p^{(0)}\!\big(X_j \mid \hat X_{\mathrm{pa}_G(j)}, S{=}1\big),$$

其中被干预的变量 $X_i\,(i\in I^{(k)})$ 换成干预后的条件分布，其余变量保持观测分布，且全程对 $S=1$ 取条件——这正是"所有样本（无论观测还是干预）都先过了选择"的写照。

### 关键设计

**1. 用增广 DAG 显式建模后处理选择：把选择变量 $S$ 与干预指示子 $\psi$ 一起请进图里**

现有框架之所以分不开后处理选择和因果，是因为它们的图模型里压根没有"选择"这个对象。本文先把"改变干预目标"这个动作显式化：在原 DAG $G$（顶点为观测变量 $X$、潜在混杂 $L$）之外，加一组外生二值指示子 $\psi=\{\psi_{I^{(k)}}\}$，每个指向它所干预的变量 $X_{I^{(k)}}$。于是"第 $k$ 次干预是否改变了 $X_A$ 的边缘分布"这件事，被非参地翻译成 CI 关系 $\psi_{I^{(k)}}\not\perp\!\!\!\perp X_A$，进一步对应增广 DAG 里的 d-分离 $\psi_{I^{(k)}}\not\perp\!\!\!\perp_d X_A$。在此基础上再加入选择变量 $S$（作用于至少两个观测变量），所有分析都在 $S=1$ 下进行。这样观测数据写成 $p(X\mid\psi{=}0,S{=}1)$、干预数据写成 $p(X\mid\psi{=}1,S{=}1)$，二者被统一进同一张增广 DAG（论文 Definition 1）。这个建模动作本身就是后续一切的地基：只有把 $S$ 摆上台面，才有可能把它的干预反应和因果的干预反应区分开。

**2. 刻画 Markov 性质：找出能把选择、混杂、因果三者分开的 CI 指纹**

光把 $S$ 画进图还不够，得证明数据里的统计信号确实和图结构一一对应。作者证明（Theorem 1）：在 $S=1$ 下，增广 DAG 里 $\psi\cup X$ 之间的 d-分离精确蕴含数据中的 CI 与不变性——即"$\psi_{I^{(k)}}\perp\!\!\!\perp_d X_A\mid X_B$ 在图里成立"当且仅当"$p^{(k)}(X_A\mid X_B)=p^{(0)}(X_A\mid X_B)$"，而 $\psi_{I^{(k)}}\not\perp\!\!\!\perp_d X_A$ 则蕴含边缘分布在干预前后发生了变化。三类统计信号被系统利用起来：**干预分布变化**（$\psi$ 与受影响变量的条件依赖）、**不变关系**（条件分布跨观测/干预不变）、以及**结构对称性**——比如一个对称选择结构会同时给出 $\psi_1\perp\!\!\!\perp X_2\mid X_1,\ \psi_2\perp\!\!\!\perp X_1\mid X_2,\ \psi_1\not\perp\!\!\!\perp X_2,\ \psi_2\not\perp\!\!\!\perp X_1$（论文 Figure 4(e)），这种"两端都带 tail"的对称指纹正是直接选择区别于因果的标志。Lemma 1 还提醒：选择会引入额外依赖（条件在 $S$ 上得到的独立性比原图弱），这正是必须显式建模的原因。

**3. FI-Markov 等价 + F-PAG：造一套更细的等价类与能唯一表示它的新图语言**

传统做法只能恢复 MAG/PAG 级别的等价类，用 $\circ\!\!-\!\!\circ$ 这类"圆圈"边把大量不同结构混在一起，因而表达不了"$X_1,X_2$ 间到底有没有直接边"。本文定义 **FI-Markov 等价**（Definition 2）：两张同干预目标的增广 DAG 等价，当且仅当它们在 $X_{[N]\setminus I}$ 上有相同的骨架与 v-结构，且在被干预变量上的 $\psi$–$X$ CI 模式相同。这个等价类比传统等价类更细——它能把 Figure 1(a) 从 (b)、(c) 从 (d) 区分开。为了**唯一**地画出这个更细的等价类，作者扩展 PAG 提出 **F-PAG**（Definition 5）：新增一种端点标记"方块 $\square$"（表示该端点至少有一个 tail 又至少有一个 arrowhead），并引入 $\blacktriangleright\!\!\to$ 与 $\blacktriangleright\!\!-$ 等新边型，专门刻画那种"CI 模式和 $\to$/$-$ 相同、但中间其实没有直接因果边/直接选择"的诱导路径。判别这些新边的钥匙是 **Type I 诱导节点**（Definition 6）：诱导路径上的非端点若呈 $\to\square$（箭头射入方块）就是 Type I，正是它让这些原本混在一起的结构变得可分。

**4. F-FCI 算法：用中间节点的额外硬干预消歧，并证明 sound & complete**

有了等价类与图语言，还需要一个能从数据学出 F-PAG 的算法。**F-FCI**（Algorithm 1）分三步：Step 1 从纯观测数据用 FCI 式约束法恢复骨架；Step 2 用观测↔干预的 CI 模式给被干预变量之间的边定向——按四元组 CI 模式查表，输出 $\to$、$\leftrightarrow$、$\circ\!\!\to$、$-$、$-\square$、$\square\!-\!\square$ 等不同标记；Step 2.3 是**消歧核心**：对仍有歧义的边（如 $\circ\!\!\to$ 的不确定性，或共享同一 CI 指纹却不知有无直接边的诱导路径），算法沿路径找 Type I 诱导节点 $X_n$，对它**再做一次硬干预**并测 $\psi_n\perp\!\!\!\perp X_{I(i)}$。直觉是：硬干预 $X_n$ 能"挡住"潜在混杂上的选择效应、打开被堵的路径——若 $\psi_n\perp\!\!\!\perp X_2\mid S$ 成立，说明 $X_1\to X_2$ 并非直接因果而是经诱导路径，于是把边更新为 $\blacktriangleright\!\!-$/$\blacktriangleright\!\!\to$；直接选择（Figure 4(f)）也用同一套逻辑识别。Step 3 再对剩余边（被干预↔未干预、未干预之间）套用标准 FCI 定向规则与不变性规则（see-see/do-see/do-do）传播。作者证明了 **soundness**（Theorem 3，输出的 tail/arrowhead/square/$\blacktriangleright$ 标记都与真增广 DAG 一致）与 **completeness**（Theorem 4，被干预节点对之间每类子结构都能被对应的 CI 模式辨认出来）。

### 损失函数 / 训练策略
本文是约束式（constraint-based）因果发现，没有可微目标或训练过程；算法依赖**忠实性假设**（faithfulness，数据里没有图之外的额外 CI）与 oracle CI 检验。实践中用统计 CI 检验代替 oracle，识别质量随样本量 $n$ 提升。

## 实验关键数据

### 主实验
合成数据上与 6 个干预因果发现强基线对比（GIES、IGSP、UT-IGSP、JCI-GSP、FCI-interven、CDIS），度量 **DAG Precision（↑）** 与 **SHD（↓）**，在硬干预/软干预、变量数 $d\in\{10,\dots,25\}$、样本量 $n\in\{500,1500,2000\}$ 多种配置下评估，每点对 10 张随机 Erdős–Rényi 图取平均。

| 设定 | 度量 | F-FCI（本文） | 干预因果发现基线 | 结论 |
|------|------|------|----------|------|
| Hard / Soft 干预，$d$=10–25 | DAG Precision ↑ | 多数配置领先 | GIES/IGSP/UT-IGSP/JCI-GSP/FCI-interven/CDIS | 平均高出 >5% |
| 同上 | SHD ↓ | 更低 | 同上 | 结构误差更小 |

基线之所以落后，是因为它们会把潜在混杂与后处理选择诱导的伪依赖误当成真因果边。

### 消融 / 分析实验

| 配置 / 分析 | 关键发现 | 说明 |
|------|---------|------|
| 区分后处理选择能力（Table 1） | F-FCI 能辨认出选择结构 | 基线无此能力 |
| 噪声鲁棒性（Fig 12） | 不同噪声下仍稳定 | $\epsilon\sim\text{Unif}([0,2]\cup[2,4])$ |
| 可扩展性（Fig 11） | 随变量数可扩展 | — |
| F1 / recall（Fig 10） | 与 Precision 趋势一致 | 附录补充 |

### 关键发现
- **可识别性依赖 Type I 诱导节点**：能否分辨直接因果边/直接选择，关键在路径上有没有 Type I 诱导节点供算法做额外硬干预；这既是方法的能力来源，也是其边界（见局限）。
- **真实数据验证**：在 Norman 单细胞基因扰动数据（人肺上皮细胞 HLEC）上做基因调控网络发现，F-FCI 同时报出调控（因果）边与后处理选择诱导的伪依赖，并用 Enrichr 先验知识库做验证。

## 亮点与洞察
- **问题本身就是最大贡献**：把"后处理选择"从被忽视的角落里挖出来，指出它和因果关系共享同一组干预指纹、因而在现有框架里根本不可识别——这是一个清晰且此前没人系统处理的 representational gap。
- **"放宽视野到中间节点"这一招很巧**：端点上分不开的东西，通过对路径中间的 Type I 诱导节点再做一次硬干预就能分开（$\psi_n\perp\!\!\!\perp X$ 测试）。这把"在哪做干预"从只盯因果端点扩展到了诱导路径内部，是可迁移的思路。
- **新图语言配套完整**：方块标记 $\square$ 与 $\blacktriangleright\!\!\to$/$\blacktriangleright\!\!-$ 边让 FI-Markov 等价能被唯一画出，而不是停在"圆圈一片说不清"的 PAG，理论自洽且有 sound+complete 保证。
- **打通理论到生物落地**：质控筛细胞这种后处理选择在单细胞实验里无处不在，方法直接对应到基因调控网络发现，动机不是空想出来的。

## 局限与展望
- **作者承认的局限**：可识别性"严重依赖 Type I 诱导节点的存在"。若一条诱导路径上全是 Type II 诱导节点（相邻两方块 $\square\square$），如何识别其上的因果结构仍是开放问题。
- **未区分生物约束 vs 后处理选择**：作者引用 Luo et al. (2025) 指出生物约束也会过滤细胞、引入额外依赖，如何把"生物约束"从"后处理选择"中进一步分离尚未解决。
- **依赖强假设**：方法建立在 faithfulness 与可靠 CI 检验之上；有限样本下 CI 检验误差会直接传导到结构识别，论文主要在合成数据与单一真实数据集上验证，更大规模真实场景的稳健性有待观察。
- **可改进方向**：把消歧判据从"硬干预 Type I 节点"推广到软干预或部分可干预场景，能显著扩大适用面。

## 相关工作与启发
- **vs 传统干预等价类方法（GIES / IGSP / UT-IGSP / JCI-GSP）**：它们用"干预改变边缘、保持条件不变"的指纹定向，但这套指纹被后处理选择完美模仿，故把"有无直接边"混进同一等价类；本文加选择变量 $S$ 并细化到 FI-Markov 等价，越过这道墙。
- **vs 含潜在混杂的 FCI-interven (Kocaoglu et al., 2019)**：它处理潜在混杂但不建模后处理选择；F-FCI 沿用其 see-see/do-see/do-do 不变性规则做边传播，但在被干预节点对之间额外引入 Type I 诱导节点消歧与新边型，信息量更高。
- **vs CDIS / "selection meets intervention" (Dai et al., 2025)**：后者关注 pre-treatment selection 与干预相遇的复杂性；本文聚焦 **post-treatment** selection（干预之后才发生的筛选），这是被现有 formulation 漏掉的另一半。
- **启发**：当两类机制在端点统计上不可分时，"主动在路径中间节点施加额外干预/扰动来打开被堵路径"是一个通用的可识别性增强思路，可迁移到其他被混杂/选择困住的因果发现任务。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统形式化后处理选择，并配套全新等价类（FI-Markov）、图语言（F-PAG）与算法（F-FCI）。
- 实验充分度: ⭐⭐⭐⭐ 合成数据多配置对比 6 基线 + 单细胞真实数据，理论有 sound/complete 证明；真实数据集偏少。
- 写作质量: ⭐⭐⭐⭐ 动机用 Figure 1/2 的反例讲得清楚，但符号与定义密集，对非因果发现背景读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 直击单细胞/临床等领域普遍存在的质控选择偏差，理论与应用都有分量。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[ICLR 2026\] Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)
- [\[ICLR 2026\] Stochastic Neural Networks for Causal Inference with Missing Confounders](stochastic_neural_networks_for_causal_inference_with_missing_confounders.md)
- [\[ICLR 2026\] Conditional Independent Component Analysis for Estimating Causal Structure with Latent Variables](conditional_independent_component_analysis_for_estimating_causal_structure_with_.md)
- [\[ICML 2025\] Latent Variable Causal Discovery under Selection Bias](../../ICML2025/causal_inference/latent_variable_causal_discovery_under_selection_bias.md)

</div>

<!-- RELATED:END -->
