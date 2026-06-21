---
title: >-
  [论文解读] Active Learning with Low-Rank Structure for Data Selection
description: >-
  [ICML 2026][学习理论][低秩近似] 针对"现有 coreset 数据选择假设数据有几何聚类结构、但很多现代数据集其实是全局代数（低秩）结构"的错配，本文提出基于低秩近似 + 残差敏感度采样的数据选择框架，用一个 $\tilde{O}(k+1/\varepsilon^2)$ 大小的加权子集把全量平均损失逼近到 $(1\pm\varepsilon)$ 相对误差（外加正比于最优 rank-$k$ 近似代价 $\Phi_k$ 的可加项），在表格数据和 Llama3-8B / Qwen2.5-3B 微调上都优于均匀采样与聚类敏感度采样。
tags:
  - "ICML 2026"
  - "学习理论"
  - "数据选择"
  - "Coreset"
  - "低秩近似"
  - "敏感度采样"
  - "主动学习"
---

# Active Learning with Low-Rank Structure for Data Selection

**会议**: ICML 2026  
**arXiv**: [2606.16045](https://arxiv.org/abs/2606.16045)  
**代码**: 未发布  
**领域**: 学习理论 / 数据选择 / Coreset  
**关键词**: 数据选择, coreset, 低秩近似, 敏感度采样, 主动学习

## 一句话总结
针对"现有 coreset 数据选择假设数据有几何聚类结构、但很多现代数据集其实是全局代数（低秩）结构"的错配，本文提出基于低秩近似 + 残差敏感度采样的数据选择框架，用一个 $\tilde{O}(k+1/\varepsilon^2)$ 大小的加权子集把全量平均损失逼近到 $(1\pm\varepsilon)$ 相对误差（外加正比于最优 rank-$k$ 近似代价 $\Phi_k$ 的可加项），在表格数据和 Llama3-8B / Qwen2.5-3B 微调上都优于均匀采样与聚类敏感度采样。

## 研究背景与动机
**领域现状**：基础模型时代训练成本高昂，而"用全量数据其实没必要、精心挑的小子集就够"已成共识。Sener & Savarese（ICLR 2018，[SS18]）把主动学习重新表述为 **coreset 选择**问题——给定数据的 embedding 表示，用 $k$-center 聚类的启发式挑代表点；Axiotis 等（ICML 2024，[ACH+24]）进一步用 $k$-means 聚类 + 敏感度采样构造 coreset。这条线的核心是：只要 coreset 上的平均损失（或梯度）逼近全量，那么在子集上训练就近似等价于在全量上训练。

**现有痛点**：这些方法都假设数据集存在**内在几何结构、能被聚类有效刻画**。但作者指出，许多现代数据集并非"局部成簇"，而是具有**全局代数结构**——更适合用低秩近似 / 主成分分析（PCA）来刻画。在高维数据上，聚类只关注点的**局部分组**，会错过数据的**主方差方向**，导致选出的子集丢掉最有信息量的成分。

**核心矛盾**："聚类衡量点之间的**距离**" vs "数据的信息其实集中在少数**主方向（谱方向）**上"。聚类隐含地认为点间距离比点的方向更重要；而像 LoRA 这类工作恰恰说明，低秩更新能抓住参数空间里最重要的成分，朴素地对 embedding 聚类反而会忽略关键方向。

**本文目标**：在数据集近似低秩这一现实假设下，构造一个**小而有理论保证**的加权子集，使其平均损失逼近全量平均损失，同时把对昂贵模型损失 $\ell$ 的查询次数压到很小。

**切入角度 / 核心 idea**：用"先在一小撮数据上算准确但昂贵的损失分数，再用快速可算的 embedding/sketch 捕捉数据集主方向"两者结合。具体地——**构造数据集的低秩 sketch 来估计 leverage score（杠杆分数），按分数比例对行（数据点）采样**，让选出的子集反映主方差方向，而不是像聚类那样只追求几何多样性。

## 方法详解

### 整体框架
方法把数据选择形式化为 **row subset selection（行子集选择）+ 保损失 coreset 构造**问题。设数据矩阵 $D\in\mathbb{R}^{n\times m}$，目标是挑一个加权子集 $S\subseteq D$（权重 $w(x)$）使加权损失和逼近全量损失和 $\Delta(S)=\big|\sum_{x\in D}\ell(x)-\sum_{x\in S}w(x)\ell(x)\big|$ 最小，同时尽量少地对 $\ell$ 做昂贵查询。整条 pipeline 是：取一个 $k$ 维子空间 $V$（如顶 $k$ 个奇异向量／低秩因子张成）→ 把每个点 $x$ 分解成 $V$ 内投影 $v(x)$ 与正交残差 $r(x)$ → 在一个"低秩损失正则性假设"下，给每个点算一个**敏感度分数** $\sigma(x)$ → 归一化成概率按比例采样 → 用 $1/(s\,p(x))$ 加权使估计无偏 → 用集中不等式给出高概率逼近保证。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["数据集 D + 预训练 embedding"] --> B["低秩损失正则性假设<br/>损失沿主方向可分解 + 残差二次惩罚"]
    B --> C["低秩子空间 V<br/>SVD / leverage-score 选行"]
    C --> D["残差敏感度分数 σ(x)<br/>投影损失 + 残差范数"]
    D -->|按 p(x)=σ(x)/Σσ 采样| E["加权子集 S<br/>权重 1/(s·p(x))"]
    E --> F["在 S 上训练/微调<br/>≈ 全量训练"]
```

### 关键设计

**1. 低秩损失正则性假设：把"损失是近似低秩的"写成可证明的条件**

聚类方法假设数据成簇，本文则需要一个刻画"损失沿主方向可分解、正交方向贡献可控"的假设（Assumption 2.1）。把任意点 $y$ 在子空间 $V=\mathrm{span}\{v_1,\dots,v_k\}$ 上分解为 $y=\sum_i\alpha_i v_i + r(y)$，其中 $\alpha_i=\langle y,v_i\rangle$、$r(y)=\mathrm{Proj}_{V^\perp}(y)$。假设存在常数 $\lambda,\gamma>0$ 使得

$$|\ell(y)-\ell(v(y))|\le\lambda\|r(y)\|_2^2,\qquad |\ell(v(y))-\textstyle\sum_i\alpha_i^2\ell(v_i)|\le\gamma\sum_i|\alpha_i^2+1|\,\ell(v_i).$$

第一式是 smoothness 型条件——点偏离子空间 $V$ 带来的损失至多随到 $V$ 的距离**二次增长**（Lipschitz 梯度／有界曲率即满足）；第二式说子空间内的损失可被沿各基方向损失的加权组合（权重 $\alpha_i^2$）逼近，误差由 $\gamma$ 控制，且**不要求**损失可加、可分或恰为二次，交互项与高阶效应都吸收进 $\gamma$ 项。关键的 regime 是 $\lambda\gg\gamma$：正交方向信息少、建模差，正是本文针对的场景。这把"数据近似低秩"从直觉变成可推导保证的前提，PCA、低秩回归、矩阵补全、LLM 微调里的 LoRA 都天然落在这个 regime。

**2. 残差敏感度分数：让采样概率正比于"点对总损失的真实贡献"**

聚类敏感度采样按"点离簇心的几何距离"打分，容易偏向离群点。本文改用基于低秩结构的**敏感度分数**（Algorithm 1 第 7 行）：

$$\sigma(x)=(\gamma+1)\big(\alpha_1^2\ell(v_1)+\dots+\alpha_t^2\ell(v_t)\big)+\gamma k\xi+\lambda\|r(x)\|_2^2,$$

由三部分构成——**投影损失项**（点在主方向上的损失贡献，由低秩 sketch 上算出的 landmark 损失 $\ell(v_i)$ 加权）、**基损失项** $\gamma k\xi$、以及**残差损失项** $\lambda\|r(x)\|_2^2$（点偏离主子空间的程度）。归一化得概率 $p(x)=\sigma(x)/\sum_y\sigma(y)$，按其采样 $s=\lceil\varepsilon^{-2}(2+2\varepsilon/3)\rceil$ 个点，并赋权 $w(x)=1/(s\,p(x))$ 保证估计无偏。这样高贡献点（无论是落在主方向上损失大，还是残差大）被选中概率更高，子集反映的是**谱主轴 + 真实损失**，而非单纯的几何多样性。值得注意的是 $\lambda$ 的角色：$\lambda$ 太大时分数被几何残差 $r$ 主导，会把采样推向离群点而非高损失区域（实验里 $\lambda=1$ 且簇数增大时聚类逼近反而变差，正源于此）。

**3. 两种 coreset 保证：sketch 因子可不在数据集内 vs 严格选数据集内的行**

Theorem 2.2 给出一般保证：存在随机算法构造大小 $s=\mathcal{O}(1/\varepsilon^2)$ 的加权子集 $S$，以 $\ge0.9$ 概率满足

$$\Big|\sum_{x\in D}\ell(x)-\sum_{x\in S}w(x)\ell(x)\Big|\le\varepsilon\cdot\Big(\sum_{x\in D}\ell(x)+\gamma\|D\|_F^2+\gamma k|D|\max\ell+2\lambda\,\Phi_k(D)\Big),$$

即加权平均损失落在真实平均损失的 $(1\pm\varepsilon)$ 倍内，外加正比于 $\Phi_k(D)/n$ 的可加项，其中 $\Phi_k(D)=\min_{\mathrm{rank}(D_k)\le k}\|D-D_k\|_F^2=\sum_{i>k}\sigma_i^2$ 是最优 rank-$k$ 近似代价（由 Eckart–Young–Mirsky 定理刻画）。数据越接近低秩，$\Phi_k$ 越小，coreset 越准——这与聚类方法"数据有离群/高秩噪声时界变松"是类比的，但低秩法显式瞄准高方差方向，在高维/不平衡场景更鲁棒。Theorem 2.2 的隐忧是 $S$ 不一定是 $D$ 的子集（因子可能要额外标注），于是 Theorem 2.3 把目标换成 **row subset selection** 代价 $\Phi_k(D)=\min_{C\subseteq D,|C|=k}\min_A\|D-CA\|_F^2$，保证选出的 $S\subseteq D$ 是**真实数据点**且享有同样的逼近界；实现上可借助"多选 $2k$ 列即得常数因子近似"的双标准算法（Theorem 5.1，[GS12]）。

### 损失函数 / 训练策略
方法本身不引入训练损失，而是在既有模型损失 $\ell(x,y;\mathcal{A})$ 上做选择。作者把批量选择的期望损失分解为**泛化误差 + 训练误差 + coreset 损失**三项，本文的低秩敏感度采样只针对最后一项 **coreset 损失**做最小化，且用的是当前模型损失（而非重训后损失），无需对标签分布或零训练损失做假设。实践中子空间维度 / landmark 数默认取数据集的 20%，表格实验用 `TruncatedSVD(n_components=5)`、$\gamma=5,\lambda=1$；LLM 实验用 BERT embedding、leverage-score 选 landmark、$\gamma=0$，并用带 RBF 核的 Kernel Ridge Regression 在 embedding 空间拟合 landmark 损失的线性组合得到 $\alpha$。

## 实验关键数据

### 主实验
**表格数据（Default of Credit Card Clients，30000 条 23 维，22% 正例）**：比较随机采样、聚类、低秩敏感度采样在不同 coreset 大小下的逼近误差与下游精度。

| 任务 / 子集大小 | 指标 | 低秩敏感度采样（本文） | 聚类 | 随机 |
|--------|------|------|------|------|
| coreset 误差 @ $s{=}1000$ | $\|\sum w\|x\|^2-L_{\text{true}}\|$ | 最低（比随机低约一个量级，比聚类低约 50%） | 中 | 最高 |
| 下游精度 @ $s{=}5000$ | 测试准确率 | ~74% | ~70% | ~67% |

**LLM 微调（Llama3-8B-Instruct，BERT embedding，$k$ 固定为 25%）**：在 GSM8k / ViGGO / SQL 三个任务、三档采样率上比较，下表为各方法平均验证准确率。

| 方法 | 25% 平均 | 12.5% 平均 | 6.25% 平均 |
|------|---------|-----------|-----------|
| Uniform Sampling | 76.5 | 69.2 | 52.0 |
| K-Center | 74.9 | 63.9 | 52.3 |
| Graph Cut | 74.0 | 65.8 | 45.7 |
| Clustering-based SS [ACH+24] | 77.5 | **71.0** | 54.6 |
| Low-rank SS（本文） | **77.6** | 70.4 | **54.9** |
| Full（100%） | 81.1 | — | — |

低秩 SS 始终优于均匀采样，平均上多数情况优于聚类敏感度采样；在 ViGGO @ 25% 上达 88.3（聚类 86.6、Full 94.0）。选择过程只需对 20% 数据做前向传播，约相当于全量训练 6.67% 的运行时。

### 消融与分析实验

| 配置 / 分析 | 关键结果 | 说明 |
|------|---------|------|
| 数据结构诊断（GSM8k） | 低秩近似误差 < 聚类投影误差 | 验证 GSM8k 更符合低秩而非聚类结构，支撑 Assumption 2.1 |
| 平均损失逼近 $\Delta(S)$ 热图（$k,\lambda$ 网格，选 2000 点） | 低秩在各 $k,\lambda$ 下误差均更低 | 聚类在 $\lambda{=}1$ 时增大簇数反而变差（被几何距离 $r$ 主导、偏向离群点） |
| 换目标 / 换 embedding | 用梯度范数代替损失、用 GTR-base 代替 BERT 结论不变 | 低秩对 $\lambda$ 更鲁棒，始终胜过均匀采样 |
| 换模型（Qwen2.5-3B，ViGGO） | 低秩 SS 比最强基线（聚类 SS）在 25%/12.5% 分别高 4.34 / 3.08 点 | 验证跨模型泛化 |

### 关键发现
- **数据集到底是"成簇"还是"低秩"决定方法优劣**：GSM8k 诊断实验显示它更符合低秩结构，这正解释了低秩采样为何更准——方法假设与数据性质对上了。
- **$\lambda$ 是双刃剑**：$\lambda$ 控制残差项权重，太大时采样被几何离群点主导、损失逼近反而变差；低秩法对 $\lambda$ 的鲁棒性强于聚类法。
- **代价小**：仅 20% 数据前向传播即可完成选择（≈6.67% 全量训练开销），适合大模型微调前的数据筛选。

## 亮点与洞察
- **把"该用聚类还是低秩"这一选择本身理论化**：作者没有泛泛说"低秩更好"，而是给出 Assumption 2.1 这个可验证条件，并配 GSM8k 诊断实验，告诉你**什么样的数据集**该用本文方法——这种"先判结构再选方法"的思路可迁移到任何 coreset/采样任务。
- **leverage score / 谱敏感度替代几何距离**：把数据选择从"覆盖空间、拉大点间距离"转向"保住主方差方向"，是对 [SS18]/[ACH+24] 路线的一次方向性修正，且天然支持非均匀加权采样。
- **理论 + 工程双落地**：$\mathcal{O}(1/\varepsilon^2)$ 的子集大小与数据维度、样本数解耦，且 row subset selection 版保证选出的是真实数据点（可直接标注），实用性强。

## 局限与展望
- **依赖 Assumption 2.1 成立**：当数据有显著离群点/高秩噪声时，可加项 $2\lambda\Phi_k(D)$ 变大，保证退化——本文的优势严格依赖"数据近似低秩"，对真正成簇的数据未必占优（12.5% 档聚类 SS 平均反超）。
- **提升幅度温和**：LLM 微调上多数任务相比聚类 SS 只领先零点几到几个点，且并非每档每任务都赢，结论是"平均更优 + 更鲁棒"而非全面碾压。
- **超参 $\lambda,\gamma$ 需调**：$\lambda$ 敏感（实验里要 tune 取最优），$\gamma$ 在 LLM 实验直接设 0，给实际使用带来调参成本。
- **改进方向**：自动从数据诊断 $\lambda$/子空间维度 $k$、把低秩与聚类敏感度做混合（按数据局部结构自适应切换）、以及把保证从"当前模型损失"推广到"重训后损失"。

## 相关工作与启发
- **vs [SS18]（k-center coreset）**：[SS18] 把主动学习化为 $k$-center 几何 coreset、追求覆盖与多样性；本文改用低秩 sketch + 残差敏感度，针对主方差方向，理论上在高维/低秩数据更鲁棒，且支持加权采样。
- **vs [ACH+24]（聚类敏感度采样）**：二者都用敏感度采样，但 [ACH+24] 的敏感度来自 $k$-means 聚类（几何距离），本文来自低秩近似（谱结构）；当数据更接近低秩（如 GSM8k）本文更优，反之聚类可能占优。
- **vs LoRA 系列工作**：本文借用了"低秩更新抓住参数空间最重要成分"的观察（[HSW+22] 等）并平移到**数据空间**——既然参数适配是低秩的，数据的损失结构也常是低秩的，故按谱方向选数据更合理。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把数据选择从聚类几何视角切换到低秩谱视角，并给出可验证假设与保证，方向性清晰
- 实验充分度: ⭐⭐⭐⭐ 覆盖表格 + 两个 LLM、三任务三采样率、结构诊断与超参/embedding 消融，但提升幅度温和且非全胜
- 写作质量: ⭐⭐⭐⭐ 假设—定理—算法—实验链条完整，动机讲得透；定理可加项偏多略增阅读负担
- 价值: ⭐⭐⭐⭐ 给"近似低秩数据集"提供了有保证、低开销的数据筛选方案，适合大模型微调前置环节

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)
- [\[ICML 2026\] The Data Manifold under the Microscope](the_data_manifold_under_the_microscope.md)
- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](when_sample_selection_bias_precipitates_model_collapse.md)
- [\[ICML 2026\] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](../../NeurIPS2025/learning_theory/adaptive_data_analysis_for_growing_data.md)

</div>

<!-- RELATED:END -->
