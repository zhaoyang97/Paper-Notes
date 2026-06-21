---
title: >-
  [论文解读] Graph Random Features for Scalable Gaussian Processes
description: >-
  [ICLR 2026][图学习][Graph Random Features] 用基于随机游走的图随机特征（GRF）构造稀疏、无偏的图节点核估计，把图上高斯过程的贝叶斯推断从 $\mathcal{O}(N^3)$ 降到 $\mathcal{O}(N^{3/2})$，并带概率精度保证，从而在单卡上跑通了超过 100 万节点的图贝叶斯优化。
tags:
  - "ICLR 2026"
  - "图学习"
  - "Graph Random Features"
  - "Gaussian Process"
  - "Graph Node Kernel"
  - "随机游走"
  - "共轭梯度"
  - "贝叶斯优化"
  - "可扩展推断"
---

# Graph Random Features for Scalable Gaussian Processes

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=89SQfLguNn](https://openreview.net/forum?id=89SQfLguNn)  
**代码**: [https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes](https://github.com/MatthewZhang473/Graph-Random-Features-for-Scalable-Gaussian-Processes)  
**领域**: 图学习 / 概率方法（图上高斯过程）  
**关键词**: Graph Random Features, Gaussian Process, Graph Node Kernel, 随机游走, 共轭梯度, 贝叶斯优化, 可扩展推断  

## 一句话总结
用基于随机游走的图随机特征（GRF）构造稀疏、无偏的图节点核估计，把图上高斯过程的贝叶斯推断从 $\mathcal{O}(N^3)$ 降到 $\mathcal{O}(N^{3/2})$，并带概率精度保证，从而在单卡上跑通了超过 100 万节点的图贝叶斯优化。

## 研究背景与动机
**领域现状**：高斯过程（GP）在带不确定性的函数建模上很强，但当输入空间是图节点（如交通路网、社交网络）时，欧氏距离核并不合适，需要定义在图上的节点核（diffusion 核、Matérn 核等）。这些核通常写成加权邻接矩阵 $\mathbf{W}$ 的幂级数 $\mathbf{K}_\alpha(\mathbf{W})=\sum_{r=0}^{\infty}\alpha_r\mathbf{W}^r$。

**现有痛点**：和欧氏 GP 一样，图上精确 GP 仍然要对 $N\times N$ 矩阵做 $\exp(\cdot)$ 或求逆这类稠密运算，光评估核就是 $\mathcal{O}(N^3)$，大图上根本算不动。现有提速手段各有损失：图傅里叶特征用截断特征值展开，会丢掉高频核信息；稀疏核族限制了灵活性；局部子图核则放弃了在整图上做推断。

**核心矛盾**：图节点核既要**保留全图结构和高频信息（灵活性）**，又要**避免稠密矩阵运算（可扩展性）**——这两者在已有方法里难以兼得。

**本文目标**：找到一种既稀疏可扩展、又能无偏估计灵活核族的图节点核近似，并把它端到端接入 GP 的超参数学习、后验推断乃至贝叶斯优化全流程。

**核心 idea**：**用随机游走做蒙特卡洛估计**——前人提出的图随机特征（GRF）是邻接矩阵幂级数的无偏估计器（类比 Von Neumann 的俄罗斯轮盘赌估计器），每个特征只有 $\mathcal{O}(1)$ 个非零元却满足指数集中不等式。本文证明用 GRF 的稀疏核 $\hat{\mathbf{K}}=\mathbf{\Phi}\mathbf{\Phi}^\top$ 替代稠密核，配合共轭梯度迭代解，就能把推断压到 $\mathcal{O}(N^{3/2})$，且不必显式 materialise 核矩阵。

## 方法详解

### 整体框架
方法分三步：先用随机游走为每个节点采样出稀疏特征 $\varphi(i)$，拼成对核矩阵 $\mathbf{K}_\alpha$ 的无偏估计 $\hat{\mathbf{K}}=\mathbf{\Phi}\mathbf{\Phi}^\top$；再以最大化对数边缘似然学超参（含调制函数 $f$ 和噪声），梯度项通过共轭梯度（CG）解线性系统 + Hutchinson 迹估计来避免显式求逆；最后用 pathwise conditioning 把后验采样转化为"先验采样 + 一次稀疏线性求解"，从而支撑 Thompson 采样的贝叶斯优化。全程只碰 $\mathcal{O}(N)$ 个非零元，空间复杂度也是 $\mathcal{O}(N)$。

```mermaid
flowchart LR
    A[图 G + 调制函数 f] --> B[随机游走采样<br/>构造稀疏 GRF φ_i]
    B --> C["稀疏核估计<br/>K̂ = ΦΦᵀ"]
    C --> D[超参学习<br/>CG + Hutchinson 估梯度]
    D --> E[后验推断<br/>pathwise conditioning + CG]
    E --> F[贝叶斯优化<br/>Thompson 采样, >1M 节点]
```

### 关键设计

**1. 随机游走构造的稀疏无偏核估计：把矩阵幂级数变成蒙特卡洛采样。** GRF 的出发点是对系数序列做"反卷积"，找一组调制函数 $(f_l)$ 满足 $\sum_{l=0}^{r}f_l f_{r-l}=\alpha_r$，使得 $\mathbf{\Psi}=\sum_{l=0}^{\infty}f_l\mathbf{W}^l$ 在对称 $\mathbf{W}$ 下满足 $\mathbf{\Psi}^\top\mathbf{\Psi}=\mathbf{K}_\alpha$。由于 $\mathbf{W}^l_{ij}$ 统计的是节点 $i,j$ 间长度 $l$ 的（带权）游走数，整个核就可以解释成"对各长度游走加权求和"。GRF 的关键是从每个节点模拟随机游走，对每条游走的前缀子路（前 $l$ 步）记录三件事——所走边权之积、调制函数 $f$、以及该前缀在采样机制下的边缘概率 $p$——按 $(\prod \text{edge weights})\cdot f(\text{len})/p(\text{prefix})$ 累加进特征向量（Alg.1 第 8 行）。除以 $1/p$ 的**重要性采样**是精髓：它让长而稀有的游走被适当上权，从而保证 $\mathbb{E}(\varphi(i)^\top\varphi(j))=[\mathbf{K}_\alpha]_{i,j}$ 无偏。每个特征只有 $\mathcal{O}(1)$ 非零元，整个 $\hat{\mathbf{K}}$ 也只有 $\mathcal{O}(N)$ 非零元，却满足指数集中界（Theorem 1），且所需游走数 $n$ 与图大小 $N$ 无关。

**2. 稀疏核的良条件性 → 共轭梯度的 $\mathcal{O}(N^{3/2})$ 求解（本文新理论）。** 光稀疏还不够，关键要证明用它解线性系统也快。本文的新结果 Theorem 2 给出 $\hat{\mathbf{K}}$ 两条性质：(1) 因 $\mathcal{O}(N)$ 非零元，矩阵向量乘 $\hat{\mathbf{K}}\boldsymbol{v}$ 只需 $\mathcal{O}(N)$；(2) 在 $\|\varphi(i)\|_1\le c$ 的假设下，正则化后的条件数被控制为 $\kappa(\hat{\mathbf{K}}+\sigma_n^2\mathbf{I})\le 1+\frac{Nc^2}{\sigma_n^2}=\mathcal{O}(N)$。由此推出 Lemma 1：用共轭梯度解 $(\hat{\mathbf{K}}+\sigma_n^2\mathbf{I})\boldsymbol{v}=\boldsymbol{b}$ 只需 $\sqrt{\kappa}=\mathcal{O}(N^{1/2})$ 次迭代，每次迭代 $\mathcal{O}(N)$，合起来 $\mathcal{O}(N^{3/2})$，远低于精确法的 $\mathcal{O}(N^3)$。实践中 $\hat{\mathbf{K}}$ 甚至不必显式拼出，直接用 $\mathbf{\Phi}(\mathbf{\Phi}^\top\boldsymbol{v})$ 两次线性时间的矩阵向量乘代替。

**3. 端到端的迭代化 GP 工作流：超参学习 + pathwise conditioning。** 超参学习上，最大化对数边缘似然 $\mathcal{L}(\boldsymbol{\theta})=-\frac12\boldsymbol{y}^\top\mathbf{H}_{\boldsymbol{\theta}}^{-1}\boldsymbol{y}-\frac12\log\det\mathbf{H}_{\boldsymbol{\theta}}-\frac{N}{2}\log(2\pi)$，其中 $\mathbf{H}_{\boldsymbol{\theta}}=\hat{\mathbf{K}}_{xx}+\sigma_n^2\mathbf{I}$；梯度里的求逆全部用 CG 迭代解，迹项 $\mathrm{tr}(\mathbf{H}_{\boldsymbol{\theta}}^{-1}\partial\mathbf{H}_{\boldsymbol{\theta}})$ 用 Hutchinson 估计器化成一批随机探针的线性系统一起解。后验推断则用 pathwise conditioning，把后验样本写成"先验样本 + 修正项"：$\boldsymbol{g}_{|\boldsymbol{y}}(\cdot)=\boldsymbol{g}(\cdot)+\hat{\mathbf{K}}_{(\cdot)x}(\hat{\mathbf{K}}_{xx}+\sigma_n^2\mathbf{I})^{-1}(\boldsymbol{y}-(\boldsymbol{g}(\boldsymbol{x})+\boldsymbol{\varepsilon}))$。其中先验样本可由 $\boldsymbol{g}=\mathbf{\Phi}\boldsymbol{w},\ \boldsymbol{w}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 高效采样（因 $\mathrm{Cov}(\mathbf{\Phi}\boldsymbol{w})=\hat{\mathbf{K}}$），修正项里的逆又交给 CG。这样从超参学习到后验采样全程只依赖稀疏矩阵向量乘，天然契合 GRF 的结构，最终为大图上的 Thompson 采样 BO 提供可扩展的后验样本。

## 实验关键数据

### 主实验：可扩展性与回归
- **稀疏 vs 稠密 GRF（合成图）**：8192 节点上稀疏实现相比显式 materialise 核矩阵的稠密实现，总 wall-clock 提速 **50×**；稀疏版扩展到 **100 万节点**，稠密版因显存受限只能到 8192 节点。
- **交通速度预测（San Jose 路网）**：当游走数 $n\gtrsim 500$ 时，可学习调制函数的 GRF 核在 NLPD 和 RMSE 上**超过精确 diffusion 核** $\mathbf{K}_{\text{diff}}$，且始终优于"diffusion 形状"的 GRF 变体。
- **全球风场插值（ERA5，10K 节点 kNN 图）**：精确 diffusion 核因 $\mathcal{O}(N^3)$ 被省略；可学习 GRF 核随 $n$ 增大持续改善，NLPD/RMSE 均优于 diffusion 形状变体。

### 消融实验

| 实验 | 设置 | 结论 |
|------|------|------|
| 经验缩放指数 | 稀疏 GRF | Memory $1.00$、Init $0.81$、Train $1.04$、Infer $1.04$（$\mathcal{O}(N^b)$） |
| 经验缩放指数 | 稠密 GRF | Memory $2.00$、Train $1.97$、Infer $2.16$（验证稀疏带来质变） |
| 重要性采样消融 | 去掉 $1/p(\text{prefix})$ 重权（Alg.1 第8行改 Eq.13） | 回归性能**显著退化**，长程依赖难以建模 |
| 调制函数 | 可学习 $f_l$ vs 冻结成 diffusion 形状 | 可学习版一致更优，体现隐式核学习的价值 |

### 关键发现
- GRF 的稀疏性本身是**合理的归纳偏置**：两节点只有当各自游走集合相遇才有非零协方差，使预测主要依赖局部邻域，同时以低概率采样长程依赖，反而比稠密核更不易被噪声驱动的虚假长程相关"过平滑"。
- 在多种合成图、社交网络（Enron / Facebook / Twitch / YouTube）和 ERA5 风速 BO 任务上，GRF-BO（Thompson 采样）在几乎所有数据集上 regret 都低于随机搜索 / BFS / DFS。
- Cora 引文网络多分类（变分推断、非共轭设定）中，稀疏 GRF 核同样取得很强性能（作为 future work 初探）。

## 亮点与洞察
- **把"稀疏 + 无偏 + 良条件"串成完整复杂度证明**：Theorem 2 关于条件数 $\mathcal{O}(N)$ 的结论是新的，正是它把"稀疏特征"接上"CG 迭代次数 $\mathcal{O}(N^{1/2})$"，从而严格得到 $\mathcal{O}(N^{3/2})$。
- **灵活性与可扩展性罕见地双赢**：可学习调制函数让 GRF 核反而能超过精确 diffusion 核，打破"近似必然损失精度"的直觉。
- **稀疏即归纳偏置**的观察很有启发——把蒙特卡洛近似的副产物（稀疏性）重新解读为对图信号的合理先验。
- **工程落地扎实**：在一块 RTX 2080 Ti（11GB）上跑通 >1M 节点 BO，可复现代码开源。

## 局限与展望
- **分类/非共轭设定缺乏复杂度保证**：变分 GP 分类下 pathwise conditioning 不平凡，Lemma 1 的 $\mathcal{O}(N^{3/2})$ 保证不再适用，作者明确留作 future work。
- **理论依赖常数 $c$ 有限**：要求 $c=\sum_r|f_r|(\max_{ij}\mathbf{W}_{ij}d_i/(1-p))^r$ 收敛，即谱半径需落在收敛域内；虽然作者论证实践中（截断最大游走长度）自然成立，但对某些重尾度数分布或大边权图仍需谨慎。
- **近线性趋势是有限尺度现象**：作者指出训练/推断的近线性是固定迭代预算所致，"conditioning 效应在当前规模尚未主导"，更大规模下 $\mathcal{O}(N^{3/2})$ 项可能显现。
- **游走数 $n$ 仍需调**：$n$ 直接决定方差与非零元数，虽与 $N$ 无关但需按精度需求选取。

## 相关工作与启发
- **图随机特征（GRF）原始工作**（Choromanski 2023；Reid et al. 2023, 2024a/b）：本文站在其肩上，补齐了"稀疏性加速 + BO 应用"两块前人未充分挖掘的拼图（Reid et al. 2024a 仅在小图 diffusion 核上做理论后验，未利用稀疏性也未做 BO）。
- **欧氏随机特征**（Rahimi & Recht 2007）：本文是其在离散图域的对应物，思想一脉相承（点积期望等于核）。
- **迭代式 GP 推断**（Gardner et al. 2018；Lin et al. 2024）与 **pathwise conditioning**（Wilson et al. 2020/2021）：本文的可扩展工作流正是把这两条线接到 GRF 稀疏结构上。
- **启发**：当一个估计器同时具备"无偏 + 稀疏 + 可控条件数"三性质时，往往能直接撬动整套迭代数值线性代数工具链（CG / Hutchinson / pathwise），这是把蒙特卡洛核方法做到可扩展的通用范式，值得迁移到其他离散结构（超图、单纯复形）上。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 在已有 GRF 估计器上补齐关键的条件数新理论（Theorem 2）并打通 BO 应用，组合创新扎实但底层估计器非本文首创。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖缩放消融、重要性采样消融、回归、社交网络/风场 BO、分类初探，规模直达 1M 节点；缺与更多近似 GP baseline 的横向对比。
- **写作质量**: ⭐⭐⭐⭐ 理论—算法—实验逻辑清晰，复杂度推导完整，对 GRF 直觉解释到位。
- **价值**: ⭐⭐⭐⭐ 让图上 GP/BO 真正进入百万节点尺度且保持竞争力，对图信号建模、网络优化等应用有实际推动力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FLOCK: A Knowledge Graph Foundation Model via Learning on Random Walks](flock_a_knowledge_graph_foundation_model_via_learning_on_random_walks.md)
- [\[ICLR 2026\] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)
- [\[ICLR 2026\] HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature](hgnet_scalable_foundation_model_for_automated_knowledge_graph_generation_from_sc.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ACL 2026\] LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](../../ACL2026/graph_learning/logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)

</div>

<!-- RELATED:END -->
