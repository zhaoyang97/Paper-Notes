---
title: >-
  [论文解读] HSG-12M: A Large-Scale Benchmark of Spatial Multigraphs from the Energy Spectra of Non-Hermitian Crystals
description: >-
  [ICLR 2026][物理/科学计算][spatial multigraph] 本文把非厄米晶体的能谱"画"成图——提出自动化流水线 Poly2Graph 把一维晶体哈密顿量映射为复能量平面上的谱图，并据此构建 HSG-12M：首个大规模"空间多重图"基准（1160 万静态 + 510 万动态图，1401 类），暴露出现有 GNN 在保留几何信息的多重边上学习的全新挑战。
tags:
  - "ICLR 2026"
  - "物理/科学计算"
  - "spatial multigraph"
  - "non-Hermitian physics"
  - "Hamiltonian spectral graph"
  - "graph benchmark"
  - "图神经网络"
  - "凝聚态物理"
---

# HSG-12M: A Large-Scale Benchmark of Spatial Multigraphs from the Energy Spectra of Non-Hermitian Crystals

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=YxuKCME576](https://openreview.net/forum?id=YxuKCME576)  
**代码**: [Poly2Graph](https://github.com/sarinstein-yan/Poly2Graph) / [HSG-12M](https://github.com/sarinstein-yan/HSG-12M)  
**领域**: 科学计算 / 图表示学习 / 数据集与基准  
**关键词**: spatial multigraph, non-Hermitian physics, Hamiltonian spectral graph, graph benchmark, GNN, 凝聚态物理  

## 一句话总结
本文把非厄米晶体的能谱"画"成图——提出自动化流水线 Poly2Graph 把一维晶体哈密顿量映射为复能量平面上的谱图，并据此构建 HSG-12M：首个大规模"空间多重图"基准（1160 万静态 + 510 万动态图，1401 类），暴露出现有 GNN 在保留几何信息的多重边上学习的全新挑战。

## 研究背景与动机
**领域现状**：AI for Science 的成功（蛋白折叠、材料发现、多体物理）高度依赖大规模高质量的领域数据集，但物理科学方向恰恰缺这类数据。与此同时，非厄米物理近年发现：一维晶体在开放边界条件下的能谱会在复能量平面上聚成弧线与环路，形成结构远比 Chern number、$\mathbb{Z}/\mathbb{Z}_2$ 等传统拓扑不变量更丰富的"哈密顿谱图"（Hamiltonian spectral graph），是电子行为的天然指纹。

**现有痛点**：谱图过去只能靠人手画、肉眼看，仅限玩具规模，没有自动化工作流，更没有大规模数据集，系统性研究无从谈起。另一方面，图表示学习几乎所有公开基准都把数据当作**简单图**（任意两点间至多一条边），即便原始数据存在多重边也会被聚合成单条属性边，丢弃关键几何信息。

**核心矛盾**：真实世界大量网络（城市街道、生物神经网络、蛋白结构）本质是**空间多重图**——嵌入度量空间、两点间存在多条几何上各异的连接路径；当我们既关心连接拓扑又关心连接几何时，把多重边塌缩成一条就是不可逆的信息损失。但因为缺数据，针对空间多重图的方法学几乎是空白。

**本文目标**：同时解决两端的缺口——给物理学家一个能自动批量提取谱图的工具，给图学习社区一个真正的大规模空间多重图基准。

**核心 idea**：**[算法]** 用"非布洛赫能带理论 + 代数几何 + 形态学图像处理"三件套把哈密顿量自动转成谱图（Poly2Graph）；**[数据]** 借此蒸馏 177 TB 谱势数据为 1200 万个空间多重图（HSG-12M）；**[洞察]** 进一步指出谱图是多项式/向量/矩阵的"通用拓扑指纹"，搭起 algebra-to-graph 的新桥梁。

## 方法详解

### 整体框架
Poly2Graph 是首个端到端、把任意一维晶体哈密顿量转成谱图的高性能流水线，比已有最优代码快约 $10^5$ 倍。它的关键观察是：谱图本质上是"谱势landscape"的山脊线，而谱势可由特征多项式的根直接算出，于是整条管线被组织成"哈密顿量 → 特征多项式 → 谱势/态密度图像 → 骨架化提取图"四步。

```mermaid
flowchart LR
    A["1-D 晶体哈密顿量 H(z)"] --> B["Laurent 特征多项式 P(z,E)=det[H(z)-EI]"]
    B --> C["解根 + Ronkin 谱势 Φ(E)"]
    C --> D["态密度 ρ(E)=-∇²Φ/2π (2D 图像)"]
    D --> E["二值化 + 自适应细化 + 骨架化"]
    E --> F["NetworkX MultiGraph (含完整边几何)"]
```

### 关键设计

**1. 从哈密顿量到谱势：把"解谱"变成"解多项式根"。** 对一个 $s$-band 紧束缚晶体链，布洛赫哈密顿量写作 $H(z)=\sum_{j=-p}^{q} T_j z^j,\ z=e^{ik}$，其开放边界谱完全由 Laurent 特征多项式 $P(z,E)=\det[H(z)-E I_s]=\sum_{n=-p}^{q} a_n(E)z^n$ 的根决定。于是在复能量平面取一个最小包围方域 $\Omega$（默认对角化 $L=40$ 的实空间小哈密顿量来估），离散成网格；对每个能量 $E$，解 $P(z,E)=0$ 的根并按模排序。这里是朴素方法的瓶颈——要对约 $10^6$ 个网格点各解一遍多项式根。作者用 Frobenius 友矩阵 + 并行特征值求解器（带 GPU 后端自动检测）把单次求根从小时压到毫秒，这是 $10^5$ 倍提速的来源。

**2. Ronkin 谱势与态密度：让"图"成为可计算的图像。** 拿到根之后，依据非布洛赫能带理论把谱势（即代数几何里的 Ronkin function）算成 $\Phi(E)=-\log|a_q(E)|-\sum_{i=p+1}^{p+q}\log|z_i(E)|$，再取其拉普拉斯得到态密度 $\rho(E)=-\frac{1}{2\pi}\nabla^2\Phi(E)$。物理上 $\rho(E)$ 是复平面单位面积的本征态数，$\rho(E)>0$ 的区域恰好勾勒出谱图；几何上态密度是谱势的二阶曲率，所以谱图就是谱势landscape的"山脊"。作者还利用特征多项式的对称性（实系数 ⇒ 谱图关于实轴对称，纯虚系数 ⇒ 关于虚轴对称）只算半平面再镜像，对符合条件的多项式再省最多 50% 算力。

**3. 自适应分辨率 + 形态学提图：解决"分辨率—算力"两难。** 谱图通常只占 $\Omega$ 极小一块面积，全域高分辨率太贵，低分辨率又会丢掉小环、相邻节点等拓扑特征。方案分两阶段：先在 256×256 粗网格上算 DOS、二值化、做 2×2 圆盘形态学膨胀，得到一个保守掩膜把谱图包住、排除约 95–99% 的无关区域；再只在掩膜内把每像素细分成 $m\times m$（默认 4）子网格重算，等效达到 1024×1024 分辨率却只算 1–5% 的点。随后对高分辨 DOS 迭代形态学细化直到一像素宽骨架，识别三类点——交叉节点（≥3 条路径相交）、叶节点（路径终点）、边点（连续段上的点），输出 NetworkX MultiGraph。**关键在于每条边都存下完整的几何信息**：一串有序的 $(\mathrm{Re}\,E,\mathrm{Im}\,E)$ 坐标，既保连接又保每条谱曲线的精确形状，这正是"空间多重图"区别于简单图的本质。

**4. 物理驱动的类别采样：避免冗余、覆盖真实晶体。** 数据集按哈密顿量族（即特征多项式类）分类。采样时尊重数学对称性以避免虚假重复——例如多项式满足 $z$-互易 $P(z)=z^{p+q}P(1/z)$（物理上对应把晶体链左右翻转，谱不变），就只保留一份。具体从基多项式 $\hat P(z,E)=-E^s+z^{-p}+z^q$ 出发，给两个选定单项式赋两个自由复系数 $(a,b)$，对 1–3 band、hopping range 4–6 遍历所有组合（已覆盖现实 1D 紧束缚晶体），去冗余后得 24 个一带类、275 个二带类、1102 个三带类共 1401 类；每类把两个自由系数在 $-10-5i$ 到 $10+5i$ 间各取 13 实×7 虚值，得 $(13\times7)^2=8281$ 个样本。沿单个系数实/虚部连续变化则得到 T-HSG-5M 的时序谱图（拓扑会在相变点突变）。

## 实验关键数据

### 数据集统计
所有变体都源自 HSG-12M，因而全部是空间且不可约的多重图。

| 数据集 | #图 | #类 | 最大/最小类比 | 时序 |
|---|---|---|---|---|
| HSG-one-band | 198,744 | 24 | 1.0 | - |
| HSG-two-band | 2,277,275 | 275 | 1.0 | - |
| HSG-three-band | 9,125,662 | 1102 | 1.0 | - |
| HSG-topology（去同构、不均衡） | 1,812,325 | 1401 | 660.2 | - |
| T-HSG-5M（时序） | 5,099,640 | 1401 | 1.0 | ✓ |
| **HSG-12M（完整）** | **11,601,681** | **1401** | 1.0 | - |

### 主实验：8 个主流 GNN 的图分类
统一对齐可学习参数量、固定训练预算（max epochs=100, max steps=1000），3 个随机种子，报 Top-1 Accuracy / Macro F1 / Top-10 Acc.（节选）：

| 模型 | 指标 | one-band | three-band | topology | HSG-12M |
|---|---|---|---|---|---|
| GCN | Acc / Top-10 | .711 / .999 | .337 / .816 | .397 / .825 | .365 / .841 |
| GAT | Acc / Top-10 | .677 / .998 | .344 / .825 | .434 / .855 | .365 / .846 |
| GIN | Acc / Top-10 | .799 / 1.000 | .050 / .295 | .095 / .390 | .063 / .339 |
| **GINE** | Acc / Top-10 | .764 / 1.000 | **.379 / .872** | **.533 / .927** | **.460 / .921** |
| MF | Acc | .589 | .271 | .348 | .295 |

（GraphSAGE 在匹配参数与预算下整体最佳，HSG-12M 上 Top-10 达 95.2%，CGCNN 94.8%。）

### 关键发现
- **难度随复杂度单调上升**：从 one-band 到 HSG-12M，各项指标随图变大、多重边几何更丰富、同构更复杂、类别更多而稳定下降；显存也随之增长（SAGE 0.067→0.511 MB/图）。
- **边属性至关重要**：边感知的 GINE 远超边无关的 GIN（HSG-12M 上 0.460 vs 0.063），证明多重边的空间几何（长度、直线距离、中点、平均谱势/DOS）携带不可约信号。
- **Top-k 很高，利于逆向设计**：尽管 Top-1 仅中等，Top-10 在完整集上高达 94–95%、易子集近饱和（one-band 99%+），意味着可检索出一小撮候选哈密顿量族交给专家筛选，支撑材料逆向设计。
- **GraphSAGE 在紧预算下最优**：相同参数与预算下全面领先，说明它对空间多重图有更强归纳偏置或样本/算力效率。

## 亮点与洞察
- **把物理问题"翻译"成图问题**：谱图=谱势山脊这一几何视角，让原本要靠精确对角化、人工描点的能谱研究变成一条可批量、可 GPU 加速的图像→图流水线，$10^5$ 倍提速是"能不能做大规模"的分水岭。
- **空间多重图这一图类型的首次大规模落地**：同时保留多重边的**拓扑（边数）**与**几何（边形状）**，直接戳中现有图基准"简单图假设"的盲区，给几何感知图学习提供了真实、丰富、带物理意义的练兵场。
- **algebra-to-graph 的通用桥梁**：作者进一步论证谱图是多项式/向量/矩阵（无论实复）的通用拓扑指纹，把这套工具的适用面从凝聚态物理外推到更广的代数对象，野心不止于一个数据集。
- **属性丰富、featurization 不强加**：原始数据保留完整节点/边几何，作者只给参考的固定长度边摘要特征做 PyG 基准，鼓励社区探索曲线序列、曲率、样条编码等更强表征。

## 局限与展望
- **数值不稳定性**：在 DOS 极低的交叉节点附近偶有数值不稳，靠合并近邻节点、收缩过短边来缓解，复杂情形仍可能出错。
- **范围限定 1D**：目前只覆盖一维紧束缚晶体（hopping range ≤6、≤3 bands），二维/三维晶体、长程跳跃尚未涉及。
- **featurization 仍是开放问题**：参考方案用的是"方向无关的固定长度边摘要"，Top-1 与 Top-10 的差距说明更强的边几何编码（曲率/挠率、高阶矩、Bézier/折线序列）有提升空间，尤其在高类别多样性子集上。
- **任务相对单一**：当前基准聚焦图级分类，时序集的外推/早序列分类、以及"谱图→哈密顿量"的逆向设计任务还停留在 promising 阶段。

## 相关工作与启发
- **非厄米能带理论**：建立在 Ronkin function / 非布洛赫能带的近期进展（Tai & Lee 2023；Xiong & Hu 2023；Wang et al. 2024）之上，把"谱势=Ronkin 函数"这一代数几何工具落到可计算管线。
- **图分类基准**：相比 MUTAG、PROTEINS、OGB（ppa/molpcba）、MALNET 等，HSG-12M 在图数与类数上均居首，且唯一是大规模空间多重图；最接近的 OpenStreetMap 规模更小、缺 ML 任务。
- **时序图**：现有时序图数据多聚焦节点/边级任务，T-HSG-5M 是首个面向图级任务的大规模动态图集合。
- **启发**：当一个领域的"物理量"能被几何化为图（山脊、骨架），就可能借图学习做规模化发现；反过来，物理结构化数据也能逼图学习正视"几何 + 多重边"这一长期被简化掉的维度。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个大规模空间多重图基准 + 首个图级动态图大集 + algebra-to-graph 新视角，跨凝聚态物理与图学习的真正空白点。
- **实验充分度**: ⭐⭐⭐⭐ 8 个主流 GNN × 5 个规模变体、多指标 + 显存/吞吐，结论清晰（边属性重要、Top-k 高、难度梯度合理）；但缺少针对空间多重边的新方法，基准更多在"暴露问题"。
- **写作质量**: ⭐⭐⭐⭐ 物理与 ML 双语境讲得清楚，图1/图2 直观，管线与采样方案交代完整；部分公式与附录依赖较重。
- **价值**: ⭐⭐⭐⭐⭐ 工具（MIT）+ 数据（CC BY 4.0）双开源，既服务材料逆向设计，也为几何感知图学习提供长期可用的练兵基准，潜在影响面广。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data](realpdebench_a_benchmark_for_complex_physical_systems_with_real-world_data.md)
- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](../../ICML2025/physics/erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)
- [\[ICML 2025\] Rethink the Role of Deep Learning towards Large-scale Quantum Systems](../../ICML2025/physics/rethink_the_role_of_deep_learning_towards_large-scale_quantum_systems.md)
- [\[ICLR 2026\] PINFDiT: Energy-Based Physics-Informed Diffusion Transformers for General-purpose Time Series Tasks](pinfdit_energy-based_physics-informed_diffusion_transformers_for_general-purpose.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](../../NeurIPS2025/physics/titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)

</div>

<!-- RELATED:END -->
