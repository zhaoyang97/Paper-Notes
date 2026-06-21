---
title: >-
  [论文解读] LEAP: Local ECT-Based Learnable Positional Encodings for Graphs
description: >-
  [ICLR 2026][图学习][位置编码] LEAP 把"局部欧拉示性变换"（ℓ-ECT）做成一个端到端可训练的局部结构位置编码：对每个节点的 m-hop 邻域算一张可微 ECT 矩阵，再用可学习投影压成低维向量插进 GNN，从而同时编码几何与拓扑信息，且复杂度与一步消息传递同阶。 领域现状：图表示学习长期以消息传递（MP…
tags:
  - "ICLR 2026"
  - "图学习"
  - "位置编码"
  - "Euler Characteristic Transform"
  - "拓扑数据分析"
  - "图神经网络"
  - "可微拓扑"
---

# LEAP: Local ECT-Based Learnable Positional Encodings for Graphs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=8XFPhByERc](https://openreview.net/forum?id=8XFPhByERc)  
**代码**: 待确认  
**领域**: 图学习 / 几何拓扑表示学习  
**关键词**: 位置编码, Euler Characteristic Transform, 拓扑数据分析, 图神经网络, 可微拓扑  

## 一句话总结
LEAP 把"局部欧拉示性变换"（ℓ-ECT）做成一个端到端可训练的局部结构位置编码：对每个节点的 m-hop 邻域算一张可微 ECT 矩阵，再用可学习投影压成低维向量插进 GNN，从而同时编码几何与拓扑信息，且复杂度与一步消息传递同阶。

## 研究背景与动机
**领域现状**：图表示学习长期以消息传递（MPNN）为主，节点反复聚合邻居信息。为缓解 MPNN 的固有缺陷，社区转向给节点注入位置编码（PE）/结构编码（SE），如随机游走编码 RWPE、拉普拉斯编码 LaPE，以及可学习的 SignNet 等。

**现有痛点**：现有 PE/SE 大多只抓**几何**（坐标、曲率、距离）**或拓扑**（拉普拉斯谱、随机游走）单一侧面，表达力受限；MPNN 本身又有过度压缩（oversquashing）、过度平滑（oversmoothing）、无法高效计数子结构等问题。而 ECT（欧拉示性变换）这一"几何-拓扑联合不变量"虽然在足够多方向下对几何图是**单射**（理论上比消息传递更强），但此前以两种方式被使用：DECT 面向**图级**全局描述符、ℓ-ECT 则是一个**静态、不可训练**的节点特征扩展，且把邻域当点云、丢掉了邻域连通性。

**核心矛盾**：ECT 同时具备几何+拓扑表达力和理论单射性，却没有被做成一个**局部、可端到端训练、能直接插进任意 GNN** 的位置编码。

**本文目标**：把 ℓ-ECT 的局部性与 DECT 的可微性结合起来，提出一个既保留拓扑表达力、又能随下游任务联合优化的局部结构 PE。

**核心 idea**：**[可微 + 局部 + 可学习投影]** 对每个节点取 m-hop 邻域 → 算可微 ECT 矩阵 M → 用可学习投影 φ 压成 k 维 PE，方向集 Θ 既可固定也可随训练优化，整条链路端到端可导并与消息传递同阶复杂度。

## 方法详解

### 整体框架
LEAP（Local ECT And Projection）为图中每个节点生成一个 k 维位置编码，分四步：**取邻域 → 特征归一化 → 算可微 ECT 矩阵 → 可学习投影降维**。它不是预处理步骤，而是可嵌入 GNN 内部、对静态特征或 MPNN 隐状态都能作用，并随下游任务端到端训练。

```mermaid
flowchart LR
    A[节点 v 的<br/>m-hop 邻域 Nm] --> B[邻域特征<br/>中心化+归一到单位球]
    B --> C["可微 ECT 矩阵 M ∈ R^|Θ|×|T|<br/>(sigmoid 近似指示函数)"]
    C --> D["可学习投影 φ<br/>(Linear/1D Conv/DeepSets/Attention)"]
    D --> E[k 维位置编码 PE_v]
    E --> F[拼接进 GNN 节点特征<br/>端到端训练]
```

### 关键设计

**1. 可微局部 ECT 作为编码骨架：把拓扑不变量变成可导张量。** 给定特征图 $(G,x)$、方向集 $\Theta\subset S^{d-1}$ 与阈值集 $T\subset[0,1]$，对节点 $v$ 先取 $m$-hop 子图 $N_m(v,G)$。ECT 沿方向 $\theta$ 的本意是统计子图沿 $\langle\theta,x(\cdot)\rangle$ 的过滤序列上"节点数减边数"这一欧拉示性曲线（ECC）：$\mathrm{ECT}(\theta,t)=\sum_{v}\mathbf{1}_{[\langle\theta,x(v)\rangle,\infty)}(t)-\sum_{e}\mathbf{1}_{[\max_{u\in e}\langle\theta,x(u)\rangle,\infty)}(t)$。原始 ECT 因指示函数对方向和坐标不可导而无法进网络，LEAP 沿用 DECT 用 sigmoid 近似指示函数，得到对每个节点一张可微矩阵 $M\in\mathbb{R}^{|\Theta|\times|T|}$，从而让"几何+拓扑"信息以梯度可回传的形式参与训练。靠 ECT 在足够多方向下对几何图单射这一性质，理论上该编码比消息传递更具表达力，并可做子结构计数。

**2. 邻域特征归一化保证结构不变性：让编码只认结构、不认尺度与旋转。** ECT 直接吃原始特征会把无关的尺度、平移混进编码。第二步对邻域节点特征集先**均值中心化**，再除以中心化后集合中的最大范数，把全部特征映到单位球 $S^{d-1}$ 上得到 $F=\{f(u_i)\}$。这一步带来两个直接后果：一是 LEAP 成为名副其实的**局部结构编码**——两个 $m$-hop 邻域相同的节点必然得到相同 PE（与 Rampášek 等人对 local structural encoding 的定义对齐）；二是即便节点特征完全不含信息（合成任务里特征从单位圆盘均匀采样），编码仍能区分不同的边数/拓扑，从而支持非属性图上的学习。

**3. 五种 ECT 投影策略，权衡置换不变性与参数量。** 第四步要把 $|\Theta|\times|T|$ 的 ECT 矩阵压成 $k$ 维向量，且理想上应对 ECC（方向顺序）置换不变以抵消旋转。论文给出五种 $\phi$：**Linear**（拉平后乘 $W\in\mathbb{R}^{k\times D}$，$D=|\Theta||T|$，不置换不变，参数 $O(|\Theta||T|)$）；**1D 卷积**（把 ECT 当多通道时间序列，阈值为时间步、方向为通道，参数 $O(|\Theta|+|T|)$）；**DeepSets**（$\mathrm{PE}=\mathrm{MLP}_2(\sum_{\theta}\mathrm{MLP}_1(\mathrm{ECC}_\theta))$，对方向置换不变、参数与 $|\Theta|$ 无关）；**Attention**（单头无位置编码的 transformer 编码器，置换不变）；**Attention w/ PE**（把每条 $\mathrm{ECC}_\theta$ 与对应方向 $\theta$ 拼接再送入编码器，既置换不变又保留方向信息）。实验显示不同数据集偏好不同投影，给了灵活性。

**4. 可学习方向 + 局部性可调：从"静态扩展"升级为"端到端组件"。** 与把 ℓ-ECT 当静态特征扩展的前作不同，LEAP 的方向集 $\Theta$ 可随机初始化后**固定（LEAP-F）或随训练优化（LEAP-L）**；它能作用于学习到的隐状态（如 HIV 数据集把类别特征经可学习嵌入层映到 $\mathbb{R}^3$ 再算 LEAP），并天然适配图级与节点级任务。局部性由 hop 数 $m$ 控制：默认 1-hop 使复杂度与一步消息传递同阶（有界度下 $O(n)$），也可加大 $m$ 或对多个 $m$ 分别计算后拼接，刻画邻域随尺度演化的轨迹。

## 实验关键数据

### 主实验：TU 分类数据集（最佳方法相对最差基线的相对提升）

| 数据集 | 最佳方法 | 最差 | 最佳 | 相对增益(%) |
|---|---|---|---|---|
| LETTER-H | NoMP + LEAP-L + 1D Conv | 41.6 | 81.6 | 96.2 |
| LETTER-M | NoMP + LEAP-L + 1D Conv | 57.8 | 88.5 | 53.1 |
| LETTER-L | NoMP + LEAP-L + 1D Conv | 80.4 | 98.0 | 21.9 |
| FINGERPRINT | NoMP + LEAP-L + Linear | 48.8 | 55.7 | 14.1 |
| COX2 | GAT + LEAP-L + Attn w/ PE | 77.7 | 80.1 | 3.1 |
| BZR | NoMP + LEAP-L + Linear | 78.3 | 84.7 | 8.2 |
| DHFR | GCN + LEAP-L + Attn w/ PE | 70.1 | 77.6 | 10.7 |

所有数据集上最佳的"架构-PE"组合**都用 LEAP 且方向可学习**；每个数据集-架构组合里两种 LEAP 变体（F/L）都拿下第一与第二。合成数据集（4 万张三节点图、特征从单位圆盘随机采样，按边数四分类）上 LEAP 增强模型达 **100.0±0.0** 准确率，而单独 GCN/GAT 仅 71.83/69.44，证明其能在节点特征无信息时捕获结构。

### 消融：邻域大小与嵌入维度

| 邻域 $N_m$ | LETTER-H | LETTER-M | LETTER-L |
|---|---|---|---|
| LEAP-F, 1-hop | 81.29 | 88.00 | 96.27 |
| LEAP-F, 2-hop | 74.44 | 84.31 | 94.13 |
| LEAP-F, {1,2} | 77.91 | 84.76 | 96.09 |

| 嵌入维 | PE | LETTER-H | LETTER-M | LETTER-L |
|---|---|---|---|---|
| 2 | LaPE | 66.31 | 94.71 | 76.76 |
| 2 | RWPE | 73.82 | 94.62 | 83.47 |
| 2 | LEAP-F | 81.16 | 96.27 | 86.76 |
| 2 | LEAP-L | 78.53 | 95.56 | 87.38 |

### 关键发现
- **1-hop 已足够**：加大邻域反而常常掉点，1-hop 在保持与消息传递同阶复杂度的同时表现最好。
- **大数据集收益更大**：在 20.2 万图的 Alchemy 上 LEAP-L（R²）显著超过所有基线；小数据集（COX2/BZR/DHFR）因样本少收益较弱。
- **与全局 PE 互补**：HIV（AUROC）上 LEAP 是唯一未压制全部基线的情形，LaPE（全局信息）单独最强，但 **LaPE+LEAP 拼接取得整体最佳**，说明局部结构与全局位置信息互补。
- **NoMP 现象**：在 Letter-High/Low 上，无消息传递的 NoMP（仅靠 PE）反超基线 GNN，侧面印证 MPNN 的结构表达局限。

## 亮点与洞察
- **把拓扑不变量端到端化**：ECT 此前要么图级（DECT）、要么静态（ℓ-ECT），LEAP 第一次让"局部 + 可微 + 可学习投影 + 可学习方向"四者合一，真正变成能插进任意 GCN 的训练组件。
- **几何与拓扑一锅端**：相比 RWPE/LaPE 只抓单一侧面，ECT 天生融合两者，且有单射理论背书，表达力上限更高。
- **复杂度友好**：默认 1-hop 下与一步消息传递同阶，且各节点 ℓ-ECT 可并行，工程上易落地。
- **互补而非替代**：与全局 PE（LaPE）拼接能进一步提升，定位清晰。

## 局限与展望
- **理论分析偏轻**：作者明言主要贡献是"新局部 PE + 实证评估"，把更深入的表达力理论分析留作未来工作。
- **全局信息缺失**：作为局部编码，LEAP 抓不到全局位置，HIV 上需借 LaPE 补足。
- **投影策略需选型**：五种投影在不同数据集上各有胜负，缺乏统一最优，实践中要调。
- **方向数/分辨率超参**：ECT 用 16 方向 × 16 阈值，方向与分辨率如何随任务自适应仍未充分探索。

## 相关工作与启发
- **位置/结构编码谱系**：RWPE（局部结构）、LaPE（全局位置）、SignNet（可学习），LEAP 补上了"基于 ECT 的局部可学习结构编码"这一格。
- **拓扑数据分析进网络**：DECT（可微 ECT）、ℓ-ECT（局部静态扩展）是直接前作，LEAP 把二者优点合并。von Rohrscheidt & Rieck (2025) 证明 1-hop ℓ-ECT 含单步消息传递所需的全部信息，是本文的理论起点。
- **启发**：把"有理论保证的几何-拓扑不变量"做成可微、可学习、可插拔模块，是给 GNN 注入归纳偏置的通用范式；与现有全局 PE 拼接的"互补思路"也值得在更多结构编码上复用。

## 评分
- **新颖性**: ⭐⭐⭐⭐ —— 首次把局部 ECT 做成端到端可训练、方向可学的局部结构 PE，几何+拓扑联合且有单射理论支撑，组合创新清晰。
- **实验充分度**: ⭐⭐⭐⭐ —— 覆盖合成 + 多个 TU 数据集 + 20 万图 Alchemy + HIV，含架构/投影/hop/维度多维消融，但理论分析偏轻、部分小数据集增益有限。
- **写作质量**: ⭐⭐⭐⭐ —— 背景—方法—性质—实验脉络清楚，ECT 自包含介绍到位，图示与表格充分。
- **价值**: ⭐⭐⭐⭐ —— 提供了可直接插进任意 GNN 的拓扑感知 PE，且与全局 PE 互补，对图表示学习管线有实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Understanding Truncated Positional Encodings for Graph Neural Networks](../../ICML2026/graph_learning/understanding_truncated_positional_encodings_for_graph_neural_networks.md)
- [\[ICML 2025\] Positional Encoding meets Persistent Homology on Graphs](../../ICML2025/graph_learning/positional_encoding_meets_persistent_homology_on_graphs.md)
- [\[ICML 2025\] L-STEP: Learnable Spatial-Temporal Positional Encoding for Link Prediction](../../ICML2025/graph_learning/learnable_spatial-temporal_positional_encoding_for_link_prediction.md)
- [\[ICML 2025\] Diss-l-ECT: Dissecting Graph Data with Local Euler Characteristic Transforms](../../ICML2025/graph_learning/diss-l-ect_dissecting_graph_data_with_local_euler_characteristic_transforms.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)

</div>

<!-- RELATED:END -->
