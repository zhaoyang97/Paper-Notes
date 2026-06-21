---
title: >-
  [论文解读] Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs
description: >-
  [ICLR 2026][图学习][Temporal Knowledge Graph] 本文提出 HEDRA，第一个在时序知识图谱（TKG）**事件级别**做异质因果解耦的表示学习框架，通过反事实检测、工具变量引导、演化正交三个模块逐级剥离非因果、伪因果，并分离动态/静态因果，在五个真实数据集上取得 SOTA。
tags:
  - "ICLR 2026"
  - "图学习"
  - "Temporal Knowledge Graph"
  - "Event Prediction"
  - "Causal Disentanglement"
  - "Instrumental Variable"
  - "Structural Causal Model"
---

# Beyond Entity Correlations: Disentangling Event Causal Puzzles in Temporal Knowledge Graphs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RdoXks7VmJ](https://openreview.net/forum?id=RdoXks7VmJ)  
**代码**: 待确认  
**领域**: 时序知识图谱 / 图表示学习 / 因果解耦  
**关键词**: Temporal Knowledge Graph, Event Prediction, Causal Disentanglement, Instrumental Variable, Structural Causal Model  

## 一句话总结
本文提出 HEDRA，第一个在时序知识图谱（TKG）**事件级别**做异质因果解耦的表示学习框架，通过反事实检测、工具变量引导、演化正交三个模块逐级剥离非因果、伪因果，并分离动态/静态因果，在五个真实数据集上取得 SOTA。

## 研究背景与动机
**领域现状**：TKG 由事件四元组 $(s, r, o, t)$ 构成，事件预测任务要根据历史事件序列预测未来实体对之间可能发生的关系。主流方法（RE-GCN、TiRGN、DECRL 等）普遍用 GCN+RNN 建模**实体或关系层面的相关性**，或借助超图、演化簇等派生结构捕捉高阶相关。

**现有痛点**：TKG 数据集（如 ICEWS 国际政治事件库）本质是从事件中构造的，事件之间天然包含**异质因果关系**。只盯着实体/关系级别的相关性，无法刻画事件之间真正的因果驱动，对事件预测是不充分的。

**核心矛盾**：作者把事件级因果拆成四类——**静态因果**（时不变的因果依赖，如 IAEA 安全评估为日本排放核污水提供制度框架）、**动态因果**（随时间戳演化的因果，如日本排放当天中国宣布禁止日本海产品进口）、**非因果**（与预测无关，如金砖峰会）、**伪因果**（误导模型的虚假关联，如台风"兰"登陆日本干扰交通，过度关注会让模型把出口变化错误归因于台风而非政策）。难点在于：现有 TKG 缺乏显式监督信号来区分这四类因果，从观测数据中识别和估计它们是非平凡的。

**本文目标**：建立事件级 TKG 结构因果模型（SCM）作为理论框架，并设计可落地的解耦机制，把动态/静态因果留下、把非因果/伪因果剔除，从而学到更具判别力的表示。

**核心 idea（加粗标签）**：
- **理论奠基**：提出事件级 TKG-SCM，用后门调整（backdoor adjustment）+ do-calculus 形式化定义四类因果，识别出 $N$、$P$ 是混杂因子（confounder），$D$、$S$ 互为混杂。
- **逐级解耦**：用三个模块依次阻断后门路径——反事实检测器剥非因果、IV 引导剥伪因果、演化正交分离动态/静态因果。

## 方法详解

### 整体框架
HEDRA 在每个时间戳先用关系感知 GCN 更新实体/关系表示并构造事件表示，随后三个模块串联级联：**反事实检测器**（CDM）借助事件重要性与分布差异生成非因果掩码、配对比损失；**IV 引导解耦模块**（IVDM）构造工具变量分数，把边切成真因果/伪因果并用多视图子图传播 + 鲁棒损失增强；**演化正交模块**（EOM）把真因果表示正交投影成静态/动态分量，用演化损失保持动态分量的时间依赖与静态分量的时间独立。最后由静/动态因果构造事件图、经事件 GCN 精化表示，交给 ConvTransE 解码器预测关系。

```mermaid
flowchart LR
    A[四元组 s,r,o,t] --> B[关系感知GCN<br/>事件表示构造]
    B --> C[CDM<br/>反事实检测器<br/>剥非因果 M^NC]
    C --> D[IVDM<br/>工具变量引导<br/>剥伪因果 M^P]
    D --> E[EOM<br/>演化正交<br/>分离动态/静态]
    E --> F[事件GCN精化<br/>ConvTransE解码]
    F --> G[事件预测 p r̂|s,o]
```

### 关键设计

**1. 反事实检测器：用重要性与分布差异联合判别非因果。** 为避免全连接事件对的二次复杂度，先用 kNN 在表示空间构造候选图 $C$。对每条候选边 $i\to j$，一方面计算注意力式的**事件重要性** $A_{ij}$（重要性越高越可能存在因果依赖），另一方面把每个事件表示映射成对角高斯后验 $q_i=\mathcal{N}(\mu_i, \mathrm{diag}(\sigma_i^2))$，用 KL 散度度量**分布差异** $D_{ij}$（差异越大越不可能因果）。两者在候选图上融合成软非因果掩码 $M^{NC}=1-\sigma\big((\alpha_{attn}\cdot\mathrm{logit}(A+\varepsilon)-\beta_{KL}\cdot D)\odot C\big)$，其中 $\alpha_{attn}+\beta_{KL}=1$ 固定为 0.5。再配一个对比损失 $L_{con}$，让非因果权重低的事件对靠近、权重高的拉远，把无关事件从图中弱化。

**2. IV 引导解耦：用工具变量分离真伪因果并多视图鲁棒化。** 剥掉非因果后，剩余边仍混杂真因果与伪因果。借鉴因果学习中的工具变量（IV）思想，用一个 IV 编码器 $f_{IV}$ 输出每条边的 IV 分数 $\Pi_{ij}=f_{IV}(h^t_{event,i}, h^t_{event,j}, \mathrm{logit}(A_{ij}+\varepsilon), -D_{ij})$。由于 $\Pi_{ij}$ 不直接进入最终打分函数，构成"排他性约束"的神经类比，近似满足 IV 独立性假设。基于门控矩阵 $\tilde\Pi=M^C\odot\Pi$ 取 $\alpha$ 分位数阈值，选 top-$\alpha$ 边为真因果掩码 $M^P$、其余为伪因果。为防 IV 估计不完美，构造真因果/伪因果/全视图三个子图，分别做异质卷积得互补表示，并用鲁棒损失 $L_{rob}=\lambda_{align}L_{align}+\lambda_{sep}L_{sep}$——对齐项把全视图拉向真因果视图、分离项把伪因果视图推离真因果视图。消融显示该模块贡献最大。

**3. 演化正交：正交投影分离静态与动态因果。** 把真因果表示用两个 MLP 编码器投影成静态分量 $h^{S}$ 和原始动态分量；再把原始动态分量相对静态分量做 Gram-Schmidt 式正交化得到纯动态分量 $h^{D,t}_{event,i}=h^{raw,D,t}_{event,i}-\frac{\langle h^{raw,D},h^{S}\rangle}{\|h^{S}\|_2^2+\varepsilon}h^{S}$，保证两者解耦。用分类器判别动/静因果得 $M^D$、$M^S$。再设演化损失 $L_{evo}=\lambda_{dyn}L_{dyn}+\lambda_{stat}L_{stat}$：动态项用 GRU 记忆历史、要求动态分量保持时间依赖；静态项用历史均值、要求静态分量保持时间独立。最后在静态视图与动态视图子图上分别异质卷积、融合归一化更新表示。

**4. 后门调整理论支撑。** 全流程对应 TKG-SCM 上的后门阻断：估计 $C\to Y$ 因果效应时需对混杂 $N$、$P$ 调整，估计 $D\to Y$ 时需对 $S$ 调整，形式化为 $P(Y|do(D))=\sum_S P(Y|do(D),S)P(S|do(D))$。三个模块正是 do-calculus 调整在 TKG 上的可落地实现，把"先剥非因果、再剥伪因果、再分静/动"的级联顺序与理论解耦目标对齐。

## 实验关键数据

### 主实验表格（ICEWS14 / ICEWS18，MRR & Hits）

| 方法 | ICEWS14 MRR | Hits@1 | Hits@3 | Hits@10 | ICEWS18 MRR | Hits@1 |
|---|---|---|---|---|---|---|
| TiRGN (IJCAI 2022) | 41.28 | 29.52 | 46.69 | 70.66 | 42.26 | 30.19 |
| DHyper (TOIS 2024) | 41.71 | 29.37 | 45.69 | 69.32 | 42.84 | 29.96 |
| DECRL (NeurIPS 2024) | 42.90 | 30.49 | 47.06 | 70.01 | 43.36 | 30.64 |
| **HEDRA** | **47.86** | **35.28** | **53.32** | **75.65** | **46.77** | **33.66** |
| Improvement | +11.56% | +15.71% | +13.30% | +2.16% | +7.86% | +9.86% |

五个数据集（ICEWS14/18、GDELT、WIKI、YAGO）平均比 runner-up 提升 MRR 5.70%、Hits@1 7.51%、Hits@3 7.21%、Hits@10 2.30%。GDELT 上 MRR 提升 8.36%、Hits@3 提升 13.11%。WIKI/YAGO 因基线已接近饱和（>98），提升幅度较小但仍稳居第一。唯一例外：ICEWS18 的 Hits@10 略低于 RE-NET（其全局图机制召回更广候选）。

### 消融实验表格（ICEWS14）

| 变体 | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| HEDRA-w/o-CDM（去反事实检测） | 47.11 | 34.25 | 52.12 | 75.04 |
| HEDRA-w/o-EI（去事件重要性） | 47.34 | 34.65 | 52.33 | 75.39 |
| HEDRA-w/o-DD（去分布差异） | 47.25 | 34.46 | 52.63 | 75.35 |
| HEDRA-w/o-IVDM（去IV解耦） | 46.47 | 33.77 | 51.65 | 74.75 |
| HEDRA-w/o-EOM（去演化正交） | 46.24 | 33.49 | 51.79 | 74.10 |
| **HEDRA（完整）** | **47.86** | **35.28** | **53.32** | **75.65** |

### 关键发现
- **IVDM 与 EOM 贡献最大**：去掉这两个模块掉点最明显（MRR 分别降至 46.47、46.24），说明剥离伪因果、分离动/静因果是涨点主因；非因果（CDM）对预测增益相对有限。
- **整体框架自带强骨架**：所有消融变体都共享事件级因果解耦框架，相比传统实体级基线已是强骨架，因此模块级消融掉点数值温和，反映模型鲁棒性。
- **超参不敏感**：对历史窗口长度不敏感，但对候选图的邻居数较敏感；$\alpha_{attn}$、$\lambda_{align}$ 等系数固定 0.5 即可，无需逐数据集调。
- **案例研究**：在 Obama-Xi、香港警察-示威者两个样本上，HEDRA 的 top-5 预测命中正确关系数多于 DECRL，说明因果解耦带来更准的高排名。

## 亮点与洞察
- **第一个把因果解耦下沉到"事件级"的 TKG 工作**：跳出实体/关系相关性的惯性，把事件之间的异质因果作为一等公民。
- **理论与实现对齐漂亮**：先有 TKG-SCM + 后门调整定义四类因果与混杂，再用三个模块逐一对应 do-calculus 的调整顺序，不是"模块堆砌后补故事"。
- **工具变量的神经化用法值得借鉴**：用 IV 分数不进最终打分函数来近似排他性约束，把计量经济学的 IV 思想迁移到无监督因果解耦。
- **正交投影分离动/静因果**简洁有效，Gram-Schmidt + 时间依赖/独立约束的组合可移植到其他动态图任务。

## 局限与展望
- **缺乏显式因果监督**：四类因果的解耦完全靠无监督信号 + 假设（IV 独立性、排他性近似），真因果/伪因果的切分阈值 $\alpha$ 是设计选择，正确性难以直接验证。
- **计算开销增加**：候选图 kNN + 多视图异质卷积 + 三模块级联带来运行时增长（作者称在显著性能提升下可接受），但 GDELT 等大图上 DHyper 都 OOM，HEDRA 的扩展性边界值得关注。
- **对邻居数敏感**：候选图邻居数影响明显，实际部署需调；不同时间粒度（15 分钟 vs 年级）下的最优配置可能差异大。
- **可解释性留白**：虽给了 case study，但缺乏对"模型究竟剥掉了哪条伪因果边"的定量解释或可视化诊断。

## 相关工作与启发
- **TKG 表示学习**：RE-GCN、TiRGN（GCN+RNN 主线），EvoExplore、GTRL、DHyper、DECRL（派生结构/超图建高阶相关）。本文指出它们都停在实体/关系相关性层面。
- **图因果学习**：静态图因果靠生成可解释子图（GNNExplainer、PGExplainer 等）；动态图因果探索时空因果（Zhao & Zhang 2024）。本文补上"伪因果"与"事件级解耦的理论框架"两块空白。
- **启发**：因果解耦 + 工具变量 + 正交投影这套组合，可推广到推荐系统的时序去偏、动态社交网络的影响力归因等需要区分真伪关联的场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个事件级 TKG 因果解耦框架，TKG-SCM 理论 + IV 神经化 + 演化正交三件套都有原创性。
- **实验充分度**: ⭐⭐⭐⭐ 五个数据集全面对比 11 个基线、消融到子模块级、含 case study；但缺因果解耦正确性的直接验证与效率细节。
- **写作质量**: ⭐⭐⭐⭐ 用核污水排放的现实例子讲清四类因果，理论与方法衔接清晰；公式偏密集。
- **价值**: ⭐⭐⭐⭐ 在 ICEWS/GDELT 上平均涨点 5-7%，为 TKG 推理提供了"从相关到因果"的新范式，理论框架有迁移价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Inductive Reasoning for Temporal Knowledge Graphs with Emerging Entities](inductive_reasoning_for_temporal_knowledge_graphs_with_emerging_entities.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](revisting_node_affinity_prediction_in_temporal_graphs.md)
- [\[ICLR 2026\] TGM: A Modular and Efficient Library for Machine Learning on Temporal Graphs](tgm_a_modular_and_efficient_library_for_machine_learning_on_temporal_graphs.md)
- [\[ICLR 2026\] UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction](urbangraph_physics-informed_spatio-temporal_dynamic_heterogeneous_graphs_for_urb.md)

</div>

<!-- RELATED:END -->
