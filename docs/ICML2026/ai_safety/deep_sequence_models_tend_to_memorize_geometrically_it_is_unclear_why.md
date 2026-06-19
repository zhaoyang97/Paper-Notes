---
title: >-
  [论文解读] Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why
description: >-
  [ICML 2026][LLM安全][几何记忆] 本文指出 Transformer / Mamba 在死记硬背图的边时并不会真的退化成查找表（联想记忆），而是会自发把节点嵌入排成一种编码了多跳全局结构的"几何记忆"，并通过 path-star 实验证明这种几何让隐式推理变得反常地容易，但其出现既不能归因于监督、容量也不能归因于优化压力，留下一个新的"记忆之谜"。
tags:
  - "ICML 2026"
  - "LLM安全"
  - "几何记忆"
  - "联想记忆"
  - "隐式推理"
  - "谱偏置"
  - "Node2Vec"
  - "Transformer"
---

# Deep Sequence Models Tend to Memorize Geometrically; It Is Unclear Why

**会议**: ICML 2026  
**arXiv**: [2510.26745](https://arxiv.org/abs/2510.26745)  
**代码**: https://github.com/shahriarnz14/geometric_memory  
**领域**: 可解释性 / 表示学习理论  
**关键词**: 几何记忆, 联想记忆, 隐式推理, 谱偏置, Node2Vec, Transformer

## 一句话总结
本文指出 Transformer / Mamba 在死记硬背图的边时并不会真的退化成查找表（联想记忆），而是会自发把节点嵌入排成一种编码了多跳全局结构的"几何记忆"，并通过 path-star 实验证明这种几何让隐式推理变得反常地容易，但其出现既不能归因于监督、容量也不能归因于优化压力，留下一个新的"记忆之谜"。

## 研究背景与动机
**领域现状**：当下解释 Transformer 参数化记忆的主流抽象是"联想记忆"（associative memory）：每个 token 拿到一个近似随机/正交的嵌入 $\Phi(u)$，事实 $(u,v)$ 则被写入一个权重矩阵 $W_{\text{assoc}}$，对外暴露 logit $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$，本质上就是邻接矩阵在一组随机基下的转写。这种抽象漂亮地解释了 n-gram 统计、key-value 记忆条等现象，已经在大量近期理论与机制可解释性工作中被默认采用。

**现有痛点**：联想记忆只暴露"局部"，做 $\ell$-hop 推理必须把查找操作复合 $\ell$ 次，理论上需要 $\exp(\ell)$ 量级的样本/算力才能在无中间监督时学会。然而越来越多实验现象与这套叙事冲突——路径上的多跳预测、两跳 grokking、知识编辑会牵一发动全身等等都不像"查表"该有的行为。

**核心矛盾**："模型只是个查找表"这套故事和"模型能做隐式多跳推理"这套故事不可能同时正确。要么模型并没在做真正的多跳推理（只是数据有捷径），要么模型存的根本不是查找表。

**本文目标**：(1) 构造一个隐式推理无法被任何捷径解释的极端干净场景，把"模型确实在多跳推理"钉死；(2) 给出一个能容纳这种行为的新型记忆抽象；(3) 解释或至少诚实地承认现有学习理论无法解释这种抽象为什么会出现。

**切入角度**：作者把推理任务搬进权重——graph 不是 in-context 给的而是逼模型把所有边背进参数；然后只对"第一跳"算梯度，把所有可能的隐式课程、链式监督、训练-测试路径重叠这些"作弊空间"全部堵死。

**核心 idea**：把参数化记忆抽象成两套相互竞争的数据结构——联想记忆（邻接矩阵直存）vs 几何记忆（邻接矩阵的低秩谱因子分解 $\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$），实验上反复证实"明明联想记忆更省、更易找，可梯度下降偏偏挑了几何记忆"。

## 方法详解

### 整体框架
这篇论文不提新算法，而是提出一种新的记忆解释框架并用受控实验把它钉死。作者用 path-star 图当沙盒：从一个 root 节点发出 $d$ 条互不相交、长度都为 $\ell$ 的链；训练数据混合"边记忆"（输入节点预测邻居，覆盖全图的边）和"路径寻找"（输入 leaf 输出 root→leaf 整条路径，只在 75% 的 leaf 上训），测试用剩下 25% 端到端从没见过的 leaf。整套设计的目的就是把所有能把多跳问题降阶成单跳的捷径堵死，看模型还能不能预测对"从 leaf 出发第一跳该往哪走"——而这个第一 token 在联想记忆视角下本该需要 $\ell$ 次复合查找。

### 关键设计

**1. In-weights path-star 沙盒：把所有"作弊空间"拔光，逼出真正的隐式推理**

先前关于 in-weights reasoning 的证据要么图太小（<200 节点）、要么只 2 跳、要么训练路径和测试路径有重叠，作者认为这些都不足以证伪"模型只是查表"。于是作者把救命稻草一根根拔掉：(i) 把图烧进权重而不是塞进 context；(ii) 边对称化以避开 reversal curse；(iii) 路径长度全部统一成 $\ell$，去掉隐式课程；(iv) 训练/测试 leaf 路径不相交，去掉子串拼接；(v) 砍掉除第一个 token 之外的所有梯度。在这套极端干净的设定下，即便把图扩到 $5\times10^4$ 节点，Transformer 和 Mamba 仍能学会"第一 token 选哪个 child"。而这个行为在 $f(u)[v]=\Phi(u)^T W_{\text{assoc}}\Phi(v)$ 这种近随机正交嵌入下需要 $\Omega(\exp(\ell))$ 步搜索——救命稻草全拔掉后模型依然成功，谜题就此立住。

**2. 几何记忆 vs 联想记忆的双抽象：用谱因子分解解释"指数难为何变成 1 步"**

要解释 in-weights 推理为什么能成，作者把存储方式重写成 $f(u)[v]=\Phi_{\text{geom}}(u)^T \Phi_{\text{geom}}(v)$，也就是邻接矩阵 $A$ 的（典型低秩的）因子分解，其中 $\Phi_{\text{geom}}$ 与图拉普拉斯 $A-D$ 的顶部特征向量对齐。在这种抽象下，同一条路径上的节点嵌入会聚到一个共同方向 $\mathbf{z}_i$，于是"从 leaf 出发找第一跳"瞬间退化成"在 root 的邻居里挑余弦最大的那个"，$\ell$ 跳问题被降阶成 1 跳。作者用余弦相似度热力图（leaf×first-hop 出现明显对角线）和 3D embedding 可视化，把这套几何结构在 Transformer、Mamba 乃至一个 3 层 MLP 上都重现出来——这正好让"任务成功"和"理论上应该指数难"两件看似矛盾的事同时成立：模型根本没在查 $\ell$ 次表，而是在一个被预先组织好的几何空间里走了 1 步。

**3. Memorization puzzle：把三类经典解释逐一否掉，留下一个诚实的开放问题**

既然联想记忆更省、更易找，为什么梯度下降偏偏挑了几何？作者把三类最常被默认的解释一个个用实验封死。监督压力：把路径寻找监督删掉只剩边记忆，热力图里的几何对角线照样出现。容量/正则压力：把模型调得非常宽（参数足够装下整张 $A$ 的联想存储）并关掉 dropout/weight decay，几何依旧浮现；反过来把嵌入冻住，模型轻松学到纯联想记忆，说明架构本身并不禁止联想，几何是被"主动"选出来的。优化偏置：实验显示梯度下降 2 步就能拼出近似 $\sum_{(u,v)\in E}\Phi(u)\otimes\Phi(v)$ 的联想查找表，而几何要 100 步才浮现——联想其实"更近、更易找"；而且在 sparse 图的记忆任务上联想与几何的 bit/norm 复杂度几乎相等，所以"梯度下降偏好更简洁的解"也救不了几何。三连否之后，作者没有再编一个新故事，而是把"为什么仍然挑了几何"诚实地列为开放问题，逼后续工作从更深层（平坦度、谱范数、深度引发的隐式秩最小化）去找答案——这是全文最大的方法论贡献。

### 损失函数 / 训练策略
全程使用标准 next-token 交叉熵，无额外正则、无 chain-of-thought 监督、无 curriculum；对照实验里也只调三件事：是否冻嵌入、是否保留 weight decay/dropout、是否给路径寻找监督。最小理论分析在 2 层 weight-tied Node2Vec 上做：动力学 $\dot V(t)=\eta\, C(t)\, V(t)$，作者实证显示 $V$ 列向量逐步收敛到拉普拉斯的 Fiedler-like 特征向量，同时 $C$ 的零空间也吃下这些向量，从而梯度自停（"自稳定"），并证明这套谱偏置可以在交叉熵 + 无 bottleneck + 1-hop 监督的极简设定下自然出现，挑战了 Levy & Goldberg 等基于 squared-error 或显式瓶颈的传统 Node2Vec 理论。

## 实验关键数据

### 主实验
在 in-weights path-star 任务上，两类架构（Transformer GPT-mid, Mamba）都在远超随机基线的水平上完成第一 token 预测，而完全相同的图换成 in-context 路径任务时它们都失败，反衬出"几何只在记进权重时才长出来"。

| 设定 | 图规模 / 路径长度 | 第一 token 准确率 | 备注 |
|------|-------------------|-------------------|------|
| In-weights, 完整路径监督 | $5\times 10^4$ 节点, $\ell=6\sim10$ | ≈ 100% (远高于 $1/d$ 基线) | 路径训练/测试不相交 |
| In-weights, 仅第一 token loss | 同上, 75% 路径用于训练 | ≈ 100% | 砍掉所有中间监督仍成功 |
| In-context path-star (B&N'24) | 小图也照样翻车 | 接近 $1/d$ | 作为反例对照 |
| 冻结嵌入的 in-weights | 同 in-weights 设定 | 跌至基线 | 几何被冻死后退化为联想，任务即失败 |

### 消融实验

| 配置 | 现象 | 含义 |
|------|------|------|
| 完整模型（边记忆 + 路径监督） | 出现强对角线热力图 + 几何嵌入 | 几何记忆默认形态 |
| 去掉路径寻找监督，只留边记忆 | 几何依旧出现（Fig. 5），Mamba 信号更强 | 监督压力不是几何成因 |
| 关掉 dropout / weight decay，宽模型 | 几何依旧出现 | 显式容量/正则压力不是成因 |
| 冻结 embedding，其余不变 | 联想记忆可学，路径推理失败 | 架构能表达联想，几何是被"主动"选出的 |
| 优化过程时间线 | 2 步出联想表，100 步才出几何 | 联想 strictly easier-to-find，优化偏置反向 |
| 2 层 weight-tied Node2Vec | 顶部 Fiedler-like 方向独占嵌入秩 | 谱偏置可在无 bottleneck 下自然涌现 |

### 关键发现
- 当模型既能表达联想也能表达几何时，几何总会姗姗来迟地胜出，而且联想记忆形成所需的步数比几何少一到两个数量级——这把"梯度下降偏好简洁解"这套常用解释直接否掉。
- Node2Vec 学到的几何比 Transformer 干净得多（Fig. 1 第三列 vs 第二列），作者由此提出"被联想记忆污染的谱偏置"猜想：Transformer 的 embedding 是 Node2Vec 谱解被局部联想记忆掺了点沙子的版本，意味着调高几何性是有显著 headroom 的实践改进方向。
- weight-untied 或多层模型还会出现"zigzag 几何"——相邻节点嵌入反号、二跳邻居反而靠近——对应邻接矩阵的负特征向量，可解释为模型在"用负方向抵消对角线上的自环 logit"，加 self-loop 即可消失，这与近期关于"加 identity 改善 two-hop"的现象不谋而合。
- 在 2 层 Node2Vec 上观测到的"自稳定"动力学（嵌入往 Fiedler 方向缩，系数矩阵 $C$ 的零空间同时吃下 Fiedler 方向）说明几何并不需要任何 bottleneck/正则项就能出现，这与 Levy & Goldberg 的传统理论形成直接冲突。
- 不同图拓扑（path-star / cycle / grid / irregular）下 Fiedler-like 方向各异，但模型嵌入都自动落在前 2~3 个非退化特征向量上，体现了 cross-entropy 损失下"低秩偏置"的普适性。

## 亮点与洞察
- 把"模型记住事实"的默认抽象从"查找表"换成"嵌入几何"，这种切换看似只是换坐标系，但带来的下游推论非常硬：知识编辑会触发"表征碎裂"、unlearning 难以局部化、可能凭空生成"幻觉关联"，这些都是把记忆视为孤立 key-value 时根本无法预测的现象，但在几何视角下是顺理成章的副作用。
- 实验设计是这篇论文真正的杀招：path-star 加上"只对第一 token 算 loss、路径完全不相交、固定长度去掉课程"，把"模型其实在做隐式多跳推理"这一行为钉到不能再硬的地步，可复用为后续任何想验证"in-weights 推理 vs 子串拼接"的标准 benchmark。
- "联想 strictly easier-to-find，但模型偏偏选了几何"这一观察把"梯度下降偏好简洁解"这条万金油解释拆穿，迫使理论界要么换更精细的复杂度度量（平坦度、谱范数），要么承认深度与因子分解才是真正的隐藏推手。

## 局限与展望
- 几乎所有结论建立在 path-star + cycle + grid 这类 toy graph 上，作者只能用"语言/算术任务里的已知几何（superposition、linear representations、world models）是这种 toy 几何的连续放大版"来做外推性论证，并没在大规模自然语言模型上直接复现。
- 谱偏置的"自稳定"证明只在 2 层 weight-tied Node2Vec 上是闭式的，深度模型里联想/几何如何竞争（哪个时间窗口、哪个学习率、哪个 weight decay 才决定胜负）目前只有定性的初步分析。
- "几何记忆能让 Transformer 拥有更强的全局推理"这一结论意味着 retrieval-style 应用里 generative retrieval 可能优于 dual encoder，但这一推论本文没有验证；同样，对于"几何越强 → 知识编辑越脆"的预测，作者也只是写在 implications 里没做实验。
- "memorization puzzle"留给后人的开放问题非常具体：在交叉熵 + 深度 + 因子分解 + 无瓶颈这组条件下，是否存在某个 max-margin 风格的隐式偏置可以推导出 Fiedler 收敛？目前甚至连 2 层 Node2Vec 的收敛速率都没闭式解。

## 相关工作与启发
- **vs Khona et al. 2024 / Wang et al. 2024（隐式推理 toy 实验）**：他们的路径任务用了路径长度多样性（隐式课程）、给整条路径的监督、训练-测试路径有重叠；本文把这些救命稻草全部拔掉后第一跳预测仍然成功，因此把"几何"的证据强度提到一个新档位。
- **vs Saxe et al.（深度网络的因子分解理论）**：传统因子分解理论用 squared-error，并隐含瓶颈/早停假设；本文证明 cross-entropy 在无瓶颈无正则下也能自然产生谱偏置，把 Node2Vec 理论从"特例"推广到"通用"。
- **vs 经典 Hopfield-style 联想记忆视角（Bietti, Sukhbaatar, Geva 等）**：本文不否定联想记忆视角在解释 disjoint facts 上的有效性，而是论证当事实之间存在隐性结构（哪怕这种结构没在监督里出现）时，联想记忆抽象会系统性地误导我们对 capacity、scaling law、unlearning、knowledge editing 的直觉。
- **vs Huang et al. 2024（two-hop reasoning emergence）**：他们把 two-hop 的出现归因于 attention key/query 矩阵不被合并的因子分解效应，本文将其推广为更普适的"几何记忆 + 谱偏置"机制，并指出即使没有注意力（Mamba、纯 MLP）这种几何仍会出现。
- **vs grokking 文献（Nanda, Power 等）**：grokking 通常被解释为"先记忆、后泛化"的相变；本文把这条相变重写为"先联想存储、后几何重组"，提供了一个新的最小化、可视化、可解析的 grokking sandbox。
- **vs Platonic Representation Hypothesis (Huh et al.)**：本文给"为什么不同模型学到的表征长得像"提供了一种可能的微观机制——只要训练目标里包含 next-token cross-entropy，谱偏置就会把所有模型推向相同的拉普拉斯特征子空间。
- **vs Nichani et al. 2024（associative memory capacity）**：他们给出"在 $m^2$ 参数下 MLP 可装下 $m^2$ 关联"的存在性定理，本文承认这一容量界，但反过来追问"既然装得下，为什么模型不装"，给传统容量分析换了一个方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Geometrically Constrained Outlier Synthesis](geometrically_constrained_outlier_synthesis.md)
- [\[ICML 2026\] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs](old_habits_die_hard_how_conversational_history_geometrically_traps_llms.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](../../NeurIPS2025/ai_safety/factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)

</div>

<!-- RELATED:END -->
