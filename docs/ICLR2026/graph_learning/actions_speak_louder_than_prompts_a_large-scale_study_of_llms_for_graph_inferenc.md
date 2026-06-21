---
title: >-
  [论文解读] Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference
description: >-
  [ICLR2026][图学习][LLM-图交互] 这是一篇大规模、可控的实证研究，系统对比 LLM 处理文本图的三种"交互模式"——直接 prompting、ReAct 式工具调用、以及让 LLM 写代码查图的 Graph-as-Code——发现**让 LLM 自己写代码操作图**（而不是把图塞进 prompt）在节点分类上整体最强，尤其在长文本、高度数的稠密图上优势巨大，并且它能在结构、特征、标签三种信号间自适应地切换依赖。
tags:
  - "ICLR2026"
  - "图学习"
  - "LLM-图交互"
  - "节点分类"
  - "Graph-as-Code"
  - "ReAct 工具调用"
  - "依赖性分析"
---

# Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=MgJUj9Sk3C](https://openreview.net/forum?id=MgJUj9Sk3C)  
**代码**: 待确认  
**领域**: 图学习 / LLM 推理  
**关键词**: LLM-图交互, 节点分类, Graph-as-Code, ReAct 工具调用, 依赖性分析

## 一句话总结
这是一篇大规模、可控的实证研究，系统对比 LLM 处理文本图的三种"交互模式"——直接 prompting、ReAct 式工具调用、以及让 LLM 写代码查图的 Graph-as-Code——发现**让 LLM 自己写代码操作图**（而不是把图塞进 prompt）在节点分类上整体最强，尤其在长文本、高度数的稠密图上优势巨大，并且它能在结构、特征、标签三种信号间自适应地切换依赖。

## 研究背景与动机
**领域现状**：在欺诈检测、推荐、信息检索这类"文本丰富的图"场景里，节点分类是核心任务。过去主导范式是图神经网络（GNN），但 GNN 通常要按任务/数据集单独训练、跨域不迁移。LLM 凭借其世界知识和文本理解能力，正快速成为替代方案——做法是把图"语言化"（把邻居、边、标签序列化成文本）塞进 prompt 让 LLM 直接判断。

**现有痛点**：绝大多数已有工作只针对特定域、特定图、特定任务报性能，缺一个**原则性的认识**：LLM 到底擅长怎么和图数据交互？什么时候靠特征、什么时候靠结构、什么时候靠标签？而且大家几乎只用一种交互方式——把图序列化进 prompt——这种做法在节点度数高或节点文本长时会**迅速撑爆 token 预算**，根本塞不下足够的邻域信息。

**核心矛盾**：图信息（结构 + 特征 + 标签）是高度结构化、可能很大的对象，而 prompting 把它压成一段线性文本喂给 LLM，既低效又有损；更糟的是，已有机制分析（Guan et al. 2025）发现 LLM 在 prompting 下往往是在"模仿 prompt 的格式"而非真的在执行图计算。换句话说，**交互媒介本身**可能比模型大小或 prompt 措辞更决定成败，但这一维度从没被系统研究过。

**本文目标**：把影响 LLM-图推理的关键变量全部"因子化"，做一次受控的大规模评测，回答两个问题：(1) 哪种交互模式最强、在什么条件下强；(2) 不同模式各自依赖结构/特征/标签的程度有何不同。

**切入角度**：作者的核心观察是——"行动胜过提示"（Actions Speak Louder than Prompts）。与其把整张图塞进上下文让模型被动阅读，不如给模型**主动行动的能力**：让它发起工具调用、甚至现写代码去按需查询图。主动性越高，模型越能针对每个实例的结构特点裁剪自己的检索和推理。

**核心 idea**：用"让 LLM 写代码操作图（Graph-as-Code）"取代"把图塞进 prompt"，并通过受控变量 + 依赖性剥离实验，证明前者整体更强、更鲁棒、且能自适应切换信息依赖。

## 方法详解

### 整体框架
本文不是提出一个新模型，而是搭建一个**受控评测框架**：把节点分类任务固定下来，沿多个轴系统变化，逐一隔离每个因素的影响。

任务形式化：给定图 $G=(V,E,X,Y)$，已知部分节点标签 $Y_K$、图结构（邻接矩阵 $A$ 或边集 $E$）、以及所有节点的文本特征 $X$，要预测查询节点集合 $Q$ 的标签 $Y_Q$。三种交互模式被统一抽象成函数 $\phi_{\text{prompt}}, \phi_{\text{tool}}, \phi_{\text{code}}: \mathcal{T} \times \mathcal{T}^N \times \{0,1\}^{N\times N} \to \mathcal{T}$，它们各自把"对话历史 + 节点特征 + 图结构"编码成一段有限 token 序列，再交给同一个底层 LLM $\text{LLM}_\theta$ 处理。三种模式的区别**只在于编码与交互方式**，模型本身不变，从而把"交互媒介"这个变量干净地隔离出来。

评测沿六条轴展开：(1) **交互模式**（prompting / 工具调用 / 写代码）；(2) **数据集域**（引文、网页链接、电商、社交）；(3) **结构特性**（同配 homophilic vs. 异配 heterophilic）；(4) **特征长度**（短文本 vs. 长文本）；(5) **模型规模**（从 Llama 到 GPT-5）；(6) **是否带推理能力**（reasoning vs. non-reasoning 变体）。主力模型是 o4-mini。除了报准确率，作者还额外做一层**依赖性剥离**：分别截断文本特征、删除边、移除标签，画成二维准确率热力图，看每种模式各自靠哪类信息吃饭。

### 关键设计

**1. Prompting $\phi_{\text{prompt}}$：把 k-hop 邻域序列化成单轮 prompt**

这是当前最主流也最被本文当作"基线靶子"的模式。它一次性把所有上下文塞进单轮推理：给出全部候选类别、目标节点的文本描述与已知标签、再按跳数（hop distance）分组序列化目标节点的 k-hop 邻域，每个邻居标注其描述和标签（被隐藏的节点标 None）。跳数 $k$ 是控制邻域信息量的超参，作者实验了 0-hop、1-hop、2-hop 三个变体。它的致命弱点是**token 预算**：在高度数图（如 products 平均度 61）或长文本图（如 wiki-cs 平均文本 3215 字）上，2-hop 邻域很快撑爆上下文窗口，表格里直接标 TokenLimit（跑不出来）。为缓解，作者加了一个 **budget prompt** 变体，对每跳邻居做下采样封顶——但下采样会引入噪声和信息丢失，性能仍然垫底。

**2. GraphTool / GraphTool+ $\phi_{\text{tool}}$：ReAct 式 think–act–observe 循环**

受 ReAct 启发，把节点分类变成一个迭代的"思考—行动—观察"回路：每一步 LLM 先推理"已知什么、还缺什么"，然后从固定工具集里发起**单个动作**，环境在图上执行并把结果追加进交互历史，循环直到模型决定终止并给出标签。基础版 GraphTool 提供四个动作：提交最终标签（终止）、只取某节点的邻居（拓扑，不耗特征 token）、只取某节点的文本描述（特征）、只取某节点标签（训练集内才返回，否则 None，防止泄漏）。增强版 GraphTool+ 再加两个"精确 k 跳"批量检索动作：一次取回离某节点恰好 k 跳的所有节点的文本 / 标签。相比 prompting 一股脑全塞，工具调用让模型**有针对性地按需检索**，减少无关曝光和 token 消耗。

**3. Graph-as-Code $\phi_{\text{code}}$：让 LLM 现写代码查图，突破固定动作集**

这是本文主推、也是最强的模式。它把 ReAct 从"固定预定义动作集"推到极致——不再给固定工具，而是把图表示成一张**按 node_id 索引的带类型表**，列包括 features（文本）、neighbors（节点 ID 列表）、label（整数或 None）；LLM 迭代地**生成紧凑程序 → 执行 → 对输出推理**，直到决定终止并预测。它的关键优势在于**组合式访问**：一段代码就能把多步工具序列折叠成单次查询（如"过滤出所有标签为 X 的 2 跳邻居并计数"），既省步数又省 token，同时保持透明可审计。正是这种"按需检索 + 组合"的能力，让它在长文本/稠密图上不会撑爆 token 窗口——它只取回该查询真正需要的结构和特征，而不是把整个邻域线性化塞进去。

**4. 依赖性剥离分析：用截断特征 / 删边 / 删标签量化每种模式靠什么吃饭**

光看总准确率看不出"模型到底依赖哪类信息"。作者设计了一套受控扰动：对 1000 个随机采样的测试节点（5 个种子取平均），分别按比例**截断每个节点的文本到固定 token 百分比**（特征剥离，选截断是因为它简单、模型无关、可复现）、**随机删边**（结构剥离）、**随机删已知标签**（标签剥离），然后把准确率画成二维热力图——一个轴是特征/标签删除率，另一个轴是边删除率。对比 prompting 与 Graph-as-Code 两张热力图的形状，就能读出二者各自对特征、结构、标签的依赖模式是否一致、谁更鲁棒。这套方法论是本文"超越准确率、看内在机制"的核心抓手。

## 实验关键数据

### 主实验
作者按数据集类型分三组报告（短文本同配、异配、长文本同配），统一用 o4-mini，对比三种交互模式与 Random / Majority / Label Propagation 三个经典基线。

短文本同配数据集（节点度数或文本一旦变大就开始撑 token）：

| 数据集 | cora | pubmed | arxiv | products |
|--------|------|--------|-------|----------|
| Label Propagation | 76.61 | 80.80 | 68.00 | 70.40 |
| 1-hop prompt | 81.92 | 91.30 | 73.80 | 82.20 |
| 2-hop prompt | 83.43 | 91.80 | 74.30 | **TokenLimit** |
| GraphTool+ | 81.40 | 91.90 | 73.30 | 78.50 |
| Graph-as-Code | **85.16** | 89.90 | **74.40** | **82.70** |

短文本同配上 prompting 和 Graph-as-Code 咬得很紧（Finding 1）；但注意 products 平均度 61，2-hop prompt 直接 TokenLimit 跑不出，而 Graph-as-Code 照样拿到最高分。

长文本同配数据集（这里 Graph-as-Code 拉开差距）：

| 数据集 | reddit | computer | photo | wiki-cs |
|--------|--------|----------|-------|---------|
| 2-hop prompt | TokenLimit | TokenLimit | TokenLimit | TokenLimit |
| 2-hop budget prompt | 54.40 | 86.00 | 85.60 | 80.80 |
| GraphTool+ | 61.80 | 83.10 | 81.30 | 80.50 |
| Graph-as-Code | 61.60 | **86.20** | **86.40** | **82.20** |

2-hop prompt 在几乎所有长文本图上全线 TokenLimit；即便退到下采样的 budget prompt，仍普遍最弱，而 Graph-as-Code 凭按需检索稳居最优（Finding 4）。

异配数据集（挑战"LLM 在低同配下崩溃"的成见）：

| 数据集 | cornell | texas | washington | wisconsin |
|--------|---------|-------|------------|-----------|
| 同配率 Hom.(%) | 11.55 | 6.69 | 17.07 | 16.27 |
| Label propagation | 41.74 | 78.90 | 15.07 | 14.21 |
| 0-hop prompt | 81.57 | 53.20 | 80.14 | 84.78 |
| Graph-as-Code | **92.70** | 73.60 | 81.96 | 89.17 |

即便同配率低到 6–17%，所有 LLM 交互模式都远超 majority/LP 基线（Finding 3）——说明 LLM 能利用非局部的、基于特征的线索，而非只靠"邻居投票"。

### 关键发现（依赖性剥离）

| 发现 | 内容 |
|------|------|
| Finding 5 | 当 prompt 不撑 token 时，Prompting 与 Graph-as-Code 对**特征 vs. 结构**的依赖几乎一致：同配图（cora/arxiv）删边掉点最多，异配图（cornell）删特征掉点最多 |
| Finding 6 | Graph-as-Code 对特征/结构/标签删除都更鲁棒；当结构被完全删除但特征完好时，Graph-as-Code 仍保持高准确率，而 Prompting 崩溃——因为前者即使没边也能访问其他节点的特征和标签，后者必须靠边来检索这些信息 |
| Finding 7 | 一旦 prompt 触到 token 上限（如长文本稠密图 photo），两者行为**分叉**：Prompting 因撑爆窗口骤降，甚至"丢掉部分特征文本反而更好"，Graph-as-Code 选择性检索不超预算，保持高分 |
| Finding 8–9 | 在**标签 vs. 结构**剥离上二者模式截然不同：Prompting 沿两轴都快速退化，必须同时有结构和标签；Graph-as-Code 只要特征或标签其一在，删边几乎不掉点——它**自适应切换**到当前最有信息量的信号，只有多种信号同时严重退化时才脆弱 |

## 亮点与洞察
- **"交互媒介"是被忽视的关键变量**：本文最大的贡献不是又跑了一遍 benchmark，而是把"LLM 怎么和图交互"提升为一个独立研究维度，并证明它比 prompt 措辞更决定成败——同一个 LLM，换成写代码就显著更强。这个视角可迁移到任何"LLM × 结构化数据"的任务（表格、知识库、数据库）。
- **Token 预算是 prompting 的根本天花板，而非可调超参**：高度数图 products、长文本图 wiki-cs/photo 上 2-hop prompt 直接 TokenLimit。作者点明"即便上下文窗口在扩，LLM 也常用不好长输入"，所以"重组图信息、只突出关键结构"的 Graph-as-Code 不会因模型变大而过时——这是个有前瞻性的判断。
- **自适应依赖切换是 Graph-as-Code 鲁棒性的根源**：它不是"忽略结构"，而是"只在结构比其他信号更有信息量时才用"。这种按需选择信号的能力，正好解释了它为何在异配图、缺信息场景下都稳——一个很漂亮的机制级洞察（Finding 9）。
- **挑战了"LLM 在异配图上崩溃"的流行结论**：之前基于纯 prompting 得出的悲观结论被证明是交互模式的局限，而非 LLM 本身的局限。

## 局限与展望
- **只覆盖节点分类一个任务**：作者自己把范围限定在 node classification，结论能否推广到链接预测、图分类、知识图谱推理尚不清楚。
- **Graph-as-Code 的代价没被量化**：写代码 + 执行 + 多轮迭代显然比单轮 prompting 更慢、更贵，论文主打准确率，但缺少对延迟 / 调用次数 / 美元成本的系统对比——实际部署时这是绕不开的权衡。
- **依赖外部代码执行环境**：Graph-as-Code 需要一个能安全执行 LLM 生成代码的沙箱，存在工程复杂度和安全面（生成代码的正确性、越权访问），论文未深入讨论失败/出错时的回退。
- **截断作为特征剥离的近似**：用"截断到固定 token 百分比"模拟特征缺失虽然可复现，但和真实世界的特征缺失（结构化字段丢失、噪声）未必同分布，热力图结论需谨慎外推。
- **主力只用 o4-mini**：跨规模/推理能力的结果放在附录，正文主结论建立在单一主力模型上，不同模型家族下三种模式的相对排序是否稳定还需更多验证。

## 相关工作与启发
- **vs 图语言化 + prompting（Fatemi et al. 2024；Huang et al. 2024a）**：他们把图编码成邻接表/边列表/叙事文本喂给 LLM，并据此得出"LLM 在异配图崩溃""性能取决于同配度"的结论。本文证明这些结论很大程度是 prompting 这一交互模式的产物——换成工具调用或写代码，异配图照样能打，token 瓶颈也被绕开。
- **vs 工具调用 / ReAct 范式（Yao et al. 2023；Schick et al. 2023）**：以往工具调用用于图多是专用 workflow 或概念验证（GraphRAG、知识图谱推理）。本文把 ReAct 式工具调用系统化、参数化（GraphTool / GraphTool+），并跨多域多结构地评测它对特征/标签/结构的依赖，把它从"能用"推进到"什么时候用、靠什么吃饭"。
- **vs 可学习的图-LLM 组件（Perozzi et al. 2024；Zhao et al. 2024）**：那条线引入可学习模块编码结构与特征。本文走的是纯推理时、无需训练的交互模式对比，主张"给 LLM 行动能力"本身就能带来大收益，是一种更轻量、更可迁移的路线。

## 评分
- 新颖性: ⭐⭐⭐⭐ 不提新模型，但首次把"LLM-图交互模式"立成研究维度并系统证明 Graph-as-Code 的优势，视角新。
- 实验充分度: ⭐⭐⭐⭐⭐ 六轴变量 × 多域多结构数据集 × 依赖性剥离热力图，受控且全面，结论扎实。
- 写作质量: ⭐⭐⭐⭐ 九条 Finding 组织清晰、逻辑层层递进；但部分依赖分析需对照原文热力图才好懂。
- 价值: ⭐⭐⭐⭐⭐ 给实践者直接可用的选型指南（高密度/长文本优先用 Graph-as-Code），对"LLM × 结构化数据"方向有普适启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](../../ACL2026/graph_learning/evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[ICLR 2026\] AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM](atlaskv_augmenting_llms_with_billion-scale_knowledge_graphs_in_20gb_vram.md)
- [\[ICLR 2026\] Discrete Bayesian Sample Inference for Graph Generation](discrete_bayesian_sample_inference_for_graph_generation.md)
- [\[ICLR 2026\] Training-Free Counterfactual Explanation for Temporal Graph Model Inference](training-free_counterfactual_explanation_for_temporal_graph_model_inference.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)

</div>

<!-- RELATED:END -->
