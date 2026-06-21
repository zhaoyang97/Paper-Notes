---
title: >-
  [论文解读] CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers for Causally Constrained Predictions
description: >-
  [ICLR 2026][因果推理][Transformer] 本文提出 **Causal Transformer (CaT)**，把一张预先给定的因果有向无环图（DAG）的邻接矩阵作为掩码注入到 transformer 的交叉注意力中，让网络在保留强函数逼近能力的同时严格遵守因果结构，从而对协变量漂移更鲁棒、更可解释，并能直接估计干预效应。
tags:
  - "ICLR 2026"
  - "因果推理"
  - "Transformer"
  - "DAG 约束"
  - "掩码注意力"
  - "协变量漂移鲁棒性"
  - "干预效应估计"
---

# CaTs and DAGs: Integrating Directed Acyclic Graphs with Transformers for Causally Constrained Predictions

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZIQactmQxb](https://openreview.net/forum?id=ZIQactmQxb)  
**代码**: [https://github.com/matthewvowels1/Causal_Transformer](https://github.com/matthewvowels1/Causal_Transformer)  
**领域**: 因果推断 / 结构归纳偏置  
**关键词**: 因果 Transformer, DAG 约束, 掩码注意力, 协变量漂移鲁棒性, 干预效应估计  

## 一句话总结
本文提出 **Causal Transformer (CaT)**，把一张预先给定的因果有向无环图（DAG）的邻接矩阵作为掩码注入到 transformer 的交叉注意力中，让网络在保留强函数逼近能力的同时严格遵守因果结构，从而对协变量漂移更鲁棒、更可解释，并能直接估计干预效应。

## 研究背景与动机
- **领域现状**：全连接网络和 transformer 是极其灵活的函数逼近器，但它们纯粹依赖统计相关性，不内置任何关于数据生成过程（DGP）的先验知识。
- **现有痛点**：不尊重 DGP 的模型会利用训练集中的非因果关联（经典例子是"沙漠背景=骆驼、草地背景=牛"），一旦这些虚假相关在测试时变化就严重失效——即对**协变量漂移**极度敏感，也难以解释。
- **核心矛盾**：要做无偏的因果效应估计（如 ATE），必须显式施加结构约束来剥离混淆，而这恰恰是"纯统计"的神经网络给不了的；但已有的神经因果方法（CEVAE、CFR、GANITE 等）大多把每个变量当作**单个标量**，无法直接吃高维 embedding（骨架关节、语音/多模态特征、多题量表）。
- **本文目标**：提出一个**通用建模框架**——不是刷某个 benchmark 的 SOTA，而是把"DAG 这种结构归纳偏置"干净地塞进流行的神经架构里，让它既能处理标量表格数据也能处理任意维度的多模态 embedding。
- **核心 idea**：**[掩码即约束]** 把拓扑排序后的 DAG 邻接矩阵 $A$ 作为 Hadamard 掩码作用在交叉注意力分数上，使得每个节点只能"注意"到它在因果图中的合法父节点，从架构层面强制满足 DAG 蕴含的条件独立性。

## 方法详解

### 整体框架
CaT 把输入看成一串与 DAG 节点对应的 $d$ 维 embedding 序列，用一个可学习的"空白"查询向量 $\gamma$ 通过**因果掩码交叉注意力**反复从输入 embedding 中"按图索骥"地抽取信息，逐块（CBlock）传播后由各节点的输出头给出重建/预测。论文还给出一个不带注意力的对照基线 **CFCN**（把 MADE 式自回归掩码推广到 DAG 的全连接网络），用来证明收益来自结构掩码本身而非 transformer 的注意力机制。

```mermaid
flowchart LR
    X[输入 X<br/>B×|Z|×C] -->|每节点独立线性层| XE[嵌入 XE<br/>B×|Z|×dE]
    DAG[输入 DAG<br/>邻接矩阵 A] --> Mask[拓扑排序 + 掩码]
    gamma[可学习查询 γ] --> CA
    XE -->|K, V| CA[因果交叉注意力<br/>softmax A⊤∘QKᵀ/√hs · V]
    Mask --> CA
    CA --> Block[CBlock: 多头 + FF + 两条残差]
    XE -. 每层都重新喂入 .-> Block
    Block -->|堆叠 × Blocks| Out[各节点输出头<br/>重建/预测 X̂]
```

### 关键设计

**1. DAG 掩码交叉注意力：把因果约束写进 softmax**。这是 CaT 的核心修改。输入 $X$（形状 $B\times|Z|\times C$）先用 $|Z|$ 个**独立**线性层嵌入成 $X_E$（$B\times|Z|\times d_E$，要求 $d_E\ge C$ 且 $d_E>1$，否则网络分不清不同变量的贡献）。注意力的键值来自输入、查询来自 $\gamma$：$K=X_EW_K,\ Q=\gamma W_Q,\ V=X_EW_V$。关键一步是用拓扑排序后的转置邻接矩阵 $A^\top$ 对注意力分数做逐元素相乘后再 softmax：

$$O = \mathrm{softmax}\!\left(\frac{A^\top \circ QK^\top}{\sqrt{h_s}}\right)\cdot V.$$

这样每个节点（查询位置）只能把注意力权重分配给它的因果父节点，DAG 蕴含的条件独立性在前向传播中被严格保留（论文在附录 S13 给出证明）。多个头并行后拼接、投影，组成因果多头交叉注意力 CMCA。

**2. 可学习查询 $\gamma$ + 每层重喂输入：让"当前估计"对照"观测"**。与常规交叉注意力在两个有信息的 embedding 之间做不同，CaT 的查询 $\gamma$ 初始是随机的"空白" embedding（$|Z|\times d_E$），它在网络中逐块被填充成输出 $O^r_{\text{Block}}$。精妙之处在于：$X_E$ 在**每一层每一块都被重新喂入**作为键值，而 $\gamma$（及其后继 $O^r_{\text{Block}}$）始终作为查询。这让下游的 $O^r_{\text{Block}}$ 既能注意自己的上一状态，又能注意观测输入，从而"比较当前估计与真实观测之间的落差（surprise）"，决定还需要从输入里（在 DAG 约束下）抽取什么。CBlock 用两条残差连接 + 可选 BatchNorm 把 CMCA 和两层 Swish 前馈包起来：$O^{r,\prime}_{\text{CMCA}}=\mathrm{BN}(\mathrm{CMCA}(O^{r-1}_{\text{Block}},X_E)+O^{r-1}_{\text{Block}})$，$O^{r}_{\text{Block}}=\mathrm{BN}(\mathrm{FF}(O^{r,\prime}_{\text{CMCA}})+O^{r,\prime}_{\text{CMCA}})$。特别地，论文**不用 LayerNorm**——因为它会重缩放可能已校准的干预值，破坏因果量的尺度。

**3. CFCN 对照基线：层级 identity 处理的 DAG 全连接掩码**。为隔离"结构掩码"本身的价值，作者把 MADE 的思路从自回归顺序推广到任意 DAG：对每层预分配掩码 $M_r$，前向 $o_r=\sigma(o_{r-1}(W_r\circ M_r)+b_r)$，并约定每层神经元数不少于输入维、且为相邻层的整数倍/因子，再把邻接矩阵 reshape 到各层权重形状。一个关键技巧是**从第二层起给 DAG 加单位对角**：第一层充当"屏障"禁止变量看到自己（保证无父节点的外生变量不自我泄漏、也防止输入直接旁路到输出），但中间层一旦发生了因果合法的交互，就需要 identity 把这些信号继续向输出传播。CFCN 像个"输入即目标"的自编码器，但目标只能由其结构父节点预测，输出反映 DAG 蕴含的条件分布。

**4. 递归干预推断：用 g-formula 在拓扑序上传播 do 操作**。约束模型不仅鲁棒，还能回答可识别的干预查询（ATE/CATE）。做法是：把被干预变量设为干预值，然后沿拓扑序**递归更新所有后代**——以父节点的当前值重新评估对应节点头，按 g-formula 把效应一层层往下游传。例如混淆链 $D\to M\to Y$ 加 $D\leftarrow L\to Y$，通过 do-calculus 识别 $Y\mid do(D{=}1)=Y\mid D{=}1,L$（条件于 $L$ 切断虚假路径），且要排除中介 $M$（传递性约简，估计总效应时 $D\to M\to Y$ 简化为 $D\to Y$）。损失上 CaT 用**逐变量损失**（连续 MSE / 二值 BCE / 类别 softmax / 向量 embedding MSE），不需要任何特殊正则项，这正是其简洁性所在。

## 实验关键数据

### 主实验：因果推断 benchmark（Twins / Jobs，越低越好）

| Model | Twins eATE (ws) | Twins eATE (os) | Jobs Risk (ws) | Jobs eATT (os) |
|---|---|---|---|---|
| **CaT** | .0110 | .0133 | .25 | .086 |
| **CFCN** | .0098 | .0102 | .25 | .084 |
| GANITE | .0058 | .0089 | .13 | .06 |
| CFR-wass | .0112 | .0284 | .17 | .09 |
| BART | .1206 | .1265 | .23 | — |
| C Forest | .0286 | .0335 | .19 | .07 |

CaT/CFCN **未做任何超参搜索、仅假设"非处理非结果变量都可能是混淆"** 这一最弱先验，却与专门为因果推断设计调优的方法（GANITE、CFR、TVAE 等）打到同一量级。

### 关键对照：玩具模拟（最低 MSE ≠ 正确因果，越低越好）

| Model | MSE | eATE |
|---|---|---|
| CaT True DAG | 1.004 | **0.058** |
| CaT False DAG 1 | 0.555 | 2.314 |
| CFCN True DAG | 1.023 | 0.149 |
| Transformer | 0.615 | 2.379 |
| MLP | 0.559 | 2.325 |
| Rand. Forest | 0.622 | 2.311 |

### 关键发现
- **只有用对的因果图，eATE 才最低**：用错误 DAG 或非因果模型反而常有更低的预测 MSE（更高的拟合力），但 ATE 估计偏差大、鲁棒性差。
- **协变量漂移下 CaT/CFCN 稳如磐石**：Fig.1 显示随机森林/MLP/transformer 的预测随虚假/混淆因素剧烈波动，而 CaT/CFCN 因只用漂移不变的因果关系而保持稳定。
- **真实心理学应用**：在 COVID-19 依恋风格→抑郁的分析（895 人）中，CaT 直接吃**全部问卷题目**（而对照方法只能用聚合量表分），无需调参就给出与半参数 Targeted Learning 一致的结论，展示了多维输入的灵活性。

## 亮点与洞察
- **掩码注意力 = 结构归纳偏置的极简实现**：一行 Hadamard 乘法就把 DAG 的条件独立性焊死进注意力，无需子网络、无需额外损失/正则。
- **高维变量的因果建模**：突破了既往神经因果方法"每变量一个标量"的束缚，可把每个变量当作向量 embedding，天然支持多模态/多题量表。
- **"低 MSE 陷阱"的清晰示范**：用一张简单模拟表说明预测精度高不等于因果正确，强调评估因果模型要看 eATE 而非拟合误差。
- **诚实的定位**：作者反复强调目标不是刷 SOTA，而是给社区一个可改造的通用基座，附带 CFCN 作为消融对照证明收益来自结构掩码而非注意力本身。

## 局限与展望
- **递归推断的误差累积**：当 DAG 有长中介链、且干预节点位于因果序很早的位置时，逐节点把中介预测反馈回模型会让误差层层放大；可用传递性约简删掉不重要的中介节点缓解。
- **依赖已知正确 DAG**：全文假设 DAG 先验已知，现实中很多复杂现象难以指定正确因果图，图错则约束错。
- **依赖强且不可检验的假设**：因果识别仍需 ignorability、positivity、SUTVA 等通常无法检验的前提。
- **非为打榜设计**：在专门优化过的因果 benchmark 上不指望领先，真实应用（如心理学）也无 ground truth，结果主要作为"可用性"演示。

## 相关工作与启发
- **结构 vs 函数归纳偏置**：本文属于"结构偏置"（约束变量间交互），区别于卷积、权重衰减等"函数偏置"；与解耦表示（β-VAE 等）、神经因果推断（CEVAE/CFR/GANITE/TVAE）同源但路线不同。
- **掩码网络谱系**：CFCN 直接继承 MADE 的自回归掩码思想，并与 Graphical Normalizing Flows、Structured Neural Networks 一脉相承，但 CaT 把掩码从全连接搬到了交叉注意力。
- **与"自回归=因果"的区分**：作者明确指出 transformer 常说的"causal masking"（只看过去 token）只是 Granger 因果式的弱约束，无法施加高层解耦或干预识别，CaT 的 DAG 掩码是真正的结构因果约束。
- **启发**：把领域知识以邻接矩阵形式注入注意力，是一种通用且低成本的"知识即架构"范式，可推广到任何有已知依赖结构的预测任务（不限于因果）。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把 DAG 邻接矩阵作为掩码注入交叉注意力的做法干净优雅，且支持高维 embedding，相比"每变量标量"的既有神经因果方法是清晰的概念推进。
- **实验充分度**: ⭐⭐⭐ 覆盖模拟、两个标准 benchmark 和一个真实心理学应用，且有 CFCN 消融；但刻意不调参、不追 SOTA，benchmark 上并不领先，说服力偏"演示"性质。
- **写作质量**: ⭐⭐⭐⭐ 动机、方法、对照逻辑清晰，"低 MSE ≠ 因果正确"的表设计很有教学价值，公式与图示完整。
- **价值**: ⭐⭐⭐⭐ 作为可复用、可扩展的通用因果建模基座，对需要鲁棒性/可解释性/因果效应的应用领域（医学、心理、政策评估）有实际推广价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](../../ACL2025/causal_inference/causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)
- [\[ICLR 2026\] Theoretical Guarantees for Causal Discovery on Large Random Graphs](theoretical_guarantees_for_causal_discovery_on_large_random_graphs.md)
- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](../../AAAI2026/causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[ICLR 2026\] Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions](learning_dynamic_causal_graphs_under_parametric_uncertainty_via_polynomial_chaos.md)

</div>

<!-- RELATED:END -->
