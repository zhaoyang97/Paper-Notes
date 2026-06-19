---
title: >-
  [论文解读] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion
description: >-
  [NeurIPS 2025][图学习][知识图谱推理] DuetGraph 提出双通路（消息传递 + 全局注意力）并行融合模型与粗到精推理优化策略，通过分离而非堆叠局部/全局信息处理来缓解 KG 推理中的分数过平滑问题，在归纳与传导推理任务上取得 SOTA，MRR 最高提升 8.7%、训练加速 1.8×。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "知识图谱推理"
  - "双通路融合"
  - "粗到精优化"
  - "过平滑"
  - "图神经网络"
---

# DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion

**会议**: NeurIPS 2025  
**arXiv**: [2507.11229](https://arxiv.org/abs/2507.11229)  
**代码**: [https://github.com/USTC-DataDarknessLab/DuetGraph](https://github.com/USTC-DataDarknessLab/DuetGraph)  
**领域**: 图学习  
**关键词**: 知识图谱推理, 双通路融合, 粗到精优化, 过平滑, 图神经网络

## 一句话总结
DuetGraph 提出双通路（消息传递 + 全局注意力）并行融合模型与粗到精推理优化策略，通过分离而非堆叠局部/全局信息处理来缓解 KG 推理中的分数过平滑问题，在归纳与传导推理任务上取得 SOTA，MRR 最高提升 8.7%、训练加速 1.8×。

## 研究背景与动机

**领域现状**：知识图谱（KG）推理旨在补全缺失的实体/关系，主流方法包括消息传递网络（捕捉局部邻域）和 Transformer（捕捉全局依赖）。近年最优方法（KnowFormer 等）将两者在单阶段中堆叠，同时利用局部和全局信息。

**现有痛点**：单通路堆叠设计导致**分数过平滑**——正确答案和错误答案的得分趋于一致（如 Figure 1 所示，多数基线中大量错误实体得分接近正确答案），使推理质量下降。

**核心矛盾**：
   - **Challenge 1**：消息传递层和注意力层各自堆叠就会加剧过平滑，二者串联堆叠这些效应累积更严重。
   - **Challenge 2**：单阶段模型判别能力有限，一次性推理生成答案无法有效区分候选实体。

**本文目标**：同时缓解两个过平滑来源——信息混合（Challenge 1）和单阶段判别不足（Challenge 2）。

**切入角度**：将局部/全局信息处理**分离到两条独立通路**而非堆叠；将推理分解为两阶段：粗粒度筛选 + 精细化判别。

**核心 idea**：用"分离代替堆叠"的双通路融合 + "粗到精"两阶段推理，从架构和推理策略双管齐下缓解 KG 推理中的分数过平滑。

## 方法详解

### 整体框架
输入为知识图谱 $\mathcal{G}=\{\mathcal{V},\mathcal{E},\mathcal{R}\}$ 和查询 $(h,r,?)$。框架包含两大组件：
- **训练阶段（Steps ①-④）**：双通路融合模型——GNN 编码器输出实体/关系表示，然后全局注意力通路和局部消息传递通路分别计算权重矩阵，最后通过 MLP 自适应融合。
- **推理阶段（Steps ⑤-⑧）**：粗到精优化——粗模型生成初始实体得分表，按 Top-k 划分高/低分子表，精模型更新两个子表，基于阈值 $\Delta$ 选择最终答案。

### 关键设计

1. **双通路全局-局部融合模型（Dual-Pathway Fusion）**:

    - 功能：将局部消息传递和全局注意力分离为两条独立通路，各自生成表示后自适应融合。
    - 核心思路：全局通路用简单全局注意力机制得到 $Z_{\text{global}}$，局部通路用查询感知消息传递网络得到 $Z_{\text{local}}$，最终融合为 $Z = \alpha \cdot Z_{\text{local}} + (1-\alpha) \cdot Z_{\text{global}}$，其中 $\alpha$ 为可学习参数。
    - 设计动机：相比单通路串联，双通路并行处理避免了消息传递和注意力层之间的相互干扰。理论证明（Theorem 1）当 $\alpha < \frac{\sigma_{\max}(\mathcal{M}_D)+\sigma_{\max}(\mathcal{M}_O)}{1+\sigma_{\max}(\mathcal{M}_O)}$ 时，双通路模型的分数差上界衰减更慢，即更抗过平滑。
    - 时间复杂度优势：双通路并行执行，复杂度为 $\mathcal{O}(\max(L_m(|\mathcal{E}|d+|\mathcal{V}|d^2), L_t|\mathcal{V}|d^2))$，优于单通路的 $\mathcal{O}(L_m|\mathcal{E}|d+(L_m+L_t)|\mathcal{V}|d^2)$。

2. **粗到精推理优化（Coarse-to-Fine Reasoning）**:

    - 功能：将单阶段推理拆分为粗粒度排序 + 精细化判别两个阶段。
    - 核心思路：
        - **粗阶段**：用已有模型（如 HousE、RED-GNN）生成初始得分表 $\mathcal{T}$，按 Top-k 分为高分子表 $\mathcal{T}^{\text{high}}$ 和低分子表 $\mathcal{T}^{\text{low}}$。
        - **精阶段**：用双通路模型分别更新两个子表的得分；取各子表最高分实体 $(e_h, s_{e_h})$ 和 $(e_l, s_{e_l})$，若 $\gamma = s_{e_l} - s_{e_h} > \Delta$ 则选低分子表中的最高分实体，否则选高分子表。
    - 设计动机：Theorem 2 证明两个子表最高分的期望差距下界为 $\mathbb{E}[|s_{e_h}-s_{e_l}|] > |(\frac{1}{N_h^2+1}-\frac{1}{N_l^2+1})\cdot\sigma|$，由于 $N_l/N_h > 1000$，该下界超过 $0.1\sigma$，远大于基线方法的 $0.02\sigma$。Theorem 3 进一步证明 $P > P'$，即粗到精优化提高了正确预测的概率。

3. **损失函数**:

    - 对每个训练三元组 $(h,r,t)$：$\mathcal{L} = -\log(\sigma(t|h,r)) - \sum_{t'}\log(1-\sigma(t'|h,r))$
    - 负采样通过 mask 正确答案后从剩余实体均匀采样，使用 Adam 优化器。

## 实验关键数据

### 主实验（归纳推理）

| 数据集 | 版本 | 指标 | DuetGraph | KnowFormer (之前SOTA) | 提升 |
|--------|------|------|-----------|----------------------|------|
| FB15k-237 | v1 | MRR | **0.507** | 0.466 | +8.8% |
| FB15k-237 | v2 | MRR | **0.549** | 0.532 | +3.2% |
| WN18RR | v3 | MRR | **0.501** | 0.467 | +7.3% |
| NELL-995 | v1 | MRR | **0.850** | 0.827 | +2.8% |
| NELL-995 | v2 | MRR | **0.543** | 0.465 | +16.8% |
| NELL-995 | v4 | MRR | **0.464** | 0.378 | +22.8% |

### 主实验（传导推理）

| 数据集 | 指标 | DuetGraph | AdaProp / HousE | 提升 |
|--------|------|-----------|-----------------|------|
| FB15k-237 | MRR | **0.419** | 0.417 | +0.5% |
| WN18RR | MRR | **0.573** | 0.562 | +2.0% |
| NELL-995 | MRR | **0.560** | 0.554 | +1.1% |
| YAGO3-10 | MRR | **0.580** | 0.573 | +1.2% |

### 消融实验

| 配置 | FB15k-237 v1 MRR | NELL-995 v1 MRR | 说明 |
|------|-------------------|-----------------|------|
| Full DuetGraph | **0.507** | **0.850** | 完整模型 |
| w/o coarse-to-fine | ~0.48 | ~0.83 | 去掉粗到精后推理质量下降 |
| w/o dual-pathway (single) | ~0.47 | ~0.82 | 退化为单通路堆叠，过平滑加剧 |
| 不同粗模型选择 | varying | varying | HousE/RED-GNN 等均可作为粗模型 |

### 关键发现
- **双通路融合**和**粗到精优化**各自独立贡献性能提升，两者组合效果最佳。
- DuetGraph 在 NELL-995 数据集上提升最为显著（MRR 最高提升 22.8%），说明在关系多样性较高的场景下优势更大。
- 训练效率提升 1.8×，得益于双通路并行计算。
- 超参数 $k$（Top-k 分割点）的选择具有一定鲁棒性，但过小会丢失正确答案，过大则子表区分不够。

## 亮点与洞察
- **"分离代替堆叠"**思想很巧妙：既有的方法默认将不同类型信息层层叠加，本文指出这恰恰是过平滑的根源，分离处理后反而两种信息各自保持 discriminative power。这个思路可迁移到其他多源信息融合任务。
- **粗到精的两阶段推理**本质上是增加了一个"对比"环节：不是直接取最高分，而是在两个子集之间比较，放大了正确/错误答案的分数差距。类似 cascaded refinement 策略可用于推荐系统等排序任务。
- 理论分析扎实，从 Lemma 1-3 到 Theorem 1-3 形成完整的证明链，涵盖过平滑上界、双通路优势和粗到精有效性。

## 局限与展望
- 粗模型的选择目前依赖人工指定，若粗模型质量差（初始排序混乱），Top-k 分割可能将正确答案分到低分子表，影响最终效果。
- 仅关注尾实体补全 $(h,r,?)$，虽然其他任务可转换，但直接多任务联合训练可能更高效。
- $\alpha$ 是全局标量，对不同查询类型的自适应能力有限；如果改为 query-dependent $\alpha$ 可能进一步提升。
- 在传导任务上提升相对较小（<2%），说明过平滑问题在归纳场景下更严重。

## 相关工作与启发
- **vs KnowFormer**: KnowFormer 是当前 SOTA 的混合方法，将消息传递和 Transformer 堆叠在单阶段中。DuetGraph 的双通路设计直接对标其单通路方案，在 12 个归纳数据集上全面胜出。
- **vs RED-GNN / A*Net**: 这些是纯消息传递方法，DuetGraph 通过额外的全局通路补充了长程依赖建模能力。
- **vs TransE / RotatE**: 三元组方法忽略图拓扑，DuetGraph 结合局部+全局结构信息后在传导任务上也明显优于它们。

## 评分
- 新颖性: ⭐⭐⭐⭐ 双通路分离融合和粗到精推理的组合设计新颖，但各组件的灵感来源（并行通路、级联筛选）在其他领域已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 12个归纳子集+4个传导数据集+三元组分类+消融+效率分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论分析完整，但符号较多，部分证明需要反复对照
- 价值: ⭐⭐⭐⭐ 分数过平滑是 KG 推理的真实痛点，提出的方案实用且有理论支撑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](../../ICML2026/graph_learning/l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[ICML 2026\] DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA](../../ICML2026/graph_learning/dtkg_dual-track_knowledge_graph-verified_reasoning_framework_for_multi-hop_qa.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[NeurIPS 2025\] SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation](spot-trip_dual-preference_driven_out-of-town_trip_recommendation.md)

</div>

<!-- RELATED:END -->
