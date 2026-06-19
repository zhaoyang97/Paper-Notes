---
title: >-
  [论文解读] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions
description: >-
  [ICML 2026][可解释性][知识-预测差距] 本文揭示 LLM 在多选题上普遍存在"知识-预测差距"——隐藏层已线性编码正确答案但最终预测却偏离，通过几何分析将该差距归因于知识子空间与预测子空间的错位，并提出 KAPPA 方法在推理时用闭式仿射变换对齐两个子空间，跨模型跨基准一致缩小差距并提升准确率。
tags:
  - "ICML 2026"
  - "可解释性"
  - "知识-预测差距"
  - "线性探针"
  - "子空间对齐"
  - "推理时干预"
  - "多选题"
---

# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions

**会议**: ICML 2026  
**arXiv**: [2509.23782](https://arxiv.org/abs/2509.23782)  
**代码**: https://github.com/holi-lab/KAPPA  
**领域**: 可解释性  
**关键词**: 知识-预测差距、线性探针、子空间对齐、推理时干预、多选题  

## 一句话总结

本文揭示 LLM 在多选题上普遍存在"知识-预测差距"——隐藏层已线性编码正确答案但最终预测却偏离，通过几何分析将该差距归因于知识子空间与预测子空间的错位，并提出 KAPPA 方法在推理时用闭式仿射变换对齐两个子空间，跨模型跨基准一致缩小差距并提升准确率。

## 研究背景与动机

**领域现状**：LLM 在多选题（MCQ）基准上的评估是当前主流做法，但模型频繁出现"能力不一致"——同一问题在自由生成场景下能给出正确答案，切换为 MCQ 格式后却选错。已有研究表明，即使模型答错，对其隐藏层施加简单线性分类器也能提取出正确答案，暗示模型内部已编码足够知识。

**现有痛点**：先前工作主要将 MCQ 错误归因于选项偏见、表面线索或风格伪迹等"表层因素"，但缺乏将这些失败统一联系到模型内部表示的解释框架。对"知识-预测差距"的研究也局限在真实性检测和简单算术等狭窄场景，尚未推广到多样化 MCQ 任务。

**核心矛盾**：模型残差流中同时线性编码了正确答案（知识信号）和实际输出答案（预测信号），但两个信号沿几何上不同的方向路由，导致最终生成时预测信号"覆盖"了知识信号。这不是知识缺失，而是知识利用失败。

**本文目标**：(1) 量化知识-预测差距在多种 MCQ 基准和模型家族上的普遍性与严重程度；(2) 从残差流几何角度解释差距的结构性成因；(3) 设计无需额外训练的推理时干预来弥合差距。

**切入角度**：训练两个线性探针——知识探针预测真实答案、预测探针预测模型输出——将各自权重矩阵视为定义子空间的基向量。如果两个子空间对齐，模型预测就应与其内部知识一致；实测发现差距大的基准上两个子空间主夹角严重偏离。

**核心 idea**：用最小 $\ell_2$ 扰动将隐藏状态在预测子空间中的坐标修正为其在知识子空间中的坐标——即"把预测对齐到知识"。

## 方法详解

### 整体框架

给定一个 MCQ 输入，KAPPA 在残差流的中间层提取隐藏状态 $h$，分别投影到由知识探针和预测探针权重定义的两个 $k$ 维子空间，计算各自坐标（探针 logits）。当两组坐标不一致时，KAPPA 对 $h$ 施加闭式仿射变换，使其在预测子空间中的坐标与知识子空间坐标对齐，修改后的 $h'$ 回写残差流继续前向传播。整个过程无需梯度更新，仅需两组预训练好的线性探针权重。

### 关键设计

**1. 双探针差距量化：把"模型知道什么"和"模型会输出什么"拆成两个可比的信号**

先前工作只能说"模型答错了"，却无法跨基准、跨模型横向比较差距到底有多大。KAPPA 的做法是在每一层 $l$ 提取残差流激活 $h^l(x)$，构建两个并行数据集——知识数据集 $D_{\text{know}}^{(l)} = \{(h^l(x), y)\}$ 把激活配上真实标签，预测数据集 $D_{\text{pred}}^{(l)} = \{(h^l(x), \tilde{y})\}$ 把同一激活配上模型自己的输出标签——再各训一个 $k$ 类线性分类器，得到知识分布 $p_K$ 与预测分布 $p_M$。两者的落差用两个互补指标刻画：预测一致率 $\text{AGR}(x) = \mathbb{I}[\arg\max p_K(x) = \arg\max p_M(x)]$ 捕捉"选对还是选错"的硬性差异，KL 散度 $\text{KLD}(x) = \text{KL}(p_M \| p_K)$ 捕捉置信度分布的软性偏移。单靠准确率会丢掉置信度信息，AGR 与 KLD 一硬一软，正好把差距量到可比的尺度上。

**2. 子空间几何分析：把"模型不听自己的知识"锚定到残差流的几何错位上**

光知道差距大还不够，得解释它从哪来。KAPPA 把每个探针的权重矩阵 $W \in \mathbb{R}^{d \times k}$ 的列向量当作一个子空间的基——知识探针张成"知识子空间"，预测探针张成"预测子空间"——再用主夹角均值（mean principal angle）和 CKA 衡量两个子空间的对齐程度。结果是深层主夹角趋近 $90°$（接近随机基线），CKA 落在 0.4–0.8 的中间带，说明知识信号和预测信号确实共存于同一条残差流，却沿几何上几乎正交的方向各自传播。更关键的是跨 8 个基准的 Spearman 相关分析：子空间错位越严重，实测差距就越大（Llama 3.1 8B 上 $\rho = 0.976, p = 0.001$）。这把抽象的"知识利用失败"坐实成了一个可测量的结构性成因，也为下一步的干预指明了方向——既然问题出在两个子空间错位，那就把它们对齐。

**3. KAPPA 推理时对齐：用闭式最小扰动把预测拉回知识**

既然成因是几何错位，KAPPA 就在推理时直接改隐藏状态，让它在预测子空间里的坐标对齐到知识子空间里的坐标。形式上是一个约束最优化：在保持改动量最小的前提下 $\min_{\tilde{h}'} \|\tilde{h}' - \tilde{h}\|_2^2$，约束修改后的状态满足 $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \tilde{W}_{\text{know}}^\top \tilde{h}$，即"预测坐标 = 知识坐标"。这个问题有闭式解

$$h' = h + W_{\text{pred}}(W_{\text{pred}}^\top W_{\text{pred}})^{-1}(\tilde{W}_{\text{know}}^\top \tilde{h} - \tilde{W}_{\text{pred}}^\top \tilde{h})$$

修改后的 $h'$ 写回残差流继续前向传播。扩展版再引入两个超参收紧对齐：$\tilde{W}_{\text{pred}}^\top \tilde{h}' = \alpha \cdot \tilde{W}_{\text{know}}^\top \tilde{h} + \beta \cdot \text{sign}(\tilde{W}_{\text{know}}^\top \tilde{h})$，其中 $\alpha$ 放大选项之间的相对差异、$\beta$ 把每个选项的 logit 往极值方向推。和 CAA 那类沿固定方向做激活引导的方法不同，KAPPA 为每个输入动态算出"刚好够"的最小扰动，只动预测子空间内的分量、保留正交方向的其它信息；而且闭式解不需要任何迭代优化或梯度更新，推理开销几乎可忽略。

## 实验关键数据

### 主实验

在 6 个差距显著的基准上，KAPPA 跨模型一致提升 ACC 和 AGR：

| 基准 (选项数) | 模型 | Base ACC | KAPPA(6) ACC | Δ ACC | Base AGR | KAPPA(6) AGR |
|---|---|---|---|---|---|---|
| TruthfulQA (4) | Llama 3.1 8B | 56.7 | 73.5 | +16.8 | 62.1 | 77.6 |
| TruthfulQA (4) | Qwen 2.5 7B | 58.8 | 64.1 | +5.3 | 61.8 | 67.3 |
| BBQ-Age (3) | Llama 3.1 8B | 59.9 | 76.8 | +16.9 | 59.2 | 81.1 |
| BBH-Algo (4) | Llama 3.1 8B | 45.1 | 50.1 | +5.0 | 62.1 | 82.5 |
| GSM8k (4) | Llama 3.1 8B | 32.6 | 36.6 | +4.0 | 53.7 | 75.9 |
| BBH-NLP (4) | Qwen 2.5 7B | 61.1 | 63.6 | +2.5 | 69.8 | 74.9 |

跨模型 TruthfulQA 结果（KAPPA(6) vs Base）：

| 模型 | Base ACC | KAPPA(6) ACC | Base AGR | KAPPA(6) AGR |
|---|---|---|---|---|
| Mistral 7B v0.3 | 40.7 | 58.3 | 46.6 | 62.3 |
| Llama 3.1 8B | 56.7 | 73.5 | 62.1 | 77.6 |
| Qwen 2.5 7B | 58.8 | 64.1 | 61.8 | 67.3 |
| Qwen3 4B | 56.5 | 61.4 | 60.0 | 66.1 |
| Qwen3 14B | 71.6 | 77.7 | 76.0 | 83.7 |

### 消融与分析

| 分析维度 | 关键指标 | 说明 |
|---|---|---|
| 对比 CAA/DoLA | KAPPA 在 12/12 个设置上 ACC 均优 | 已有干预方法无法系统性缩小差距 |
| 干预层数 (1/3/6) | 6 层 > 3 层 > 1 层（多数设置） | 多层干预累积效应更强 |
| α, β 超参扫描 | 增大 α 或 β 均单调提升 AGR | 两个超参因果地控制对齐强度 |
| 训练数据量敏感性 | 仅用 10% 数据训练探针仍优于 Base | 低数据场景依然有效 |
| 跨数据集迁移 | TruthfulQA → BBQ-Age: +5.72 AGR | 同类技能任务间子空间部分共享 |
| 自由生成迁移 | TruthfulQA ACC: 41.7 → 44.2 | MCQ 探针可泛化到开放生成 |

### 关键发现

- 知识-预测差距在真实性/偏见基准上最大（TruthfulQA 的知识探针比模型高 +19–21 个点），推理基准次之，知识密集基准最小
- 子空间错位程度与差距高度相关（$\rho = 0.976$），证实差距的几何根源
- KAPPA 并非直接修改答案 token 的 logit（干预层的预测子空间与 logit 空间主夹角约 65°–70°），而是通过修改中间层抽象表示间接影响后续决策

## 亮点与洞察

- **闭式最小扰动对齐**：将知识-预测对齐建模为约束最优化问题并推导闭式解，无需迭代优化，计算开销可忽略，同时保证修改量最小——这个"能用数学解就不用梯度"的思路在推理时干预领域很有启发性
- **双探针作为诊断工具**：同一隐藏状态上训练两个目标不同的线性探针，再比较它们定义的子空间的几何关系，提供了一种通用的"模型内部信号分歧"诊断框架，可迁移到幻觉检测、对齐审计等场景
- **跨格式泛化**：MCQ 上训练的探针和干预策略可迁移到自由生成，说明中间层子空间编码的是抽象语义方向而非特定答案符号，深化了对 LLM 内部表示层次结构的理解

## 局限与展望

- 仅处理线性可访问的知识信号，非线性编码的更深层知识未被触及
- 探针训练需要标注数据和模型预测标签，在完全黑盒场景下不适用
- 自由生成迁移效果有限（GSM8k 自由生成准确率反而微降 0.9 个点），说明 MCQ 子空间与开放生成子空间仍有差异
- 未来可探索：高维非线性对齐、无监督探针发现、与 CoT 提示联合使用以同时弥合推理层和表示层的差距

## 相关工作与启发

- **知识-预测差距文献**：Marks & Tegmark (2024) 在真实性任务上首次发现隐藏层可提取正确答案，本文将该现象推广到通用 MCQ 并给出几何解释
- **推理时干预**：CAA (Rimsky et al., 2024) 用均值差向量做激活引导，DoLA (Chuang et al., 2024) 对比层间 logit——两者均非针对知识-预测差距设计，本文实验显示它们效果有限
- **机制可解释性**：与 Geva et al. (2023)、Park et al. (2024) 关于残差流中高层特征如何被后续层转化为 token 预测的发现一致，KAPPA 的有效性进一步佐证了这一信息流图景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ICML 2026\] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/interpretability/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[CVPR 2026\] Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation](../../CVPR2026/interpretability/where_culture_fades_revealing_the_cultural_gap_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
