---
title: >-
  [论文解读] DISCO: Diversifying Sample Condensation for Efficient Model Evaluation
description: >-
  [ICLR 2026][LLM评测][高效评测] DISCO 提出"挑选让模型们意见最不一致的样本"这一极简准则来压缩评测集，配合"模型签名 + 简单回归"直接预测全量性能，在 MMLU/HellaSwag/Winogrande/ARC 上用 100 个样本就把评测成本砍掉 99% 而误差仅约 1 个百分点，刷新了高效评测的 SOTA。
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "高效评测"
  - "样本压缩"
  - "模型分歧"
  - "性能预测"
  - "锚点选择"
---

# DISCO: Diversifying Sample Condensation for Efficient Model Evaluation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SoOgBHa3dZ](https://openreview.net/forum?id=SoOgBHa3dZ)  
**代码**: [DISCO Codebase（项目页提供）](https://openreview.net/forum?id=SoOgBHa3dZ)  
**领域**: 高效模型评测 / LLM Evaluation  
**关键词**: 高效评测, 样本压缩, 模型分歧, 性能预测, 锚点选择  

## 一句话总结
DISCO 提出"挑选让模型们意见最不一致的样本"这一极简准则来压缩评测集，配合"模型签名 + 简单回归"直接预测全量性能，在 MMLU/HellaSwag/Winogrande/ARC 上用 100 个样本就把评测成本砍掉 99% 而误差仅约 1 个百分点，刷新了高效评测的 SOTA。

## 研究背景与动机
**领域现状**：现代大模型的评测已经贵到离谱——LMMs-Eval 在 8×A100 上跑一个模型要 30 到 1400 小时，HELM 单模型超过 4000 GPU 小时。为了省钱，高效评测方法应运而生，主流框架分两步：先从全量测试集里挑出一小撮"锚点(anchor)"样本，再训练一个从锚点准确率到全量结果的映射来外推。

**现有痛点**：锚点选择普遍依赖**聚类**——把样本按它们在一批参考模型上诱发的"响应相似度"分组，再从每个簇里取代表点(如 Anchor Points、tinyBenchmarks)。聚类本身复杂、对设计选择(距离度量、簇数、嵌入方式)敏感；而性能预测端又往往要先估计模型的隐变量(如 IRT 的能力参数 θ)再做预测，引入了额外的心理测量学复杂度。

**核心矛盾**：大家默认"锚点要覆盖样本多样性、要代表数据难度分布"。但本文反问：**真正决定能不能区分和排序模型的，并不是样本本身多样，而是样本能否引发模型间响应的多样**。一个所有模型都答对(或都答错)的样本，对区分模型几乎零信息；只有那些"有的模型对、有的模型错"的样本才真正有判别力。

**本文目标**：把高效评测的两端都做简单——选样本端用逐样本统计替代全局聚类，预测端用原始输出直接回归替代隐变量建模。

**核心 idea**：**【信息论最优的贪心选样】** 证明"模型间分歧(inter-model disagreement)"在以区分/排序模型为目标时，是估计基准性能信息论意义上最优的逐样本信号，于是只需按分歧分数排序取 top-k，无需任何聚类。**【模型签名直接回归】** 把模型在所选子集上的原始输出拼接成"模型签名"，直接喂给简单预测器(随机森林/kNN)，绕开 IRT 的隐参数估计。

## 方法详解

### 整体框架
DISCO 沿用"选样本 → 预测性能"两段式，但两段都换成更简单的部件。给定一批已知全量性能的**源模型** $F=\{f^1,\dots,f^M\}$ 和全量测试集 $D$，第一步用逐样本的分歧分数对样本排序、取 top-k 得到压缩子集 $D_{\text{DISCO}}$；第二步把任意模型在该子集上的输出拼成"模型签名"，经 PCA 降维后用回归器映射到全量性能。训练阶段在源模型上拟合预测器，测试阶段对未见过的**目标模型**只需在 100 个锚点上推理一次即可估出其全量表现。

```mermaid
flowchart LR
    A[全量测试集 D] --> B[源模型 F 在 D 上推理]
    B --> C[逐样本算分歧分数 PDS/JSD]
    C --> D[排序取 top-k<br/>得压缩子集 D_DISCO]
    D --> E[任意模型在 D_DISCO 上输出]
    E --> F[拼接成模型签名]
    F --> G[PCA 降维]
    G --> H[回归器 RF/kNN]
    H --> I[估计全量性能]
```

### 关键设计

**1. 信息论最优的分歧选样：把"该测哪些样本"还原成一道互信息最大化题。** DISCO 的出发点是 Proposition 1：设模型下标 $m$ 在源模型集合上均匀采样，$\bar{y}_i$ 为集成均值预测对应的类别随机变量，则"模型身份诱发的性能函数 $S(m)$"与"样本 $i$ 上的预测 $\hat{y}_i$"之间的互信息恰好等于该样本上各模型预测分布的广义 Jensen-Shannon 散度：

$$\mathrm{MI}_{m,\hat y_i}\big(S(m);\hat y_i\big)=H(\hat y_i)-\mathbb{E}_m\big[H(\hat y_i^m)\big]=\mathrm{JSD}\big(\hat y_i^1,\dots,\hat y_i^M\big).$$

直观说，一个样本携带的"用于区分模型"的信息量，正比于各模型在它上面的预测分布有多分散。因此最优的贪心准则就是按 JSD 从大到小取样本——这给"选模型最不一致的样本"提供了严格依据，而不是靠聚类去近似"代表性"。

**2. PDS：JSD 的可解释替身，把分歧落到可计算的连续打分上。** 直接算多分布 JSD 较繁，作者改用预测多样性分数 PDS(源自 OOD 检测)，它是"$M$ 个源模型 argmax 类别去重个数"的连续推广：

$$\mathrm{PDS}\big(\hat y_i^1,\dots,\hat y_i^M\big):=\frac1C\sum_c \max_m f_c^m(x_i).$$

PDS 越大表示越多模型把概率质量压在不同类别上、分歧越强。Proposition 2 进一步给出 PDS 与 JSD 的双向夹逼不等式 $\frac{2}{M^2\ln 2}(\mathrm{PDS}_i-1)^2\le \mathrm{JSD}_i\le \frac{M}{M-1}\log M\cdot(\mathrm{PDS}_i-1)$，说明用 PDS 排序与用 JSD 排序高度一致。实验里两者都试，PDS 配随机森林往往最稳。

**3. 模型签名 + 简单回归：用原始输出而非标量准确率做外推。** 以往预测端只用锚点上的(加权/校正)准确率这种标量摘要，信息被压扁。DISCO 把目标模型在子集上的原始输出直接拼接成高维"模型签名" $f(D_{\text{DISCO}})=[f(x_1),\dots,f(x_L)]$，保留了远比标量丰富的判别信号。由于维度可能很高(类别数 × 样本数)，先用 PCA 把签名压到如 256 维 $Q\circ f(D_{\text{DISCO}})$ 抑制过拟合，再走两条朴素预测路线：**kNN**——在源模型签名里找最近的 K 个、平均其性能；**参数化回归**——训练一个映射 $R$(线性/随机森林/神经网)让 $R\circ Q\circ f^m(D_{\text{DISCO}})$ 逼近真值。作者刻意选最简单的部件，正是要论证"简单即最优"。

**4. 时间切分(chronological split)：用更贴近现实的方式检验泛化。** 元模型方法历来被诟病依赖已有模型分布、对未来新模型失效。DISCO 不用按性能高低制造的人为压力测试，而是按发布时间切分——源模型取 2024-01-13 之前发布的，测试集取之后的(9:1)，模拟"用老模型训练预测器、去估新模型"的真实场景。实验显示这种更现实的切分下排序相关性 .987，与均匀切分的 .986 几乎一致，说明方法稳健。

## 实验关键数据

### 主实验表格（语言域，每个数据集压到 100 个样本，MAE 越低/Rank 越高越好）

| 方法 | 选样 | 预测 | MMLU MAE/Rank | HS MAE/Rank | WG MAE/Rank | ARC MAE/Rank |
|------|------|------|------|------|------|------|
| Baseline | Random | Direct | 3.45 / .916 | 2.85 / .839 | 3.60 / .827 | 2.61 / .898 |
| tinyBenchmarks | Random | gp-IRT | 2.79 / .922 | 1.96 / .819 | 1.64 / .928 | 2.22 / .921 |
| Anchor-corr | — | gp-IRT | 2.08 / .927 | 1.27 / .937 | 1.95 / .918 | 2.18 / .948 |
| Metabench† | Best for val. | ability-IRT | 2.08 / .904 | 0.80 / .974 | 1.23 / .947 | 1.14 / .971 |
| Model signature | Random | Sig.+RF | 1.81 / .933 | 1.36 / .938 | 1.29 / .926 | 1.72 / .938 |
| **DISCO** | High PDS | Sig.+kNN | 1.31 / .972 | 1.32 / .956 | 1.19 / .951 | 1.96 / .937 |
| **DISCO** | **High PDS** | **Sig.+RF** | **1.07 / .987** | **1.01 / .984** | **1.00 / .967** | 1.47 / .971 |
| DISCO | High JSD | Sig.+RF | 1.30 / .987 | 0.86 / .972 | 1.09 / .973 | 1.75 / .938 |

† Metabench 收敛需更多样本(MMLU/ARC 用 150，HS 用 450，WG 用 200)，不完全可比。

### 消融实验表格（MMLU，100 样本，Rank 相关性）

| 维度 | 设置 | Rank |
|------|------|------|
| (a) 模型切分 | 时间切分 / 均匀切分 | .987 / .986（稳健） |
| (b) 分层抽样 | 关 / 开 | .987 / .978（PDS 下分层无益） |
| (c) 源模型数 | 100 / 382 | .969 / .987（100 个就超 tinyBenchmarks 的 .927） |
| (d) 降维 | 不降(3100 维) / PCA-256 | .918 / .987 |
| (e) 预测器 | 随机森林最佳 | .987 |

### 关键发现
- **拆解贡献**：仅"模型签名 + RF"(随机选样)就已达 1.81%p / .933 的 SOTA 水平；再叠加 PDS 选样把 MMLU 推到 1.07%p / .987，证明两项创新各自有效且可叠加。
- **极端压缩**：样本数低到 10 个时，非参数的 kNN 反而比随机森林更稳，提示极端压缩下宜用非参数预测器。
- **跨域泛化**：迁到视觉域 ImageNet 验证集(400 个 timm 模型)，压到 100 点即降低 99.8% 推理成本，DISCO 取得 0.63%p / .969，全面超过 Lifelong Benchmark(.838/2.06)与 SSEPY(.762/3.05)。

## 亮点与洞察
- **把"挑样本"从工程直觉提升为信息论命题**：Proposition 1 用一行互信息等式说明"模型分歧 = 区分模型的信息量"，给了"选分歧最大样本"以理论正当性，而非又一个启发式聚类技巧。
- **逆共识设计**："样本要多样、要覆盖难度谱"是该领域的隐含共识，本文直接否定并用实验证明分层抽样在 PDS 下反而无益——视角转换很干净。
- **简单即最优的实证**：用最朴素的随机森林/kNN + PCA 就超过了带 IRT 隐变量的 Metabench，且只需 100 而非 150~450 个样本，工程上极易复现部署。

## 局限与展望
- **对模型分布漂移敏感**：预测器在源模型上训练，当出现全新架构/训练范式/目标的模型时，会引入训练时未见过的模式而退化——作者承认这是主要软肋，建议用自适应选样或周期性重训缓解。
- **仅适用于"预定义选项概率"的封闭式任务**：方法依赖每题在若干候选类别上的预测概率(Proposition 1 的 classes)，因此不适配翻译/摘要等开放式生成任务，需先人为定义正确/错误输出集合。
- **依赖源模型池规模**：虽然 100 个源模型已够强，但极少源模型或源模型同质化严重时，签名空间的判别力会下降。

## 相关工作与启发
- **锚点/高效评测谱系**：Anchor Points(Vivek 2023)、tinyBenchmarks(Polo 2024, IRT)、Metabench(Kipnis 2024)、动态锚点(Hofmann 2025)都属"选锚点 + 预测"框架，DISCO 在选样(分歧 vs 代表性)和预测(签名 vs 隐变量)两端都做了简化与超越。
- **PDS 的跨界复用**：分歧分数 PDS 原本是 OOD 检测工具(Rubinstein 2024)，这里被借来度量样本信息量，是"OOD 不确定性度量 → 评测样本选择"的一次有趣迁移，启发把集成分歧类信号用到更多"选数据"场景(主动学习、数据剪枝、课程构建)。
- **与 active testing 的区别**：主动测试把标注预算投给信息丰富样本，但需先在全集推理；DISCO 关心的是推理成本本身，因此选静态锚点、对目标模型只推理一次——两者互补。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — "分歧即信息量"的信息论刻画 + 逆共识的极简选样，思路清爽且有理论支撑，虽然部件(PDS、签名、RF)多为已有积木，但组合与论证角度新。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 4 个语言基准 + ImageNet 视觉域、424/400 个真实模型、完整因子分析与跨压缩率曲线，时间切分设计贴近现实；缺开放式生成任务验证。
- **写作质量**: ⭐⭐⭐⭐ — 动机—命题—方法—实验逻辑顺，两条命题给方法定调，表格清晰；公式较密但解释到位。
- **价值**: ⭐⭐⭐⭐ — 99% 成本削减且误差约 1%p，对训练中频繁监控、有限算力评测、部署后抽检都很实用，工程落地门槛低。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization](sparseeval_efficient_evaluation_of_large_language_models_by_sparse_optimization.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/llm_evaluation/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICLR 2026\] Fewer Battles, More Gain: An Information-Efficient Framework for Arena-based LLM Evaluation](fewer_battles_more_gain_an_information-efficient_framework_for_arena-based_llm_e.md)
- [\[ICLR 2026\] Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](dont_passk_a_bayesian_framework_for_large_language_model_evaluation.md)
- [\[ICLR 2026\] NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation](naipv2_debiased_pairwise_learning_for_efficient_paper_quality_estimation.md)

</div>

<!-- RELATED:END -->
