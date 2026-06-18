---
title: >-
  [论文解读] Explaining CLIP Zero-shot Predictions Through Concepts
description: >-
  [CVPR 2026][信息检索/RAG][CLIP] 本文提出 EZPC，通过学习一个线性投影矩阵将 CLIP 的图像-文本嵌入映射到可解释的概念空间，在几乎不损失零样本分类精度的前提下（CIFAR-100/CUB/ImageNet-100 上 H-mean 仅差约 1%），为 CLIP 的预测提供基于人类可理解概念的忠实解释，且推理开销仅增加约 0.1ms。
tags:
  - "CVPR 2026"
  - "信息检索/RAG"
  - "CLIP"
  - "零样本分类"
  - "概念瓶颈模型"
  - "可解释性"
  - "视觉语言模型"
---

# Explaining CLIP Zero-shot Predictions Through Concepts

**会议**: CVPR 2026  
**arXiv**: [2603.28211](https://arxiv.org/abs/2603.28211)  
**代码**: [https://github.com/oonat/ezpc](https://github.com/oonat/ezpc)  
**领域**: 信息检索  
**关键词**: CLIP, 零样本分类, 概念瓶颈模型, 可解释性, 视觉语言模型

## 一句话总结
本文提出 EZPC，通过学习一个线性投影矩阵将 CLIP 的图像-文本嵌入映射到可解释的概念空间，在几乎不损失零样本分类精度的前提下（CIFAR-100/CUB/ImageNet-100 上 H-mean 仅差约 1%），为 CLIP 的预测提供基于人类可理解概念的忠实解释，且推理开销仅增加约 0.1ms。

## 研究背景与动机

1. **领域现状**：CLIP 等视觉语言模型(VLM)在零样本图像识别上取得了巨大成功，通过将图像和文本对齐到共享语义空间，无需任务特定训练就能识别任意类别。与此同时，概念瓶颈模型（CBM）通过人类定义的概念中间层提供可解释的推理，但依赖概念标注且无法泛化到未见类别。

2. **现有痛点**：CLIP 的高维嵌入是纠缠的黑箱——用户无法理解为什么模型将某张图像与某个标签关联。CBM 虽然可解释但需要概念监督且局限于封闭世界（固定类别集合）。SpLiCE 将 CLIP 嵌入分解为概念组合但需要逐图优化（比CLIP慢59倍），Z-CBM 需要大型概念库和昂贵回归。

3. **核心矛盾**：可解释性和开放世界泛化能力似乎不可兼得——CBM 有可解释性但无法泛化，CLIP 能泛化但不可解释。

4. **本文目标** 如何在保持 CLIP 零样本能力的同时，让其预测通过人类可理解的概念来解释？

5. **切入角度**：CLIP 的内部表示可能已经隐式编码了人类可理解的语义结构，只需要一个恰当的投影就能将其"解码"出来。

6. **核心 idea**：学习单个线性投影矩阵 $A$ 将 CLIP 的图像-文本嵌入共同映射到预定义的概念空间，同时用匹配损失保持可解释性、用重建损失保持语义忠实性。

## 方法详解

### 整体框架
EZPC 想做的事很直接：让 CLIP 的每一次零样本判断都能被拆成"这张图激活了哪些人类能读懂的概念"，而又不动 CLIP 本身、不额外拖慢推理。它先准备一组 $m$ 个文字描述的概念（如 "has feathers"、"made of metal"），然后只训练一个线性投影矩阵 $A \in \mathbb{R}^{d \times m}$，把 CLIP 的 $d$ 维图像嵌入 $v_x$ 和文本嵌入都送进这个 $m$ 维概念空间。图像变成概念激活向量 $c_x = v_x A$，每个类别也变成概念向量 $c_k$，分类就是在概念空间里取最近的类别 $\hat{y} = \arg\max_k \langle c_x, c_k \rangle$。由于这是点积，类别得分天然等于各概念贡献之和 $\langle c_x, c_k \rangle = \sum_{j=1}^{m} s_{x,k}^{(j)}$，其中 $s_{x,k} = c_x \odot c_k$，于是"为什么判成这类"直接读元素乘积里哪几维最大就行。整个训练只优化 $A$ 这一个矩阵，靠两个损失分别拉住可解释性和忠实性。

### 关键设计

**1. 共享线性概念投影：用一个矩阵把图文一起搬进概念空间**

CLIP 的高维嵌入是纠缠的黑箱，无法告诉用户某次匹配究竟基于什么语义。EZPC 不去逐图求解（那是 SpLiCE、Z-CBM 慢几十倍的根源），而是学一个全局共享的 $A$，让图像和所有类别文本都经过同一投影：$c_x = v_x A$、$C_\mathcal{Y} = T A$。之所以坚持用**线性**投影而非更强的非线性映射，是因为线性让"解释"和"决策"严格是同一回事——类别 logit $\langle c_x, c_k\rangle$ 就是把各概念得分 $s_{x,k}^{(j)}$ 加起来，每一维的贡献可以原样拆出来，不存在事后归因的近似偏差。这种 faithfulness-by-construction 比 saliency map 之类的事后解释更可信，而代价只是一次矩阵乘法，推理几乎免费。

**2. 匹配损失：把投影列锚在真实概念方向上，防止训歪**

投影矩阵如果放任优化，列向量很可能漂到一些数值上分类很好、却不再对应任何人类概念的方向，解释就失效了。EZPC 先用 CLIP 文本编码器把全部概念短语编码成 $\Phi \in \mathbb{R}^{d \times m}$，用它初始化 $A = \Phi$，再在训练中加一个 MSE 约束把 $A$ 拽回概念基底附近：

$$\mathcal{L}_{\text{match}} = \frac{1}{dm}\sum_{i,j}(A_{ij} - \Phi_{ij})^2$$

这相当于给每个概念方向一个软锚点——允许 $A$ 微调以适配下游分类，但不许它跑远到"概念 $j$ 这一列其实已经不再表示概念 $j$"的地步，从而在灵活性和可解释性之间留住平衡。

**3. 重建损失：保证概念空间里的判断和原始 CLIP 一致**

光有可解释性还不够，如果投影后模型的分类倾向和原版 CLIP 不一样，那解释的就不是 CLIP 而是另一个模型了。EZPC 用 KL 散度逼着概念空间的类别分布去对齐 CLIP 原始嵌入空间的类别分布：

$$\mathcal{L}_{\text{recon}} = \frac{1}{B}\sum_{i=1}^{B} \text{KL}\big(\text{softmax}(c_i C_\mathcal{Y}^\top) \,\|\, \text{softmax}(v_i T^\top)\big)$$

左边是投影后在概念空间算的相似度分布，右边是 CLIP 原始 $v_i T^\top$ 的分布。把前者拉向后者，等于要求"加了可解释层之后，模型对每张图的判断排序基本不变"，这正是 EZPC 能把精度损失压到 1% 以内、不牺牲 CLIP 零样本能力的关键。

### 损失函数 / 训练策略
总损失把两项合起来，$\lambda$ 调可解释性与忠实性的权重：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{match}} + \lambda \mathcal{L}_{\text{recon}}$，多数数据集取 $\lambda = 1$，CUB 和 Places365 用 $\lambda = 5$。概念集复用 LF-CBM 里 GPT-3 生成的概念词表，并并入 ImageNet-1k 的大型概念池以扩大覆盖；所有实验按 80/20 划分 seen/unseen 类别来检验开放世界泛化。

## 实验关键数据

### 主实验

**广义零样本性能 (Harmonic Mean)**:

| 数据集 | CLIP | Z-CBM | SpLiCE | EZPC |
|--------|------|-------|--------|------|
| CIFAR-100 | 0.408 | 0.365 | 0.270 | **0.403** |
| ImageNet-100 | 0.693 | 0.585 | 0.389 | **0.682** |
| CUB | 0.474 | 0.189 | 0.070 | **0.465** |
| ImageNet-1k | 0.530 | 0.462 | 0.300 | 0.481 |
| Places365 | 0.362 | 0.357 | 0.282 | 0.352 |

**推理效率对比**:

| 方法 | 延迟 (ms/img) | 开销倍数 |
|------|--------------|----------|
| CLIP | 5.77 | 1.0× |
| Z-CBM | 542.34 | 94.0× |
| SpLiCE | 338.51 | 58.7× |
| EZPC | 5.90 | ~1.0× |

### 消融实验

| $\lambda$ | Zero-shot Seen | Unseen | GZS H-mean |
|-----------|----------------|--------|------------|
| 0.01 | 0.377 | 0.508 | 0.358 |
| 0.1 | 0.654 | 0.820 | 0.630 |
| 1 | 0.699 | 0.851 | 0.682 |
| 10 | 0.707 | 0.859 | 0.695 |
| 100 | 0.706 | 0.857 | 0.692 |

### 关键发现
- **EZPC 在大多数数据集上与 CLIP 性能差距在 1% 以内**（CIFAR-100: -0.5%, ImageNet-100: -1.1%, CUB: -0.9%），而 SpLiCE 和 Z-CBM 常差 10-15%
- **$\lambda$ 存在量化-质化权衡**：大 $\lambda$ 提升量化指标（更好保持 CLIP 分布），但定性分析显示小 $\lambda$（如1）产生更语义相关的概念激活
- **概念空间具有良好的空间对齐性**：在 CUB 的 Indigo Bunting 类上，正向概念"蓝灰色身体"的 Pointing Accuracy 达 96.7%，负向概念"红色面部"几乎为 0
- **跨数据集迁移有效**：在 ImageNet-100 上训练的投影矩阵可以直接迁移到 CIFAR-100 和 CUB，性能接近 CLIP

## 亮点与洞察
- **极简的可解释性方案**：仅需一个线性投影矩阵就实现了概念级解释，推理开销几乎为零（0.1ms），这使其适合大规模部署。与需要逐图优化的 SpLiCE（慢 59 倍）和 Z-CBM（慢 94 倍）形成鲜明对比。
- **解释的忠实性保证**：由于概念得分直接构成预测 logit（$\langle c_x, c_k \rangle = \sum_j s_{x,k}^{(j)}$），解释不是事后归因而是构造性忠实（faithfulness by construction）。这一点比 saliency map 等事后解释方法更可信。
- **匹配-重建双目标的平衡设计**：$\mathcal{L}_{\text{match}}$ 保可解释性，$\mathcal{L}_{\text{recon}}$ 保性能，$\lambda$ 控制两者平衡——这种设计模式可以迁移到其他需要在可解释性和性能间权衡的任务。

## 局限与展望
- **线性投影假设限制表达能力**：高度非线性的语义关系可能无法在概念空间中完全捕获
- **概念集质量依赖**：可解释性取决于概念词汇表的质量和多样性，概念集的偏差会影响解释的忠实度
- **仅限分类任务**：当前方法聚焦于分类，扩展到多模态推理、VQA 等任务是开放问题
- **ImageNet-1k 上性能差距较大**（5%），大规模设置下概念分解的信息损失更明显
- **改进方向**：非线性概念映射、自适应概念发现、与 LLM 集成动态扩展概念词汇

## 相关工作与启发
- **vs SpLiCE**: SpLiCE 将 CLIP 嵌入稀疏分解为概念向量组合，但需要逐图优化，慢 59 倍；EZPC 学习统一投影，推理近乎免费
- **vs Z-CBM**: Z-CBM 从概念库重建嵌入进行零样本 CBM，但需要大型概念库和昂贵回归（慢 94 倍）；EZPC 显著更高效且性能更好
- **vs LF-CBM**: LF-CBM 需要概念标注训练且是闭集方法；EZPC 利用 LF-CBM 的概念集但实现开放世界零样本
- CLIP 内部表示天然编码人类可对齐的结构这一发现，对理解大规模预训练模型的语义组织方式有理论价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 线性投影+双损失的方案简洁优雅，但技术深度有限
- 实验充分度: ⭐⭐⭐⭐ 5个数据集+多种定性分析+跨域实验+效率对比，较全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，数学符号规范，实验展示直观
- 价值: ⭐⭐⭐⭐ 为 VLM 可解释性提供了实用且高效的方案，有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EZSR: Event-based Zero-Shot Recognition](../../CVPR2025/information_retrieval/ezsr_event-based_zero-shot_recognition.md)
- [\[ICML 2026\] BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](../../ICML2026/information_retrieval/blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)
- [\[ICML 2026\] How can embedding models bind concepts?](../../ICML2026/information_retrieval/how_can_embedding_models_bind_concepts.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[ICCV 2025\] External Knowledge Injection for CLIP-Based Class-Incremental Learning](../../ICCV2025/information_retrieval/external_knowledge_injection_for_clip-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
