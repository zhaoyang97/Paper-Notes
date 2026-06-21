---
title: >-
  [论文解读] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms
description: >-
  [ICLR 2026][预训练][正无标签学习] 提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。 领域现状：PU 学习（仅有正样本和…
tags:
  - "ICLR 2026"
  - "预训练"
  - "正无标签学习"
  - "benchmark"
  - "模型选择"
  - "标签偏移"
  - "公平评估"
---

# Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms

**会议**: ICLR 2026  
**arXiv**: [2509.24228](https://arxiv.org/abs/2509.24228)  
**代码**: 有 (benchmark)  
**领域**: 弱监督学习  
**关键词**: 正无标签学习, benchmark, 模型选择, 标签偏移, 公平评估

## 一句话总结
提出首个 PU 学习统一基准，系统解决两个关键问题：(1) 用代理准确率和代理 AUC 实现无负样本的模型选择；(2) 发现并通过将正样本并入无标签集的简单校准方法解决单样本设置下的内部标签偏移问题，使双样本算法在单样本评估中得到公平比较。

## 研究背景与动机

**领域现状**：PU 学习（仅有正样本和无标签样本的二分类）近年算法众多，但实验设置高度不一致，难以判断哪个算法更好。

**现有痛点**：(a) 许多算法依赖包含负样本的验证集做模型选择，这与 PU 学习"无负样本"的前提矛盾；(b) PU 学习有单样本 (OS) 和双样本 (TS) 两种设置，现有评估偏向 OS 设置但忽略两者的关键差异——OS 设置下无标签数据的类别先验与边缘分布不同。

**核心矛盾**：TS 算法假设无标签数据来自边缘分布 $p(x)$，但 OS 设置下无标签数据的密度是 $\bar{p}(x) = \bar{\pi}p(x|y=+1) + (1-\bar{\pi})p(x|y=-1)$，其中 $\bar{\pi} \neq \pi$。这导致 TS 算法的风险一致性被破坏。

**本文目标** 提供可访问、现实、公平的 PU 学习评估框架。

**切入角度**：(a) 从信息论推导仅用正/无标签数据即可计算的代理指标；(b) 识别内部标签偏移问题并给出理论保证。

**核心 idea**：将正样本并入无标签集恢复边缘分布的无偏性，一句代码改动即可消除 OS 和 TS 设置间的公平性偏差。

## 方法详解

### 整体框架
这篇论文不提新算法，而是搭一套能公平评判已有 PU 算法的评估系统，针对三件被长期忽视的事逐一给出对策——这三件事的顺序也正是下面三个关键设计的顺序。第一件，模型选择偷偷用了负样本：现有论文做早停和超参选择时普遍依赖含负样本的验证集，违背了 PU 学习「无负样本」的前提，本文给出只用正/无标签数据就能算的代理指标，把模型选择拉回设定之内。第二件，单样本 (OS) 设置下藏着一个让跨设置比较失真的偏移：本文点出这个「内部标签偏移」并用一行代码校准，让原本只在双样本 (TS) 设置下成立的算法在 OS 下也能被公平地量出真实水平。第三件，各家实现细节不一致污染比较：本文把数据生成、训练、评估协议统一成一套标准基准，让最终读出的差距来自算法本身。

> 这是评估/基准型论文（两条独立的方法学贡献 + 一套统一协议，无串行数据流 pipeline），不适合画框架图，故跳过；三者一致性靠下文「关键设计」与整体框架同序同名来保证。

### 关键设计

**1. 代理准确率 (PA) 与代理 AUC (PAUC)：不用负样本也能做模型选择**

PU 学习的前提是手头根本没有干净的负样本，可现有论文做早停和超参选择时却普遍依赖含负样本的验证集（即所谓 oracle accuracy），这等于偷偷违背了自己的设定。本文从信息论出发推导出两个只用正/无标签验证数据就能算的代理指标。PA 建立在 $\mathbb{E}[\text{PA}(f)] = \text{ACC}(f) - 1 + 2\pi$ 这个等式上，把无法直接观测的真实准确率与可计算的代理量挂钩，代价是需要知道类别先验 $\pi$；PAUC 则把无标签数据当成"带噪声的负样本"来算 AUC，连 $\pi$ 都不需要。Proposition 1-2 证明这两个代理指标对模型排序保序——也就是说用它们选出来的模型，和用真值选出来的模型排序一致，从而让模型选择这一步真正回到"无负样本"的轨道上。

**2. 内部标签偏移 (ILS) 的识别与校准：一行代码恢复 TS 算法在 OS 下的一致性**

TS 算法的风险一致性建立在"无标签数据来自边缘分布 $p(x)$"这个假设上，但在 OS 设置里，正样本是从总体中抽走后再划出无标签集的，剩下无标签数据的正类比例已经从 $\pi$ 悄悄降到了 $\bar{\pi} = (1-c)\pi/(1-c\pi)$，密度也变成了 $\bar{p}(x)=\bar\pi\,p(x|y{=}+1)+(1-\bar\pi)\,p(x|y{=}-1)$。这个偏移本文称作内部标签偏移，是文献里第一次被点出来——此前大家都在 OS 设置下评测 TS 算法却从不校准，于是流传多年的比较其实都不公平。校准的手段反而极其朴素：算无标签损失时把正样本也并进无标签集，即用 $\mathcal{D}_k^U \cup \mathcal{D}_k^P$ 代替 $\mathcal{D}_k^U$，正好把被抽走的正类质量补回来，恢复边缘分布的无偏性。Theorem 1 证明这样得到的风险估计器无偏，Theorem 2 给出收敛保证，而代码层面只是一行改动、几乎零开销。

**3. 统一基准框架：抹掉实现细节带来的噪声**

就算指标和校准都对了，各家在数据增强、warm-up、训练协议上的差异仍会污染比较。本框架把数据生成、训练和评估协议统一起来，覆盖代价敏感、样本选择、有偏 PU 学习三大算法族（共 17 个代表算法），在 CIFAR-10、ImageNette 两个图像数据集与 USPS、Letter 两个 UCI 表格数据集上跑同一套流程，让最终读出的差距来自算法本身而非工程细节。

### 损失函数 / 训练策略
统一使用 logistic 损失。校准只改无标签损失的计算方式（把正样本并入无标签集），不触碰各算法的核心，因此可以原样套到任意 TS 算法上。

## 实验关键数据

### 主实验

**CIFAR-10 PU 版本 (不同正样本量):**

| TS算法 | OS设置(未校准) | OS设置(校准后) | TS设置 |
|--------|-------------|------------|-------|
| uPU | 显著下降 | ≈TS性能 | 基线 |
| nnPU | 下降 | 恢复 | 基线 |
| Dist-PU | 下降 | 恢复 | 基线 |
| VPU | 下降 | 恢复 | 基线 |

### 消融实验

**模型选择标准对比:**

| 选择标准 | 是否需要负样本 | ACC 相关性 | AUC 相关性 |
|---------|------------|----------|----------|
| Oracle Accuracy (OA) | ✓ | 最佳 | 最佳 |
| Proxy Accuracy (PA) | ✗ | 接近OA | 中等 |
| Proxy AUC (PAUC) | ✗ | 中等 | 接近OA |

### 关键发现
- 没有单一算法在所有数据集和指标上全面领先；一些早期简单方法已经很强
- TS 算法在 OS 设置下不校准时性能显著退化，校准后恢复——验证了 ILS 问题的实际影响
- PA 适合做准确率相关的模型选择，PAUC 适合做 AUC 相关的选择——不同测试指标需用不同选择标准
- 校准方法在各种设置和算法上一致有效，计算开销几乎为零

## 亮点与洞察
- **一行代码的影响**：校准方法本质上就是把正样本也放到无标签损失里计算。如此简单的改动却有理论保证且效果显著——体现了"理解问题比设计方法更重要"
- **ILS 的首次识别**：之前所有论文都在 OS 设置下评估 TS 算法但不做校准，导致流传多年的不公平比较。这个发现可能改变该领域的评估标准
- **代理指标的实用价值**：证明了不需要负样本就能做有效的模型选择，使 PU 学习的评估真正自洽

## 局限与展望
- 代理准确率需要知道或估计类别先验 $\pi$，估计不准确时可能影响模型选择
- 基准目前仅覆盖图像分类数据集，文本/表格等模态有待扩展
- 校准方法假设正样本的标注概率 $c$ 是常数（SCAR 假设），非均匀标注场景未覆盖
- 深度学习场景中的超参数搜索可能使代理指标的理论保证在有限样本下不够紧

## 相关工作与启发
- **vs 现有 PU 学习论文**: 几乎所有论文都用含负样本的验证集做早停/选择，本文指出这是不现实的并提供了替代方案
- **vs 弱监督学习基准**: 类似 CleanLab 对噪声标签学习的贡献，本文为 PU 学习提供了急需的标准化基准
- **vs du Plessis et al. 2015**: uPU 的风险一致性在 OS 设置下被破坏，校准方法恢复了它——对经典方法的重要补充

## 评分
- 新颖性: ⭐⭐⭐⭐ ILS 问题的发现和校准方法简洁优雅，代理指标有理论保证
- 实验充分度: ⭐⭐⭐⭐⭐ 6种算法族×多数据集×OS/TS两种设置×校准前后对比
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，理论推导严谨，benchmark 贡献实质性大
- 价值: ⭐⭐⭐⭐⭐ 作为首个 PU 学习基准，可能改变整个领域的评估实践

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](../../AAAI2026/llm_pretraining/rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICLR 2026\] DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks](duet_optimizing_llm_training_data_mixtures_via_noisy_feedback_from_unseen_downst.md)
- [\[ICLR 2026\] Learning Facts at Scale with Active Reading](learning_facts_at_scale_with_active_reading.md)
- [\[ACL 2025\] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)

</div>

<!-- RELATED:END -->
