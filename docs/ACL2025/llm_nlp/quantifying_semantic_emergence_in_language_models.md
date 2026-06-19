---
title: >-
  [论文解读] Quantifying Semantic Emergence in Language Models
description: >-
  [ACL 2025][LLM 其他][信息涌现] 提出了 Information Emergence (IE) 这一基于信息论的定量指标，通过比较 Transformer 各层中宏观（序列级）与微观（token级）的互信息差异，量化 LLM 从 token 中提取语义的能力。 核心矛盾 核心矛盾：领域现状：大语言模型（LLM…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "信息涌现"
  - "语义理解"
  - "互信息"
  - "大语言模型"
  - "可解释性"
---

# Quantifying Semantic Emergence in Language Models

**会议**: ACL 2025  
**arXiv**: [2405.12617](https://arxiv.org/abs/2405.12617)  
**代码**: [github](https://github.com/Zodiark-ch/Emergence-of-LLMs)  
**领域**: LLM/NLP  
**关键词**: 信息涌现, 语义理解, 互信息, 大语言模型, 可解释性

## 一句话总结

提出了 Information Emergence (IE) 这一基于信息论的定量指标，通过比较 Transformer 各层中宏观（序列级）与微观（token级）的互信息差异，量化 LLM 从 token 中提取语义的能力。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：大语言模型（LLMs）被广泛认为具有出色的语义理解能力，但目前缺乏一种**定量且任务无关**的度量指标来衡量这种能力。

现有评估方法的局限性：

**依赖特定任务**：如指令遵循、搜索、推理等任务的准确率只能间接反映语义理解能力，且数据集构建耗时

**粒度过粗**：现有评估通常关注文本层面表现，无法对更细粒度的 token 行为提供解释

**指标不统一**：不同任务使用不同评测指标，可能导致相互矛盾的结论

因此，作者提出了一种**闭式（closed-form）、任务无关**的指标——信息涌现（IE），用于确定性地量化 LLM 从 token 中提取有意义语义的能力。

## 方法详解

### 整体框架

核心思想源自信息论中的「涌现」概念：语义是 token 集合在宏观层面呈现出的有意义组织，它在微观（单个 token）层面不可观测，但在宏观（整个序列）层面可观测。作者将 Transformer 块间的 token 表示传递类比为马尔可夫过程，并通过互信息来量化宏观与微观层面的熵减差异。

### 关键设计

1. **马尔可夫过程类比**：将 NTP（下一个 token 预测）机制视为马尔可夫随机过程。对于 Transformer 的第 $l$ 层，token $t$ 的输出表示 $h_{l+1}^t$ 依赖于第 $l$ 层中位置 $\leq t$ 的所有输入表示。微观变量（如 $h^0$）仅依赖自身，宏观变量（如 $h^{T-1}$）聚合了所有前置 token 的信息。

2. **信息涌现（IE）定义**：对于第 $l$ 个 Transformer 块，IE 定义为宏观互信息与微观互信息均值之差：
    $E(l) = MI(h_{l+1}^{ma}, h_l^{ma}) - \frac{1}{T}\sum_{t=0}^{T-1} MI(h_{l+1}^{mi\_t}, h_l^{mi\_t})$
   $E(l) > 0$ 表示该层在整个序列上的不确定性降低（熵减）大于单个 token 的熵减，意味着模型成功捕获了集体语义。

3. **微观变量的计算**：为确保微观变量仅依赖自身，每个 token 被单独作为输入序列送入模型，避免自回归机制引入的上下文影响。宏观变量则取完整序列中最后一个 token 的表示。

### 损失函数 / 训练策略

使用轻量级互信息估计器（10层线性 + LeakyReLU 网络）来近似高维连续空间中的 KL 散度。基于 MINE 方法（Belghazi et al.），通过优化一个对比式的误差函数来估计互信息：
- 正样本：同一序列中相邻层的表示对 $(h_{l+1,s}^{ma}, h_{l,s}^{ma})$
- 负样本：不同序列的表示对 $(h_{l+1,s}^{ma}, h_{l,s'}^{ma})$
- 批量大小设为 300,000，学习率从 1e-4 多项式衰减到 1e-8，迭代 10k 个 epoch

## 实验关键数据

### 主实验

**ICL 场景（合成数据集）**：

| 数据集 | 实体数 | 样本量 | token长度 | shots数 |
|--------|--------|--------|-----------|---------|
| Country | 25 | 303,600 | 8 | 4 |
| Animal | 16 | 524,160 | 10 | 5 |
| Color | 15 | 360,360 | 10 | 5 |

**自然句子场景**：从 OpenOrca 和 OpenHermes 各随机选取 300,000 条自然序列（每条 8 个 token）。

**模型范围**：GPT2-large (812M), GPT2-XL (1.6B), GEMMA (2.51B), OpenLlama (3B)。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| ICL vs 自然文本 | IE 增长模式差异 | ICL 中 IE 仅在新 demonstration 出现时增长；自然文本中逐 token 递增 |
| 模型大小增长 | IE 值上升 | 与模型参数量正相关，符合更大模型语义捕获能力更强的直觉 |
| ICL shots 饱和 | ~7th shot | 三种 ICL 类别在第 7 个 demonstration 趋于饱和 |
| 人类文本 vs LLM文本 | 人类 IE 更低 | LLM 生成文本的 IE 值显著高于人类文本（GPT-4: ~39.2 vs 人类: ~19.4） |

### 关键发现

1. **ICL 语义增强机制**：ICL 通过 demonstration 提升语义确定性，但最终在一定数量后饱和；每个 demonstration 内增加 token 长度不会改变这种"阶梯式上升"模式。

2. **IE 与幻觉的关联**：当 IE 停止增长且标准差达到峰值时，LLM 更容易生成错误回复（如重复错误）。这与现有关于幻觉的研究一致：LLM 难以在生成错误后自我纠正。

3. **人类与 LLM 文本的区分**：不同 LLM（GPT-4、Claude3、Llama3）生成的文本展现出不同的 IE 值和增长模式，甚至可以通过 IE 来区分文本来源——且无需计算目标 LLM 自身的 Transformer 表示。

## 亮点与洞察

- **理论贡献突出**：首次将信息论中的涌现概念系统性地应用于 LLM 语义理解的量化度量
- **实用价值**：提出的轻量级估计器不需要访问 LLM 内部，可通过小模型估计大模型/闭源模型的 IE 值
- **跨领域洞察**：IE 的发现为 ICL 机制、幻觉检测、AI 生成文本检测等多个方向提供了新的视角
- **与 Emergence 的关联**：在 $10^8$ 到 $10^{10}$ 参数范围内，IE 呈现与任务性能类似的急剧上升

## 局限与展望

1. **位置敏感性**：要求每个 token 位置有特定含义（如句子开头/结尾），直接应用于现有任务可能缺乏可解释性
2. **样本量要求巨大**：需要超过 30 万样本来保证高维连续表示的联合与边际分布估计准确性
3. **模型与文本长度受限**：受计算资源限制，未能在更大模型和长文本上进行验证
4. **因果关系未建立**：IE 与幻觉之间目前仅展示了相关性，未证明因果关系

## 相关工作与启发

- **信息论基础**：借鉴了 Rosas et al. 的信息涌现理论和 MINE（Belghazi et al.）的互信息估计方法
- **与 LLM 涌现的区别**：明确区分了信息涌现（宏微观差异可量化的现象）和 LLM 涌现（小模型没有但大模型有的能力）
- **启发方向**：IE 的 token 级分析可用于理解注意力机制、层间信息流动、以及不同架构的语义捕获差异

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出任务无关的语义理解定量指标，将信息论涌现概念引入 LLM 分析
- 实验充分度: ⭐⭐⭐⭐ 涵盖了 ICL 和自然句子两类场景及多种模型，但模型规模受限
- 写作质量: ⭐⭐⭐⭐ 从理论到实验逻辑清晰，数学推导严谨，但公式密度较高
- 价值: ⭐⭐⭐⭐ 为 LLM 可解释性、幻觉检测、AI 文本鉴别等提供了新工具和新视角

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ACL 2025\] Semantic Exploration with Adaptive Gating for Efficient Problem Solving with Language Models](semantic_exploration_adaptive_gating.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](argument_mining_in_the_age_of_large_language_models.md)
- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](../../AAAI2026/llm_nlp/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)

</div>

<!-- RELATED:END -->
