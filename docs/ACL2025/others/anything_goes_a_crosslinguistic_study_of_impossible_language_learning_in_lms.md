---
title: >-
  [论文解读] Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs
description: >-
  [ACL 2025][不可能语言] 在12种语言上训练GPT-2 small，系统性测试语言模型是否能区分可能语言(自然语言)与不可能语言(打乱词序等)，发现LM展现出部分类人的学习偏向但并非完美——能在单语言内区分但无法跨语言完全分离，而名词短语词序实验中泛化测试(而非困惑度)能反映类型学偏好。 领域现状：关于LLM能否作…
tags:
  - "ACL 2025"
  - "不可能语言"
  - "语言类型学"
  - "语言模型认知"
  - "GPT-2"
  - "Greenberg Universal 20"
---

# Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs

**会议**: ACL 2025  
**arXiv**: [2502.18795](https://arxiv.org/abs/2502.18795)  
**代码**: [GitHub](https://github.com/picol-georgetown/multilingual-LM)  
**领域**: 计算语言学 / 认知科学  
**关键词**: 不可能语言, 语言类型学, 语言模型认知, GPT-2, Greenberg Universal 20

## 一句话总结

在12种语言上训练GPT-2 small，系统性测试语言模型是否能区分可能语言(自然语言)与不可能语言(打乱词序等)，发现LM展现出部分类人的学习偏向但并非完美——能在单语言内区分但无法跨语言完全分离，而名词短语词序实验中泛化测试(而非困惑度)能反映类型学偏好。

## 研究背景与动机

**领域现状**：关于LLM能否作为人类语言习得的认知模型，学界争论激烈。支持者认为LLM的语言能力可以反映人类语言理论，反对者(如Chomsky等)认为LLM与人类认知机制根本不同，能学习任何任意输入，因此其成功对理解人类语言毫无意义。

**现有痛点**：Kallini等(2024)首次在英语上证明GPT-2能区分可能与不可能语言，但研究仅限于英语，结论能否跨语言泛化未知。此外，对类型学上未证实(但可能存在)的语言的学习行为也未探索。

**核心矛盾**："anything goes"假说（LLM对所有语言一视同仁）vs LM实际展现的偏好是否与人类一致。

**本文目标**：(1) LM是否能跨语言区分可能vs不可能语言？(2) LM是否展现类型学偏好，即是否更容易学习类型学上常见的词序？

**切入角度**：构建了两个平行语料库(OPUS12: 12种语言10M词，OPUS30: 30种语言0.7M词)，确保跨语言可比性，同时引入Greenberg Universal 20的NP词序测试。

## 方法详解

### 整体框架

三组实验：(1) 单语言内比较：每种自然语言vs其不可能变体；(2) 跨语言比较：所有自然语言vs所有不可能语言是否可分离；(3) 已证实vs未证实NP词序的学习差异。

### 关键设计

1. **平行语料库构建 (OPUS12/OPUS30)**:
    - 功能：从5个OPUS来源构建句子对齐的多语言平行语料库
    - 核心思路：OPUS12包含12种语言(4个语族)，英语部分约10M词(相当于儿童2-5岁的输入量)。OPUS30包含30种语言，用于测试集。平行语料确保不同语言的内容(信息量)一致，从而隔离语言形式特征对可学习性的影响
    - 设计动机：现有研究多用非平行语料，不同语言文本的信息量差异会混淆可学习性的比较

2. **不可能语言构造**:
    - 功能：对每种自然语言生成多种"不可能"变体——确定性打乱(3种seed)、局部窗口打乱(w=2,3,5,10)、完全逆序、奇偶重排
    - 核心思路：打乱操作是确定性的，即原始语言可通过逆变换恢复。如果LM只是通用模式匹配器(如反对者所声称)，应能同样好地学习这些变体
    - 设计动机：选择打乱操作因为(a)它们被Kallini等认定为"最不可能"的语言类型；(b)人类研究也表明人类对规则化有强烈偏好

3. **NP词序泛化测试 (ΔGenScore)**:
    - 功能：提出ΔGenScore指标，测试在不同NP词序上训练的模型的泛化能力
    - 核心思路：$\Delta\text{GenScore} = \text{GenScore}_{\checkmark} - \text{GenScore}_{\times}$，比较在自然语言上训练的模型与在非自然词序上训练的模型，哪个能更好地泛化到对方的测试数据。ΔGenScore > 0 表示自然语言模型有更好的泛化能力
    - 设计动机：困惑度在NP词序实验中无法区分自然vs非自然（因为规范化的NP实际上更规则、熵更低），但泛化测试可以揭示模型的内在偏好

### 损失函数 / 训练策略

每种语言独立训练GPT-2 small，使用各语言预训练的BPE tokenizer(约50k词表)。每种配置3个随机种子，最大1200训练步，120步warmup。评估使用几何均值困惑度(在10K句子平行测试集上)。

## 实验关键数据

### 主实验

实验1(单语言内)：12种语言中除意大利语外，所有自然语言的困惑度均低于其不可能变体。

| 语言 | 自然语言困惑度 | 最近不可能变体困惑度 | 差异显著？ |
|------|-------------|-------------------|----------|
| 英语 | ~15 | ~17 (shuffle_local w=2) | 是 |
| 中文 | ~8 | ~10 (shuffle_local w=2) | 是 |
| 阿拉伯语 | ~35 | ~37 (shuffle_local w=2) | 是 |
| 意大利语 | ~20 | ~19.5 (shuffle_local w=2) | 否(p=0.353) |

实验2(跨语言)：线性SVM分类器的macro F1 = 0.75，说明无法完全分离。

### 消融实验

实验3(NP词序)——ΔGenScore分析：

| NP词序变体 | 类型学状态 | 英语ΔGenScore | 中文ΔGenScore |
|-----------|-----------|--------------|--------------|
| Nnda | 未证实 | +正向(一致) | +正向(一致) |
| anNd | 未证实 | +正向(一致) | +正向(一致) |
| daNn | 少量证实 | +正向 | +正向 |
| dnaN(≈英语) | 大量证实 | +正向 | +正向 |
| dnNa(≈意大利语) | 大量证实 | 混合 | +正向 |

### 关键发现

1. **局部打乱(小窗口)比全局打乱更难区分**：shuffle_local(w=2)的困惑度最接近自然语言，某些语言(意大利语)甚至无法区分
2. **跨语言无法完全分离**：部分不可能语言(如英语shuffle_local w=3)的困惑度低于某些自然语言(如俄语、阿拉伯语)
3. **困惑度与TCW(每词token数)的相关性不显著**(ρ=0.564, p=0.076)，说明形态复杂度不是困惑度差异的主要原因
4. **ΔGenScore能区分类型学偏好**：未证实词序(Nnda, anNd)一致产生正向ΔGenScore，说明模型在自然词序上训练后泛化更好

## 亮点与洞察

- **温和立场的实证支持**：LM既不是"anything goes"的万能学习器，也不完全具备类人的学习偏向——它们在连续体中某处
- **困惑度vs泛化测试的分离现象**：困惑度作为衡量标准可能不够敏感（受文本熵影响），泛化测试(minimal pair)更能揭示内在偏好
- **constituency保持假说**：保持短语结构的打乱比破坏短语结构的打乱更容易学习，解释了NP内部打乱困惑度低于全局打乱的原因
- **多语言平行语料方法论**：通过构建内容一致的平行语料控制了信息量变量，是跨语言可学习性比较的方法论贡献

## 局限与展望

- 仅使用GPT-2 small，更大模型可能表现不同
- 训练数据仅10M词，相对较小
- NP词序实验仅覆盖4种有constituency parser的语言
- 平行语料可能包含噪声，未做人工检查
- 未探索count-based grammar等其他不可能语言类型

## 相关工作与启发

- **Kallini et al. (2024)**：本文的直接前身，在英语上首次证明LM能区分可能/不可能语言
- **Culbertson & Newport (2015)**：人类对harmonic NP词序(如dnaN)有学习偏好，本文发现LM也表现出类似但更弱的偏好
- 启发：LM的学习偏向可能来自自回归架构本身对local dependency的敏感性，而非对"语言"的深层理解

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将不可能语言研究扩展到12种语言，引入NP词序测试和ΔGenScore指标
- 实验充分度: ⭐⭐⭐⭐ 12种语言、多种打乱方式、多个分析维度，但受限于GPT-2 small
- 写作质量: ⭐⭐⭐⭐ 研究问题清晰，方法论严谨，讨论深入
- 价值: ⭐⭐⭐⭐ 对LLM作为认知模型的争论提供了重要实证，ΔGenScore方法有方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Little Human Data Goes A Long Way](a_little_human_data_goes_a_long_way.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)
- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)

</div>

<!-- RELATED:END -->
