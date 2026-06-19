---
title: >-
  [论文解读] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks
description: >-
  [NeurIPS 2025][LLM安全][verbal confidence] 首次系统研究 LLM 语言化置信度（verbal confidence）在对抗攻击下的鲁棒性，提出基于扰动和越狱的攻击框架，揭示攻击可导致置信度下降最高 30%、答案翻转率高达 100%，且现有防御策略基本无效。 领域现状：随着 LLM 广泛部…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "verbal confidence"
  - "对抗攻击"
  - "LLM robustness"
  - "置信度校准"
  - "越狱攻击"
---

# On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks

**会议**: NeurIPS 2025  
**arXiv**: [2507.06489](https://arxiv.org/abs/2507.06489)  
**代码**: 无  
**领域**: AI安全  
**关键词**: verbal confidence, 对抗攻击, LLM robustness, 置信度校准, 越狱攻击

## 一句话总结
首次系统研究 LLM 语言化置信度（verbal confidence）在对抗攻击下的鲁棒性，提出基于扰动和越狱的攻击框架，揭示攻击可导致置信度下降最高 30%、答案翻转率高达 100%，且现有防御策略基本无效。

## 研究背景与动机

**领域现状**：随着 LLM 广泛部署，获取模型对自身预测的置信度估计变得至关重要。由于大多数 SOTA LLM 不开放 logit 访问，verbal confidence（让模型用自然语言输出一个数值置信度）成为最实用的置信度获取方式，已被工业界多个系统采用（如 TLM、云故障根因分析、客户对话评分等）。

**现有痛点**：已有大量工作研究 verbal confidence 的校准和机制，但几乎没有工作关注其在对抗攻击下的鲁棒性——即攻击者是否能通过微小输入修改来操纵模型输出的置信度分数。

**核心矛盾**：verbal confidence 的易获取性（任何黑盒用户都能拿到）恰好成为攻击者的利器：攻击者可以利用这些分数作为优化信号来制造对抗样本，不需要任何模型内部信息。这与传统需要 logit 的对抗攻击形成鲜明对比。

**本文要解决**：(i) verbal confidence 在各类对抗攻击下有多脆弱？ (ii) 如何有效地制造针对置信度的攻击？ (iii) 现有防御手段能否缓解这类攻击？

**切入角度**：作者观察到即使语义保持不变的微小扰动（如错别字、同义词替换）就能显著改变模型输出的置信度数值，这说明 LLM 的 verbal confidence 并不稳健。

**核心 idea**：设计专门以 verbal confidence 为优化目标的攻击框架（VCA），包括扰动式和越狱式两大类，全面评估 LLM 置信度表达的安全隐患。

## 方法详解

### 整体框架
输入为用户查询 $\mathbf{X}$、任务提示 $\mathcal{P}$；LLM 生成回答 $\mathcal{Y}$ 和置信度 $\mathcal{C}$。攻击目标是在保持语义不变的约束下，生成对抗输入使 $\mathcal{C}$ 最小化：$\min_{\hat{\mathbf{X}}} \text{CEM}(\text{LLM}(\hat{\mathbf{X}}, \mathcal{P}))$，s.t. $\text{Sim}(\mathbf{X}, \hat{\mathbf{X}}) > \tau$。攻击可针对三种威胁向量：用户查询、系统提示、one-shot 示例。

### 关键设计

1. **置信度引出方法（CEM）**:

    - 定义四种 CEM：Base（直接输出答案+置信度）、CoT（链式思维后输出）、Multi-Step（分步打分再汇总）、Self-Consistency（多次采样取平均）
    - 核心思路：CEM 是从输出 token 序列到数值置信度的映射 $\text{CEM}: \mathcal{Y} \to \mathcal{C}$
    - 设计动机：覆盖不同复杂度的置信度获取策略，全面评估攻击效果

2. **扰动式攻击（VCA-TF / VCA-TB / Typos / SSR）**:

    - VCA-TF 基于 TextFooler 改造，先按 token 对置信度的重要性排序，再对最重要的 token 做同义词替换
    - VCA-TB 基于 TextBugger，额外支持字符级修改（拼写错误、相似字符替换等）
    - Typos：随机引入常见拼写错误（键盘邻键、字符删除、相邻字符交换），概率 0.1
    - SSR（SubSwapRemove）：随机进行同义词替换、相邻 token 交换或删除
    - 关键改进：原始算法使用 logit 类别概率作为评分函数，本文改为使用 verbal confidence 数值

3. **越狱式攻击（ConfidenceTriggers / ConfidenceTriggers-AutoDAN）**:

    - ConfidenceTriggers：使用遗传算法优化一组 trigger token，附加到系统提示后，使后续所有查询的置信度降低
    - 完全黑盒：不需要模型权重或 tokenizer，仅需 verbal confidence 反馈
    - 初始种群从约 2000 个与"不确定性"相关的词汇中抽样，经过锦标赛选择、交叉、变异迭代优化
    - AutoDAN 变体：生成更自然的 trigger 文本，使用 GPT-4 进行句子级改写实现交叉/变异
    - 一旦优化完成，trigger 可无限复用于任意查询

### 攻击向量对比
系统提示和示例比用户查询更容易被攻击——前者更有效地降低置信度，后者更容易导致答案翻转。这意味着 prompt 注入攻击的威胁尤为突出。

## 实验关键数据

### 主实验：扰动式攻击效果

| 攻击方法 | 模型 | 平均 Δ 置信度 | 受影响样本% | 受影响样本 Δ Cf. | 正确→翻转% | 错误→翻转% |
|---------|------|-------------|-----------|----------------|----------|----------|
| VCA-TF | Llama-3-8B | 5.3% | 30.4% | 最高 70% | 最高 68.8% | 最高 100% |
| SSR | Llama-3-8B | 6.9% | 41.7% | 最高 60.8% | 最高 100% | 最高 100% |
| VCA-TB | GPT-3.5 | 6.9% | 35.8% | 最高 29.3% | 最高 38.3% | 最高 83.1% |
| Typos | GPT-3.5 | 6.5% | 40.5% | 最高 35.8% | 最高 62.2% | 最高 83.1% |

### 越狱攻击效果（ConfidenceTriggers，Llama-3-8B）

| CEM | 数据集 | Δ 置信度 | 受影响% | 正确→翻转% |
|-----|-------|---------|--------|----------|
| CoT | TQA | 29.5% | 66.0% | 100% |
| MS | TQA | 24.9% | 97.5% | 100% |
| Base | MQA | 8.7% | 33.5% | 28.4% |
| CoT | SQA | 24.7% | 72.5% | 22.7% |

### 防御效果（ConfidenceTriggers-AutoDAN, Llama-3-8B）

| 防御方法 | 对原始样本的影响 Δ Cf. | 对对抗样本的减轻 Δ Adv. | 结论 |
|---------|---------------------|---------------------|-----|
| Paraphrase | 高达 -24.8%（严重干扰正常样本） | 6.4%~53.2% | 防御本身破坏正常行为 |
| SmoothLLM | 高达 -14.4% | 0%~15.8% | 类似，副作用严重 |
| 困惑度过滤 | 对抗样本困惑度(350)远低于真实世界文本(457) | - | 无法有效区分 |

### 关键发现
- Multi-Step CEM 因初始置信度较低，最容易被攻击（Δ Cf. 最大）
- 较大模型（GPT-3.5 / GPT-4o / Llama-3-70B）不见得更鲁棒，VCA-TB 在 GPT-3.5 上平均影响 40.5% 样本
- 置信度的降低与答案翻转并不完全绑定——置信度可在答案不变的情况下大幅下降
- 攻击产生的置信度异常与 token logit 概率的错位加剧，损害模型诚实性

## 亮点与洞察
- **首次系统研究 verbal confidence 的对抗鲁棒性**：填补了 LLM 安全领域的重要空白，之前所有对抗攻击工作都聚焦于准确率或 logit 置信度
- **ConfidenceTriggers 的通用性**：只需优化一次 trigger，即可对任意后续查询降低置信度，类似"一次注入，永久生效"的 prompt 后门
- **防御困境的深刻揭示**：输入扰动防御（paraphrase、SmoothLLM）对正常样本的副作用太大，困惑度过滤无法区分对抗样本和真实世界噪声文本，LLM-Guard 过滤率也很低
- **置信度稳定性的反直觉发现**：即使删除大部分 token，置信度变化通常 <15%，说明有效攻击需要精确优化而非简单破坏输入

## 局限与展望
- 由于攻击成本高，大模型（GPT-4o、70B）实验规模有限，仅用子集验证
- 主要聚焦字符/词级攻击，句子级攻击尚未探索
- 只研究了置信度降低方向，提升置信度的攻击（导致过度自信）也值得关注
- 测试数据集生态效度偏低（MCQA 格式），开放式生成场景未覆盖
- 未提出有效的防御方案——论文主要是攻击+现有防御失败的分析，缺少建设性解决方案

## 相关工作与启发
- **vs 传统 NLP 对抗攻击（TextFooler/TextBugger）**：本文将这些方法的评分函数从 logit 概率改为 verbal confidence，实现了黑盒场景下的攻击迁移
- **vs logit-based 置信度攻击（galil2021, obadinma2024）**：它们需要模型内部信息，本文方法仅需文本输出，适用范围更广
- **vs GCG/AutoDAN 越狱**：本文将越狱目标从绕过安全护栏改为降低置信度，是越狱框架的新应用方向
- 对 AI 安全部署有直接启示：任何依赖 verbal confidence 做决策的系统都需要额外的鲁棒性保障

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将对抗攻击系统性地应用于 verbal confidence，问题定义清晰且有实际意义
- 实验充分度: ⭐⭐⭐⭐⭐ 4 种扰动攻击 + 2 种越狱攻击 × 4 种 CEM × 3 数据集 × 多模型，极其全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式化定义完整，但表格数据量巨大读起来有些累
- 价值: ⭐⭐⭐⭐ 揭示了一个重要的安全盲区，但缺乏有效防御方案略显遗憾

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](../../ACL2025/llm_safety/tip_iceberg_adversarial_attacks.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)
- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)
- [\[ACL 2025\] CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](../../ACL2025/llm_safety/cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
