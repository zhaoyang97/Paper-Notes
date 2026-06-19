---
title: >-
  [论文解读] Aligning to What? Limits to RLHF Based Alignment
description: >-
  [ACL 2025][LLM对齐][RLHF] 本文通过系统实验发现RLHF（包括DPO、ORPO、RLOO等方法）在减少LLM隐性种族偏见方面基本无效，且SFT在RLHF之前进行会"固化"模型偏见，揭示了当前对齐技术在处理模糊目标（如消除偏见）方面的根本局限。 领域现状：RLHF已成为大模型对齐的标准范式…
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "RLHF"
  - "隐性偏见"
  - "DPO"
  - "ORPO"
  - "方言偏见"
  - "对齐局限"
---

# Aligning to What? Limits to RLHF Based Alignment

**会议**: ACL 2025  
**arXiv**: [2503.09025](https://arxiv.org/abs/2503.09025)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: RLHF、隐性偏见、DPO、ORPO、方言偏见、对齐局限

## 一句话总结
本文通过系统实验发现RLHF（包括DPO、ORPO、RLOO等方法）在减少LLM隐性种族偏见方面基本无效，且SFT在RLHF之前进行会"固化"模型偏见，揭示了当前对齐技术在处理模糊目标（如消除偏见）方面的根本局限。

## 研究背景与动机

**领域现状**：RLHF已成为大模型对齐的标准范式，被用于使LLM符合"有用、无害、诚实"等人类偏好。主流方法包括在线RL方法（PPO、RLOO）和免奖励方法（DPO、ORPO）。然而，RLHF在处理偏见这类更微妙、更模糊的对齐目标方面的有效性尚未被系统研究。

**现有痛点**：Hofmann等人(2024)的研究发现，经过RLHF训练的现成LLM反而持有最强的隐性偏见（covert bias），但还没有研究系统地检验RLHF训练过程本身是否会加剧或减轻这些偏见。偏好数据集本身也存在质量问题——标注者之间对于什么构成"无害"并不总是一致。

**核心矛盾**：RLHF擅长优化明确、可度量的目标（如回答长度、格式合规），但"减少偏见"是一个模糊的、难以用偏好数据精确编码的目标。现有偏好数据主要关注回答质量而非公平性。

**本文目标**：系统分析多种RLHF技术（DPO、ORPO、RLOO）、不同底座模型（Llama 3、Mistral）、不同偏好数据集对LLM隐性和显性偏见的影响。

**切入角度**：采用社会语言学中的匹配假面测试（matched-guise probing），通过比较模型对非裔美国英语（AAE）和标准美国英语（SAE）文本的反应差异来量化隐性偏见。

**核心 idea**：用严格的偏见度量方法系统评估RLHF对模型内在态度的影响，证明当前对齐技术无法有效处理偏见消除这类模糊目标。

## 方法详解

### 整体框架
以Llama 3 8B为主要实验对象，分别使用DPO、ORPO和RLOO三种RLHF技术进行后训练，然后通过匹配假面探测法测量训练前后的隐性偏见（covert bias）和显性偏见（overt bias）变化。还在Mistral 7B上验证结论的泛化性，以及探索不同偏好数据集和训练轮次的影响。最后将偏见测量方法扩展到多模态模型（Llama 3.2 Vision 11B）。

### 关键设计

1. **匹配假面探测法（Matched-Guise Probing）**:

    - 功能：量化LLM对不同方言群体的隐性和显性态度
    - 核心思路：给定AAE和SAE两组文本，计算模型对各种人格特质（如"聪明""懒惰"等）的条件概率差异。关联分数 $q(t;\theta) = \frac{1}{|X|}\sum_i \log\frac{p(t|f(x_i);\theta)}{p(t|f(y_i);\theta)}$ 反映特质 $t$ 与AAE文本的关联程度。正值表示该特质更多与AAE关联，负值表示更多与SAE关联。使用Princeton trilogy研究中的人格特质和职业声望评分来评判关联是否构成偏见
    - 设计动机：比直接测试"你觉得XX群体怎样"更能揭示模型的"真实态度"，因为隐性测试绕过了模型的安全对齐

2. **多维度RLHF实验设计**:

    - 功能：系统评估各变量对偏见的影响
    - 核心思路：控制变量实验矩阵包括：(1) RLHF方法维度——DPO/ORPO/RLOO；(2) 底座模型维度——Llama 3/Mistral；(3) 数据集维度——Anthropic HH-RLHF/PKU-SafeRLHF/OLMo偏好数据；(4) 训练策略维度——有无SFT前置、训练1/3个epoch；(5) 方言暴露维度——使用AAE翻译后的偏好数据训练。所有模型使用LoRA (rank=16) 微调
    - 设计动机：排除单一变量的影响，全面理解RLHF与偏见的关系

3. **多模态偏见扩展（Multimodal Bias Extension）**:

    - 功能：将偏见测量从纯文本模型扩展到视觉语言模型
    - 核心思路：对Llama 3.2 Vision 11B，隐性偏见仍通过文本输入测量。显性偏见则通过将不同种族人物图片作为视觉输入（替代文本中的种族标识词）来测量，利用UTKFace数据集中的人脸图像。视觉输入提供了更丰富的种族信息，减少了仅靠少数文本标识词带来的高方差问题
    - 设计动机：拓宽偏见度量范围，同时VLM的图像输入可以提供更稳定的显性偏见度量

### 损失函数 / 训练策略
各RLHF方法使用标准实现：DPO使用标准偏好对损失，RLOO用NCSOFT OffsetBias-RM作为奖励模型（训练时排名在Huggingface奖励模型排行榜前10），ORPO使用Odds Ratio目标函数。SFT阶段在SlimOrca的10万样本上训练。

## 实验关键数据

### 主实验

| 模型配置 | 隐性特质偏见变化(均值) | 隐性职业偏见变化(均值) | 显性特质偏见变化(均值) | 说明 |
|---------|---------------------|---------------------|---------------------|------|
| Llama 3 (base) | - | - | - | 基线 |
| +DPO | 0.175 | -0.022 | -0.365 | 隐性偏见基本不变 |
| +ORPO | -0.026 | 0.151 | 0.076 | 显性偏见略有增加 |
| +RLOO | 0.003 | 0.135 | -0.177 | 变化不显著 |
| +SFT+DPO | 方差显著更低 | 方差显著更低 | 方差显著更低 | SFT固化了偏见 |
| Mistral+DPO | 0.044 | 0.097 | -0.116 | 变化比Llama更小 |

### 模型间奖励对比

| 模型配置 | ArmoRM奖励 | OffsetBias奖励 | 说明 |
|---------|-----------|---------------|------|
| Llama 3 base | 0.062 | -6.837 | 基线 |
| +DPO | 0.071 | -6.324 | DPO效果最好 |
| +ORPO | 0.062 | -7.004 | 几乎无提升 |
| +RLOO | 0.064 | -7.098 | 几乎无提升 |
| Llama 3 Instruct | 0.095 | -4.742 | 大规模后训练效果明显 |

### 关键发现
- **RLHF对隐性偏见基本无效**：所有RLHF方法（DPO、ORPO、RLOO）在所有实验设置下都未显著改变模型的隐性偏见模式。隐性偏见的抛物线型趋势（极端正面和负面特质都更多与AAE关联）在训练后保持不变
- **SFT固化偏见**：SFT在DPO之前进行会降低后续RLHF改变偏见的能力，Table 1显示L3+SFT配置下关联分数的变化方差显著比不做SFT时更低
- **不同底座模型有不同基线偏见**：Mistral的偏见比Llama 3更难被RLHF改变（变化方差更低），说明模型架构/预训练数据也影响偏见的可塑性
- **AAE偏好数据有微弱效果**：使用AAE翻译后的数据训练使关联分数略微向AAE方向偏移，但效果微弱
- **VLM的隐性和显性偏见可能相反**：Llama 3.2 Vision的隐性偏见将极端特质与AAE关联，但显性偏见（图像输入）却将相同特质与白人关联

## 亮点与洞察
- 实验设计非常系统，覆盖了RLHF的几乎所有主要变量（方法、模型、数据、训练策略），结论因此更加可信。这种控制变量的实验范式可以作为对齐研究的方法论模板
- "SFT固化偏见"的发现非常重要——意味着当前"SFT→RLHF"的标准训练流水线可能在SFT阶段就锁定了某些不良属性
- VLM中隐性和显性偏见可能截然相反的发现揭示了多模态对齐中一个全新的挑战维度

## 局限与展望
- 受资源限制只使用了LoRA微调而非全参数训练，全参数训练是否能更有效地改变偏见有待验证
- 偏见测量聚焦于非裔美国人vs白人的二元对比，其他种族群体和其他类型的偏见未涉及
- 用于评估偏见的AAE/SAE文本均来自社交媒体，可能不代表真实用户与LLM的交互场景
- 未来需要开发专门针对偏见消除的偏好数据集和对齐方法，而不是依赖通用的安全性数据

## 相关工作与启发
- **vs Hofmann et al. (2024)**: Hofmann发现现成RLHF模型有最强隐性偏见，本文进一步证明RLHF训练过程本身无法改善这一问题
- **vs Constitutional AI**: Claude的CAI方法设定了明确的规则约束，可能比RLHF更适合处理偏见这类目标
- **vs D'Oosterlinck et al. (2024)**: 指出偏好数据不够对比鲜明是RLHF效果不佳的一个原因，与本文的发现一致

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统地分析RLHF与隐性偏见的关系
- 实验充分度: ⭐⭐⭐⭐⭐ 实验矩阵非常完整，变量控制严格
- 写作质量: ⭐⭐⭐⭐ 跨学科工作表述清晰
- 价值: ⭐⭐⭐⭐⭐ 对理解RLHF的局限性和推动更好的对齐方法有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Synergistic Weak-Strong Collaboration by Aligning Preferences](synergistic_weak-strong_collaboration_by_aligning_preferences.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](../../NeurIPS2025/llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[ACL 2025\] Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](reward_fairness_rlhf.md)
- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](../../ICML2025/llm_alignment/dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)

</div>

<!-- RELATED:END -->
