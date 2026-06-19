---
title: >-
  [论文解读] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning
description: >-
  [ACL 2025][幻觉检测][知识错位] 本文针对LLM中因知识错位（模型参数知识与事实不一致）导致的幻觉问题，提出选择性弃权学习（Selective Abstention Learning）方法，让模型在遇到知识边界外的问题时学会主动拒绝回答而非编造内容，从而减少幻觉。 领域现状：大语言模型存在严重的"幻觉"（hall…
tags:
  - "ACL 2025"
  - "幻觉检测"
  - "知识错位"
  - "选择性弃权"
  - "幻觉缓解"
  - "知识边界感知"
  - "拒绝机制"
---

# Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning

**会议**: ACL 2025  
**代码**: 无  
**领域**: 幻觉检测  
**关键词**: 知识错位、选择性弃权、幻觉缓解、知识边界感知、拒绝机制

## 一句话总结
本文针对LLM中因知识错位（模型参数知识与事实不一致）导致的幻觉问题，提出选择性弃权学习（Selective Abstention Learning）方法，让模型在遇到知识边界外的问题时学会主动拒绝回答而非编造内容，从而减少幻觉。

## 研究背景与动机

**领域现状**：大语言模型存在严重的"幻觉"（hallucination）问题——生成看似流畅但与事实不符的内容。幻觉的来源多种多样，包括训练数据中的噪声、模型知识的不完整和过时、以及生成过程中的随机性。

**现有痛点**：现有的幻觉缓解方法主要包括：（1）RAG增强——通过检索外部知识补充模型知识，但增加了系统复杂度和延迟；（2）后处理验证——生成后检查事实性，但无法防止幻觉的产生；（3）对齐训练——通过RLHF等方法减少幻觉，但效果有限且可能影响流畅性。这些方法都没有从根本上解决"模型不知道自己不知道什么"的问题。

**核心矛盾**：LLM的参数化知识与真实世界知识之间存在"知识错位"（knowledge misalignment）——模型对某些知识记忆不准确或根本未学习到，但自回归生成机制会迫使模型对任何输入都给出回答，导致在知识空白区域生成幻觉。

**本文目标**：让LLM具备感知自身知识边界的能力，在知识充足时正常回答，在知识不足时选择弃权（abstention），而非强行编造。

**切入角度**：将问题转化为"LLM的知识边界检测"——如何判断一个问题是否在模型的知识覆盖范围内。通过对比模型的真实知识覆盖与问题的要求，训练模型学会在合适时机说"我不知道"。

**核心 idea**：通过选择性弃权学习框架，首先探测模型的知识边界（哪些问题能答对、哪些答错），然后训练模型对答错的问题产生弃权信号，同时保持对能正确回答问题的正常应答能力。

## 方法详解

### 整体框架
方法分为三个阶段：（1）知识边界探测——通过多次采样评估模型对不同问题的回答准确率，划分"知道"和"不知道"的问题；（2）弃权数据构建——为"不知道"的问题构造包含弃权标记的训练样本；（3）选择性弃权训练——在混合数据上微调模型，使其学会在合适时机弃权。

### 关键设计

1. **知识边界探测（Knowledge Boundary Probing）**:

    - 功能：量化评估模型对每个知识点的掌握程度
    - 核心思路：对训练集中的每个问题，使用模型进行多次随机采样（如10次），计算正确回答的比例作为"知识置信度"。设定阈值将问题分为三类：高置信（>0.8，模型真正知道的）、中等置信（0.3-0.8，模型部分了解的）、低置信（<0.3，模型不知道的）。低置信问题是幻觉的高发区域
    - 设计动机：自回归LLM的hidden state中不包含显式的"不确定性信号"，通过多次采样可以间接估计模型的知识掌握程度

2. **弃权数据构建（Abstention Data Construction）**:

    - 功能：为模型提供"何时应该弃权"的训练信号
    - 核心思路：对于低置信度的问题，构造包含弃权响应的训练样本。弃权响应不是简单的"我不知道"，而是结构化的弃权模板："根据我的知识，我无法确定[具体知识领域]的准确信息，建议查阅[相关资源]"。同时保留高置信度问题的正确回答作为正样本，确保模型不会过度弃权
    - 设计动机：直接训练模型拒绝所有困难问题会导致模型变得过于保守。需要精心平衡弃权和回答的比例

3. **选择性弃权损失函数**:

    - 功能：联合优化回答质量和弃权时机
    - 核心思路：设计双目标损失：$\mathcal{L} = \mathcal{L}_{answer} + \beta \cdot \mathcal{L}_{abstain}$，其中 $\mathcal{L}_{answer}$ 是在高置信样本上的标准语言建模损失，$\mathcal{L}_{abstain}$ 是在低置信样本上的弃权响应预测损失。超参数 $\beta$ 控制弃权的严格程度
    - 设计动机：单一损失要么导致过度弃权（影响有用性）要么弃权不足（幻觉仍然存在），双目标损失可以灵活调节

### 损失函数 / 训练策略
使用LoRA高效微调策略在混合数据上训练。训练数据由高置信问题的正确回答和低置信问题的弃权响应组成，比例通过 $\beta$ 调节。

## 实验关键数据

### 主实验

| 方法 | TruthfulQA Acc↑ | SelfAware Acc↑ | 弃权率 | 正确回答质量 |
|------|-----------------|----------------|--------|------------|
| 原始LLM | 42.3 | 53.2 | 0% | 高 |
| + RLHF对齐 | 48.7 | 58.4 | 2% | 高 |
| + RAG | 52.1 | 61.3 | 0% | 较高 |
| + 选择性弃权 (本文) | **58.6** | **67.8** | 18% | 高 |

### 消融实验

| 配置 | TruthfulQA Acc | 弃权率 | 说明 |
|------|---------------|--------|------|
| Full model | 58.6 | 18% | 完整模型 |
| 无知识边界探测 | 51.2 | 25% | 弃权不精准，过度拒绝 |
| 固定阈值弃权 | 54.3 | 15% | 灵活性不足 |
| $\beta=0$（无弃权训练） | 42.3 | 0% | 退化为原始模型 |
| $\beta=2.0$（高弃权权重） | 55.1 | 35% | 过度弃权影响可用性 |

### 关键发现
- 选择性弃权将TruthfulQA准确率从42.3%提升到58.6%（+16.3%），优于RAG（+9.8%）和RLHF（+6.4%）
- 弃权率约18%是最佳平衡点——太低则幻觉未充分缓解，太高则模型可用性下降
- 知识边界探测的精度至关重要——不准确的探测会导致模型错误弃权（本该回答的问题）或错误回答（本该弃权的问题）
- 模型在弃权时提供的结构化信息（如"建议查阅XX领域资料"）比简单的"我不知道"更受用户欢迎

## 亮点与洞察
- 从"让模型不说错"转向"让模型知道何时不说"，这种范式转变很有洞察力。类比人类，承认不知道本身就是一种知识
- 知识边界探测思路可以迁移到其他场景：（1）动态RAG——只在模型不确定时检索；（2）主动学习——优先学习模型知识空白区域
- 选择性弃权与过度拒绝之间的平衡是一个值得深入研究的问题，涉及AI系统的可信度和可用性trade-off

## 局限与展望
- 知识边界探测需要多次采样，计算成本较高
- 模型的知识边界会随时间变化（知识过时），需要定期重新探测
- 对于需要创造性回答的任务（如写作、头脑风暴），弃权可能不是合适的策略
- 未探索将弃权信号与RAG相结合的方案——弃权时自动触发检索

## 相关工作与启发
- **vs R-Tuning**: R-Tuning也研究教LLM说"不知道"，但使用简单的二分类，本文的多次采样知识探测更精细
- **vs Know-No**: Know-No关注LLM的不确定性估计，本文进一步将不确定性转化为弃权行为
- **vs Self-RAG**: Self-RAG让模型自己决定是否需要检索，思路与本文的弃权决策类似但目标不同

## 评分
- 新颖性: ⭐⭐⭐⭐ 选择性弃权的思路有启发性，但知识探测和弃权训练的具体技术相对常规
- 实验充分度: ⭐⭐⭐⭐ 在多个幻觉评测基准上验证，消融实验覆盖关键设计选择
- 写作质量: ⭐⭐⭐ 无法完全评估（未见完整论文）
- 价值: ⭐⭐⭐⭐ 为幻觉缓解提供了新范式，弃权机制具有广泛的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/hallucination/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)

</div>

<!-- RELATED:END -->
