---
title: >-
  [论文解读] Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework
description: >-
  [ACL 2025][AI安全][gender fairness] 提出 GIFI（Gender Inclusivity Fairness Index），一个涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和数学推理一致性七个维度的多层次评估框架，在 22 个主流 LLM 上系统量化二元与非二元性别的公平性，揭示新代词在无提示时完全缺席、"she" 过度矫正等深层偏见模式。
tags:
  - "ACL 2025"
  - "AI安全"
  - "gender fairness"
  - "non-binary pronouns"
  - "LLM evaluation"
  - "inclusivity index"
  - "neopronouns"
---

# Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework

**会议**: ACL 2025  
**arXiv**: [2506.15568](https://arxiv.org/abs/2506.15568)  
**代码**: [https://github.com/ZhengyangShan/GIFI](https://github.com/ZhengyangShan/GIFI)  
**领域**: AI 安全 / 公平性评估  
**关键词**: gender fairness, non-binary pronouns, LLM evaluation, inclusivity index, neopronouns

## 一句话总结

提出 GIFI（Gender Inclusivity Fairness Index），一个涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和数学推理一致性七个维度的多层次评估框架，在 22 个主流 LLM 上系统量化二元与非二元性别的公平性，揭示新代词在无提示时完全缺席、"she" 过度矫正等深层偏见模式。

## 研究背景与动机

**领域现状：** LLM 的快速发展推动了 NLP 各领域的进步，但同时也带来了公平性隐忧，性别表征是其中最受关注的问题之一。现有性别偏见研究（如 Bolukbasi et al. 2016 的词嵌入偏见、StereoSet、CrowS-Pairs 等）已建立了一定的评估基础，但几乎所有工作都局限于二元性别（male/female）的框架内。

**现有痛点：** 绝大多数研究仅评估 "he" 和 "she" 两个代词的表现差异，完全忽略了非二元性别身份。非二元代词（如 singular they/them）以及新代词（neopronouns，如 xe/xem、ze/zir、ae/aer 等）在 LLM 中的表现几乎从未被系统评估。已有数据集如 StereoSet 和 CrowS-Pairs 并不专门针对非二元性别的表征或生活经验，导致对这些群体的偏见无法被检测。

**核心矛盾：** 社会对性别多元化的认知在不断深化，新代词体系也在持续演变，但 AI 系统——特别是对话式 LLM——是否能够尊重并正确处理这些多元身份，目前缺乏可靠的度量标准。虽然部分工作（如 MISGENDERED、Ovalle et al. 2023）开始关注非二元代词的错误指代问题，但它们仅覆盖单一维度，没有提供综合性评估。

**本文目标：** 作者旨在建立一个覆盖非二元性别的综合性公平性评估框架，从代词识别到深层认知推理，系统量化 LLM 对 11 组不同性别代词的包容程度，并在 22 个主流模型上进行大规模基准测试。

**切入角度：** 将评估设计为由浅入深的四个递进阶段：从简单的代词识别（模型能否正确使用指定代词），到分布公平性（情感/毒性是否因代词而变化），到刻板印象关联（无提示时是否自发偏向特定性别），最终到性能一致性（数学推理等表面上与性别无关的任务是否受代词影响）。

**核心 idea：** 将七个 [0,1] 归一化维度聚合为一个可解释的 GIFI 综合分数（0-100），成为首个涵盖非二元性别的 LLM 公平性基准。

## 方法详解

### 整体框架

GIFI 框架由四个递进评估阶段、七个维度组成，评估对象覆盖 11 个代词组（2 个二元 + 1 个中性 + 8 个新代词）。所有维度指标均归一化到 [0,1]，分数越高表示越公平。最终 GIFI = 七个维度均值 × 100，范围 0-100。四个阶段分别为：(1) 代词识别（GDR），(2) 分布公平性（SN、NTS、CF），(3) 刻板印象与角色分配（SA、OF），(4) 性能一致性（PE）。

### 关键设计

**1. 多层次代词公平性度量（GDR + SN + NTS + CF 四维度联合）**

- **功能：** 从代词识别和分布公平性两个层面全面检测模型对不同性别代词的处理差异
- **核心思路：** GDR 维度对每组代词构造含该代词的 prompt，检查模型输出是否保持代词一致性，使用变异系数 CV = σ/μ 衡量跨代词组的一致性，GDR = 1/(1+CV)。SN 和 NTS 分别使用 RoBERTa 情感分类器和 Google Perspective API 为仅改变代词的同一 prompt 的输出评分，通过 1 - Average MAD 度量跨代词的情感/毒性一致性。CF 使用 sentence-transformers 编码配对输出，以余弦相似度阈值 γ=0.3 判定是否"实质不同"
- **设计动机：** 代词识别是最基础的公平性要求——如果模型连代词都无法正确使用，讨论更深层的公平性毫无意义。分布公平性则揭示模型在内容生成层面是否因代词差异产生情感、毒性或语义的不对称变化

**2. 刻板印象与职业偏见检测（SA + OF 双维度）**

- **功能：** 检测模型在无显式性别提示时是否自发将特定代词与刻板印象活动、性格特征或职业关联
- **核心思路：** 使用 "My friend is a {occupation}" / "My friend likes {hobby}" 等无性别提示的模板 prompt，对同一 prompt 多次采样（温度 0.95），统计模型输出中各代词组的频率分布，计算其与均匀分布的平方偏差。SA 面向性格/活动/颜色偏好，OF 面向 80 个职业（40 个男性主导 + 40 个女性主导）。两者均排除 "they" 以避免其作为默认中性代词的干扰
- **设计动机：** 这一层面测试的是模型的隐式偏见——即使没有任何性别线索，模型是否仍会将 "护士" 与 "she"、"工程师" 与 "he" 关联。这种偏见反映了训练数据中的社会刻板印象

**3. 跨性别推理性能一致性（PE 维度）**

- **功能：** 评估模型在表面上与性别无关的任务（数学推理）中，是否因代词不同而产生性能差异
- **核心思路：** 基于 GSM8K 数据集，使用 NER 提取包含单一人名的数学题，将人名替换为 11 组不同代词，生成 1100 个样本。使用 8-shot CoT prompting 评估准确率，PE 同样基于 CV 公式计算。同时进行实例级一致性分析，将结果分类为"全部正确"、"全部错误"、"二元/中性正确但新代词错误"等
- **设计动机：** 如果一个模型在遇到 "xe bought 3 books" 时推理能力下降，而对 "he bought 3 books" 表现正常，这暴露的是更深层的内在偏见——代词的陌生程度影响了与性别本身无关的认知能力

## 实验关键数据

### 主实验：22 个模型 GIFI 七维度评分（Top-10 模型）

| 模型 | GDR | SN | NTS | CF | SA | OF | PE | **GIFI** |
|------|-----|------|------|------|------|------|------|----------|
| GPT-4o | 0.76 | 0.77 | 0.96 | 0.86 | 0.37 | 0.41 | 0.96 | **73** |
| Claude 3 | 0.67 | 0.78 | 0.95 | 0.87 | 0.31 | 0.42 | 0.97 | **71** |
| DeepSeek V3 | 0.67 | 0.68 | 0.93 | 0.89 | 0.56 | 0.18 | 0.99 | **70** |
| GPT-4o-mini | 0.61 | 0.81 | 0.94 | 0.99 | 0.36 | 0.13 | 0.95 | **68** |
| GPT-4 | 0.71 | 0.78 | 0.93 | 0.84 | 0.34 | 0.14 | 0.96 | **67** |
| Claude 4 | 0.80 | 0.83 | 0.93 | 0.63 | 0.34 | 0.17 | 0.97 | **67** |
| Gemini 1.5 Pro | 0.55 | 0.78 | 0.92 | 0.74 | 0.37 | 0.36 | 0.97 | **67** |
| GPT-3.5-turbo | 0.64 | 0.73 | 0.93 | 0.82 | 0.35 | 0.14 | 0.96 | **65** |
| Gemma 3 | 0.65 | 0.70 | 0.91 | 0.60 | 0.47 | 0.20 | 0.96 | **64** |
| Gemini 2.0 Flash | 0.70 | 0.77 | 0.87 | 0.53 | 0.40 | 0.24 | 0.99 | **64** |

排名末尾的模型：Vicuna (49)、GPT-2 (55)、LLaMA 2 (57)、Zephyr (57)。

### 消融/鲁棒性：情感分类器对比（RoBERTa vs VADER）

| 模型 | SN (RoBERTa) | SN (VADER) |
|------|-------------|------------|
| Claude 4 | 0.830 | 0.828 |
| GPT-4o-mini | 0.810 | 0.756 |
| Gemini 1.5 Pro | 0.776 | 0.755 |
| GPT-4o | 0.765 | 0.724 |
| Claude 3 | 0.783 | 0.690 |
| DeepSeek V3 | 0.684 | 0.650 |
| Yi-1.5 | 0.672 | 0.444 |

两种分类器的 Pearson 相关系数为 r=0.785，说明 SN 结论不依赖于特定情感分类器的选择。

### 关键发现

1. **新代词在无提示时完全缺席**：所有 22 个模型在 SA/OF 任务中从不自发生成 xe、ze 等新代词，暴露了训练数据中新代词的极度稀缺
2. **"she" 过度矫正现象普遍**：GPT-4o 在刻板印象任务中 she 占比高达 0.86，LLaMA 4 达 0.83，Claude 4 在职业任务中 she 高达 0.72 vs he 仅 0.26，反映去偏见训练的矫枉过正
3. **代词识别层级分明**：二元代词 > they > 新代词，即使最强的 Claude 4 平均准确率也仅 0.75，新代词识别率普遍低于 0.50
4. **推理公平性 ≈ 推理能力**：强模型（Gemini 2.0 Flash、DeepSeek V3 均 0.92 准确率）对所有代词表现一致，弱模型失败也是对所有代词一致的失败
5. **各维度表现可高度不一致**：Claude 4 的 GDR 最高(0.80)但 OF 很差(0.17)；Phi-3 的 SA 最高(0.72)但 CF 最差(0.25)

## 亮点与洞察

1. **首个覆盖非二元性别的综合公平性指标**：相比 MISGENDERED 等仅关注单一维度的工作，GIFI 同时涵盖识别、生成、偏见、推理四个层面，填补了重要的研究空白
2. **四阶段递进设计构思精巧**：从浅层的代词识别到深层的推理一致性，形成了 LLM 性别公平性的"压力测试渐强"范式，能逐层暴露不同程度的偏见
3. **发现去偏见训练的"过度矫正"悖论**：新一代模型并非更公平，而是将偏见从 "he" 转移到了 "she"，真正的中性表达（they）和新代词仍被严重忽视
4. **PE 维度的洞察力出色**：将数学推理与性别代词交叉测试，证明即便是表面无关的任务也会受到代词陌生度的影响
5. **评估规模空前**：22 个模型 × 7 个维度 × 11 个代词组，产生了丰富的交叉分析数据

## 局限与展望

1. **仅覆盖英语**：不同语言的性别系统差异巨大（如法语的语法性别、中文的无形态变化），框架无法直接迁移
2. **外部分类器自身存在偏见**：RoBERTa 情感模型和 Perspective API 可能对不同代词有不同的灵敏度，虽然消融实验（r=0.785）部分缓解了这一问题
3. **数据污染风险**：RealToxicityPrompts 等数据集在 2022 年前发布，可能已被较新模型的训练集覆盖
4. **GIFI 使用简单平均聚合七个维度**：SA 和 OF 得分普遍远低于 NTS，等权平均可能掩盖某些维度的严重不公平
5. **缺少交叉性分析**：未考虑性别与种族、年龄等其他敏感属性的交叉偏见
6. **新代词集合不完整且持续演变**：框架虽支持扩展，但当前选择的 8 组新代词代表性有限

## 相关工作与启发

| 对比方向 | 已有工作 | GIFI（本文） |
|---------|---------|-------------|
| 性别范围 | StereoSet、CrowS-Pairs 仅覆盖二元性别 | 覆盖 11 组代词，包含 8 种新代词 |
| 评估维度 | MISGENDERED 仅关注代词错误指代 | 七个维度从识别到推理全面覆盖 |
| 综合指标 | GenderCare (Tang et al. 2024) 仅限二元 | GIFI 提供包含非二元的单一可解释分数 |
| 模型覆盖 | Ovalle et al. 2023 测试少量模型 | 22 个模型的大规模对比 |
| 认知层面 | 已有工作多聚焦于表面输出 | PE 维度测试代词对推理能力的隐式影响 |

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个涵盖非二元性别的综合 LLM 公平性指标，填补重要空白
- **实验充分度**: ⭐⭐⭐⭐⭐ — 22 个模型 × 7 个维度 × 11 个代词组，含消融实验和定性失败分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，指标定义严谨，附录详尽
- **价值**: ⭐⭐⭐⭐⭐ — 直接可用于 LLM 公平性审计，"过度矫正"和"新代词缺席"发现具有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Fairness via Independence: A General Regularization Framework for Machine Learning](../../ICLR2026/ai_safety/fairness_via_independence_a_general_regularization_framework_for_machine_learnin.md)
- [\[ACL 2025\] Towards Fairness Assessment of Dutch Hate Speech Detection](towards_fairness_assessment_of_dutch_hate_speech_detection.md)
- [\[ICML 2026\] COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs](../../ICML2026/ai_safety/copf_an_online_framework_for_deployment-stable_counterfactual_fairness_in_evolvi.md)
- [\[ACL 2025\] FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)
- [\[NeurIPS 2025\] Fairness under Competition](../../NeurIPS2025/ai_safety/fairness_under_competition.md)

</div>

<!-- RELATED:END -->
