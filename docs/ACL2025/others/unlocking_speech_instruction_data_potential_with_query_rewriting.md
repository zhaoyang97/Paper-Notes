---
title: >-
  [论文解读] Unlocking Speech Instruction Data Potential with Query Rewriting
description: >-
  [ACL 2025][语音指令数据] 提出基于多LLM知识融合的查询重写框架与多智能体标注验证方法，将超出TTS词汇范围的文本指令重写为适合语音合成的形式，使语音指令数据可用率从72%提升至93%，为端到端大型语音语言模型(LSLM)构建高质量语音指令数据集。 领域现状： 端到端大型语音语言模型(LSLM)在响应延迟和语音理…
tags:
  - "ACL 2025"
  - "语音指令数据"
  - "Query Rewriting"
  - "TTS"
  - "多智能体标注"
  - "知识融合"
---

# Unlocking Speech Instruction Data Potential with Query Rewriting

**会议**: ACL 2025  
**arXiv**: [2507.08603](https://arxiv.org/abs/2507.08603)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: 语音指令数据, Query Rewriting, TTS, 多智能体标注, 知识融合  

## 一句话总结

提出基于多LLM知识融合的查询重写框架与多智能体标注验证方法，将超出TTS词汇范围的文本指令重写为适合语音合成的形式，使语音指令数据可用率从72%提升至93%，为端到端大型语音语言模型(LSLM)构建高质量语音指令数据集。

## 研究背景与动机

**领域现状**: 端到端大型语音语言模型(LSLM)在响应延迟和语音理解方面展现出巨大潜力，但其遵循语音指令的能力因缺乏高质量数据集和训练任务严重偏向(如ASR重复任务)而未被充分释放。

**现有痛点**: (1) 人工采集和标注语音指令数据成本极高，难以大规模构建；(2) 早期方法利用LLM续写ASR数据的语义来构造语音指令，但LLM生成结果与真实人类回答存在差距，续写方法会进一步放大此缺陷；(3) 利用TTS合成语音时，TTS模型词汇有限，无法正确转换缩写、复合词、数学公式等分布外文本，造成语言信息丢失。

**核心矛盾**: 高质量语音指令数据集的构建需要在低成本自动化和高语言保真度之间取得平衡——直接用TTS合成会因分布外文本导致信息丢失，而人工标注又不可扩展。

**本文目标**: 如何低成本、自动化地将文本指令转换为语言信息等价的高质量合成语音，构建大规模语音指令数据集。

**切入角度**: 通过多LLM零样本重写将分布外文本转换为TTS友好的形式，并用多ASR模型+多Embedding模型组成"多智能体"自动标注和验证合成语音质量。

**核心 idea**: 用多个LLM重写文本让TTS"读得准"，用多个ASR+Embedding模型验证"读得对"，再通过知识融合处理疑难重写样本。

## 方法详解

### 整体框架

给定原始文本指令 $c_o$，首先用多个LLM（Llama3、Phi3、Qwen2）将其重写为多个候选文本，与原始文本构成候选集 $C = \{c_o, c_l, c_p, c_q\}$；然后用TTS模型（Parler-TTS）合成对应语音集 $S$；接着用3个ASR模型（Whisper-large-v3、Canary-1b、Parakeet-tdt-1.1b）识别语音中的语言信息，用3个Embedding模型计算与原始文本的语义相似度，选择最优的合成结果；最后对仍失败的样本进行知识融合训练，微调出更强的重写模型处理疑难case。

### 关键设计

1. **多智能体标注与验证 (Multi-agent Annotation & Verification)**
    - **功能**: 准确评估合成语音的语言信息保真度
    - **核心思路**: 用3个不同架构的ASR模型（性能相近但互补）识别语音，用3个Embedding模型计算语义相似度的均值，取最优ASR结果的相似度作为质量分数 $q = \max_j F(c_o, \bar{c}_{o,j})$
    - **设计动机**: 单一ASR模型的识别错误会导致不当数据过滤；类比人类标注中整合多个标注者意见的做法，利用模型间的正交性减少一致性错误

2. **基于多LLM的查询重写 (Query Rewriting with Multi-LLM)**
    - **功能**: 将TTS无法正确合成的文本（缩写、公式、复合词）重写为TTS可处理的形式
    - **核心思路**: 分别用Llama3-8B、Phi3-small、Qwen2-7B对原始文本进行零样本重写，从4个候选（含原始）中选择合成质量最高的作为TTS输入
    - **设计动机**: 手工设计规则的重写方法依赖人工难以扩展；不同LLM在零样本重写上的表现具有正交性，多个候选可覆盖更多情况

3. **知识融合处理疑难重写 (Knowledge Fusion for Challenging Rewriting)**
    - **功能**: 解决多LLM联合仍无法成功重写的困难样本
    - **核心思路**: 收集成功重写的 $\langle c_i, \hat{c}_i \rangle$ 样本对作为训练数据，用LoRA微调Llama-3-8B-Instruct，融合多个LLM的重写知识训练出更强的重写模型，专门处理失败样本
    - **设计动机**: 困难重写任务需要多视角能力（如同时理解上下文和领域知识），知识融合可将多个模型的互补优势整合到单一模型中

### 损失函数 / 训练策略

- **知识融合训练**: 使用标准自回归语言模型损失 $\mathcal{L} = -\sum_{i=0}^{M} \log P(y_i | x, c, y_{<i})$，其中 $\langle c, y \rangle$ 为成功重写样本对
- **LoRA配置**: 知识融合阶段 r=8, a=16, 学习率3e-4；LSLM微调阶段 r=8, a=32, 学习率3e-5
- **质量阈值**: $\alpha = 0.9$ 用于划分成功/失败重写样本

## 实验关键数据

### 主实验

在7个QA数据集上评估，Multi-Speaker Setting下使用Parler-TTS-Large-v1：

| 方法 | SIM (Avg, ↑) | PASS (Avg, ↑) |
|------|-------------|--------------|
| Original（不重写） | 93.14 | 72.19 |
| Text Normalization | 95.34 | 82.05 |
| Phi3单独重写 | 97.05 | 88.48 |
| Llama3单独重写 | 96.94 | 88.36 |
| Ours w/o KF | 97.91 | 92.52 |
| **Ours** | **97.99** | **93.07** |

### 消融实验

LSLM（Qwen2-Audio-7B-Instruct）训练效果对比：

| 训练目标 | 阈值 | 数据方法 | DROP | Quoref | ROPES | NarrativeQA | Avg |
|---------|------|---------|------|--------|-------|-------------|-----|
| 无微调 | - | - | 17.40 | 55.98 | 42.69 | 43.02 | 39.77 |
| Golden | 0 | Original | 29.25 | 76.01 | 55.42 | 48.34 | 52.26 |
| LLM Continue | 0 | Ours | 30.08 | 75.05 | 57.15 | 47.88 | 52.54 |
| Golden | 0.90 | Ours | **44.35** | **86.81** | **64.24** | **56.76** | **63.04** |

### 关键发现

1. 多LLM联合重写比任一单一LLM高约4.5% PASS，验证了正交性假设
2. 多ASR联合标注比单一ASR模型平均降低WER约1.7个百分点
3. Golden（人类标注回答）作为对齐目标显著优于LLM续写（Avg 60.50 vs 52.54），说明真实人类响应不可替代
4. 质量阈值 t=0.90 最优，过高(0.95)反而因丢弃过多数据而性能下降

## 亮点与洞察

- 方法全自动化无需人工标注，从数据构建到质量评估端到端完成
- 多智能体标注验证思路优雅——借鉴众包标注"多标注员取共识"的思想
- 实验揭示重要洞察：语音指令对齐中LLM续写数据无法替代人类标注数据
- 方法跨TTS模型泛化性强，将不同TTS模型间的质量差距从13.95%缩小到0.9%

## 局限与展望

- 仅在英语QA数据集上验证，多语言场景有效性未知
- 依赖GPT-4生成说话人描述且仅192个，多样性可能不够
- 知识融合仅基于LoRA微调单一backbone，多模型融合训练可能更好
- 未探讨合成语音中韵律、情感等副语言信息的保真度

## 相关工作与启发

- Parler-TTS用自然语言描述控制语音风格的思路值得借鉴
- 多智能体标注思想可推广到其他自动数据构建场景（多模态、代码等）
- 知识融合的思路启发了跨模型能力互补的研究方向

## 评分

- **新颖性**: 3/5 — 各组件为已有技术组合，但整体方案设计合理
- **技术深度**: 3/5 — 以工程设计为主
- **实验充分度**: 4/5 — 多设置、多消融验证
- **实用性**: 4/5 — 对构建语音指令数据集具有直接参考价值
- **综合评分**: 3.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)
- [\[ACL 2025\] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)
- [\[ACL 2025\] Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection](tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](../../CVPR2025/others/potential_field_based_deep_metric_learning.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)

</div>

<!-- RELATED:END -->
