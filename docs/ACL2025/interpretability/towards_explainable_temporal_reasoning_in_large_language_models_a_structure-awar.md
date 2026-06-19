---
title: >-
  [论文解读] Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework
description: >-
  [ACL2025][可解释性][时序推理] 提出 GETER 框架，通过轻量级 Structure-Text Adapter 将时序知识图谱的结构信息注入 LLM，使模型在时序推理任务中既能给出准确预测又能生成可解释的推理说明。 领域现状： 时序推理（Temporal Reasoning）是 NLP 中的核心能力…
tags:
  - "ACL2025"
  - "可解释性"
  - "时序推理"
  - "时序知识图谱"
  - "图结构与文本对齐"
  - "指令微调"
---

# Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework

**会议**: ACL2025  
**arXiv**: [2505.15245](https://arxiv.org/abs/2505.15245)  
**代码**: [carryTatum/GETER](https://github.com/carryTatum/GETER)  
**领域**: 可解释性  
**关键词**: 时序推理, 可解释性, 时序知识图谱, 图结构与文本对齐, 指令微调

## 一句话总结

提出 GETER 框架，通过轻量级 Structure-Text Adapter 将时序知识图谱的结构信息注入 LLM，使模型在时序推理任务中既能给出准确预测又能生成可解释的推理说明。

## 研究背景与动机

**领域现状**: 时序推理（Temporal Reasoning）是 NLP 中的核心能力，在搜索推荐、新闻聚合等场景至关重要。近年来 LLM 在时序推理上取得了显著进展，多项工作通过 ICL、CoT、微调等方式提升准确率。

**现有痛点**: 几乎所有已有工作都"只关注性能、忽视可解释性"——LLM 给出预测结果但无法解释其推理过程，缺乏透明度和可信度。

**核心矛盾**: LLM 在仅依赖文本信息时，经常产生幻觉（hallucination），难以生成令人信服的时序推理解释；而传统可解释方法（逻辑规则、强化学习路径）的解释能力有限且泛化性差。

**本文目标**: 如何让 LLM 在复杂时序推理场景中同时做出准确预测并清晰展示推理过程？

**切入角度**: 利用时序知识图谱（TKG）的结构化信息弥补纯文本推理的不足，通过图结构-文本对齐来增强 LLM 的可解释时序推理能力。

**核心 idea**: 用 temporal encoder 编码 TKG 结构信息为 soft graph token，经轻量 adapter 投射到 LLM 文本空间，与 instruction-tuning prompt 拼接后生成解释文本。

## 方法详解

### 整体框架

GETER（Graph structures with text for Explainable TEmporal Reasoning）由三个核心模块组成：
1. **Temporal Encoder**：利用 RE-GCN 等 TKG 模型在时序知识图谱上学习实体和关系的结构表示
2. **Structure-Text Prefix Adapter**：将图结构表示投射到 LLM 的文本嵌入空间
3. **Instruction Tuning**：用 LoRA 微调 LLM，通过 soft graph token + prompt token 生成解释文本

### 关键设计 1：ETR Benchmark 构建

- **功能**: 构建覆盖多种时间粒度（分钟/天/年）的可解释时序推理基准。
- **为什么**: 现有 benchmark 不评估解释质量，缺乏正/负/中性样本的全面考量。
- **怎么做**:
    - 从 TKG 中用 BFS 提取推理链，转为自然语言；
    - 用 GPT-4o 基于 query + 推理链生成高质量解释文本；
    - 通过实体替换构造负样本（反事实），通过 NLI 模型筛选语义中性关系构造中性样本；
    - 涵盖 ICEWS14/ICEWS05-15/ICEWS18/GDELT/WIKI 五个数据集，共 ~60k 训练样本和 ~9k 测试样本。

### 关键设计 2：Structure-Text Adapter

- **功能**: 将 query 和推理链的图结构表示融合并投射到 LLM 嵌入空间。
- **为什么**: LLM 仅依赖文本无法捕获 TKG 中事件间的结构化时序模式。
- **怎么做**:
    - 对推理链中所有三元组 $(e_s', r', e_o')$ 的结构嵌入做拼接求和：$S_{C} = \sum (e_s' \| r' \| e_o')$；
    - 与 query 结构表示 $S_q$ 取平均后，通过线性投影矩阵 $W_p \in \mathbb{R}^{3d_s \times d_x}$ 映射到 LLM 空间；
    - 得到单个 soft graph token $S_{graph}$，作为前缀拼接到文本嵌入前。

### 关键设计 3：Instruction Tuning

- **功能**: 训练 LLM 在 soft graph token 和文本信息的联合引导下生成解释。
- **为什么**: 需要将结构信息与语义信息有机结合，同时控制微调开销。
- **怎么做**: 最终输入为 $X' = S_{graph} \| X$，优化目标为最大化解释文本 $Y_A$ 的似然；采用 LoRA 进行参数高效微调。

### 损失函数 / 训练策略

- 标准 autoregressive language modeling loss：$P(Y_A | X', X_I) = \prod_{j=1}^{L} P_\theta(y_j | X', X_I, Y_{<j})$
- LoRA 微调，temporal encoder 预训练后参数冻结
- 使用 DeepSpeed 加速训练和推理

## 实验关键数据

### 主实验：预测 F1（%）

| 模型 | ICEWS14 Overall | GDELT Overall | ICEWS05-15 Overall |
|------|:-:|:-:|:-:|
| GPT-4o zero-shot | 39.95 | 36.83 | 40.58 |
| Llama3-8B LoRA | 65.59 | 56.44 | 65.86 |
| **GETER (Llama3)** | **74.25** | **72.51** | **81.84** |
| Qwen2.5-7B LoRA | 71.90 | 46.95 | 73.72 |
| **GETER (Qwen2.5)** | **78.12** | **73.27** | **80.23** |
| Mistral-7B LoRA | 71.18 | 65.05 | 76.07 |
| **GETER (Mistral)** | **79.08** | **72.02** | **81.80** |

GETER 相比 LoRA-only 在 Overall F1 上提升 7~28%，相比 GPT-4o zero-shot 提升约 100%。

### 消融实验（Mistral，Overall F1 %）

| 变体 | ICEWS14 | GDELT | ICEWS05-15 |
|------|:-:|:-:|:-:|
| GETER (full) | 79.08 | 72.02 | 81.80 |
| w/o Structure-Text Adapter | 71.18 (↓7.90) | 65.05 (↓6.97) | 76.07 (↓5.73) |
| w/o Reasoning Chains Text | 72.05 (↓7.03) | 68.89 (↓3.13) | 77.82 (↓3.98) |
| w/o Both | 66.79 (↓12.29) | 47.80 (↓24.22) | 61.95 (↓19.85) |

### 关键发现

1. **结构信息至关重要**: 去掉 Structure-Text Adapter 导致 F1 下降 5.7~7.9%，说明图结构特征对时序推理建模有独特价值。
2. **推理链文本也不可或缺**: 推理链文本提供了时序上下文背景，去掉后性能明显下降。
3. **两者互补**: 同时去掉两个组件（Line 4），性能断崖式下降（GDELT 上 -24.22%），证明结构 + 文本的融合是 GETER 成功的关键。
4. **对 temporal encoder 鲁棒**: 替换为 CEN/CENET/SiMFy 等不同编码器后，GETER 仍然显著优于 LoRA-only。
5. **MLP 深度影响小**: 1 层 MLP 已足够，更深的网络无额外收益。
6. **推理链顺序**: 降序排列效果最优（80.68%），但即使随机排列也能保持 77.57%，体现了框架的鲁棒性。

## 亮点与洞察

1. **首次系统性研究 LLM 的可解释时序推理**：不仅要求预测正确，还要求解释合理，填补了该方向的空白。
2. **ETR benchmark 设计精巧**：正/负/中性三类样本 + 多时间粒度覆盖，评估维度全面。
3. **轻量级对齐方案**：仅用一个线性投影就完成图结构→文本空间的桥接，证明简单方法在跨模态对齐中依然有效。
4. **实际意义**: 可解释的时序推理对新闻分析、金融预测、事件预警等场景有直接应用价值。

## 局限与展望

1. **计算开销**: LLM 微调和推理仍然资源密集，尽管用了 LoRA 和 DeepSpeed。
2. **推理链噪声**: 部分 BFS 提取的推理链可能含噪声，影响解释质量。
3. **GPT-4o 生成的 ground-truth 解释**: benchmark 的"金标准"解释由 GPT-4o 生成，而非人类标注，可能引入系统偏差。
4. **时间粒度限制**: 虽然覆盖了分钟/天/年，但未涉及更细粒度（秒级）或更复杂的时间推理（区间推理、周期性推理）。
5. **soft graph token 为单个向量**: 所有图结构信息压缩为一个 token，对复杂推理链可能会造成信息瓶颈。

## 相关工作与启发

### vs. 逻辑规则方法（TLogic 等）
逻辑规则方法通过显式规则模板保证可解释性，但泛化能力差，难以处理复杂场景。GETER 借助 LLM 的生成能力，生成更灵活、更自然的推理解释，同时保留了图结构的精确性。

### vs. 强化学习路径方法（CluSTeR、TITer 等）
RL 方法通过预定义奖励机制构建推理路径，但决策过程隐式，可解释性受限。GETER 直接生成自然语言解释，可读性远优于 RL 路径表示。

### vs. 纯文本 LLM 方法（CoT、ICL）
纯文本方法忽略了 TKG 的结构化信息，容易产生幻觉。GETER 通过引入结构先验，有效减少幻觉并提升解释质量。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将图结构信息注入 LLM 以增强可解释时序推理，benchmark + framework 双贡献
- **实验充分度**: ⭐⭐⭐⭐ — 5 个数据集、4 个 LLM backbone、详细消融与讨论，实验全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机阐述充分
- **价值**: ⭐⭐⭐⭐ — 可解释时序推理是重要且未被充分探索的方向，benchmark 和框架均有后续研究价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](../../ACL2026/interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)

</div>

<!-- RELATED:END -->
