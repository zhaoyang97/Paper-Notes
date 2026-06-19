---
title: >-
  [论文解读] Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6×6 Sudoku
description: >-
  [ACL 2025][模型压缩][LLM reasoning] 评估五个LLM在求解和解释6×6数独谜题上的能力，发现即使o1-preview能解出65%的题目，其推理解释在忠实性、清晰度和教育价值方面仍严重不足。 问题定义：LLM在人机协作决策支持场景中的有效性不仅取决于能否给出正确答案，更取决于能否提供逐步的、一致的、可…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "LLM reasoning"
  - "Sudoku"
  - "explanation generation"
  - "human-AI collaboration"
  - "puzzle solving"
---

# Explaining Puzzle Solutions in Natural Language: An Exploratory Study on 6×6 Sudoku

**会议**: ACL 2025  
**arXiv**: [2505.15993](https://arxiv.org/abs/2505.15993)  
**代码**: 待公开  
**领域**: 模型压缩  
**关键词**: LLM reasoning, Sudoku, explanation generation, human-AI collaboration, puzzle solving  

## 一句话总结

评估五个LLM在求解和解释6×6数独谜题上的能力，发现即使o1-preview能解出65%的题目，其推理解释在忠实性、清晰度和教育价值方面仍严重不足。

## 研究背景与动机

**问题定义：** LLM在人机协作决策支持场景中的有效性不仅取决于能否给出正确答案，更取决于能否提供逐步的、一致的、可信赖的解释。本文以6×6数独谜题为受控实验环境，研究LLM在多步推理任务中的解释生成能力。

**现有方法局限：** 已有工作（如PuzzleLM）通过结合符号求解器（Z3）实现正确答案生成，但对LLM能否为解题过程提供有用的自然语言解释关注甚少。符号推理器虽能生成正确且忠实的推理步骤，但其输出通常以形式化证明呈现，对非专业用户难以理解。

**核心动机：** 好的推理者应当能够清晰暴露其演绎过程。作者认为推理解释应同时具备忠实性（faithfulness）、为每个决策提供充分依据、并且传达可推广的解题策略。数独作为需要演绎推理和规则排除的逻辑谜题，提供了一个理想的测试平台。

**研究问题：** (1) LLM能否求解6×6数独？(2) LLM能否成功解释到达给定解的推理步骤？

## 方法详解

### 整体框架

本文为评估性研究（evaluation study），分两阶段进行：
1. **求解能力评估：** 在2,293个Z3生成的6×6数独上测试五个LLM的求解准确率
2. **解释能力评估：** 从中选取20个覆盖不同难度级别的谜题子集，由三位专家评估o1-preview生成解释的质量

### 关键设计

1. **最小化数据集构建：** 使用Z3求解器生成数独谜题；从11个给定数字开始，逐步移除数字直到移除任何一个都会破坏解的唯一性。最终约96%的谜题包含9-11个给定数字。采用Explainer Rating（ER）系统量化难度，涵盖Easy（ER 1.0-1.2，73.4%）到Diabolical（ER 6.2+，1.5%）五个级别
2. **多粒度准确率体系：** 从cell-wise → row-wise → column-wise → box-wise → 完全正确solution五个层面逐级评估，可精确定位模型在约束整合中的失败模式
3. **三维度解释质量评估：** 三位专家使用三点Likert量表（Yes / Maybe / No），从Justification（是否为解提供了理由）、Clarity（是否逻辑清晰、术语一致）、Educational Value（是否传达可推广的解题策略）三个维度独立评分

### 评价指标

使用加权Kappa衡量评注者间一致性，得到教育价值κ=0.6、论证性κ=0.6、清晰度κ=0.4，表明中等至良好的一致性。人类评估者在20个谜题上的难度评分与ER的Spearman相关系数达0.87。

## 实验

### 主实验结果：求解能力

| 模型 | Cell (%) | Row (%) | Column (%) | Box (%) | 完全正确 (%) |
|------|----------|---------|------------|---------|------------|
| Mistral-7B-Instruct-v0.3 | 21.96 | 0.59 | 0.17 | 0.17 | 0.00 |
| Llama-3.1-8B-Instruct | 39.65 | 2.22 | 2.01 | 1.36 | 0.00 |
| Llama-3.1-70B-Instruct | 45.93 | 4.59 | 6.29 | 2.70 | 0.04 |
| Gemma-2-9B-Instruct | 50.12 | 6.74 | 6.53 | 3.87 | 0.04 |
| OpenAI o1-preview (100题) | 83.53 | 67.50 | 67.00 | 66.83 | 59.00 |

### 解释质量评估

| 维度 | Yes (%) | Maybe (%) | No (%) |
|------|---------|-----------|--------|
| Justification | 5.0 | 52.5 | 42.5 |
| Clarity | 7.5 | 32.5 | 60.0 |
| Educational Value | 2.5 | 52.5 | 45.0 |

### 关键发现

1. **开源模型几乎无法求解：** 即使单元正确率达40-50%，行列约束满足率仅约7%，完全正确率近0%，说明模型在全局约束整合上存在根本困难
2. **o1-preview求解与解释能力的鲜明反差：** o1在Easy/Medium难度上100%正确，整体解出59%，但解释质量极差——模型虽能找到正确数字放置，但几乎从不解释"为什么"选择这些放置
3. **解释缺乏系统性：** 评估者一致反馈解释缺乏逻辑递进、未能论证推理步骤、术语使用不一致、未能清晰表达解题路径
4. **难度敏感性：** o1-preview在Diabolical级别（ER 6.2+）完全正确率降至40%，cell-wise降至57.2%

## 亮点

- 首次系统研究LLM在多步推理任务中的**解释生成能力**，将研究焦点从"能否解"转向"能否解释"
- 使用Z3构建有理论保证（唯一解、最小化）的基准数据集，确保评估的科学严谨性
- 人类评估设计完善，包含多维度评分、评注者一致性验证和难度校准
- 提出LLM+逻辑推理器结合的未来研究方向：用符号推理器生成可靠推理链，再由LLM翻译为人类可理解的自然语言

## 局限性

- 仅使用6×6数独，复杂度有限，可能低估LLM在标准9×9数独上的局限
- o1-preview的评估规模较小（100题求解 + 20题解释），统计充分性有限
- 未探索微调（fine-tuning）对解释能力的影响
- 未测试更新版本的推理模型（如o1正式版、o3等）

## 相关工作

- **LLM谜题求解：** PuzzleVQA (Chia et al., 2024) 诊断多模态推理、PuzzleLM (Mittal et al., 2024) 结合SMT求解器
- **推理能力评估：** GSM-Symbolic (Mirzadeh et al., 2024) 揭示LLM数学推理局限
- **LLM+形式化推理：** DeepSeek-Prover (Xin et al., 2024) 探索LLM与Lean 4结合、Draft-Sketch-Prove (Jiang et al., 2022) 用非形式化证明引导形式化证明器
- **不忠实推理：** Turpin et al. (2024) 证明CoT提示下LLM的推理链可能不反映其实际推理过程

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 实用性 | ⭐⭐ |
| 实验充分度 | ⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Explaining and Mitigating Crosslingual Tokenizer Inequities](../../NeurIPS2025/model_compression/explaining_and_mitigating_crosslingual_tokenizer_inequities.md)
- [\[ACL 2025\] A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](gist_token_context_compression.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[NeurIPS 2025\] Navigating Simply, Aligning Deeply: Winning Solutions for Mouse vs. AI 2025](../../NeurIPS2025/model_compression/navigating_simply_aligning_deeply_winning_solutions_for_mouse_vs_ai_2025.md)
- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](../../NeurIPS2025/model_compression/a_granular_study_of_safety_pretraining_under_model_abliteration.md)

</div>

<!-- RELATED:END -->
