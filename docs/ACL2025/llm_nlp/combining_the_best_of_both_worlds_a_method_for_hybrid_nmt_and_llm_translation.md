---
title: >-
  [论文解读] Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation
description: >-
  [ACL2025][LLM 其他][机器翻译] 提出基于源句特征的NMT与LLM混合翻译调度策略（PPLT与JDM），在保持翻译质量最优的同时将LLM调用比例降至约25-30%，大幅减少计算开销。 1. LLM翻译成本高延迟大：LLM在翻译任务上表现优异，但推理成本远高于传统NMT系统，对实际部署造成障碍…
tags:
  - "ACL2025"
  - "LLM 其他"
  - "机器翻译"
  - "NMT-LLM融合"
  - "调度策略"
  - "二分类决策器"
  - "翻译质量估计"
---

# Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation

**会议**: ACL2025  
**arXiv**: [2505.13554](https://arxiv.org/abs/2505.13554)  
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: 机器翻译, NMT-LLM融合, 调度策略, 二分类决策器, 翻译质量估计

## 一句话总结
提出基于源句特征的NMT与LLM混合翻译调度策略（PPLT与JDM），在保持翻译质量最优的同时将LLM调用比例降至约25-30%，大幅减少计算开销。

## 背景与动机

1. **LLM翻译成本高延迟大**：LLM在翻译任务上表现优异，但推理成本远高于传统NMT系统，对实际部署造成障碍。
2. **大部分句子NMT即可胜任**：作者标注发现约95%的句子属于"简单"类型，NMT与LLM在这些句子上的DA分数差距仅1.41分，无需动用LLM。
3. **LLM在复杂句上优势明显**：仅约5%的"困难"句子中LLM比NMT高3.80分DA，说明互补性集中在少数难句。
4. **现有QE方法存在缺陷**：Hendy等人提出的QET方法依赖质量估计模型对NMT翻译打分，但无法判断LLM翻译是否真的更好，可能引入更差的LLM结果。
5. **NMT与LLM在不同领域各有优势**：LLM在文学、网络用语方面更强，而NMT在技术领域表现更好，单一模型无法覆盖所有场景。
6. **低成本高速融合方案需求迫切**：工业界需要在不增加显著延迟的前提下获得最优翻译质量，仅在"必要时"调用LLM是最务实的方案。

## 核心问题
如何仅基于源句特征快速判断应使用NMT还是LLM进行翻译，在尽量少用LLM的前提下获得最优翻译质量？

## 方法详解

### 整体思路
与QET（先NMT翻译→QE评分→决定是否再用LLM）不同，本文直接从源句出发做二元决策，无需运行NMT翻译和QE模型，推理流程更简单、更快。

### 方法一：PPL阈值法（PPLT）
- 使用NMT训练数据中的单语数据训练一个小型语言模型（LM）
- 对输入源句计算困惑度（PPL），PPL越高表示句子越复杂
- 设定阈值：PPL > 阈值时调用LLM翻译，否则使用NMT
- 阈值确定：对100万条单语数据排序，取前25%对应的PPL值作为阈值
- **核心假设**：句子复杂度（PPL）是预测LLM优势的有效代理指标

### 方法二：联合决策法（JDM）
- **训练数据构造**：对100万条双语数据分别获取NMT和LLM翻译结果，用COMET计算质量分数 $Q_{NMT}$ 和 $Q_{LLM}$
- **正样本筛选条件**：$Q_{NMT} < T_1$ 且 $Q_{LLM} - Q_{NMT} > T_2$，即NMT翻译差且LLM明显更好
- 阈值 $T_1$ 取 $Q_{NMT}$ 排序后第10万位，$T_2$ 取差值排序后第1万位
- 最终获得1万正样本 + 3万负样本
- **决策器**：基于 xlm-roberta-base 微调的二分类模型，输入仅为源句，输出为"用NMT"或"用LLM"
- **推理时**：源句直接过决策器，无需先翻译或评估

### 关键设计选择
- JDM的两个条件确保LLM仅在"NMT差 + LLM更好"时才被调用，避免了QET无法保证LLM结果更优的问题
- 决策器仅依赖源句，不需要NMT翻译结果，因此推理延迟极低

## 实验关键数据

### 实验设置
- **NMT模型**：Deep Transformer-Big架构，每语言对1亿双语数据训练
- **LLM**：Llama-3.1-8B-Instruct，不做微调
- **测试集**：WMT22 News、Flores、自建Literary（500句）、自建Tech（500句），覆盖中英/英中/德英/日英
- **评估指标**：DA分数（COMET）、BLEURT、LLM调用比例（$LLM_p$）

### 中→英结果（Table 2）

| 方法 | News DA | Literary DA | Tech DA | 平均DA | 平均BLEURT | 平均LLM% |
|------|---------|-------------|---------|--------|------------|----------|
| NMT | 78.99 | 59.71 | 83.38 | 77.29 | 65.12 | 0% |
| LLM | 80.13 | 66.69 | 77.68 | 77.80 | 64.49 | 100% |
| QET | 79.13 | 63.88 | 80.21 | 77.58 | 64.78 | 30.55% |
| PPLT | 79.24 | 63.49 | 82.02 | 77.95 | 65.58 | 32.31% |
| **JDM** | **79.69** | **65.70** | **82.71** | **78.81** | **66.65** | **29.52%** |
| Oracle | 82.25 | 68.41 | 84.80 | 80.91 | 68.78 | 51.56% |

### 英→中结果（Table 3）

| 方法 | News DA | Literary DA | Tech DA | 平均DA | 平均BLEURT | 平均LLM% |
|------|---------|-------------|---------|--------|------------|----------|
| NMT | 86.17 | 71.68 | 86.30 | 83.01 | 67.65 | 0% |
| LLM | 85.17 | 76.30 | 78.40 | 81.63 | 63.24 | 100% |
| QET | 86.08 | 72.73 | 80.57 | 81.80 | 65.36 | 22.16% |
| **JDM** | **86.18** | **75.15** | **85.39** | **83.62** | **67.86** | **23.37%** |

### 关键发现
- JDM在所有设置下平均DA和BLEURT均最优，且LLM调用约25-30%
- JDM能自适应调整LLM使用比例：文学测试集高达80%（LLM擅长），技术测试集低至7%（NMT擅长）
- 仅用PPL（PPLT）也能超过QET，说明源句复杂度是有效的调度信号
- QET在技术领域表现差，因为它无法判断LLM是否真的更好

## 亮点

1. **思路简洁有效**：仅基于源句特征做调度决策，不需要运行NMT+QE两步评估，推理更快
2. **JDM决策器自适应性强**：在不同领域自动调整LLM使用比例，文学域重用LLM、技术域几乎不用
3. **正样本筛选的双条件设计精巧**：同时要求NMT差且LLM更好，避免引入更差的LLM翻译
4. **实际工业部署友好**：来自华为翻译服务中心，方案面向真实生产环境，成本敏感

## 局限与展望

1. **互补性依赖**：当NMT与LLM能力完全对等时，融合无法带来提升，方法前提是两者在不同场景有差异化优势
2. **决策器需要大量标注数据训练**：JDM需要100万条双语数据分别跑NMT和LLM获取训练信号，前期成本不低
3. **仅测试了8B规模LLM**：未探索更大规模LLM（如70B+）或GPT-4级别模型，随着LLM能力提升互补空间可能缩小
4. **阈值设定依赖语言对**：每个语言对需要单独调阈值，泛化性讨论不够充分
5. **未考虑LLM的其他优势**：如风格一致性、专有名词处理等非指标可衡量的翻译质量维度

## 与相关工作的对比

### vs QET (Hendy et al., 2023)
QET先用NMT翻译再用QE模型评分决定是否调LLM，存在两个问题：(1)需要QE模型额外推理成本；(2)无法判断LLM结果是否真的更好。本文JDM仅看源句即可决策，且训练时保证正样本中LLM确实优于NMT，避免了"盲目换用LLM"的问题。

### vs Cooperative Decoding (Zeng et al., 2024)
合作解码在每次翻译时都需要同时运行NMT和LLM进行联合解码，计算成本等同于全量使用LLM。本文方案仅在约25%的句子上调用LLM，计算开销大幅降低。

### vs MBR Ensembling (Farinhas et al., 2023)
MBR集成方法需要从多个模型采样多条候选翻译再选最优，推理成本极高。本文是"路由"而非"集成"，每句只走一个模型。

## 启发与关联
- **路由/调度思路通用性强**：这种"小模型能搞定就不用大模型"的策略可以推广到各种NLP任务中的大小模型协作
- 源句复杂度作为调度信号的发现可以启发其他任务（如摘要、问答）中的模型选择策略
- 决策器的训练范式（基于两个模型的输出差异构造正负样本）可迁移到其他需要模型路由的场景

## 评分
- 新颖性: ⭐⭐⭐ — 将"模型路由"思路应用到MT中，PPLT和JDM设计合理但不算突破性
- 实验充分度: ⭐⭐⭐⭐ — 4语言对×4领域，多指标评估，有Oracle上界分析
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，方法对比图直观，结构紧凑
- 价值: ⭐⭐⭐⭐ — 工业实用性强，来自华为翻译中心，成本-质量权衡分析到位

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] LLM Social Simulations Are a Promising Research Method](../../ICML2025/llm_nlp/llm_social_simulations_are_a_promising_research_method.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)

</div>

<!-- RELATED:END -->
