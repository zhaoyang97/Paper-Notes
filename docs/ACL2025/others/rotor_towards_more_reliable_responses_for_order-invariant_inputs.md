---
title: >-
  [论文解读] RoToR: Towards More Reliable Responses for Order-Invariant Inputs
description: >-
  [ACL 2025][位置偏差] 提出 RoToR，一种基于全局排序和循环位置编码分配的零样本顺序不变语言模型，通过最小化位置 ID 修改来实现稳定的顺序不变性，并设计选择路由（Selective Routing）机制自适应处理混合输入类型。 语言模型对输入的顺序非常敏感,但很多实际场景中列表式输入（如表格行、多选项、检索文…
tags:
  - "ACL 2025"
  - "位置偏差"
  - "顺序不变性"
  - "位置编码"
  - "因果语言模型"
  - "选择路由"
---

# RoToR: Towards More Reliable Responses for Order-Invariant Inputs

**会议**: ACL 2025  
**arXiv**: [2502.08662](https://arxiv.org/abs/2502.08662)  
**代码**: 有 ([https://github.com/soyoung97/RoToR](https://github.com/soyoung97/RoToR))  
**领域**: 其他  
**关键词**: 位置偏差, 顺序不变性, 位置编码, 因果语言模型, 选择路由

## 一句话总结

提出 RoToR，一种基于全局排序和循环位置编码分配的零样本顺序不变语言模型，通过最小化位置 ID 修改来实现稳定的顺序不变性，并设计选择路由（Selective Routing）机制自适应处理混合输入类型。

## 研究背景与动机

语言模型对输入的顺序非常敏感,但很多实际场景中列表式输入（如表格行、多选项、检索文档集合）的顺序应该是无关紧要的。这种"位置偏差"（positional bias）问题广为人知：

- 在 LLM-as-a-judge 场景中，模型对第一个答案有高达 75% 的偏好
- 在 MMLU 上，仅改变选项顺序就能使模型排名变动 8 个位置
- "lost-in-the-middle" 现象：中间位置的信息被严重忽略

现有零样本顺序不变方法存在两个关键局限：

**训练-推理分布不匹配**：PCW 完全隔离段间注意力，PINE 为每个 query token 动态重新分配位置 ID，导致位置编码分配与预训练时的分布差距太大

**无法适应混合输入**：实际问题（如 MMLU）既包含顺序无关的选项（A、B、C），也包含顺序敏感的选项（如"None of the above"），现有方法用单一策略处理两者

RoToR 的核心思路是：**用一个全局排序 + 循环分配替代 PINE 的逐 query 动态排序**，大幅减少位置 ID 的扰动；同时引入选择路由来自适应处理顺序敏感和不敏感的输入。

## 方法详解

### 整体框架

RoToR 由两个阶段组成：
1. **RoToR 核心**：一种新的位置 ID 分配方案，使用全局排序和循环排列实现顺序不变性
2. **选择路由（Selective Routing）**：根据原始模型和不变模型的置信度自适应选择输出

### 关键设计

1. **全局排序（Global Ordering）**

    - 功能：为所有输入段确定一个统一的排列顺序，而非 PINE 的逐 query 排列
    - 核心思路：
        - 提供三种全局排序算法：
       - **词典排序（Lexicographical）**：基于 token 序列的字典序，开销最小
       - **MonoT5 排序**：使用 pointwise reranker 按与问题的相关性排序
       - **频率排序**：基于 token 的逆频率归一化排序
        - 排序结果在所有 query token、所有层、所有注意力头之间共享
    - 设计动机：
        - PINE 为每个 query token 重新计算排序，导致 $O(n^2d)$ 的额外计算和频繁的位置 ID 变更
        - 全局排序只需排一次，复杂度降至 $O(nk\log k)$，且一致的位置 ID 分配减少了分布偏移

2. **循环排列（Circular Arrangement）**

    - 功能：在因果 LM 中模拟双向注意力，让每个段都能"看到"其他所有段
    - 核心思路：
        - 给定全局排序 A→B→O→K→G，构建有向循环图
        - 当段 B 作为 query 时，按循环顺序将 B 放在最后：O→K→G→A→B
        - 当段 K 作为 query 时：G→A→B→O→K
        - 关键：所有 suffix 和生成 token 使用全局排序的前后部分拼接，**不再逐 token 变化位置 ID**
    - 设计动机：
        - 在因果注意力中，位于序列末尾的 token 能看到前面所有 token
        - 循环排列让每个段轮流处于"最后位置"，实现事实上的双向访问
        - 与 PINE 不同，suffix token 的位置 ID 保持不变，大幅减少 OOD 风险

3. **选择路由（Selective Routing）**

    - 功能：自适应选择使用原始模型还是顺序不变模型的输出
    - 核心思路：
        - 原始模型和 RoToR 模型分别对同一输入生成答案及置信度（最大 token 概率）
        - 如果原始模型置信度 + 偏置 α > RoToR 置信度，选择原始模型的答案；否则选择 RoToR
        - α = 0.2（通过验证集搜索确定），略倾向于原始模型
    - 设计动机：
        - 实际任务（如 MMLU）中部分选项是顺序敏感的（如"None of the above"）
        - 顺序不变模型在这些选项上可能反而更差
        - 基于置信度的路由可以自适应选择最合适的模型

### 计算复杂度分析

| 方法 | 额外计算开销 |
|------|------------|
| PINE | $O(n^2d + nk\log k)$（每个 query 都要重新计算无 RoPE 注意力 + 排序）|
| RoToR (词典排序) | $O(nk\log k)$（单次全局排序）|
| RoToR (基数排序优化) | $O(nk)$ |

## 实验关键数据

### 主实验（Lost in the Middle 基准, best_subspan_em %）

| 方法 | ndoc=10 | ndoc=20 | ndoc=30 |
|------|---------|---------|---------|
| **Llama-3.1-8B-Instruct** | | | |
| Original | 50.2~54.7 | 51.0~54.8 | 43.5~56.8 |
| PCW | 11.9~12.4 | 3.7~4.0 | 1.8~2.3 |
| Set-Based Prompting | 42.5 | 26.3 | 14.1 |
| PINE | 58.6~59.0 | 55.5~56.2 | 53.7~54.8 |
| **RoToR-lexical** | **61.4~61.6** | **59.6~61.4** | **59.0~59.5** |
| RoToR-MonoT5 | 61.2~61.4 | 60.7~61.2 | 60.7~60.9 |
| **Llama-3.1-70B-Instruct** | | | |
| Original | 65.7~66.2 | 64.3~66.2 | — |
| PINE | 67.5~67.9 | 65.5~65.9 | — |
| **RoToR** | **69.3~69.6** | **67.6~67.9** | — |

### KGQA 实验（N=30 段，best_subspan_em %）

| 方法 | Llama-8B Acc. | Llama-70B Acc. | Qwen-4B Acc. | Qwen-7B Acc. |
|------|-------------|---------------|-------------|-------------|
| Original | 50.2 | 61.6 | 30.7 | 31.5 |
| PINE | 51.5 | 63.1 | 31.6 | 32.3 |
| **RoToR** | **53.1** | **63.6** | **32.0** | **34.3** |
| RoToR-MonoT5 | 51.6 | — | 32.3 | 32.9 |

### 关键发现

1. **RoToR 在所有模型和设置下一致优于 PINE**：在 LitM 基准上平均提升 2-5 个百分点
2. **顺序不变性极佳**：打乱段顺序后 RoToR 的标准差极小（0.02-0.11），远优于 Original 模型（0.07-0.75）
3. **简单的词典排序就足够好**：不需要复杂的 MonoT5 排序，RoToR-lexical 已有显著收益
4. **计算开销远低于 PINE**：消除了 $O(n^2d)$ 项，随段数 k 增加优势更明显
5. **PCW 和 Set-Based Prompting 在段数增多时几乎失效**：ndoc=30 时 PCW 仅有 2%
6. **选择路由有效**：在 MMLU 上帮助处理顺序敏感的特殊选项

## 亮点与洞察

- **名字巧妙**：RoToR 是回文结构，呼应"顺序不变性"的主题，同时暗示"旋转"（Rotary）
- **简洁是力量**：全局排序 + 循环排列的方案极其简洁，却在数学上保证了顺序不变性
- **OOD 视角独到**：将位置偏差问题转化为训练-推理分布不匹配问题，并通过最小修改来缓解
- **实验中的细节**：发现 bfloat16 精度下 PINE 的注意力分数会出现大量 tied values，导致排序的非确定性，这是一个重要的实践洞察

## 局限与展望

1. **无法处理完全任意的输入结构**：仍需要明确的段划分
2. **选择路由需要两次前向传播**：在实际部署中增加了推理成本
3. **全局排序不保证最优**：词典排序虽然简单有效，但相关文档放在更近位置可能更好（MonoT5 排序的优势）
4. **大规模实验受限**：70B 模型在 ndoc=30 时因资源限制未能实验
5. **未在 LLM-as-a-judge 等高影响场景中直接验证**

## 相关工作与启发

- PINE 是最直接的前驱工作，RoToR 通过全局排序消除了其逐 query 排序的核心缺陷
- 与 Set/Graph ML 中的顺序不变性方法（Murphy et al., 2019）有概念联系，但循环分配和预训练 LM 上的应用是新贡献
- 选择路由的思想可推广到其他"需要异构处理"的场景（如 RAG 中是否使用检索结果的决策）
- 对 RoPE 和因果注意力的改造技术可启发其他需要修改注意力机制的工作

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 循环排列 + 全局排序的组合方案简洁新颖，将 OOD 视角引入位置偏差分析是独特贡献
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 LitM / KGQA / MMLU 三大类任务，5 种模型规模，多种排序算法变体，包括方差和时间分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示生动，与 PINE 的对比非常直观
- **价值**: ⭐⭐⭐⭐ — 解决了 decoder-only LM 中一个持久且实际的问题，方法简洁实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Enhancing Transformers for Generalizable First-Order Logical Entailment](enhancing_fol_entailment.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[CVPR 2025\] Sufficient Invariant Learning for Distribution Shift](../../CVPR2025/others/sufficient_invariant_learning_for_distribution_shift.md)

</div>

<!-- RELATED:END -->
