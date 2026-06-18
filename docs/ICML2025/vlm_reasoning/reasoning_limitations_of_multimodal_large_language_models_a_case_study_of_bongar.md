---
title: >-
  [论文解读] Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems
description: >-
  [ICML 2025][多模态VLM][Bongard Problems] 系统评估4个闭源+4个开源MLLM在经典合成Bongard Problems、Bongard HOI、Bongard-OpenWorld三个数据集上的抽象视觉推理能力，提出7种解题策略和新数据集Bongard-RWR（用真实图像表达合成BP概念），揭示MLLM在合成BP上的极差表现并非因域差异而是固有的抽象推理局限。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "Bongard Problems"
  - "abstract visual reasoning"
  - "MLLM evaluation"
  - "Bongard-RWR"
  - "few-shot concept learning"
---

# Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems

**会议**: ICML 2025  
**arXiv**: [2411.01173](https://arxiv.org/abs/2411.01173)  
**代码**: [GitHub](https://github.com/pavonism/bongard-rwr)  
**领域**: 多模态推理评估 / 抽象视觉推理  
**关键词**: Bongard Problems, abstract visual reasoning, MLLM evaluation, Bongard-RWR, few-shot concept learning

## 一句话总结

系统评估4个闭源+4个开源MLLM在经典合成Bongard Problems、Bongard HOI、Bongard-OpenWorld三个数据集上的抽象视觉推理能力，提出7种解题策略和新数据集Bongard-RWR（用真实图像表达合成BP概念），揭示MLLM在合成BP上的极差表现并非因域差异而是固有的抽象推理局限。

## 研究背景与动机

**领域现状**：Bongard Problems (BPs)是经典的抽象视觉推理任务——给定左右各6张图像，要求发现区分两侧的共同概念并用自然语言描述。这要求联合感知与推理，是评估AI抽象能力的标杆测试。

**现有痛点**：

1. 传统深度学习方法将BP简化为二分类任务，回避了自然语言答案生成的挑战

2. 现有包含真实世界图像的BP数据集（Bongard HOI/OpenWorld）与经典合成BP的概念不同，无法直接对比域差异的影响

3. MLLM的抽象推理能力尚未被系统评估——是受限于合成图像的域外特性，还是推理能力本身不足？

**核心问题**：MLLM在合成BP上表现差，是因为训练数据的域差异还是固有的抽象推理缺陷？

## 方法详解

### 整体框架

设计7种适配MLLM的BP解题策略（3种直接生成 + 4种分步推理），在4个BP数据集上用8个MLLM进行评估，并通过新建的Bongard-RWR数据集控制域差异变量。

### 关键设计

1. **7种解题策略**

    - **Direct**：直接输入完整BP矩阵图像，要求模型一次性输出答案
    - **Descriptive**：逐张描述每个面板，再基于文本描述推理答案
    - **Descriptive-iterative**：在同一上下文窗口中迭代地描述同侧图像，逐步精化
    - **Descriptive-direct**：Descriptive + 附加完整矩阵图像辅助推理
    - **Contrastive**：逐对比较左右两侧对应图像的差异
    - **Contrastive-iterative**：在同一上下文中迭代对比，逐步积累理解
    - **Contrastive-direct**：Contrastive + 附加完整矩阵图像
    - 关键发现：Descriptive策略在所有数据集上最优，Contrastive反而更差

2. **Bongard-RWR 数据集构建**

    - 目标：用真实世界图像表达合成BP的相同抽象概念（如"凸 vs 非凸"）
    - 流程：GPT-4o生成10种真实世界文本描述 → Pexels API搜索图像 → GPT-4o筛选 → 人工调整
    - 最终60个问题：12个全自动生成，24个半自动+人工调整，24个完全人工构建
    - 提供4个变体：原始、RWR-S（正方形裁剪）、RWR-G（灰度）、RWR-SG（正方形灰度）

3. **自动化答案评估**

    - 使用4个闭源MLLM组成的集成来判断生成答案与标准答案是否描述相同概念
    - 至少2个模型同意则判为正确
    - 避免了自然语言答案的多样性导致的评估困难

### 评估设置

- 闭源模型：GPT-4o, GPT-4 Turbo, Gemini 1.5 Pro, Claude 3.5 Sonnet
- 开源模型：InternVL2-8B, LLaVA-1.6 Mistral-7B, Phi-3.5-Vision, Pixtral 12B
- 数据集：100个合成BP + 100个Bongard HOI + 100个Bongard-OpenWorld + 60个Bongard-RWR
- 人类基线：Bongard-RWR上平均正确39.2/60（65%）

## 实验关键数据

### 主实验

**自然语言生成的正确数（数据集总数见列标题）**

| 模型 | 合成BP /100 | HOI /100 | OpenWorld /100 | RWR /60 |
|------|-----------|---------|---------------|---------|
| GPT-4o (Descriptive) | 17 | **42** | **46** | 8 (13.3%) |
| GPT-4 Turbo (Desc.) | 15 | **45** | **57** | 5 (8.3%) |
| Claude 3.5 Sonnet (Desc.) | 19 | 44 | 53 | **13 (21.7%)** |
| Gemini 1.5 Pro (Desc.) | **21** | 40 | 32 | 7 (11.7%) |
| Pixtral 12B (Desc.) | 4 | 27 | 34 | 1 (1.7%) |
| InternVL2-8B | 0 | 2 | 18 | 0 (0%) |
| 人类基线 | - | - | - | 39.2 (65%) |

### 消融实验

**策略对比（所有模型合计解出的不重复问题数）**

| 数据集 | Descriptive系 | Contrastive系 |
|--------|-------------|--------------|
| 合成 BP | 44 | 44 |
| Bongard HOI | **82** | 63 |
| Bongard-OpenWorld | **90** | 76 |
| Bongard-RWR | **20** | 11 |

### 关键发现

- **所有MLLM在合成BP上表现极差**：最佳模型（Gemini 1.5 Pro, Descriptive）仅解出21/100
- **Bongard-RWR证实域差异非主因**：GPT-4o在真实世界HOI上解42题但在真实世界RWR上仅8题，因为RWR保留了合成BP的抽象概念
- **Descriptive >> Contrastive**：MLLM更擅长逐张描述再综合推理，而非直接对比——这与人类的推理方式相反
- **闭源 >> 开源**：闭源模型在35/40个(数据集,策略)组合中领先
- **迭代推理反而变差**：Descriptive-iterative比Descriptive更差，说明当前模型难以有效利用上下文窗口中的历史信息
- 人类在Bongard-RWR上65%正确率，最佳MLLM仅21.7%

## 亮点与洞察

- 通过Bongard-RWR巧妙地控制了"概念相同、域不同"的对比变量，证实了MLLM的抽象推理缺陷是固有的
- 7种解题策略的系统对比揭示了MLLM的推理模式偏好：更适合文本描述→综合推理的pipeline
- "对比推理反而更差"这一发现揭示了MLLM与人类在类比推理上的根本差异
- 数据集变体（正方形/灰度）的消融显示去除颜色和裁剪空白能提升分类准确率

## 局限与展望

- Bongard-RWR仅60个问题，规模较小
- 自动评估依赖MLLM集成判断语义等价，可能引入偏差
- 未评估微调后的MLLM（如在部分BP上微调能否改善推理）
- 8个模型虽然覆盖主流但缺少近期更强模型（如GPT-4o-mini、Claude 3.5 Opus）
- 仅考虑零样本设定，未探索few-shot in-context learning

## 相关工作与启发

- **vs Bongard-LOGO**：Bongard-LOGO用合成数据做二分类，本文用MLLM做自然语言答案生成，更贴近BP的原始定义
- **vs Wüst et al. (2024)**：同期工作也评估MLLM在合成BP上的表现，本文额外引入Bongard-RWR进行域差异控制实验
- **vs ARC/RPM**：这些任务在文本化后可被LLM解决，但BP需要联合视觉感知和抽象推理
- 启发：当前MLLM的"推理"能力可能更多是模式匹配而非真正的抽象概念形成

## 评分

- 新颖性: ⭐⭐⭐⭐ Bongard-RWR数据集和7种策略的系统设计是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 8个模型×4个数据集×7种策略的大规模评估
- 写作质量: ⭐⭐⭐⭐ 实验设计严谨，分析深入
- 价值: ⭐⭐⭐⭐ 揭示MLLM抽象推理的固有局限，对未来模型设计有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](../../ICLR2026/multimodal_vlm/bongard-rwr_real-world_representations_of_fine-grained_concepts_in_bongard_probl.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](../../ACL2025/multimodal_vlm/mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[NeurIPS 2025\] Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning](../../NeurIPS2025/multimodal_vlm/can_llms_reason_over_non-text_modalities_in_a_training-free_manner_a_case_study_.md)

</div>

<!-- RELATED:END -->
