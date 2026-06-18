---
title: >-
  [论文解读] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval
description: >-
  [NeurIPS 2025][多模态VLM][多模态检索] 提出首个R1风格的推理型多模态检索框架Retrv-R1，通过信息压缩模块降低token消耗、细节检查机制保留困难候选的完整信息、课程式RL奖励兼顾效果与效率，在通用多模态检索benchmark上实现SOTA。 通用多模态检索要求单一模型处理文本到图像、图像到文本、组…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态检索"
  - "强化学习"
  - "推理MLLM"
  - "信息压缩"
  - "DeepSeek-R1"
---

# Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval

**会议**: NeurIPS 2025  
**arXiv**: [2510.02745](https://arxiv.org/abs/2510.02745)  
**代码**: 暂无  
**领域**: 多模态VLM  
**关键词**: 多模态检索, 强化学习, 推理MLLM, 信息压缩, DeepSeek-R1

## 一句话总结

提出首个R1风格的推理型多模态检索框架Retrv-R1，通过信息压缩模块降低token消耗、细节检查机制保留困难候选的完整信息、课程式RL奖励兼顾效果与效率，在通用多模态检索benchmark上实现SOTA。

## 研究背景与动机

通用多模态检索要求单一模型处理文本到图像、图像到文本、组合查询等多种检索任务。现有方法分两类：(1) 基于MLLM嵌入的相似度匹配，精度和鲁棒性不足；(2) 将检索转化为QA任务（如LamRA），利用MLLM直接生成检索结果，但**缺乏显式推理过程**，面对复杂案例能力有限。

DeepSeek-R1展示了RL训练CoT推理的巨大潜力，启发了一个自然的问题：**R1范式能否提升检索MLLM的能力？** 然而直接应用GRPO到检索任务面临两大障碍：

**高计算成本**：多候选输入+CoT推理产生大量token，可能超出上下文长度

**训练不稳定**：直接RL训练收敛困难，模型常生成错误推理链，导致次优结果

本文从模型结构（信息压缩）和训练策略（激活+增强两阶段）两方面系统性地解决上述问题。

## 方法详解

### 整体框架

Retrv-R1采用两阶段检索流水线：第一阶段用嵌入模型 $\phi$ 粗筛top-K候选；第二阶段用推理MLLM $\theta$（带信息压缩模块ICM）从K个候选中精选最佳匹配。核心创新集中在第二阶段模型的结构设计和训练策略。

### 关键设计

1. **信息压缩模块（Information Compression Module, ICM）**

   核心思想是将每个候选的token序列压缩为2个token，为CoT推理腾出上下文空间。ICM放置在MLLM的语言模型之前，对每个候选 $c_k$ 生成两种压缩token：

    - **内容token** $t_{con}^{c_k}$：用可学习嵌入 $e_{con}$ 作为query，候选token序列 $T_{c_k}$ 作为key/value，通过两层注意力压缩：
    $t_{con}^{c_k} = \text{ATT}_1(\mathbf{Q}_{e_{con}}, \mathbf{K}_{T_{c_k}}, \mathbf{V}_{T_{c_k}})$

    - **关系token** $t_{rel}^{c_k}$：先用候选token对查询token做交叉注意力建模关系特征 $R_{q,c_k}$，再压缩为单token：
    $R_{q,c_k} = \text{ATT}_2(\mathbf{Q}_{T_{c_k}}, \mathbf{K}_{T_q}, \mathbf{V}_{T_q})$

   ICM通过自对齐策略（Self-Alignment）预训练：冻结LM，用交叉熵损失约束压缩token的LM输出与原始完整token的LM输出一致，确保压缩后保留检索关键信息。

2. **细节检查机制（Details Inspection Mechanism, DIM）**

   尽管压缩token对大多数候选足够，但困难候选需要完整信息。DIM引入两个特殊token `<inspection-index-start>` 和 `<inspection-index-end>`，让MLLM在CoT过程中**自主决定**哪些候选需要"仔细检查"，自动取回其完整token序列作为补充。这使得模型能在效率和精度之间自适应权衡。

3. **课程式RL训练（Curriculum Efficiency Constraint）**

   训练分三阶段：(a) ICM预训练（自对齐）；(b) SFT激活推理能力——用Qwen2.5-VL-72B合成包含四阶段结构的CoT标注数据（猜测→快速排否→精细验证→生成结果）；(c) GRPO强化学习增强推理。

   RL的奖励函数由格式奖励 $r_f$ 和结果-效率奖励 $r_r$ 组成：

    $r_r = \mathbb{1}(\hat{c} = \hat{c}_{gt})(1 - \lambda \frac{N_{ins}}{K})$

   其中 $N_{ins}$ 是检查完整token的候选数，$K$ 是总候选数。关键创新是**课程式调度** $\lambda_i = i / N_{iter}$：训练初期允许模型多用完整信息，后期逐步增强效率约束。

### 损失函数 / 训练策略

- 预训练阶段：交叉熵自对齐损失
- SFT阶段：合成CoT数据上的交叉熵损失，ICM和MLLM联合训练
- RL阶段：GRPO目标 + 格式奖励 + 课程式结果-效率奖励，在10K难样本上训练
- 视觉编码器全程冻结，语言模型用LoRA微调

## 实验关键数据

### 主实验：M-BEIR测试集（16个子任务平均）

| 方法 | 参数量 | 平均Recall |
|------|--------|-----------|
| CLIP-L | - | 32.5 |
| UniIR-CLIP | - | 50.6 |
| MM-Embed-7B | 7B | 52.7 |
| LamRA-7B | 7B | 63.7 |
| **Retrv-R1-3B** | **3B** | **65.5** |
| **Retrv-R1-7B** | **7B** | **69.2** |

注：3B模型已超越所有7B方法。

### 效率对比（CIRR任务，K=50）

| 方法 | R@5 | 推理时间比 | 显存比 |
|------|-----|-----------|--------|
| Qwen2.5-VL-7B | 55.1 | 4.79x | 2.44x |
| Vision-R1-7B | 57.7 | 7.23x | 3.28x |
| LamRA-Rank-L-7B | 66.2 | 4.98x | 2.46x |
| **Retrv-R1-7B** | **72.3** | **1.00x** | **1.00x** |

### 消融实验

| 配置 | CIRR R@5 | OVEN R@5 | 说明 |
|------|----------|----------|------|
| Retrv-R1-3B（完整） | 67.9 | 83.3 | 基准 |
| w/o ICM | 68.8 | 84.4 | 性能微升但时间7.4x |
| w/o 内容token $t_{con}$ | 60.7 | 77.6 | 下降7.2 |
| w/o 关系token $t_{rel}$ | 64.7 | 80.7 | 下降3.2 |
| w/o 自对齐预训练 | 64.3 | 80.7 | 下降3.6 |
| w/o 细节检查(DIM) | 62.3 | 78.1 | 下降5.6 |
| w/o SFT阶段 | 63.0 | 79.4 | 下降4.9 |
| w/o RL阶段 | 61.1 | 78.4 | 下降6.8 |

### 关键发现

- 课程式效率约束优于任何固定 $\lambda$，固定 $\lambda=0.5$ 时R@5下降1.3%
- 泛化性强：在未见过的数据集（CIRCO、GeneCIS等）和未见过的任务类型上均大幅领先
- 在多模态推荐（Amazon Review）上微调后同样达到SOTA

## 亮点与洞察

- **R1推理范式首次成功应用于检索**：证明了CoT推理对检索任务的价值，不是简单的嵌入匹配
- 信息压缩模块的设计很精巧：内容token + 关系token两路压缩，分别捕获候选自身信息和与query的关联
- 细节检查机制赋予模型**自主决定精度-效率权衡**的能力，是一种优雅的自适应设计
- 课程式RL奖励的设计符合直觉：模型先学会做对，再学会做快

## 局限与展望

- 仍依赖第一阶段嵌入模型的召回质量，端到端优化可进一步提升
- ICM的两层注意力结构虽然简单但可能在极端情况下信息损失过大
- 合成CoT数据依赖72B教师模型，成本较高
- 当前K=50，更大K值下的扩展性有待验证

## 相关工作与启发

- LamRA首先将检索形式化为MLLM的QA任务，Retrv-R1在此基础上引入推理
- DeepSeek-R1的GRPO和Vision-R1对MLLM推理的探索是本文的直接灵感来源
- BLIP-2的预训练对齐思路被借鉴用于ICM的自对齐
- 启发：R1风格推理可推广到排序、推荐等更多信息检索子任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个R1风格检索MLLM，ICM和DIM设计原创性强
- **实验充分度**: ⭐⭐⭐⭐⭐ M-BEIR全任务+泛化性+效率+消融+推荐任务
- **写作质量**: ⭐⭐⭐⭐ 结构完整，动机清晰
- **价值**: ⭐⭐⭐⭐⭐ 为多模态检索开辟了推理增强的新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ACL 2025\] MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](../../ACL2025/multimodal_vlm/megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](../../ICLR2026/multimodal_vlm/shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[NeurIPS 2025\] ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism](elasticmm_efficient_multimodal_llms_serving_with_elastic_multimodal_parallelism.md)

</div>

<!-- RELATED:END -->
