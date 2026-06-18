---
title: >-
  [论文解读] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation
description: >-
  [NeurIPS 2025][信息检索/RAG][多模态RAG] 提出Windsock+DANCE双组件框架解决多模态RAG的三个核心问题：Windsock模块根据查询自适应决定**何时检索**和**检索什么模态**（文本/图像/不检索），DANCE指令微调策略通过动态选择模型薄弱模态进行噪声鲁棒训练来提升**如何利用**检索信息的能力，整体性能提升17.07%同时减少8.95%检索次数。
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "多模态RAG"
  - "自适应检索"
  - "模态选择"
  - "噪声鲁棒训练"
  - "检索增强生成"
---

# Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.22694](https://arxiv.org/abs/2510.22694)  
**代码**: 暂无  
**领域**: 信息检索  
**关键词**: 多模态RAG, 自适应检索, 模态选择, 噪声鲁棒训练, 检索增强生成

## 一句话总结

提出Windsock+DANCE双组件框架解决多模态RAG的三个核心问题：Windsock模块根据查询自适应决定**何时检索**和**检索什么模态**（文本/图像/不检索），DANCE指令微调策略通过动态选择模型薄弱模态进行噪声鲁棒训练来提升**如何利用**检索信息的能力，整体性能提升17.07%同时减少8.95%检索次数。

## 研究背景与动机

多模态检索增强生成（MRAG）通过引入外部知识提升MLLM的事实性和时效性，但现有方法存在三大关键缺陷：

**When问题——盲目检索**：现有方法对所有查询一律进行检索（retrieve-for-all策略），即使模型的参数化知识已足以回答。这带来不必要的计算开销，且不可靠的检索器返回的噪声信息反而会降低回答质量。

**What问题——模态选择僵硬**：现有方法要么固定检索图像，要么只检索文本（如Wikipedia），不考虑不同查询对信息类型的不同需求。实际上，历史事件更需要文本检索，而绘画风格问题更需要图像检索——不同查询天然需要不同模态的信息。

**How问题——检索信息利用不足**：MLLM对检索到的无关文档敏感，无论是统计型（BM25）还是向量型（CLIP）检索器都可能返回不相关内容，导致事实错误和幻觉。模型需要学会"取其精华去其糟粕"。

现有工作（如ReflectiVA、mR2AG）虽尝试引入自适应检索，但依赖**昂贵的人工或GPT-4标注**，且忽略了多模态选择。Windsock通过**自评估**方式自动构建训练数据，且同时解决when和what两个问题。

## 方法详解

### 整体框架

三个核心组件构成完整pipeline：
1. **Windsock模块**（轻量分类器）：接收用户查询，输出三分类决策——不检索(NA)/检索图像(Visual)/检索文本(Textual)
2. **检索器**：根据Windsock决策，从相应模态知识库检索top-k文档
3. **MLLM生成器**（经DANCE训练）：整合查询和检索文档生成最终回答

### 关键设计

1. **Windsock——查询感知的自适应检索决策器**：
   
   基于Flan-T5-Small骨干实现三分类映射：
    $c = \mathcal{W}(Q) \in \{\text{NA}, \text{Visual}, \text{Textual}\}$
   
   根据分类结果执行对应策略：
    $\begin{cases} r^\varnothing = \mathcal{G}(Q, \varnothing), & \text{if } c = \text{NA} \\ r^V = \mathcal{G}(Q, \mathcal{R}(Q, \mathbb{D}^I)), & \text{if } c = \text{Visual} \\ r^T = \mathcal{G}(Q, \mathcal{R}(Q, \mathbb{D}^T)), & \text{if } c = \text{Textual} \end{cases}$
   
   设计优点：（a）跳过不必要的检索减少开销和噪声引入；（b）选择最合适的模态提升信息质量；（c）模块化设计可**即插即用**到任意MLLM。实验显示Windsock仅增加10.25ms（1.83%）的推理开销。

2. **自评估训练数据构建**（无需GPT-4标注）：
   
   对每个QA对 $\{Q, A\}$，用MLLM分别以三种策略生成回答（不检索/检索图像/检索文本），然后用下游任务评估指标（如F1）评分：
    $s^\varnothing = \epsilon(r^\varnothing, A), \quad s^I = \epsilon(r^I, A), \quad s^T = \epsilon(r^T, A)$
    $c^* = \arg\max_c(s^\varnothing, s^I, s^T)$
   
   最优策略 $c^*$ 作为Windsock的训练标签。这种方法直接利用MLLM自身能力评估不同策略的效果，无需外部标注。还能发现检索有害的情况（当 $s^\varnothing > \max(s^I, s^T)$ 时）。

3. **DANCE——动态噪声抵抗指令微调**：
   
   核心思路：不是随机注入噪声，而是智能识别模型**最薄弱的模态**进行针对性训练。具体地，对每个样本选择MLLM表现最差的模态：
    $\arg\min_M (s^I, s^T) \in \{I, T\}$
   
   该模态的检索结果大概率包含噪声/无关信息。用这些"困难样本"构建指令微调数据：$\{Q, \mathcal{R}(Q, \mathbb{D}^M), A\}$，然后用标准指令微调训练模型学会在噪声中提取有用信息。

   与SURf对比：SURf用图像相似度做硬样本挖掘且需逐文档生成回答，DANCE通过下游指标高效识别困难案例且不同模态可并行处理，数据构建速度快2倍。

### 损失函数 / 训练策略

- **Windsock训练**：使用AdamW优化器（lr=5e-4），batch size 16，5 epochs，带类别权重的交叉熵损失平衡训练数据不均衡
- **DANCE指令微调**：使用LLaMA-Factory框架的默认LoRA配置，1 epoch训练
- **检索器**：VBGE-base，返回top-3检索结果

## 实验关键数据

### 主实验——WebQA F1分数

| 方法 | 生成器 | Single | Multiple | All |
|------|------|:---:|:---:|:---:|
| Zero-Shot | Qwen2-VL-7B | 61.76 | 37.09 | 44.04 |
| Vanilla RAG | Qwen2-VL-7B | 62.96 | 38.36 | 45.29 |
| Windsock only | Qwen2-VL-7B | 65.92 | 38.63 | 46.32 |
| SURf | Qwen2-VL-7B-SURf | 62.72 | 55.60 | 57.61 |
| **DANCE** | Qwen2-VL-7B-DANCE | 66.42 | 57.45 | 59.97 |
| **Windsock+DANCE** | Qwen2-VL-7B-DANCE | **70.12** | **59.32** | **62.36** |

### 消融——检索策略效率分析

| 检索策略 | 时间(s)↓ | Single↑ | Multiple↑ | All↑ |
|------|:---:|:---:|:---:|:---:|
| NA（不检索） | 0.46 | 61.76 | 37.09 | 44.04 |
| Visual only | 0.67 | 64.88 | 36.46 | 44.47 |
| Textual only | 0.79 | 52.87 | 36.70 | 41.25 |
| **Windsock** | **0.56** | **65.92** | **38.63** | **46.32** |

### 消融——指令微调策略对比

| 策略 | Single | Multiple | ALL |
|------|:---:|:---:|:---:|
| Easy（选最好模态训练） | 58.83 | 51.24 | 53.38 |
| Random（随机选模态） | 60.98 | 53.94 | 55.12 |
| **DANCE（选最差模态）** | **66.42** | **57.45** | **59.97** |

### 关键发现

- **自适应检索比固定策略全面优胜**：Windsock比纯Visual或纯Textual检索都好，且推理时间介于两者之间。在WebQA上跳过8.96%的查询检索；加入简单MS-COCO查询后跳过率升至26.99%——体现了良好的适应性
- **"在薄弱处训练"是DANCE的关键**：选择模型最差模态进行训练（DANCE）比选最好模态（Easy）高6.59%，比随机选（Random）高4.85%——针对性训练远优于随机训练
- **Windsock+DANCE互补增益**：Windsock解决输入端问题（给什么信息），DANCE解决模型端问题（怎么用信息），结合后提升最大
- **DANCE的副作用**：在通用MLLM基准MME上性能有所下降，存在专用与通用能力的权衡

## 亮点与洞察

1. **问题分解的清晰性**：将MRAG问题清晰分解为when/what/how三个维度，每个维度有针对性的解决方案
2. **自评估替代GPT-4标注**是实用且经济的思路：直接用目标模型自身评估不同策略效果，无需昂贵的外部标注器。数据构建效率是SURf的2倍
3. **"在失败处学习"的训练哲学**：DANCE选择模型最弱模态进行训练，类似课程学习的思路但方向相反（硬样本优先），实验证明这种反直觉的策略效果最好
4. **Windsock的极轻量设计**：使用Flan-T5-Small作为骨干，仅增加1.83%推理开销就实现了显著的效率和性能提升

## 局限与展望

- 当前仅支持文本和图像两种模态检索，未支持表格等其他模态
- DANCE训练在通用MLLM基准（MME）上存在性能下降，专用化和通用化之间的平衡需要进一步探索
- Windsock使用纯文本骨干（Flan-T5），未利用查询中可能包含的图像信息
- 自评估方法的训练数据质量依赖于基础MLLM的回答质量和检索器的检索质量
- 三分类方式（NA/Visual/Textual）可能过于粗粒度，未支持混合检索

## 相关工作与启发

- **Self-RAG的启发**：Self-RAG使用特殊token决定是否检索，Windsock将这一思路扩展到多模态并增加了模态选择维度
- **SURf的对比**：SURf通过指令微调增强模型的信息选择利用能力，但使用图像相似度挖掘困难样本。DANCE使用下游指标直接评估，更高效且更任务对齐
- **对MRAG系统设计的启示**：未来的MRAG系统应将"是否检索"和"检索什么"作为可学习的决策而非固定策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ When+What+How的三维分解和自评估数据构建方法新颖实用
- **实验充分度**: ⭐⭐⭐⭐ WebQA和MultimodalQA两个数据集，多个基线对比，丰富消融分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，实验分析全面，可视化丰富
- **价值**: ⭐⭐⭐⭐ 对MRAG系统的实际部署有直接指导意义，自评估流水线开箱即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](../../ACL2025/information_retrieval/towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](../../ACL2025/information_retrieval/seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](../../ACL2025/information_retrieval/accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)

</div>

<!-- RELATED:END -->
