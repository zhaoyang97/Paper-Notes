---
title: >-
  [论文解读] Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories
description: >-
  [NEURIPS2025][LLM评测][属性泛化] 本文首次系统评估了属性预测任务中训练/测试划分策略对泛化性能的影响,提出了基于 LLM 语义分组、嵌入相似度、嵌入聚类和超类标签的四种渐进式难度划分方案,发现无监督聚类划分在不依赖标注的情况下实现了与真值超类划分相当的去泄漏效果,同时保留了更好的预测性能。
tags:
  - "NEURIPS2025"
  - "LLM评测"
  - "属性泛化"
  - "训练/测试划分"
  - "语义泄漏"
  - "聚类"
  - "线性探测"
  - "视觉表征"
---

# Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories

**会议**: NEURIPS2025  
**arXiv**: [2509.06998](https://arxiv.org/abs/2509.06998)  
**代码**: 待确认  
**领域**: LLM评测  
**关键词**: 属性泛化, 训练/测试划分, 语义泄漏, 聚类, 线性探测, 视觉表征  

## 一句话总结

本文首次系统评估了属性预测任务中训练/测试划分策略对泛化性能的影响,提出了基于 LLM 语义分组、嵌入相似度、嵌入聚类和超类标签的四种渐进式难度划分方案,发现无监督聚类划分在不依赖标注的情况下实现了与真值超类划分相当的去泄漏效果,同时保留了更好的预测性能。

## 研究背景与动机

**属性的跨类别能力**: 属性（如"有四条腿"、"有条纹"）是人类描述物体的核心方式,具有天然的跨类别迁移潜力——"有条纹"可以从斑马学到并迁移至蜜蜂和老虎。

**现有数据集的缺陷**: 当前属性预测基准要么在分类学上范围狭窄（如 AwA 只有动物、CUB 只有鸟类），要么未控制训练/测试集的相异度（如 VAW、MIT States），使模型可以利用分类学捷径（semantic leakage）而非真正学会属性抽象。

**语义泄漏问题**: 当训练集和测试集包含语义相近的类别时（如训练有"狗"测试有"狼"），模型可能通过识别类别而非理解属性来获得高分,导致对泛化能力的评估产生偏差。

**缺乏系统评估**: 此前没有工作明确控制训练/测试集之间概念的语义和感知距离,来评估属性泛化的真实水平。

**核心研究问题**: 模型能否将从一组类别学到的属性知识泛化到语义和感知上完全不相关的类别？例如从"狗"上学到的"有四条腿"能否迁移到"椅子"？

**基准构建需求**: 需要为属性推理任务构建更公平、更具挑战性的评测协议,以推动表征学习的进步。

## 方法详解

### 整体框架

本文提出了一套渐进难度的训练/测试划分策略。给定一组概念（如猫、草莓、椅子），每个概念标注了二值属性,目标是评估预训练视觉嵌入中是否编码了这些属性。评估方法是对每个属性训练一个线性分类器（linear probe），在训练概念上训练、测试概念上评估。关键创新在于如何划分训练/测试集：通过不同方法将相似概念分组,确保相似概念在同一划分中,以控制跨划分的语义泄漏程度。

### 模块一：LLM 语义分组 (A. LLM-based)

- **功能**: 使用 ChatGPT-4o 识别语义高度相似的概念对（如 cup 和 mug），将这些高相似度对共同分配到训练集
- **核心思路**: 利用 LLM 的世界知识进行启发式语义去重,避免训练集和测试集之间出现直接的语义重叠
- **设计动机**: LLM 能捕捉人类级别的语义相似性判断,但其覆盖率有限——仅能识别 12% 概念的分组关系,大量未分配的概念仍可能跨越划分边界产生泄漏

### 模块二：嵌入相似度阈值划分 (B. Embeddings Similarity)

- **功能**: 计算概念嵌入之间的余弦相似度,将与其他概念最大相似度最高的 top 概念分配到训练集
- **核心思路**: 将语义密集区域集中在训练集中,最小化训练/测试边界上的高相似度对
- **设计动机**: 基于数据驱动而非人工判断,但只根据排名前 600 的相似度组构建分组,约 60% 的样本未被分配,覆盖率不足导致去泄漏效果有限

### 模块三：嵌入聚类划分 (C. Embeddings Clustering)

- **功能**: 对概念嵌入执行 K-Means 聚类,整个聚类被完整地分配到训练集或测试集
- **核心思路**: 聚类确保全覆盖（每个概念都属于某个聚类），同时以适中的粒度控制跨划分的语义重叠。选择 k=100 在 F1 和相关性之间取得最优平衡
- **设计动机**: 解决了前两种方法覆盖率不足的根本问题。完全无监督,不需要地面真值标签,同时在 CS（与超类的相关性）指标上达到接近真值划分的去泄漏水平

### 模块四：真值超类划分 (GT: Supercategory Labels)

- **功能**: 基于 THINGSplus 的 53 个人工标注超类将概念分组,每个超类完整地分配到训练集或测试集
- **核心思路**: 作为严格的上界控制——完全消除已知分类学边界的泄漏
- **设计动机**: 虽然去泄漏最彻底(CS≈0.06),但组过大导致某些属性完全集中在单一超类中（如 "has_4_legs" 集中在哺乳动物），使任务变得不可学习,性能下降严重

### 评测框架

- **线性探测**: 使用 scikit-learn 的 LogisticRegression（balanced class weights, 无正则化, max 1000 iterations）,共 211 个二分类任务
- **评估指标一 F1 selectivity**: F1 得分与随机基线的差值,衡量属性预测的有效性
- **评估指标二 CS (Correlation with Supercategory)**: 每个属性的 F1 selectivity 与其超类优势度的 Pearson 相关,衡量模型是否依赖分类学捷径

## 实验关键数据

### 表 1：不同划分策略下的 F1 Selectivity (↑)

| 视觉模型 | Random | A. LLM | B. Similarity | C. Clustering | GT: Supercategory |
|---------|--------|--------|--------------|--------------|-------------------|
| SigLIP | 45.0 | 43.7 | 42.8 | 39.9 | 32.1 |
| CLIP | 43.6 | 42.0 | 40.9 | 38.6 | 33.2 |
| Swin-V2 | 43.2 | 42.0 | 39.2 | 34.3 | 25.1 |
| DINOv3 | 40.0 | 38.2 | 36.9 | 34.3 | 27.1 |

### 表 2：不同划分策略下的 CS (Correlation with Supercategory, ↓)

| 视觉模型 | Random | A. LLM | B. Similarity | C. Clustering | GT: Supercategory |
|---------|--------|--------|--------------|--------------|-------------------|
| SigLIP | 0.36 | 0.35 | 0.36 | 0.12 | 0.01 |
| CLIP | 0.39 | 0.40 | 0.42 | 0.19 | 0.04 |
| Swin-V2 | 0.36 | 0.35 | 0.32 | 0.02 | -0.14 |
| DINOv3 | 0.37 | 0.35 | 0.36 | 0.14 | 0.03 |
| 平均 | 0.37±0.01 | 0.36±0.03 | 0.36±0.04 | 0.12±0.07 | 0.06±0.08 |

### 关键发现

1. **性能随泄漏减少而骤降**: 从 Random 到 GT Supercategory,SigLIP 的 F1 selectivity 从 45.0 降至 32.1（下降 28.7%），Swin-V2 从 43.2 降至 25.1（下降 41.9%），说明现有模型严重依赖分类学捷径
2. **聚类是最优折衷**: Clustering 在 CS 上达到 0.12±0.07（接近 GT 的 0.06±0.08），同时 F1 显著高于 GT（如 SigLIP: 39.9 vs 32.1），证明无监督聚类在去泄漏和可学习性之间取得了最佳平衡
3. **LLM 和相似度方法效果有限**: A 和 B 的 CS 与 Random 几乎相同（~0.36），低覆盖率导致大量概念未被控制,泄漏未被真正减少
4. **纯视觉模型与视觉-语言模型差异**: Swin-V2/DINOv3 在困难划分下性能下降更剧烈,而 CLIP/SigLIP 因语言监督信号提供了更抽象的属性表征,泛化能力更强
5. **k=100 的 K-Means 最优**: 在 k∈[10,400] 的消融实验中,k=100 达到最高 F1 selectivity 同时保持低 CS

## 亮点与洞察

1. **首次系统化评测属性泛化**: 此前没有工作明确控制训练/测试划分的语义距离来评估属性预测,本文填补了这一空白
2. **揭示了"虚高"的评测问题**: Random split 下的高性能很大程度上是分类学捷径而非真正的属性理解,这对属性预测和零样本学习领域的评测范式提出了根本性质疑
3. **无监督聚类的工程优雅性**: 不需要任何人工标注或 LLM 调用,仅用 K-Means 就能达到接近 GT 标签的去泄漏效果,具有高度可扩展性
4. **对表征学习的深层启示**: 当前视觉嵌入更多编码的是"什么类别"而非"什么属性",属性的跨类别抽象仍然是未解决的挑战

## 局限性

1. **数据集规模有限**: 仅在 McRae×THINGS 一个数据集上验证,包含 1,854 个概念和 277 个属性（过滤后 211 个），代表性可能不足
2. **仅使用线性探测**: 线性分类器只能检测线性可分的属性信息,无法评估嵌入中可能存在的非线性属性编码
3. **聚类依赖嵌入质量**: 嵌入聚类方法的效果依赖于用于聚类的嵌入模型（Swin-V2），不同嵌入可能产生不同的分组和泄漏控制效果
4. **缺乏下游任务验证**: 仅在探测任务上评估,未验证结论是否迁移到零样本学习、组合泛化等实际下游任务
5. **属性粒度单一**: 所有属性均为二值标签,未考虑连续属性或程度差异（如"非常圆"vs"稍微圆"）
6. **未探索改进方向**: 只诊断了问题但未提出如何训练出更好泛化能力的表征

## 相关工作与启发

- **属性预测与零样本学习**: Lampert et al. 的 AwA 数据集开创了基于属性的零样本分类,但仅限于动物类别;Farhadi et al. 提出以丰富属性描述取代简单类别命名的范式
- **组合泛化**: MIT States、UT-Zappos50K、C-GQA 等数据集测试 (属性,物体) 新组合的识别能力,但未控制概念间的不相似度
- **探测分类器**: Alain & Bengio 的线性探测方法和 Hewitt & Liang 的控制任务设计为本文提供了方法论基础
- **跨不相似类别的属性推理**: CORE 和 Find-the-Common (FTC) 虽然目标类似,但规模小或评估结构不适合属性泛化
- **启发**: 划分策略（splitting strategy）是评测设计中被严重低估的维度,本文的方法论可推广至其他任务（如关系推理、组合泛化）中的泄漏控制

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次系统研究划分策略对属性泛化的影响,问题定义清晰且重要)
- 实验充分度: ⭐⭐⭐ (四种视觉模型×五种划分策略的完整矩阵,但仅单一数据集和线性探测)
- 写作质量: ⭐⭐⭐⭐ (结构紧凑、图表直观、贡献描述准确)
- 价值: ⭐⭐⭐⭐ (对属性预测/零样本学习的评测公平性具有根本性启示,提供了可复现的划分工具)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Language Model Probabilities are Not Calibrated in Numeric Contexts](../../ACL2025/llm_evaluation/language_model_probabilities_are_not_calibrated_in_numeric_contexts.md)
- [\[ACL 2025\] Beyond One-Size-Fits-All: Tailored Benchmarks for Efficient Evaluation](../../ACL2025/llm_evaluation/beyond_one-size-fits-all_tailored_benchmarks_for_efficient_evaluation.md)
- [\[ACL 2025\] ONEBench to Test Them All: Sample-Level Benchmarking Over Open-Ended Capabilities](../../ACL2025/llm_evaluation/onebench_to_test_them_all_sample-level_benchmarking_over_open-ended_capabilities.md)
- [\[ICML 2025\] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](../../ICML2025/llm_evaluation/evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](../../ICCV2025/llm_evaluation/rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)

</div>

<!-- RELATED:END -->
