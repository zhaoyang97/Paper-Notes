---
title: >-
  [论文解读] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning
description: >-
  [NEURIPS2025][目标检测][文本检测] 提出 DETree 框架，通过构建层次亲和树（HAT）建模不同人机协作文本生成过程之间的层次关系，并设计树结构对比损失（TSCL）对齐表示空间，在混合文本检测和 OOD 场景下取得了显著优势。 随着 LLM 的广泛应用，AI 参与文本生成的方式日益多样化：人类撰写后由 AI…
tags:
  - "NEURIPS2025"
  - "目标检测"
  - "文本检测"
  - "human-AI collaborative text"
  - "hierarchical representation learning"
  - "对比学习"
  - "out-of-distribution generalization"
---

# DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning

**会议**: NEURIPS2025  
**arXiv**: [2510.17489](https://arxiv.org/abs/2510.17489)  
**代码**: [heyongxin233/DETree](https://github.com/heyongxin233/DETree)  
**领域**: 目标检测  
**关键词**: AI text detection, human-AI collaborative text, hierarchical representation learning, contrastive learning, out-of-distribution generalization

## 一句话总结

提出 DETree 框架，通过构建层次亲和树（HAT）建模不同人机协作文本生成过程之间的层次关系，并设计树结构对比损失（TSCL）对齐表示空间，在混合文本检测和 OOD 场景下取得了显著优势。

## 背景与动机

随着 LLM 的广泛应用，AI 参与文本生成的方式日益多样化：人类撰写后由 AI 润色、AI 生成后由人类修改、多个 AI 模型协作生成等。不同场景对 AI 参与的容忍度不同（如广告文案可接受、学术写作严格禁止），因此检测方法不仅需要判断文本是否涉及 AI，还需要识别 AI 参与的具体方式和程度。

现有方法存在明显局限：

- **二分类方法**：仅区分纯人类文本和纯 AI 文本，无法应对人机协作的复杂场景
- **粗粒度估计**：如回归 AI 参与度或基于预定义 prompt 的内容/风格特征分类，表达能力有限
- **泛化能力弱**：在训练分布上效果好，但在 OOD 场景下性能严重下降

作者的关键观察是：不同生成过程产生的文本在表示空间中天然存在层次化的聚类关系。例如 "Llama3_polish_GPT-4o" 与 "Claude3.5_paraphrase_GPT-4o" 的相似度高于它们与 "human_polish_Gemini1.5" 的相似度，而与纯人类文本的相似度最低。

## 核心问题

1. 如何对人机协作文本的多种生成过程之间的内在关系进行结构化建模？
2. 如何构建一个能涵盖多样化人机协作模式的大规模基准数据集？
3. 如何在 OOD 场景下（新模型、新领域）保持强泛化能力？

## 方法详解

### 1. RealBench 数据集构建

聚合 MAGE、M4、TuringBench、OUTFOX、RAID 等数据集的原始样本，引入改写（paraphrasing）、扩展（extension）、润色（polishing）、翻译（translation）等混合文本构造策略，并集成 11 种基于扰动的攻击类型。最终覆盖 **1,204 个文本类别**，约 **1,640 万条文本样本**。

### 2. Hierarchical Affinity Tree (HAT) 构建

- **第一步**：使用监督对比学习微调编码器，将每个类别视为独立类别，计算类间相似度矩阵 $E \in \mathbb{R}^{N \times N}$
- **第二步**：基于相似度矩阵采用凝聚层次聚类生成初始二叉树结构
- **第三步**：引入可编辑的自顶向下子树重组算法。预定义三种先验（混合文本归入人类类别/AI 类别/独立类别），对每个子树枚举所有可能划分，用 Silhouette Score 选最优分区，递归执行直到满足停止条件
- 计算复杂度为 $\mathcal{O}(N^2 \log N)$

### 3. Tree-Structured Contrastive Loss (TSCL)

核心思想是：在 HAT 中共同祖先越近的类别，在表示空间中应越相似。

对叶节点 $c$ 深度为 $d_c$，定义第 $i$ 层层次分区集合：

$$H_c^{(i)} = \{x \mid x \in \text{leaf}(f_c^{(i)}) \setminus \text{leaf}(f_c^{(i-1)})\}$$

基于 Theorem 3.1 的等价性，将层次相似度约束转化为可优化的对比学习目标。在每个层次 $i$ 定义正集 $P_i = H_c^{(i)}$ 和负集 $N_i = \bigcup_{j=i+1}^{d_c} H_c^{(j)}$，完整 TSCL 为：

$$\mathcal{L}_{\text{TSCL}}(x;\theta) = \frac{1}{d_c} \sum_{i=0}^{d_c-1} \mathcal{L}_c^{(i)}(x;\theta)$$

### 4. Virtual Class Prototype (VCP)

由于类别数量很大（1,204），单个 mini-batch 难以覆盖所有类别。为每个类别引入可学习的原型向量 $\mathbf{v}_c \in \mathbb{R}^d$ 作为持久锚点参与对比学习，不增加额外内存开销。

### 5. 推理与 OOD 适应

- **K-Means 数据库压缩**：对每个类别用 K-means 聚类生成紧凑代表，减少检索开销同时保持平衡的类别表示
- **基于检索的 Few-Shot 适应**：利用少量目标域样本重建检索数据库，调整决策边界以适应域偏移

### 实现细节

- 编码器：RoBERTa-large + LoRA 微调
- 优化器：AdamW，余弦退火，初始学率 3e-5，2000 步线性预热
- 训练：10 epochs，8×RTX 4090，batch size 64，最大输入长度 512
- 推理：Faiss-GPU 加速 K-means 和 KNN，表示层选自 17-19 层，k=5 或 50
- 温度参数 $\tau = 0.07$

## 实验关键数据

### 监督检测

| 方法 | MAGE AvgRec | M4-mono AvgRec | TuringBench AvgRec | 平均 AvgRec |
|------|-------------|----------------|--------------------| ------------|
| DeTeCTive | 96.15 | 98.44 | 99.74 | 96.94 |
| DETree (prior1) | **96.87** | **99.86** | **99.74** | **97.88** |

### OOD 泛化（AUROC）

| 数据集 | Binoculars | MAGE | DETree zero-shot | DETree 10-shot |
|--------|-----------|------|------------------|----------------|
| MAGE-Unseen | 96.84 | 95.20 | 99.13 | **99.81** |
| MAGE-Paraphrase | 75.87 | 83.35 | 92.66 | **98.77** |
| DetectRL-MultiDomain | 83.95 | 86.67 | 98.94 | **99.88** |
| Beemo-GPT4o-Edited | 78.15 | 67.79 | 83.79 | **88.54** |

### 混合文本检测（HART，AUROC）

| 检测器 | Level-1 ALL | Level-2 ALL | Level-3 ALL |
|--------|------------|------------|------------|
| HART(Binoculars) | 0.838 | 0.848 | 0.883 |
| DETree (prior1) | **0.998** | **0.992** | **0.988** |

99.32% 的三元组满足 Theorem 3.1 中定义的层次相似度约束，验证了 TSCL 的有效性。

## 亮点

1. **层次化建模思路新颖**：将文本检测从扁平分类提升为基于树结构的层次化表示学习，HAT 能自主捕获文本来源间的关联（如同模型族聚类、同操作策略聚类）
2. **理论支撑完整**：Theorem 3.1 证明了层次相似度约束的等价性，为 TSCL 设计提供理论基础
3. **OOD 泛化能力强**：通过检索式 few-shot 适应，仅需 5-10 个目标域样本即可大幅提升跨域检测性能（MAGE-Paraphrase 上 +15.55 AUROC）
4. **数据集规模大**：RealBench 覆盖 1,204 类别和 1,640 万样本，是目前最全面的人机协作文本基准
5. **鲁棒性强**：即使不做对抗训练，DETree 在多数攻击场景下仍优于经对抗训练的 RoBERTa-large
6. **有趣发现**：混合文本中 AI 痕迹比人类特征更显著；人类编辑不会根本改变基础文本的 AI 特征

## 局限与展望

1. 未探索通过模型微调进行对抗性逃避检测的场景
2. 目前聚焦三作者（两 AI 或一 AI + 人类）的混合文本，更多协作者的场景有待研究
3. 遇到完全未见的稀有领域时，仍需少量目标域样本调整决策边界，无法做到完全零样本泛化
4. 论文分类归入 object_detection 值得商榷，实际属于 NLP/AI 安全领域

## 与相关工作的对比

| 维度 | 传统方法 | HART | DETree |
|------|---------|------|--------|
| 分类粒度 | 二分类 | 三层风险等级 | 层次化类别树 |
| 建模方式 | 扁平特征 | 内容/风格解耦 | 树结构对比学习 |
| OOD 泛化 | 差 | 中等 | 强（few-shot 适应） |
| 数据规模 | 万级 | 十万级 | 千万级（RealBench） |

与 Hierarchical Text Classification (HTC) 的区别：HTC 的标签层次是预定义且固定的，而混合文本检测的标签本质上是模糊且动态演化的，HAT 的构造是数据驱动的自适应过程。

## 启发与关联

- 层次化建模思路可推广到其他需要细粒度类别关系建模的任务，如恶意代码家族检测、deepfake 视频溯源
- K-Means 数据库压缩 + KNN 检索的推理范式在大规模分类场景中具有通用价值
- Few-shot 检索适应方法为训练式检测器在 OOD 场景下提供了新的可行方向
- RealBench 的构建策略（组合式类别扩展）可复用于其他需要大规模多类别数据集的研究

## 评分
- 新颖性: 8/10 — 层次亲和树和树结构对比损失在 AI 文本检测领域属首创
- 实验充分度: 9/10 — 覆盖监督/OOD/混合检测/鲁棒性/压缩等多维度，消融实验完整
- 写作质量: 8/10 — 结构清晰，理论推导完整，图表信息量丰富
- 价值: 8/10 — 方法设计合理且实用，数据集贡献显著，对实际 AI 文本审核有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](../../CVPR2025/object_detection/show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[ICML 2025\] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](../../ICML2025/object_detection/self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)
- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[CVPR 2026\] Expert-Teacher-Student Collaborative Learning for Domain Adaptive Object Detection](../../CVPR2026/object_detection/expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti.md)

</div>

<!-- RELATED:END -->
