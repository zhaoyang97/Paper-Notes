---
title: >-
  [论文解读] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning
description: >-
  [AAAI 2026][3D视觉][学术信息抽取] 提出GSAP-ERE——一个面向机器学习领域的细粒度学术实体与关系抽取数据集，包含10种实体类型和18种关系类型，在100篇全文论文上标注了63K实体和35K关系，实验表明微调模型（NER: 80.6%, RE: 54.0%）大幅超越LLM提示方法（NER: 44.4%, RE: 10.1%）。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "学术信息抽取"
  - "命名实体识别"
  - "关系抽取"
  - "知识图谱"
  - "机器学习可复现性"
  - "细粒度标注"
---

# GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning

**会议**: AAAI 2026  
**arXiv**: [2511.09411](https://arxiv.org/abs/2511.09411)  
**作者**: Wolfgang Otto, Lu Gan, Sharmila Upadhyaya, Saurav Karmakar, Stefan Dietze (GESIS)  
**代码**: [https://data.gesis.org/gsap/gsap-ere](https://data.gesis.org/gsap/gsap-ere)  
**领域**: 图学习  
**关键词**: 学术信息抽取, 命名实体识别, 关系抽取, 知识图谱, 机器学习可复现性, 细粒度标注  

## 一句话总结

提出GSAP-ERE——一个面向机器学习领域的细粒度学术实体与关系抽取数据集，包含10种实体类型和18种关系类型，在100篇全文论文上标注了63K实体和35K关系，实验表明微调模型（NER: 80.6%, RE: 54.0%）大幅超越LLM提示方法（NER: 44.4%, RE: 10.1%）。

## 研究背景与动机

### 问题背景
机器学习研究发展迅速，但可复现性持续下降——已有研究发现仅4%的ML论文能在原作者不回复的情况下被复现。理解模型、数据集和任务之间的依赖关系对于提升研究可复现性和可重用性至关重要。学术信息抽取（Scholarly IE）通过从论文中自动提取实体及其关系，为构建知识图谱、监控研究可复现性提供了规模化路径。

### 已有工作的不足
- **实体类型粗粒度**：SciERC仅6种实体类型，SciER仅3种（Dataset, Task, Method），无法识别模型架构、数据来源等关键元信息
- **标注范围有限**：ScienceIE仅标注段落，SciERC和SemEval 2018仅标注摘要，未覆盖全文的多样语言风格
- **LLM直接应用效果差**：现有LLM在细粒度领域特定IE任务上远逊于微调模型，不适合直接用于高质量学术IE
- **缺乏非正式实体标注**：SciER仅包含显式命名实体，忽略了大量非正式提及（如"the model"、"this dataset"），导致关系标注不完整
- **关系覆盖不全面**：现有数据集最多9种关系类型（SciER），无法捕获模型设计、数据溯源、同行比较等多维关系

### 核心动机
构建一个覆盖全文、包含细粒度实体和关系标注的综合数据集，支撑从知识图谱构建到AI研究可复现性监控等多种下游任务。

## 方法详解

### 数据模型设计
基于GSAP-NER数据集扩展，定义了完整的实体-关系体系：

**10种实体类型**（三大类别）：
- ML模型相关：MLModel, ModelArchitecture, MLModelGeneric, Method, Task
- 数据集相关：Dataset, DatasetGeneric, DataSource
- 其他：ReferenceLink, URL

**18种关系类型**（七个语义组）：
1. **Model Design**（模型设计）：usedFor, architecture, isBasedOn——捕获模型/方法之间的组成和衍生关系
2. **Task Binding**（任务绑定）：appliedTo, benchmarkFor——关联模型/数据集与任务
3. **Data Usage**（数据使用）：trainedOn, evaluatedOn——训练和评估依赖
4. **Data Provenance**（数据溯源）：sourcedFrom, transformedFrom, generatedBy——数据来源和转化
5. **Data Properties**（数据属性）：size, hasInstanceType——数据集规模和模态
6. **Peer Relations**（同行关系）：coreference, isPartOf, isHyponymOf, isComparedTo——同类型实体间关系
7. **Referencing**（引用关系）：citation, url——外部来源链接

### 标注策略
采用两阶段"标注-精炼"策略：
- **标注阶段**：2名计算机科学背景的学生标注员，10篇论文双标注，90篇分配单标注，使用INCEpTION平台
- **精炼阶段**：2名博士生和2名博后研究员审查标注对齐情况，提取不一致模式并修正

### 评估设置
设计了四种RE评估严格度：
- **RE+**：实体类型、关系标签和实体边界全部严格匹配
- **RE**：仅要求关系标签和实体边界匹配，不限制实体类型
- **RE+≈**：标签严格匹配，实体边界允许部分重叠
- **RE≈**：仅要求关系标签正确且实体边界重叠

### 基线模型
- **监督Pipeline**：PL-Marker——先NER后RE，使用Packed Levitated Marker建模实体对交互
- **监督联合模型**：HGERE——基于PL-Marker框架引入超图神经网络，NER和RE联合优化
- **LLM提示**：Qwen 2.5（32B/72B）和LLaMA 3.1（70B），两阶段pipeline提示（先NER后RE）

## 实验关键数据

### 实验1：监督模型 vs LLM提示方法

| 方法 | 模型 | NER | NER≈ | RE | RE≈ | RE+ | RE+≈ |
|------|------|-----|------|-----|-----|-----|------|
| 监督联合 | HGERE | **80.6** | **85.8** | **54.0** | **59.8** | **46.9** | **51.3** |
| 监督Pipeline | PL-Marker | 72.6 | 77.7 | 41.4 | 46.2 | 36.3 | 39.9 |
| LLM Pipeline | Qwen 2.5 72B | 44.4 | 59.1 | 10.1 | 15.7 | 8.2 | 11.9 |
| LLM Pipeline | Qwen 2.5 32B | 42.0 | 56.9 | 7.2 | 14.6 | 7.2 | 10.9 |
| LLM Pipeline | LLaMA 3.1 70B | 40.5 | 55.0 | 6.4 | 9.6 | 5.7 | 7.8 |

监督HGERE在所有指标上全面领先：NER超出LLM最佳36.2个百分点，RE超出43.9个百分点。推理速度上，PLM方法比LLM快182倍（4分钟 vs 12.5小时）。

### 实验2：Few-Shot示例选择策略对NER的影响（Qwen2.5 32B验证集）

| 选择策略 | k=0 | k=1 | k=2 | k=5 | k=10 | k=20 |
|---------|-----|-----|-----|-----|------|------|
| random (micro-F1) | 19.1 | 24.7 | 23.1 | 29.7 | 34.1 | 34.4 |
| similar+diverse (micro-F1) | 19.1 | 34.7 | 38.2 | 40.4 | **40.9** | 27.8 |
| random (NER≈ micro-F1) | 33.0 | 41.8 | 37.1 | 50.1 | 53.3 | 50.3 |
| similar+diverse (NER≈ micro-F1) | 33.0 | 53.8 | 56.7 | 58.1 | **58.4** | 39.4 |

similar+diverse策略在k=10时效果最佳，比random策略高约5个百分点；k=20时性能急剧下降。RE最佳配置为k=1（micro-F1: 10.7%），增加示例数反而无益。

### 数据集规模对比

| 数据集 | 标注单元 | 论文数 | 实体类型 | 关系类型 | 实体数 | 关系数 | 关系/篇 |
|--------|---------|--------|---------|---------|--------|--------|---------|
| **GSAP-ERE** | 全文 | 100 | **10** | **18** | **62,619** | **35,302** | **353.0** |
| SciER | 全文 | 106 | 3 | 9 | 24,518 | 12,083 | 114.0 |
| SciERC | 摘要 | 500 | 6 | 7 | 8,094 | 4,648 | 9.3 |
| SemEval18 | 摘要 | 500 | - | 6 | 7,505 | 1,583 | 3.3 |
| ScienceIE | 段落 | 500 | 3 | 2 | 9,946 | 672 | 3.1 |

GSAP-ERE在实体数、关系数、类型丰富度和标注密度上全面超越现有数据集。

## 亮点

- **最大规模学术IE数据集**：63K实体+35K关系，18种关系类型覆盖7个语义维度，标注密度（353关系/篇）远超同类数据集
- **细粒度数据模型**：区分正式和非正式实体提及，捕获模型设计、数据溯源、同行比较等多维关系，直接服务于ML研究可复现性监控
- **全文级标注**：相比仅标注摘要或段落的数据集，全文标注覆盖更丰富的语言风格和信息
- **严格质量控制**：两阶段标注-精炼流程，NER标注者间一致性达0.82 macro-F1，经精炼后显著提升
- **揭示LLM局限**：实证表明当前最强LLM在细粒度学术IE上与微调模型差距巨大（RE差距达43.9%），为领域数据集的必要性提供了有力证据

## 局限与展望

- **领域局限**：仅覆盖ML和应用ML领域的论文，未涉及其他学科（如生物医学、物理等），泛化性待验证
- **句子级标注**：当前仅支持句子级实体和关系标注，缺乏文档级跨句关系标注，无法捕获长距离依赖
- **数据集规模**：仅100篇论文，训练集80篇，对于深度学习模型可能不够充分
- **RE性能上限低**：即使最好的监督模型RE+ F1仅46.9%，表明任务难度极高或数据模型需要进一步优化
- **标注者间一致性参差**：Model Design语义组RE+一致性仅38.4%，反映部分关系定义边界模糊
- **未考虑嵌套关系**：虽数据集包含嵌套和重叠实体，但未深入分析这些复杂结构对性能的影响

## 与相关工作的对比

- **SciERC (Luan et al. 2018)**：6种实体+7种关系，仅标注500篇摘要；GSAP-ERE在类型丰富度和标注密度上远超，且为全文标注
- **SciER (Zhang et al. 2024)**：3种实体+9种关系，排除非正式实体；GSAP-ERE保留非正式提及，增加关系覆盖完整性
- **DMDD (Huitong et al. 2023)**：基于远程监督自动标注，无关系标注；GSAP-ERE为人工精标注且包含丰富关系
- **SciREX (Jain et al. 2020)**：关系标注限于mention聚类而非成对关系，忽略上下文信息
- **PL-Marker (Ye et al. 2022)**：Pipeline方法在GSAP-ERE上NER 72.6%、RE 41.4%，低于联合方法HGERE
- **HGERE (Yan et al. 2023)**：联合方法在GSAP-ERE上取得最佳性能（NER 80.6%、RE+ 46.9%），验证了超图网络在学术IE中的有效性
- **LLM提示方法**：Qwen 2.5 72B和LLaMA 3.1即使采用10-shot similar+diverse策略，RE性能仍不足11%，与Zhang et al. (2024)在SciER上的观察一致

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个结合10种细粒度实体和18种语义分组关系的全文学术IE数据集，填补领域空白
- 实验充分度: ⭐⭐⭐⭐ — 覆盖监督和无监督方法，消融了few-shot策略，但缺少跨领域泛化实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据模型定义详尽，对比表格丰富
- 价值: ⭐⭐⭐⭐ — 为ML研究可复现性监控和知识图谱构建提供了高质量基准，揭示LLM在领域IE上的不足
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] iSplat: Iterative Learning for Fine-Grained Gaussian Splatting](../../CVPR2026/3d_vision/isplat_iterative_learning_for_fine-grained_gaussian_splatting.md)
- [\[CVPR 2026\] Animator-Centric Skeleton Generation on Objects with Fine-Grained Details](../../CVPR2026/3d_vision/animator-centric_skeleton_generation_on_objects_with_fine-grained_details.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[CVPR 2026\] Fresco: Frequency-Spatial Consistent Optimization for Fine-Grained Head Avatar Modeling](../../CVPR2026/3d_vision/fresco_frequency-spatial_consistent_optimization_for_fine-grained_head_avatar_mo.md)
- [\[CVPR 2026\] InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields](../../CVPR2026/3d_vision/infinidepth_arbitrary-resolution_and_fine-grained_depth_estimation_with_neural_i.md)

</div>

<!-- RELATED:END -->
