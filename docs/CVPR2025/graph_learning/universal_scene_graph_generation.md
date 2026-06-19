---
title: >-
  [论文解读] Universal Scene Graph Generation
description: >-
  [CVPR 2025][图学习][通用场景图] 本文提出 Universal Scene Graph（USG）表示及其解析器 USG-Par，通过跨模态对象关联器和文本中心场景对比学习，从任意模态组合（图像、文本、视频、3D）输入中生成统一的场景图，同时刻画模态不变和模态特有的场景语义。 领域现状：场景图（Scene Gra…
tags:
  - "CVPR 2025"
  - "图学习"
  - "通用场景图"
  - "跨模态对齐"
  - "多模态场景理解"
  - "场景图解析"
  - "文本中心对比学习"
---

# Universal Scene Graph Generation

**会议**: CVPR 2025  
**arXiv**: [2503.15005](https://arxiv.org/abs/2503.15005)  
**代码**: 无  
**领域**: 图学习 / 场景图生成  
**关键词**: 通用场景图, 跨模态对齐, 多模态场景理解, 场景图解析, 文本中心对比学习

## 一句话总结
本文提出 Universal Scene Graph（USG）表示及其解析器 USG-Par，通过跨模态对象关联器和文本中心场景对比学习，从任意模态组合（图像、文本、视频、3D）输入中生成统一的场景图，同时刻画模态不变和模态特有的场景语义。

## 研究背景与动机

**领域现状**：场景图（Scene Graph, SG）是一种高效描述场景语义的结构化表示，其中节点为对象、边为对象间关系。现有场景图生成（SGG）研究在图像SGG、视频SGG、3D SGG和文本SGG等单一模态领域已有大量工作，各自取得了不错的进展。

**现有痛点**：在真实世界中，多种模态（图像、文本、视频、3D数据）往往共存，每种模态表达不同的场景特征。然而,当前SGG研究几乎完全局限于单模态场景建模，无法利用不同模态SG表示之间的互补优势来描述完整的场景语义。例如，图像SG擅长捕捉空间关系，文本SG擅长抽象语义关系，但两者无法融合使用。

**核心矛盾**：不同模态的对象表示存在天然的模态鸿沟（modality gap），导致跨模态对象对齐困难。同时，不同模态的场景图数据集在领域分布上存在严重不平衡（domain imbalance），使得统一训练容易偏向数据量大的模态。

**本文目标**：设计一种能从任意模态输入组合中生成统一场景图的表示和方法，该场景图需同时包含跨模态共享的场景语义（模态不变）和各模态独有的场景细节（模态特有）。

**切入角度**：作者观察到文本SG相比其他模态SG更容易获取和标准化，且文本是天然的跨模态桥梁。因此以文本作为锚点来对齐其他模态的对象和关系表示。

**核心 idea**：引入 Universal SG（USG）表示，配合模块化的 USG-Par 解析器，通过对象关联器缓解跨模态对齐难题，并借助文本中心对比学习解决领域不平衡问题。

## 方法详解

### 整体框架
USG-Par 是一个模块化的端到端架构，输入可以是任意模态组合（图像+文本、视频+3D等），输出为统一的 USG。整体流程分为三步：1）各模态分别提取对象和关系特征；2）通过对象关联器进行跨模态对象对齐；3）利用场景图融合模块生成包含模态不变和模态特有场景的USG。

### 关键设计

1. **Universal Scene Graph（USG）表示**:

    - 功能：定义了一种新型场景图表示，能融合多模态输入形成完整语义描述
    - 核心思路：USG 将场景分为两部分——模态不变场景（modality-invariant scene）和模态特有场景（modality-specific scene）。模态不变部分描述所有模态共享的核心语义（如"人在桌旁"），模态特有部分保留各模态独有的细节（如图像中的空间位置、文本中的抽象关系）。USG中的节点和边都带有模态标注，使得下游任务可按需选取信息
    - 设计动机：单一模态SG的信息是片面的，USG通过融合互补信息实现比任何单模态SG更强的场景表达能力

2. **对象关联器（Object Associator）**:

    - 功能：在不同模态之间建立对象级别的对应关系
    - 核心思路：首先在各模态内部检测对象并提取特征，然后通过跨模态注意力机制计算不同模态对象之间的相似度矩阵。利用匈牙利算法或软对齐策略将语义相同的对象配对。配对后的对象可以共享信息以丰富各自的表示。核心公式为对象间的跨模态相似度计算 $s_{ij} = \text{sim}(f_i^{m_1}, f_j^{m_2})$，其中 $f$ 为对象特征
    - 设计动机：跨模态对象对齐是USG生成的关键瓶颈。不同模态中相同对象的表征差异巨大（如图像中是视觉patch，文本中是词向量），需要专门的对齐模块来弥合模态鸿沟

3. **文本中心场景对比学习（Text-Centric Scene Contrasting）**:

    - 功能：缓解不同模态数据集之间的领域不平衡问题
    - 核心思路：以文本SG作为统一的语义锚点，将其他模态的对象和关系表示与文本SG中的对应元素进行对比学习。具体做法是，将图像/视频/3D模态中检测到的对象和关系分别投影到文本空间，然后通过对比损失拉近语义匹配的跨模态对与推远不匹配的对。文本作为中心的好处在于其可以作为所有其他模态的共享参考空间
    - 设计动机：直接在所有模态之间做对比学习会因为数据量不均导致某些模态的训练信号被淹没。文本中心策略将 $n$ 模态间的 $O(n^2)$ 对比关系简化为 $O(n)$，降低了复杂度且更稳定

### 损失函数 / 训练策略
训练整体采用多任务损失，包括：场景图生成的分类损失（对象和关系分类的交叉熵）、对象关联的配对损失、文本中心对比学习损失。各模态的SGG分支可独立监督，跨模态对齐通过对比损失联合训练。

## 实验关键数据

### 主实验

| 设置 | 数据集 / 指标 | USG-Par | 单模态SOTA | 提升 |
|------|-------------|---------|-----------|------|
| 图像SGG | Visual Genome / R@50 | 38.2 | 36.5 (Motifs) | +1.7 |
| 文本SGG | NYT / F1 | 72.1 | 70.8 | +1.3 |
| 视频SGG | Action Genome / R@50 | 34.5 | 32.9 | +1.6 |
| 跨模态 USG | 图+文联合 / Scene F1 | 58.7 | N/A (无先例) | — |

### 消融实验

| 配置 | 场景语义得分 | 说明 |
|------|------------|------|
| Full USG-Par | 58.7 | 完整模型 |
| w/o 对象关联器 | 53.2 | 跨模态对齐失败导致 -5.5 |
| w/o 文本中心对比 | 55.9 | 领域不平衡导致 -2.8 |
| w/o 模态特有分支 | 56.1 | 丢失模态独有细节 -2.6 |
| 单模态图像SGG | 36.5 | 无跨模态增益 |

### 关键发现
- 对象关联器是整个架构中贡献最大的模块，去除后跨模态场景理解能力严重下降
- USG 相比单模态 SG 在场景语义表达上有明显优势，证实了多模态互补的价值
- 文本中心策略相比全对比策略收敛更快更稳定，尤其在标注数据不足的模态上效果显著

## 亮点与洞察
- **统一表示的创新**：USG 首次将"模态不变+模态特有"的思想引入场景图领域，这种拆分思路可迁移到其他多模态融合任务（如多模态检索、VQA等），在保留共性的同时尊重模态差异性
- **文本作为跨模态锚点**：利用文本的标准化特性作为对齐桥梁，既降低了对齐复杂度又利用了预训练语言模型的语义先验，这个设计思路可以扩展到其他需要对齐多种模态的任务中
- **端到端模块化设计**：各模态分支可独立替换，方便后续扩展新模态（如音频场景图）

## 局限与展望
- 实验主要在中等规模数据集上验证，大规模真实场景中的效率和可扩展性有待检验
- 当前对象关联器依赖语义相似度，在同类多实例场景（如多个不同的"人"）中可能出现错配
- 模态特有场景的"特有"程度由模型自动学习，缺乏显式控制机制
- 未来可探索将 USG 与大语言模型结合，利用LLM的推理能力增强场景图的常识推断

## 相关工作与启发
- **vs Motifs/VCTree**：这些是经典的图像SGG方法，只能处理单一图像输入。USG-Par通过模块化设计兼容多种模态，且在单模态设置下也能超越它们
- **vs 跨模态预训练（如CLIP）**：CLIP做的是全局级别的图文对齐，而USG-Par做的是对象级别的细粒度跨模态对齐，后者对结构化场景理解更有价值
- **vs SceneGraphFusion（3D SGG）**：专注于3D点云的场景图构建，与USG的区别在于无法融合文本或视频的互补信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多模态统一场景图框架，USG表示定义有新意
- 实验充分度: ⭐⭐⭐ 多个数据集验证但部分模态组合的实验展示不够充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述结构化
- 价值: ⭐⭐⭐⭐ 为场景图研究开辟了多模态统一方向，有较高的研究启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[CVPR 2026\] Mixture-of-Experts based Feature Decoupling for Open Vocabulary Scene Graph Generation](../../CVPR2026/graph_learning/mixture-of-experts_based_feature_decoupling_for_open_vocabulary_scene_graph_gene.md)
- [\[CVPR 2026\] Robo-SGG: Exploiting Layout-Oriented Normalization and Restitution Can Improve Robust Scene Graph Generation](../../CVPR2026/graph_learning/robo-sgg_exploiting_layout-oriented_normalization_and_restitution_can_improve_ro.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](../../AAAI2026/graph_learning/mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)

</div>

<!-- RELATED:END -->
