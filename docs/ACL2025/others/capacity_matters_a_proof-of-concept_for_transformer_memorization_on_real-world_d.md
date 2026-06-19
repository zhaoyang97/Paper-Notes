---
title: >-
  [论文解读] Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data
description: >-
  [ACL 2025 (L2M2 Workshop)][Transformer] 本文以SNOMED医学知识图谱为数据源，系统研究了decoder-only Transformer在结构化数据上的记忆容量，发现嵌入维度是决定学习速度和容量的主要因素，而增加层数收效甚微，Softmax激活函数表现最稳定。
tags:
  - "ACL 2025 (L2M2 Workshop)"
  - "Transformer"
  - "知识图谱"
  - "嵌入维度"
  - "激活函数"
  - "边缘部署"
---

# Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data

**会议**: ACL 2025 (L2M2 Workshop)  
**arXiv**: [2506.14704](https://arxiv.org/abs/2506.14704)  
**代码**: [有](https://github.com/um-dacs-nlp/capacity/)  
**领域**: 其他  
**关键词**: Transformer记忆容量, 知识图谱, 嵌入维度, 激活函数, 边缘部署

## 一句话总结

本文以SNOMED医学知识图谱为数据源，系统研究了decoder-only Transformer在结构化数据上的记忆容量，发现嵌入维度是决定学习速度和容量的主要因素，而增加层数收效甚微，Softmax激活函数表现最稳定。

## 研究背景与动机

- Transformer已在NLP任务中取得巨大成功，但**模型如何存储和召回信息**（尤其是事实性/结构化知识）的机理仍不清楚
- 实际应用场景：在医疗可穿戴设备（如智能眼镜、智能手表）上部署小型本地Transformer，需要模型在有限参数下**精确记忆**特定领域的事实知识
- 已有理论工作（Kim et al., 2023; Kajitsuka & Sato, 2024）给出了Transformer记忆容量的数学上界$O(d + n + \sqrt{nN})$，但缺乏在**真实世界结构化数据**上的经验验证
- 本文不关注泛化能力，而专注于**可控条件下的记忆能力**测量，作为连接理论分析与实际评估的概念验证

## 方法详解

### 整体框架

本文设计了一套完整的实验框架：(1) 从SNOMED知识图谱生成结构化数据集；(2) 训练小规模decoder-only Transformer进行逐token预测；(3) 使用Maximum Attainable Capacity (MAC)指标衡量记忆容量。

### 数据生成

#### Triplets数据集
- 从SNOMED知识图谱中提取 **(Concept, Property, Related Concept)** 三元组
- 过滤非信息性属性，对同一(Concept, Property)对存在多个related concept时随机选一个保证唯一性
- 数据规模：50K到100K条样本

#### Sequences数据集
- 模拟图遍历，生成形如 **(node₁, edge₁, node₂, ..., nodeₙ)** 的序列
- 使用BFS（深度5跳）构建子图，然后随机游走生成序列，每个序列包含4-6个节点（3-5条边）
- 通过零填充统一序列长度，使用node mask区分节点和边token
- 数据规模：20K到100K条序列

### 模型架构

- **Decoder-only Transformer**：包含嵌入层（学习的位置编码）、Transformer解码层（多头注意力）、线性输出层
- 参数规模：2.9M到44.5M，主要随嵌入维度和层数变化
- 预测任务：基于前序token预测下一个concept
- 准确率定义：正确预测的related concept数 / 总预测数

### 关键度量：Maximum Attainable Capacity (MAC)

- MAC测量模型在给定大库（large library）时**能记住的最大样本数**
- 相比Maximum Library Size (MLS)方法（需要迭代训练不同规模数据集），MAC计算效率更高
- 先前研究已验证MAC与MLS具有强相关性

### 四组实验设置

**Setup 1 - 数据规模影响**：1层，embedding=128，4头注意力，ReLU，batch=64，500 epochs，数据50K-100K

**Setup 2 - 架构与激活函数**：层数1/2/4；激活ReLU/GELU/RReLU/Softmax；保持总参数量恒定(`embedding_size = ⌊base_params / n_layers⌋`)，batch=128，1000 epochs

**Setup 3 - 参数量与深度交互**：1/2层，base参数{16,32,64,128}，仅Softmax，batch=128，500 epochs

**Setup 4 - 序列记忆**：embedding=64，4头注意力，1/2/4层，RReLU和Softmax，batch=128，400 epochs

### 损失函数 / 训练策略

- 损失函数：**交叉熵损失**
- 优化器：**Adam** (lr=0.001)
- 每个实验重复10次（Setup 1-2）或3次（Setup 3-4），报告均值±2标准差
- 计算资源：NVIDIA A100 16GB，共训练546个模型，约3100 GPU小时

## 实验关键数据

### 主实验（数据规模 vs 记忆容量）

| 数据规模 | 准确率(%) | 容量(MAC) |
|---------|----------|----------|
| 50,000 | 93.62±0.3 | 46,811±149 |
| 60,000 | 92.42±0.2 | 55,455±126 |
| 70,000 | 91.1±1.08 | 63,773±756 |
| 80,000 | 89.63±1.66 | 71,706±1326 |
| 90,000 | 87.24±1.66 | 78,517±2173 |
| 100,000 | 86.78±2.42 | 86,776±2484 |

### 序列数据集记忆容量（100K序列）

| 激活函数 | 层数 | 容量(MAC) | 总预测数 |
|---------|------|----------|---------|
| RReLU | 1 | 166,934±243 | 167,965 |
| RReLU | 4 | 165,271±1,068 | 167,965 |
| Softmax | 1 | 166,992±110 | 167,965 |
| Softmax | 4 | 166,825±319 | 167,965 |

### 消融实验

- **嵌入维度 vs 层数**：embedding=16的1层模型与embedding=16/层的2层模型收敛速度几乎相同 → 嵌入维度决定学习速度
- **激活函数对比**：Softmax在所有配置下最稳定；ReLU/RReLU在深层模型中变异性增大
- **容量瓶颈**：100K数据上，2层embedding=8的模型容量下降至85,935±153（vs 其他配置~88,200）

### 关键发现

1. **嵌入维度是核心因素**：相同嵌入维度下不同层数模型的学习曲线几乎一致
2. **增加层数无益甚至有害**：对简单结构化数据，额外层降低训练速度和最终容量
3. **Softmax全面领先**：在容量、稳定性和深度扩展性上优于ReLU/GELU/RReLU
4. **序列 > 三元组**：序列数据更少epoch即可达近乎完美记忆（20K序列达100%）
5. **存在~70K阈值**：超过此规模后训练动态发生质变，需显著更多epoch

## 亮点与洞察

- 首次将大型医学本体转化为可用于记忆研究的tokenized数据集，提供理论到实践的桥梁
- 反直觉发现：**更多层≠更好记忆**，这对边缘设备部署有实践意义——应用浅层宽嵌入架构
- 数据结构本身编码了有助于记忆的关系模式（序列优于三元组）
- 实用建议：医疗可穿戴设备上的小型Transformer应采用1-2层 + 大嵌入维度

## 局限与展望

- 未分析未学习样本的模式（可用curriculum learning或loss re-weighting改善）
- 未在更长序列和更大规模数据上验证
- 缺乏层间冗余/相似性的直接评估（可用probing和pruning研究）
- 未考虑量化和稀疏化对架构建议的影响
- 序列生成方式与临床推理模式的对齐有待加强
- 仅在SNOMED上测试，未验证其他生物医学图谱的泛化性

## 相关工作与启发

- Kim et al. (2023) 给出Transformer记忆容量理论上界$O(d+n+\sqrt{nN})$，本文经验验证
- Härmä et al. (2024) 用随机数字序列评估记忆能力，本文扩展到结构化知识图谱
- 可结合稀疏自编码器区分记忆与泛化，或用curriculum learning改善未学习样本

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes](../../CVPR2026/others/multi-view_crowd_tracking_transformer_with_view-ground_interactions_under_large_.md)
- [\[CVPR 2025\] Zero-Shot Head Swapping in Real-World Scenarios](../../CVPR2025/others/zero-shot_head_swapping_in_real-world_scenarios.md)
- [\[ICML 2025\] Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Settings](../../ICML2025/others/suitability_filter_a_statistical_framework_for_classifier_evaluation_in_real-wor.md)
- [\[ACL 2025\] Partial Colexifications Improve Concept Embeddings](partial_colexifications_improve_concept_embeddings.md)
- [\[CVPR 2026\] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos](../../CVPR2026/others/videoworld_2_learning_transferable_knowledge_from_real-world_videos.md)

</div>

<!-- RELATED:END -->
