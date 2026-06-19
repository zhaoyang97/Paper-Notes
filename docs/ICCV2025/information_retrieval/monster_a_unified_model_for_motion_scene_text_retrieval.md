---
title: >-
  [论文解读] MonSTeR: a Unified Model for Motion, Scene, Text Retrieval
description: >-
  [ICCV 2025][信息检索/RAG][三模态检索] 提出 MonSTeR：——首个运动-场景-文本三模态检索模型：，通过受拓扑深度学习启发的高阶关系建模，构建统一隐空间以捕获三模态之间的内在依赖关系，在多项检索任务上大幅超越仅依赖单模态表征的基线，并可用于人-场景交互模型的评估。 人类在复杂环境中活动时需要平衡意图与环…
tags:
  - "ICCV 2025"
  - "信息检索/RAG"
  - "三模态检索"
  - "运动-场景-文本"
  - "高阶关系建模"
  - "对比学习"
  - "人-场景交互评估"
---

# MonSTeR: a Unified Model for Motion, Scene, Text Retrieval

**会议**: ICCV 2025  
**代码**: [GitHub](https://github.com/colloroneluca/MonSTeR)  
**领域**: 信息检索  
**关键词**: 三模态检索, 运动-场景-文本, 高阶关系建模, 对比学习, 人-场景交互评估  
**作者**: Luca Collorone, Matteo Gioia 等 (Sapienza University of Rome, Technion/NVIDIA)

## 一句话总结

提出 **MonSTeR**——首个**运动-场景-文本三模态检索模型**，通过受拓扑深度学习启发的高阶关系建模，构建统一隐空间以捕获三模态之间的内在依赖关系，在多项检索任务上大幅超越仅依赖单模态表征的基线，并可用于人-场景交互模型的评估。

## 研究背景与动机

人类在复杂环境中活动时需要平衡意图与环境所提供的可能性。例如，"坐在椅子上"这个意图加上相应的动作，如果环境中没有椅子，就会显得不合理。这揭示了**意图（文本）、运动（动作）、环境（场景）**三者之间的强内在一致性。

然而，现有研究存在明显缺口：

**文本-运动检索模型**（TMR、MoPa）无法整合环境上下文，运动脱离场景存在

**人-场景交互模型的评估**缺乏全局一致性/真实性度量，往往拆分为碰撞检测、目标距离等独立指标，忽视了路径合理性、运动可信度等重要方面

**现有多模态对齐方法**要么仅做两两对齐，要么通过共享编码器处理所有模态但缺乏显式的跨模态交互建模

核心挑战：如何在统一的隐空间中有效表征三模态的多对多关系——同一文本可对应多种运动，同一运动在不同场景下可能有不同含义。

## 方法详解

### 整体架构

MonSTeR 构建于三种编码器之上：

1. **单模态编码器**：基于 Transformer 的变分自编码器，分别编码文本 $t$、运动 $m$、场景 $s$，输出隐变量 $v_t$, $v_m$, $v_s$
2. **跨模态编码器**：将单模态编码器的输出 token 两两拼接，通过跨模态编码器生成联合隐变量 $v_{st}$, $v_{mt}$, $v_{ms}$

### 数据表示

- **文本** $t \in \mathbb{R}^{768}$：DistilBERT 特征
- **场景** $s \in \mathbb{R}^{N \times 6}$：彩色点云 (x,y,z + RGB)
- **运动** $m \in \mathbb{R}^{T \times 3 \times 22}$：T 帧×3D 坐标×22 关节

### 高阶关系的拓扑建模

受拓扑深度学习启发，将三模态关系建模为包含节点-边-面的拓扑结构：

- **节点** $\mathcal{V} = \{t, s, m\}$：单模态表征
- **边** $\mathcal{E} = \{ts, sm, mt\}$：跨模态表征
- **面** $\mathcal{P} = \{tsm\}$：三模态全局关系

通过对齐单模态与单模态、单模态与跨模态来编码高阶关系：
- $(st, m)$：场景-文本跨模态与运动对齐
- $(mt, s)$：运动-文本跨模态与场景对齐
- $(ms, t)$：运动-场景跨模态与文本对齐

### 训练目标

对比学习损失中包含的项集合：
$$K = \{(t,s), (m,t), (m,s), (st,m), (mt,s), (ms,t)\}$$

排除了可能导致退化解的项（如 $(st, t)$ 或 $(st, s)$），以防止跨模态编码器学习恒等函数。

对每对 $(i,j) \in K$，计算 $N \times N$ 余弦相似度矩阵 $C_{i,j}$，用 InfoNCE 聚合：
$$\mathcal{L}_{\text{tot}} = \frac{1}{|K|} \sum_{(i,j) \in K} \frac{\mathcal{L}_{\text{NCE}}(C_{i,j})}{N}$$

### 检索推理

统一隐空间支持灵活检索：
- **双模态→单模态**：st2m, ms2t, mt2s（给定两个模态检索第三个）
- **单模态→双模态**：m2st, t2ms, s2mt（给定一个模态检索两个）
- **单模态→单模态**：t2m, m2t, s2m, m2s, t2s, s2t

## 实验关键数据

### 主实验：HUMANISE+ 检索结果（All 协议，mRecall）

| 方法 | st2m | m2st | ms2t | t2sm | tm2s | s2mt | 平均 |
|------|------|------|------|------|------|------|------|
| TMR + S | 4.10 | 3.30 | 5.81 | 4.79 | 1.08 | 1.98 | 2.72 |
| MoPa + S | 2.10 | 2.45 | 1.62 | 1.94 | 3.28 | 3.06 | 1.88 |
| **MonSTeR** | **13.91** | **13.14** | **8.46** | **10.39** | **4.09** | **4.45** | **4.80** |

MonSTeR 在 st2m 上相对最佳基线提升 **209%**，平均 mRecall 超越最佳场景感知模型 **76.47%**。

### 消融实验：高阶关系建模的必要性

| 变体 | st2m | m2st | t2m | m2t | 平均 |
|------|------|------|-----|-----|------|
| **MonSTeR** | 13.91 | 13.14 | 3.62 | 3.11 | 5.63 |
| w/o cross-modal | 5.20 | 3.77 | 4.35 | 3.21 | 3.79 |
| w/o single | 11.91 | 12.93 | 0.22 | 0.29 | 4.36 |
| w tri-modal | 6.14 | 6.00 | 4.16 | 4.37 | 4.14 |

| 变体 | Small Batches 平均 |
|------|-------------------|
| **MonSTeR** | **60.00** |
| w/o cross-modal | 56.07 |
| w/o single | 41.77 |
| w tri-modal | 53.70 |

关键发现：
- 移除跨模态编码器后双模态→单模态任务大幅下降
- 移除单模态项后单模态→单模态任务几乎崩溃
- 三模态共享编码器方案不如分离的跨模态编码器

### 路径合理性评估

将测试集运动旋转 0 到 $\pi$ 弧度后，MonSTeR 的 FID 和 Recall 均持续恶化（符合预期），且包含碰撞的运动得分更低——证明 MonSTeR 隐空间**内化了运动不应穿透场景物体**的先验。

### 用户研究

MonSTeR 的排序与人类偏好的一致率达 **66.5%**（1122 标注，224 评估者）。

### 下游任务：运动标注

| 方法 | BLEU 1 | BLEU 4 | ROUGE L | CIDER | BERT F1 |
|------|--------|--------|---------|-------|---------|
| MotionGPT | 42.16 | 17.47 | 40.23 | 11.13 | 22.16 |
| **MonSTeR + GPT2** | **42.93** | **23.59** | **50.85** | **13.70** | **35.57** |

MonSTeR 的嵌入用于运动标注时在 BLEU 4（+6.12）、ROUGE L（+10.62）、BERT F1（+13.41）上大幅超越 MotionGPT。

### 零样本场景物体放置

利用 mt2s 评分在 5×5×5 网格中定位物体，平均误差仅 **18 cm**（随机基线 58.98 cm），证明 MonSTeR 在零样本设定下具备精确的空间推理能力。

## 亮点与洞察

1. **首创三模态检索**：运动-场景-文本的三模态统一隐空间此前从未被探索，MonSTeR 填补了这一空白
2. **拓扑启发的高阶建模**：不同于简单的两两对齐或全模态混合编码，通过边-节点对齐来编码高阶关系，理论有拓扑学支撑
3. **评估功能**：MonSTeR 可替代传统的碰撞检测+距离度量，提供整体一致性评分，且与人类判断对齐
4. **灵活的检索能力**：支持 12 种检索任务组合，包括给定单/双模态检索单/双模态
5. **零样本迁移能力**：场景物体放置和运动标注的零样本/下游实验证明了隐空间的丰富语义

## 局限性

1. **仅在对齐数据上训练**：跨模态编码器仅使用配对数据，未利用非配对数据的潜力
2. **静态场景假设**：人类动作不会改变场景布局，限制了对动态交互的建模
3. **数据集规模**：HUMANISE+（19.6K 样本）和 TRUMANS+（15 小时动捕）规模仍然有限
4. **TRUMANS+ 上 TMR 仍在 t2m 任务占优**：说明当场景信息不具区分性时，专门的双模态模型可能更优

## 相关工作与启发

- **与 TMR/MoPa 的关系**：MonSTeR 通过引入场景维度扩展了文本-运动检索，将其推广到三模态
- **与 CLIP 类多模态对齐的差异**：CLIP 系方法多选择一个参考模态进行对齐，MonSTeR 实现了全对全对齐
- **对 HSI 评估的启示**：传统碰撞+距离指标可被单一的三模态一致性评分取代
- **与拓扑深度学习的联系**：将 Bodnar et al. 2021 的拓扑理论应用于多模态对齐，是一种新颖的跨领域思路

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个运动-场景-文本三模态检索模型，拓扑启发的高阶建模思路独特
- **实验**: ⭐⭐⭐⭐ — 12 种检索任务全面评估，消融到位，下游应用多样（标注、物体放置、用户研究）
- **写作**: ⭐⭐⭐⭐ — 结构清晰，Beatles 引言有趣，拓扑动机阐述到位
- **价值**: ⭐⭐⭐⭐ — 开辟了三模态检索新方向，对人-场景交互的评估具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ACL 2026\] SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation](../../ACL2026/information_retrieval/skmteb_slovak_massive_text_embedding_benchmark_and_model_adaptation.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](../../ICLR2026/information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[ACL 2025\] SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](../../ACL2025/information_retrieval/saferag_benchmarking_security_in_retrieval-augmented_generation_of_large_languag.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)

</div>

<!-- RELATED:END -->
