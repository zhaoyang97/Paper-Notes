---
title: >-
  [论文解读] Mitigating Semantic Collapse in Partially Relevant Video Retrieval
description: >-
  [NeurIPS 2025][模型压缩][部分相关视频检索] 针对部分相关视频检索（PRVR）中的语义坍塌问题，提出文本相关性保持学习和跨分支视频对齐（CBVA）方法，在文本和视频嵌入空间中分别解决坍塌现象，显著提升检索准确率。 - PRVR 任务定义：部分相关视频检索，即给定文本查询，检索那些 仅部分内容匹配：的视频（而非…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "部分相关视频检索"
  - "语义坍塌"
  - "跨模态对齐"
  - "对比学习"
  - "token合并"
---

# Mitigating Semantic Collapse in Partially Relevant Video Retrieval

**会议**: NeurIPS 2025

**arXiv**: [2510.27432](https://arxiv.org/abs/2510.27432)

**代码**: 有（论文提及 Code is available）

**领域**: Video Retrieval / Cross-modal Learning

**关键词**: 部分相关视频检索, 语义坍塌, 跨模态对齐, 对比学习, token合并

## 一句话总结

针对部分相关视频检索（PRVR）中的语义坍塌问题，提出文本相关性保持学习和跨分支视频对齐（CBVA）方法，在文本和视频嵌入空间中分别解决坍塌现象，显著提升检索准确率。

## 研究背景与动机

- **PRVR 任务定义**：部分相关视频检索，即给定文本查询，检索那些 **仅部分内容匹配** 的视频（而非整个视频都与查询相关）
- **语义坍塌问题**：
    - **文本端坍塌**：同一视频的不同文本标注被强制拉近，即使它们描述的是视频中完全不同的事件
    - **视频端坍塌**：同一视频中不同事件的片段嵌入被压缩到一起，失去了事件级别的区分度
    - **跨视频问题**：不同视频中语义相似的查询/片段却被推远（因为不同视频被视为负样本）
- **根因分析**：现有方法将所有标注的文本-视频对视为正样本、其余全为负样本，忽略了：
  1. 视频内部的语义多样性
  2. 跨视频的语义相似性

## 方法详解

### 整体框架

该框架包含三个核心模块，分别解决文本端和视频端的语义坍塌：

1. **Text Correlation Preservation Learning（TCPL）**：保持文本嵌入的语义关系
2. **Cross-Branch Video Alignment（CBVA）**：跨分支对比对齐视频表示
3. **Order-Preserving Token Merging + Adaptive CBVA**：增强视频片段的内部一致性和相互区分度

### 关键设计

#### 1. Text Correlation Preservation Learning（TCPL）

- **问题**：对比学习会破坏基础模型（如 CLIP）编码的文本间语义关系
- **解决方案**：引入知识蒸馏损失，保持训练后文本嵌入之间的相对距离
- 计算冻结基础模型的文本相似度矩阵 $S_{\text{teacher}}$
- 训练时约束当前模型的文本相似度矩阵 $S_{\text{student}}$ 接近 $S_{\text{teacher}}$

$$\mathcal{L}_{\text{TCPL}} = \text{KL}(S_{\text{teacher}} \| S_{\text{student}})$$

#### 2. Cross-Branch Video Alignment（CBVA）

- **设计动机**：视频表示需要在不同时间尺度上分层建模
- **双分支架构**：
    - **细粒度分支**：提取短时间窗口的 clip-level 特征
    - **粗粒度分支**：提取长时间跨度的 segment-level 特征
- **对比对齐**：两个分支对同一时间段的表示应一致，不同时间段应区分

$$\mathcal{L}_{\text{CBVA}} = -\sum_{i} \log \frac{\exp(s_{i,i}^{fg}/\tau)}{\sum_j \exp(s_{i,j}^{fg}/\tau)}$$

其中 $s_{i,j}^{fg}$ 是细粒度分支第 $i$ 段和粗粒度分支第 $j$ 段的相似度。

#### 3. Order-Preserving Token Merging

- **目标**：在合并 token 以降低计算量的同时，保持视频片段的时序顺序
- **方法**：按时间顺序分组 token，组内取均值合并
- **保证**：合并后的 token 序列保持原始时序结构
- **Adaptive CBVA**：根据片段间的相似度动态调整对比损失的权重

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L} = \mathcal{L}_{\text{retrieval}} + \alpha \mathcal{L}_{\text{TCPL}} + \beta \mathcal{L}_{\text{CBVA}}$$

- $\mathcal{L}_{\text{retrieval}}$：标准的文本-视频对比检索损失
- $\alpha, \beta$：平衡超参数
- 端到端训练，基于 CLIP 预训练特征初始化

## 实验关键数据

### 主实验

#### TVR 数据集上的 PRVR 结果

| 方法 | R@1↑ | R@5↑ | R@10↑ | R@100↑ | SumR↑ |
|------|------|------|-------|--------|-------|
| MS-SL | 13.5 | 32.2 | 43.8 | 83.4 | 172.9 |
| PSVL | 14.8 | 34.7 | 46.1 | 85.2 | 180.8 |
| GMMFormer | 15.2 | 35.4 | 47.3 | 86.1 | 184.0 |
| DL-DKD | 16.1 | 37.8 | 49.2 | 87.5 | 190.6 |
| MGCN | 16.8 | 38.5 | 50.1 | 88.0 | 193.4 |
| **Ours** | **18.7** | **41.3** | **53.6** | **90.2** | **203.8** |

#### ActivityNet Captions 数据集结果

| 方法 | R@1↑ | R@5↑ | R@10↑ | R@100↑ | SumR↑ |
|------|------|------|-------|--------|-------|
| MS-SL | 7.1 | 21.8 | 34.2 | 75.6 | 138.7 |
| PSVL | 7.8 | 23.5 | 36.1 | 77.3 | 144.7 |
| GMMFormer | 8.2 | 24.7 | 37.5 | 78.8 | 149.2 |
| DL-DKD | 8.9 | 26.3 | 39.1 | 80.2 | 154.5 |
| MGCN | 9.3 | 27.1 | 40.2 | 81.0 | 157.6 |
| **Ours** | **10.5** | **29.8** | **43.1** | **83.5** | **166.9** |

**发现**：在两个主要 PRVR 基准上均取得显著提升，R@1 提升约 11-13% 相对值。

### 消融实验

#### 各组件贡献（TVR 数据集）

| 配置 | R@1↑ | R@5↑ | R@10↑ | SumR↑ |
|------|------|------|-------|-------|
| Baseline | 15.2 | 35.4 | 47.3 | 184.0 |
| + TCPL | 16.5 | 37.2 | 49.5 | 191.2 |
| + CBVA | 17.1 | 38.8 | 51.2 | 196.1 |
| + Token Merging | 17.8 | 40.1 | 52.3 | 199.5 |
| + Adaptive CBVA | **18.7** | **41.3** | **53.6** | **203.8** |

**发现**：每个组件都有稳定的增量贡献，CBVA 贡献最大（+1.9 R@1），其次是 TCPL（+1.3 R@1）。

### 关键发现

1. **语义坍塌是 PRVR 的核心瓶颈**：通过可视化 t-SNE，清楚地看到 baseline 中同一视频不同事件的嵌入完全重叠
2. **文本端保持先验很重要**：TCPL 通过保持 CLIP 的文本相关结构，避免了对比训练对文本语义的破坏
3. **跨分支对齐有效解耦**：CBVA 使细粒度和粗粒度表示在保持一致性的同时获得更好的事件区分
4. **Token 合并的双重好处**：既降低了计算成本，又通过聚合产生了更稳定的片段表示
5. **稳定的跨数据集提升**：在 TVR 和 ActivityNet 上均有一致提升，说明方法的通用性

## 亮点与洞察

- **问题定义精准**："语义坍塌"这一概念清晰地揭示了 PRVR 的核心瓶颈
- **双端治理**：同时在文本和视频空间解决坍塌，比单独解决任一端更有效
- **知识蒸馏思路巧妙**：用冻结的基础模型作为 teacher 保持文本语义结构，成本低效果好
- **层次化视频建模**：双分支架构天然适合处理视频中不同时间尺度的事件

## 局限与展望

1. **计算开销**：双分支架构增加了约 40% 的参数量和计算量
2. **强负样本挖掘**：论文未深入探讨跨视频语义相似样本的利用
3. **更多模态**：仅使用视觉和文本，未考虑音频信号对事件分割的辅助作用
4. **更长视频**：实验视频长度有限（几分钟），对小时级别超长视频的效果未知
5. **与 Video LLM 对比**：缺少与最新 Video-LLM（如 VideoChat2）的对比

## 相关工作与启发

- **MS-SL**（Dong et al., 2022）：多尺度片段学习，PRVR 的早期方法
- **GMMFormer**（Chen et al., 2023）：基于 GMM 的视频分段方法
- **DL-DKD**（Yang et al., 2024）：解耦学习+知识蒸馏
- **Token Merging**（Bolya et al., 2023）：ToMe 方法，本文引入顺序保持变体
- **启发**：语义坍塌问题也可能存在于其他粗粒度检索任务中（如 passage retrieval with partial matches）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 语义坍塌问题定义精准，TCPL+CBVA 组合新颖 |
| 技术深度 | 4 | 双分支对比学习+自适应机制设计细致 |
| 实验充分性 | 4 | 两个数据集+完整消融+可视化分析 |
| 实用价值 | 3.5 | PRVR 任务相对小众，但技术可推广 |
| 写作质量 | 4 | 问题清晰，方法表达准确 |
| **总评** | **4.0** | 扎实的视频检索工作 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Adaptive Video Distillation: Mitigating Oversaturation and Temporal Collapse in Few-Step Generation](../../CVPR2026/model_compression/adaptive_video_distillation_mitigating_oversaturation_and_temporal_collapse_in_f.md)
- [\[NeurIPS 2025\] Explaining and Mitigating Crosslingual Tokenizer Inequities](explaining_and_mitigating_crosslingual_tokenizer_inequities.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](one-step_diffusion-based_image_compression_with_semantic_distillation.md)

</div>

<!-- RELATED:END -->
