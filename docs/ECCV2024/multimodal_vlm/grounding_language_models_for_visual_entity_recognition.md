---
title: >-
  [论文解读] Grounding Language Models for Visual Entity Recognition
description: >-
  [ECCV 2024][信息检索/RAG][视觉实体识别] 提出 AutoVER——首个将多模态大语言模型（MLLM）应用于大规模视觉实体识别的方法，通过将检索能力集成到 MLLM 内部，结合对比训练和前缀树约束解码，在 Oven-Wiki 基准上大幅超越 PaLI-17B 等先前方法。 - 视觉实体识别（VER）的挑战：…
tags:
  - "ECCV 2024"
  - "信息检索/RAG"
  - "视觉实体识别"
  - "多模态大语言模型"
  - "检索增强生成"
  - "约束解码"
  - "知识接地"
---

# Grounding Language Models for Visual Entity Recognition

**会议**: ECCV 2024  
**arXiv**: [2402.18695](https://arxiv.org/abs/2402.18695)  
**代码**: [https://github.com/MrZilinXiao/AutoVER](https://github.com/MrZilinXiao/AutoVER)  
**领域**: 信息检索  
**关键词**: 视觉实体识别, 多模态大语言模型, 检索增强生成, 约束解码, 知识接地

## 一句话总结

提出 AutoVER——首个将多模态大语言模型（MLLM）应用于大规模视觉实体识别的方法，通过将检索能力集成到 MLLM 内部，结合对比训练和前缀树约束解码，在 Oven-Wiki 基准上大幅超越 PaLI-17B 等先前方法。

## 研究背景与动机

- **视觉实体识别（VER）的挑战**：
  1. 答案空间超过 600 万 Wikipedia 实体，分类器方法不可行
  2. 生成式 VQA 模型容易产生幻觉——生成的文本可能不对应任何合法实体
  3. 现有方法忽略了候选实体的图像信息
  4. 对域外未见实体的泛化困难
- **Oven-Wiki 基准**：给定图像+问题，从 Wikipedia 中找到精确实体答案
- **核心思路**：将实体识别重构为受约束的序列到序列生成问题

## 方法详解

### 整体框架

AutoVER = MLLM（基于 LLaVA/Vicuna）+ 多模态实体编码器 + 检索增强约束解码

### 关键设计

**1. 联合对比-生成训练**
- MLLM 侧：添加特殊 token `<ret>`，其最后一层隐状态作为查询表示 Q
- 实体编码器侧：2 层 Transformer 融合实体图像和文本描述，输出实体表示 E
- 对比损失（InfoNCE）：$\mathcal{L}_{query2ent} = -\frac{1}{N}\sum \log \frac{\exp(sim(Q_i, E_i)/\tau)}{\sum_j \exp(sim(Q_i, E_j)/\tau)}$
- 语言建模损失（next token prediction）：$\mathcal{L}_{LM}$
- 总损失：$\mathcal{L} = \mathcal{L}_{LM} + \lambda_r \cdot \mathcal{L}_{query2ent}$

**2. 硬负例挖掘**
- **vision-hard**：使用预训练 ViT 分类器识别视觉相似实体（共享预测类别）
- **kb-hard**：利用 Wikidata 类别层次结构，共享父节点的实体为知识相似实体
- 通过 rejection sampling 避免同 batch 内出现重复实体

**3. 检索增强约束解码（推理阶段）**
- 用训练好的实体编码器预缓存所有实体向量 → 构建实体向量数据库
- 推理时用 `<ret>` token 的表示执行 top-k 相似度搜索，检索 k=300 个候选
- 动态构建**前缀树（trie）**覆盖候选实体标识符
- 自回归生成时，前缀树约束每一步的合法 token，消除无效解码路径
- **保证生成内容一定匹配知识库中的实体**

### 损失函数 / 训练策略

- 初始化：LLaVA 架构，Vicuna-7B/13B, CLIP-ViTL/14-336px
- 训练数据：Oven-Wiki ~500 万 query-entity 对
- 32× V100 GPU，batch size 256
- λ_r = 1，实体描述截断至 77 tokens

## 实验关键数据

### 主实验（Oven-Wiki 验证集准确率）

| 方法 | Entity seen | Entity unseen | Entity hm | Query seen | Query unseen | Overall hm |
|------|-------------|---------------|-----------|------------|--------------|------------|
| CLIP Fusion | 32.7 | 4.3 | 7.7 | 33.4 | 2.2 | 5.4 |
| PaLI-17B | 30.6 | 12.4 | 17.6 | 44.2 | 22.4 | 22.1 |
| GPT-4V (zero-shot) | 29.8 | 19.3 | 23.4 | 56.5 | 52.7 | 32.9 |
| **AutoVER-7B** | **61.5** | **21.7** | **32.1** | **69.0** | **31.4** | **36.8** |
| **AutoVER-13B** | **63.6** | **24.5** | **35.6** | **68.6** | **32.3** | **39.2** |

### 消融实验

- 去掉对比训练 → entity seen 下降约 10%
- 去掉约束解码 → entity unseen 下降显著
- 去掉硬负例挖掘 → 细粒度识别能力下降

### 关键发现

- Entity seen 准确率从 PaLI-17B 的 30.6% 翻倍至 61.5%（参数量更少）
- 约束解码**消除幻觉**：保证生成内容对应真实实体
- 在 A-OKVQA-Ent 零样本转移中表现优异，证明泛化能力
- 人类评估集上 AutoVER-13B 达到 53.7%（Human+Search 为 77.7%，仍有差距）

## 亮点与洞察

1. **统一检索+生成框架**：检索能力内置于 MLLM，无需外部检索器
2. **前缀树约束解码**确保生成结果可接地（grounded），彻底杜绝幻觉
3. **对比+生成联合训练**兼顾检索精度和生成质量
4. 硬负例挖掘策略（视觉+知识库双路）有效提升细粒度区分能力

## 局限与展望

- Entity unseen 子集准确率仍较低（21.7%），域外泛化仍是主要瓶颈
- 实体数据库预缓存需要大量存储和 top-k 搜索开销
- 前缀树在候选集极大时（>万级）可能面临效率问题
- Query split 的推理需求（空间关系、常识）仍有提升空间

## 相关工作与启发

- FROMAGe/GILL 启发了 MLLM 中添加检索 token 的设计
- 生成式实体链接（GENRE）为文本域的知识接地提供了思路
- 可启发：将约束解码框架推广到其他需要结构化输出的任务（如知识图谱补全、结构化信息抽取）

## 评分

- 新颖性：⭐⭐⭐⭐⭐（统一 RAG 框架 + 约束解码消除幻觉）
- 技术深度：⭐⭐⭐⭐⭐
- 实验充分度：⭐⭐⭐⭐⭐（多 split + 零样本转移 + 消融）
- 写作质量：⭐⭐⭐⭐
- 综合推荐：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Towards Open-Ended Visual Recognition with Large Language Model](towards_open-ended_visual_recognition_with_large_language_models.md)
- [\[ECCV 2024\] Multi-Label Cluster Discrimination for Visual Representation Learning](multi-label_cluster_discrimination_for_visual_representation_learning.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](../../ACL2026/information_retrieval/quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[CVPR 2025\] EZSR: Event-based Zero-Shot Recognition](../../CVPR2025/information_retrieval/ezsr_event-based_zero-shot_recognition.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)

</div>

<!-- RELATED:END -->
