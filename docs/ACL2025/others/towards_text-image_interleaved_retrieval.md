---
title: >-
  [论文解读] Towards Text-Image Interleaved Retrieval
description: >-
  [ACL 2025][interleaved retrieval] 定义文本-图像交错检索（TIIR）新任务，构建基于 wikiHow 的首个 TIIR 基准数据集（155K 文档、7654 测试对），并提出 Matryoshka Multimodal Embedder（MME）通过多粒度视觉 token 压缩解决 MLLM 中视觉 token 过多导致的效率和语义偏差问题，大幅提升检索性能。
tags:
  - "ACL 2025"
  - "interleaved retrieval"
  - "多模态"
  - "visual token compression"
  - "Matryoshka embedding"
  - "MLLM"
---

# Towards Text-Image Interleaved Retrieval

**会议**: ACL 2025  
**arXiv**: [2502.12799](https://arxiv.org/abs/2502.12799)  
**代码**: [https://github.com/vec-ai/wikiHow-TIIR](https://github.com/vec-ai/wikiHow-TIIR)  
**领域**: 其他  
**关键词**: interleaved retrieval, multimodal retrieval, visual token compression, Matryoshka embedding, MLLM

## 一句话总结
定义文本-图像交错检索（TIIR）新任务，构建基于 wikiHow 的首个 TIIR 基准数据集（155K 文档、7654 测试对），并提出 Matryoshka Multimodal Embedder（MME）通过多粒度视觉 token 压缩解决 MLLM 中视觉 token 过多导致的效率和语义偏差问题，大幅提升检索性能。

## 研究背景与动机

**领域现状**：现有多模态信息检索主要针对单图输入（query 或 document 最多一张图），无法满足教程、手册等需要多张图片交错排列的真实场景。

**现有痛点**：单图检索模型无法处理交错内容；直接将多图拼接为单图会丢失交错语境信息；用文本描述 caption 替代图片则丢失视觉语义。

**核心矛盾**：虽然 MLLM 天然支持交错输入，但每张图产生数百个视觉 token（如 576 个），多图场景下序列过长导致：(1) 计算效率低；(2) 视觉信息在 embedding 空间中占比过大；(3) 超出上下文长度后被截断丢失关键语义。

**本文目标** 定义 TIIR 任务、构建基准数据集、提出有效的 TIIR 模型。

**切入角度**：借鉴 Matryoshka 嵌套表示学习的思想，对视觉 token 进行多粒度压缩。

**核心 idea**：用 Matryoshka 风格的视觉 token 压缩打造高效的交错多模态检索器。

## 方法详解

### 整体框架
1. **任务定义**：query 和 document 都是文本-图像交错序列 $X = [x_1, ..., x_n]$（$x_i$ 可以是文本块或图像），目标是从语料库中检索与 query 最相关的文档
2. **基准构建**：基于 wikiHow 教程文章构建 155K 交错文档语料库 + 自动生成交错查询管线 + 人工标注测试集
3. **模型**：以 DeepSeek-VL 为 backbone 的 DPR 基线 → 在此基础上提出 MME 视觉 token 压缩

### 关键设计

1. **wikiHow-TIIR 数据集构建**

    - **文档构建**：从 wikiHow 教程中提取目标、步骤标题和对应图片组成交错文档
    - **查询生成管线**（三阶段）：
        - (a) 用 Idefics3-8B 对文档图片生成 caption → Qwen2.5-72B 基于文本和 caption 生成文本查询
        - (b) BM25 找最具信息量的句子 → LLM 选择实体/动作转化为图片 caption → 重写查询文本去除已图片化的信息
        - (c) FLUX.1-dev 文生图 → 与重写后文本合并形成交错查询
    - **测试集标注**：人工审核 10K 对，过滤非法内容、不合理内容、图文不一致等，保留 7654 对高质量query-document pair

2. **DPR 基线**

    - 使用 DeepSeek-VL-1.3B 作为 backbone，天然支持交错输入
    - [EOS] 状态作为序列级 embedding
    - InfoNCE 对比学习损失训练

3. **Matryoshka Multimodal Embedder (MME)**

    - 在 MLLM 的视觉投影后加入平均池化层，将 $24 \times 24 = 576$ 个视觉 token 压缩为 $N \times N$ 个
    - $N \in \{1, 2, 3, 4, 6, 8, 12, 24\}$，推理时可灵活选择
    - **三种训练策略**：
        - Random (Rand)：每个 micro-batch 随机采样一个 N
        - Matryoshka Learning (MRL)：同时训练所有 M 个粒度，加权求和损失
        - Mean Learning (Mean)：计算所有 $M \times M$ 种 query-document 大小组合的损失均值

### 损失函数 / 训练策略

- InfoNCE 对比学习损失：$\mathcal{L} = -\log \frac{\exp(s(X^Q, X_+^D)/\tau)}{\sum_{i=1}^N \exp(s(X^Q, X_i^D)/\tau)}$
- 温度 $\tau = 0.05$，batch size 32，学习率 $5 \times 10^{-5}$
- in-batch 负样本 + 1 个随机 hard negative
- 训练 3 个 epoch，线性 warmup

## 实验关键数据

### 主实验

| 模型 | 类型 | Recall@5 | MRR@10 | nDCG@10 |
|------|------|----------|--------|---------|
| VISTA | 单图拼接 | 45.06 | 33.73 | 35.22 |
| GME-Qwen2-VL-2B | 单图拼接 | 65.85 | 51.65 | 54.06 |
| MM-Embed | 单图拼接 | 68.73 | 53.67 | 56.37 |
| BGE-v1.5 | 文本+caption | 39.66 | 29.14 | 30.56 |
| GTE-Qwen2-7B | 文本+caption | 47.24 | 35.28 | 36.85 |
| Fine-tuned CLIP | 向量融合 | 69.41 | 54.73 | 57.15 |
| DPR baseline | 原生交错 | 74.79 | 60.87 | 63.28 |
| **MME (N=3)** | **原生交错** | **77.40** | **63.40** | **65.91** |

- MME 相比 DPR baseline：Recall@5 +2.61、MRR@10 +2.53、nDCG@10 +2.63
- 所有非交错模型均不如原生交错模型，说明交错上下文建模的必要性

### 消融实验

1. **交错上下文有效性验证**：打乱图片顺序、打乱图片位置、同时打乱两者，均导致显著性能下降（图4），证明交错上下文被有效建模
2. **适配策略有效性**：将现有模型适配至 TIIR 后，多数情况下反而不如纯文本检索（表3），说明简单适配策略（拼接、caption替换）会引入噪声
3. **Visual Document 适配**：GME 的截图模式有效（高于纯文本），因为保留了交错信息结构

### 关键发现

1. **最优视觉 token 数**：性能呈倒U形曲线，N=3（每图9个token）时最优；token太少丢失语义，太多导致视觉信息在 embedding 空间中占主导地位
2. **训练策略**：Mean Learning > MRL > Random，Mean 训练的跨粒度损失计算提供了更好的泛化
3. **视觉主导性分析**：N=3 时视觉与文本信息最平衡，分布最对称
4. **文本 caption 替代**：用 caption 替代图片后文本检索器性能反而上升（BGE 从 29.14→44.55），说明 caption 虽丢失视觉语义但文本检索能力更强；而多模态检索器适配后下降则说明图片拼接策略引入了噪声

## 亮点与洞察

1. **新任务定义**：TIIR 是首个关注查询和文档都包含交错文本-图像的检索任务，贴近真实 RAG 场景
2. **精心的数据构建**：三阶段自动生成管线 + 人工标注，平衡了规模和质量
3. **Matryoshka 压缩**：将 Matryoshka 嵌套表示学习思想巧妙应用于视觉 token 压缩，推理时灵活调节粒度
4. **深入分析**：5 个研究问题逐一实验验证（交错上下文有效性、适配策略、视觉模态、token数量、训练策略）

## 局限与展望

1. 当前仅用 DeepSeek-VL-1.3B 作为 backbone，未验证更大 MLLM（如 7B+）的效果
2. 训练数据全部来自 wikiHow 教程，领域单一，泛化到其他交错场景（如论文、产品页面）待验证
3. 每张图像仍需经过 ViT 编码，多图场景计算开销仍然较大
4. 查询生成管线依赖文生图模型（FLUX.1），生成图片质量和真实性有限

## 相关工作与启发

- **Matryoshka Representation Learning**：嵌套表示学习的思想启发了多粒度视觉 token 压缩
- **ColPali / Visual Document Retrieval**：截图模态检索的新范式，表3证明其在保留交错结构方面有效
- **DeepSeek-VL**：原生支持交错输入的 MLLM，是构建 TIIR 模型的理想 backbone
- **启发**：随着 RAG 场景越来越复杂，检索器需要从单图单模态走向多模态交错，视觉 token 压缩是关键挑战

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| **总评** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)

</div>

<!-- RELATED:END -->
