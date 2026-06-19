---
title: >-
  [论文解读] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation
description: >-
  [ICML 2025][视频理解][长视频描述] 提出 SGVC 框架，通过将视频各段的文本描述解析为场景图、用 Hungarian 算法迭代合并为统一图表示、再用轻量图到文本解码器生成视频级描述，以极低计算开销实现了超越 LLM-based 方法的零样本长视频描述性能。 视觉-语言模型（VLM）在图片和短视频描述上已经取得…
tags:
  - "ICML 2025"
  - "视频理解"
  - "长视频描述"
  - "场景图"
  - "图合并"
  - "零样本视频描述"
  - "图到文本生成"
---

# Fine-Grained Captioning of Long Videos through Scene Graph Consolidation

**会议**: ICML 2025  
**arXiv**: [2502.16427](https://arxiv.org/abs/2502.16427)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 长视频描述, 场景图, 图合并, 零样本视频描述, 图到文本生成

## 一句话总结

提出 SGVC 框架，通过将视频各段的文本描述解析为场景图、用 Hungarian 算法迭代合并为统一图表示、再用轻量图到文本解码器生成视频级描述，以极低计算开销实现了超越 LLM-based 方法的零样本长视频描述性能。

## 研究背景与动机

视觉-语言模型（VLM）在图片和短视频描述上已经取得了优秀的效果，但生成长视频的连贯、全面的描述仍然是一个重大挑战。核心矛盾在于：现有模型的时间感受野有限，无法一次性编码整个长视频。

对此，现有解决方案分为三类，但各有缺陷：

**Memory-based / 递归框架**：需要在目标数据集上进行监督微调，泛化性差

**LLM-based 汇总方法**（如 VidIL、Video ChatCaptioner）：利用 LLM 总结多片段信息，无需微调但**推理开销巨大**（需调用商用 GPT API、参数量 7B+），且 LLM 有时会忽略场景细节或产生幻觉

**零样本方法**（如 ZeroCap、MAGIC）：通过 CLIP 引导语言模型，但在复杂事件视频上效果不佳

本文的切入角度非常独特：**不要用大语言模型做文本级别的汇总，而是在结构化的场景图层面做信息融合**。每段视频描述被解析为场景图（对象+属性+关系），多个场景图通过图合并算法融合为统一表示，最后用一个仅 235M 参数的轻量解码器生成最终描述。这样既保留了对象级细粒度信息，又避免了 LLM 的高计算开销。

## 方法详解

### 整体框架

SGVC（Scene Graph-based Video Captioning）框架由四个阶段组成：
1. **段级描述生成**：将视频均匀分为多个片段（帧或短视频段），使用现成的 VLM（如 BLIP、BLIP2、InternVL2.5）为每段生成描述
2. **场景图解析**：用文本场景图解析器（FACTUAL-MR parser）将每段描述转换为场景图
3. **场景图合并**：迭代合并所有段级场景图为统一图表示
4. **视频描述生成**：用图到文本模型将合并后的场景图解码为最终的视频级描述

整个框架是 **training-free** 的——无需在目标长视频数据集上微调任何组件，可搭配任意现成 VLM 使用。

### 关键设计

1. **场景图定义与解析**：场景图 $G = (\mathcal{O}, \mathcal{E})$ 由对象集和边集组成。每个对象 $o_i = (c_i, \mathcal{A}_i)$ 包含类别标签和属性集合，有向边 $e_{i,j}$ 带有关系标签 $r_{i,j}$。使用 FACTUAL-MR parser 将文本描述先映射为中间语义表示（对象、属性、关系），再确定性地转换为场景图。这种结构化表示比纯文本更适合做信息融合——同一对象在不同帧的出现可以被精确匹配和合并。

2. **场景图合并算法**：这是方法的核心。合并过程是迭代式的：每轮选择图集合中**最相似的一对图**进行合并，直到只剩一个统一图。

   两图合并的具体步骤：
    - 使用**图编码器** $\phi(\cdot)$ 对两个图进行编码，得到每个对象的嵌入表示
    - 通过 **Hungarian 算法**求解最优对象匹配，目标函数基于对象嵌入的**余弦相似度**：
    $\pi^* = \arg\max_{\pi \in \Pi} \sum_i \frac{\psi_i(\phi(G^s))}{\|\psi_i(\phi(G^s))\|} \cdot \frac{\psi_i(\phi(G_\pi^t))}{\|\psi_i(\phi(G_\pi^t))\|}$
    - 对于相似度超过阈值 $\tau$ 的匹配对 $(o_p^s, o_q^t)$，将两个对象**合并为一个**：$\hat{o}_m = (\hat{c}, \mathcal{A}_p^s \cup \mathcal{A}_q^t)$，其中类别标签 $\hat{c}$ 可能与原始标签不同（通过编码器推断）
    - 更新合并图的边集：将原来连接到被合并对象的边重定向到新的合并对象

   数量不同的对象通过引入 dummy objects 来对齐后再做匹配。阈值 $\tau$ 在 0.80-0.95 范围内性能稳定，实验统一使用 0.9。

3. **优先子图提取**：当需要简洁的视频描述时，在合并过程中跟踪每个节点的**合并计数**作为重要性度量，选取 top-k 个合并次数最多的节点及其子图。高合并计数意味着该对象在视频多个帧中反复出现，大概率是关键实体。k=1 产生最精简子图（利于精确率指标），k 增大则保留更多上下文（利于召回率指标）。

4. **图到文本模型**：

    - **图编码器**：基于 BERT-base，使用**注意力掩码**限制注意力仅在场景图定义的边上传播（而非全局注意力），保留图结构信息。额外加入一个可学习的 **全局 embedding token** 使得断开的子图之间也能交换信息
    - **文本解码器**：使用 T5-base 的解码器部分
    - 总参数量仅 **235M**（对比 Mistral-7B 的 7.5B）

### 损失函数 / 训练策略

- 图到文本模型使用 **next-token prediction** 目标训练：$\mathcal{L}(\theta) = \sum_{i=1}^{N} \log P_\theta(t_i | t_{1:i-1}, G)$
- 训练数据：约 **250 万**图-文本对，来自 MS-COCO、Flickr30k、TextCaps、Visual Genome 等图像描述数据集，以及用 LLaVA-NeXT-7B 为 Kinetics-400 视频生成的描述
- 训练 1K 迭代，batch size 512，AdamW 优化器，learning rate 0.0001
- 视频段落描述任务进一步在 Visual Genome Paragraph Captions 上微调 400 迭代
- 推理使用 beam search（5 beams，max length 32，length penalty 0.6）

## 实验关键数据

### 主实验

零样本视频描述（MSR-VTT 和 MSVD）：

| 方法 | Backbone | B@4 | METEOR | CIDEr | F_BERT |
|------|----------|-----|--------|-------|--------|
| VidIL (零样本) | BLIP+CLIP | 3.2 | 14.8 | 3.1 | 0.225 |
| Video ChatCaptioner | BLIP2 | 13.2 | 22.0 | 16.5 | 0.436 |
| **SGVC（本文）** | **BLIP2** | **18.4** | **23.1** | **26.1** | **0.487** |
| VidIL†（few-shot） | BLIP+CLIP | 13.6 | 20.0 | 20.2 | 0.490 |

在 MSR-VTT 上，SGVC 的零样本 CIDEr 分数（26.1）甚至超过了使用了参考描述做 few-shot 的 VidIL†（20.2）。

零样本视频段落描述（ActivityNet Captions）：

| 方法 | Backbone | B@4 | METEOR | CIDEr | F_BERT |
|------|----------|-----|--------|-------|--------|
| Video ChatCaptioner | BLIP2 | 2.4 | 8.9 | 1.6 | 0.200 |
| Summarization w/ GPT-4o mini | InternVL2.5 | 5.8 | 11.4 | 15.3 | 0.336 |
| **SGVC（本文）** | **InternVL2.5** | **8.0** | **13.2** | **24.1** | **0.338** |

在长视频段落描述任务上，SGVC 的 CIDEr（24.1）远超 GPT-4o mini 总结方法（15.3），提升超过 57%。

### 消融实验

subgraph extraction 的 k 值影响（MSR-VTT，BLIP2 backbone）：

| k值 | METEOR | CIDEr | P_BERT | R_BERT | F_BERT |
|-----|--------|-------|--------|--------|--------|
| 1 | 23.1 | **26.1** | **0.467** | 0.542 | **0.487** |
| 3 | **23.8** | 24.9 | 0.454 | **0.554** | 0.486 |

合并阈值 $\tau$ 的影响（MSVD，稳定性分析）：

| τ | CIDEr | F_BERT |
|---|-------|--------|
| 0.95 | 50.0 | 0.589 |
| 0.90 | **50.2** | **0.589** |
| 0.85 | 49.9 | 0.589 |
| 0.80 | 49.9 | 0.589 |

### 关键发现

- **计算效率优势显著**：SGVC（BLIP backbone）仅需 0.74B 参数、5.07GB 显存、1.14s/视频，而 Mistral-7B 总结需要 7.5B 参数、14.5GB 显存、1.27s/视频。SGVC 用不到 1/10 的参数取得了更好的效果
- **场景图合并 vs LLM 汇总**：在完全相同的段级描述输入下，场景图合并在 CIDEr 上大幅领先 LLM 总结（24.0 vs 10.8 on MSR-VTT with BLIP），说明结构化表示的信息保持能力远优于纯文本汇总
- **LLM 方法的幻觉问题**：Video ChatCaptioner 通过多轮 QA 聚合信息，经常产生幻觉（如"公园场景中没有动物"），而基于场景图的方法通过结构化表示有效避免了这个问题
- **Backbone 灵活性**：框架可即插即用不同 VLM（BLIP、BLIP2、InternVL2.5），更强的 backbone 带来一致的性能提升

## 亮点与洞察

1. **"结构化中间表示"的思路极其优雅**：与其让 LLM 在文本空间做模糊的总结，不如将文本转为结构化的场景图，在图空间做精确的对象匹配和合并。这一思路避免了 LLM 丢失细节和产生幻觉的问题
2. **Hungarian 算法保证最优匹配**：不同于简单的文本相似度比较，Hungarian 算法在全局最优意义上匹配两个图中的对象，这对正确关联跨帧实体至关重要
3. **轻量设计哲学**：235M 参数的图到文本模型 + CPU 可运行的图合并算法，相比动辄 7B+ 的 LLM 方案，真正做到了"以小博大"
4. **训练仅需文本数据**：图到文本模型训练只需图-文本对，完全不需要视频-文本配对数据，这大大扩展了可用训练数据的规模

## 局限与展望

1. **依赖文本场景图解析器质量**：FACTUAL-MR parser 的解析精度直接影响下游效果，若描述中的实体和关系解析不准确，合并后的图质量会下降
2. **场景图合并是在文本空间进行**：对象匹配依赖预训练图编码器的语义表示，可能忽略视觉层面的相似性（如同一个人穿不同衣服可能匹配失败）
3. **图到文本模型的生成复杂度受限**：235M 参数的模型在生成更长、更复杂的段落描述时可能力不从心
4. **未考虑时序信息**：场景图合并算法基于相似度而非时序顺序选择合并对，可能导致时间线信息丢失
5. **图合并在 CPU 运行**：虽然目前已经很快，但 GPU 实现可进一步加速

## 相关工作与启发

- **场景图在视频描述中的新用法**：场景图以前主要用于视觉关系检测和视觉问答，本文将其作为跨段信息融合的"中间表示"是一个新颖且有效的应用角度
- **与 Memory-based 方法的互补性**：本文的图合并策略可以作为 Memory-based 方法的即插即用替代方案，用于任何需要聚合多段视频信息的场景
- **启发**：这种"先结构化、再合并、后生成"的 pipeline 可以推广到其他多文档/多模态信息融合任务，例如多文档摘要、多视角场景理解等

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](../../CVPR2025/video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](../../ECCV2024/video_understanding/finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)

</div>

<!-- RELATED:END -->
