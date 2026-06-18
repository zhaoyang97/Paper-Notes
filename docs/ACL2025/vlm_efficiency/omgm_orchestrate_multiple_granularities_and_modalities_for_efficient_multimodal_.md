---
title: >-
  [论文解读] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval
description: >-
  [ACL 2025][多模态VLM][多模态检索] 提出OMGM——一个面向知识密集型视觉问答(KB-VQA)的多模态RAG系统，通过粗到细三步检索策略协调查询与知识库在不同粒度和模态间的匹配，在InfoSeek和E-VQA上取得SOTA检索性能和极具竞争力的问答结果。 问题背景 知识密集型视觉问答(KB-VQA)要求系统不…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态检索"
  - "知识增强VQA"
  - "RAG"
  - "粒度对齐"
  - "多模态融合重排序"
  - "粗到细检索"
---

# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval

**会议**: ACL 2025  
**arXiv**: [2505.07879](https://arxiv.org/abs/2505.07879)  
**作者**: Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, Jiang Bian (Microsoft Research Asia)
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: 多模态检索, 知识增强VQA, RAG, 粒度对齐, 多模态融合重排序, 粗到细检索

## 一句话总结

提出OMGM——一个面向知识密集型视觉问答(KB-VQA)的多模态RAG系统，通过粗到细三步检索策略协调查询与知识库在不同粒度和模态间的匹配，在InfoSeek和E-VQA上取得SOTA检索性能和极具竞争力的问答结果。

## 研究背景与动机

### 问题背景
知识密集型视觉问答(KB-VQA)要求系统不仅理解图像内容，还需从外部知识库中检索图像主体的相关知识来回答问题。检索增强生成(RAG)是解决此类任务的主流方案，但多模态检索面临两大核心挑战：

**多模态性**：查询和知识库都包含图像与文本两种模态，需要灵活运用单模态、跨模态和多模态匹配策略

**多粒度性**：查询中图像提供粗粒度实体信息，问题指向细粒度知识；知识库中实体文章包含粗粒度概览图和摘要以及细粒度的详细章节

### 已有工作的不足
- **单步检索方法**（PreFLMR、MuKA等）：直接在全库上做多模态检索，需要昂贵的任务特定预训练，推理时全量搜索计算代价高
- **多步检索方法**（Wiki-LLaVA、EchoSight等）：采用层次化策略但忽略了检索步骤间模态与粒度的协调设计，各步骤的相似度分数未充分传递和融合
- 现有方法普遍缺乏对查询与知识库间粒度对齐的系统性思考

### 核心动机
设计一个能在每一步选择合适模态和粒度的粗到细多步检索流程，通过跨步骤相似度传播实现全局最优的实体定位和知识筛选。

## 方法详解

### 整体框架
OMGM采用三阶段粗到细检索策略：
1. **粗粒度跨模态实体搜索**：用查询图像匹配实体摘要，从百万级知识库中筛选top-k候选实体
2. **混合粒度多模态融合重排序**：用多模态融合特征对候选实体进行精排，结合上一步相似度分数选出最相关实体
3. **细粒度文本增强生成**：在选定实体内用文本重排序器筛选最相关章节，作为上下文送入生成器回答问题

### 关键设计1：粒度对齐的实体搜索
核心思想是让查询和检索索引在信息粒度上对齐。查询图像本身承载的是粗粒度的实体身份信息，因此检索索引也应是粗粒度的实体概述，而非完整文章或标题。

具体做法：
- **离线摘要生成**：用预训练语言模型对知识库中所有实体文章生成简洁摘要 $s_i = M_s(P, a_i)$，作为检索索引
- **图像-摘要匹配**：用CLIP的视觉编码器和文本编码器分别编码查询图像和实体摘要，通过FAISS进行内积相似度搜索，保留top-k个实体

消融实验验证了Image→Summary的检索效果显著优于Image→Article、Image→Image和Image→Title，证明粒度对齐的重要性。

### 关键设计2：多模态融合重排序与跨步骤相似度传播
在top-k候选实体中进行精排，需要利用查询的完整多模态信息（图像+问题）和候选的多模态信息（实体图像+章节文本）。

**多模态融合特征提取**：采用Q-Former架构将图像和文本融合为统一的特征矩阵：

$$Q = E_m(I_q, T_q), \quad C_{sec_e^h} = E_m(I_e, sec_e^h)$$

**Late Interaction细粒度匹配**：通过Max-Sum操作计算查询与候选在token级别的细粒度相似度：

$$sim_m^{sec_e^h} = \sum_{i=1}^{l_Q} \max_{j=1}^{l_C} Q_i {C_{sec_e^h}^j}^\top$$

**跨步骤相似度融合**：将第一步的实体相似度 $sim_c^e$ 与当前步的多模态相似度加权融合，选出最终实体：

$$e_{top1} = \arg\max_{e \in Ent_k} \left(\alpha \cdot sim_c^e + (1-\alpha) \cdot \max_h sim_m^{sec_e^h}\right)$$

训练时利用第一步检索的hard negative构建对比学习，正样本为正确实体的主图+证据章节，负样本为候选实体的主图+非证据章节。

### 关键设计3：细粒度章节筛选
在确定top-1实体后，结合文本重排序器和多模态重排序器的分数，筛选最相关章节：

$$sec_{e_{top1}}^{best} = \arg\max_{sec \in e_{top1}} \left(\beta \cdot sim_m^{sec} + (1-\beta) \cdot sim_t^{sec}\right)$$

最终将筛选出的章节作为上下文与查询一起输入下游生成器。

## 实验关键数据

### 检索性能对比

| 方法 | E-VQA R@1 | E-VQA R@20 | InfoSeek R@1 | InfoSeek R@20 |
|------|-----------|------------|--------------|---------------|
| CLIP I-T | 3.3 | 16.5 | 32.0 | 68.2 |
| Wiki-LLaVA | 3.3 | 13.2 | 36.9 | 71.9 |
| EchoSight (w. rerank) | 36.5 | 48.8 | 53.2 | 77.9 |
| ReflectiVA | 15.6 | 49.8 | 56.1 | 86.4 |
| **OMGM (w. rerank)** | **42.8** | **58.7** | **64.0** | **84.8** |

OMGM在E-VQA的R@1上比EchoSight高出6.3个百分点，在InfoSeek的R@1上比ReflectiVA高出7.9个百分点。重排序步骤单独贡献了E-VQA上23.7%和InfoSeek上11.4%的R@1提升。

### VQA性能对比

| 方法 | 生成器 | Gen. FT | E-VQA | InfoSeek Overall |
|------|--------|---------|-------|------------------|
| Wiki-LLaVA | LLaVA-1.5-7B | ✓ | 21.8 | 28.9 |
| mR2AG | LLaVA-1.5-7B | ✓ | - | 40.2 |
| ReflectiVA | LLaVA-MORE-8B | ✓ | 35.5 | 40.1 |
| **OMGM** | LLaVA-1.5-7B | ✓ | **50.17** | **43.49** |
| OMGM | InternVL-2.5-8B | ✗ | 48.72 | 36.1 |

OMGM在微调LLaVA-1.5-7B下，E-VQA达到50.17，InfoSeek达到43.49，均为最佳。值得注意的是，仅微调检索器（不微调生成器）的OMGM仍超过多数微调生成器的方法。

### 效率对比

| 方法 | 平均检索时间(s) | 平均推理时间(s) | E-VQA VQA结果 |
|------|---------------|---------------|--------------|
| LLaVA-1.5-7B | - | 1.432 | 17.00 |
| PreFLMR | 0.984 | 2.196 | 54.45 |
| **OMGM** | **0.402** | 2.023 | **63.39** |

多步检索反而比单步PreFLMR更快（0.4s vs 0.98s），因为每步只搜索缩小后的候选空间。

## 关键发现

- **粒度对齐至关重要**：Image→Summary检索R@20达58.7%，远超Image→Article的41.7%和Image→Image的48.8%，说明查询与索引的信息粒度匹配是有效检索的前提
- **多模态融合双端优于单端**：查询端和候选端都用多模态融合的(I,T)→(I,T)模式R@1达40.2%，大幅优于纯文本T→T的30.7%
- **每步检索递进提升**：三步分别贡献VQA提升——第一步16.25%、第二步14.36%、第三步2.0%（E-VQA），第二步多模态重排序贡献最大
- **跨步相似度传播有效**：将前序步骤的相似度分数传递并融合到后续步骤，比各步独立计算更稳健

## 亮点与洞察

- **系统性的粒度-模态协调设计**：不是简单堆叠多步检索，而是在每一步精心选择合适的查询模态、索引模态和信息粒度，形成互补的检索链路
- **轻量高效**：仅需训练一个Q-Former重排序器，不依赖大规模生成器微调即可大幅提升检索和VQA性能；多步检索通过逐步缩小搜索空间反而比全量单步检索更快
- **跨步骤信息流**：通过相似度分数传播实现各检索步骤的协同，而非孤立地串联多个检索器
- **摘要作为检索索引的洞察**：用LLM生成的实体摘要替代原始文档或标题作为索引，既压缩了信息又保留了与查询图像的语义对齐，这一思路可推广到其他RAG场景

## 局限性

- **未利用知识库的辅助图像**：实体文章中常包含与特定章节关联的细粒度图片，当前方法仅使用主图，忽略了这些潜在的多模态线索
- **生成器端未深度集成**：检索到的多模态融合特征仅作为文本上下文传入生成器，未探索如何将融合特征直接注入生成过程以进一步提升回答质量
- **摘要生成的质量依赖**：第一步的检索效果高度依赖LLM生成的摘要质量，对于小众或专业领域的实体，摘要质量可能下降
- **仅在Wikipedia风格知识库上验证**：虽然在OK-VQA上做了泛化测试，但整体框架对非结构化或非百科全书式知识库的适用性有待进一步验证
- **重排序范围k的扩展性**：k从20增至100时检索持续提升但时间也线性增长，大规模知识库下如何高效扩展重排序范围未深入讨论

## 相关工作与启发

- **EchoSight** (Yan & Xie 2024)：同为多步检索，但未设计粒度对齐和跨步传播机制，OMGM在此基础上系统化地优化了每一步的模态-粒度选择
- **PreFLMR** (Lin et al. 2024)：单步多模态全量检索，编码成本高；OMGM证明分步检索可同时提升效果和效率
- **ReflectiVA** (Cocchi et al. 2024)：通过反思式token驱动检索迭代，侧重生成器与检索器的交互；OMGM则侧重检索链路内部的粒度-模态协调
- **启发**：粒度对齐的思想可迁移到文档级RAG——先用文档摘要粗筛，再用段落级重排序精选，最后用句子级抽取，每步使用最适合的表示与匹配方式

## 评分

- 新颖性: ⭐⭐⭐⭐ — 粗到细多粒度多模态协调的系统性设计有明确创新，但各单元技术（CLIP检索、Q-Former重排、对比学习）均为成熟组件
- 实验充分度: ⭐⭐⭐⭐⭐ — 两个主流数据集+OK-VQA泛化测试，消融覆盖每个设计选择，效率分析完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，方法描述系统化，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 在KB-VQA检索上效果显著，粒度对齐的设计理念对通用多模态RAG有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MMOne: Representing Multiple Modalities in One Scene](../../ICCV2025/multimodal_vlm/mmone_representing_multiple_modalities_in_one_scene.md)
- [\[ICLR 2026\] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](../../ICLR2026/multimodal_vlm/multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)
- [\[ACL 2025\] Progressive Multimodal Reasoning via Active Retrieval](progressive_multimodal_reasoning_via_active_retrieval.md)
- [\[NeurIPS 2025\] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)
- [\[ACL 2025\] Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](con_instruction_universal_jailbreaking_of_multimodal_large_language_models_via_n.md)

</div>

<!-- RELATED:END -->
