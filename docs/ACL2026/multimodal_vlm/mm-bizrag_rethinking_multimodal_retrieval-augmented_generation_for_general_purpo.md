---
title: >-
  [论文解读] MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A
description: >-
  [ACL2026][信息检索/RAG][MM-RAG] MM-BizRAG 证明企业多模态 RAG 不能只依赖页面截图和视觉 embedding，而应先按文档结构区分报告与幻灯片，再显式解析文本、表格和图片，并在推理时组装多模态上下文，从而在 SlideVQA、FinRAGBench-V 和内部企业数据上显著超过视觉中心 baseline。
tags:
  - "ACL2026"
  - "信息检索/RAG"
  - "MM-RAG"
  - "企业文档"
  - "结构感知解析"
  - "多模态检索"
  - "FastRAGEval"
---

# MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A

**会议**: ACL2026  
**arXiv**: [2606.04231](https://arxiv.org/abs/2606.04231)  
**代码**: 无（cache 未给出开源仓库）  
**领域**: 多模态检索增强生成 / 企业问答  
**关键词**: MM-RAG、企业文档、结构感知解析、多模态检索、FastRAGEval

## 一句话总结
MM-BizRAG 证明企业多模态 RAG 不能只依赖页面截图和视觉 embedding，而应先按文档结构区分报告与幻灯片，再显式解析文本、表格和图片，并在推理时组装多模态上下文，从而在 SlideVQA、FinRAGBench-V 和内部企业数据上显著超过视觉中心 baseline。

## 研究背景与动机
**领域现状**：企业知识库中常见 PDF、DOCX、PPTX、HTML、TXT 等文档，里面混合文本、表格、图像、图表和复杂版式。多模态 RAG 近期流行一种简化路线：把页面当图片，用视觉 embedding 做检索，再把页面图交给 VLM 生成答案。

**现有痛点**：页面图路线省掉了解析，但也把阅读顺序、表格结构、图文位置关系全部交给预训练 VLM 隐式理解。企业文档里的结构化信息往往不在通用模型训练分布内，尤其是财报、合规材料、技术文档和多页业务报告。

**核心矛盾**：RAG 检索需要轻量、稳定、可索引的 representation；答案生成又需要丰富、保留结构的多模态上下文。把同一种页面截图同时用于检索和生成，既可能检索不准，也可能丢掉表格和阅读顺序。

**本文目标**：作者希望构建一个不微调模型、可落地到异构企业文档的 MM-RAG 系统，显式处理结构化 artifact，并系统比较不同 ingestion representation 和 embedding 策略的影响。

**切入角度**：论文先按文档结构把 vertical documents（报告、PDF、filings）和 horizontal documents（slide decks）分流，再分别设计解析与 chunking。检索阶段用更适合索引的 representation，生成阶段再把原始 artifact 组装回来。

**核心 idea**：将“用于检索的表示”和“用于生成的上下文”解耦：检索时轻量索引文本/描述/页面表示，生成时依据 placeholder 和元数据恢复表格 markdown、图片和页面图，构造更接近原文结构的多模态证据。

## 方法详解

### 整体框架
MM-BizRAG 想解决的是企业知识库里那堆异构文档：PDF 财报、PPTX 幻灯片、DOCX 技术文档混在一起，光靠页面截图 + 视觉 embedding 会丢掉表格结构和阅读顺序。它的主线是 document structure-aware ingestion——先判断文档是纵向结构（报告、filings，有自然阅读顺序）还是横向结构（幻灯片，每页是完整语义单元），再分别走不同的解析路线。

对纵向文档，Docling 等工具抽出文本块、表格、图片和页面图：文本里插入表格/图片 placeholder 来锁住阅读顺序，表格转 markdown 后由 LLM 生成逐行描述，图片由 VLM 描述并过滤掉 logo 这类无信息内容。对横向文档则不强拆，每页保留页面图加一段 VLM 生成的 slide-level 描述。检索阶段按变体建立 text 或 multimodal embedding；推理时 query rewriter 先结合对话历史改写查询，再用 dense + BM25 hybrid retrieval，RRF 融合取 top 30 送进 LLM list-wise reranker，最终选 top 20 chunks 组装成多模态上下文交给 GPT-4.1 生成答案。论文据此定义三种变体（TCTE、PCMHE、TCMIE），差别在 chunk 粒度、embedding 模型和 artifact 注入时机。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["企业文档<br/>PDF / PPTX / DOCX / HTML / TXT"] --> B["结构感知文档分流<br/>纵向/横向分类器"]
    B -->|纵向：报告/财报| C["版式解析<br/>文本+占位符 / 表格→markdown→LLM描述 / 图片→VLM描述"]
    B -->|横向：幻灯片| D["整页表示<br/>页面图 + slide级VLM描述"]
    C --> E["索引侧轻量表示<br/>text / multimodal embedding（解耦·检索侧）"]
    D --> E
    E --> F["查询改写<br/>结合对话历史补全自含查询"]
    F --> G["混合检索<br/>dense 70 + BM25 100 → RRF top30"]
    G --> H["list-wise 重排<br/>取 top20 chunks"]
    H --> I["推理时上下文组装<br/>顺占位符还原表格markdown/图片（解耦·生成侧）"]
    I --> J["GPT-4.1 生成答案"]
```

### 关键设计

**1. 结构感知文档分流：报告和幻灯片不能用同一套解析**

报告依赖段落与表格的线性顺序，幻灯片依赖整页的空间布局，硬塞进一套 chunking 必然两头不讨好。MM-BizRAG 因此先做一个 vertical/horizontal 分类，纵向文档走 layout-aware parsing 保留文本阅读顺序、表格 markdown、图片描述和页面图，横向文档则把每页当成不可分割的语义单元、只用页面图加 VLM 描述表示。实验里这个分类器 precision 达 100、recall 83.28，分流足够可靠才让后续按结构定制的解析有意义。

**2. 检索表示与生成上下文解耦：索引用轻量描述，生成时再把原始 artifact 装回去**

表格 markdown 和图片 base64 体积大、直接全塞进索引会让向量库膨胀、检索也变慢，但生成答案时又确实需要这些原始证据。MM-BizRAG 把「用于检索的表示」和「用于生成的上下文」拆开：以 TCTE 变体为例，索引侧只用文本 chunk、表格描述和图片描述做 text embedding；当检索命中某条表格或图片描述时，系统顺着它的 placeholder 找回所在的父文本 chunk，再把表格 markdown、图片 artifact 和描述注入到 placeholder 的原始位置。这样索引保持精简、检索保持高效，而生成拿到的是接近原文结构的多模态证据——很多 RAG 系统把 index schema 和 prompt context schema 绑死，要么检索很重要么证据太贫乏，这里靠 inference-time assembly 同时绕开两个坑。

**3. FastRAGEval 单调用评测：用一次 LLM 调用算事实级 recall**

企业 QA 的答案往往是长段落，token overlap 和 exact match 衡量不了关键事实是否被召回；而 RAGChecker 那种两阶段 claim decomposition + matching 成本和延迟都高，撑不起大规模系统评测。FastRAGEval（FRE）把这件事压进一次 LLM 调用：同时抽取 reference 和 prediction 的 atomic facts 并算 precision、recall、F1，论文主要用 FRE recall。它和人类判断的相关性反而更高（Pearson 0.808 vs RAGChecker 0.748），说明单调用并没有牺牲一致性，却大幅降了 judge 成本。

### 一个完整示例：一个查询如何拿到表格证据
设用户对一份财报追问「Q3 海外营收同比变化」。query rewriter 先结合对话历史补全成自含查询；hybrid retrieval 里 dense 取 70 个、BM25 取 100 个 chunks，RRF 融合后 top 30 进 list-wise reranker。假设排在前面的命中是一条「该表展示分地区季度营收」的表格描述——此时系统并不会把这句干瘪描述直接喂给生成器，而是顺着它的 placeholder 回到父文本 chunk，把对应的表格 markdown（带具体数字的那张表）和上下文段落一并注入原位。最终 top 20 chunks 里就含有结构完整、数字可读的原始表格，GPT-4.1 据此作答，而不是去猜一张截图里的单元格。

### 损失函数 / 训练策略
本文不训练专用检索器或生成模型，重点全在 ingestion、retrieval 和 assembly 的设计上。用到的组件包括 Docling、EasyOCR、PyPdfium2、Tableformer、OpenAI text-embedding-3-large、nomic-multimodal-embed-3b、cohere-embed-v4、GPT-4.1 系列、ColPali 和 VisRAG-Ret。推理 pipeline 的固定档位是：dense retrieval 取 70 个 chunks、BM25 取 100 个 chunks，RRF 后 top 30 进 list-wise reranker，最终 top 20 用于答案生成。

## 实验关键数据

### 主实验
| Pipeline | SlideVQA FRE | FinRAGBench-V FRE | Internal FRE | 关键结论 |
|------|------|------|------|------|
| Text-Only | 67.8 | 60.3 | 83.7 | 在纯文本问题上强，但表格/图片弱 |
| ColPali | 83.6 | 49.3 | - | 幻灯片强于 text-only，报告式文档弱 |
| VisRAG | 78.8 | 46.0 | - | 同样受限于页面图中心路线 |
| TCTE (OAI v3-large) | 87.3 | 80.2 | 88.1 | 推荐生产配置，垂直文档延迟更低 |
| PCMHE (Nomic) | 89.9 | 79.6 | 87.6 | SlideVQA 最强 |
| PCMHE (Cohere) | 89.06 | 82.4 | 87.8 | FinRAGBench-V 最强 |
| TCMIE (Cohere) | 88.2 | 76.9 | 88.0 | 内部数据接近 TCTE |

### 消融实验
| 分析项 | 数值 / 现象 | 说明 |
|------|------|------|
| Vertical-horizontal classifier | Precision 100.00, Recall 83.28, F1 90.87 | 文档结构分流较可靠，但召回仍可提升 |
| FRE vs RAGChecker Pearson | 0.808 vs 0.748 | FRE 与人类判断相关性更高 |
| FRE vs RAGChecker Spearman | 0.808 vs 0.736 | 排序一致性更好 |
| FRE vs RAGChecker Kendall | 0.808 vs 0.725 | 单调用 metric 没牺牲一致性 |
| 人类标注一致性 | Cohen's kappa 0.966 | 200 个实例双标注高度一致 |
| TCTE 延迟 | FinRAGBench-V 11.9s, Internal 11.1s | 垂直文档约为 PCMHE Cohere 的一半 |

### 关键发现
- 在 SlideVQA 上，MM-BizRAG 最高 FRE recall 为 89.9，相比 ColPali 的 83.6 提升 6.3 个百分点；说明即便幻灯片适合视觉路线，显式文本/视觉融合仍有收益。
- 在 FinRAGBench-V 上，PCMHE Cohere 的 FRE recall 为 82.4，ColPali 只有 49.3，VisRAG 只有 46.0；报告式文档中页面图中心方法退化非常明显。
- 内部数据包含 1,908 个问题、1,048 个文档、20,429 页，MM-BizRAG 各变体 FRE recall 都高于 text-only 的 83.7。
- TCTE 是推荐生产配置：它在垂直文档上 recall 距最优通常只有 1-3 个百分点，但延迟大约只有 PCMHE 的一半。
- 所有 MM-BizRAG 变体的 faithfulness 超过 90%，说明结构化 assembly 并没有用更丰富上下文换来更多无依据生成。

## 亮点与洞察
- 论文反驳了“VLM 足够强后就不需要解析”的直觉。企业文档的表格、页眉页脚、跨页叙事和图文顺序仍需要显式建模。
- 检索表示和生成上下文解耦是非常工程化但重要的设计。很多 RAG 系统把 index schema 和 prompt context schema 绑死，导致要么检索很重，要么生成证据太贫乏。
- TCTE 的表现很有现实意义：最强配置不一定是最适合生产的配置，延迟、成本和 recall 的 trade-off 需要一起看。
- FastRAGEval 也很实用。企业 QA 的答案往往是长段落，用 token F1 或 exact match 没意义；单调用 fact-level recall 更接近业务评价。

## 局限与展望
- 公开 slide 评测主要依赖 SlideVQA，而该数据集相对简单，不能完全代表复杂企业级演示文稿。
- FinRAGBench-V 只处理了作者能拿到 PDF 格式的 213 个英文文档子集，没有覆盖完整 1,100+ 文档，也没有评估多语言企业场景。
- 公开基线只包括 ColPali 和 VisRAG 两个视觉中心系统，缺少更多近期或闭源企业 RAG 系统对比。
- 内部企业数据因隐私和组织约束不能发布，可复现性受限；未来若能发布匿名或合成版本会更有价值。
- 系统依赖 GPT-4.1 生成描述、重写 query、rerank 和回答，成本、速率限制和模型版本变化都会影响部署表现。

## 相关工作与启发
- **vs ColPali**: ColPali 用视觉语言模型做文档页面检索，适合视觉页面匹配；MM-BizRAG 显式恢复文本、表格、图片结构，在报告式文档上优势明显。
- **vs VisRAG**: VisRAG 也强调页面图和多模态检索，MM-BizRAG 则认为生成上下文需要 artifact-aware assembly，而不是只传页面图。
- **vs Text-only RAG**: Text-only 在文本密集问题上仍强，但面对表格和图片会掉点；MM-BizRAG 在不放弃文本优势的同时补上多模态证据。
- **启发**: 企业 RAG 的 ingestion 不应只问“用什么 embedding”，还要问“文档是什么结构、哪些 artifact 用于索引、哪些 artifact 在生成时恢复”。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 组件多为已有技术，但结构分流、placeholder alignment 和 inference-time assembly 的系统组合很有工程创新。
- 实验充分度: ⭐⭐⭐⭐☆ 内部大规模数据加两个公开 benchmark 支撑较强，但公开可复现和 baseline 范围有限。
- 写作质量: ⭐⭐⭐⭐☆ 系统设计讲得清楚，变体对比有用；企业系统细节较多，读者需要跟住符号和 pipeline。
- 价值: ⭐⭐⭐⭐⭐ 对企业级多模态 RAG 落地非常有参考价值，尤其提醒大家不要过早放弃显式文档解析。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation](utility-oriented_visual_evidence_selection_for_multimodal_retrieval-augmented_ge.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/information_retrieval/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/information_retrieval/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](../../ACL2025/information_retrieval/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)

</div>

<!-- RELATED:END -->
