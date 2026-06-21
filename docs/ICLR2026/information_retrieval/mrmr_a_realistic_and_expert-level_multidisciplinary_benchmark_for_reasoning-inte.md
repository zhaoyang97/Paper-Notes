---
title: >-
  [论文解读] MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval
description: >-
  [ICLR 2026][信息检索/RAG][多模态检索] MRMR 构建了首个面向**专家级、多学科、推理密集**场景的多模态检索基准——含 1,435 条横跨 23 个领域的查询，把查询和文档统一表示为图文交错序列，并首创"矛盾检索"任务，评测发现当前多模态检索模型在需要推理的任务上大幅落后于"文本检索器 + 图像描述"这种朴素方案。
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "多模态检索"
  - "推理密集检索"
  - "专家领域"
  - "矛盾检索"
  - "图文交错"
  - "MMMU-Pro"
---

# MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XZNXSM4rHG](https://openreview.net/forum?id=XZNXSM4rHG)  
**代码**: [https://github.com/rebeccaz4/MRMR](https://github.com/rebeccaz4/MRMR) · [数据集](https://huggingface.co/datasets/MRMRbenchmark)  
**领域**: 信息检索 / 多模态检索 / Benchmark  
**关键词**: 多模态检索, 推理密集检索, 专家领域, 矛盾检索, 图文交错, MMMU-Pro  

## 一句话总结
MRMR 构建了首个面向**专家级、多学科、推理密集**场景的多模态检索基准——含 1,435 条横跨 23 个领域的查询，把查询和文档统一表示为图文交错序列，并首创"矛盾检索"任务，评测发现当前多模态检索模型在需要推理的任务上大幅落后于"文本检索器 + 图像描述"这种朴素方案。

## 研究背景与动机
**领域现状**：以 DeepResearch 为代表的 LLM agent 越来越依赖外部检索来补足知识，而科学、医学、工程、金融等高价值场景的信息天然是多模态的（图像 + 文本），因此一个鲁棒的多模态检索组件成了刚需。已有多模态检索基准（InfoSeek、OVEN、ViDoRe、MMDocIR 等）虽有进展，但都不足以刻画真实 agentic 场景的复杂度。

**现有痛点**：作者指出三个系统性缺口——(1) **领域过窄**：多数基准建在 Wikipedia 文本/图像上，停留在通用知识，而 SOTA 的 LLM 早已能轻松搞定这类知识；(2) **缺乏推理**：现有任务主要是语义匹配和信息检索，很少需要对专家领域图像（如显微切片诊断）做深层理解和逻辑推理；(3) **模态单一**：大多只支持"单图 + 补充文本"的查询，而真实查询和文档往往是多图 + 图文交错的混合内容。

**核心矛盾**：检索模型在"语义匹配/找相似"上已经很强，但在"看懂专家图像 + 做逻辑推理判断相关性"上几乎是空白，而后者恰恰是 agentic 检索最需要的能力——现有基准无法把这个能力短板暴露出来。

**本文目标**：造一个能区分"会匹配"和"会推理"的多模态检索基准，覆盖多学科专家领域，并支持图文交错的真实输入形态。

**核心 idea**：**[数据来源升级]** 直接复用 MMMU-Pro 这类专家级 VQA 题目作为查询源（保证难度和领域广度），由人类专家从互联网采集并核验正样本文档；**[任务创新]** 在传统"找支持证据"之外首创 **Contradiction Retrieval（矛盾检索）**——要求模型找出与查询相冲突的规则/陈述，强制引入逻辑推理；**[形态统一]** 查询和文档都用图文交错序列表示，逼近真实使用场景。

## 方法详解

### 整体框架
MRMR 把多模态检索形式化为：给定查询集 $Q$ 和文档库 $D$，每个查询/文档都是文本片段与图像交错的序列 $(x_1,\dots,x_k)$；对查询 $q$，文档若为推理链提供支撑的原理/定理则记为正样本 $d^+$，否则为负样本 $d^-$。围绕"推理"这一核心，基准设计了三类任务——Knowledge（专家知识检索）、Theorem（定理检索）、Contradiction（矛盾检索），三类共 1,435 条查询、覆盖 23 个领域 / 6 大学科。数据构造统一走"从 MMMU-Pro 选题 → GPT-Search 检索候选 → OCR 切块保留图文 → GPT 初筛 + 专家复核 → 补充负样本"的半自动 + 人工核验流水线。

```mermaid
flowchart LR
    A[MMMU-Pro 专家题] --> B{GPT-5 分类}
    B -->|知识题| K[Knowledge 任务]
    B -->|计算题| T[Theorem 任务]
    C[COCO/DesignQA/驾驶手册] --> D[Contradiction 任务]
    K --> E[GPT-Search 检索网页<br/>OCR切块+专家核验]
    T --> F[BRIGHT定理库<br/>+爬取定理]
    D --> G[合成/规则文档<br/>引入冲突]
    E --> H[图文交错语料库]
    F --> H
    G --> H
    H --> I[14个前沿模型<br/>nDCG@10评测]
```

### 关键设计

**1. Knowledge 任务：用专家题反推"能帮你答题的网页"**。这条任务的关键在于"正样本难采集"——不像语义检索那样有现成图文对，得真的找到能支撑答案的多模态资料。作者先用 GPT-5 把 MMMU-Pro 题目分成知识题与计算题，并滤掉那些只靠浅层图文推理、不依赖外部专家知识就能答的题；保留题目后给每张图生成详细描述作为上下文。随后让 GPT-Search 围绕题目推理并给出带参考链接（Wikipedia、书籍、论文、博客）的解释，把这些网页截成 PDF、用 MonkeyOCR 抽取图文交错内容并切块（保留图像引用），再经 GPT-5 初筛 + 专家复核：GPT 与人类都判为相关的留作正样本，都判无关的作硬负样本，30–60% 的歧义样本直接丢弃；GPT-Search 没找到资料的题目（占 38.2%）由专家手工搜索并创建一篇支撑文档。最后从 PIN-14M（含 PMC 医学文献、OBELICS 网页）大规模采样补充负样本，凑成 26,223 篇的语料库。

**2. Theorem 任务：把"找解题定理"扩展到多模态计算题**。沿用 BRIGHT 的思路——给一道新的数理计算题，应能检索到求解所需的定理陈述。作者用 MMMU-Pro 的计算题作查询，先用 GPT-5 排除那些题面已经明说所需定理的题，再把剩余题归到 Math/Physics/Engineering/Business 四大领域，并让 GPT-5 逐题推理、给出答案并总结所用关键定理，剔除 GPT-5 答错的题，最终留 512 题。语料库以 BRIGHT 的约 13.8k 条定理陈述为主（反映"定理多为纯文本"的现实），用总结出的定理作查询、Qwen3-Embedding 检索 top-10 候选，由 GPT-5 从中挑出最相关者作正样本、其余作负样本；对 BRIGHT 里没有对应定理的，从 Wikipedia 等网页爬取定理（可带插图）并用 GPT-5 改写成 BRIGHT 格式，最终 63.6% 正样本来自网页爬取。

**3. Contradiction 任务：首创"找矛盾"，强制逻辑推理**。这是 MRMR 区别于所有已有基准的核心——查询是一个案例描述，文档库是一堆强制性规则，正样本是"被该案例违反"的那条规则，模型不仅要做语义匹配还要判断概念冲突。它由三个子任务组成：**Negation**（受 NegBench 启发的合成任务，给一张 COCO 图配四条描述，其中一条断言不存在的物体存在或存在的物体不存在，模型从四选一中挑出矛盾项，用 Hit@1 评测）；**Vehicle Design**（基于 Formula SAE 规则书和 DesignQA，给一个车辆设计案例，检索它违反的具体设计规范条款，例如轴距短于规定最小值）；**Traffic Case**（把官方驾驶手册切块作语料，挑出若干交通规则并配标注违规案例，再用 Qwen-Image 把文字要素替换成 AI 生成图像来增广，例如跟车仅 3 米远小于安全距离）。

**4. 四类检索范式 + 14 个前沿模型的统一评测协议**。为公平对比不同技术路线，作者把待评模型归为四类并统一处理多图输入：**T2T**（文本检索器 BGE-M3 / NV-Embed-v2 / Qwen3-Embedding 配 MLLM 生成的图像描述）；**IT2IT 双流融合**（EVA-CLIP / SigLIP / OpenCLIP / JinaCLIP，文本块拼成一个文本嵌入 $t$、图像纵向拼接成图像嵌入 $i$，按 MTEB 用融合嵌入 $e = t + i$ 打分）；**IT2IT 合并图像**（VISTA / E5-V / MM-Embed / VLM2Vec / Ops-MM-Embedding / GME-Qwen2-VL，因只支持单图输入故把多图拼接）；**T2I 文档即图像**（ColPali，把整篇多模态文档编码成截图、查询图像替换为 LLM 描述）。除 Negation 用 Hit@1 外，全部用 nDCG@10。

## 实验关键数据

### 主实验（nDCG@10，Avg 为 11 个子任务均值）

| 模型 | 类别 | Knowledge(Art) | Theorem(Eng.) | Contradiction(Traffic) | **Avg.** |
|------|------|------|------|------|------|
| **Qwen3-Embedding** | T2T | 71.9 | 42.9 | 54.2 | **54.1** |
| Ops-MM-Embedding | IT2IT 合并图 | 79.3 | 30.1 | 45.8 | 48.1 |
| Ops-MM-Embedding | T2I 文档即图 | 67.7 | 29.2 | 46.3 | 45.6 |
| NV-Embed-v2 | T2T | 70.7 | 32.9 | 42.2 | 44.8 |
| MM-Embed | IT2IT 合并图 | 65.6 | 27.4 | 34.9 | 39.8 |
| GME-Qwen2-VL | IT2IT 合并图 | 54.3 | 30.2 | 29.6 | 36.2 |
| ColPali | T2I | 36.1 | 13.5 | 18.2 | 25.2 |
| OpenCLIP | 双流融合 | 56.0 | 7.3 | 12.4 | 18.0 |
| E5-V | IT2IT 合并图 | 25.1 | 2.5 | 2.1 | 8.6 |

**最强是"文本检索器 + 图像描述"的 Qwen3-Embedding（54.1）**，比最强多模态模型 Ops-MM-Embedding（48.1）高 6.0 分；CLIP 式双流模型全面垫底（11.6–18.0）。

### 关键发现
- **多模态模型在推理任务上崩**：Ops-MM-Embedding 在 Knowledge 子任务平均 67.4，但 Theorem 跌到 37.4、Contradiction 跌到 36.6，落差近 30 分；问题主要出在推理能力而非领域知识。
- **Negation 接近随机**：四选一任务下，多数模型 Hit@1 低于 25%（≈随机），人能轻松看出的矛盾概念，连强文本嵌入模型也抓不住。
- **领域差异显著**：同为多模态模型，E5-V 仅 8.6 而 Ops-MM-Embedding 达 48.1；Art 任务靠"匹配视觉相似画作"就能成功，但医学影像需要识别底层病理/放射特征，难度远高。
- **失败模式两类**（30 例案例分析）：① 视觉偏置压过上下文相关性（线虫 SEM 图像像查询里的蚯蚓就被排高）；② 高阶推理失败（交通案例中模型看到都有车/隧道/车道线就给负样本高分，却推不出"压线"违反"保持车道"）。
- **测试时扩展有效**：用 MLLM 生成推理轨迹（问题摘要 + CoT）替换原查询，Qwen2-VL-2B 提升 +5.1、Qwen2.5-VL-72B 提升 +14.8，Knowledge 任务收益最大，但更大模型多花 20–66% token 成本。

## 亮点与洞察
- **首创 Contradiction Retrieval**：把"找支持证据"翻转成"找冲突证据"，是检索语义之外真正需要逻辑推理的设计，对法律合规、工程审查、安全驾驶等场景有直接价值。
- **暴露了一个反直觉结论**：在专家级推理场景下，端到端多模态嵌入模型还打不过"先把图转成文字描述、再用强文本检索器"的朴素管线——说明当前多模态嵌入的视觉推理能力被严重高估。
- **数据构造务实**：用 GPT-Search + OCR 切块 + GPT/人类双重核验的半自动流水线平衡了规模与质量，并明确丢弃 30–60% 的歧义样本来保证标注可靠性。
- **多学科细粒度评测**：23 领域 / 6 学科的拆分让人能看清同一模型在 Art 强、在 Medicine 弱这类"按领域取胜"的差异，对模型选型有指导意义。

## 局限与展望
- **语料规模可再扩**：作者承认语料还能通过采样更多专家领域文档来增大检索难度（也会提升假负样本概率），留作未来工作。
- **负样本假负风险**：从 PIN-14M 大规模采样负样本依赖"低假负概率"假设，虽做了人工误差分析但仍是潜在噪声源。
- **依赖 GPT 系列做核验**：选题、初筛相关性、改写定理都重度依赖 GPT-5 / GPT-Search，构造流程的偏差会传导进基准。
- **测试时扩展只做文本**：作者主要研究文本侧 query 扩展，图像侧的扩展/处理留白。
- **理想模型缺席**：原生图文交错检索模型 TIIR 最契合 MRMR 格式但未公开，导致最贴合该形态的方法无法评测。

## 相关工作与启发
- **推理密集检索**：纯文本侧的 BRIGHT 首次提出"需要推理才能判定相关性"的检索基准，MRMR 是其向多模态域的延伸，并直接复用了 BRIGHT 的定理库。
- **多模态检索基准谱系**：从语义匹配（NIGHTS/SciMMIR）、组合图像检索（FashionIQ/CIRR/CIRCO）、信息检索（InfoSeek/OVEN/ViDoRe/MMDocIR）到图文交错的 TIIR——MRMR 在"专家领域 + 推理 + 交错"三个维度同时拉满，是首个三者兼具的基准。
- **多模态嵌入模型**：评测覆盖 CLIP/BLIP 式双流到 MLLM 微调的统一嵌入（E5-V、MM-Embed、VLM2Vec、Ops-MM-Embedding 等），相当于给这条技术线做了一次专家级推理压力测试。
- **启发**：结果强烈暗示，下一代多模态检索器需要把"视觉推理"显式纳入训练目标（而非只对齐表层图文语义）；同时"caption + 文本检索"作为强基线，提醒社区在宣称多模态优势前先过这道门槛。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个"专家级 + 多学科 + 推理密集 + 图文交错"四合一的多模态检索基准，并首创矛盾检索任务，填补了真实 agentic 检索评测的空白。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 4 类检索范式 / 14 个前沿模型 / 23 领域细粒度评测，外加 30 例失败案例分析和测试时扩展实验，相当扎实；略欠在语料规模和图像侧扩展的探索。
- **写作质量**: ⭐⭐⭐⭐ 三类任务的动机—构造—评测线索清晰，表格和案例图丰富；数据构造细节多但组织得当。
- **价值**: ⭐⭐⭐⭐⭐ 暴露了"多模态嵌入打不过 caption+文本检索"这一反直觉短板，为多模态 RAG / DeepResearch 类系统的检索组件指明改进方向，数据与代码均开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval](retro_optimizing_llms_for_reasoning-intensive_document_retrieval.md)
- [\[ICLR 2026\] Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks](frustratingly_simple_retrieval_improves_challenging_reasoning-intensive_benchmar.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](../../ACL2026/information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] A Survey of Reasoning-Intensive Retrieval: Progress and Challenges](../../ACL2026/information_retrieval/a_survey_of_reasoning-intensive_retrieval_progress_and_challenges.md)

</div>

<!-- RELATED:END -->
