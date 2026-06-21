---
title: >-
  [论文解读] MobileKGQA: On-Device KGQA System on Dynamic Mobile Environments
description: >-
  [ICLR 2026][图学习][KGQA] MobileKGQA 把高维 LLM embedding 压成二进制哈希码喂给 GNN 推理模块，再配一套逐步推理的标注自动生成方法，让知识图谱问答系统第一次能在手机/边缘设备上**直接训练并适应不断累积的用户数据**，在 Jetson Orin Nano 上以 30.4% 的能耗换来 20.3% 的性能提升。
tags:
  - "ICLR 2026"
  - "图学习"
  - "KGQA"
  - "端侧训练"
  - "embedding hashing"
  - "分布漂移"
  - "标注自动生成"
---

# MobileKGQA: On-Device KGQA System on Dynamic Mobile Environments

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SjLo76SpoW](https://openreview.net/forum?id=SjLo76SpoW)  
**代码**: [https://github.com/jyahn215/mobileKGQA](https://github.com/jyahn215/mobileKGQA)  
**领域**: 图学习 / 知识图谱问答 / 端侧推理  
**关键词**: KGQA, 端侧训练, embedding hashing, 分布漂移, 标注自动生成  

## 一句话总结
MobileKGQA 把高维 LLM embedding 压成二进制哈希码喂给 GNN 推理模块，再配一套逐步推理的标注自动生成方法，让知识图谱问答系统第一次能在手机/边缘设备上**直接训练并适应不断累积的用户数据**，在 Jetson Orin Nano 上以 30.4% 的能耗换来 20.3% 的性能提升。

## 研究背景与动机
**领域现状**：手机里大量用户信息（社交关系、活动日志、位置偏好）以知识图谱（KG）形式存储并持续累积，端侧 LLM 若能借助 KGQA 检索这些结构化知识，就能给出更个性化、可解释、少幻觉的回答。

**现有痛点**：现有 KGQA 系统全都不适合端侧部署——IR-based 模型要存高维 embedding 且反复构子图、算力高；SP-based 模型在新关系出现时常生成不可执行的逻辑式；LLM-based 的 ICL 方法要反复调用 LLM、延迟极高；微调 LLM 类方法则要调数十亿参数。

**核心矛盾**：用户数据持续累积会引发剧烈的**分布漂移**，导致 KGQA 性能随时间衰减，因此系统必须能持续适应新数据。但适应需要重训，而重训通常在中心化服务器上进行，既受端侧资源约束，又把私人数据上传带来隐私风险。两个无法回避的研究问题随之产生：(1) 如何把 KGQA 的训练资源砍到边缘设备能承受的水平？(2) 如何在不泄露数据的前提下为新知识生成监督训练所需的标注？

**本文目标**：做出首个能在手机上**直接部署且直接训练**的 KGQA 系统，用极少参数（0.136M）对抗资源约束，并自带适应分布漂移的能力。

**核心 idea**：**[哈希压缩 + 自生成标注]** 用「最大化原始 embedding 与哈希码互信息」的哈希模块把 GB 级浮点 embedding 压成几 MB 的二进制码，让 GNN 推理在常数算力下完成；再用「逐步推理拆解」的标注生成方法本地造出 question-answer 对，从而完全在设备上闭环训练与适应。

## 方法详解

### 整体框架
MobileKGQA 分三个阶段：**哈希阶段**把问题与关系的 LLM embedding 投影成保语义的二进制哈希码；**检索阶段**用基于 GNN 的推理模块在哈希码上预测答案候选并抽取连接路径，交给端侧 LLM 选最优路径生成答案；**适应阶段**在新数据到来时用自动标注生成方法造监督信号，重训哈希与推理模块以对抗分布漂移。

```mermaid
flowchart LR
    A[问题/关系 embedding<br/>EM 模型] --> B[哈希模块<br/>MLP→BN→tanh→sgn]
    B --> C[二进制哈希码 h∈{-1,1}]
    C --> D[GNN 推理模块<br/>ReaRev]
    KG[知识图谱 G] --> D
    D --> E[答案候选 + 最短推理路径]
    E --> F[端侧 LLM 选路径生成答案]
    G[新数据累积<br/>分布漂移] -.触发.-> H[逐步标注生成<br/>采样→口语化→合并→遮蔽→生成精炼]
    H -.监督信号.-> B
    H -.监督信号.-> D
```

### 关键设计

**1. 保余弦相似度的 embedding 哈希：用互信息把浮点压成二进制。** 哈希模块要把高维浮点 embedding $z\in\mathbb{R}^D$ 压成二进制码 $h\in\{-1,1\}^d$ 同时不丢语义，作者把它形式化为「在 $h=\psi(\phi(z))$ 约束下最大化互信息 $I(z,h)$」。由于二进制是离散优化，他们拆成两段可微映射——先用 MLP 做低维映射 $\phi$、再用批归一化把分布拉到零中心使哈希空间分布更均匀、最后 $d=\tanh(\mathrm{BN}(\mathrm{MLP}_\phi(z)))$、$h=\mathrm{sgn}(d)$ 二值化。理论上他们证明（Theorem 1）：在「embedding 的模长与方向独立」假设下，只要 $\phi$ 和 $\psi$ 都**保持任意 embedding 对之间的余弦相似度**，互信息 $I(z,h)$ 就达到最大。于是优化目标落到一个 log-ratio 回归损失上，把映射前的余弦相似度作为回归目标去逼近：$\mathcal{L}_{hash}=\ell_\phi(Z_a,Z_i,Z_j)+\alpha\ell_\psi(d_a,d_i,d_j)$。这套设计的妙处在于：哈希码体积只有原 embedding 的 0.25%，且训练算力与 embedding 模型大小**完全解耦**——不像 SOTA 的 GNN-RAG 要在训练时反复重算 embedding，因此可以放心换用更大的多语种 embedding 模型。

**2. 哈希码上的 GNN 推理 + 最短路径检索。** 拿到问题哈希码 $h_q$、关系哈希码 $h_r$ 和图 $G$ 后，推理模块（复用 ReaRev 架构，利用图结构作归纳偏置）输出全图节点表示 $H=\{h_i\}_{i=1}^N=\mathrm{RM}(h_q,h_r,G)$，以答案节点为标签用 KL 散度训练：$\mathcal{L}_{reason}=D_{KL}(Q\,\|\,\mathrm{softmax}(WH))$。预测出答案候选 $\hat a_j$ 后，从问题实体 $e_q$ 到候选之间抽取**最短路径集合** $P$ 作为推理路径喂给 LLM。选最短路径不是随意为之：一方面前人工作已证明最短路径是有效的启发式，另一方面路径长度增加会让搜索空间指数膨胀，限定最短路径相当于在固定检索预算下最优化召回。因为推理全在低维二进制码上做，整个检索阶段的计算与存储开销都被压到极低。

**3. 逐步推理的标注自动生成：本地造 QA 对对抗漂移。** 新知识累积后没有现成的 question-answer 标注，而本地资源又不允许像前人那样多次调 LLM 或调云端 LLM。作者把「为新三元组造问题」拆成五个可解释小步：先**采样并过滤**推理路径（用 tokenize 后 token-字符比偏高识别并丢弃加密/非自然字符实体）；再把过滤后的三元组**口语化**成自然语句让 LLM 更好处理；接着**合并**成一句描述答案的话，帮模型理清答案与其它信息的关系；然后把答案**替换成带类型提示的占位符**，避免答案直接出现在问题里；最后 LLM 从遮蔽句**生成问题并精炼**，若缺少作为推理起点的问题实体就触发补全。把复杂推理拆成简单步骤，既提升了标注质量，又大幅减少输出 token——在 Phi-4 上只用 CoT 21% 的 token 和 26% 的时间就拿到更高的标注质量。生成的 QA 对随即用式(3)和式(6)的监督损失去重训哈希与推理模块，闭环完成本地适应。

## 实验关键数据

### 主实验（WebQSP，训练成本与推理性能）

| 维度 | MobileKGQA | SOTA (GNN-RAG) | 对比 |
|---|---|---|---|
| 调参参数 | 0.1M | 0.8–1.7M | 仅 9% |
| 训练 PFLOPs | 0.6 | 2.4–25.1 | 仅 7.2%（GTE-large 设定）|
| 训练时间 | 1.63h | 1.85–2.22h | 最短 |
| Hit / F1（Gemma2 2B）| 79.8 / 67.0 | 80.0 / 66.9 | 几乎持平 |

跨各种 embedding 模型与 0.5B–14B 端侧 LLM，MobileKGQA 与 GNN-RAG 的差距平均仅 +0.46 Hit / −0.16 F1，最坏也只 −0.2 Hit；且算力**不随 embedding 模型增大而增长**（GTE-Qwen2-1.5B 下 GNN-RAG 算力可达其 41.8 倍）。相比 ICL 方法（ToG 延迟达 2400+s），延迟与性能全面占优。

### Jetson Orin Nano 边缘平台（vs GNN-RAG）

| 模式 | 指标 | MobileKGQA | GNN-RAG |
|---|---|---|---|
| 7W | 训练时间 / 能耗(Wh) | 2.1 / 11.8 | 5.9 / 28.7 |
| 7W | Hit / F1 | 74.2 / 62.9 | 61.7 / 54.3 |
| 15W | 是否限频 throttle | 否 | 是 |

只用 30.6% 训练时间、30.4% 能耗，性能反而 +20.3% Hit；且唯一能在设备上跑通**最优配置**、两种功耗模式都不触发限频的模型。

### 消融与分析

| 实验 | 结论 |
|---|---|
| 哈希维度（Table 5）| 64bit 时性能掉 <2.9%，256bit 几乎无损；存储省 99.75%，推理模块参数/PFLOPs 降 46.9%/82.2% |
| 标注质量（Table 6）| ROUGE-L 42.8 / BERTScore 48.7，全面超 RLM、CoT；token 仅 CoT 的 21% |
| 标注消融（Table 7）| 适应后 total Hit 64.9，优于 RLM(62.3)、CoT(61.4) |
| 分布漂移（Table 4）| 两次域适应后 total Hit 全面领先所有 baseline；CWQ 上适应新域后原域性能反升（知识正迁移）|

### 关键发现
- 哈希压缩在大幅省资源的同时几乎不掉点，验证「保余弦相似度即保语义」的理论假设站得住。
- 单纯缩小超参（如硬压 GNN-RAG）无法保住推理性能，说明哈希策略对端侧 KGQA 是必要而非可替代的。

## 亮点与洞察
- **把 embedding 哈希引入 KGQA 并给出互信息最优性证明**：不是工程化压缩，而是从「保余弦相似度⇒互信息最大」的理论条件出发设计损失，思路干净。
- **算力与 embedding 模型大小解耦**：哈希码固定维度，意味着可以白嫖未来更强的大 embedding 模型而端侧成本不变——这是相对 GNN-RAG「训练时反复重算 embedding」的结构性优势。
- **本地闭环对抗分布漂移**：自动标注生成让系统不必把私人数据传服务器即可持续学习，把隐私与适应性这对矛盾在端侧化解，是真正面向落地的设计。

## 局限与展望
- **依赖端侧 LLM 的标注质量**：标注生成仍要调本地 LLM，2B 级模型上质量虽优于 baseline，但绝对 ROUGE-L/BERTScore（42.8/48.7）离人工还有距离，错误标注可能累积偏差。
- **最短路径启发式的天花板**：用最短路径近似真实推理路径，在需要长程/多跳复杂推理时可能漏掉真正的逻辑链。
- **评测仍限于 WebQSP/CWQ 两个标准 benchmark**：分布漂移是人工切分 D1/D2/D3 模拟的，真实手机上长期累积的数据漂移更复杂，泛化性待进一步验证。
- 作者也在 ethics 中指出个性化系统可能引入偏见，留作未来工作。

## 相关工作与启发
- **KGQA 三大流派**：IR-based（NSM、EmbedKGQA、ReaRev）构子图做图推理；SP-based（QGG、DecAF、UnifiedSKG）把问题转成 SPARQL 等查询语言；LLM-based 分 ICL（KB-BINDER、ToG、StructGPT）与微调（RoG、ChatKBQA、GNN-RAG、SubgraphRAG）。本文站在 IR-based + LLM 路径检索的交叉点。
- **启发**：哈希 + 互信息保语义的思路可迁移到其他「端侧检索增强」场景（如端侧 RAG）；把复杂监督信号生成拆成可解释多步、以减少 token 的做法，对资源受限下的合成数据生成有普适价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 首个端侧可训练 KGQA，embedding 哈希+互信息证明、逐步标注生成两个组件都不是简单拼装，针对性强。
- **实验充分度**: ⭐⭐⭐⭐ — 真机 Jetson 测能耗/限频、两 benchmark、多 embedding/多 LLM、哈希维度与标注质量消融齐全，分布漂移设计扎实。
- **写作质量**: ⭐⭐⭐⭐ — 两个研究问题牵引全文，方法与动机对应清晰，图表完整；理论部分略压缩需查附录。
- **价值**: ⭐⭐⭐⭐ — 隐私+资源约束下的端侧个性化 KGQA 有明确落地价值，为端侧 RAG/KGQA 提供了可复用的工程与理论范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning](gdgb_a_benchmark_for_generative_dynamic_text-attributed_graph_learning.md)
- [\[ICLR 2026\] Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](adaptive_mixture_of_disentangled_experts_for_dynamic_graph_out-of-distribution_g.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[ICLR 2026\] One for Two: A Unified Framework for Imbalanced Graph Classification via Dynamic Balanced Prototype](one_for_two_a_unified_framework_for_imbalanced_graph_classification_via_dynamic_.md)
- [\[ACL 2026\] IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance](../../ACL2026/graph_learning/industryasseteqa_a_neurosymbolic_operational_intelligence_system_for_embodied_qu.md)

</div>

<!-- RELATED:END -->
