---
title: >-
  [论文解读] OSCAR: Online Soft Compression for RAG
description: >-
  [ICLR 2026][信息检索/RAG][RAG] OSCAR 让一个轻量压缩器在线地、依据当前 query 把每篇检索文档压成几个 embedding token，再交给生成器作答，从而在 1B–24B 的 LLM 上实现 2–5× 端到端推理加速、且几乎不掉点。 领域现状：RAG 通过把检索到的文档拼进 prompt…
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "RAG"
  - "软压缩"
  - "查询相关压缩"
  - "序列级蒸馏"
  - "重排序"
---

# OSCAR: Online Soft Compression for RAG

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ideKAUWvFE](https://openreview.net/forum?id=ideKAUWvFE)  
**代码**: https://github.com/naver/pisco （模型权重 naver/oscar）  
**领域**: 信息检索 / RAG 上下文压缩  
**关键词**: RAG、软压缩、查询相关压缩、序列级蒸馏、重排序

## 一句话总结
OSCAR 让一个轻量压缩器在线地、依据当前 query 把每篇检索文档压成几个 embedding token，再交给生成器作答，从而在 1B–24B 的 LLM 上实现 2–5× 端到端推理加速、且几乎不掉点。

## 研究背景与动机
**领域现状**：RAG 通过把检索到的文档拼进 prompt 来增强 LLM，但上下文一长，推理成本急剧上升。为压缩检索内容，主流有两条路线：**硬压缩**（hard compression）在文本层面做摘要或剪枝（如 Provence、RECOMP），**软压缩**（soft compression）把文本映射成连续向量 / KV cache（如 COCOM、PISCO、xRAG）。

**现有痛点**：硬压缩是在线、query-aware 的，能边检索边压、还能顺手重排，但因为产物仍是文本，压缩率很有限（≈×2），省不了多少。软压缩能做到很高压缩率（≈×16），但代价是明显掉点，而且——关键问题——**现有软压缩全是离线的**：它们用一个和生成器同样大的 LLM 做压缩前向，只能预先把整个语料库压好存起来，没法在线对开放网页 / 大规模动态语料即时压缩。

**核心矛盾**：在线 + query-aware + 高压缩率三者难以兼得。软压缩之所以离线，是因为它依赖「压缩器和生成器同构、同尺寸」来对齐隐空间；可压缩器一旦和生成器一样大，在线压缩的开销就把生成阶段省下的时间全吃掉了，整体反而不快。而且现有软压缩**压缩时根本不看 query**，硬要把整篇文档的所有信息塞进固定向量。

**本文目标**：造一个足够快、又 query 相关的在线软压缩器，既保留软压缩的高压缩率，又拿回硬压缩的在线 + query-aware 优势。

**切入角度**：作者观察到两点——其一，压缩时引入 query，可以只保留对当前问题有用的信息，让固定数量的 embedding 装下更精炼的内容；其二，压缩器**不必和生成器同构同尺寸**，只要能把隐状态对齐到生成器的输入空间即可，于是可以用一个很小的 backbone（小模型或生成器的前几层）来把压缩做快。

**核心 idea**：用一个轻量压缩器，把「(query, 文档)」对在线映射成 l 个 memory embedding，再用序列级蒸馏让整条压缩管线逼近无压缩 RAG 的输出——即「query 相关的在线软压缩」。

## 方法详解

### 整体框架
OSCAR 的推理管线很直接：检索 → 压缩 → 生成。检索拿到 top-k 文档后，一个**压缩器 LLM** 把每个「query + 文档」对映射成少量 embedding token（默认每篇 128-token 文档压成 8 个，即 ×16 压缩率）；这些压缩向量连同原始 query 一起塞进 RAG prompt，交给**生成器 LLM** 直接作答。由于每篇文档从上百 token 缩成几个 embedding，生成阶段的注意力计算大幅变快。

训练侧用**序列级蒸馏**：先用标准无压缩 RAG 管线（教师 = Mistral-7B）跑一遍，把它的回答存成监督标签，再让 OSCAR 的「压缩器 + 生成器」端到端拟合这些教师回答，梯度同时回传到两个模型。整条管线不需要任何人工标注答案。此外，压缩器还可顺带学一个 `[RR]` 重排序 token，把压缩和重排合并进一次前向，让压缩在已含重排的 RAG 管线里近乎「免费」。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Query + top-k 检索文档"] --> B["轻量压缩器骨干<br/>小模型 或 生成器前 N 层"]
    B --> C["查询相关在线压缩<br/>(q, d) → l 个 [MEM] embedding"]
    C -->|同一前向多出 [RR] token| D["重排序<br/>预测相关性分数 r_i"]
    C --> E["生成器 LLM<br/>读 query + 压缩向量作答"]
    E -.序列级蒸馏对齐.-> F["教师 Mistral-7B<br/>无压缩 RAG 回答"]
```

### 关键设计

**1. 查询相关的 memory-token 压缩：让固定数量向量只装「对这个问题有用」的信息**

这是 OSCAR 区别于所有先前软压缩的核心。以往方法（COCOM、PISCO、xRAG）压缩文档时**不看 query**，强行把整篇文档塞进固定向量；OSCAR 把 query $q$、第 $i$ 篇文档 $d_i$ 和一组可学习的 memory token $[\text{MEM}_j]_{j=1\ldots l}$ 一起喂给压缩器 $C$，取最后一层在这些 memory token 位置上的隐状态作为该文档的压缩表示 $c_i = (c_i^1,\ldots,c_i^l) = C(q, d_i)$。这些 `[MEM]` token 扮演类似 BERT `[CLS]` 的角色——一个提示模型「把信息存到我这」的任务专属 token。因为压缩时已知 query，固定的 $l$ 个向量可以只编码与问题相关的内容，从而在同样压缩率下保住准确率。消融显示，去掉 query 依赖（query-independent）在 ×16 时掉 4 个点、×128 时掉 6 个点，证明 query 确实被用来优化了压缩表示；needle-in-a-haystack 分析也显示压缩 embedding 与「针」附近文本的余弦相似度最高，印证了强 query 依赖。

**2. 轻量压缩器骨干：把软压缩从离线拉回在线**

软压缩之所以一直是离线的，根因在于「压缩器和生成器同构同尺寸」——这让隐空间对齐变容易，但在线跑这么大的压缩前向就抵消了生成阶段的加速。OSCAR 提出两种小骨干来打破这一点。其一是 **OSCAR-N-Layers**：直接取预训练 backbone 的前 $N$ 层（去掉 head）当压缩器，与生成器同源，因此**无需额外预训练**就能对齐隐空间，效率由 $N$ 控制，通常取总层数的 1/4–1/3（实验用 $N=5,8,10$）。其二是 **OSCAR-llama**：用一个更小的独立 LLM（默认 Llama-3.2-1B）当压缩器，在它最后一层隐空间上接两层带 ReLU 的全连接层，把表示映射到生成器的 embedding 空间；由于压缩器与生成器异构，这个映射需要先在自编码 / 文本续写任务上预训练，再做 QA 微调。两者都把压缩开销压到远低于生成开销，使整体在线加速成为可能；实验里 OSCAR-llama 通常最快也最强。

**3. 序列级蒸馏训练目标：不要标注，直接逼近无压缩 RAG**

OSCAR 希望压缩后的管线尽量复现无压缩版本的行为，于是不依赖任何 ground-truth 答案，而是用教师 LLM（Mistral-7B）在标准无压缩 RAG 下生成的回答 $a_1,\ldots,a_r$ 作监督。训练目标对压缩器 $C$ 和生成器 $G$ 联合优化：

$$\mathcal{L}(C,G) = -\sum_{i=1}^{r} \log G(a_i \mid q, c_1,\ldots,c_k, a_{<i}),\quad c_i = C(q,d_i)$$

其中 $k$ 是用于生成的文档数。梯度同时回传到生成器和压缩器。实践上检索结果和教师回答只需在训练集上预存一次，之后 OSCAR 训练就退化成一次普通的有监督微调，1B–24B backbone 只需 1–5 GPU-day。消融还发现：生成器必须和压缩器**联合训练**，冻结生成器会掉 6 个点——说明生成器也需要适应这种 embedding 形式的输入。

**4. 同一前向顺带重排序：让压缩在 RAG 管线里近乎免费**

作者借鉴 Provence 的洞察：query 相关的在线压缩，本质上和「cross-encoder 重排序」做的事很像——都是把文档相对 query 做上下文化编码。既然强 RAG 管线本就含重排步骤，那不如用**一次前向同时产出压缩表示和重排分数**，这样压缩成本就被重排步骤吸收掉、近乎免费。具体做法是在压缩器 prompt 里加一个 `[RR]` token，再接一层全连接把它的隐状态映射成相关性分数 $r_i$，并在式 (1) 上加一项点式蒸馏损失 $\lambda \sum_{i=1}^{k}(r_i - r_i')^2$，让 $r_i$ 拟合参考 reranker 的分数 $r_i'$（默认 $\lambda=0.05$）。实验显示这种简单点式蒸馏就足够有效：joint 训练后 standalone 与 e2e（用 OSCAR 自己的重排结果再压缩）几乎不掉点，在 BEIR 上的重排能力也接近教师。

### 损失函数 / 训练策略
- 主损失：式 (1) 的序列级蒸馏负对数似然，教师统一用 Mistral-7B，梯度同时回传压缩器与生成器。
- 重排分支（可选）：附加点式蒸馏 MSE，权重 $\lambda=0.05$。
- 训练配置：生成器用 LoRA 适配器，压缩器用全量微调（实测优于 LoRA）；训练时 $k=5$ 篇文档，推理时可泛化到 10–50 篇。OSCAR-llama 需在自编码 + 文本续写上预训练。

## 实验关键数据

### 主实验
在 Mistral-7B 上对比各压缩方法（Average 为 6 个 QA 数据集准确率均值，Total 为压缩+生成总 T-FLOPs，括号为相对无压缩的加速）：

| 方法 | Average 准确率 | Total T-FLOPs | 端到端加速 |
|------|------|------|------|
| No compression | 0.68 | 20.33 | 1× |
| RECOMP（硬） | 0.67 | 8.13 | 2.5× |
| Provence（硬） | 0.68 | 9.43 | 2.2× |
| PISCO（离线软） | 0.65 | 3.49 | 5.8×（离线） |
| OSCAR-llama | 0.68 | 6.15 | 3.3× |
| OSCAR-8-Layers | 0.68 | 8.36 | 2.4× |

跨 backbone 看，OSCAR 越大模型越划算：Mistral-24B 上 OSCAR-llama 把总 FLOPs 从 64.29 降到 13.37（**4.8×**）且 Average 反升到 0.69；Llama-1B 上 OSCAR-5-Layers 也有 2.1× 加速并把 Average 从 0.55 提到 0.60。GPT-4o 成对比较显示 OSCAR-llama 在更快的同时与无压缩、Provence、PISCO 基本持平、平均略优于 Provence。

### 消融实验
均基于 Mistral-7B，报告 5 个 QA 任务的点式 LLM-eval 均值（∆ 相对 OSCAR-llama ×16 的 0.77）：

| 配置 | Avg | ∆ | 说明 |
|------|------|------|------|
| OSCAR-llama ×16 | 0.77 | — | 完整模型 |
| query-independent ×16 | 0.73 | -0.04 | 压缩不看 query |
| 无压缩器预训练 ×16 | 0.70 | -0.07 | OSCAR-llama 去掉预训练 |
| 冻结生成器 ×16 | 0.71 | -0.06 | 生成器不参与训练 |
| OSCAR-llama ×128 | 0.75 | -0.02 | 压缩率拉到 128 |
| query-independent ×128 | 0.71 | -0.06 | 高压缩率下 query 更关键 |
| Modern-bert-large 压缩器 ×16 | 0.76 | -0.01 | 低延迟的替代压缩器 |

### 关键发现
- **query 依赖是核心**：去掉后 ×16 掉 4 点、×128 掉 6 点，压缩率越高 query 越关键。
- **联合训练 + 预训练缺一不可**：冻结生成器掉 6 点；OSCAR-llama 去掉压缩器预训练掉 7 点（异构压缩器必须预训练对齐隐空间）。
- **压缩率可拉到 ×128 仅掉 2 点**，说明软压缩潜力大；压缩器选型上 Llama-1B 最好，Modern-bert-large 是低延迟场景的好替代。
- **鲁棒性**：换成弱检索（BM25 无重排）后，OSCAR 相对 Mistral-7B 的掉点幅度与无压缩一致，说明它能处理噪声文档；文档数增到 50（≈7k token 上下文）时因高压缩率反而比无压缩快 5×。

## 亮点与洞察
- **「压缩时看 query」这一步看似简单却是范式转变**：它把软压缩从「离线把整库压死」变成「在线按需压缩」，从而能接开放网页 / 动态语料的即插即用 RAG。
- **压缩器不必和生成器同构**这个松绑很关键：用前 N 层（免预训练）或小 LLM（需预训练对齐）两条路，把在线软压缩的延迟真正打下来——这是先前软压缩一直绕不开的死结。
- **压缩与重排合并进一次前向**是个漂亮的工程洞察：既然强 RAG 本就要重排，那压缩就被「藏」进重排成本里、近乎免费，可直接迁移到任何含 reranker 的检索管线。
- 序列级蒸馏「无需标注、直接拟合教师无压缩输出」的训练范式，让方法对新 backbone 的迁移成本极低（1–5 GPU-day）。

## 局限与展望
- **backbone 专属**：OSCAR 需为每个生成 LLM 单独重训，不像硬压缩那样 LLM-agnostic，部署到新模型有训练成本（虽然不大）。
- **OSCAR-llama 需要额外预训练**：异构压缩器要先做自编码 / 续写预训练才能对齐，流程比 N-Layers 复杂。
- 评测集中在 QA 类任务（NQ/TriviaQA/HotpotQA/ASQA/PopQA/BIOASQ），对长生成、多跳复杂推理等场景的压缩表现仍有待验证。
- 作者指出与 KV-cache 压缩方法正交，二者结合是潜在方向，但本文未做。

## 相关工作与启发
- **vs Provence / RECOMP（硬压缩）**：它们在文本层面剪枝、LLM-agnostic 且在线，但压缩率受限于文本（≈×2）；OSCAR 在 embedding 层压缩，压缩率更高、推理更快，准确率持平甚至略优，代价是需为每个 backbone 重训。
- **vs PISCO / COCOM（离线软压缩）**：它们压缩时不看 query、且压缩器与生成器同尺寸只能离线预压；OSCAR 引入 query 依赖 + 轻量压缩器，把软压缩做成在线，并在同等压缩率下少掉点。
- **vs FiD-light / DODO**：FiD-light 用 encoder-decoder 做了一种 query 相关压缩但压缩率很低；DODO 主要面向长上下文，用于多文档 RAG 压缩时效果弱。OSCAR 在「高压缩率 + 在线 + query 相关」三点上同时达成。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个在线软压缩 RAG 方法，「压缩看 query + 轻量异构压缩器」打通了长期被认为难以兼得的三角。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 1B–24B 四种 backbone、6 个 QA 集，含 FLOPs、LLM-eval、GPT-4o 成对比较与多维消融。
- 写作质量: ⭐⭐⭐⭐ 动机与方法清晰，图表信息密度高，部分公式与附录交叉引用较多。
- 价值: ⭐⭐⭐⭐⭐ 2–5× 加速几乎不掉点，且重排免费、可接动态语料，对生产级 RAG 很实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Less Is More: Elevating RAG via Performance-Driven Context Compression](../../ICML2026/information_retrieval/less_is_more_elevating_rag_via_performance-driven_context_compression.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](../../ACL2026/information_retrieval/codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](../../ACL2025/information_retrieval/exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)

</div>

<!-- RELATED:END -->
