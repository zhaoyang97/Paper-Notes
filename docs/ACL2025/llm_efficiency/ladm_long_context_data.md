# LADM: Long-context Training Data Selection with Attention-based Dependency Measurement

**会议**: ACL 2025  
**arXiv**: [2503.02502](https://arxiv.org/abs/2503.02502)  
**代码**: [https://github.com/ZNLP/LADM](https://github.com/ZNLP/LADM)  
**领域**: LLM 效率 / Long-context Modeling  
**关键词**: long-context, data selection, attention mechanism, dependency measurement, continual pre-training  

## 一句话总结

提出 LADM 框架，利用注意力机制的内在检索能力来度量长上下文数据中的跨 span 依赖关系，从大规模预训练语料中高效筛选高质量长上下文训练数据，仅用 1B token 持续预训练即可显著提升多种 LLM 的长上下文能力。

## 研究背景与动机

长上下文建模是 LLM 的重要能力（GPT-4 支持 128K，Gemini 1.5 支持 1M token），持续预训练是赋予 LLM 长上下文处理能力的主流方法。然而，训练数据质量的度量仍是开放挑战：若训练数据由短样本拼接而成、缺乏跨上下文的依赖关系，模型可能无法学会处理长距离、多样化的上下文依赖，甚至加剧 LLM 忽略远距离信息的倾向。现有方法如 ProLong 通过 delta perplexity 来度量片段间依赖，但忽略了长上下文内部的固有结构和关系，导致度量不够准确。作者希望利用注意力分布来更全面地捕获上下文内的依赖关系。

## 方法详解

### 整体框架

LADM 分为三个层次的度量：
1. **Pairwise Focus Score (PFS)**：度量两个 span 之间的依赖强度
2. **Aggregated Focus Score (AFS)**：聚合单个 span 与所有前序 span 的依赖
3. **Contextual Dependency Score (CDS)**：汇总所有 span 的 AFS 得到样本级质量分数

根据 CDS 排序后，选取高分样本进行持续预训练。

### 关键设计

- **Long Attention Calculator**：使用 TinyLlama-1.1B 在随机采样的 32K 序列上训练，使其具备基础的长上下文建模能力。该小模型作为注意力计算器，用于后续的数据筛选，兼顾效率和准确性。
- **PFS 计算**：将长上下文样本分为 N=256 个长度为 l=128 的 span，计算 span j 对 span i 的累积注意力权重，量化 span i 对 span j 最终表示的影响。公式为 PFS(i,j) = Sum(Softmax(Q_j K_{0:j}^T / √d_k)[:, i])。
- **AFS 设计考量**：排除首部 m=1 个 span 和局部 n=4 个 span（避免 attention sink 和局部依赖的干扰），对距离加权（鼓励长距离依赖），引入方差项σ_j（高方差表示依赖模式更多样、结构更丰富）。以步幅 d=4 采样以提高效率。
- **CDS 聚合**：按 span 位置加权求和所有 AFS，位置越靠后的 span 权重越高（因为后部 span 有更多前序 span 可依赖），排除初始 n₀=16 个 span 的 AFS。最终 CDS 越高表示样本的长距离依赖越丰富。

### 损失函数 / 训练策略

训练配置如下：

- Long Attention Calculator 用 5B token 训练（也提供了 1B token 版本的对比实验）
- 持续预训练使用 32K 序列，RoPE base 从 10000 调整到 500000
- 数据选择保持原始领域分布，所有方法统一使用 1B token
- 在 OpenLlama-3B、Llama2-7B/13B、Mistral-7B 上验证

数据选择时按 CDS 排序，从每个领域中选取 top-ranking 样本，确保领域分布不变。

## 实验关键数据

### 主实验

**初步实验（Needle-in-the-Haystack）**：用不同长度拼接的 32K 数据训练 Llama2-7B，平均检索准确率随原始长度递减：32K 原生 0.88 → 16K×2 拼接 0.85 → 8K×4 拼接 0.69 → 4K×8 拼接 0.57，证明上下文依赖对长上下文能力至关重要。

**Perplexity 评估（Proof-Pile）**：LADM 在所有模型和上下文窗口下均优于 Random 和 ProLong。例如 Llama2-7B 32K 窗口：Random 2.458, ProLong 2.470, LADM **2.453**。Mistral-7B 32K：Random 2.455, ProLong 2.346, LADM **2.332**。

**LongBench 实测（核心结果）**：LADM 在四个模型上平均比 ProLong 提升 **2.16%**。具体亮点：
- Mistral-7B Single-doc QA：ProLong 23.76 → LADM **33.85**（+10.09%）
- Mistral-7B Multi-doc QA：ProLong 24.43 → LADM **29.09**（+4.66%）
- Llama2-7B Single-doc QA：ProLong 29.50 → LADM **32.24**（+2.74%）
- Llama2-7B Multi-doc QA：ProLong 29.04 → LADM **31.10**（+2.06%）
- Llama2-13B Multi-doc QA：ProLong 33.20 → LADM **35.53**（+2.33%）
- OpenLlama-3B Single-doc QA：ProLong 25.21 → LADM **26.63**（+1.42%）

**Needle-in-the-Haystack 合成任务**：LADM 在 Llama2-7B/13B 和 Mistral-7B 上以仅 1B token 训练即接近 100% 检索率，而其他方法在中间位置和长距离检索时表现较差。

### 关键发现

1. **数据效率极高**：仅用 random sampling 一半的训练 token 就能达到更好的效果。
2. **跨架构泛化**：在 3B/7B/13B 不同规模、OpenLlama/Llama2/Mistral 不同架构上均有效。
3. **不损害短上下文性能**：附录实验证明 LADM 在提升长上下文能力的同时保持了短上下文表现。
4. **注意力机制可区分依赖质量**：Long Attention Calculator 的注意力分数能有效区分原生长文本和拼接文本。
5. **图 2 直观验证**：原生 32K 样本在远距离 span 上的中位注意力分数显著高于拼接版本，证实注意力分布可以作为依赖质量的有效指标。

## 亮点与洞察

- **核心思想简洁有力**：利用注意力机制本身的检索特性来度量数据质量，方法论上将"数据质量"与"模型注意力分布"巧妙关联。
- **计算效率好**：使用 1.1B 小模型做数据筛选，比基于大模型 perplexity 的方法更高效。步幅采样进一步减少计算量。
- **AFS 中引入方差项**的设计巧妙——不仅关注依赖强度，还关注依赖多样性，这对识别结构丰富的长文本很关键。
- **与 ProLong 的对比有说服力**：ProLong 将样本拆段后独立计算 delta PPL，丢失了完整上下文信息；LADM 在完整上下文中计算注意力分布，保留了全局依赖信息。
- **初步实验设计巧妙**：通过不同长度拼接的对比实验，清晰展示了上下文依赖性对长上下文建模的关键影响，为后续方法提供了直观动机。

## 局限性

- 仅在 32K 长度上实验，未验证更长序列（64K/128K）的效果。
- Long Attention Calculator 需要额外的训练成本（5B token），虽然 1B 版本也可用但效果略差。
- 预训练数据仅用 Pile corpus，未验证在其他数据源上的适用性。
- CDS 中多个超参数（m, n, d, n₀, l）需要调整，未充分讨论敏感性。
- 方法假设注意力分布能准确反映数据依赖质量，但不同层、不同 head 的注意力模式差异较大，论文未详细讨论使用哪些层/head。

## 相关工作

- **长上下文建模**：PI (Chen et al., 2023)、NTK、YaRN (Peng et al., 2024)、LongLoRA 等位置编码扩展方法。
- **训练免方法**：StreamingLLM (Xiao et al., 2024)、SelfExtend (Jin et al., 2024)。
- **长上下文数据**：ProLong (Chen et al., 2024a) 用 delta PPL 度量依赖强度；相似性分组（Staniszewski et al., 2023）；信息密集训练（An et al., 2024b）。
- **注意力机制的检索能力**：Mittal et al. (2022)、Wu et al. (2024) 揭示注意力机制内在的检索操作。

## 评分

- **新颖性**: 4/5 — 用注意力分布度量长上下文数据质量是新颖的视角
- **技术深度**: 4/5 — PFS→AFS→CDS 三级度量体系设计完整，考虑周全
- **实验充分性**: 4/5 — 4 个模型、3 类任务、多个 baseline，实验较为充分
- **实用价值**: 4/5 — 对长上下文预训练的数据筛选有直接指导意义
- **推荐指数**: ⭐⭐⭐⭐
