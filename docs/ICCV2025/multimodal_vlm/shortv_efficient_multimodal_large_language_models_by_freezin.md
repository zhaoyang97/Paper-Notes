# ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers

**会议**: ICCV 2025  
**arXiv**: [2504.00502](https://arxiv.org/abs/2504.00502)  
**代码**: [https://github.com/icip-cas/ShortV](https://github.com/icip-cas/ShortV)  
**领域**: 多模态VLM / 高效推理 / 层级冗余  
**关键词**: layer redundancy, visual token freezing, training-free, Layer Contribution metric, MLLM efficiency  

## 一句话总结
发现MLLM中约60%的层对视觉token的变换几乎不影响模型输出（Layer Contribution极低），提出ShortV方法在这些"ineffective layers"中冻结视觉token（不参与attention query和FFN），在LLaVA-NeXT-13B上实现50% FLOPs降低且性能几乎不变，且与token剪枝方法（如FastV）正交可叠加。

## 研究背景与动机
1. **领域现状**：MLLM的计算开销主要来自两方面：LLM backbone的大规模参数和大量视觉token（LLaVA-NeXT每张图2880 token）。现有加速方法主要关注减少视觉token数量（如FastV剪枝50%的token）。
2. **现有痛点**：Token剪枝方法从"减少token数量"角度优化，但忽视了另一个维度——MLLM的很多层对视觉token的处理本身就是冗余的。文本LLM中已发现约25%的层是"ineffective"的，但MLLM中这种层级冗余尚未被系统研究。
3. **核心矛盾**：视觉token和文本token在MLLM中存在模态差距（modality gap），导致它们的层级冗余分布不同。直接套用文本LLM的层级冗余分析方法（perplexity或cosine similarity）对视觉token不适用。
4. **本文要解决什么**：(1) 提出能准确衡量MLLM各层对视觉/文本token贡献度的指标；(2) 基于此发现冻结哪些层的视觉token可以不影响性能；(3) 实现与token剪枝正交的新效率维度。
5. **切入角度**：提出Layer Contribution（LC）指标——逐层冻结特定token后测量输出logits的KL散度变化，发现MLLM的初始层和深层对视觉token贡献极小。
6. **核心idea一句话**：MLLM约60%的层对视觉token的变换是无效的，在这些层中冻结视觉token（跳过Q投影、FFN和attention查询）可以大幅降低计算量而不影响性能。

## 方法详解

### 整体框架
(1) 用少量样本（40个）计算每层对视觉token的LC分数；(2) 按LC升序排列，选择LC最低的N层标记为"ineffective"；(3) 将这些层替换为ShortV层——视觉token的hidden state保持不变（不作为attention query、不过FFN），仅文本token正常处理。

### 关键设计

1. **Layer Contribution (LC) 指标**:
   - 做什么：量化每层对特定类型token的贡献度。
   - 核心思路：在第$i$层冻结token类型$X$（保持hidden state不变），计算修改后模型的输出logits与原始模型的KL散度：$LC_i^X = KL(\text{logits}(M), \text{logits}(\mathcal{M}_i^X))$。LC越低说明该层对这类token越"无效"。
   - 关键发现：(1) 视觉token的LC普遍低于文本token——层对视觉token比对文本token更冗余；(2) 视觉token的冗余分布不同于文本——初始层和深层（包括最后一层）最冗余，而文本token在中到深层最冗余；(3) 最后一层对视觉token的LC恒为0，因为最后一层视觉token不参与next-token prediction。
   - 为何不用perplexity：即使完全不输入视觉token，MLLM的perplexity变化也很小（但视觉任务性能显著下降）。perplexity无法区分视觉信息对模型的真实影响。
   - 为何不用cosine similarity：cosine similarity不考虑层的位置——浅层的小变换会传播到所有后续层，但cosine similarity无法捕捉这种级联效应，导致高估浅层冗余、低估深层冗余。

2. **ShortV稀疏层设计**:
   - 做什么：在被标记为ineffective的层中，视觉token完全跳过计算。
   - 核心思路：ShortV层中，视觉token不参与attention（不作为Q），不经过$W_Q$和$W_O$投影，不经过FFN。文本token正常计算，且仍然可以attend到视觉token的KV（视觉token的KV仍然存在但不更新）。
   - 设计动机：既然这些层对视觉token的变换不影响输出，那么直接跳过这些计算就是安全的。但文本token仍需正常处理（它们的LC不为0）。
   - 默认配置：7B模型冻结19/32层（~60%），13B模型冻结24/40层（~60%）。

3. **与Token剪枝的正交性**:
   - ShortV减少每个视觉token在每层的计算量（层级维度）
   - FastV减少视觉token的数量（token维度）
   - 两者可以叠加：ShortV+FastV在LLaVA-1.5-7B上仅需29% FLOPs，同时性能几乎不降（MMBench 64.2 vs 基线64.1）。

### FLOPs分析
假设文本token数$t$，视觉token数$v$，hidden size $h$，FFN中间维度$m$。原始层FLOPs：$2(t+v)(4h+3m)h + 4(t+v)^2h$。ShortV层FLOPs：$2t(4h+3m)h + 4vh^2 + 4t(t+v)h$。主要节省来自视觉token不过FFN和不作为Q投影。

## 实验关键数据

### 主实验

| 模型 | 方法 | FLOPs比 | MME | MMBench | MMMU | SEED | GQA | Flickr30K |
|------|------|--------|-----|---------|------|------|-----|-----------|
| LLaVA-1.5-13B | Vanilla | 100% | 1531 | 68.9 | 35.4 | 68.2 | 63.3 | 79.6 |
| | FastV | 57% | 1507 | 68.3 | 34.6 | 67.8 | 59.4 | 73.4 |
| | VTW | 55% | 1533 | 68.5 | 34.9 | 68.2 | 60.6 | 65.9 |
| | **ShortV** | **55%** | **1536** | **68.6** | **35.8** | **68.0** | **62.0** | **76.4** |
| LLaVA-NeXT-13B | Vanilla | 100% | 1570 | 69.3 | 35.9 | 71.9 | 65.7 | 66.7 |
| | FastV | 51% | 1546 | 68.5 | 35.9 | 71.5 | 62.9 | 66.0 |
| | **ShortV** | **50%** | **1553** | **70.2** | **36.2** | **71.8** | **63.6** | **67.5** |

### ShortV layers数量消融（LLaVA-1.5-13B，40层）

| ShortV层数N | FLOPs比 | Avg | 性能保留 |
|------------|--------|-----|---------|
| 0 (基线) | 100% | 61.2 | 100% |
| 8 | 85% | 60.9 | 99.5% |
| 16 | 70% | 61.0 | 99.7% |
| 24 (默认) | 55% | 60.7 | 99.2% |
| 32 | 40% | 55.7 | 91.0% |

ShortV+FastV叠加（LLaVA-1.5-7B）:

| 方法 | FLOPs比 | MMBench | MMMU | SEED | GQA |
|------|--------|---------|------|------|-----|
| FastV | 58% | 64.3 | 35.8 | 65.4 | 60.2 |
| ShortV | 55% | 64.8 | 36.2 | 66.2 | 60.9 |
| **ShortV+FastV** | **29%** | **64.2** | **37.1** | **65.1** | **59.3** |

### 消融：层选择策略

| 策略 | FLOPs比 | MMBench | MMMU | SEED | GQA |
|------|--------|---------|------|------|-----|
| Random | 55% | 58.4 | 33.6 | 60.5 | 56.1 |
| Cosine Similarity | 55% | 60.8 | 34.2 | 62.7 | 59.5 |
| **LC (ours)** | **55%** | **64.8** | **36.2** | **66.2** | **60.9** |

### 消融：冻结哪些token

| 冻结类型 | MMBench | MMMU | SEED | GQA |
|---------|---------|------|------|-----|
| (a) Text | 2.1 | 23.7 | 8.9 | 2.9 |
| (b) Text+Visual | 1.3 | 26.6 | 0.8 | 0.0 |
| (c) Random | 1.5 | 22.9 | 5.5 | 2.3 |
| **(d) Visual (ours)** | **64.8** | **36.2** | **66.2** | **60.9** |

### 关键发现
- **60%的层可以冻结视觉token**（13B模型可冻结24/40层），性能保持99.2%。相比之下，文本LLM移除25%层的文本变换就会掉10%性能。这说明视觉token在LLM中的处理确实有大量冗余。
- **LC指标远优于cosine similarity**：LC选择的层冻结后MMBench 64.8 vs cosine similarity选择的60.8。cosine similarity高估浅层冗余导致误选。
- **只有冻结视觉token有效**：冻结文本token导致性能崩溃（MMBench 2.1），即使在对文本token最冗余的层中也不行。这验证了视觉token和文本token在MLLM中的处理方式确实不同。
- **ShortV+FastV可以叠加到29% FLOPs**，两种方法从不同维度（层级vs token数量）减少计算，正交互补。
- 被冻结的层主要集中在最后几层和前几层，中间层反而更重要（与文本LLM不同）。
- 实测加速：LLaVA-NeXT-13B默认配置1.52×加速，80%层冻结时1.84×。

## 亮点与洞察
- **发现了MLLM中视觉token的层级冗余远大于文本token**：这是一个重要的分析性发现。MLLM在大部分层中对视觉token做的变换是"无用功"，暗示了视觉信息在进入LLM后主要在少数关键层被吸收，其余层只是"路过"。
- **LC指标设计精巧**：通过直接测量输出logits的KL散度变化，避免了perplexity和cosine similarity的缺陷。特别是对浅层变换的级联效应的考虑，使得LC能准确反映每层的真实重要性。
- **与token剪枝正交意味着效率可以倍增**：29% FLOPs几乎不掉点，这是单独剪token或单独跳层都无法达到的。两个维度的组合为MLLM加速提供了新空间。

## 局限性 / 可改进方向
- 粒度较粗：整层冻结，未区分attention block和FFN。He et al.发现它们在文本LLM中冗余程度不同。
- 仅在LLaVA-1.5和LLaVA-NeXT（7B/13B）上验证，未测试更新更大的模型。
- LC需要少量校准数据（40个样本），虽然成本很低但不是完全零样本。
- 未分析在视频理解等长context场景中的效果。
- 作者在局限性中提到：可以用小网络在frozen层中做轻量更新而非完全跳过。

## 相关工作与启发
- **vs FastV**: FastV在第2层后剪掉50% token（token维度），ShortV在60%的层中冻结全部visual token的计算（层维度）。ShortV在GQA上保持更好（60.9 vs 60.2），且两者可以叠加到29% FLOPs。
- **vs VTW (Visual Token Withdrawal)**: VTW在某层之后完全丢弃视觉token，更激进但在Flickr30K上崩溃（65.9→44.5 for 7B）。ShortV不丢弃token只是冻结更新，更稳定。
- **vs SparseVILA（同批论文）**: SparseVILA从prefill-decode阶段解耦稀疏，ShortV从层级角度冻结视觉token。两者切入点不同但互补：SparseVILA选择哪些token在decode时激活，ShortV决定哪些层需要处理视觉token。理论上可以结合。
- **vs ShortGPT**: ShortGPT对文本LLM移除约25%冗余层，但直接应用于MLLM不适用（因为视觉和文本的冗余分布不同）。ShortV针对多模态特性做了token类型级别的冻结。

## 评分
- 新颖性: ⭐⭐⭐⭐ LC指标和层级视觉token冗余的发现很有洞察力，方法简洁有效
- 实验充分度: ⭐⭐⭐⭐ 4个模型变体，7个benchmark，丰富消融（层数、选择策略、冻结类型），实测加速
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，LC指标的讨论（vs perplexity vs cosine sim）非常透彻
- 价值: ⭐⭐⭐⭐ 揭示了MLLM中视觉token层级冗余的本质，与token剪枝正交的新加速维度
