# AutoVER: Grounding Language Models for Visual Entity Recognition

**会议**: ECCV 2024  
**arXiv**: [2402.18695](https://arxiv.org/abs/2402.18695)  
**代码**: [https://github.com/MrZilinXiao/AutoVER](https://github.com/MrZilinXiao/AutoVER)  
**领域**: 多模态VLM  
**关键词**: 视觉实体识别, 检索增强生成, 约束解码, 对比学习, 多模态大语言模型

## 一句话总结
提出 AutoVER，在多模态大语言模型中统一集成对比检索和前缀树约束解码，将 600 万级 Wikipedia 实体空间先缩小到数百候选再做受限生成，在 Oven-Wiki 上将 entity seen 准确率从 PaLI-17B 的 30.6% 翻倍到 61.5%，同时在 unseen/query split 上也大幅领先。

## 研究背景与动机

1. **领域现状**：视觉实体识别（VER）要求模型给定图像和问题后，从 Wikipedia 600 多万个实体中精确定位答案。此前方案分两类——判别式（CLIP 微调后做检索排序）和生成式（PaLI/GiT 生成文本后用 BM25 匹配实体）。
2. **现有痛点**：
   - 判别式方法受限于浅层 dot-product 交互，难以处理需要推理的 query split；
   - 生成式方法存在严重的**幻觉问题**——生成文本无法保证属于 600 万实体空间中的真实实体；
   - 两类方法都不利用实体侧的视觉信息（entity image），无法区分名称相似但外观不同的实体。
3. **核心矛盾**：标签空间巨大（600 万+）且实体粒度极细（如 ATR 42 vs British Aerospace 146），分类器不可行；无约束的自回归生成又会产生不存在的实体名。
4. **本文要解决**：(a) 如何在 MLLM 中同时做检索和生成？(b) 如何保证生成结果始终落在合法实体空间中？(c) 如何处理训练中从未见过的 unseen 实体？
5. **切入角度**：将 VER 重新定义为"先检索缩小候选集、再用前缀树约束自回归解码"的 RAG 问题。关键观察是：如果能学到好的 query 表征从 600 万实体中检索出 top-k 候选，再用 trie 约束解码路径，就能同时解决幻觉和 unseen 问题。
6. **核心 idea 一句话**：在 MLLM 内部统一训练对比检索能力（不需外部 retriever），推理时用检索结果动态构建前缀树来约束解码，确保生成的实体名一定有效。

## 方法详解

### 整体框架
AutoVER 由两个模块组成：(1) 基于 LLaVA 架构的多模态大语言模型 $f_\phi$（Vicuna-7B/13B + CLIP-ViT-L/14 视觉编码器），负责处理 query 图像+问题并进行自回归生成；(2) 轻量级多模态实体编码器 $F_\varphi$（两层 Transformer），负责融合实体图像和描述文本产生实体表征。

**训练阶段**：query 侧通过特殊 `<ret>` token 的 last-layer hidden state 得到 query 表征 $Q$，entity 侧通过 $F_\varphi$ 的 fusion token 得到实体表征 $E$，然后同时优化 in-batch 对比损失和语言建模损失。

**推理阶段**：先用 $Q$ 在预缓存的实体向量库中做 top-k 相似度检索（k=300），再用检索到的实体标识符动态构建前缀树，最后在约束下自回归生成实体名称。

### 关键设计

1. **统一的检索-生成架构（无需外部 retriever）**
   - 做什么：在 MLLM 内部同时实现检索和生成，不依赖独立的检索模型
   - 核心思路：在 MLLM 词表中增加 `<ret>` token，其 last-layer hidden state 经过维度匹配投影和 L2 归一化后作为 query 表征。由于 causal attention mask 的存在，`<ret>` token 能够感知完整的图像+文本输入，但不会泄露 label 信息
   - 实体侧用 frozen CLIP 视觉编码器提取实体图像特征 $\mathbf{Z}_{im}$，frozen CLIP 文本编码器提取实体描述特征 $\mathbf{Z}_{text}$，再通过两层 Transformer 的 fusion token 做多模态融合
   - 设计动机：避免维护一个独立 retriever 的额外开销，且让检索能力和生成能力在同一模型中协同优化

2. **In-Batch 对比训练 + 拒绝采样**
   - 做什么：通过 InfoNCE 损失训练 query-to-entity 检索能力
   - 核心思路：对 mini-batch 中 $N$ 对 $(Q_i, E_i)$，正对为对角线，其余为负样本，最小化 $\mathcal{L}_{query2ent} = -\frac{1}{N}\sum_{i}\log\frac{\exp(\text{sim}(Q_i,E_i)/\tau)}{\sum_j\exp(\text{sim}(Q_i,E_j)/\tau)}$
   - 关键细节：VER 中多个 query 可能对应同一 entity，若 batch 内出现重复 entity 则对比学习语义错误。因此使用**拒绝采样**：遇到"冲突 batch"（同 entity 出现多次）时重新采样，保证负样本的纯净性
   - 注意只训练 query-to-entity 方向，不训练反向，因为目标是从 query 检索 entity

3. **硬负例挖掘（vision-hard + kb-hard）**
   - 做什么：构造更有区分度的训练 batch，迫使模型学会区分细粒度相似实体
   - **vision-hard**：用预训练 ViT 图像分类器对实体图像做分类，共享同一预测类别的实体被归为一组。采样时优先从同组中选取负样本
   - **kb-hard**：利用 Wikidata 的类别层级结构，共享父节点的实体被视为"知识相似"实体
   - 设计动机：普通随机采样产生的负样本太容易区分，无法让模型学到细粒度的实体差异

4. **检索增强约束解码（Retrieval-Augmented Constrained Decoding）**
   - 做什么：推理时确保生成的实体名一定存在于知识库中
   - 核心思路：(a) 用训练好的 $Q$ 表征在预缓存的实体向量库 $\mathcal{V}\in\mathbb{R}^{n\times d}$ 中做 top-k 检索得到 k=300 个候选实体；(b) 用这些候选的文本标识符动态构建 prefix tree (trie)；(c) 在自回归解码的每一步，trie 限制下一个 token 只能是有效路径上的 token，消除所有无效解码路径
   - 设计动机：解决了两个问题——① 将 600 万空间缩小到 300，让 MLLM 做小范围精确决策；② trie 约束彻底杜绝了幻觉，保证输出一定是合法实体

### 损失函数 / 训练策略
最终训练损失为语言建模损失与对比损失的线性组合：
$$\mathcal{L} = \mathcal{L}_{LM} + \lambda_r \cdot \mathcal{L}_{query2ent}$$
其中 $\lambda_r = 1$。语言建模损失 $\mathcal{L}_{LM}$ 只对目标序列（实体标识符）反向传播，不对输入序列（图像+问题）计算损失。训练使用 5M query-entity pairs，batch size 256，32 张 V100 GPU。为节省计算预算，消融实验使用 10% 数据，主实验使用 50% 数据。

## 实验关键数据

### 主实验（Oven-Wiki 验证集）

| 方法 | Entity Seen | Entity Unseen | Entity HM | Query Seen | Query Unseen | Query HM | Overall HM |
|------|------------|---------------|-----------|------------|--------------|----------|------------|
| CLIP-ViT-L/14 | 5.4 | 5.3 | 5.4 | 0.8 | 1.4 | 1.0 | 1.7 |
| CLIP Fusion | 32.7 | 4.3 | 7.7 | 33.4 | 2.2 | 4.2 | 5.4 |
| PaLI-3B | 21.6 | 6.6 | 10.1 | 33.2 | 14.7 | 20.4 | 13.5 |
| PaLI-17B | 30.6 | 12.4 | 17.6 | 44.2 | 22.4 | 29.8 | 22.1 |
| GPT-4V (zero-shot) | 29.8 | 19.3 | 23.4 | 56.5 | 52.7 | 54.5 | 32.9 |
| **AutoVER-7B** | **61.5** | 21.7 | 32.1 | 69.0 | 31.4 | 43.2 | 36.8 |
| **AutoVER-13B** | **63.6** | **24.5** | **35.6** | **69.0** | **32.3** | **43.9** | **39.2** |

Entity seen 从 PaLI-17B 的 30.6% 直接翻倍到 61.5%（7B）/ 63.6%（13B），在 query split 同样以超过 20 个百分点的优势领先。

### 消融实验（Oven-Wiki Entity Val，10% 训练数据）

| 配置 | Seen | Unseen | HM |
|------|------|--------|-----|
| AutoVER-7B-0.1 (完整) | 48.9 | 19.0 | 27.4 |
| w/o 检索增强 | 50.7 | 0.6 | 1.2 |
| w/o 约束解码 | 46.8 | 0.6 | 1.2 |
| w/ LoRA (r=128) | 43.5 | 2.8 | 5.3 |
| [CLS] 分类器变体 | 12.8 | 0.1 | 0.2 |

### 零样本泛化（A-OKVQA-Ent）

| 方法 | Multi-choice Seen | Multi-choice Unseen | MC Overall | Entity-match Overall |
|------|------------------|--------------------|-----------|--------------------|
| OpenFlamingo-9B | 5.9 | 9.0 | 6.9 | 36.8 |
| InstructBLIP-7B | 53.7 | 49.4 | 52.4 | 37.2 |
| LLaVA-v1-7B | 13.0 | 10.3 | 12.1 | 42.7 |
| **AutoVER-7B** | **67.7** | **52.5** | **62.8** | **55.0** |
| LLaVA-v1.5-7B (微调过) | 72.4 | 73.7 | 72.8 | 46.2 |

### 关键发现
- **检索增强是 unseen 性能的命脉**：去掉检索后 unseen 从 19.0% 暴跌到 0.6%，说明没有检索缩小候选集，模型完全无法处理训练中未见过的实体
- **约束解码有效抑制幻觉**：去掉约束解码后 seen 也下降 2.1%，说明即使是见过的实体，自由生成也容易"说错"
- **LoRA 不适合此任务**：全参数微调远优于 LoRA（r=128），LoRA 在 seen 上就掉了 5.4%，在 unseen 上基本不可用
- **分类器方案完全失败**：[CLS] 变体在 seen 只有 12.8%、unseen 接近 0，说明 600 万实体的分类器范式不可行
- **零样本泛化强于同等规模模型**：在 A-OKVQA-Ent 上，AutoVER 甚至在 entity-match 指标上超过了微调过 A-OKVQA 数据的 LLaVA-v1.5（55.0% vs 46.2%）

## 亮点与洞察
- **统一的检索-生成范式**：不需要独立 retriever，用一个特殊 token `<ret>` 就把检索能力嵌入了 MLLM 内部，训练时对比学习和语言建模共享参数、协同优化。这个设计非常优雅，可以迁移到任何需要"先检索再生成"的 MLLM 任务
- **Trie 约束解码彻底消灭幻觉**：从"600 万→300→唯一实体"的渐进缩小策略，用 prefix tree 在 token 级别做硬约束，比 post-hoc 的 BM25 匹配从根本上更可靠。这个 trick 可以直接用于任何答案空间有限的生成任务
- **拒绝采样解决对比学习的语义冲突**：VER 中多 query 对应同一 entity 的特殊性质会导致 batch 内"假负样本"，拒绝采样是一个简单有效的解法
- **硬负例挖掘的双源设计**：同时利用视觉相似性（ViT 分类）和知识图谱结构（Wikidata 层级），从两个互补维度提升细粒度区分能力

## 局限性 / 可改进方向
- **Unseen 上仍有较大差距**：Entity unseen 最高只有 24.5%，与人类+搜索引擎表现（human eval unseen 79.3%）差距巨大，说明检索能力对完全未见实体仍然不足
- **计算开销较大**：需要对 600 万实体预计算向量库，且训练需要 32 张 V100，对大多数研究组不友好
- **只使用了 50% 训练数据**：受计算预算限制未用全量数据，实际性能上限仍未探索
- **实体编码器较简单**：只有两层 Transformer 做融合，更强的实体编码方式（如 cross-attention 多层融合）可能进一步提升检索质量
- **检索数 k=300 固定**：不同难度的 query 可能需要不同数量的候选，自适应 k 值可能更优
- **LoRA 失败未深入分析**：为何 LoRA 在此任务上表现差？是否与对比学习需要大梯度更新有关？值得深入探索高效微调方案

## 相关工作与启发
- **vs PaLI (ICLR 2022)**：PaLI 是最强生成式基线，用 encoder-decoder 架构 + BM25 后匹配。AutoVER 用 decoder-only 架构 + trie 约束解码取代 BM25，从"先生成再匹配"变成"生成过程中就约束"，从根本上避免幻觉
- **vs GER (CVPR 2024)**：GER-400M 是同期工作，用 400M 参数的生成式方法在 entity test seen 达到 31.5%。AutoVER 用 7B 参数达到 62.8%，说明 MLLM 的推理能力对 VER 至关重要
- **vs CLIP-based 方法**：CLIP Fusion 在 entity seen 上有 32.7% 但 unseen 只有 4.3%。AutoVER 的检索增强设计专门解决了 unseen 泛化问题
- **vs GPT-4V**：GPT-4V 在 query split 上表现出色（54.5% HM），但在 entity split 上远不及 AutoVER。说明通用大模型的知识广度有余但实体识别的精确度不足
- **启发**：这种"在 MLLM 内部集成检索能力 + 外部知识约束解码"的范式可以推广到更多知识密集型多模态任务，如细粒度图像描述、视觉 fact checking 等

## 评分
- 新颖性: ⭐⭐⭐⭐ 在 MLLM 中统一检索和生成、trie 约束解码的组合是新颖的，但各组件（对比学习、prefix tree、RAG）都是已有技术
- 实验充分度: ⭐⭐⭐⭐ 主实验、消融、零样本泛化、case study 都有，消融揭示了各组件贡献；不足是受计算限制未用全量数据
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰、方法展示系统化、图表质量高，但部分数学符号较重
- 价值: ⭐⭐⭐⭐ 在 VER 这个具体任务上取得了显著突破，检索+约束解码范式对知识密集型多模态任务有参考价值
