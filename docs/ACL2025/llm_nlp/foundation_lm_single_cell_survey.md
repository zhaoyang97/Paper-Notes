# A Survey on Foundation Language Models for Single-cell Biology

**会议**: ACL 2025 (Long Paper)  
**arXiv**: 无  
**代码**: 无  
**领域**: 计算生物学 / NLP交叉  
**关键词**: single-cell biology, foundation language model, pre-trained language model, large language model, gene expression  

## 一句话总结
首篇从语言建模视角系统综述单细胞生物学基础语言模型的工作，将现有模型划分为 PLM（从头预训练）和 LLM（利用已有大模型）两大类，全面分析了数据 tokenization 策略、预训练/微调范式以及下游任务。

## 背景与动机
语言模型（BERT、GPT 等）的成功已经渗透到计算生物学领域。研究者发现可以将细胞（cell）类比为"句子"，将基因（gene）类比为"词/token"，从而利用语言模型来构建统一的单细胞基础模型。这类模型能够获得跨数据集、跨任务的通用细胞表示，在细胞类型注释、基因扰动预测、药物响应等下游任务上超越传统专用模型。然而，此前的综述大多从 Transformer 角度出发，缺乏从语言建模视角的系统分析。本文填补了这一空白。

## 核心问题
如何系统地理解和分类当前为单细胞生物学构建的基础语言模型？这些模型在数据表示（tokenization）、预训练策略、以及下游任务适配方面各自如何设计？当前面临的核心挑战是什么，未来方向在哪里？

## 方法详解

### 整体框架
论文将单细胞基础语言模型分为两大阵营：

1. **Single-cell PLMs**（预训练语言模型）：将基因视为 token、细胞视为句子，从头在大规模单细胞数据上预训练（如 scBERT、scGPT、GeneFormer、scFoundation 等）。流程为：数据收集 → tokenization → 预训练 → 下游任务微调/零样本推理。

2. **Single-cell LLMs**（大语言模型）：不从头预训练，而是利用已有的通用 LLM（如 GPT-2/3.5/4、LLaMA、T5），通过将细胞数据转换为文本格式后进行微调或直接推理。流程为：数据收集 → 细胞转文本 → 微调/零样本 → 下游任务。

### 关键设计

#### 1. Tokenization 策略（PLM 端）
将细胞的基因表达矩阵 (N×G) 转化为语言模型可理解的格式，主要有三种方式：
- **离散 token**：binning（scBERT, CellLM）将连续基因表达值离散化为整数；rank value encoding（GeneFormer 家族）按基因表达排序后用基因词表编码
- **连续嵌入**：利用蛋白质语言模型获取基因嵌入（UCE, scPRINT）；可学习层映射（CellPLM）；分层贝叶斯下采样（scFoundation）等
- **辅助信息**：融入元数据（细胞状态、器官来源、供体信息、测序技术等）或利用蛋白质基础模型的先验知识

#### 2. 预训练范式（PLM 端）
- **掩码语言建模（MLM）**：最主流，随机掩码 15%-30% 基因后重建（scBERT, UCE, GeneFormer, CellPLM, scFoundation, Nicheformer）
- **下一个 token 预测（NTP）**：自回归预训练，仅 tGPT 和 scGPT 采用。论文指出 NTP 在单细胞领域不流行，因为（1）数据规模相比文本仍不够大，（2）细胞数据稀疏导致大量 ground truth 为零，模型倾向学到平凡解
- **多任务预训练**：在 MLM 基础上叠加对比学习、分类、细胞生成、元数据预测、去噪等监督任务（CellLM, LangCell, scCello, scPRINT, scMulan, GeneCompass, CellFM）

#### 3. 细胞-文本转换与微调范式（LLM 端）
转换方式：
- **Cell-to-Sentence**：按表达量排序选 top-100 基因名拼成文本句子（Cell2Sentence, CHATCELL, CELLama）
- **Text-level Gene Embeddings**：用 LLM 获取每个基因的功能描述嵌入，再用表达值加权组合（GenePT, scELMo, scInterpreter）

微调范式：
- **指令微调**：将任务转为 QA 格式（Cell2Sentence, CHATCELL），局限在于很多任务难以自然地转化为 QA
- **嵌入微调**：直接利用细胞/基因嵌入进行监督微调，是目前主流
- **免调优**：LLM 作为 agent 直接生成 Python 代码执行分析（scChat）

### 下游任务体系
- **细胞级**：细胞类型注释（最基础）、新细胞类型发现、批次效应校正、细胞聚类、多组学整合、细胞生成
- **基因级**：基因网络分析、基因扰动预测、基因功能/表达预测
- **药物相关**：药物敏感性预测、药物响应预测
- **空间相关**：空间转录组补全、空间标签预测、空间组成分析

## 实验关键数据
本文为综述论文，不包含原创实验。但提供了详细的模型对比表：

| 模型 | Tokenization | 预训练范式 | 预训练数据规模 |
|------|-------------|-----------|--------------|
| scBERT | Binning | MLM | 1M cells |
| GeneFormer | Rank Value Encoding | MLM | 27.4M cells |
| scFoundation | Downsampling | MLM | 50M cells |
| scGPT | Binning+Metadata | NTP | 33M cells |
| GeneCompass | Ranking+Metadata | Multi-task | 126M cells |
| CellFM | Padding+MLP | Multi-task | 100M cells |
| Nicheformer | Ranking+Metadata | MLM | 57M cells |

LLM 端模型基于 GPT-2/3.5、T5、LLaMA-13B、all-MiniLM-L12-v2 等。

### 消融实验要点
- 综述无消融实验，但总结了各模型的核心区别：MLM 优于 NTP 在单细胞领域（数据量和稀疏性限制）；多任务预训练整合了自监督和监督信号效果更好；仅 scGPT 和 scELMo 验证了多组学整合能力。

## 亮点
- **清晰的分类体系**：PLM vs LLM 两大类，配合精细的子分类（tokenization 三策略、预训练三范式、LLM 三种微调模式），使读者快速建立全景视图
- **语言建模新视角**：首次完全从 NLP 的语言建模角度审视单细胞基础模型，而非传统的生物信息学视角
- **Cell = Sentence 的类比**：将基因视为 token、细胞视为句子的统一框架简洁优雅，是跨领域迁移的典范
- **系统的挑战分析**：从数据质量（稀疏性、位置信息、批次效应、多组学缺乏）、模型设计（统一 tokenizer、scaling law 未现）、评测协议（缺乏统一基准、可用性差）三方面指出问题

## 局限性 / 可改进方向
- **技术深度有限**：综述主要停留在"是什么"层面，对各模型的具体实验效果缺乏横向量化对比（因为现有模型大多在私有数据集上评测，难以公平对比）
- **生物动机分析不足**：作者自己承认，论文侧重技术分析，对设计背后的生物学意义讨论不够深入
- **时效性**：后续已有大量新模型涌现（如 CellVerse 等），但截止投稿时未覆盖
- **评估空白**：缺乏统一 benchmark 是整个领域的痛点，论文指出但未提出具体解决方案
- **Scaling law 分析缺失**：现有最大单细胞 PLM 不到 1B 参数，scaling 行为尚不明确

## 与相关工作的对比
- **vs Lan et al. (2024), Szałata et al. (2024)**：这些综述从 Transformer 架构角度分析单细胞模型，本文首次从"语言建模"视角（PLM vs LLM 二分法）进行分析，更贴近 NLP 社区的思维方式
- **vs LLM4Cell (Dip et al., 2025)**：后者覆盖了 agentic models，时间更晚，补充了 agent 范式的讨论

## 启发与关联
- 这篇综述揭示了一个重要趋势：NLP 领域的"预训练-微调"范式正在被系统性地迁移到生物数据领域，其中 tokenization 是最关键的桥梁
- Cell-to-Sentence 的思路可以推广到其他非文本模态数据（如时间序列、传感器数据）的 LLM 适配
- 统一 benchmark 的缺失是普遍问题，在 medical AI、科学计算等交叉领域同样存在

## 评分
- 新颖性: ⭐⭐⭐ 综述本身无新方法，但首次从语言建模视角切入是一个新角度
- 实验充分度: ⭐⭐⭐ 综述无原创实验，模型总结表格较为完整但缺乏量化对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰、分类体系完整、配图直观，适合快速入门该领域
- 对我的价值: ⭐⭐⭐ 对了解 NLP 与计算生物学交叉领域有参考价值，但与主要研究方向关联不大
