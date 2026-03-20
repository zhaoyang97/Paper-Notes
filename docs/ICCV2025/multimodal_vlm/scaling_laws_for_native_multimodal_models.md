# Scaling Laws for Native Multimodal Models

**会议**: ICCV 2025 (Oral)  
**arXiv**: [2504.07951](https://arxiv.org/abs/2504.07951)  
**代码**: 无  
**领域**: 多模态VLM / Scaling Laws  
**关键词**: native multimodal, early fusion, late fusion, scaling laws, MoE, 架构设计  

## 一句话总结
通过训练457个不同架构和训练配比的模型进行系统性scaling law研究，发现Native Multimodal Models（NMM）的early-fusion架构（不依赖视觉编码器/tokenizer）在小参数量时优于late-fusion，训练更高效且部署更简单，结合MoE可进一步显著提升性能。

## 背景与动机
当前主流VLM（如LLaVA/InternVL）采用late-fusion架构——先独立预训练视觉编码器（如CLIP-ViT）和LLM，再通过connector连接进行多模态训练。这种方式样本效率高，但存在问题：视觉编码器的归纳偏置限制灵活性，多组件协调复杂，且不清楚这种架构是否inherently更好。另一个方向是Native Multimodal Models（NMM）——从零开始在所有模态上训练的统一模型。但NMM的架构设计空间（early vs late fusion、有无tokenizer、MoE等）缺乏系统性的scaling law研究。

## 核心问题
对于从零训练的native multimodal模型，early-fusion和late-fusion架构谁更优？在不同模型规模和数据规模下的scaling行为如何？如何优化NMM的架构选择？

## 方法详解

### 整体框架
这是一项实验研究而非方法论文。作者系统训练了457个不同配置的模型，覆盖：(1) 不同架构（early-fusion无视觉编码器 vs late-fusion有视觉编码器 vs 有视觉tokenizer）；(2) 不同模型规模；(3) 不同训练数据混合比例；(4) 不同MoE配置。通过拟合scaling law来分析各因素的影响。

### 关键发现
1. **Early-fusion不比late-fusion差**：核心发现——在相同参数量和训练数据下，不使用预训练视觉编码器的early-fusion模型并不天然劣于late-fusion模型。这挑战了社区的传统认知。更进一步，在**较小参数规模**时，early-fusion实际上表现**更好**——因为它不需要为视觉编码器分配额外参数和计算。

2. **Early-fusion的优势**：(a) 训练更高效——不需要先独立预训练视觉组件；(b) 部署更简单——只有一个统一模型而非多组件pipeline；(c) 更灵活——不受视觉编码器分辨率/宽高比的限制。这些实际优势使early-fusion成为更有前途的方向。

3. **MoE显著提升NMM性能**：将MoE引入NMM允许模型为不同模态学习特定的权重路径（modality-specific weights）。这与EVEv2的Divide-and-Conquer思想一致——模态间的干扰是NMM的核心挑战，MoE提供了一种高效的解耦方式。MoE在early-fusion架构上的提升尤为显著。

4. **Scaling Law的可预测性**：NMM的性能可以用标准power law关于模型参数和训练token数来拟合，这意味着可以通过小规模实验预测大规模训练的结果——降低了NMM研究的试错成本。

### 损失函数 / 训练策略
标准的next-token prediction（文本）+ 扩散/重建loss（视觉），不同架构变体有不同的具体配置。

## 实验关键数据
- 总共训练了**457个模型**，覆盖多种架构×规模×数据配比
- Early-fusion在小规模时优于late-fusion，大规模时持平
- MoE版本在各架构变体上一致带来显著提升
- Scaling law可以准确预测更大规模的性能
- 28张图表+13张表格的详尽分析

### 消融实验要点
- 视觉tokenizer（离散化）方案表现最差——信息损失不可恢复
- 数据混合比例对不同架构的影响不同——early-fusion对视觉数据比例更敏感
- MoE的专家数和激活比例有最优区间
- 训练效率：early-fusion达到相同性能所需的训练计算量更少

## 亮点
- **ICCV Oral，457个模型的大规模实证研究**：这是NMM领域迄今最系统的架构对比研究
- **"无视觉编码器更好"的反直觉发现**：直接挑战了CLIP-ViT+LLM的主流范式，与EVEv2和Web-SSL的发现形成闭环
- **MoE for NMM**的验证：为模态间干扰问题提供了额外的实证支持
- **Scaling Law的实用价值**：使NMM研究从"试错"走向"预测"，大幅降低研究成本
- **Apple出品**（Joshua Susskind），产业界对NMM方向的重视信号

## 局限性 / 可改进方向
- 457个模型虽多，但最大规模仍受限于计算资源
- 尚未在text-to-image/video generation任务上验证scaling law
- 数据质量的影响未充分探讨——高质量标注（如EVEv2的DenseFusion++）可能改变结论
- Scaling law的外推到极大规模（100B+）的可靠性未验证

## 与相关工作的对比
- **vs. EVEv2**：EVEv2专注于encoder-free VLM的最优训练策略（Divide-and-Conquer）；本文提供更系统的架构对比和scaling law——两者高度互补
- **vs. Scaling Language-Free Visual Repr**：Web-SSL证明SSL在相同数据上可以匹配CLIP；本文进一步证明NMM在from-scratch训练时early-fusion可以匹配late-fusion——共同指向"预训练视觉编码器并非必要"
- **vs. Chinchilla/Kaplan scaling laws**：将LLM的scaling law方法论扩展到多模态领域，填补了NMM的关键空白
- **vs. Mono-InternVL**：Mono-InternVL是encoder-free VLM的工程实践；本文是系统性的科学研究

## 启发与关联
- **核心启发**：如果early-fusion NMM足够好，那么整个VLM社区的默认范式（CLIP+LLM）可能需要重新审视
- MoE for NMM的发现与EVEv2的Divide-and-Conquer和Dynamic-DINO的MoE-Tuning形成一致性——模态/任务级别的解耦是多模态模型的关键设计原则
- Scaling law使得NMM研究可以"cheaper at scale"——用小模型实验预测大模型行为

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 457模型规模的系统研究前所未有，"early-fusion不比late-fusion差"的发现是paradigm-level的贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 28图13表，极其详尽的实验设计和分析
- 写作质量: ⭐⭐⭐⭐⭐ Oral水准的科学叙事，结论清晰有力
- 价值: ⭐⭐⭐⭐⭐ 对VLM社区的架构选择有深远指导意义，Scaling law为NMM研究建立了科学基础
