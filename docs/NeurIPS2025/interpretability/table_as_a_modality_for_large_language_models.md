---
title: >-
  [论文解读] Table as a Modality for Large Language Models
description: >-
  [NeurIPS 2025][可解释性][Table Reasoning] 提出 TaMo 框架，将表格作为独立模态通过超图神经网络编码其结构信息，与 LLM 的文本模态融合，在多个表格推理基准上相比纯文本方法平均提升 42.65%，且在结构鲁棒性上接近 GPT-4。 当前 LLM 处理表格任务的主流方式是将表格序列化为文本…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "Table Reasoning"
  - "多模态"
  - "Hypergraph"
  - "Permutation Invariance"
  - "Table QA"
---

# Table as a Modality for Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2512.00947](https://arxiv.org/abs/2512.00947)  
**代码**: [有](https://github.com/liyaooi/TAMO)  
**领域**: 可解释性  
**关键词**: Table Reasoning, Multimodal LLM, Hypergraph, Permutation Invariance, Table QA

## 一句话总结

提出 TaMo 框架，将表格作为独立模态通过超图神经网络编码其结构信息，与 LLM 的文本模态融合，在多个表格推理基准上相比纯文本方法平均提升 42.65%，且在结构鲁棒性上接近 GPT-4。

## 研究背景与动机

当前 LLM 处理表格任务的主流方式是将表格序列化为文本（如 Markdown 格式）后直接输入模型。然而这种方式存在根本性问题：**表格的结构信息在序列化过程中丢失了**。

论文通过设计 StructQA 基准（专注于表格结构理解的诊断数据集）揭示了这一问题。该基准考虑了表格数据特有的**置换不变性**属性：行列任意重排不应改变表格语义，模型应对结构等价的表格给出一致的答案。实验发现：

- 包括 Llama2-7B、GPT-3.5、GPT-4 在内的主流 LLM 在面对置换后的表格时性能大幅下降
- 除 GPT-4 外，所有模型的答案鲁棒性低于 40%
- 专门为表格训练的 TableLlama 在 StructQA 上仅达 6.47%

这种失败对人类而言微不足道——说明根本原因不是理解能力不足，而是**序列化带来的表示瓶颈**。论文的核心洞察是：正如图像和音频需要专用编码器，表格也应被视为一种独立模态。

## 方法详解

### 整体框架

TaMo 是一个多模态框架，包含三个组件：
1. **超图增强表格编码器**：捕获表格结构信息，生成结构嵌入
2. **模态对齐接口**：将结构嵌入投射到 LLM 语义空间
3. **LLM 推理**：融合结构嵌入和文本嵌入，自回归生成答案

### 关键设计

1. **超图建模表格结构**：

    - 将表格建模为超图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：叶子单元格（不包含子单元格的）为节点，分支单元格（包含子单元格的表头等）为超边
    - 简单平表：每个单元格是节点，每行/列是一个超边
    - 复杂层级表（如 HiTab）：根据层级关系自然转化为超图
    - 核心动机：行列置换只改变排列不改变图结构（节点和边集合不变），天然满足置换不变性

2. **HyperTrans 编码器**：

    - 采用两个多集函数（multiset functions）交替更新节点和超边表示
    - 节点→超边聚合：$\mathbf{x}_e^{t+1} = \text{Fusion}(\mathbf{x}_e^t, \text{Multiset}_1(\{\mathbf{x}_v^t | v \in e\}))$
    - 超边→节点聚合：$\mathbf{x}_v^{t+1} = \text{Multiset}_2(\{\mathbf{x}_e^{t+1} | v \in e\})$
    - 多集函数用 Set Transformer 参数化，包含多头注意力和前馈网络
    - 多集函数的置换不变性保证了编码器整体的置换不变性

3. **模态对齐与融合**：

    - 用 MLP 将超图输出的节点和超边表示（pooling 后）投射到 LLM 嵌入空间：$\mathbf{X}_{st} = \text{MLP}(\text{Pooling}(\hat{\mathbf{X}}_\mathcal{V}, \hat{\mathbf{X}}_\mathcal{E}))$
    - 同时保留文本序列化的表格嵌入 $\mathbf{X}_{tt}$（提供细粒度语义内容 "what"）
    - 结构嵌入 $\mathbf{X}_{st}$ 以类似 soft prompt 的方式注入 LLM 前端（提供全局关系上下文 "where"）
    - 两个流互补非冗余：消融实验证明去掉任何一个都会损失性能

### 损失函数 / 训练策略

- 使用标准的 next token prediction（自回归交叉熵损失）
- 支持三种训练模式：
    - **Frozen LLM**：仅训练表格编码器和对齐层
    - **LoRA**：加上 LLM 的 LoRA 微调
    - **SFT**：全参数监督微调
- 每个数据集独立训练以建立性能上界

## 实验关键数据

### 主实验

| 设置 | StructQA | HiTab | WikiTQ | WikiSQL | FeTaQA |
|------|----------|-------|--------|---------|--------|
| Zero-shot | 8.60 | 7.77 | 14.50 | 21.44 | 20.08 |
| Prompt Tuning | 37.80 | 26.26 | 29.86 | 61.24 | 29.94 |
| **TaMo (Frozen)** | **59.07** | **48.86** | **37.06** | **76.45** | **36.52** |
| △ vs Prompt Tuning | ↑56.27% | ↑86.06% | ↑24.11% | ↑24.84% | ↑21.98% |
| LoRA | 45.67 | 50.76 | 37.13 | 57.10 | 35.80 |
| **TaMo+LoRA** | **70.80** | **59.22** | **43.53** | **84.43** | **37.43** |
| SFT | 62.73 | 54.80 | 43.28 | 79.86 | 37.37 |
| **TaMo+SFT** | **71.60** | **63.89** | **45.81** | **85.90** | **39.01** |
| GPT-4 | 51.40 | 48.40 | 68.40 | 47.60 | 21.70 |
| DeepSeek-R1 | 57.47 | 63.89 | 75.76 | 71.91 | 13.10 |

Frozen LLM 设置下 TaMo 平均提升 **42.65%**；在 StructQA 和 WikiSQL 上 TaMo+SFT 显著超越 GPT-4.1 和 DeepSeek-R1。

### 消融实验：表格编码器结构学习评估

| 配置 | F1 Score | 说明 |
|------|----------|------|
| MLP head (无编码器) | 5.39 | 无法识别行列结构 |
| + 随机初始化编码器 | 49.73 | 超图归纳偏置本身有效 |
| + StructQA 预训练编码器 | **71.32** | 结构学习最佳 |
| + WikiTQ 预训练编码器 | 62.63 | 跨数据集泛化 |
| + WikiSQL 预训练编码器 | 68.00 | 跨数据集泛化 |

### 关键发现

- 结构信息对冻参 LLM 的提升最大（+42.65%），说明结构是文本序列化无法获取的信息
- 注意力可视化显示 TaMo 使 LLM 更关注与正确答案相关的 token（如正确单元格和相关上下文）
- 置换鲁棒性测试中 TaMo 始终优于纯文本方法，答案一致性最高
- 7B 参数的 TaMo+SFT 在结构密集型任务上超越了 GPT-4.1 和 DeepSeek-R1

## 亮点与洞察

- **将表格视为独立模态**的理念很新颖，类比图像/音频的多模态 LLM 设计思路自然且有效
- 超图建模表格是一个精巧的选择：天然处理层级表格、内置置换不变性、统一了简单和复杂表格
- StructQA 基准本身就是重要贡献——揭示了当前 LLM 表格理解的根本缺陷
- 作为"即插即用"模块，不需要修改 LLM 架构，通用性强

## 局限与展望

- 依赖预结构化的表格输入，嵌入在非结构化文本中的表格需要额外预处理
- 目前仅支持单轮静态表格理解，动态多步推理和多轮对话待探索
- 不同文本序列化模板（Markdown vs SQL）与结构模态的交互效果未系统研究
- 虽然跨数据集泛化有一定展示，但缺乏大规模多模态指令数据的预训练

## 相关工作与启发

- 与 TAPAS/TAPEX 等传统表格模型的区别：TaMo 专为 decoder-only LLM 设计模态接口
- 与表格编码器（TaBERT、TabNet、HyTrel）的区别：那些仅做表示学习，不能处理文本+表格联合推理
- 对未来 LLM 处理结构化数据（知识图谱、数据库）的设计有借鉴意义

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — "表格即模态"的理念是首次提出并系统验证
- 实验充分度: ⭐⭐⭐⭐ — 5 个数据集 + 多种训练设置 + 消融 + 可视化
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，StructQA 的诊断实验很有说服力
- 价值: ⭐⭐⭐⭐⭐ — 对 LLM 表格推理有实质性推进，即插即用设计实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](../../ACL2025/interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](emergence_of_linear_truth_encodings_in_language_models.md)
- [\[NeurIPS 2025\] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)

</div>

<!-- RELATED:END -->
