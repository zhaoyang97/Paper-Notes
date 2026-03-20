# GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding

**会议**: ACL 2025  
**arXiv**: [2409.04183](https://arxiv.org/abs/2409.04183)  
**代码**: https://github.com/codefuse-ai/GALLa  
**领域**: LLM/NLP  
**关键词**: code understanding, graph alignment, GNN, AST/DFG, model-agnostic

## 一句话总结
提出 GALLa，通过 GNN 编码代码的 AST/DFG 结构图并用跨模态适配器对齐到 LLM 嵌入空间，在微调时作为辅助任务注入代码结构信息，推理时丢弃 GNN 和 adapter 实现零额外开销，在 5 个代码任务 × 7 个基线 LLM（350M-14B）上持续提升。

## 研究背景与动机

1. **领域现状**：代码 LLM（Code LLaMA、DeepSeek-Coder等）通过扩大模型和数据规模取得了显著进展，但本质上仍是标准的 decoder-only Transformer，仅将源代码视为文本 token 序列。
2. **现有痛点**：代码具有丰富的图结构语义（AST 抽象语法树、DFG 数据流图、CFG 控制流图），这些信息在纯文本表征中被忽略。现有图增强方法分三类：(1) 修改注意力掩码编码图结构（如 GraphCodeBERT，不兼容 decoder-only LLM）；(2) 将图线性化为文本（仅适用简单树，如 AST，不适用含环的 DFG）；(3) 修改位置编码（需大规模重训练）。
3. **核心矛盾**：图增强和大规模预训练 LLM 不兼容——要么改架构丢预训练知识，要么不用图丢结构信息。
4. **切入角度**：受 VLM 领域（LLaVA 等）用轻量 adapter 桥接视觉和语言模态的启发，用外部 GNN + adapter 处理图信息，LLM 架构完全不变。
5. **核心 idea 一句话**：图信息通过 GNN+adapter 在训练时注入 LLM 理解，推理时丢弃——训练时有图，推理时零开销。

## 方法详解

### 整体框架
三模块：GNN（编码 AST/DFG）→ Adapter（投影到 LLM 嵌入空间）→ LLM（解码生成）。训练分两阶段，数据分图对齐数据和下游任务数据，两者完全独立。

### 关键设计

1. **图编码与输入**:
   - GNN 编码 AST 和 DFG 的节点特征（使用文本编码器初始化）和边结构，输出上下文化节点表征 $H \in \mathbb{R}^{n_v \times d_{gnn}}$。
   - Adapter 用可学习的 query vectors 通过交叉注意力将节点表征压缩为固定数量的图 token $X_g \in \mathbb{R}^{n_g \times d_{lm}}$。
   - 图 token 与文本 token 拼接输入 LLM，loss 仅在文本 token 上计算。

2. **两阶段训练**:
   - **Stage 1（图编码器预训练）**: 冻结 LLM，仅训练 GNN + adapter。任务：Graph2Code（给图生成源代码），让 GNN+adapter 学习生成 LLM 可理解的图 token。
   - **Stage 2（图-LLM 对齐）**: 解冻 LLM，三模块联合训练。任务：Graph2Code + GraphQA（预测边是否存在、预测节点的父/子节点）。**同时**在下游任务数据上微调 LLM（此时不走 GNN，直接文本输入）。

3. **推理时丢弃 GNN**:
   - 训练完成后，丢弃 GNN 和 adapter，LLM 独立推理——速度与基线 LLM 完全一致。
   - 关键假设：通过图对齐训练，LLM 内部已经习得了代码结构理解的能力，推理时不再需要显式图输入。

4. **模型和任务无关性**:
   - GNN 选择取决于图类型（有向/无向），LLM 选择取决于应用场景——框架本身不限制。
   - 图对齐数据来自 CodeNet（240K Python + 75K Java），与下游任务数据完全独立。

## 实验关键数据

### 主实验（多任务微调 MFT）

| 模型 | 无 GALLa | G2C | G2C+GraphQA | 平均提升 |
|------|:---:|:---:|:---:|:---:|
| CodeGen 350M | 34.7 | 35.2 (+1%) | 36.6 (+5%) | +5% |
| StarCoder 1B | 14.0 | 16.7 (+20%) | 18.9 (+36%) | +36% |
| Phi-1 1.3B | 基线 | 提升 | 更大提升 | 显著 |
| LLaMA3-8B | 基线 | 提升 | 提升 | 一致 |
| Qwen2.5-Coder-7B | 基线 | 提升 | 提升 | 一致 |

**5 个任务**: 代码翻译(pass@1)、克隆检测(F1)、缺陷检测(Acc)、代码摘要(BLEU)、代码修复(EM)。

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅 G2C | 有效 | 基础图理解 |
| G2C + GraphQA | 更优 | 深层结构理解 |
| AST only / DFG only | 都有效 | 两者互补 |
| 跨语言泛化 | 有效 | 对齐数据仅含 Python+Java，但在 JavaScript/C 上也有提升 |

### 关键发现
- **小模型受益更大**：StarCoder 1B 提升 36%，而大模型提升幅度较小但仍一致。
- **跨语言结构知识迁移**：图对齐数据不含的编程语言也能受益，说明 LLM 学到了通用的结构理解能力。
- **GraphQA 比 G2C 更重要**：问答式结构任务迫使模型深入理解图拓扑，而非仅仅记忆图-代码映射。

## 亮点与洞察
- **"训练时有图，推理时无图"的设计哲学**极其实用：推理零开销意味着可直接替换任何现有代码 LLM。
- **GNN 作为"图 tokenizer"的类比**：Stage 1 相当于训练一个图 tokenizer（类似 VLM 中的视觉 tokenizer），使 LLM 能"读懂"图信息。
- **图对齐数据与下游任务数据分离**：不要求下游任务数据有图结构标注，大幅降低了使用门槛。

## 局限性 / 可改进方向
- AST/DFG 提取需要语法正确的完整代码，对代码片段或有语法错误的代码不适用。
- 仅测试了 Python 和 Java 的图对齐数据，更多语言的效果有待验证。
- GNN 处理大型代码文件的图可能面临可扩展性问题。

## 相关工作与启发
- **vs GraphCodeBERT**: 通过修改注意力掩码编码 DFG，但不兼容 decoder-only LLM；GALLa 完全外部处理。
- **vs TransCoder-IR**: 用 LLVM IR 作为中间表示对齐，但 IR 也是文本 token，结构信息有限；GALLa 直接处理图结构。

## 评分
- 新颖性: ⭐⭐⭐⭐ 图结构与 LLM 的优雅桥接方案，"训练时有图推理时无图"设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 7 模型(350M-14B) × 5 任务 + 跨语言泛化验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，两阶段训练逻辑连贯
- 价值: ⭐⭐⭐⭐⭐ 推理零开销+模型无关+任务无关，实用性极强
