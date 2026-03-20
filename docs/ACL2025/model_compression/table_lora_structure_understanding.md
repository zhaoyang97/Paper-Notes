# TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2503.04396](https://arxiv.org/abs/2503.04396)  
**代码**: [https://github.com/microsoft/TableLoRA](https://github.com/microsoft/TableLoRA)  
**领域**: 模型压缩  
**关键词**: LoRA, 表格理解, 结构化数据, 2D位置编码, PEFT  

## 一句话总结
TableLoRA 提出面向表格任务的专用 LoRA 模块，通过特殊 token 编码器改善表格序列化，并用 2D LoRA 编码单元格的行列位置信息，在参数高效微调设置下相比 vanilla LoRA 在 HiTab 上提升 5.9%，弥合了 LoRA 与全量微调之间 40.56% 的性能差距。

## 研究背景与动机

1. **领域现状**：表格数据在众多领域广泛使用，LLM 在 PEFT 范式下处理表格任务越来越重要。
2. **现有痛点**：(a) 表格序列化方式（markdown/HTML）影响模型理解，但现有方法仍无法准确识别表格结构（如同一列的对应关系）；(b) 二维表格结构被压平为一维序列后，行列位置信息只能通过注意力机制隐式学习，在低参数 PEFT 下学习不充分。
3. **核心矛盾**：表格的二维位置关系是理解表格结构的关键，但 LoRA 不显式编码这种结构信息。
4. **本文要解决什么？** 在 PEFT 低参数设置下让 LLM 更好地理解表格结构。
5. **切入角度**：直接通过模型架构设计告诉模型表格结构关系，而非依赖注意力机制隐式学习。
6. **核心idea一句话**：用特殊 token 替代 markdown 标记改善序列化 + 用低秩行列位置编码注入每一层，显式告知 LLM 表格结构。

## 方法详解

### 整体框架

两个组件并行工作：(1) Special Tokens Encoder 在 Transformer 层之前引入 [tab]/[row]/[cell] 特殊 token 嵌入；(2) 2D LoRA 在每层将行列索引的低秩嵌入与 token 嵌入融合。

### 关键设计

1. **特殊 Token 编码器 (Special Tokens Encoder)**:
   - 做什么：用 [tab]、[row]、[cell] 替代 markdown/HTML 标记进行表格序列化
   - 核心思路：受 p-tuning 启发，这些特殊 token 有可学习的嵌入，通过微调时的梯度传播学习表格结构语义
   - 设计动机：传统标记符号（|、\n）不是专门为表格设计的，专用 token 可以更好地表示结构边界

2. **2D LoRA**:
   - 做什么：将行列索引信息编码为低秩嵌入，注入每层的 token 表示中
   - 核心思路：为行索引和列索引分别创建低秩嵌入 $E_{row} \in \mathbb{R}^{R \times r}$ 和 $E_{col} \in \mathbb{R}^{C \times r}$，通过上投影矩阵扩展到隐藏维度后添加到 token 表示。与原始 LoRA 并行工作
   - 设计动机：2D 位置信息量相对于 token 语义较少，用低秩编码足够且参数效率高

### 损失函数 / 训练策略
标准任务损失，与 LoRA 联合微调。2D LoRA 在每层与标准 LoRA 并行。

## 实验关键数据

### 主实验

3 个模型（Llama-2-7B、Llama-3-8B、Qwen2-7B），4个数据集（HiTab、WikiTableQuestions、TabFact、SQA）。

| 方法 | HiTab ↑ | WTQ ↑ | TabFact ↑ |
|------|---------|-------|-----------|
| LoRA | 38.5 | 55.2 | 72.8 |
| **TableLoRA** | **44.4 (+5.9)** | **57.1** | **74.5** |
| 全量微调 | 52.9 | 59.3 | 76.1 |

### 消融实验

| 配置 | HiTab Acc | 说明 |
|------|----------|------|
| TableLoRA (完整) | **44.4** | Special Tokens + 2D LoRA |
| 仅 Special Tokens | 41.2 | 去掉 2D LoRA |
| 仅 2D LoRA | 42.8 | 去掉 Special Tokens |
| LoRA 基线 | 38.5 | 无表格特定设计 |

### 关键发现
- **HiTab 上提升最大（5.9%）**：HiTab 包含层级表头，最需要精确的行列位置理解
- **弥合 LoRA 与全量微调 40.56% 的差距**：在极低额外参数下有效提升结构理解
- **两个组件互补**：Special Tokens 改善序列化表示，2D LoRA 提供位置信息

## 亮点与洞察
- **首个表格专用 LoRA**：将领域知识（二维结构）直接编码到 LoRA 架构中的思路可以推广到其他结构化数据（如图、代码 AST）
- **低秩编码位置信息**：行列索引的信息量确实远小于语义内容，用低秩编码非常合理

## 局限性 / 可改进方向
- 仅覆盖平面表格和简单层级表头，复杂合并单元格、嵌套表格未涉及
- 未与专门的表格 LLM（如 TableGPT）对比
- 最大行列数受预设嵌入大小限制

## 相关工作与启发
- **vs Vanilla LoRA**: LoRA 不了解表格结构，TableLoRA 通过专用编码解决
- **vs TableGPT/TableLLM**: 这些模型通过大规模训练学习表格理解，TableLoRA 仅需 PEFT 级别的开销

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个表格专用 LoRA，2D 位置编码的设计思路简洁有效
- 实验充分度: ⭐⭐⭐⭐ 3模型4数据集+对照实验+详细分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对表格处理社区和 PEFT 社区都有参考价值
