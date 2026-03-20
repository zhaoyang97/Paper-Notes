---
title: "Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation (CoSyn)"
conference: "ACL 2025"
arxiv: "2502.14846"
code: "https://yueyang1996.github.io/cosyn"
domain: "multimodal_vlm"
keywords: ["synthetic data", "text-rich image", "vision-language model", "code generation", "chart understanding"]
---

# CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation

## 一句话总结

提出 CoSyn 框架，利用纯文本 LLM 的代码生成能力自动合成多样化的文本丰富型图像及对应指令微调数据，构建 400K 图像 + 2.7M 指令数据集，在 7 个 benchmark 上达到开源 SOTA 并超越 GPT-4V。

## 研究背景与动机

1. 文本丰富型图像（图表、文档、表格等）的理解是 VLM 的关键应用场景，但模型在此类任务上的表现仍不理想
2. 根本原因是高质量、多样化的文本丰富型视觉-语言数据严重不足
3. 现有图表 VQA 数据集规模小、图表类型有限、问题模板化，难以支持泛化
4. 文本丰富型图像本身大多通过代码渲染生成（HTML、LaTeX、Matplotlib 等），这提供了利用代码作为中间表征的天然机会
5. 纯文本 LLM 在代码生成方面已经非常强大，可以利用这一能力合成图像
6. 现有 VLM 在训练数据外的新任务上泛化能力差，需要针对性的合成数据生成方案

## 方法详解

### 整体框架

给定自然语言查询（如"nutrition fact labels"），CoSyn 通过四个阶段生成多模态数据：
1. **Topic Generation**: 基于采样的 persona 生成多样化主题
2. **Data Generation**: 生成详细内容数据
3. **Code Generation**: 用 LLM 生成可执行代码渲染图像
4. **Instruction Generation**: 以代码为上下文生成问答和解释

核心公式: P(I,T|q) = P_LM(C|q) · P(I|C) · P_LM(T|C)

### 关键设计

- **11种渲染工具**: Matplotlib, Plotly, Vega-Lite（图表），LaTeX, HTML（文档/表格），Mermaid, Graphviz（流程图），SVG, Asymptote（向量图形），Lilypond（乐谱），RDKit（化学结构）
- **20条生成管线**: 基于 11 种工具设计，每条管线包含主题/数据/代码/指令四阶段
- **Persona 增强多样性**: 使用 200K persona（如"a sci-fi novelist who likes alien worlds"），在主题生成阶段注入，显著提升合成数据多样性
- **CoT 推理数据**: 每个样本包含 (question, explanation, short answer) 三元组，支持链式思维推理
- **合成 Pointing 数据**: 可生成坐标标注数据，支持 web agent 中的点击预测任务

### 数据集

- **CoSyn-400K**: 9 类图像（图表、文档、数学题、表格、流程图、向量图形、乐谱、电路图、化学结构）
- 400K 图像 + 2.7M 行指令微调数据
- 基于 DataDreamer 框架构建，代码/数据生成用 Claude-3.5-Sonnet，指令生成用 GPT-4o-mini

### 训练策略

- 架构: CLIP ViT-L/14 + Mistral-7B，MLP 连接
- 两阶段训练: (1) 预训练 + (2) SFT
- SFT 数据来源: 评测数据集(138K) + 辅助数据集(1M) + 合成数据集(400K)

## 实验关键数据

### 主实验（7个 text-rich benchmark 平均分）

| 模型 | 平均 | ChartQA | DocVQA | InfoVQA |
|------|------|---------|--------|---------|
| GPT-4V | 72.8 | 78.1 | 87.2 | 75.1 |
| Gemini 1.5 Flash | 76.2 | 85.4 | 89.9 | 75.3 |
| LLaVA-OV-7B | 72.4 | 80.0 | 87.5 | 68.8 |
| Llama 3.2 11B | 77.0 | 83.4 | 88.4 | 63.6 |
| **Ours (7B)** | **80.9** | **86.3** | **90.0** | **70.5** |
| Ours (zero-shot) | 74.7 | 80.8 | 82.9 | 59.8 |

### 消融实验（数据组合）

| 数据组合 | 与GPT-4V差距 |
|---------|-------------|
| 仅辅助(1M) | -14.1% |
| 仅合成(400K) | ≈ GPT-4V |
| 辅助+合成 | > GPT-4V |
| 评测+辅助 | +1.4% |
| 评测+合成 | +3.6% |

### 关键发现

1. **零样本超越 GPT-4V**: 不使用任何评测数据集训练样本，仅用辅助+合成数据即可超越 GPT-4V
2. **新任务适应**: 仅用 CoSyn 生成 7K 营养标签样本微调，即可在 NutritionQA 上超过大部分开源 VLM
3. **缓解数据偏差**: 加入合成数据后，ChartQA 人工/机器问题的性能差距从 21.8% 降至 14.2%
4. **多样性关键**: 使用多种渲染工具（5工具）比单一工具（Matplotlib）不仅多样性更高（0.607 vs 0.572），性能也更好
5. **Pointing 数据**: 合成 pointing 数据在 ScreenSpot 点击预测任务上达到 SOTA

## 亮点与洞察

- **代码即表征**: 利用代码作为图像的文本化中间表征是核心创新——代码既可渲染图像，又可作为 LLM 生成问答的上下文
- **纯文本 LLM 驱动多模态**: 完全不需要视觉模型参与数据合成过程，纯文本 LLM + 代码渲染即可生成高质量多模态数据
- **Persona 驱动多样性**: 引入 200K persona 有效解决 LLM 合成数据重复性问题
- **数据效率超强**: 400K 合成图像即可匹配需要数百万真实图像训练的开源模型

## 局限性

- 合成数据质量依赖底层 LLM（Claude-3.5-Sonnet）的代码生成能力
- 某些图像类型（照片、手绘图）无法通过代码渲染
- CoT 评估受限于现有 benchmark 的严格字符串匹配评估方式
- 合成数据的视觉风格与真实扫描文档/照片仍有 domain gap
- 仅测试 7B 规模模型，更大规模的扩展性未知

## 相关工作

- VLM 架构演进（Tsimpoukelli et al., 2021; LLaVA, Molmo）
- 文本丰富图像理解数据集（ChartQA, DocVQA, PlotQA）
- 合成数据生成（FigureQA, DVQA, PixMo-docs）
- 代码引导图像生成（Han et al., 2023; He et al., 2024）
- Persona 增强多样性（Ge et al., 2024）

## 评分

- **新颖性**: ★★★★☆ — "代码即表征"连接图像和文本的思路很有创意
- **技术深度**: ★★★★☆ — 20条管线 × 11种工具的系统化设计，工程壁垒高
- **实验充分性**: ★★★★★ — 7个基准 + 多维消融 + 新任务适应 + 偏差分析 + pointing
- **实用价值**: ★★★★★ — 直接产出可用的 400K 合成数据集和通用框架
