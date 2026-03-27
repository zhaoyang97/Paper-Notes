# Benchmarking Agentic Systems in Automated Scientific Information Extraction

**会议**: NeurIPS 2025
**arXiv**: [2510.00795](https://arxiv.org/abs/2510.00795)
**代码**: 有
**领域**: Agent / 科学信息提取
**关键词**: chemical information extraction, multimodal, agent benchmark, nanomaterials, scientific data

## 一句话总结
构建 ChemX——10 个人工标注的多模态化学数据提取数据集，系统评估 SOTA Agent 系统（ChatGPT Agent、SLM-Matrix、FutureHouse、nanoMINER）和 LLM（GPT-5 等），发现单 Agent 方法（GPT-5）在纳米酶数据集上达到 F1=0.58，超越专用多 Agent 系统。

## 研究背景与动机
1. **领域现状**：科学文献中的结构化信息提取（如化学属性、纳米材料参数）对研究至关重要但人工成本极高。
2. **现有痛点**：(1) 缺乏多模态（文本+表格+图片）的化学信息提取基准；(2) 不同 Agent 系统间缺乏公平对比；(3) 领域特定挑战（化学命名、单位转换等）未被系统评估。
3. **本文要解决什么？** 提供全面的化学信息提取评估框架和基准。

## 方法详解

### 关键设计
1. **10 个手工标注数据集**：覆盖纳米材料和小分子两大领域，包含不同提取难度
2. **系统化 Agent 评估**：统一评估流水线，比较单 Agent vs 多 Agent、专用 vs 通用方法
3. **多模态挑战**：需要从表格、图片和文本中联合提取信息

## 实验关键数据
| Agent 系统 | 纳米酶 F1 | 配合物 F1 |
|-----------|---------|---------|
| GPT-5 | 0.58 | 0.35 |
| ChatGPT Agent | 0.45 | 0.28 |
| 专用系统 | 0.40-0.50 | 0.25-0.30 |

### 关键发现
- 通用 LLM（GPT-5）在多数任务上超越专用的多 Agent 系统
- 化学命名和单位标准化仍是主要错误来源

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化化学信息提取Agent评估
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个数据集+多系统对比
- 写作质量: ⭐⭐⭐⭐ 评估全面
- 价值: ⭐⭐⭐⭐ 对 AI for Science 有重要推动
