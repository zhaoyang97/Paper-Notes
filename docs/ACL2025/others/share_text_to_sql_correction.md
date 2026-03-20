---
title: "SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL"
conference: "ACL 2025"
arxiv: "2506.00391"
code: "https://github.com/quge2023/SHARE"
domain: "others"
keywords: ["text-to-sql", "self-correction", "small language model", "action model", "error correction"]
---

# SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL

## 一句话总结

提出 SHARE 框架，通过三个专用小语言模型（SLM）组成的顺序管道，将 SQL 查询转换为逐步动作轨迹并分别修正 schema 错误和逻辑错误，从而以低成本高效辅助 LLM 进行 Text-to-SQL 自纠正。

## 研究背景与动机

1. 当前 Text-to-SQL 的自纠正方法严重依赖 LLM 的递归自调用，导致计算开销呈乘法增长，成本很高
2. Self-debugging 方法依赖数据库执行反馈，但 SQLite 等主流数据库的反馈过于简略，难以精确定位错误
3. 数据库直接执行访问在隐私敏感场景中受到严格限制，self-debugging 方法实际应用受限
4. LLM 直接对声明式 SQL 进行自纠正时，无法展示底层推理路径，存在自增强偏差（self-enhancement bias）
5. 现有方法往往需要多轮迭代加上精心设计的 prompt，人力和计算成本都很高
6. 作者提出用 SLM 作为辅助纠正器的思路，将声明式 SQL 转换为可暴露推理过程的动作轨迹，实现更精确的错误定位和更高效的修正

## 方法详解

### 整体框架

SHARE 采用 assistant-based 自纠正范式：生成器 LLM 产生初始 SQL，三个专用 SLM（均<8B参数）组成顺序管道进行纠正：

1. **Base Action Model (BAM)**: 将 SQL 转换为 pandas-like API 的动作轨迹，暴露底层推理路径
2. **Schema Augmentation Model (SAM)**: 检测并修正轨迹中的 schema 链接错误（表名、列名等）
3. **Logic Optimization Model (LOM)**: 修正逻辑推理错误（操作顺序、条件逻辑等）

### 关键设计

- **动作轨迹表示**: 将声明式 SQL 分解为 pandas-like 函数调用序列（如 where, select, groupby, orderby），使推理过程可视化
- **两阶段 SAM 训练**: 第一阶段学习用 [MASK] 标记 schema 元素，第二阶段学习正确填充 schema 链接
- **动作扰动策略**: 三种扰动类型 — ADD（插入动作）、DELETE（删除动作）、SUBSTITUTE（替换动作/参数），用于 LOM 数据增强
- **层级自演化策略 (Hierarchical Self-Evolve)**: 用 BAM 自动合成和增强 SAM/LOM 的训练数据，而非反复查询 GPT-4o，降低标注成本

### 训练策略

- BAM 使用 GPT-4o 蒸馏构建 13K 训练数据，确保高质量
- SAM/LOM 通过层级自演化利用 BAM 生成训练数据，数据效率高
- 所有模型采用 LoRA 微调，backbone 为 Llama-3.1-8B 或 Phi-3-Mini-3.8B
- 训练集约 13K-15K 样本，在 4×A100 上进行

## 实验关键数据

### 主实验（GPT-4o 生成器 + 单轮修正）

| 方法 | BIRD EX(%) | SPIDER EX(%) |
|------|-----------|-------------|
| GPT-4o (baseline) | 55.87 | 77.10 |
| Self-Correction | 55.28 (-0.59) | 75.90 |
| Self-Consistency | 58.75 | 81.80 |
| MAGIC | 59.53 | 85.66 |
| +SHARE-3.8B | 60.89 | 84.00 |
| **+SHARE-8B** | **64.14** | **85.90** |

### 跨模型泛化性（BIRD 数据集）

| 生成器 | 原始 EX(%) | +SHARE-8B EX(%) | 相对提升 |
|--------|-----------|----------------|---------|
| Claude-3.5-S | 49.41 | 63.56 | +28.64% |
| GPT-4o-mini | 49.09 | 59.64 | +21.49% |
| Llama-3.1-70B | 53.91 | 61.93 | +14.88% |
| DS-Coder-6.7B | 34.57 | 51.24 | +48.22% |

### 消融实验

- 50% 训练数据就超过 MAGIC（60.71% vs 59.53%），验证数据效率
- DK/Realistic 等 robust 测试集分别提升 11.20%/8.10%
- 泛化至未见 SQL 方言也有效

### 关键发现

1. GPT-4o 直接 self-correction 反而降低性能（55.87→55.28），说明 LLM 在声明式 SQL 上难以有效自纠正
2. SHARE 在单轮交互中即可实现 14.80%（BIRD）和 11.41%（SPIDER）的相对提升
3. 仅用 50% 训练数据即可超过 SOTA，训练效率非常高
4. SHARE 能跨模型、跨方言泛化，并非简单拟合特定模型的错误模式

## 亮点与洞察

- **小模型辅助大模型**: 用 <8B 的 SLM 辅助任意 LLM 进行纠正，计算成本远低于 LLM 自纠正
- **声明式→过程式转换**: 将声明式 SQL 转换为过程式动作轨迹是核心创新，使错误定位从"黑盒"变为"白盒"
- **模块化错误处理**: 将 schema 错误和逻辑错误拆分给不同专家模型处理，各自专精更有效
- **自演化减少标注**: 层级自演化策略避免重复调用教师 LLM，大幅降低数据构建成本

## 局限性

- 依赖预定义的 action space（pandas-like API），可能无法覆盖所有 SQL 方言的复杂特性
- BAM 的训练数据依赖 GPT-4o 蒸馏，初始质量受限于教师模型
- 训练数据量的扩展效果在 challenging 级别查询上不够稳定
- 目前仅评估在 SQLite 相关 benchmark 上，其他数据库系统的效果未知

## 相关工作

- Self-debugging: 基于执行反馈的迭代修正（Zhong et al., 2023; Li & Xie, 2024）
- Self-correction: 无需执行反馈的自主纠正（Liu & Tan, 2024; Askari et al., 2024）
- Action model: 将任务分解为过程式动作轨迹（Zhang et al., 2024）
- MAGIC: 当前 SOTA 的 text-to-SQL 自纠正方法

## 评分

- **新颖性**: ★★★★☆ — 动作轨迹转换加模块化纠正的设计很有创意
- **技术深度**: ★★★★☆ — 三阶段管道设计和自演化训练策略设计精细
- **实验充分性**: ★★★★★ — 4个benchmark + 多个生成器 + 低资源分析 + 跨方言测试
- **实用价值**: ★★★★☆ — 低成本辅助纠正在隐私受限的实际场景中非常有价值
