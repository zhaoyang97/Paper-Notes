# Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning

**会议**: ICLR 2026  
**arXiv**: [2504.01445](https://arxiv.org/abs/2504.01445)  
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: systematic generalization, meta-learning for compositionality, ARC, abstract reasoning, few-shot learning

## 一句话总结
提出 Compositional-ARC 数据集评估模型在抽象空间推理中的系统性泛化能力——从已知基础几何变换（如平移、旋转）泛化到未见过的变换组合。一个仅 5.7M 参数的 MLC 训练的 encoder-decoder 模型在系统性任务上达到 78.26%，与 ARC Prize 2024 冠军的 8B 模型+TTT 持平，远超 GPT-4o、o3-mini 等（<3%）。

## 研究背景与动机

1. **领域现状**：系统性泛化（systematic generalization）是人类认知的核心能力——理解已知组件后能自动推广到新组合。LLM 虽在各领域取得进展，但在组合性泛化测试中表现不佳。
2. **现有痛点**：Lake & Baroni (2023) 提出的 Meta-Learning for Compositionality (MLC) 在伪语言任务上实现了人类级别的系统性泛化，但这一方法是否适用于非语言领域（如空间推理）仍未探索。
3. **核心矛盾**：当前 LLM（包括 o3-mini、GPT-4o）在标准推理任务上表现优异，但面对需要从基础组件重组到新组合的场景时系统性失败——不是因为能力不够，而是缺乏组合性泛化的训练范式。
4. **本文要解决什么**：(1) 设计一个评估空间推理系统性泛化的 benchmark；(2) 验证 MLC 可超越语言领域。
5. **切入角度**：利用几何变换的封闭性（两个有效变换的组合仍为有效变换），设计 ARC 风格的 2D 网格任务，测试模型能否从基础变换和 level-1 组合推断未见过的 level-2 组合。
6. **核心 idea 一句话**：将 MLC 从语言推广到视觉空间推理，证明小模型 + 正确的训练范式可以在组合性泛化上大幅超越万倍参数的 LLM。

## 方法详解

### 整体框架

1. **数据集（Compositional-ARC）**：10×10 网格上的 2D 对象 → 5 种基本几何变换（平移/旋转/反射/扩展/变色） → 通过 3 种指示器（形状/颜色/邻居关系）选择变换目标 → level-1 组合（2 个指示器）→ level-2 组合（3 个指示器，TEST SET）
2. **模型训练（MLC）**：Transformer encoder-decoder（5.7M 参数）在 10 万个 episode 上训练，每个 episode 有不同的"视觉解释文法"
3. **评估**：给定基础变换和 level-1 组合的 study examples，预测未见过的 level-2 组合

### 关键设计

1. **Compositional-ARC 数据集**:
   - 做什么：评估模型从已知变换泛化到新变换组合的能力
   - 核心思路：
     - 5 种基础变换：平移（右/下 1 格）、旋转（±90°）、反射（水平/垂直）、扩展（左/上方向）、变色（红/橙）
     - 3 种指示器：形状（如 L 形物体做平移）、颜色（如绿色物体做反射）、邻居（邻近特定物体时做扩展）
     - 组合层次：level-1 = 2 个指示器组合（如形状+颜色 → 平移+反射）；level-2 = 3 个指示器组合（形状+颜色+邻居 → 平移+反射+扩展）
     - 系统性测试：study examples 只展示基础变换和 level-1 组合，模型必须推断未见过的 level-2 组合
   - 设计动机：利用几何变换的封闭性和 ARC 的 2D 网格表示，构建了一个干净、可控的组合性泛化测试床。

2. **Meta-Learning for Compositionality (MLC) 扩展到空间推理**:
   - 做什么：训练小模型学会从 few-shot examples 中推断视觉解释文法并组合
   - 核心思路：每个训练 episode 有不同的"视觉解释文法"（如"黄色物体做平移"在不同 episode 中可能是"黄色物体做旋转"），迫使模型学会从 study examples 推断文法而非记忆固定映射。
   - 编码方式：10×10 网格划分为 2×2 patches（25 个 patch/grid），每 patch 编码为嵌入向量。用 1D 位置编码标记 grid pair 顺序 + 2D 位置编码捕获空间信息。
   - 辅助任务：训练时还要求模型复现 study examples 的输出（copy task），增强对 study examples 的理解。
   - 模型架构：3 层 encoder + 3 层 decoder，8 heads, dim=128, FFN=768, GELU, 共 5.7M 参数。

### 损失函数 / 训练策略

- 标准交叉熵损失（预测输出 grid 的 patch 序列）
- 辅助 copy task 损失（复现 study examples 的输出）
- 训练 10 万个 episode，每个有唯一的视觉解释文法
- 训练/测试的 level-2 组合不重叠（OOD 评估）

## 实验关键数据

### 主实验

系统性泛化任务（Systematicity）上的 Exact Match Accuracy：

| 模型 | 参数量 | Exact Match (%) | 备注 |
|------|--------|----------------|------|
| GPT-4o | ~数百B | 0.99 | 通用 LLM |
| Gemini 2.0 Flash | ~数百B | 2.66 | 通用 LLM |
| o3-mini (low) | ~数百B | 0.53 | 通用 LLM（推理增强） |
| Llama-3.2-3B-ReARC | 3B | 0.87 | ARC 专用 |
| Llama-3.2-3B-ReARC + TTT | 3B | 73.70 | +测试时训练 |
| Mistral-8B-Full + TTT | 8B | 78.20 | ARC Prize 2024 冠军 |
| **MLC (ours)** | **5.7M** | **78.26** | 与冠军持平！ |

### 消融实验

| 配置 | Exact Match (%) | 说明 |
|------|----------------|------|
| MLC 完整 | 86.73 ± 6.03 | 4 个 split 均值 |
| - 去掉 copy task | 69.05 ± 9.23 | 辅助任务很重要 |
| - 去掉基础变换 examples | 75.27 ± 12.95 | 适度下降 |
| - 去掉 level-1 组合 examples | 21.01 ± 19.07 | 严重崩溃 |
| MLC (更复杂数据集) | 88.10 | 变换种类增加后仍有效 |

### 关键发现
- **5.7M << 8B 但表现持平**：MLC 训练的微型模型在系统性泛化上与 ARC Prize 2024 冠军（8B + TTT + 大量工程优化）持平，参数量差 1400 倍。
- **通用 LLM 系统性泛化近乎为零**：GPT-4o (0.99%)、o3-mini (0.53%) 在 3-shot 任务上分别达 22%/64%，但在需要组合泛化的 Systematicity 任务上几乎完全失败。
- **Level-1 组合 examples 是关键**：去掉 level-1 examples 导致准确率从 87% 降至 21%，说明中间层次的组合示例对推断更高层次组合至关重要。
- **Copy task 是隐藏增益**：辅助 copy task 贡献了 ~18pp 的提升，迫使模型更深入地理解 study examples。
- **错误模式不同**：LLM 主要错误是预测错误形状或只做基础变换；MLC 主要错误是形状微小偏差（很少退化为基础/level-1 变换）。

## 亮点与洞察
- **"正确的训练范式 > 大模型"的绝佳例证**：5.7M 参数 vs 8B 参数，核心差异不在模型大小而在 MLC 训练策略。这对"大力出奇迹"的 scaling 叙事是一个有力的反例。
- **MLC 从语言到视觉的成功迁移**：证明 MLC 不是语言特定的 trick，而是一种通用的组合性泛化训练范式——通过动态变换训练文法迫使模型学习规则而非记忆。
- **研究设计精巧**：level-0/1/2 的层次化任务设计既干净又有深度，能精确地隔离和测量组合性泛化能力。

## 局限性 / 可改进方向
- **任务相对简单**：5 种基础变换、10×10 网格、2 个物体，与真实世界的空间推理相比仍很受限。
- **组合层次仅到 level-2**：更深层次的组合（3+变换组合）是否仍可泛化未测试。
- **固定网格大小**：未测试对不同大小网格的泛化能力。
- **与 ARC 原始数据集的差异**：Compositional-ARC 的变换规则比原始 ARC 更规则化，原始 ARC 的多样性和抽象性更高。

## 相关工作与启发
- **vs Lake & Baroni (2023, MLC 原文)**：原文在伪语言任务上验证 MLC，本文首次扩展到视觉空间推理，证明 MLC 的通用性。
- **vs ARC Prize 2024 冠军 (Franzen et al.)**：冠军使用 8B 模型 + 定制 tokenizer + 数据增强 + TTT + DFS 搜索等大量工程优化，MLC 模型仅 5.7M 参数+简单训练即达到同等水平。
- **vs GPT-4o / o3-mini**：暴露了当前最强 LLM 在组合性泛化上的根本缺陷——它们可以做 pattern matching 但无法做 systematic composition。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 MLC 推广到视觉空间推理，数据集设计精巧，实验结论有冲击力。
- 实验充分度: ⭐⭐⭐⭐⭐ 多个模型对比、4 个 split 验证、详细消融、错误分析、复杂度扩展实验一应俱全。
- 写作质量: ⭐⭐⭐⭐⭐ 图文并茂，逻辑清晰，概念引入循序渐进。
- 价值: ⭐⭐⭐⭐⭐ 对 AI 系统性泛化能力的理解有重要贡献，5.7M 参数超越 GPT-4o 的结果极具启发性。
