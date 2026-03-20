# FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging

**会议**: ACL 2025  
**arXiv**: [2506.05828](https://arxiv.org/abs/2506.05828)  
**代码**: [https://github.com/BUPT-Reasoning-Lab/FinanceReasoning](https://github.com/BUPT-Reasoning-Lab/FinanceReasoning)  
**领域**: 金融推理 / Benchmark  
**关键词**: 金融数值推理, Benchmark, 大推理模型, 知识增强, 函数库

## 一句话总结

提出 FinanceReasoning benchmark，通过重标注公开数据集、构建 3,133 个 Python 金融函数库和新增 908 道专家标注难题，在可信度、全面性和挑战性三个维度提升金融数值推理评估能力。

## 研究背景与动机

1. **领域现状**: LRM（大推理模型）如 OpenAI o1、DeepSeek-R1 在通用推理任务上取得突破，但在金融等领域专用数值推理任务上仍面临挑战。
2. **现有痛点**: 现有金融推理 benchmark（如 CodeFinQA、FinanceMath）存在三大问题——(1) 标注质量差（9.72%-30% 的题目有错误或歧义）；(2) 评估标准宽松（允许 1% 误差、忽略单位和符号）；(3) 简单题过多，LRM 已饱和（>90% 准确率），难以客观评估推理能力。
3. **核心矛盾**: 现有 benchmark 无法如实反映 LRM 的真实推理改进——例如 DeepSeek-R1 在原始 CodeFinQA 上仅比 V3 低 0.88%，但在重标注版本上高 2.01%。
4. **本文要解决什么**: 构建一个可信、全面、有挑战性的金融数值推理 benchmark。
5. **切入角度**: 三管齐下——修正已有数据、构建金融函数知识库、专家标注高难度新题。
6. **核心idea一句话**: 通过"修旧题+造知识库+出难题"三步策略，打造高质量金融数值推理评估基准。

## 方法详解

### 整体框架

(1) 重标注四个公开测试集（CodeFinQA/CodeTAT-QA/FinCode/FinanceMath），修正 15.6% 的题目；(2) 从 Investopedia 提取构建 3,133 个 Python 金融函数；(3) 利用函数库引导 GPT-4o 生成 + 专家验证 908 道新题。

### 关键设计

1. **数据重标注（Updates to Public Datasets）**: 对每道题执行三类操作——消歧（Disambiguation，修复不可解或模糊题目）、详述（Elaboration，补充缺失计算步骤）、纠错（Correction，修正错误答案）。严格执行 0.2% 误差限制，指定单位/百分比/小数位数。
2. **金融函数库构建**: 从 Investopedia 收集 6,138 篇文章，用 GPT-4o 提取金融计算函数，包含语义化签名、详细 docstring（功能/参数/返回值/约束）和逐步实现代码。由 CFA 持证专家审核修正，共 3,133 个函数，覆盖 1,864 个金融概念。
3. **难度分级算法**: 首次提出基于算子数量(o)、括号对数(p)和代码行数(l)的启发式难度评估：rc = ln(max(o,1)) + ln(max(l+p,1))，将题目分为 Easy(1000)/Medium(1000)/Hard(238)。

### 损失函数 / 训练策略

本文为评估 benchmark，无训练过程。评估采用 CoT（Chain-of-Thought）和 PoT（Program-of-Thought）两种提示方法。探索了知识增强策略和 Reasoner+Programmer 模型组合方案。

## 实验关键数据

### 主实验

各模型在 FinanceReasoning 上的表现（Accuracy %）：

| 模型 | Hard(CoT) | Hard(PoT) | Medium(CoT) | Medium(PoT) | Easy(CoT) | Easy(PoT) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI o1 | 81.1 | **89.1** | 89.7 | — | 88.0 | — |
| DeepSeek-R1 | 83.2 | 85.3 | 91.1 | 89.8 | 89.8 | 89.2 |
| OpenAI o3-mini | 77.3 | 84.0 | 87.8 | 88.6 | 88.8 | 88.1 |
| GPT-4o | 65.6 | 83.6 | 84.6 | 87.9 | 86.8 | 88.1 |
| Claude 3.5 Sonnet | 68.5 | 83.6 | 85.7 | 88.2 | 87.7 | 88.4 |
| Llama 3.3-70B | 50.4 | 71.4 | 79.2 | 85.9 | 83.3 | 84.8 |

### 消融实验

重标注前后的对比（揭示 LRM 真实改进）：

| 数据集 | 评估标准 | DeepSeek-V3 | DeepSeek-R1 | R1相对提升 |
|--------|----------|:---:|:---:|:---:|
| CodeFinQA | Silver（原始） | 61.76 | 60.88 | -1.42% |
| CodeFinQA | **Gold（重标注）** | 85.41 | 87.42 | **+2.35%** |
| FinanceMath | Silver（原始） | 58.50 | 71.00 | +21.37% |
| FinanceMath | **Gold（重标注）** | 59.50 | 83.50 | **+40.34%** |

### 关键发现

- PoT 显著优于 CoT，尤其在 Hard 题目上（OpenAI o1: 81.1% → 89.1%），程序化推理对数值精度至关重要
- 知识增强有效：GPT-4o + 金融函数库从 83.2% 提升至 91.6%（+8.4%）
- Reasoner + Programmer 组合策略有效：DeepSeek-R1 + Programmer 从 83.2% → 87.8%（+4.6%）
- LRM 仍面临公式选择错误和数值精度不足的挑战
- 原始数据集中 9.72%-30% 的题目存在质量问题，严重低估了 LRM 的真实推理改进

## 亮点与洞察

- 通过重标注暴露了现有 benchmark 的严重标注质量问题，方法论贡献大
- 金融函数库（3,133 个函数）不仅用于生成题目，也可作为知识增强资源提升模型能力，一举两得
- 难度分级算法简单有效，为未来构建不同难度的领域 benchmark 提供参考

## 局限性 / 可改进方向

- Hard 题目仅 238 道，规模较小，可能不够代表性
- 难度分级基于代码统计特征，未考虑金融概念本身的理解难度
- 仅覆盖英语金融推理，中文等金融场景未涉及
- 函数库取自 Investopedia，可能存在覆盖偏差（偏向美国金融体系）

## 相关工作与启发

- 与 BizBench 的关系：FinanceReasoning 在其基础上重标注并扩展，形成更严格的评估标准
- 与通用推理 benchmark（GSM8K、MATH）的区别：强调领域知识应用和数值精度
- 启发：Reasoner + Programmer 的双模型协作模式值得在更多领域推广（如物理、工程计算）

## 补充分析

- Easy/Medium/Hard 的平均算子数分别为 1.77/3.79/10.12，代码行数为 3.13/4.27/9.49，难度梯度设计合理
- 函数库平均每个函数有 2.85 个算子、2.64 个参数，覆盖 1,864 个金融概念
- 8 名跨学科研究生 + 2 名 CFA 持证专家的标注团队规格较高，整个标注历时三个月
- 数据集全部公开，连同完整金融函数库一起开源，对社区贡献显著
- 难度分级对比图显示 FinanceReasoning 的中等和困难题目占比远高于现有数据集

## 评分

- 新颖性: ⭐⭐⭐⭐ 重标注+函数库+难题的三重策略有新意，但整体是benchmark工程
- 实验充分度: ⭐⭐⭐⭐⭐ 13个模型×多种策略，重标注对比分析很有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 高质量金融推理benchmark + 开源函数库，对领域评估有重要贡献
