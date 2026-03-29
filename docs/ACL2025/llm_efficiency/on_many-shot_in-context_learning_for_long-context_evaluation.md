# On Many-Shot In-Context Learning for Long-Context Evaluation

**会议**: ACL 2025
**arXiv**: [2411.07130](https://arxiv.org/abs/2411.07130)
**代码**: https://github.com/launchnlp/ManyICLBench
**领域**: LLM Efficiency / 长上下文评估
**关键词**: Many-shot ICL, Long-context Evaluation, ManyICLBench, Similar-Sample Learning, All-Sample Learning

## 一句话总结
深入研究 many-shot ICL 用于长上下文语言模型评估，提出 Sample Learning Ratio 指标区分 SSL 和 ASL 任务，构建 ManyICLBench 基准全面评测 12 个 LCLM。

## 研究背景与动机
1. **领域现状**：长上下文语言模型（LCLM）支持 128K 甚至 1M token 上下文，but 现有评估主要测检索能力。
2. **现有痛点**：Needle-in-a-Haystack 等合成任务只测检索，缺乏全局上下文理解的评测。
3. **核心矛盾**：LongICLBench 等已有 many-shot ICL 基准主要用分类任务，但不清楚这些任务到底在测什么能力。
4. **本文要解决什么？** (1) 哪些任务从更多示例中获益？(2) 各任务在多大程度上依赖相似样本检索 vs 全样本学习？
5. **切入角度**：提出 Sample Learning Ratio (SLR) 指标，量化 ICL 任务对检索 vs 理解的依赖程度。
6. **核心idea一句话**：many-shot ICL 分类任务实质是检索相似示例，真正的全局理解需要 ASL 任务来评测。

## 方法详解

### 整体框架
(1) 收集 12 个 ICL 数据集(21 个子任务) → (2) 用 1k~128k 上下文测试 12 个 LCLM → (3) 提出 SLR 指标分析每个任务的技能需求 → (4) 构建 ManyICLBench 基准。

### 关键设计
1. **Sample Learning Ratio (SLR)**:
   - 做什么：量化任务对相似样本检索的依赖程度
   - 核心思路：分别移除 10% 最相似和最不相似示例，比较性能变化比率
   - 设计动机：SLR >> 1 表示强依赖检索，SLR ≈ 1 表示需要全样本学习

2. **任务分类（SSL vs ASL）**:
   - 做什么：将 ICL 任务分为相似样本学习(SSL)和全样本学习(ASL)
   - 核心思路：SSL 任务(分类)主要靠检索相似示例；ASL 任务(数学/摘要)需要理解所有示例
   - 设计动机：区分两类技能，提供更全面的评测

3. **ManyICLBench 构建**:
   - 做什么：curate 一组 many-shot ICL 基准
   - 核心思路：同时包含 SSL 和 ASL 任务，覆盖 1k 到 128k token
   - 设计动机：单一维度评测不足以反映 LCLM 真实能力

### 损失函数 / 训练策略
纯评估工作，无训练。使用 greedy decoding，每个实验三种随机种子。

## 实验关键数据

### 主实验（SSL 任务，Macro F1 @ 不同 token 数）

| 模型 | 1k | 8k | 32k | 64k | 128k |
|------|-----|-----|------|------|-------|
| Qwen2-72B | 36.4 | 65.3 | 76.5 | 77.5 | 77.5 |
| Llama-3.1-70B | 38.8 | 66.1 | 76.6 | 78.5 | 65.6 |
| Gemini-1.5-Pro | 45.7 | 74.7 | 80.2 | 84.1 | 84.5 |
| GLM-4-9b | 31.6 | 57.3 | 68.3 | 72.2 | 72.9 |
| Phi-3-Mini | 30.3 | 48.1 | 57.3 | 56.8 | 48.7 |

### 任务类型与 Many-shot ICL 效果

| 任务类型 | 与上下文长度相关性 | 趋势 |
|---------|------------------|------|
| 分类 | 高正相关 | 持续改善 |
| 摘要 | 中等正相关 | 收益递减 |
| 翻译 | 无明显趋势 | 不一致 |
| 数学推理 | 有条件获益 | 需 CoT + 强模型 |
| 科学/符号推理 | 不一致 | 取决于任务特性 |

### 消融实验

| 分析 | 发现 |
|------|------|
| SSL SLR | 分类任务 SLR 远 > 1，证实依赖检索 |
| ASL SLR | 数学/摘要 SLR ≈ 1，不依赖相似检索 |
| BM25 vs SentenceTransformer | 两种检索器结论一致 |

### 关键发现
- 分类任务在 SSL 中表现优异但 ASL 中差距巨大
- SOTA 模型在 SSL 64k 可达优秀，但 ASL 16k 就开始性能下降
- 小模型（如 Phi-3-Mini）在长上下文场景严重退化

## 亮点与洞察
- SLR 指标简洁有效，一句话就能解释清楚
- 将检索 vs 理解的二分法引入 ICL 评测，框架设计优雅
- 发现 many-shot ICL 分类≈检索这一 insight 对社区很有价值

## 局限性 / 可改进方向
- SLR 基于 BM25 的相似度可能遗漏语义层面的相似性
- 仅测试了公开模型，缺少最新 GPT-4o/Claude-3.5 在 ASL 上的表现
- 未探讨示例排序对 SSL vs ASL 的影响

## 相关工作与启发
- **vs LongICLBench (Li et al. 2024)**: LongICLBench 仅用分类，本文证明分类主要测检索而非全局理解
- **vs Agarwal et al. (2024)**: 他们只测 Gemini 1.5 Pro，本文覆盖 12 个模型


## 补充细节
- 12 个模型包括 Llama-3.1、Qwen2、Phi-3、Mistral、GLM-4、Jamba、Gemini-1.5-Pro
- 上下文长度从 1k 到 128k，每次新增示例扩展上下文
- 使用 greedy decoding，三种随机种子平均
- 数学任务需要 CoT 才能从更多示例中获益

## 评分
- 新颖性: ⭐⭐⭐⭐ SLR 指标和 SSL/ASL 分类思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 12 模型 x 21 任务 x 多种上下文长度
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表信息量大
- 价值: ⭐⭐⭐⭐ 对长上下文评测社区有实际指导意义
