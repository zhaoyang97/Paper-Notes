# Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books

**会议**: ACL 2025  
**arXiv**: [2506.01796](https://arxiv.org/abs/2506.01796)  
**代码**: [https://github.com/Infinite-set/ZhuangRules](https://github.com/Infinite-set/ZhuangRules)  
**领域**: LLM NLP / Low-Resource Translation  
**关键词**: extremely low-resource translation, grammar book, code representation, Zhuang language, rule retrieval  

## 一句话总结

将语法书辅助的极低资源翻译（XLR MT）分解为语法规则检索和规则应用两步，并提出用代码格式表示语法规则以提升 LLM 在两步中的表现，在壮语翻译上实现了 13.1% BLEU 的提升。

## 研究背景与动机

大多数人类语言面临数据稀缺问题，传统预训练/微调方法对极低资源语言不可行。LLM 通过上下文学习（ICL）利用语言学资源（词典、平行句对）进行 XLR MT 展现了潜力。语法书作为系统性的语言学描述理论上最适合指导翻译，但其有效性存在争议：部分研究声称有效，另一部分认为 LLM 只是从语法书中提取了双语词汇解释作为捷径，而非真正理解语法规则。问题在于缺乏能排除这些干扰因素的数据集来评估 LLM 是否真正理解语法。

## 方法详解

### 整体框架

将语法书辅助翻译分解为两步：
1. **语法规则检索（Rule Retrieval）**：给定待翻译句子，从语法书中找到所需的语法规则
2. **语法规则应用（Rule Application）**：根据提供的规则完成翻译

引入 ZhuangRules 数据集支持受控实验，并提出代码格式语法规则来增强 LLM 的两步能力。

### 关键设计

- **ZhuangRules 数据集**：109 条壮语原子语法规则，每条平均配 5.6 个壮中平行句对（共 608 对）。每个测试句对附带覆盖所有相关词汇的壮中词典，将语法理解与词汇知识解耦。规则按动作、难度（easy/medium/hard，平均操作数 1.2/1.5/2.1）和语言学领域（形态学、词序等）标注。
- **Pilot Study 发现**：提供无关规则数量增加时翻译性能急剧下降，证明规则检索是关键瓶颈。仅提供所需规则 vs 提供全部规则，前者显著优于后者。
- **Rule-by-Rule 检索**：不一次提供整本语法书（Full-Book），而是逐条判断每条规则是否与待翻译句子相关（二分类），降低了长上下文理解的需求。
- **代码格式语法规则**：利用语法规则操作（如加词缀→算术运算，条件选择→if-else）与代码结构的天然相似性，用 GPT-4o 将文本规则转换为伪代码函数。每条代码规则包含：简要注释 + Python 伪代码函数。

### 损失函数 / 训练策略

本文不训练模型，全部基于 ICL。代码规则转换用 GPT-4o 5-shot ICL 完成。翻译实验中平行例句用 2-shot ICL。IGT（Interlinear Glossed Text）由 GPT-4o 生成，以 123 个手工 IGT 作为 ICL 示例，正确率约 72%。

## 实验关键数据

### 主实验

**规则检索（ZhuangRules, Table 1）：**

| 方法 | za→zh recall | zh→za recall |
|------|-------------|-------------|
| BM25 rec@5 | 41.6 | 27.3 |
| Full-Book Qwen-72B | 52.8 | 49.4 |
| Rule-by-Rule Qwen-72B (text) | 89.4 | 84.7 |
| Rule-by-Rule Qwen-72B (code) | **89.6** | **87.1** |
| Rule-by-Rule Llama-70B (text) | 69.7 | 75.8 |
| Rule-by-Rule Llama-70B (code) | **82.2** | **87.5** |

代码格式在 Llama-70B 上提升检索 recall 约 **+12.5%/+11.7%**，在 Qwen-7B 上提升更为显著。

**规则应用（ZhuangRules, Table 2, 平均 BLEU/chrF++）：**

| 设置 | 平均 BLEU/chrF++ |
|------|-----------------|
| No Rule | 25.5 / 38.0 |
| Gold Textual Rule | 45.7 / 60.7 |
| Gold Code Rule | 57.9 / 69.2 |
| Gold Textual Rule + Parallel Examples | 70.2 / 75.4 |
| Gold Code Rule + Parallel Examples | **72.4 / 77.9** |

代码规则相比文本规则在规则应用上提升 **+12.2 BLEU**（45.7→57.9），结合 parallel examples 后达到最优 72.4 BLEU。

**端到端最佳实践**：Code Rule + Rule-by-Rule 检索比 Full-Book + Textual Rule 端到端翻译提升 **13.1% BLEU**。

**跨语言验证（MTOB, Kalamang, Table 3）**：Gold Code Rule 在 kgv→eng (16.0 vs 14.6 BLEU) 和 eng→kgv (44.5 vs 43.8 BLEU) 上均优于 Gold Textual Rule，证明代码格式的跨语言泛化能力。

### 关键发现

1. **规则检索是主要瓶颈**：Full-Book 方式 recall 仅约 50%，BM25 基线更差。LLM 难以从完整语法书中定位所需规则。
2. **Rule-by-Rule 大幅优于 Full-Book**：将长上下文理解问题简化为二分类，recall 从 ~50% 提升到 ~89%。
3. **代码格式全面提升两步能力**：检索 recall +8.8%，应用 BLEU +12.2%，互相独立且可叠加。
4. **复杂规则仍是挑战**：涉及多个操作的 hard 规则性能降至 easy 规则的约一半。
5. **辅助元素有用**：平行例句和 IGT 都能进一步提升规则应用性能，但 IGT 仅支持低→高资源方向。

## 亮点与洞察

- **问题分解思想精妙**：将端到端语法书翻译拆为检索+应用，精确定位瓶颈（检索），并针对性解决。
- **代码表示语法规则**的灵感来自代码增强推理（Liu et al., 2023; Li et al., 2024），但应用于语言学领域是首次，且效果显著。
- **ZhuangRules 数据集设计考究**：通过提供词典排除词汇知识干扰，通过原子规则+难度标注支持受控分析，填补了 XLR MT 可解释性评估的空白。
- **Rule-by-Rule 策略简单但有效**：将长上下文检索转化为短上下文二分类，本质上是用更多 API 调用换取准确率。

## 局限性

- 仅在壮语和 Kalamang 两种极低资源语言上实验，泛化到其他语言有待验证。
- 代码规则的转换依赖 GPT-4o，对于更复杂或不规则的语法是否同样有效未可知。
- Rule-by-Rule 策略需要对每条规则单独查询 LLM，计算开销显著高于 Full-Book（109 次查询 vs 1 次）。
- ZhuangRules 仅覆盖 109 条规则，实际语法书通常更大更复杂。
- IGT 生成质量（72% morpheme 正确率）仍有提升空间。

## 相关工作

- **XLR MT**：Tanzer et al. (2024) MTOB benchmark+语法书；Zhang et al. (2024a) 壮语翻译；Aycock et al. (2024) 质疑语法书的真实有效性（词汇泄露）。
- **代码增强推理**：Liu et al. (2023) 用代码提升逻辑推理；Li et al. (2024) 代码表示促进 LLM 推理。
- **低资源翻译资源**：词典辅助 ICL、平行句对、IGT (Ginn et al., 2024)。
- **检索策略**：BM25 (Robertson et al., 2009)；长上下文理解挑战。

## 评分

- **新颖性**: 5/5 — 两步分解+代码格式语法规则的组合非常新颖
- **技术深度**: 4/5 — 实验设计受控、分析细致，数据集构建用心
- **实验充分性**: 4/5 — 3 个模型、2 个数据集、多种消融实验
- **实用价值**: 4/5 — 为极低资源翻译提供了可行的新范式
- **推荐指数**: ⭐⭐⭐⭐⭐
