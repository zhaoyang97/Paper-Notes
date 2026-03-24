# EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements

**会议**: ICLR 2026
**arXiv**: [2506.08762](https://arxiv.org/abs/2506.08762)
**代码**: [GitHub](https://github.com/SakanaAI/EDINET-Bench)
**领域**: 时间序列
**关键词**: financial benchmark, LLM evaluation, fraud detection, earnings forecasting, Japanese NLP

## 一句话总结
构建了基于日本 EDINET 十年年报的金融基准 EDINET-Bench，包含会计欺诈检测、盈利预测和行业分类三项专家级任务，发现即使是 SOTA LLM 也仅略优于逻辑回归。

## 研究背景与动机
1. **领域现状**: LLM 在数学、编程等领域已超越人类表现，基准数据集是推动进步的关键驱动力。但金融领域基准数据集相对匮乏，现有基准（FinQA、ConvFinQA 等）多为简单 QA 或数据抽取任务。
2. **现有痛点**: 现有金融基准不涉及专家级推理（如整合多张报表和文本段落），无法评估 LLM 在真实高风险金融任务上的能力。
3. **核心矛盾**: LLM 在通用任务上表现优越，但金融领域需要同时处理大量表格数据和文本信息，并进行跨年度复杂推理。
4. **本文要解决什么**: 提供首个开源的、需要专家级推理的日语金融基准，特别是首个开放的会计欺诈检测数据集。
5. **切入角度**: 利用日本 EDINET 系统（类似美国 EDGAR）十年的真实年报数据，构建三个挑战性任务。
6. **核心idea一句话**: 真实年报 + 专家级金融任务 = 揭示 LLM 在金融推理上的不足。

## 方法详解
### 整体框架
数据管线: EDINET API → edinet2dataset 工具解析 → EDINET-Corpus（~40,000 份年报）→ 三个基准任务。

### 关键设计
1. **edinet2dataset 工具**: 使用 EDINET API 下载年报，Polars 高速解析 TSV 格式，提取 Meta/Summary/BS/PL/CF/Text 六大类信息。覆盖 2014-2025 十年约 41,691 份年报。
2. **会计欺诈检测**: 从修正年报中提取 6,712 份修正报告，用 Claude 3.7 Sonnet 判断修正原因是否涉及欺诈（668份确认为欺诈），人工审核误标率 <5%。非欺诈样本随机抽取700家，按公司分割为训练集（865）和测试集（224）。
3. **盈利预测**: 随机选1000家公司，构建连续两年年报对，比较"归母净利润"增减方向作为标签。按时间分割（2020年前为训练集），549训练 + 451测试。
4. **行业分类**: 基于 SICC 的 TOPIX-33 合并为16个大类，每类约35家公司，共496测试样本。

### 评估设置
- 零样本 prompt：系统提示"You are a financial analyst"，输入年报的不同组合（Summary only / +BS+CF+PL / +Text）
- 模型：GPT-4o, o4-mini, GPT-5, Claude 3.5 Haiku/Sonnet, Claude 3.7 Sonnet, Kimi-K2, DeepSeek-V3/R1, Llama 3.3 70B
- 经典基线：Logistic Regression, Random Forest, XGBoost

## 实验关键数据
### 主实验
欺诈检测 ROC-AUC（部分）:

| 模型 | Summary | +BS/CF/PL | +Text |
|------|---------|-----------|-------|
| Claude 3.5 Sonnet | 0.64 | 0.63 | **0.73** |
| GPT-5 | 0.56 | 0.62 | 0.67 |
| Logistic Regression† | - | 0.61 | - |

盈利预测 ROC-AUC:

| 模型 | Summary | +BS/CF/PL | +Text |
|------|---------|-----------|-------|
| GPT-5 | 0.58 | 0.62 | **0.65** |
| Claude 3.7 Sonnet | 0.55 | 0.58 | 0.61 |
| Logistic Regression† | - | 0.60 | - |

### 消融实验
输入信息量的消融:

| 输入配置 | 欺诈检测(avg) | 盈利预测(avg) |
|----------|--------------|---------------|
| Summary only | ~0.58 | ~0.48 |
| +BS/CF/PL | ~0.59 | ~0.52 |
| +Text | ~0.64 | ~0.52 |

### 关键发现
- **LLM 仅略优于逻辑回归**: 在二分类任务上，最强 LLM 的 MCC 也仅在 0.1-0.3 之间
- **文本信息有帮助**: 加入 Text 段后欺诈检测 ROC-AUC 平均提升 ~0.06
- **开源模型落后**: DeepSeek-V3/R1 在金融任务上明显弱于闭源模型
- **行业分类相对简单**: 提供完整报表后 Claude 3.5 Sonnet 达 41% 准确率（16类随机基线 6.25%）
- 每份年报约30K tokens，单次推理成本约$0.1（Claude 3.7 Sonnet）

## 亮点与洞察
- **首个开源会计欺诈检测数据集**: 此前无公开的欺诈检测评估基准
- **edinet2dataset 工具开源**: 提供了从 EDINET 构建金融数据集的完整管线，基于 Polars 高速解析 TSV
- **诚实的结论**: 直言仅提供年报让 LLM 直接推理是不够的，需要更多脚手架（如模拟环境、任务特定推理支持）
- **跨语言价值**: 日语金融基准填补了非英语金融 NLP 的空白
- **实验设计严谨**: 多种输入配置的消融，经典ML基线的对比，成本分析透明
- **标签质量控制**: 欺诈标签经 Claude 判断 + 人工审核，误标率 <5%

## 局限性 / 可改进方向
- 仅评估零样本设置，缺少 few-shot 和 RAG 实验，未探索 chain-of-thought 等推理增强
- 欺诈标签由 Claude 3.7 Sonnet 生成而非完全人工标注，可能存在系统性偏差
- 评估的 LLM 多数对日语金融术语理解有限，特别是开源模型
- 数据仅覆盖日本市场，未评估跨国泛化能力
- 缺少对 LLM 推理过程的深入分析（如关注哪些报表项目、推理路径可视化等）
- fine-tuned Llama-3.2-1B 未展示完整结果，缺少小模型微调的充分探索
- 欺诈检测和盈利预测均为二分类，未探索更细粒度的回归任务
- 年报长度约30K tokens，接近部分模型的上下文限制，可能影响结果

## 相关工作与启发
- 与 FinQA/ConvFinQA 对比：EDINET-Bench 需处理完整年报而非短段落，更接近真实金融分析场景
- 与 FinanceBench 对比：FinanceBench 为开放式 QA，EDINET-Bench 要求整合多张报表+文本进行专家级推理
- 与 FAMMA 对比：FAMMA 基于 CFA 考试和教程，EDINET-Bench 基于真实企业年报
- 与 kim2024 (GPT-4 预测盈利方向) 对比：本文提供开源数据和评估代码，可复现
- 启发：金融 LLM 需要突破简单 QA，向 agent 化（模拟金融分析师工作流）发展
- 对非英语金融 NLP 的启发：各国可用类似方法构建本地金融基准（中国 CSRC 披露、美国 EDGAR 等）
- 未来方向：结合 RAG 或 multi-agent 框架可能显著提升 LLM 的金融推理表现

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个开源欺诈检测基准，但任务本身设计较为直接
- 实验充分度: ⭐⭐⭐⭐ 覆盖10+模型和3种输入配置，但缺少 few-shot 等进阶实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据构建流程详尽，表格丰富
- 价值: ⭐⭐⭐⭐ 开源工具和数据集对金融 NLP 社区有实际贡献，揭示 LLM 金融推理的不足
