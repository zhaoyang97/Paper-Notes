# Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking

**会议**: NeurIPS 2025  
**arXiv**: [2505.11065](https://arxiv.org/abs/2505.11065)  
**代码**: [GitHub](https://github.com/HKUSTDial/DeepFund)  
**领域**: LLM Agent / 金融 AI  
**关键词**: LLM trading, live benchmarking, multi-agent, fund investment, information leakage, financial evaluation

## 一句话总结
提出 DeepFund——首个实时基金投资 benchmark 工具，通过多智能体架构（Financial Planner + Analyst Team + Portfolio Manager）连接实时股市数据，避免传统回测中 LLM "时间旅行"导致的信息泄露问题。在 24 个交易日的实盘测试中，9 个旗舰 LLM 只有 Grok 3 实现盈利，揭示了当前 LLM 在主动基金管理中的重大局限。

## 研究背景与动机

1. **领域现状**：LLM 在金融任务（报告摘要、收益电话分析、资产分类）上展现了出色能力。研究者越来越关注将 LLM 用于交易策略生成和基金投资管理。现有 benchmark（如 TAT-QA、FinanceBench、InvestorBench）评估 LLM 的金融理解和交易表现。
2. **现有痛点**：现有 benchmark 依赖历史回测（back-testing）评估 LLM 交易策略，但 LLM 的预训练数据很可能包含了回测期间的历史市场数据。这导致严重的**信息泄露**——LLM 可以"时间旅行"，利用未来信息在历史数据上表现虚高。不同 LLM 的知识截止日期不同（如 GPT-4o 截至 2023.10，DeepSeek-V3 截至 2024.7），加剧了不公平性。
3. **核心矛盾**：在回测设置下，我们无法区分 LLM 是真的"预测"了市场走势，还是只是"记忆"了历史数据。这让回测结果完全不可信。
4. **本文要解决什么**：构建一个完全基于实时（live）市场数据的 LLM 基金投资评估框架，确保零信息泄露。
5. **切入角度**：使用每个模型预训练截止日期之后的实时数据进行前向测试（forward testing），而非回测。
6. **核心 idea 一句话**：将 LLM 基金投资评估从"历史回测"转变为"实盘前向测试"，彻底消除时间旅行作弊。

## 方法详解

### 整体框架
DeepFund 系统由三部分组成：(1) **Live Environment** 持续接入实时市场数据、基金资产信息和交易历史；(2) **Multi-Agent Workflow** 模拟真实基金管理流程——Financial Planner 分配任务 → Analyst Team 并行分析 → Portfolio Manager 做决策；(3) **LLM Factory** 支持灵活切换不同 LLM 后端。所有 agent 由同一个 LLM 驱动，确保一致性。

### 关键设计

1. **Live Environment（实时环境）**：
   - 做什么：提供无泄露的实时市场条件
   - 核心思路：通过模块化 API 网关接入 Yahoo Finance、Alpha Vantage 等数据源，持续获取实时股价、公司新闻、宏观经济指标、内部人交易等多源数据。所有数据均为模型预训练截止日期之后发布
   - 设计动机：从根本上消除信息泄露——使用模型不可能见过的"未来"数据

2. **Multi-Agent Decision Framework（多智能体决策）**：
   - 做什么：模拟真实基金管理的协作决策流程
   - 核心思路：  
     - **Financial Planner**：确定分析优先级，将任务分配给合适的分析师（支持确定性和动态两种模式）
     - **Analyst Team**：6 类专业分析师（Technical、Fundamental、Insider、Company News、Macro Economic、Policy），各自分析专业领域数据，输出标准化信号（Bullish/Bearish/Neutral）+ 详细理由
     - **Portfolio Manager**：综合多个分析信号做出交易决策（Buy/Sell/Hold），管理风险（持仓比例和现金），维护双记忆架构（历史交易 + 当前组合状态）
   - 设计动机：单一 agent 无法处理多源异构金融数据；分工协作更接近真实基金管理团队的工作方式

3. **评估体系**：
   - 做什么：多维度量化 LLM 交易表现
   - 核心思路：采用累积收益（CR）、Buy&Hold 累积收益（CR_bnh）、Sharpe Ratio、最大回撤（MDD）、胜率（WR）、Beta、Alpha 等标准金融指标
   - 设计动机：对标专业基金评估标准，而非仅看收益率

### 训练策略
不涉及模型训练。DeepFund 是一个评估框架，使用各 LLM 的标准 API 进行推理。

## 实验关键数据

### 整体交易表现（2025.3.17-4.17，24 交易日）
| LLM | CR(%) | SR | MDD(%) | WR(%) | Beta |
|-----|-------|-----|--------|-------|------|
| Grok 3 mini Beta | **+1.1** | 0.51 | 5.5 | 61 | 0.42 |
| Gemini 2.5 Flash | -1.9 | -1.37 | 6.4 | 61 | 0.35 |
| Claude 3.7 Sonnet | -3.7 | -1.45 | 10.1 | 70 | 0.64 |
| Llama 4 Scout | -4.3 | -2.42 | 8.9 | 61 | 0.36 |
| DeepSeek-V3 | -5.7 | -1.39 | 14.5 | 57 | 0.94 |
| GPT-4.1 | -5.9 | -1.87 | 12.8 | 52 | 0.77 |
| Qwen2.5-Max | -6.7 | -3.12 | 10.7 | 65 | 0.48 |
| GLM-4-Air | -7.5 | -2.31 | 13.2 | 57 | 0.78 |
| Doubao-1.5-pro | -8.1 | -2.35 | 13.6 | 65 | 0.84 |
| S&P 500 | -6.91 | — | 13.7 | — | 1.00 |

### 信号和决策有效性
| 指标 | 总数 | 有效数 | 有效率 |
|------|------|--------|-------|
| 分析师信号 | 4320 | 4144 | 96% |
| 交易决策 | 1080 | 1059 | 98% |

### Grok vs DeepSeek 对比分析
| 维度 | Grok 3 | DeepSeek-V3 |
|------|--------|-----------|
| 初始现金分配 | 保守（60%现金储备）| 激进（90%立即投入）|
| 交易频率 | 低频长期持有 | 高频动量驱动 |
| 行业分散度 | 好（能源+消费品）| 差（集中能源+金融）|
| 最大回撤 | 5.5% | 14.5% |
| Buy 有效率 | 7/11 (64%) | 1/3 (33%) |
| 应对关税冲击 | 高现金储备缓冲，下跌后抄底 | 低现金无法止损 |

### 关键发现
- **9 个旗舰 LLM 中只有 1 个盈利**：Grok 3 以 +1.1% 的微薄收益成为唯一赢家，大多数模型跑输 Buy&Hold 策略
- **中国 LLM 整体表现不如美国 LLM**：在关税冲击期间，中国 LLM（Qwen、GLM、Doubao）的亏损更大
- **风控能力是关键分水岭**：Grok 的成功不在于选股能力（信号质量与 DeepSeek 接近），而在于保守的现金管理和行业分散
- **所有模型都未能预测 4.9 的强反弹**（AAPL 从 172 → 198 USD），暴露了 LLM 在极端事件预测上的共同短板

## 亮点与洞察
- **"时间旅行是作弊"**：论文标题本身就是对当前 LLM 金融评估范式的尖锐批评。Live benchmark 的理念可推广到任何涉及时序数据的 LLM 评估
- **LLM 的"交易人格"分析**：Grok 像谨慎的基金经理（低频、分散、高现金），DeepSeek 像散户投机者（高频、集中、all-in）。这种拟人化分析对理解 LLM 决策风格很有启发
- **多智能体协作的实用范式**：Financial Planner → Analyst Team → Portfolio Manager 的分工模式是 LLM Agent 在金融场景的一个良好示范
- **信号质量 ≠ 盈利能力**：96% 的信号有效率和 70% 的胜率（Claude）并不能保证盈利，说明风控和仓位管理比信号准确率更重要

## 局限性 / 可改进方向
- **评估期过短**：仅 24 个交易日，且恰好经历极端波动（FOMC + 关税战），结论可能偏向特定市场环境
- **仅覆盖美股 5 只股票**：Berkshire Hathaway 重仓股均为大盘蓝筹，对中小盘、成长股等风格适用性未验证
- **未计入交易成本**：佣金、滑点、市场冲击等实际摩擦未考虑，可能进一步恶化亏损
- **单一 LLM 驱动所有 agent**：真实基金可能让不同 LLM 担任不同角色（如用擅长定量的 LLM 做技术分析）
- **改进方向**：(1) 延长评估至 6-12 个月覆盖牛熊震荡多种市况；(2) 扩展到全球市场和更多资产类别；(3) 引入交易成本和市场微结构约束；(4) 探索混合 LLM 团队

## 相关工作与启发
- **vs InvestorBench**：InvestorBench 用历史回测评估 LLM 交易，存在信息泄露问题；DeepFund 用实盘前向测试彻底解决
- **vs FinRL-Meta**：FinRL-Meta 聚焦强化学习策略的 benchmark，非 LLM-specific；DeepFund 专为 LLM Agent 设计
- **vs LiveBench/LiveCodeBench**：思路类似——都用持续更新的数据消除数据污染；DeepFund 将这一理念引入金融投资领域
- **启发**：当前 LLM 在需要真正预测未来的金融场景中表现远不如营销所暗示的那样好。"看起来能做"和"真正能做"之间的鸿沟值得警醒

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 LLM 实盘基金投资 benchmark，"时间旅行"问题的提出切中要害
- 实验充分度: ⭐⭐⭐⭐ 9 个旗舰 LLM + 多维度金融指标 + 深入的 Grok/DeepSeek 案例对比
- 写作质量: ⭐⭐⭐⭐⭐ 叙事生动（Q1-Q4 结构清晰），"交易人格"分析极具可读性
- 价值: ⭐⭐⭐⭐⭐ 对金融 AI 社区有震撼效果——大多数 LLM 在实盘上亏钱，戳破了回测泡沫
