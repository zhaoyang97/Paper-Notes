# MaintainCoder: Maintainable Code Generation Under Dynamic Requirements

**会议**: NeurIPS 2025  
**arXiv**: [2503.24260](https://arxiv.org/abs/2503.24260)  
**代码**: [https://github.com/IAAR-Shanghai/MaintainCoder](https://github.com/IAAR-Shanghai/MaintainCoder)  
**领域**: LLM Agent / 代码生成 / 软件工程  
**关键词**: maintainable code, design patterns, waterfall model, multi-agent, dynamic requirements, MaintainBench  

## 一句话总结
首次系统定义并解决 LLM 代码生成的**可维护性**问题，同时贡献基准和方法：MaintainBench 通过 4 种需求变化模式 + 动态指标评测代码在需求演化下的可维护性；MaintainCoder 将 Waterfall 模型、设计模式与 6 个专业化 Agent 结合，动态可维护性指标提升 60%+，且初始代码正确性也一并提高。

## 背景与动机

LLM 代码生成在功能正确性上进步显著（HumanEval、MBPP、SWE-Bench），但忽略了实际软件开发最关键的维度：**可维护性**——代码在需求演化时能否以最小修改量适应变化。赫拉克利特说"唯一不变的是变化本身"，软件工程中尤其如此。

**可维护性危机的现实案例**：
- **Knight Capital**：未维护的遗留代码在几分钟内蒸发 4.4 亿美元
- **千年虫问题（Y2K）**：短视的设计决策导致全球 1000 亿美元修复成本
- 行业研究一致表明 **60–80% 的软件生命周期成本**来自部署后的维护，根源在于高耦合、低内聚的结构缺陷

**三个根本性空白**：
1. 没有基准在需求演化周期中量化可维护性
2. 没有方法系统地将设计模式等软件工程原则应用于 LLM 代码生成
3. 传统静态指标（圈复杂度 CC、可维护性指数 MI）仅分析代码结构，无法捕捉动态维护场景

**已有证据支撑方法可行性**：Dong et al. 证实 Waterfall 模型可将代码正确性提高 29.9–47.1%；Hegedűs et al. 发现设计模式与软件可维护性的 Pearson 相关系数高达 0.89。

## 核心问题
如何让 LLM 生成的代码 $C_0 = \mathcal{G}(P_0)$ 在面对动态需求序列 $\{P_0, P_1, \ldots, P_n\}$ 时，以最小的累计维护代价 $\mathcal{M}(C_0) = \mathbb{E}[\sum_{i=0}^{n-1} \gamma^i \mathcal{M}(C_i \to C_{i+1})]$ 完成适应性修改？

## 方法详解

### 一、MaintainBench：首个动态可维护性基准

**数据构建**——扩展 5 个经典基准，按难度分三级（共 500+ Python 样本）：
- **入门级**：HumanEval-Dyn + MBPP-Dyn（各 30 题 → 120 题）
- **混合级**：APPS-Dyn（50 题 → 200+ 题，含入门到竞赛多种难度）
- **竞赛级**：CodeContests-Dyn + xCodeEval-Dyn（各 30 题 → 120+ 题）

**四种需求变化模式**（对应 ISO/IEC/IEEE 14764:2022 四类软件维护）：
1. **功能扩展 $\pi_{ext}$**：新增功能需求，要求调用原有函数（适应性+完善性维护）
2. **接口修改 $\pi_{int}$**：修改输入参数、返回类型等 API 契约，如外部依赖库升级（适应性维护）
3. **数据结构转换 $\pi_{dst}$**：更换数据表示以适应效率/可扩展性需求（完善性维护）
4. **错误处理增强 $\pi_{err}$**：引入 ZeroDivisionError、IndexError 等异常处理（纠正性+预防性维护）

**构建流程**：原始问题 $P_0$ → GPT-4o 按变化模式生成变体 $(P_1, S_1', T_1')$ → Python 解释器自动执行测试 → 失败则人工专家诊断并修正 → 迭代直到所有测试通过 → 多阶段专家审查确保质量。

**动态指标设计**（通过蒙特卡洛估计 + 一阶截断近似动态可维护性）：
- **Pass@k**：需求修改后的功能正确率（$k$ 最高到 5）
- **$\text{Code}_{diff}^{per}$**：修改行数占原代码的百分比（相对改动量）
- **$\text{Code}_{diff}^{abs}$**：修改的绝对行数
- **$\text{AST}_{sim}$**：修改前后抽象语法树的结构相似度

### 二、MaintainCoder：多 Agent 可维护代码生成系统

整体架构模仿人类软件开发生命周期（Waterfall 模型），基于 AutoGen 框架实现 Agent 间通信，分为两大模块共 6 个专业化 Agent：

**代码框架模块（Code Framework Module）**——将需求转化为可维护的架构蓝图：
1. **需求分析 Agent**：CoT 逐步分解问题，提取核心功能、识别关键挑战、提出高层解决方案，避免不必要的复杂性
2. **设计模式选择 Agent**：充当软件架构师，为每个功能模块选择最合适的设计模式（策略/工厂/观察者等），优先促进模块化、降低耦合、增强可扩展性和复用性，并给出备选方案
3. **架构设计 Agent**：基于设计模式构建类结构，遵循单一职责原则，明确类间依赖关系；根据评审反馈迭代修改
4. **架构评审 Agent**：审查设计清晰性、可扩展性、性能和最佳实践，识别耦合/内聚/复用问题，**防止过度细分（over-fragmentation）**，给出可操作的改进建议

**代码生成模块（Code Generation Module）**——将蓝图实现为可执行代码：
5. **代码生成 Agent**：将架构设计转化为 PEP 8/PEP 257 规范代码，注释聚焦设计意图和非显而易见的逻辑而非冗余描述；插入测试样本用于迭代调试
6. **代码优化 Agent**：执行代码 → 收集语法错误/测试失败/异常行为 → CoT 诊断根因 → 修复并重测 → 迭代至全部通过

## 实验关键数据

实验分两阶段：Phase I 用 MaintainCoder 或基线生成初始代码 $C_0$；Phase II 用固定生成器（如 GPT-4o-mini）对 $C_0$ 进行需求修改，评估动态可维护性。

### 混合级 APPS-Dyn 主实验
| 方法 | MI↑ | CC↓ | Pass@5↑ | AST_sim↑ | Code_diff^per↓ |
|------|-----|-----|---------|----------|---------------|
| GPT-4o-mini | 63.3 | 5.10 | 35.5% | 0.589 | 140% |
| AgentCoder (4o-mini) | 63.3 | 5.81 | 21.0% | 0.510 | 66.3% |
| MapCoder (4o-mini) | 67.8 | 5.98 | 30.5% | 0.583 | 73.8% |
| **MaintainCoder (4o-mini)** | **69.5** | **2.75** | **50.5%** | **0.797** | **29.4%** |
| DeepSeek-V3 | 61.8 | 7.59 | 59.5% | 0.598 | 131% |
| **MaintainCoder (DS-V3)** | **62.4** | **3.21** | **62.5%** | **0.828** | **29.2%** |
| GPT-4o | 63.0 | 4.58 | 39.5% | 0.556 | 140% |
| Claude-3.7-Sonnet | 59.3 | 6.65 | 48.5% | 0.620 | 85.2% |
| Gemini-2.5-Flash | 59.7 | 9.00 | 51.0% | 0.631 | 108% |

核心结论：MaintainCoder 的 CC 约为基线的 **1/2 到 1/3**，AST 相似度高出 **28%+**，代码改动量仅为基线的 **1/5**。

### 竞赛级数据集
- **CodeContests-Dyn**：MaintainCoder(4o-mini) Pass@5 32.6%、CC 2.68、AST_sim 0.833、Code_diff^per 23.2%——全面最优
- **xCodeEval-Dyn**：MaintainCoder(DS-V3) Pass@5 **36.7%**（超越第二名 GPT-4o 的 32.8% 达 60% 相对提升），AST_sim 0.785，Code_diff^per 33.0%
- 竞赛级问题上基线多 Agent 系统（AgentCoder/MapCoder）的 CC 膨胀至 15–20，MaintainCoder 仍然维持在 3 左右

### 正确性与可维护性的双赢
| 方法 | APPS | CodeContests | xCodeEval |
|------|------|-------------|-----------|
| GPT-4o-mini | 44% | 18% | 46% |
| **MaintainCoder (4o-mini)** | **48%** | **23%** | **57%** |
| DeepSeek-V3 | 66% | 48% | 75% |
| **MaintainCoder (DS-V3)** | **69%** | **51%** | **77%** |

问题越复杂提升越大：xCodeEval 上 GPT-4o-mini 从 46%→57%（+11%），APPS 仅 +4%。即使在已表现强劲的 DeepSeek-V3 上仍有提升。

### 人类基线与推理模型对比（CodeContests-Dyn）
| 方法 | CC↓ | Pass@5↑ | AST_sim↑ | Code_diff^per↓ |
|------|-----|---------|----------|---------------|
| 人类程序员 (CF 1700-2300) | 8.17 | 23.5% | 0.541 | 112.3% |
| o3-mini | 11.3 | 30.3% | 0.661 | 101.6% |
| EvoMAC (4o-mini) | 5.18 | 26.5% | 0.685 | 60.1% |
| MetaGPT (4.1-mini) | 7.63 | 30.3% | 0.760 | 44.3% |
| **MaintainCoder (4o-mini)** | **2.68** | **32.6%** | **0.833** | **23.2%** |
| **MaintainCoder (o3-mini)** | **3.85** | **36.4%** | **0.794** | **27.8%** |

有竞赛经验的人类程序员（Codeforces 1700-2300）在时间压力下生成的代码，可维护性指标反而不如 AI。MaintainCoder 在强推理模型 o3-mini 上依然带来显著提升。

### 静态指标 vs 动态指标
- AgentCoder/MapCoder 的 MI 显著高于基线，但 Pass@k 反而**降低**——静态指标给出误导性结论
- MI 和 CC 变化方向常常**矛盾**（如 MapCoder 的 MI=66.1 和 CC=7.32 相比 GPT-4o-mini 的 MI=57.8 和 CC=6.06，两项指标同时上升）
- 动态指标（Pass@k、AST_sim、Code_diff）之间**高度一致**——证明动态指标更准确反映真实可维护性

### 消融实验
| 消融配置 | APPS-Dyn | xCodeEval-Dyn |
|---------|----------|--------------|
| 完整 MaintainCoder (4o-mini) | 50.5% | 27.3% |
| 去掉架构评审 | 49.0% (−3.0%) | 20.3% (−25.7%) |
| 去掉代码优化 | 40.0% (−20.8%) | 17.2% (−37.1%) |

代码优化 Agent 贡献更大，但架构评审在高难度任务上也至关重要（xCodeEval 下降 25.7%）。架构评审通常只需 1 轮迭代，是轻量但高效的组件。

### 计算成本
| 数据集 | MaintainCoder | MapCoder | o3-mini | MetaGPT/ChatDev | GPT-4o-mini |
|--------|--------------|----------|---------|-----------------|-------------|
| CodeContests | 33.1k | 38.7k | 20.8k | 50k+ | 2.5k |
| xCodeEval | 29.6k | 23.5k | 21.2k | 50k+ | 2.3k |

MaintainCoder 的 token 消耗与 MapCoder 相当，远低于 MetaGPT/ChatDev，初始投入换来后续维护成本的大幅降低。

## 亮点
- **问题定义的范式转移**：首次将"可维护性"从软件工程视角引入 LLM 代码生成，用动态需求演化代替静态测试用例
- **MaintainBench 填补评测空白**：4 种需求变化模式 × 3 难度级别 × 500+ 样本 + 动态指标体系
- **反直觉发现**：多 Agent 系统（AgentCoder/MapCoder）优化单轮正确性反而**损害**长期可维护性；CoT/Self-Planning 的影响不稳定甚至随机
- **双赢而非权衡**：良好的架构设计不仅提升可维护性，也通过结构清晰性促进初始代码正确性
- **静态指标被证伪**：MI 和 CC 互相矛盾且与真实维护成本脱节
- **超越人类程序员**：竞赛选手在时间压力下的代码可维护性甚至不如 AI 生成

## 局限性
- 多 Agent 管线增加初始 API 调用成本（~30k tokens/问题），不适合实时代码辅助
- 仅在 Python 上评估，其他语言和大规模仓库级项目（如 SWE-Bench 规模）待验证
- 设计模式选择依赖 LLM 内在知识，复杂架构决策可能出错
- 仅模拟一轮需求变化（$P_0 \to P_1$），多轮连续演化评测有待探索
- MaintainBench 的需求变化由 GPT-4o 生成，可能存在分布偏差

## 相关工作对比
- **vs AgentCoder/MapCoder/EvoMAC**：这些系统优化单轮正确性但忽视结构质量，竞赛级问题上 CC 膨胀至 15–20
- **vs SWE-Bench**：SWE-Bench 评测的是解决 GitHub issue 的能力，不评估代码本身是否可维护
- **vs MetaGPT/ChatDev**：角色分工类多 Agent 但不强调设计模式和 Waterfall 流程，token 消耗也更高（50k+）
- **vs CoT/Self-Planning**：Prompt 工程方法对可维护性影响不稳定，MaintainCoder 的结构化流程显著更鲁棒

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义并解决代码可维护性生成问题，基准+方法双贡献，问题意识非常前沿
- 实验充分度: ⭐⭐⭐⭐⭐ 5 基准 × 10+ 基线（含人类/推理模型/多 Agent）+ 消融 + 静态vs动态指标 + 成本分析
- 写作质量: ⭐⭐⭐⭐ 问题动机以 Knight Capital/千年虫开场有冲击力，形式化定义完整，方法论系统
- 价值: ⭐⭐⭐⭐⭐ 范式转移——从"生成正确代码"到"生成可维护代码"，MaintainBench 有潜力成为标准评测
