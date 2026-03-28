# Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis

**会议**: AAAI 2026  
**arXiv**: [2511.04481](https://arxiv.org/abs/2511.04481)  
**代码**: [GitHub](https://github.com/DFKIEI/WebAgentEnergy)  
**领域**: LLM Agent / 可持续AI  
**关键词**: Web Agent、能耗基准测试、碳排放估算、绿色AI、可持续部署

## 一句话总结

首次系统性地从实证基准测试和理论估算两个角度量化了 Web Agent 的能耗与碳排放，发现更高能耗并不等于更好性能，并倡导在评测中引入能效指标。

## 研究背景与动机

1. **领域现状**：Web Agent（如 OpenAI Operator、Google Project Mariner）正在快速发展，能够自主浏览网页、填写表单、比较价格，代表了 LLM 应用的重要前沿方向。
2. **现有痛点**：当前 Web Agent 研究几乎完全忽略了可持续性问题——现有 benchmark 只关注任务完成率（如 Step Success Rate），没有任何能耗指标。用户面对的只是一个简单的输入框，完全无法感知背后巨大的计算能耗。
3. **核心矛盾**：不同 Web Agent 的设计哲学导致能耗差距可达 10 倍以上，但这种差距对终端用户完全不透明。高能耗的 agent 并不必然带来更好的性能。
4. **本文要解决什么**：量化不同 Web Agent 的能耗差异，让研究社区和用户意识到这一问题的紧迫性，并推动评测标准纳入能效维度。
5. **切入角度**：同时从**实证测量**（直接基准测试开源 Agent）和**理论估算**（针对使用专有 LLM 的 Agent）两个互补角度进行分析。
6. **核心idea**：建立一套双轨评估框架——对开源 Agent 用 carbontracker 在真实 GPU 上直接测量能耗，对闭源 Agent 基于模型参数规模和 token 数量进行理论估算，从而全面揭示 Web Agent 的能耗全景。

## 方法详解

### 整体框架

提出双轨评估框架：(1) **实证基准测试** —— 在 8 种 GPU 上直接测量 5 个开源 Web Agent 的能耗；(2) **理论估算** —— 基于文献信息估算使用专有 LLM 的 Agent 能耗。两种方法互补覆盖开源和闭源 Agent。

### 关键设计

**模块一：实证基准测试（Empirical Benchmarking）**

- **做什么**：在 Mind2Web benchmark 上运行 5 个开源 Web Agent（AutoWebGLM、MindAct、MultiUI、Synapse、Synatra），用 carbontracker 库直接测量 GPU 能耗。
- **核心思路**：修改原始 Agent 代码，在执行开始和结束处插入 carbontracker 标记，捕获实际 GPU 能耗。在 8 种 NVIDIA GPU（A100、RTX 3090、H100、H200、L40S 等）上各运行 5 次取平均。
- **设计动机**：直接测量是最精确的方式，但前提是 Agent 和 LLM 都开源。通过多 GPU 多次运行确保结果稳定可靠。

**模块二：理论能耗估算（Theoretical Estimation）**

- **做什么**：对使用专有 LLM（如 GPT-4）的 Agent 进行能耗估算。核心公式为 $E_{action} = \bar{N} \cdot e_{token}$，其中 $\bar{N}$ 是每次动作的平均 token 数，$e_{token}$ 是每 token 能耗。
- **核心思路**：分析 Agent 论文和开源代码，确定其内部流程（输入模态、预处理步骤、LLM 调用次数），然后分别估算每个 LLM 组件的 token 数和 per-token 能耗。对 GPT-4 基于泄露的 1.8T 参数 MoE 架构，推导 FLOP 并映射到 H100 GPU 性能。
- **设计动机**：闭源 Agent 无法直接测量，但仍需提供某种比较手段。用 MindAct 同时进行测量和估算，可以评估估算方法的准确性。

**模块三：碳排放换算与可视化**

- **做什么**：将能耗乘以不同国家的碳排放因子（挪威 20g/kWh、美国 453g/kWh、澳大利亚 800g/kWh），换算为 CO₂ 排放量，并进一步转换为汽车行驶距离。
- **核心思路**：使不同 Agent 的环境代价直观可感。
- **设计动机**：能耗数字（kWh）对大多数人缺乏直觉，但"相当于开车 X 公里"的表述让影响易于理解。

### 损失函数 / 训练策略

本文不涉及模型训练。评估指标体系为：(1) **总能耗**（kWh）；(2) **每 token 能耗**（kWh/token）；(3) **能耗-性能比**（能耗 vs. 平均 Step Success Rate）；(4) **CO₂ 排放**（g CO₂e）。

## 实验关键数据

### 主实验

在 Nvidia H100-NVL GPU 上的综合对比：

| Agent | 平均 SSR (%) | 总能耗 (kWh) | 运行时间 (min) |
|-------|-------------|-------------|---------------|
| AutoWebGLM | **53.53** | **0.33** | **57.0** |
| MindAct | 43.50 | 1.22 | 296.0 |
| MultiUI | 34.70 | 0.82 | 130.0 |
| Synapse | 21.67 | 1.74 | 356.0 |
| Synatra | 15.85 | 3.31 | 426.0 |

理论估算对比（Mind2Web 全量）：

| Agent | 方法 | 能耗 (kWh) |
|-------|------|-----------|
| MindAct | 基准测试 | 1.22 |
| MindAct | 理论估算 | 8.5 |
| LASER (GPT-4) | 理论估算 | 99.21 |

### 消融实验

- **GPU 差异**：在 8 种 GPU 上，H100-NVL 最节能；不同 GPU 间能耗差异显著但 Agent 之间的排序一致。
- **估算 vs. 实测**：MindAct 理论估算值（8.5 kWh）约为实测值（1.22 kWh）的 7 倍，说明理论估算仅能提供量级参考。
- **per-token 能耗**：主要受 LLM 规模影响，但**总能耗**主要受 token 总量影响——有效的预处理（如 MindAct 的 HTML 剪枝）才是降低总能耗的关键。

### 关键发现

1. 最节能的 AutoWebGLM 同时也是性能最好的——**更多能耗 ≠ 更好结果**。
2. LASER（GPT-4）的估算能耗约为 MindAct 的 **10 倍**以上。
3. 对于美国电网，LASER 一次跑完 Mind2Web 的碳排放相当于开车 181 公里。
4. 对完全闭源的 Agent（如 Operator、Mariner），即使理论估算也无法进行。

## 亮点与洞察

- **首次系统性量化 Web Agent 能耗**：填补了该领域的空白，建立了基准数据。
- **双轨方法设计巧妙**：用同一个 Agent（MindAct）同时做实测和估算，验证了估算方法的局限性。
- **"预处理节能"的洞见**：Web Agent 能效的关键不在模型大小，而在于能否通过巧妙的预处理减少需要处理的 token 总量。
- **碳排放换算**直观有力——将抽象的 kWh 转化为开车公里数。

## 局限性 / 可改进方向

1. **理论估算精度有限**：7 倍的高估表明当前方法仅能提供粗略量级参考。
2. **完全闭源 Agent 无法评估**：如 OpenAI Operator、Google Mariner 由于没有任何技术细节公开，连估算都无法进行。
3. **仅评估推理能耗**：未考虑有些 Agent 需要微调带来的训练能耗。
4. **Mind2Web 基准局限**：离线 benchmark 可能无法反映真实部署中的能耗模式。
5. **未涉及解决方案**：主要是诊断和测量工作，没有提出降低能耗的具体技术方案。

## 相关工作与启发

- **LLM 碳排放研究**：GPT-3 训练产生约 550 吨 CO₂，BERT 约 0.754 吨——训练和推理的能耗都不容忽视。
- **推理能耗评估**：Samsi et al. 提出的 energy-per-token 是有用的评估指标。
- **Web Agent 多样性**：从输入模态（HTML/accessibility tree/screenshot）到模型选择（开源/闭源），设计哲学差异直接影响能耗。
- **启发**：未来的 Agent 评测应同时报告性能和能耗，就像 MLPerf 同时报告准确率和吞吐量。

## 评分

⭐⭐⭐

实用价值突出——首次为 Web Agent 领域建立了能耗基准，数据详实且实验设计合理。但作为研究贡献主要停留在测量和倡导层面，缺乏降低能耗的技术方案。理论估算方法的精度也有较大提升空间。
