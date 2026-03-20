# EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation

**会议**: AAAI 2026  
**arXiv**: [2505.05440](https://arxiv.org/abs/2505.05440)  
**代码**: [https://github.com/Yi-Biao/EcoAgent](https://github.com/Yi-Biao/EcoAgent) (有)  
**领域**: Agent / 移动端自动化  
**关键词**: 设备-云端协作, 多Agent系统, 移动自动化, 隐私保护, Dual-ReACT  

## 一句话总结
提出 EcoAgent，一个闭环设备-云端协作的多 Agent 移动自动化框架，通过 Dual-ReACT 双层推理规划 + 设备端轻量验证反馈 + Pre-Understanding 文本压缩模块，在 AndroidWorld 上达到与全云端 Agent 相当的成功率，同时大幅降低延迟（3.9s vs 15.3s）、云端调用（降89%）和上行数据量（降48.6倍）。

## 研究背景与动机
1. **领域现状**：基于 MLLM 的移动 Agent 主要有三种架构——纯云端（如 M3A，双 GPT-4o）推理能力强但延迟高、成本大；纯设备端（如 ShowUI 2B）延迟低但复杂任务规划能力不足；开环设备-云端协作（如 UGround）云端规划+设备端执行但缺少反馈闭环。
2. **现有痛点**：(1) 开环协作需要频繁上传手机截图到云端做验证，暴露用户隐私且增加延迟；(2) 一旦执行出错，设备端无法反馈给云端重新规划，只能继续错误执行，无法从失败中恢复。
3. **核心矛盾**：云端 MLLM 推理能力强但通信成本高+隐私风险大；设备端 MSLM 延迟低+保护隐私但推理能力弱。需要在两者之间找到最佳的任务分配方式。
4. **本文要解决什么？** 如何实现闭环的设备-云端协作——让设备端能自主验证执行结果并在需要时向云端提供轻量反馈，使云端能据此做反思和重规划。
5. **切入角度**：观察到设备端通用 MSLM（未在大量 GUI 数据上微调的）虽然不擅长精确定位 UI 元素，但擅长理解屏幕语义和判断描述是否匹配——可以用来做轻量级的执行验证。
6. **核心idea一句话**：用 Dual-ReACT 让云端一次生成带期望的执行计划，设备端用轻量模型逐步验证并在失败时提供文本化反馈，实现隐私保护的闭环协作。

## 方法详解

### 整体框架
EcoAgent 由三个 Agent 组成：(1) 云端 Planning Agent（GPT-4o），负责 Dual-ReACT 规划和反思重规划；(2) 设备端 Execution Agent（ShowUI 2B 或 OS-Atlas 4B），负责精确的 UI 操作执行；(3) 设备端 Observation Agent（Qwen2-VL 2B），负责验证每步执行结果并通过 Pre-Understanding 模块将截图压缩为文本。工作流程：Planning Agent 生成带步骤和期望的计划 → Execution Agent 逐步执行 → Observation Agent 验证是否符合期望 → 成功则继续，失败则将文本反馈发回云端 → Planning Agent 反思重规划。

### 关键设计

1. **Dual-ReACT 双层推理规划**:
   - 做什么：将传统 ReACT 扩展为全局+局部两层推理，一次性生成完整的可执行计划。
   - 核心思路：给定用户指令 $Ins$ 和初始屏幕 $S_0$，先做 Global ReACT 将任务分解为子目标序列，再对每个子目标做 Local ReACT 生成具体操作步骤 $ST_t$ 和对应的期望结果 $EX_t$。计划 $P_0 = \text{GlReACT}(Ins, S_0) = \{\text{LoReACT}(ST_1, EX_1), \ldots, \text{LoReACT}(ST_t, EX_t)\}$。
   - 设计动机：生成的 $EX_t$ 是关键——它让设备端的 Observation Agent 不需要理解任务全局语义，只需判断当前屏幕是否匹配期望描述，将复杂的验证问题简化为简单的文本-图像匹配，使轻量模型也能胜任。

2. **Pre-Understanding Module（预理解模块）**:
   - 做什么：将屏幕截图压缩为 50-150 token 的文本描述，替代原始图像传输到云端。
   - 核心思路：$T_{t+1} = \text{PreUnderstand}(S_{t+1})$，由 Observation Agent（Qwen2-VL 2B）用简单提示将截图总结为 3-5 句话的功能描述。
   - 设计动机：(1) 原始截图约 1400+ token，压缩后仅 50-150 token，大幅降低通信开销和 MLLM token 消耗；(2) 传输文本而非原始图像，从根本上避免了屏幕内容泄露的隐私风险；(3) 重规划不需要全部屏幕细节，只需知道屏幕之间的语义变化轨迹。

3. **Memory + Reflection 反思重规划**:
   - 做什么：执行失败时，Planning Agent 利用存储的屏幕描述轨迹和失败原因进行反思，生成新计划。
   - 核心思路：Memory Module 存储文本化的屏幕轨迹和操作历史。失败时 Reflection Module 分析错误轨迹，生成新计划 $P_n = \text{Reflection}(Ins, P_{n-1}, \text{Memory})$。
   - 设计动机：借鉴 Reflexion 的迭代改进思路，使系统能从错误中学习并自适应恢复，这是闭环协作区别于开环系统的核心优势。

### 损失函数 / 训练策略
EcoAgent 不涉及端到端训练。Execution Agent 使用已有的 GUI 微调模型（ShowUI 2B / OS-Atlas 4B），Observation Agent 使用通用的 Qwen2-VL 2B（未微调），Planning Agent 使用 GPT-4o。整个框架是即插即用的。

## 实验关键数据

### 主实验
AndroidWorld 基准上的任务成功率（SR）和运营成本对比：

| 架构 | Agent | 基础模型 | SR(%) | MC(云调用) | MT(Token) |
|------|-------|---------|-------|-----------|-----------|
| 设备端 | ShowUI | ShowUI 2B | 7.0 | 0 | 0 |
| 设备端 | V-Droid | V-Droid 8B | 59.5 | – | – |
| 云端 | AppAgent | GPT-4o | 11.2 | 6.46 | 15309 |
| 云端 | M3A | GPT-4o×2 | 28.4 | 13.39 | 87469 |
| 开环 | UGround-2B | GPT-4o×2+2B | 32.8 | 12.21 | 45192 |
| **闭环** | **EcoAgent(OS-Atlas)** | GPT-4o+4B+2B | **27.6** | **1.53** | **3240** |

延迟对比：

| Agent | 延迟(s) |
|-------|---------|
| M3A | 15.3 |
| UGround-2B | 18.2 |
| **EcoAgent(ShowUI)** | **3.9** |

### 消融实验

| 配置 | SR(%) | MC | MT | 说明 |
|------|-------|----|----|------|
| 仅 Execution Agent (ShowUI) | 7.0 | 0 | 0 | 无规划能力 |
| + Planning Agent | 15.5 | 1 | 2149 | +8.5%，Dual-ReACT有效 |
| + Observation Agent | **25.6** | 1.87 | 3545 | +10.1%，闭环反馈关键 |

### 关键发现
- EcoAgent 与 M3A 成功率相当（27.6% vs 28.4%），但云端调用降低 89%、Token 降低 96%。
- 上行数据量仅 120kB/任务，比 M3A 的 5831kB 低 48.6 倍，隐私保护效果显著。
- 失败分析显示 50% 的失败来自视觉定位错误（MSLM 局限），35% 来自规划错误，说明 EcoAgent 的性能上限主要受设备端模型能力制约，未来更强的端侧模型可直接提升效果。
- Observation Agent 贡献最大：从 15.5% 提升到 25.6%（+10.1%），证明闭环反馈是关键设计。

## 亮点与洞察
- Dual-ReACT 中"期望"的设计非常巧妙：把验证问题从"理解任务是否完成"降级为"判断屏幕是否匹配描述"，使得 2B 小模型就能做设备端验证，是实现闭环的关键使能技术。
- Pre-Understanding Module 一石三鸟：降成本、保隐私、减延迟。将图像信息压缩为文本的思路可以推广到任何需要设备-云端通信的 Agent 场景。
- 框架是模型无关的（model-agnostic），可以随时替换更强的设备端模型（如 V-Droid 8B 作为 Execution Agent），具有很好的可扩展性。

## 局限性 / 可改进方向
- 27.6% 的绝对成功率仍然不高，V-Droid（纯设备端 8B）达到 59.5%，说明端侧模型能力的天花板限制了整体性能。
- Pre-Understanding 模块的信息压缩可能丢失关键细节，导致云端重规划信息不足。
- 当前只在 AndroidWorld（116 个任务）上评测，缺少更大规模和更多应用场景的验证。
- 设备端推理延迟（~3.9s/步）对于真实用户交互仍偏高。

## 相关工作与启发
- **vs M3A**: M3A 用两个 GPT-4o 做规划和验证，EcoAgent 用一个 GPT-4o + 两个 2-4B 端侧模型达到相近效果，成本降低一个数量级。闭环是核心差异。
- **vs UGround**: UGround 是开环设备-云端协作，仍需频繁上传截图到云端验证。EcoAgent 通过 Dual-ReACT 的期望机制实现了设备端验证，避免了隐私泄露。
- **vs V-Droid**: V-Droid 是纯设备端的 8B 模型，成功率最高（59.5%），但无法做复杂的长程规划。将 V-Droid 集成为 EcoAgent 的 Execution Agent 是一个很有前景的方向。

## 评分
- 新颖性: ⭐⭐⭐⭐ 闭环设备-云端协作+Dual-ReACT+Pre-Understanding的组合设计新颖，但各单一模块的创新有限。
- 实验充分度: ⭐⭐⭐⭐ 三维度评估（成功率/成本/延迟）+消融+失败分析+系统开销分析，较为完整。
- 写作质量: ⭐⭐⭐⭐ 架构图和对比图清晰，问题动机阐述到位。
- 价值: ⭐⭐⭐⭐ 为移动 Agent 的实际部署提供了一个实用的低成本方案，工程价值高。
