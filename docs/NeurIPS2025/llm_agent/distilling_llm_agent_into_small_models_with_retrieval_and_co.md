# Distilling LLM Agent into Small Models with Retrieval and Code Tools

**会议**: NeurIPS 2025  
**arXiv**: [2505.17612](https://arxiv.org/abs/2505.17612)  
**代码**: [https://github.com/Nardien/agent-distillation](https://github.com/Nardien/agent-distillation)  
**领域**: LLM Agent / 知识蒸馏  
**关键词**: agent distillation, first-thought prefix, self-consistent action generation, small language model, CodeAct  

## 一句话总结
提出 Agent Distillation 框架，将 LLM agent 的完整 reason-act-observe 交互行为（而非静态 CoT）蒸馏到 0.5B-7B 小模型中，配合 first-thought prefix 提升教师轨迹质量和 self-consistent action generation 提升推理鲁棒性，使小模型达到比其大 2-4× 的 CoT 蒸馏模型的性能。

## 研究背景与动机

1. **领域现状**：CoT 蒸馏（从大模型的推理 trace 训练小模型）是当前压缩 LLM 推理能力的主流方法，已被 Llama3、Qwen2.5、DeepSeek-R1 等广泛采用。

2. **现有痛点**：CoT 蒸馏传授的是静态推理——小模型必须将事实知识和计算过程都"记住"在参数中。当遇到训练时未见过的新知识或复杂计算时，小模型容易幻觉。例如 "2010年投资$100买Apple股票到2020年值多少" 需要事实知识（股价历史）和精确计算，CoT 蒸馏的小模型两者都易出错。

3. **核心矛盾**：小模型参数有限，不可能同时记住大量事实知识和具备精确计算能力。但如果让小模型学会使用工具（检索+代码执行），就可以将知识存储和计算委托给外部工具，只需学习"如何推理并调用工具"的行为模式。

4. **本文要解决什么**：将 LLM agent 的完整 agentic 行为（reasoning + tool use + environment interaction）蒸馏到 ≤3B 的小模型中，使其成为能用检索和代码工具的小型 agent。

5. **切入角度**：(1) 发现 instruction-tuned LLM 在被提示为 agent 时推理质量下降（尤其数学任务），因为 agent 指令与 CoT 训练分布不一致；(2) 小模型生成的代码 action 经常有语法错误或执行失败。

6. **核心idea一句话**：用 first-thought prefix（CoT 首步接入 agent 轨迹起始）修复教师 agent 的推理质量滑坡，用 self-consistent action generation（多次采样选一致结果）修复学生 agent 的代码执行失败。

## 方法详解

### 整体框架
Agent Distillation 分为两个阶段：(1) 训练时：用 32B 教师 LLM 生成 CodeAct 格式的 reason-act-observe 轨迹，在学生模型上做 SFT（只对 thought 和 action 计算 loss，observation 不计）；(2) 推理时：学生 agent 在环境中多步交互，每步通过检索工具获取知识或通过代码工具计算。

### 关键设计

1. **First-Thought Prefix (FTP)**：
   - 做什么：修复 instruction-tuned 教师模型在 agent 模式下的推理退化
   - 核心思路：先用 CoT prompt 让教师模型生成第一步推理 $y_1$，然后将 $y_1$ 作为 agent 第一个 thought 的前缀拼接到 agent prompt 中，再生成完整 agent 轨迹。这确保了 agent 从正确的推理方向起步
   - 设计动机：研究表明 LLM 推理链的第一步对最终结论有决定性影响。Agent 指令（"按 Thought/Code/Observation 循环"）可能覆盖模型原本的 CoT 推理模式。FTP 类似于"越狱攻击"中的 prefix 注入技术，但用于正面目的——引导推理方向
   - 注意事项：FTP 只用于生成教师轨迹，学生推理时不需要。但 FTP 有时会导致模型内部生成知识而非调用检索工具，增加幻觉风险

2. **Self-Consistent Action Generation (SAG)**：
   - 做什么：提升小模型 agent 在推理时的代码生成鲁棒性
   - 核心思路：每步采样 N=8 个 thought-action 序列（高温 nucleus sampling），过滤掉解析或执行失败的代码，对剩余有效结果做 majority voting 选出最一致的结果。所有候选都失败时，随机保留一个失败的 action 并将错误信息作为 observation 反馈，让模型自我修正
   - 设计动机：0.5B-3B 模型虽然预训练过代码数据，但生成有效 Python 代码的能力有限。SAG 利用"小模型能生成有效代码，只是概率较低"这一特点，通过多次采样提升选到有效代码的几率

### 损失函数 / 训练策略
- 标准 SFT loss，只对 thought 和 action token 计算，observation 不参与
- LoRA（rank 64），所有线性层，lr=2e-4，batch size 8，2 epochs
- 训练数据：1000 HotPotQA + 2000 MATH，筛选正确轨迹后约 2000 条
- 4×A100 80GB

## 实验关键数据

### 主实验

| 方法 | 0.5B Avg | 1.5B Avg | 3B Avg | 7B Avg |
|---|---|---|---|---|
| CoT Distill | 13.64 | 21.28 | 27.72 | 33.54 |
| CoT Distill + RAG | 15.90 | 24.64 | 28.53 | 32.16 |
| **Agent Distill** | **19.24** | **28.06** | **33.60** | **39.85** |
| **+FTP+SAG** | **21.90** | **30.55** | **36.60** | **42.68** |

核心发现：0.5B Agent ≈ 1.5B CoT，1.5B Agent ≈ 3B CoT，3B Agent > 7B CoT，7B Agent > 32B CoT。Agent 蒸馏使小模型达到 2-4× 大模型的 CoT 蒸馏性能。

### 消融实验

| 组件 | 贡献 | 说明 |
|---|---|---|
| FTP 对教师轨迹质量 | MATH hard: 58.4→67.1, MATH medium: 78.4→83.4 | 显著提升教师 agent 在难题上的表现 |
| SAG 对代码错误 | 0.5B AIME 解析错误减半 | 通过多采样过滤显著降低无效 action |
| FTP 对 retrieval 调用 | 减少检索次数 | FTP 导致模型更多使用内部知识而非检索，可能增加幻觉 |
| LoRA vs Full FT | LoRA: 29.11 vs Full FT: 26.24 (1.5B) | LoRA 泛化更好，Full FT 易过拟合 |
| Code-specific model | 影响微小 | Qwen2.5-Coder 作为 teacher 略有帮助但不显著 |

### 关键发现
- **Agent 蒸馏 vs RAG**：静态 RAG 在事实推理上有帮助但在数学推理上有害（预取文档与任务不匹配）。Agent 蒸馏让模型自主决定何时检索，更灵活
- **FTP 对复杂题效果更好**：MATH level 5 和 AIME 上提升最大，因为复杂推理更依赖正确的初始方向
- **token 开销基本持平**：Agent 在事实推理上生成更多 token（多次检索），但在数学推理上更少 token（用 for-loop 代替冗长计算），整体无显著差异
- **跨模型族泛化**：在 Llama-3.2-1B 和 Phi-4-mini 上也观察到一致的提升趋势

## 亮点与洞察
- **"蒸馏行为而非知识"的范式转变**：传统 CoT 蒸馏让小模型记忆推理过程，Agent 蒸馏让小模型学习如何与外部工具交互。前者受限于模型参数容量，后者将知识和计算外包给工具
- **First-Thought Prefix 的双面性**：在数学推理上通过引导初始方向显著提升性能，但在事实推理上可能导致模型"自信回答"而非调用检索，增加幻觉。这揭示了 agentic reasoning 中内部推理与外部工具使用的 tension
- **0.5B 也能当 Agent**：这是一个很有实用价值的发现。prompt 方式下 0.5B 模型几乎无法产生有效 agent 输出，但经过蒸馏后可以在多个 benchmark 上给出有意义的结果

## 局限性 / 可改进方向
- 只蒸馏了检索和代码两种工具，未探索网页浏览、模拟器交互等更复杂的 agent 场景
- 仅用单条教师轨迹训练，增加采样数可能进一步提升
- Agent 蒸馏不直接提升小模型的核心推理能力，RL 后训练可以进一步增强
- 代码执行存在安全风险，未充分讨论沙箱化方案

## 相关工作与启发
- **vs Search-R1 / ToolRL**：这些工作用 RL 训练 LLM 使用搜索/工具，本文用 SFT 蒸馏方式将 agentic 能力传递到极小模型（0.5B-3B），成本更低但不涉及策略优化
- **vs GiGPO (2505.10978)**：GiGPO 解决 Agent RL 训练中的 credit assignment 问题，本文解决 Agent 能力从大到小的蒸馏问题。两者互补——先蒸馏再 RL fine-tune 是一条可行路线
- **vs FireAct / AgentTuning**：这些工作主要在 7B+ 模型上做 agent 微调，本文首次系统研究 ≤3B 极小模型的 agent 蒸馏，并提出了专门针对小模型问题（代码生成失败、推理方向偏移）的解决方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 agent 行为蒸馏到极小模型是有价值的新方向，FTP 和 SAG 设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 4个模型规模，8个benchmark，跨模型族验证，大量消融分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Figure 1 的性能对比和 Figure 2 的概念图直观
- 价值: ⭐⭐⭐⭐ 对构建低资源可部署的 Agent 有直接指导意义，但受限于 SFT 范式
