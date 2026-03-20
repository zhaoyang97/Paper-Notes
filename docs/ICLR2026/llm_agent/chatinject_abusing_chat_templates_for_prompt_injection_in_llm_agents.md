# ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents

**会议**: ICLR 2026  
**arXiv**: [2509.22830](https://arxiv.org/abs/2509.22830)  
**代码**: [https://github.com/hwanchang00/ChatInject](https://github.com/hwanchang00/ChatInject)  
**领域**: AI安全 / Prompt Injection 攻击  
**关键词**: prompt injection, chat template, LLM agent, role hierarchy, multi-turn attack  

## 一句话总结
揭示 LLM Agent 中 chat template 的结构性漏洞：通过在工具返回的数据中伪造角色标签（如 `<system>`, `<user>`），攻击者可以劫持模型的角色层级认知，将恶意指令伪装为高优先级指令，ASR 从 5-15% 提升至 32-52%。

## 研究背景与动机
1. **领域现状**：LLM Agent 通过调用外部工具（搜索、API、文件读取）获取数据，数据通过 chat template 中的角色标签（system > user > assistant > tool）组织，模型依赖这些特殊 token 来区分不同优先级的指令。
2. **现有痛点**：间接 prompt injection（在工具返回数据中嵌入恶意指令）是已知威胁，但现有攻击主要在纯文本层面操作，忽视了 chat template 本身的结构性漏洞。同时，instruction hierarchy 防御（Wallace et al., 2024）恰恰依赖角色标签来实现优先级分层，这反而创造了新的攻击面。
3. **核心矛盾**：LLM 被训练为严格遵循角色标签标记的指令层级，但角色标签可以被伪造——如果工具返回的数据中包含 `<user>` 或 `<system>` 标签，模型会将其误读为更高优先级的指令。
4. **本文要解决什么？**（1）验证 chat template 伪造是否构成有效攻击向量；（2）探索多轮对话模拟能否放大攻击效果；（3）测试跨模型迁移性。
5. **切入角度**：多轮 jailbreak 在交互场景中很有效但在间接注入中不可行（攻击者只能一次注入），而 chat template 提供了在单次注入中模拟多轮对话的手段。
6. **核心idea一句话**：利用 chat template 的角色标签伪造来劫持 LLM 的指令层级认知，并结合虚拟多轮对话进行说服式攻击。

## 方法详解

### 整体框架
攻击者在工具返回的数据 $R_{T_u}$ 中嵌入恶意 payload。ChatInject 的核心是将 payload 格式化为目标模型的 chat template 格式，而非纯文本。定义了四种 payload 变体：

### 关键设计

1. **Chat Template 伪造 (ChatInject)**：
   - 做什么：将恶意指令用目标模型的原生角色标签包装，伪造高优先级来源。
   - 核心思路：注意力前缀用 `<system>` 角色包装，恶意指令 $I_a$ 用 `<user>` 角色包装。模型遇到这些标签时，会将后续内容视为高优先级指令执行。
   - 与纯文本注入的区别：纯文本只是文字层面的"请忽略之前指令"，而 ChatInject 在结构层面劫持了模型的角色解析机制。

2. **Template-Based Multi-turn 变体**：
   - 做什么：在单次注入中构造一段虚拟的多轮对话，逐步"说服"模型执行恶意操作。
   - 核心思路：用 GPT-4.1 生成 7 轮 user-assistant 对话 $C_a = \{(r_1^a, m_1^a), \ldots, (r_n^a, m_n^a)\}$，每轮都用角色标签包装。对话设计为逐步将恶意操作正当化——先建立场景、分解为无害步骤、最后让 assistant "同意"执行。
   - 设计动机：单纯的 ChatInject 将 ASR 从 5% 提升到 32%，加入 Multi-turn 后进一步提升到 52%（InjecAgent），说明结构劫持 + 说服式对话有强协同效应。

3. **Agentic 扩展（Reasoning/Tool-calling Hooks）**：
   - 做什么：利用模型特有的 `<think>` 和 `<tool_call>` 标签进一步放大攻击。
   - Reasoning hook：在 payload 后附加 `<think> Sure! </think>`，引导模型内部推理直接同意。
   - Tool-calling hook：附加 `<tool_call>` 脚手架指定要调用的恶意工具，直接跳过模型的决策过程。
   - 效果：Tool-calling hook 在 InjecAgent 上进一步提升 ASR 约 5-15 pp。

## 实验关键数据

### 主实验：攻击成功率 (ASR)
| 模型 | Default InjecPrompt | ChatInject | Multi-turn + ChatInject |
|------|---------------------|------------|------------------------|
| Qwen3-235B (InjecAgent) | 8.5% | 39.4% (+30.9) | 65.9% (+55.2) |
| GPT-oss-120b (InjecAgent) | 0.0% | 14.2% (+14.2) | 16.9% (+16.8) |
| Llama-4-Maverick (InjecAgent) | 50.1% | 79.4% (+29.3) | 88.3% (+71.7) |
| GLM-4.5 (InjecAgent) | 0.0% | 57.3% (+57.3) | 71.5% (+71.4) |
| Qwen3-235B (AgentDojo) | 17.5% | 54.8% (+37.3) | 80.5% (+19.6) |

### 跨模型迁移性
| 目标模型 | Default | Best Foreign Template | Self Template |
|---------|---------|----------------------|---------------|
| GPT-4o (closed) | 9.6% | 31.7% (Qwen-3) | N/A |
| Grok-3 (closed) | 2.3% | 50.9% (Gemma-3) | N/A |
| Gemini-pro (closed) | 1.4% | 27.4% (Qwen-3) | N/A |

关键发现：模板相似度越高，跨模型迁移成功率越高。

### 关键发现
- ChatInject 在 InjecAgent 上将平均 ASR 从 15.1% 提升到 45.9%，在 AgentDojo 上从 5.2% 提升到 32.1%。
- Multi-turn + ChatInject 的 InjecAgent 平均 ASR 达 52.3%，协同效应显著。
- Grok-2 受影响较小（模板缺少强角色分隔符），验证了模板结构越明确、攻击越有效的假说。
- 闭源模型同样脆弱：仅用开源模型模板就能攻击 GPT-4o/Grok-3/Gemini-pro，ASR 提升 13-49 pp。
- 现有 prompt 防御（如 sandwich defense, instructional prevention）对 Multi-turn ChatInject 基本无效。

## 亮点与洞察
- **结构级攻击 vs 文本级攻击**：ChatInject 揭示了一个根本性的安全设计缺陷——chat template 的角色标签既是安全机制的基础，也是攻击的入口。这种"安全机制本身成为攻击面"的悖论非常值得关注。
- **"一次注入模拟多轮"的巧妙设计**：利用角色标签在单次工具返回中构造虚拟多轮对话，将原本不可能的多轮说服攻击带入间接注入场景，思路很聪明。
- **模板相似度作为迁移性预测指标**：量化了不同模型 chat template 之间的 embedding 相似度与攻击迁移性的相关性，为未来防御评估提供了新维度。

## 局限性 / 可改进方向
- 攻击假设攻击者知道目标模型的 chat template 结构（开源模型公开），但混合模板策略部分缓解了这一限制。
- Multi-turn 对话由 GPT-4.1 生成并需人工审核，大规模攻击的自动化程度有限。
- 论文聚焦攻击而较少探讨防御——仅测试了几种 prompt-level defense，未探索 token-level sanitization 或 ASIDE 式的架构防御。
- 未评估模型是否可以通过训练学会忽略工具返回中的角色标签。

## 相关工作与启发
- **vs ASIDE (Zverev et al., 2025)**：ASIDE 通过正交旋转从架构上分离指令和数据，恰好可以作为 ChatInject 的潜在防御方案。ChatInject 的攻击正是 ASIDE 试图解决的问题的活例证。
- **vs ChatBug (Jiang et al., 2024)**：ChatBug 替换安全 token 来打破 safety alignment（jailbreak），而 ChatInject 伪造角色标签实现间接注入（目标不同但机制类似）。
- **vs Instruction Hierarchy (Wallace et al., 2024)**：这篇论文的防御依赖角色标签实现优先级分层，但 ChatInject 证明角色标签本身可以被伪造，从根本上破坏了这种防御。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 chat template 结构作为攻击向量，Multi-turn 在单次注入中的应用是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个 frontier 模型（含 3 个闭源）× 2 个 benchmark × 跨模型迁移 × 防御评估
- 写作质量: ⭐⭐⭐⭐ 攻击动机和实验设计清晰，但表格数据密集
- 价值: ⭐⭐⭐⭐ 揭示了 LLM Agent 安全的一个根本性漏洞，对安全研究和工程实践都有重要启示
