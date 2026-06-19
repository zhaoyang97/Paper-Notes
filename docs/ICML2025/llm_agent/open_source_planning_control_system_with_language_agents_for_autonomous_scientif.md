---
title: >-
  [论文解读] Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery
description: >-
  [ICML 2025 (Workshop on Machine Learning for Astrophysics)][LLM Agent][多智能体系统] 本文提出 cmbagent，一个由约 30 个 LLM Agent 组成的多智能体系统，采用 Planning & Control 策略编排无人干预的科研工作流，各 Agent 分别负责论文检索、代码编写、结果解读、输出评审等专业任务，并可在本地执行代码；该系统成功完成了博士级别的宇宙学任务（用超新星数据测量宇宙学参数），在两个基准测试集上优于当前最先进的 LLM。
tags:
  - "ICML 2025 (Workshop on Machine Learning for Astrophysics)"
  - "LLM Agent"
  - "多智能体系统"
  - "科研自动化"
  - "Planning & Control"
  - "宇宙学"
  - "自主科学发现"
---

# Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery

**会议**: ICML 2025 (Workshop on Machine Learning for Astrophysics)  
**arXiv**: [2507.07257](https://arxiv.org/abs/2507.07257)  
**代码**: [https://github.com/CMBAgents/cmbagent](https://github.com/CMBAgents/cmbagent) (开源)  
**领域**: LLM Agent  
**关键词**: 多智能体系统, 科研自动化, Planning & Control, 宇宙学, LLM Agent, 自主科学发现

## 一句话总结

本文提出 cmbagent，一个由约 30 个 LLM Agent 组成的多智能体系统，采用 Planning & Control 策略编排无人干预的科研工作流，各 Agent 分别负责论文检索、代码编写、结果解读、输出评审等专业任务，并可在本地执行代码；该系统成功完成了博士级别的宇宙学任务（用超新星数据测量宇宙学参数），在两个基准测试集上优于当前最先进的 LLM。

## 研究背景与动机

**领域现状**：自主科学发现（Autonomous Scientific Discovery）是 AI for Science 的前沿方向。科学研究过程包含文献调研、假设提出、实验设计、代码实现、数据分析、结果解读等多个环节，传统上完全依赖人类研究者逐步完成。近年来，大语言模型（LLM）展现了在代码生成、文本理解和推理方面的强大能力，为自动化科研流程提供了新的可能。

**现有痛点**：现有的 LLM-based 科研辅助工具通常只覆盖科研流程的单一环节——例如仅做文献检索（如 Semantic Scholar API 包装器）、仅做代码生成（如 Copilot）或仅做数据分析（如 Code Interpreter）。缺乏一个能够**端到端**编排完整科研流程的集成系统。此外，单一 LLM Agent 在处理复杂多步骤的科研任务时面临注意力分散、上下文窗口不足和专业能力有限等挑战。

**核心矛盾**：科学研究任务要求系统同时具备异质化能力——文献理解、数学推导、代码实现、实验设计、结果批判等——单个 Agent 很难同时精通所有这些领域。同时，科研流程中许多环节存在复杂的依赖关系（如需要先理解理论背景才能正确设计实验），需要精心的规划和流程控制。

**本文切入角度**：受到工业界多 Agent 框架（如 AutoGen、MetaGPT）和经典控制论中 Planning & Control 思想的启发，作者设计了一个约 30 个专业化 Agent 组成的系统，通过规划与控制策略协调各 Agent 的工作，实现从文献检索到结果产出的全流程自动化。特别关注天体物理学/宇宙学领域的科研任务自动化。

**核心 idea**：用 ~30 个专业化 LLM Agent + Planning & Control 编排策略，构建无需人类介入的端到端科研自动化系统。

## 方法详解

### 整体框架

cmbagent 是一个多智能体系统，其核心架构由以下层次组成：

1. **Planning 层（规划层）**：系统接收高层科研任务描述后，由规划模块将任务分解为有序的子任务序列，确定各子任务之间的依赖关系和执行顺序。
2. **Control 层（控制层）**：控制模块负责实时监控各 Agent 的执行状态，根据中间结果动态调整后续步骤——如果某个 Agent 的输出不合格，控制层会触发重试或切换到替代策略。
3. **Agent Pool（Agent 池）**：约 30 个 LLM Agent，每个 Agent 专注于特定任务类型，通过精心设计的 system prompt 被赋予专业能力。
4. **代码执行环境**：系统集成了本地代码执行能力，Agent 编写的代码可以直接在本地运行，获取真实的计算结果。

整个系统在执行过程中**无需人类干预**（no human-in-the-loop），从任务接收到最终结果输出完全自主运行。

### 关键设计

#### 1. 专业化 Agent 分工

系统中的 ~30 个 Agent 覆盖科研工作流的各个环节：

| Agent 类型 | 职责描述 | 核心能力 |
|-----------|---------|---------|
| 文献检索 Agent | 检索相关科学论文和代码库 | RAG、语义搜索、关键信息提取 |
| 代码编写 Agent | 根据科研需求编写实现代码 | 代码生成、API 调用、数据处理脚本 |
| 结果解读 Agent | 分析计算输出，提取科学含义 | 数值结果解读、统计分析、可视化 |
| 输出评审 Agent | 审核其他 Agent 的输出质量 | 一致性检查、科学合理性评估、错误检测 |
| 数学推导 Agent | 处理理论公式和数学推导 | 符号计算、公式验证、理论推导 |
| 实验设计 Agent | 规划实验步骤和参数设置 | 实验方案设计、参数空间定义 |

每个 Agent 通过针对性的 prompt engineering 获得特定领域的专家级能力，避免了单一 Agent 处理所有任务时的能力瓶颈。

#### 2. Planning & Control 策略

- **Planning（规划）**：采用层次化任务分解策略，将复杂科研任务（如"用超新星数据测量宇宙学参数"）递归分解为可由单个 Agent 完成的原子子任务。规划阶段确定任务的执行拓扑——哪些子任务可以并行，哪些存在前后依赖。
- **Control（控制）**：在执行过程中实时监控各 Agent 的输出质量和任务进度。控制策略包括：(a) **状态追踪**——记录每个子任务的完成状态和中间产物；(b) **质量门控**——评审 Agent 对关键输出进行评审，不合格则反馈修正；(c) **动态调整**——根据中间结果调整后续规划（如初步分析发现数据质量问题时，插入额外的数据清洗步骤）。

#### 3. 本地代码执行

与纯对话式 LLM 不同，cmbagent 可以在本地执行代码。这意味着 Agent 不仅能"讨论"如何分析数据，还能实际运行 Python/Julia 等脚本、调用科学计算库（如 NumPy、SciPy、Astropy 等宇宙学工具），获取真实的数值结果。本地执行消除了"幻觉风险"——计算结果来自真实的程序运行而非 LLM 生成。

#### 4. 跨 Agent 评审机制

系统设置了专门的评审 Agent（Critic Agent），对其他 Agent 的输出进行质量审核。这种"对抗性验证"机制有效降低了单个 Agent 输出中的错误率，类似于科研中的同行评审（peer review）过程。

## 实验关键数据

### 核心实验：博士级宇宙学任务

cmbagent 被应用于一项博士级别的宇宙学任务——使用 Ia 型超新星数据测量宇宙学参数（如哈勃常数 $H_0$、物质密度参数 $\Omega_m$ 等）。这是宇宙学研究中的经典问题，通常需要研究生具备扎实的理论背景和数据分析能力才能完成。

| 评估维度 | cmbagent 表现 | 对比基准 |
|---------|-------------|---------|
| 任务完成度 | 成功完成端到端宇宙学参数测量 | SOTA LLM 无法独立完成全流程 |
| 自动化程度 | 全程无人干预（no human-in-the-loop） | 现有工具通常需要人工介入多个环节 |
| 代码执行 | 本地执行科学计算代码并获取真实结果 | 纯 LLM 方法依赖文本生成，无法验证 |
| 科学合理性 | Agent 间交叉评审保障输出质量 | 单 Agent 方法缺乏自我纠错机制 |

### 基准测试集评估

系统在两个基准测试集上进行了评估，与 state-of-the-art LLM 进行对比：

| 对比维度 | cmbagent（多Agent） | SOTA 单一 LLM | 优势分析 |
|---------|-------------------|-------------|---------|
| 综合性能 | 优于对比方法 | 基线水平 | Planning & Control + 专业化 Agent 分工的系统性优势 |
| 任务覆盖 | 端到端科研流程 | 通常只覆盖部分环节 | ~30 Agent 分工覆盖全流程 |
| 可靠性 | 评审 Agent 交叉验证 | 单次生成无验证 | 多 Agent 评审 ≈ 同行评审机制 |
| 可解释性 | 各 Agent 输出可独立审查 | 端到端黑箱 | 模块化架构提升透明度 |
| 部署灵活性 | GitHub + HuggingFace + 云端 | 通常仅 API 调用 | 开源 + 多平台部署 |

### 关键发现

- **多 Agent 系统在复杂科研任务上显著优于单一 LLM**：约 30 个专业化 Agent 的分工协作，使系统能够处理单一 LLM 无法独立完成的博士级科研任务。
- **Planning & Control 是成功的关键**：合理的任务分解和实时的执行控制确保了复杂多步骤流程的顺利推进，避免了中间环节的错误累积。
- **代码执行消除幻觉**：通过本地执行代码获取真实计算结果，而非依赖 LLM 生成数值，从根本上解决了 LLM 在科学计算中的幻觉问题。
- **评审机制提升可靠性**：专门的 Critic Agent 对其他 Agent 输出的审核机制，类似于科研中的同行评审，有效提升了最终输出的科学质量。

## 亮点与洞察

- **Agent 数量达到 ~30 是有意义的设计选择**：相比 3-5 个 Agent 的小型系统，cmbagent 通过更细粒度的 Agent 分工实现了更强的专业化。每个 Agent 只需精通一个狭窄领域，降低了对底层 LLM 通用能力的要求。这暗示了一个趋势：未来的 Agent 系统可能会朝着"更多、更专"的方向发展。
- **科学领域的 Agent 系统需要代码执行**：与一般的对话任务不同，科学研究的核心是数值计算和数据分析，纯文本的 Agent 交互远远不够。cmbagent 的本地代码执行能力是其完成博士级任务的必要条件。
- **Planning & Control 借鉴控制论思想**：区别于简单的链式调用（chain）或图调用（graph），P&C 策略引入了反馈回路——控制层根据执行结果动态调整规划，这种闭环设计更适合不确定性较高的科研任务。
- **开源 + 多平台部署降低使用门槛**：系统同时在 GitHub（代码）、HuggingFace（模型/空间）和云端提供服务，覆盖了从开发者到终端用户的不同需求层次。

## 局限与展望

- **领域特异性较强**：目前主要聚焦于宇宙学/天体物理学领域的科研任务，系统中的许多 Agent 和检索资源都针对该领域定制。迁移到其他科学领域（如生物学、化学）需要重新设计 Agent 的专业化 prompt 和知识库。
- **Agent 编排复杂度**：约 30 个 Agent 的系统带来了显著的编排复杂度——Agent 间的通信开销、状态同步、错误传播等问题在实际部署中可能成为瓶颈。论文未充分讨论系统的可扩展性和资源消耗。
- **Workshop 论文的评估深度有限**：作为 ICML Workshop 论文，实验规模和详细程度不如主会论文，缺乏对各组件的系统消融实验和对不同 LLM backbone 的全面对比。
- **无人干预的风险**：完全自主的科研流程缺乏人类专家的关键判断介入（如实验假设的合理性、结果的物理直觉判断），在开放性研究问题上可能产生看似合理但科学上有误的结论。
- **成本效率**：~30 个 LLM Agent 的推理成本较高，对于资源有限的研究团队可能不够友好。论文未给出详细的 token 消耗和时间开销分析。
- **可复现性**：虽然代码开源，但系统依赖特定的 LLM API 和科学计算库配置，完整复现博士级任务的实验环境搭建可能存在较高门槛。

## 相关工作与启发

- **vs AutoGen / MetaGPT**：这些是通用多 Agent 框架，cmbagent 在此基础上针对科研自动化做了深度定制（~30 个领域专业化 Agent、科学论文/代码库检索、本地科学计算执行）。cmbagent 更专、更深，但通用性不如前者。
- **vs ChemCrow / Coscientist**：同样是面向科学发现的 Agent 系统，但分别聚焦于化学和实验科学。cmbagent 的独特之处在于面向计算密集型的宇宙学研究，强调代码执行和数值计算。
- **vs SciAgent / AI Scientist**：类似的科研自动化系统，但 cmbagent 的 Agent 数量更多（~30 vs 通常 3-5 个），分工更细，且采用了 Planning & Control 而非简单的链式调用。
- **启发**：cmbagent 的 Planning & Control + 大规模专业化 Agent 池的架构，为构建其他领域的科研自动化系统提供了参考范式。关键经验是：科学研究的自动化不仅需要语言能力（LLM），更需要执行能力（代码运行）和质量控制机制（跨 Agent 评审）。

## 评分

- 新颖性: ⭐⭐⭐⭐ ~30 Agent 级别的大规模科研自动化系统较为新颖，Planning & Control 策略的引入为 Agent 编排提供了新思路
- 实验充分度: ⭐⭐⭐ Workshop 论文篇幅有限，成功完成博士级宇宙学任务令人印象深刻，但缺乏系统消融和详细数值对比
- 写作质量: ⭐⭐⭐⭐ 系统架构和应用场景描述清晰，开源部署信息完整
- 价值: ⭐⭐⭐⭐ 展示了大规模 Agent 系统在真实科研任务上的可行性，开源代码和多平台部署增强了实际影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[CVPR 2026\] SciEducator: Scientific Video Understanding and Educating via Deming-Cycle Multi-Agent System](../../CVPR2026/llm_agent/scieducator_scientific_video_understanding_and_educating_via_deming-cycle_multi-.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICML 2025\] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)

</div>

<!-- RELATED:END -->
