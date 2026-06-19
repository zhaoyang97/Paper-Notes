---
title: >-
  [论文解读] GUI Agents: A Survey
description: >-
  [ACL 2025 (Findings)][LLM Agent][GUI代理] 本文对基于大型基础模型的图形用户界面（GUI）代理进行了全面综述，提出了涵盖感知、推理、规划、执行四大能力的统一分析框架，系统梳理了 GUI Agent 的基准测试、评估指标、架构设计和训练方法，并讨论了关键挑战和未来方向。
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "GUI代理"
  - "自动化交互"
  - "大模型Agent"
  - "基准评测"
  - "多平台操作"
---

# GUI Agents: A Survey

**会议**: ACL 2025 (Findings)  
**arXiv**: [2412.13501](https://arxiv.org/abs/2412.13501)  
**代码**: 无  
**领域**: Agent / 人机交互  
**关键词**: GUI代理, 自动化交互, 大模型Agent, 基准评测, 多平台操作

## 一句话总结

本文对基于大型基础模型的图形用户界面（GUI）代理进行了全面综述，提出了涵盖感知、推理、规划、执行四大能力的统一分析框架，系统梳理了 GUI Agent 的基准测试、评估指标、架构设计和训练方法，并讨论了关键挑战和未来方向。

## 研究背景与动机

**领域现状**：GUI Agent 是指能够自主与数字系统的图形用户界面交互的智能代理——它们通过观察屏幕截图、理解界面元素，然后执行点击、输入、滑动等操作来完成用户指定的任务。随着 GPT-4V、Gemini 等多模态大模型的出现，GUI Agent 从早期基于规则和手工特征的方案发展为基于大模型的端到端系统，能力有了质的飞跃。

**现有痛点**：GUI Agent 是一个快速发展但高度碎片化的领域——不同平台（Web、移动端、桌面端）有不同的交互范式，不同工作采用不同的基准和评估指标，新方法层出不穷但缺乏统一的对比框架。研究者很难快速把握全貌，也很难判断哪些技术真正有效。

**核心矛盾**：GUI Agent 需要同时具备视觉感知（识别界面元素）、语义理解（理解任务意图和界面含义）、规划推理（分解复杂任务为操作序列）和精确执行（在正确位置执行正确操作）四种能力，但现有工作通常只关注其中一两个方面，缺乏全面的能力分析视角。

**本文目标**：提供一个全面、结构化的 GUI Agent 领域综述，建立统一的分析框架，帮助研究者和从业者快速理解当前进展、核心技术和开放问题。

**切入角度**：作者从 GUI Agent 的四大核心能力出发（感知→推理→规划→执行），将现有工作组织在这个统一框架下，同时横向覆盖基准测试、训练方法和跨平台差异。

**核心 idea**：提出一个四维能力框架（Perception, Reasoning, Planning, Acting）来统一分析各种 GUI Agent 方法，并在此框架下全面梳理了基准、架构和训练策略。

## 方法详解

### 整体框架

本文的分析框架将 GUI Agent 的能力分为四个层次，形成一个从输入到输出的完整 pipeline：

1. **感知（Perception）**：从屏幕截图和可访问性树（accessibility tree）中提取界面信息
2. **推理（Reasoning）**：理解当前界面状态、用户意图和任务上下文
3. **规划（Planning）**：将复杂任务分解为可执行的操作序列
4. **执行（Acting）**：将规划的操作转化为具体的 GUI 交互动作

### 关键设计

1. **感知能力分类**:

    - 功能：总结 GUI Agent 如何"看懂"界面
    - 核心思路：感知是 GUI Agent 的基础，决定了代理能获取多少界面信息。综述将感知方法分为三类：
        - **截图理解**：直接将屏幕截图作为视觉输入，依赖视觉语言模型（VLM）理解界面布局和内容。优势是不依赖平台API，但面临分辨率限制和小元素识别困难。
        - **DOM/Accessibility Tree 解析**：利用平台提供的结构化信息（HTML DOM、Android View Hierarchy 等）获取元素位置和属性。信息准确但依赖平台支持，跨应用泛化差。
        - **混合方法**：同时使用视觉截图和结构化信息，如 Set-of-Mark（在截图上标注元素编号）方法。兼顾准确性和视觉理解。
    - 关键挑战：小图标/密集界面的识别、多分辨率适配、动态界面变化的跟踪。

2. **规划能力分析**:

    - 功能：分析 GUI Agent 如何将复杂任务分解为可执行步骤
    - 核心思路：综述将规划策略分为三类：
        - **反应式（Reactive）**：每步根据当前截图直接决定下一步操作，不做长期规划。简单高效但容易陷入局部最优。
        - **预规划式（Pre-planning）**：在执行前先生成完整的操作计划，然后逐步执行。全局优化但计划可能因界面变化失效。
        - **自适应规划（Adaptive）**：生成初始计划但在执行中根据实际界面状态动态调整。结合了两者优势，是当前主流方向。
    - 相关技术：ReAct、Reflexion、Tree-of-Thought 等推理框架被广泛用于 GUI Agent 的规划模块。

3. **基准与评估体系**:

    - 功能：系统梳理 GUI Agent 的评测生态
    - 核心思路：综述按平台分类整理了主要基准：
        - **Web 平台**：MiniWoB++（简单网页操作）、WebArena（真实网站交互）、Mind2Web（跨网站泛化）
        - **移动平台**：AndroidWorld、Mobile-Env、AITW（Android In The Wild）
        - **桌面平台**：OSWorld（跨操作系统）、WindowsAgentArena
        - **跨平台**：OmniACT（统一多平台评测）
    - 评估指标：任务成功率（最核心）、步骤正确率、效率（步数）、泛化性（未见任务/应用的表现）。
    - 关键发现：现有基准之间差异大，即使在同一基准上不同论文的实验设置也不统一，导致可比性差。作者呼吁建立更标准化的评测协议。

### 训练方法总结

综述将 GUI Agent 的训练方法分为四类：

- **Prompt Engineering**：不训练模型，通过精心设计的 prompt 和 in-context examples 引导通用 LLM/VLM 完成 GUI 操作。入门简单但性能天花板低。
- **监督微调（SFT）**：用人类操作轨迹数据微调模型。需要高质量标注数据，效果好但数据收集昂贵。
- **强化学习（RL）**：通过在环境中试错获得奖励信号来训练。不需要标注但训练不稳定，需要高效的模拟环境。
- **混合方法**：先 SFT 建立基础能力，再用 RL 在线优化。综合效果最好，是前沿方向。

## 实验关键数据

### 各平台主要基准性能对比

| 基准 | 平台 | 最佳方法 | 成功率 | 说明 |
|------|------|---------|--------|------|
| MiniWoB++ | Web (简单) | GPT-4V + SoM | ~95% | 简单任务已接近饱和 |
| WebArena | Web (真实) | 最佳Agent | ~35% | 真实网站仍然极具挑战 |
| Mind2Web | Web (泛化) | 最佳方法 | ~60% (elem acc) | 跨网站泛化差 |
| AITW | Mobile | 最佳方法 | ~70% | 移动端相对成熟 |
| OSWorld | Desktop | GPT-4V Agent | ~12% | 桌面端最难 |

### 不同方法范式对比

| 方法范式 | 代表工作 | 优势 | 劣势 |
|---------|---------|------|------|
| Prompt-only | GPT-4V + ReAct | 零样本泛化 | 性能受限 |
| SFT | CogAgent, UGround | 特定任务强 | 泛化性差、数据昂贵 |
| RL-based | DigiRL, AppAgent | 持续改进 | 训练不稳定 |
| Hybrid (SFT+RL) | WebAgent, AgentQ | 综合最优 | 复杂度高 |

### 关键发现

- **真实环境 vs 简化环境差距巨大**：MiniWoB++ 上近95%的任务成功率到 WebArena 上骤降至 ~35%，OSWorld 上更是只有 ~12%。说明 GUI Agent 在真实复杂环境中仍远未可用。
- **视觉感知是主要瓶颈之一**：很多失败案例源于模型无法正确识别小按钮、下拉菜单或动态加载的界面元素。Set-of-Mark 等辅助标注方法能显著提升感知准确率。
- **长序列规划能力不足**：需要 10+ 步操作的复杂任务成功率远低于 3-5 步的简单任务。Error propagation 是核心问题——每步的小错误会累积。
- **跨平台泛化是开放问题**：大多数方法针对单一平台优化，在其他平台上性能大幅下降。统一的 GUI 表示和操作抽象是未来的关键方向。
- **数据是制约因素**：高质量的人类操作轨迹数据收集困难，合成数据的质量和多样性不足。自动化数据收集和轨迹过滤是重要的工程问题。

## 亮点与洞察

- **四维能力框架（感知-推理-规划-执行）**提供了一个清晰的分析视角，帮助定位不同工作的贡献和 GUI Agent 的整体能力瓶颈。这个框架也适用于分析其他类型的自主代理（如具身Agent、API Agent）。
- **对基准评测生态的系统梳理**非常有用——列出了各平台的主要基准、评估指标和已有结果，是快速进入该领域的最佳参考。
- **对训练范式的分类总结**（Prompt → SFT → RL → Hybrid）清晰勾勒了 GUI Agent 训练方法的发展脉络和未来方向。
- **论文发表后被引 18 次（截至 2026 年 4 月）**，说明综述在社区有一定影响力。

## 局限与展望

- **技术细节偏浅**：作为综述覆盖面广但深度有限，对各方法的具体实现细节和失败模式分析不够深入。
- **缺少定量的横向对比表格**：虽然梳理了各基准的结果但没有在统一条件下进行严格对比，读者难以判断方法间的真正差距。
- **多模态感知的讨论不够充分**：对视觉感知（OCR、布局理解、视觉定位）的挑战分析较少，但这恰恰是当前最大瓶颈。
- **安全性和可靠性**讨论不足：GUI Agent 自动操作用户设备带来的安全和隐私风险需要更多关注。
- **未来关键方向**：
    - 跨平台统一 GUI 表示
    - 自我纠错和安全边界机制
    - 高效的数据收集和标注方案
    - 与人类协作的混合自动化模式
    - 基于 RL 的持续在线学习

## 相关工作与启发

- **vs WebAgent/AutoGPT 等具体方法**: 本文不提出新方法而是系统总结现有工作，提供了全景式的领域视图，更适合作为研究入门和方向选择的参考。
- **vs 其他 Agent Survey (如 "A Survey on Large Language Model based Autonomous Agents")**: 那些综述关注通用 LLM Agent，而本文专注于 GUI 交互这个垂直领域，对 GUI 特有的视觉感知、界面建模等问题有更深入的覆盖。
- **vs 软件测试自动化**: 传统的 UI 自动化测试（如 Selenium）基于确定性脚本，而 GUI Agent 基于 AI 感知和推理，能处理动态变化的界面，但可靠性更低。两者可以互补。

## 评分

- 新颖性: ⭐⭐⭐ 作为综述无技术新颖性，但四维能力框架的提出有组织价值
- 实验充分度: ⭐⭐⭐ 综述不含原创实验，但对已有基准和结果的梳理较全面
- 写作质量: ⭐⭐⭐⭐ 组织结构清晰，分类体系合理，30位共同作者保证了广泛的领域视角
- 价值: ⭐⭐⭐⭐ 对 GUI Agent 领域的入门和研究方向选择有很高参考价值，系统性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_a_survey_on_mllm-based_agents_for_general_computing_devices_use.md)
- [\[CVPR 2025\] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](../../CVPR2025/llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)
- [\[ACL 2026\] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](../../ACL2026/llm_agent/from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)

</div>

<!-- RELATED:END -->
