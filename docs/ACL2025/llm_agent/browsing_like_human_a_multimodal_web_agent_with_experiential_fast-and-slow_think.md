---
title: >-
  [论文解读] Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking
description: >-
  [ACL 2025][LLM Agent][Web Agent] 本文提出WebExperT框架，模拟人类"快思考与慢思考"的规划模式，并通过从失败中反思的经验学习机制不断改进决策，在Mind2Web基准上取得了监督和无监督设置下的优异表现。 领域现状：自动化网页导航任务要求Agent根据用户指令（如"预订一张从北京到上海的…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Web Agent"
  - "快思考慢思考"
  - "经验学习"
  - "多模态"
  - "网页导航"
---

# Browsing Like Human: A Multimodal Web Agent with Experiential Fast-and-Slow Thinking

**会议**: ACL 2025  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-long.697/)
**代码**: 无  
**领域**: LLM Agent / Web导航 / 多模态推理  
**关键词**: Web Agent, 快思考慢思考, 经验学习, 多模态, 网页导航

## 一句话总结

本文提出WebExperT框架，模拟人类"快思考与慢思考"的规划模式，并通过从失败中反思的经验学习机制不断改进决策，在Mind2Web基准上取得了监督和无监督设置下的优异表现。

## 研究背景与动机

**领域现状**：自动化网页导航任务要求Agent根据用户指令（如"预订一张从北京到上海的机票"）在真实网站上完成复杂交互操作。近年来，基于LLM和多模态模型的Web Agent受到广泛关注，现有方法通常具备视觉感知、规划和记忆能力。

**现有痛点**：尽管现有Web Agent在技术层面不断进步，但它们的推理过程仍然偏离人类的认知模式。具体而言：(1) 面对复杂任务时缺乏系统性的任务分解策略；(2) 遇到失败后无法有效学习和调整，重复犯相同错误；(3) 对简单操作和复杂操作采用统一的处理方式，效率低下。

**核心矛盾**：人类在浏览网页时会根据任务复杂度自适应地切换思考模式——面对熟悉的简单操作（如点击按钮）快速执行，面对复杂决策（如多步骤表单填写）则深入思考。现有Agent缺乏这种自适应能力。

**本文目标**：设计一个更贴近人类认知的Web Agent框架，具备(1)自适应的快慢思考切换能力和(2)从失败中持续学习的经验积累能力。

**切入角度**：受Daniel Kahneman的"思考快与慢"理论启发，将任务规划分为快思考（System 1，处理常规操作）和慢思考（System 2，处理复杂决策），并引入经验学习模块积累执行经验。

**核心 idea**：用双系统思考模型（快/慢）来分解和执行Web导航任务，同时通过反思失败经验不断优化规划和决策。

## 方法详解

### 整体框架

WebExperT的输入是用户自然语言指令和当前网页的视觉截图，输出是一系列交互动作序列（点击、输入、选择等）。整个框架分为三个核心部分：快思考模块（Fast Thinking）、慢思考模块（Slow Thinking）和经验学习模块（Experiential Learning）。

### 关键设计

1. **快思考模块（Fast Thinking / System 1）**:

    - 功能：快速处理常规、简单的网页操作
    - 核心思路：维护一个经验库（experience pool），存储过去成功执行的操作模式。当面对新的网页状态时，先通过相似度检索查找匹配的历史经验。如果找到高置信度的匹配，直接复用历史操作策略，无需深度推理。类似于人类对熟悉操作的"肌肉记忆"
    - 设计动机：大量Web操作是重复性的（如"点击确认按钮"、"选择日期"），对这些操作进行深度推理是浪费计算资源。快思考模块可以显著提升效率

2. **慢思考模块（Slow Thinking / System 2）**:

    - 功能：对复杂的多步骤子任务进行深入规划和推理
    - 核心思路：当快思考模块无法找到匹配经验时，触发慢思考模块。该模块使用多模态LLM（如GPT-4V）对当前网页截图和用户指令进行深入分析，将复杂任务分解为子目标序列，并为每个子目标生成详细的执行计划。规划过程考虑网页的结构化信息（DOM元素）和视觉布局
    - 设计动机：复杂交互（如跨多页面的预订流程）需要全局规划能力，单步贪心策略容易陷入错误路径

3. **经验学习模块（Experiential Learning）**:

    - 功能：从执行结果中学习，特别是从失败中反思
    - 核心思路：每次任务执行后，无论成功还是失败，都将完整的（状态、动作、结果）轨迹存入经验库。对于失败的轨迹，使用LLM进行失败原因分析，生成"教训"（lesson learned）标签。下次遇到类似场景时，经验库不仅提供成功范例，还提供失败警示，帮助Agent避免重蹈覆辙
    - 设计动机：人类从失败中学习的能力是持续进步的关键。现有Agent通常只存储成功经验，忽略了失败经验的巨大价值

### 训练策略

WebExperT在监督设置下使用Mind2Web的标注数据微调多模态模型，在无监督设置下通过自我博弈（self-play）生成训练数据。经验库随着交互次数增加不断扩充。

## 实验关键数据

### 主实验

| 测试集 | 指标 | WebExperT | MindAct | SeeAct | 提升 |
|--------|------|-----------|---------|--------|------|
| Mind2Web-Cross-Task | Element Acc | 显著领先 | 基线 | 基线 | ~5-8% |
| Mind2Web-Cross-Website | Element Acc | 显著领先 | 基线 | 基线 | ~4-7% |
| Mind2Web-Cross-Domain | Element Acc | 最优 | 基线 | 基线 | ~3-6% |
| 监督设置总体 | Step Success Rate | 最优 | - | - | 显著 |
| 无监督设置总体 | Task Completion | 显著提升 | - | - | 明显 |

### 消融实验

| 配置 | Step Acc | 说明 |
|------|---------|------|
| Full WebExperT | 最优 | 完整模型 |
| w/o Fast Thinking | 下降明显 | 失去快速决策能力，效率降低 |
| w/o Slow Thinking | 大幅下降 | 复杂任务无法有效分解 |
| w/o Experiential Learning | 中等下降 | 重复犯错，无法持续改进 |
| w/o Failure Reflection | 轻微下降 | 证明失败经验确实有价值 |

### 关键发现
- 慢思考模块贡献最大，说明任务分解和深度规划是Web导航的核心能力
- 快思考模块在效率上的提升很显著——对于重复性任务，推理速度提升约2-3倍
- 从失败中学习（Failure Reflection）在跨网站和跨领域场景中效果更明显，因为新场景更容易犯错
- 在无监督设置下，WebExperT的经验积累机制使其随交互次数增加而持续提升

## 亮点与洞察
- **双系统思考框架**是一个优雅的Agent设计范式——将Kahneman的认知理论与AI Agent结合，既有理论深度又有实践效果。这种思路可以迁移到其他需要自适应决策的Agent任务
- **失败经验的显式利用**是本文的另一个亮点——大多数Agent只存储成功经验，忽略了失败经验的价值。将失败轨迹结构化为"教训"并用于未来决策是一个可复用的策略
- 经验库的设计使得Agent具有"记忆"和"成长"特性，比无状态的LLM调用更接近真实的人类行为

## 局限与展望
- Mind2Web是静态网页基准，真实网页的动态变化（弹窗、异步加载等）未被覆盖
- 经验库的检索依赖网页状态的相似度计算，对界面变化较大的场景效果可能受限
- 快慢思考的切换策略较为规则化（基于检索命中率），未来可以学习自适应的切换策略
- 未在真实浏览器环境（如WebArena）中进行验证

## 相关工作与启发
- **vs MindAct**: MindAct使用单一推理流程处理所有操作，WebExperT的双系统设计在处理异质化操作时更灵活
- **vs SeeAct**: SeeAct侧重视觉grounding，WebExperT在此基础上增加了规划和学习维度
- **vs Reflexion**: Reflexion也利用反思来改进Agent，但专注于一般推理任务；WebExperT将反思机制与Web导航的快慢思考框架结合

## 评分
- 新颖性: ⭐⭐⭐⭐ 快慢思考+经验学习的组合在Web Agent领域较为新颖
- 实验充分度: ⭐⭐⭐⭐ Mind2Web上多维度评测，消融实验清晰
- 写作质量: ⭐⭐⭐⭐ 动机阐述清楚，框架图直观
- 价值: ⭐⭐⭐⭐ 对Web Agent的认知启发式设计提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)
- [\[CVPR 2026\] iSHIFT: Lightweight Slow-Fast GUI Agent with Adaptive Perception](../../CVPR2026/llm_agent/ishift_lightweight_slow-fast_gui_agent_with_adaptive_perception.md)
- [\[ACL 2025\] Caution for the Environment: Multimodal LLM Agents are Susceptible to Environmental Distractions](caution_environment_gui_agent_distractions.md)
- [\[ACL 2025\] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)

</div>

<!-- RELATED:END -->
