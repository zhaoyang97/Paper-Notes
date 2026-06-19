---
title: >-
  [论文解读] AutoGLM: Autonomous Foundation Agents for GUIs
description: >-
  [AAAI 2026][LLM Agent][GUI Agent] AutoGLM 基于 ChatGLM 构建了面向 Web 浏览器和 Android 手机的 GUI 基础智能体，通过中间接口设计分离规划与定位行为，并提出自进化在线课程强化学习框架，在 VAB-WebArena-Lite 上达到 55.2% 成功率，大幅超越 GPT-4o 的 18.2%。
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "强化学习"
  - "中间接口设计"
  - "自进化课程学习"
  - "基础智能体"
---

# AutoGLM: Autonomous Foundation Agents for GUIs

**会议**: AAAI 2026  
**arXiv**: [2411.00820](https://arxiv.org/abs/2411.00820)  
**代码**: [https://github.com/THUDM/AutoGLM](https://github.com/THUDM/AutoGLM)  
**领域**: Agent  
**关键词**: GUI Agent, 强化学习, 中间接口设计, 自进化课程学习, 基础智能体

## 一句话总结
AutoGLM 基于 ChatGLM 构建了面向 Web 浏览器和 Android 手机的 GUI 基础智能体，通过中间接口设计分离规划与定位行为，并提出自进化在线课程强化学习框架，在 VAB-WebArena-Lite 上达到 55.2% 成功率，大幅超越 GPT-4o 的 18.2%。

## 研究背景与动机
1. **领域现状**：基础模型（LLM/LMM）在知识获取和语言理解方面表现出色，但在动态真实环境中的决策能力仍然不足。GUI Agent 作为连接基础模型与数字设备的桥梁，近年来受到广泛关注，但仍处于初期阶段。
2. **现有痛点**：现有 GUI Agent 方法面临三个核心问题：（1）互联网预训练数据中缺乏决策相关的动态知识，导致模型难以学会与环境交互；（2）高质量的专家轨迹数据极度稀缺且采集成本高；（3）端到端方法将规划（planning）和定位（grounding）耦合在一起，导致两者都难以单独优化。
3. **核心矛盾**：规划和定位对模型的要求截然不同——规划需要灵活性和错误恢复能力，而定位需要精准的动作执行。将二者耦合训练会相互干扰，且纯行为克隆无法让智能体学会从错误中恢复。
4. **本文要解决什么？**（1）如何设计合适的接口让规划和定位解耦？（2）如何在数据极度稀缺的情况下通过 RL 持续提升智能体能力？（3）如何构建可部署到真实用户的基础智能体系统？
5. **切入角度**：作者观察到现有 LLM/LMM 在规划方面的能力远优于定位——大部分错误来自于元素识别错误（如点击了错误的坐标），而非规划方向错误。这一观察启发了将两者分离的思路。
6. **核心idea一句话**：通过中间接口设计将 GUI Agent 的规划与定位解耦并分别优化，再通过自进化在线课程强化学习让智能体在真实环境中自主学习和进化。

## 方法详解

### 整体框架
AutoGLM 的整体 pipeline 由三大部分组成：（1）**中间接口设计**：将端到端的 Agent 行为拆分为 Planner（生成语义化动作描述）和 Grounder（将描述转化为精确坐标）两个模块；（2）**自进化在线课程强化学习**（WebRL）：通过 actor-critic 框架进行在线 RL，解决任务数据稀缺和策略分布漂移问题；（3）**部署基础设施**：针对 Web 浏览器和 Android 两种场景构建可交付给用户的实际系统。

### 关键设计

1. **中间接口设计（Intermediate Interface Design）**:
    - 做什么：将传统的端到端动作格式 `do(action="Click", element_coordinates=[823,684])` 改为分离式格式——Planner 生成 `do(action="Click", element_description="the 'Submit' button on the bottom right")`，Grounder 再将描述映射为坐标。
    - 核心思路：Planner 只需关注"做什么"（语义层面），Grounder 只需关注"在哪做"（视觉定位层面），两者可以独立训练和优化。Grounder 的训练数据可以从无监督环境观察中大量自动构建，远比端到端的专家轨迹容易获取。
    - 设计动机：实验表明（Table 1），仅仅通过这种解耦，GPT-4o 在 VAB-WebArena-Lite 上的成功率就从 18.2% 提升至 27.3%（+9.1%），GPT-4-vision-preview 更是从 18.8% 提升至 36.4%（+17.6%），说明定位错误确实是性能瓶颈。

2. **自进化在线课程强化学习（Self-Evolving Online Curriculum RL / WebRL）**:
    - 做什么：在仅有约 1000 条行为克隆数据的基础上，通过在线 RL 持续提升 Planner 能力，解决数据稀缺和策略分布漂移两大难题。
    - 核心思路：（1）**任务数据自进化**：利用约 1000 条 BC 数据初始化模型到 22.4% 成功率后，在在线 roll-out 过程中对失败任务指令进行变异（增加难度或降低难度），通过 critic 筛选后用于下一轮训练；（2）**KL 约束策略更新**：为防止课程学习中的策略分布漂移，引入 KL 散度约束保持策略更新的稳定性；（3）**Actor 置信度过滤经验回放**：只利用 actor 有较高置信度的经验进行回放，提升训练效率。
    - 设计动机：传统 BC 只能模仿专家的逐步行为，无法理解目标本身，更无法学会从错误中恢复。在线 RL 能让模型从失败中学习，但面临环境模拟效率低和采样多样性不足的挑战。课程学习渐进式增加难度可缓解这些问题。

3. **奖励建模（Reward Modeling）**:
    - 做什么：构建通用的奖励模型为在线 RL 提供监督信号，而非依赖特定任务的规则奖励函数。
    - 核心思路：区分 outcome-supervised ORM（结果监督）和 process-supervised PRM（过程监督），为开放世界的通用任务提供不同粒度的评估反馈。
    - 设计动机：基础智能体的目标是完成广泛的真实世界任务，特定任务的奖励函数无法覆盖所有场景。

### 训练策略
训练采用渐进式课程：先用预训练引入弱监督决策信号 → 高分辨率视觉 SoM 提示增强感知 → BC 初始化基本能力 → 在线课程 RL（WebRL）逐步提升规划能力。整个流程形成"弱到强"的渐进式增强。

### 部署架构
- **Web 端**：通过 Chrome/Edge 浏览器插件（智谱清言）对外提供服务，Planner 作为后端服务，Grounder 在浏览器侧执行元素定位。
- **Android 端**：通过 AccessibilityService 实现对物理手机的自主控制，而非传统的 Android 虚拟设备（AVD），更贴近真实用户场景。
- **二次尝试机制**：当首次执行失败时，AutoGLM 可自动检测失败状态并发起重试，这种错误恢复能力是 RL 训练的直接产物。

## 实验关键数据

### 主实验

| 基准测试 | 指标 | AutoGLM | GPT-4o | Claude-3.5-Sonnet | 提升 |
|----------|------|---------|--------|-------------------|------|
| VAB-WebArena-Lite | 成功率 | 55.2% | 18.2% | - | +37.0% |
| VAB-WebArena-Lite (二次尝试) | 成功率 | 59.1% | - | - | +3.9% |
| OpenTable Eval | 成功率 | 96.2% | 62.6% | - | +33.6% |
| AndroidLab (VAB-Mobile) | 成功率 | 36.2% | 31.2% | 29.0% | +5.0% |
| 中文APP (人工评估) | 成功率 | 89.7% | - | - | - |

### 消融实验（中间接口设计效果）

| 配置 | GPT-4o (text) | GPT-4o (visual) | GPT-4-vision (visual) |
|------|---------------|-----------------|----------------------|
| End-to-End Agent | 14.3% | 18.2% | 18.8% |
| + Intermediate Interface | 18.1% (+3.8%) | 27.3% (+9.1%) | 36.4% (+17.6%) |

### 关键发现
- 中间接口设计对视觉输入的模型提升更为显著（GPT-4-vision 提升 17.6%），说明视觉定位是当前 GUI Agent 的最大瓶颈。
- 自进化在线课程 RL 从仅 1000 条 BC 数据的 22.4% 起步，最终将模型推至 55.2%，验证了 RL 在数据稀缺场景下的巨大潜力。
- KL 约束和置信度过滤经验回放是防止训练崩溃的关键设计，缺一不可。
- 在真实中文 APP 场景下，AutoGLM 达到 89.7% 成功率，即使未完全成功的任务也大多完成了部分操作（Partial），仍有实用价值。
- 允许二次尝试时成功率从 55.2% 提升至 59.1%，说明错误恢复机制确实有效。
- OpenTable 评估使用的是真实网站交互，而非模拟环境，96.2% 的成功率充分说明了 AutoGLM 在实际场景中的可靠性。
- Agent Q 的 MCTS 方法在 OpenTable 上仅达到 81.7%，表明课程 RL + 中间接口的组合方案在实践中优于纯搜索策略。

## 亮点与洞察
- **规划与定位解耦的简洁有效性**：仅通过改变动作表示格式就能带来巨大提升，这个设计极其简洁但效果显著。其核心洞察是"不同能力需要不同的优化策略"，这种思想可迁移到其他复合型任务。
- **数据自进化机制**：在数据极度稀缺（仅 1000 条）的情况下，通过在线 roll-out 中的指令变异实现数据自我扩充，巧妙解决了没有足够用户任务指令的冷启动问题。
- **面向部署的设计哲学**：论文不仅关注基准测试，还将系统真正部署为可供用户使用的浏览器插件和 Android 应用，这种从研究到产品的闭环在学术论文中较为罕见。

## 局限性 / 可改进方向
- **多模态感知能力有限**：当前主要依赖 SoM 提示和文本描述，对复杂视觉布局（如重叠元素、动态加载页面）的理解能力可能不足。
- **仅在特定基准环境中训练**：WebRL 主要在 WebArena 环境中进行，迁移到其他 Web 环境或全新 APP 时的泛化能力存疑。
- **评估覆盖面有限**：Android 评估仅覆盖 7 个常用中文 APP，对小众应用或英文应用的表现未知。
- **安全与隐私考量不足**：论文未深入讨论自主智能体操作真实设备时的安全风险，如误操作导致的财务损失或隐私泄露。
- **RL 训练的可复现性**：在线课程 RL 涉及大量环境交互和自进化指令生成，训练过程的随机性较大，其他团队可能难以精确复现相同结果。
- **对长 horizon 任务的处理**：论文中展示的任务大多在 10-20 步内完成，对于需要几十步甚至上百步的复杂任务，当前方案的表现未知。

## 相关工作与启发
- **vs Agent Q**：Agent Q 同样利用 RL 训练 Web Agent 并在 OpenTable 上评估，但 AutoGLM 通过中间接口设计和自进化课程 RL 取得了更好效果（96.2% vs 81.7%）。Agent Q 的 MCTS 搜索策略与 AutoGLM 的课程 RL 代表了两种不同的探索策略，未来可考虑结合。
- **vs DigiRL**：DigiRL 同样提出用在线 RL 训练 Device-control Agent，但其课程策略是从固定指令集中按当前能力筛选任务，而 AutoGLM 的自进化方法可以自主生成新的训练指令，数据利用效率更高。
- **vs CogAgent**：CogAgent 强调高分辨率视觉输入的重要性，AutoGLM 继承了这一发现并进一步通过中间接口设计将视觉感知与决策规划解耦。

## 评分
- 新颖性: ⭐⭐⭐⭐ 中间接口设计和自进化课程RL各自不算全新，但组合在一起形成了有效的系统方案
- 实验充分度: ⭐⭐⭐⭐ 覆盖Web和Android双场景，有真实部署验证，但消融实验偏少
- 写作质量: ⭐⭐⭐⭐ 工业级系统论文写得清晰有条理，但部分训练细节因篇幅限制较为粗略
- 价值: ⭐⭐⭐⭐⭐ 首个真正部署到用户的基础GUI Agent系统，对该领域有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents](co-epg_a_framework_for_co-evolution_of_planning_and_groundin.md)
- [\[AAAI 2026\] Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design](physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)
- [\[ICML 2025\] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](../../ICML2025/llm_agent/aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)
- [\[ICLR 2026\] WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](../../ICLR2026/llm_agent/weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](../../ACL2026/llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)

</div>

<!-- RELATED:END -->
