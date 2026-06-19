---
title: >-
  [论文解读] The PokeAgent Challenge: Competitive and Long-Context Learning at Scale
description: >-
  [NeurIPS 2025][多智能体][游戏AI基准] 提出 PokéAgent Challenge，一个基于宝可梦对战和RPG速通的双赛道大规模AI基准，通过NeurIPS 2025竞赛验证了专家RL方法远超通用LLM方法，并揭示宝可梦对战衡量的能力与现有49个LLM基准近乎正交。 领域现状： 部分可观测性、博弈论推理和…
tags:
  - "NeurIPS 2025"
  - "多智能体"
  - "游戏AI基准"
  - "宝可梦对战"
  - "长期规划"
  - "强化学习"
  - "LLM Agent"
---

# The PokeAgent Challenge: Competitive and Long-Context Learning at Scale

**会议**: NeurIPS 2025  
**arXiv**: [2603.15563](https://arxiv.org/abs/2603.15563)  
**代码**: [https://pokeagentchallenge.com](https://pokeagentchallenge.com)  
**领域**: LLM效率  
**关键词**: 游戏AI基准, 宝可梦对战, 长期规划, 强化学习, LLM Agent

## 一句话总结

提出 PokéAgent Challenge，一个基于宝可梦对战和RPG速通的双赛道大规模AI基准，通过NeurIPS 2025竞赛验证了专家RL方法远超通用LLM方法，并揭示宝可梦对战衡量的能力与现有49个LLM基准近乎正交。

## 研究背景与动机

**领域现状**: 部分可观测性、博弈论推理和长期规划是序列决策的核心挑战，但现有基准往往只孤立地测试其中一个维度——不完全信息博弈侧重短回合均衡计算，开放环境测试探索但缺乏对抗性对手。宝可梦同时涉及所有三个维度：对战中需要在隐藏信息下对抗战略对手推理，单人战役则需数千步的探索、资源管理和战斗决策。

**现有痛点**: 2025年宝可梦引起AI界广泛关注（Claude Plays Pokémon、Gemini完成蓝版、GPT-5完成红版），但这些尝试各自为政——不同游戏版本、不同harness、不同评估标准，使得有意义的比较变得不可能。无法区分成功归因于Agent架构、底层模型还是简化感知的硬编码假设。

**核心矛盾**: 宝可梦具有约10^564种可能对战状态，1000+物种的队伍构建，持续演变的竞技元游戏，但缺乏标准化的评估基础设施来公平比较RL、LLM和混合方法。同时，宝可梦知识虽大量存在于预训练语料中（招式属性、伤害公式、竞技分级），但将这些潜在知识转化为部分可观测下的多轮序列决策与简单的检索任务有本质区别。

**本文目标**: 建立统一的宝可梦AI评估框架，提供标准化基线、大规模数据集和公开排行榜，使不同范式（RL vs. LLM vs. 混合）可以在相同条件下公平比较。

**切入角度**: 设计双赛道基准——对战赛道（Pokémon Showdown上的竞技对战）测试博弈推理和泛化能力，速通赛道（Pokémon Emerald RPG速通）测试长期规划和序列决策。通过NeurIPS 2025竞赛验证基准质量并推动社区参与。

**核心 idea**: 将宝可梦的对战系统和RPG环境标准化为一个活的基准(living benchmark)，同时评估RL和LLM在高风险竞技和长期决策中的能力差距。

## 方法详解

### 整体框架

PokéAgent Challenge 包含两个互补赛道：

- **对战赛道 (Competitive Battling Track)**: 基于Pokémon Showdown开源模拟器，评估部分可观测下两人零和博弈中的战略推理。每回合从约9个动作中选择，对战持续20-100回合。
- **速通赛道 (Speedrunning Track)**: 将RPG游戏形式化为周期MDP M=(S,A,T,R,γ)，动作为按钮输入，代理接收视觉帧和有限状态信息（队伍组成、等级、HP值），但谜题状态、动态障碍和招式集不暴露。

### 关键设计

#### 对战赛道评估体系
- **功能**: 建立多维度的技能评分系统和标准化对战环境。
- **核心思路**: 使用Full-History Bradley-Terry (FH-BT)评分作为主要指标，辅以Glicko-1和GXE。对战在专用修改版Showdown服务器上进行，避免干扰人类玩家。
- **设计动机**: Showdown的在线评分（Elo）为大规模人类玩家设计，在AI代理小池子、密集匹配、固定策略场景下噪声过大。FH-BT基于完整历史的Bradley-Terry模型配合bootstrap不确定性，更适合此场景。

#### 大规模数据集
- **功能**: 提供目前最大的公开宝可梦对战数据集。
- **核心思路**: 包含4M+人类演示RL轨迹（从旁观者视角重放推断私有信息，重建每位玩家视角的对战）、18M+合成对战轨迹、200K+精选竞技队伍。
- **设计动机**: 人类演示对策略引导至关重要，但竞技性能通常需要自我对弈的规模。原始"回放"从旁观者视角记录，不反映每位玩家决策时的私有信息，因此需要推断和重建。

#### LLM基线框架
- **功能**: 将PokéChamp扩展为通用的reasoning model harness框架。
- **核心思路**: 将游戏状态转换为结构化文本，提供可配置harness（包括深度限制的minimax搜索+LLM评估）。支持前沿API模型（GPT/Claude/Gemini）和开源模型（Llama/Gemma/Qwen）。
- **设计动机**: 即使小型开源模型在harness支持下也能获得有意义的性能。默认回合计时器（60-90秒）对LLM推理不足，因此提供Extended Timer设置。

#### RL基线
- **功能**: 扩展Metamon，发布30个跨越竞技技能阶梯的预训练Agent。
- **核心思路**: 覆盖从紧凑RNN到200M参数Transformer的多个规模点。基于大规模人类演示和自我对弈训练。
- **设计动机**: 提供跨多个人类技能水平的高质量参考点，使研究者能在可及硬件上探索算力-性能权衡。

#### 速通赛道：多智能体编排系统
- **功能**: 发布首个开源的长期RPG游玩多智能体编排系统。
- **核心思路**: 协调MCP工具（A*寻路、按钮输入、知识检索）与专门子Agent（战斗策略、自我反思、道馆谜题、目标验证）。中央编排器维护高层路线规划，根据游戏上下文动态派遣子Agent，自动上下文压缩以管理数千推理步骤。
- **设计动机**: 原始前沿VLM在此赛道上实现0%完成度——宝可梦RPG是OOD任务，需要harness作为取得任何进展的前提条件。Harness × Model 分析框架从五个维度（状态表示S、工具T、记忆M、反馈F、微调Φ）解耦系统性能。

### 损失函数 / 训练策略

- **RL基线训练**: 基于大规模离线RL，使用人类演示(4M轨迹)引导策略，再通过自我对弈(18M轨迹)提升竞技水平。
- **竞赛优胜方案**: 对战赛道冠军FoulPlay使用根并行MCTS搜索；速通冠军Heatz采用Scripted Policy Distillation (SPD)——LLM分解任务生成脚本策略，通过模仿学习蒸馏到神经网络后用RL精化。
- **评估协议**: 对战使用FH-BT评分+bootstrap不确定性，要求最低样本量。速通评估完成度百分比（15个里程碑）和完成时间，同时报告墙钟时间和步数以区分推理速度和样本效率。

## 实验关键数据

### 主实验

**对战赛道：Agent对人类的Showdown天梯表现**

| 方法类别 | 代表Agent | Gen 1 OU | Gen 9 OU | 备注 |
|---------|----------|----------|----------|------|
| RL基线 (新) | Metamon-Transformer-200M | 接近Top 500人类 | 接近Top 500人类 | 最强基线，但仍未超人类 |
| RL基线 (旧) | Prior Metamon | 中等水平 | 中等水平 | 此次显著改进 |
| LLM基线 | GPT-4o + minimax harness | 低于RL | 低于RL | Extended Timer下表现改善 |
| LLM基线 | PC-Llama3.1-8B | 有意义但偏低 | 有意义但偏低 | 小模型+harness也有效 |
| 人类 | Top 500 Showdown | 参考上界 | 参考上界 | 当前AI上界仍低于超人水平 |

**速通赛道：首个道馆完成情况**

| 方法 | 完成时间 | 步数 | 类型 |
|------|---------|------|------|
| 人类速通最佳 | 18分钟 | - | 人类 |
| 人类平均 | 1:22:05 | - | 人类 |
| Heatz (冠军, SPD) | 40:13 | 1,608 | RL蒸馏 |
| Hamburg (亚军, PPO) | ~1:20 | - | RL |
| Deepest (评委选择) | 排名第5 | 649 (最少) | LLM harness |
| Gemini 3 Flash (组织方) | ~2:24 | - | LLM harness |
| Claude Sonnet 4.5 (组织方) | 6:25-20:45 (高方差) | - | LLM harness |
| anthonys (最佳纯LLM) | 1:29:17 | - | LLM harness |

### 消融实验

**BenchPress正交性分析：宝可梦 vs. 49个标准LLM基准**

| 分析维度 | 指标 | 结果 |
|---------|------|------|
| 与最佳基准的Spearman相关 | max ρ | 0.77 |
| 与所有基准的平均相关 | mean \|ρ\| | 0.45 |
| 标准基准Rank-2 SVD解释率 | 方差占比 | 91% |
| GXE的Rank-2 SVD解释率 | 方差占比 | 仅27% |

**竞赛参与统计**

| 指标 | 数量 |
|------|------|
| 参赛队伍 | 100+ |
| 比赛场次 | 100K+ |
| Discord成员 | 650+ |
| 有效提交 | 150+ |
| 速通完成全部里程碑 | 6/22队 |
| 对战资格赛：RL/Search vs. LLM | 16席中13席为RL扩展，3席为独立RL/MCTS |

### 关键发现

1. **专家方法碾压通用LLM**: 两个赛道均显示一致模式——RL和搜索方法全面超越LLM。对战资格赛16个席位中无LLM方法入围。速通冠军Heatz(RL蒸馏)比最佳纯LLM快2×以上。
2. **LLM作为先验，RL作为精化**: Heatz的成功模式——LLM分解任务和生成初始策略，RL蒸馏和精化执行——提供了一种有潜力的混合范式。
3. **宝可梦暴露标准基准遗漏的推理失败**: 弱模型出现"恐慌行为"——小战术失误后不断犯更多错误。不同模型家族有不同的失败模式：记忆损坏级联、目标振荡、过度计划承诺、计算麻痹。这些在编程/数学基准中不会出现。
4. **宝可梦对战与标准LLM基准正交**: BenchPress矩阵中83个模型×49个基准近似秩为2，5个基准足以预测其余44个（误差~7分）。但宝可梦GXE打破了这种低秩结构，Rank-2 SVD仅能解释27%方差。
5. **Harness是必要条件而非可选优化**: 原始前沿VLM在速通赛道实现0%完成度。常见CLI-Agent架构（Claude Code、Codex CLI、Gemini CLI）也无法维持数千步序列决策的一致性。

## 亮点与洞察

- **框架格局大**: 同时覆盖对抗博弈(对战)和长期规划(速通)两个维度，提供统一基础设施，这在现有游戏AI基准中独一无二。
- **数据规模空前**: 20M+对战轨迹、200K+队伍数据集，远超此前任何宝可梦AI工作。
- **正交性论证有力**: 通过BenchPress矩阵严格证明宝可梦衡量的能力与现有评估套件正交，不是简单声称"这里有gap"，而是定量地证明了。
- **Harness × Model解耦框架**: 首次系统化地分离模型能力和外部工具链的贡献，解决了之前各路演示无法比较的核心问题。
- **活的基准设计**: 从竞赛过渡到持续运行的排行榜，竞技元游戏的自然演化确保基准不会饱和。

## 局限与展望

1. **速通赛道仅覆盖第一个道馆**: 尽管合理（降低成本），但完整游戏的评估将更全面地测试长期记忆和规划。
2. **最强RL基线仍未超人类**: 当前上界低于人类顶尖水平，留有提升空间但也说明基准确实足够难。
3. **视觉感知未充分测试**: 对战赛道使用符号状态表示而非视觉输入，速通赛道虽有视觉帧但受限于简单像素级输入。
4. **开源模型基线不足**: 无开源模型完成完整RPG，限制了更广泛研究社区的参与。
5. **四个开放挑战值得后续**: VLM-SLAM（视觉空间定位）、LLM-RL差距弥合、开源模型全游戏通关、逼近人类速通时间。

## 相关工作与启发

- **与FightLadder/Hanabi的关系**: 这些专注不完全信息竞争评估，但不结合长期规划。PokéAgent同时覆盖两者。
- **与NetHack/BALROG的关系**: 测试长期推理，但主要单人、非对抗。宝可梦增加了对抗维度。
- **与MineRL/Neural MMO的关系**: NeurIPS竞赛先例，但各自只推进一个研究轴。PokéAgent的双赛道设计联合评估。
- **LLM as prior + RL as refinement范式**: Heatz的成功与近期"LLM指导RL"研究方向一致，可能推广到机器人和其他决策领域。
- **对Benchmark设计的启发**: BenchPress正交性分析方法可用于评估任何新基准是否真正测量了既有基准未覆盖的能力。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 双赛道设计+living benchmark理念新颖，正交性论证独到，但核心组件(PokéChamp/Metamon)是已有工作的扩展
- 实验充分度: ⭐⭐⭐⭐⭐ — 100+参赛队、100K+对战、多维度分析(BenchPress正交性、harness解耦、跨赛道洞察)，实验极为充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表信息量大，但论文偏长，竞赛细节占比较多
- 价值: ⭐⭐⭐⭐⭐ — 填补了游戏AI基准的重要空白，正交性发现对LLM评估社区有广泛影响，living benchmark可持续驱动研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CoOT: Learning to Coordinate In-Context with Coordination Transformers](../../ICML2026/multi_agent/coot_learning_to_coordinate_in-context_with_coordination_transformers.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](multi-agent_collaboration_via_evolving_orchestration.md)
- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/multi_agent/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ACL 2025\] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](../../ACL2025/multi_agent/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)

</div>

<!-- RELATED:END -->
