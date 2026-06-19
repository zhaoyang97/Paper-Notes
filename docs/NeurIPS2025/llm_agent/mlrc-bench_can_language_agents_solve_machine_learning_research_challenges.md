---
title: >-
  [论文解读] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?
description: >-
  [NeurIPS 2025][LLM Agent][研究Agent] 本文提出MLRC-Bench，一个基于ML会议竞赛任务的动态benchmark，用于客观评估LLM agent提出和实现新研究方法的能力，发现最强agent（gemini-exp-1206）也仅缩小了baseline与人类顶级方案之间9.3%的差距，且LLM主观评分的"创新性"与实际效果之间几乎无相关性。
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "研究Agent"
  - "基准测试"
  - "ML竞赛"
  - "方法创新"
  - "LLM-as-Judge"
---

# MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?

**会议**: NeurIPS 2025  
**arXiv**: [2504.09702](https://arxiv.org/abs/2504.09702)  
**代码**: [HuggingFace](https://huggingface.co/spaces/launch/MLRC_Bench)  
**领域**: LLM Agent / AI for Science  
**关键词**: 研究Agent, 基准测试, ML竞赛, 方法创新, LLM-as-Judge

## 一句话总结

本文提出MLRC-Bench，一个基于ML会议竞赛任务的动态benchmark，用于客观评估LLM agent提出和实现新研究方法的能力，发现最强agent（gemini-exp-1206）也仅缩小了baseline与人类顶级方案之间9.3%的差距，且LLM主观评分的"创新性"与实际效果之间几乎无相关性。

## 研究背景与动机

**领域现状**：LLM research agent的评估主要有两个方向：一是类似AI Scientist的端到端科学发现（提idea→写代码→跑实验→写论文），但评估依赖LLM-as-Judge或人类评审，缺乏客观标准；二是类似MLE-Bench的Kaggle式ML工程竞赛，但这些竞赛很少要求真正的方法创新，只需调参或集成已有方法。

**现有痛点**：（1）End-to-end评估（AI Scientist）的主观性——LLM-as-Judge可能给出过于乐观的评价；（2）Kaggle式评估（MLE-Bench）的浅层性——不需要提出新方法；（3）现有benchmark多为single-file代码，与实际研究中的repository-level开发脱节；（4）很多benchmark缺少计算约束（runtime/GPU限制），无法鼓励高效方案。

**核心矛盾**：如何在一个framework中同时评估agent的"方法创新能力"和"客观性能"——既要求提出novel的方法，又要用可量化的指标衡量效果？

**本文目标** 构建一个客观、可扩展、面向前沿研究问题的agent benchmark，评估LLM agent能否提出并实现真正有效的新方法。

**切入角度**：ML会议竞赛天然具备评估两面性——开放性问题要求创新，公开leaderboard提供客观比较基准。直接利用这些竞赛作为benchmark任务。

**核心 idea**：把ML会议竞赛重构为agent-agnostic的标准化环境，用baseline到人类冠军方案的相对进步作为客观度量，同时分析LLM主观评分与客观表现的相关性。

## 方法详解

### 整体框架

MLRC-Bench包含7个ML竞赛任务，覆盖LLM合并、后门触发恢复、时序动作定位、降雨预测、机器遗忘、商品推荐、跨域元学习等领域。每个任务提供标准化的代码仓库、baseline方案、dev/test数据集和评估指标。Agent在methods/目录中修改代码实现新方法，在development集上迭代优化，最终在test集上评估。主指标为Relative Improvement to Human：$s'_{\text{agent}} = \frac{s_{\text{agent}} - s_{\text{baseline}}}{s_{\text{top\_human}} - s_{\text{baseline}}} \times 100\%$。

### 关键设计

1. **Repository-level代码框架**:

    - 功能：让agent在真实的研究项目结构中工作，而非single-file提交
    - 核心思路：每个竞赛被重构为标准化的项目结构，用`python main.py --method my_method --phase dev/test`启动。Agent只能修改methods/目录下的代码，评估脚本为只读。test集数据对agent不可见（开发阶段文件权限控制）
    - 设计动机：真实ML研究涉及多文件协作、依赖复用等，single-file模式过于简化。Repository-level也支持多agent协作（文献review agent、coding agent、评估agent分工）

2. **防过拟合的评估协议**:

    - 功能：防止agent在test集上过拟合
    - 核心思路：agent在试验中多次修改代码并在dev集上评估，系统在每次代码修改后保存快照。试验结束后，选择dev集上表现最好的那个快照，用该版本在test集上做最终评估。这严格遵循标准ML实践
    - 设计动机：仿照真实ML研究中的model selection流程，避免agent通过反复提交到test集来"刷分"

3. **客观+主观双重评估**:

    - 功能：量化评估agent方案的效果，并分析LLM主观评价的可靠性
    - 核心思路：客观指标包括效果（竞赛指标）、效率（runtime）、简洁性（LLoC）。主观指标由o1模型在5个维度上评分（有效性、清晰度、严谨性、泛化性、创新性），分为有代码和无代码两个设置。计算Spearman相关系数分析客观和主观指标的关联
    - 设计动机：验证AI Scientist等工作中LLM-as-Judge评估的可靠性，为评估方法论提供实证基础

### 损失函数 / 训练策略

不涉及模型训练。Agent使用MLAB框架（ReAct风格），单次试验限50步/5小时，每个配置8次试验取最好结果。

## 实验关键数据

### 主实验

不同LLM在MLAB框架下的Relative Improvement to Human（%）：

| Agent/LLM | 时序定位 | LLM合并 | 元学习 | 商品推荐 | 降雨预测 | 机器遗忘 | 后门触发 | **平均** |
|-----------|---------|---------|--------|---------|---------|---------|---------|----------|
| gemini-exp-1206 | -0.5 | 5.0 | -1.1 | 0.1 | 43.1 | 5.6 | 12.9 | **9.3** |
| llama-3.1-405b | 0.5 | -1.0 | -4.9 | 0.0 | 31.5 | 6.2 | 11.5 | 6.3 |
| o3-mini | 0.3 | -1.0 | -4.9 | 0.1 | 25.1 | 3.6 | 6.2 | 4.2 |
| claude-3.5-sonnet | 0.8 | 5.0 | -4.9 | 3.0 | 14.6 | -94.7 | 39.9 | -5.2 |
| gpt-4o | 0.3 | 2.0 | -4.9 | 0.6 | 47.5 | -18.0 | 10.4 | 5.4 |
| gpt-4o + Human Idea | 0.5 | -1.0 | -4.9 | 2.2 | 12.3 | 6.8 | 8.8 | 3.5 |
| gpt-4o + CoI Idea | 0.4 | -1.0 | -4.9 | 0.1 | 39.4 | 11.8 | 4.0 | 7.1 |

### 消融实验

| 分析 | 发现 |
|------|------|
| 提供AI生成idea | 不一定改善效果，有时反而更差 |
| 提供人类idea | 也不一定改善，实现能力是瓶颈 |
| 创新性 vs 效果 Spearman相关 | -0.06（几乎无关） |
| 迭代优化趋势 | 代码量和runtime持续增长，但性能增量递减 |
| 成本效益 | llama-3.1-405b性价比最高 |
| Pass@k扩展 | 高质量idea+多次尝试有帮助，但人类idea比AI idea更有效 |

### 关键发现

- **Agent能力严重不足**：最强agent仅缩小9.3%的baseline-human差距，大多数任务上agent甚至不如baseline
- **给idea没用，实现能力才是瓶颈**：提供人类甚至expert的idea也不能稳定提升性能，说明agent的"idea→code→调优"能力才是关键短板
- **LLM-as-Judge不可靠**：创新性评分与实际效果几乎零相关（-0.06），LLM评审可能给出过于乐观的评价
- **Agent过度优化**：随着迭代轮次增加，代码变复杂、runtime增长，但性能不成比例提升
- **Claude在机器遗忘上惨败（-94.7%）**：case study显示它把"遗忘"和"保持"作为独立目标分别优化，而未联合优化
- **降雨预测任务上偏高**：可能因为类似方案（U-Net变体）在网上广泛可得

## 亮点与洞察

- **Benchmark设计的三个巧思**：（1）用竞赛baseline和冠军方案做归一化，使跨任务比较成为可能；（2）Repository-level代码结构+文件权限控制，模拟真实研究场景；（3）动态更新机制——可持续加入新竞赛、淘汰饱和任务
- **对AI Scientist类工作的重要警示**：如果LLM-as-Judge的创新性评分与实际效果无关，那么完全依赖LLM评审的研究agent评估（如AI Scientist）可能大幅高估了agent的研究能力
- **Agent错误模式的insight**：11.5%的步骤因工具参数错误失败（幻觉参数名），且仅17.2%的代码执行错误能被agent自修复，揭示了LLM在复杂codebase中的脆弱性

## 局限与展望

- **仅7个任务**：数量较少，可能不具代表性。但作者强调质量优先于数量
- **计算成本高**：每个配置8次试验×5模型×3框架设置 = 大量API费用，限制了更多模型和更多试验的探索
- **baseline可能偏弱**：某些任务（如后门触发）的baseline本身很弱，agent比baseline好不代表有真正的方法创新
- **Agent框架有限**：主要测试MLAB，其他框架（AIDE、SELA等）因single-file假设不兼容
- **未测试多agent协作**：Repository-level设计支持多agent分工，但实验中未探索

## 相关工作与启发

- **vs AI Scientist**：AI Scientist做端到端（idea→paper），用LLM/人评审。MLRC-Bench聚焦method proposal+implementation，用客观指标。两者互补
- **vs MLE-Bench / MLAgentBench**：它们用Kaggle式任务，不要求方法创新，single-file提交。MLRC-Bench要求novel methods，repository-level代码
- **vs RE-Bench**：RE-Bench也要求ML研究能力，但任务多集中在语言模型和CIFAR-10等旧领域，由专家手工策划难以更新。MLRC-Bench直接从竞赛sourcing，更易持续更新

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地用ML竞赛评估research agent的方法创新能力
- 实验充分度: ⭐⭐⭐⭐ 5个LLM、3个框架设置、多维评估，分析全面深入
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，实验设计严谨
- 价值: ⭐⭐⭐⭐⭐ 对LLM research agent领域有重要的benchmark和方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](../../ACL2025/llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ICLR 2026\] CoMind: Towards Community-Driven Agents for Machine Learning Engineering](../../ICLR2026/llm_agent/comind_towards_community-driven_agents_for_machine_learning_engineering.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](../../ACL2025/llm_agent/repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)
- [\[NeurIPS 2025\] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)
- [\[NeurIPS 2025\] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments](defenderbench_a_toolkit_for_evaluating_language_agents_in_cybersecurity_environm.md)

</div>

<!-- RELATED:END -->
