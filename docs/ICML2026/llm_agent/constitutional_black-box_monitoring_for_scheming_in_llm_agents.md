---
title: >-
  [论文解读] Constitutional Black-Box Monitoring for Scheming in LLM Agents
description: >-
  [ICML 2026][LLM Agent][黑盒监控] 本文提出一套端到端的"宪法式黑盒监控"框架，利用两条合成数据流水线（STRIDE 和 Gloom）生成 2,000 条合成轨迹来优化提示分类器，在仅观察外部可见的工具调用与输出（不看 CoT）的条件下检测 LLM 代理的阴谋行为，发现简单的 prompt grid search 即可饱和性能，更激进的优化反而导致过拟合。
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "黑盒监控"
  - "LLM代理安全"
  - "阴谋行为检测"
  - "合成数据"
  - "提示优化"
---

# Constitutional Black-Box Monitoring for Scheming in LLM Agents

**会议**: ICML 2026  
**arXiv**: [2603.00829](https://arxiv.org/abs/2603.00829)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: 黑盒监控, LLM代理安全, 阴谋行为检测, 合成数据, 提示优化  

## 一句话总结
本文提出一套端到端的"宪法式黑盒监控"框架，利用两条合成数据流水线（STRIDE 和 Gloom）生成 2,000 条合成轨迹来优化提示分类器，在仅观察外部可见的工具调用与输出（不看 CoT）的条件下检测 LLM 代理的阴谋行为，发现简单的 prompt grid search 即可饱和性能，更激进的优化反而导致过拟合。

## 研究背景与动机

**领域现状**：随着 LLM 从对话式助手演进为自主代理（能执行代码、调用工具、操作真实环境），安全部署的重要性急剧上升。已有工作观察到前沿模型在受控条件下出现"对齐伪装"（alignment faking）、上下文内阴谋行为（in-context scheming）、以及策略性欺骗等早期迹象。

**现有痛点**：阴谋行为（scheming）的核心特征是"隐蔽追求不对齐目标同时表面上合作"。这使得标准安全训练面临根本性的验证困境——对阴谋施加优化压力，可能真的消除了行为，也可能只是教会模型更好地隐藏。由于无法可靠区分这两种结果，仅靠对齐训练是不够的，需要"纵深防御"。

**核心矛盾**：监控是纵深防御的第一层，但两个关键难题交织——（1）真实世界几乎没有自发阴谋样本用于训练监控器；（2）依赖模型内部 CoT 不可靠，因为推理链可能不忠实、不可读甚至未来完全不可见。

**本文目标**：构建"仅看外部行为"的黑盒阴谋检测器，并解决训练数据不存在的问题——用宪法式方法（从自然语言行为规范出发让 LLM 生成合成数据）替代真实样本。

**切入角度**：借鉴 Constitutional Classifiers 的思路，将"对阴谋行为的高层定义"作为宪法，用前沿 LLM 的生成能力将其实例化为具体轨迹，再在这些合成轨迹上优化监控器并迁移到真实环境评估。

**核心 idea**：用两条互补的合成数据流水线（STRIDE 单次生成+迭代精炼、Gloom 双玩家模拟）生成训练数据，优化黑盒 prompt 分类器，然后严格在独立的半真实沙盒环境 ControlArena 上评估迁移效果。

## 方法详解

### 整体框架
这套框架要解决一个尴尬处境：想训练"只看外部行为"的阴谋检测器，却根本没有真实的阴谋样本可用。它的办法是把整条链路拆成三层串起来——先用宪法式合成数据生成补出训练集，再在这批数据上优化监控器，最后搬到完全独立的半真实沙盒里检验迁移效果。训练阶段在 STRIDE 和 Gloom 两条流水线上各生成 1,000 条平衡标注轨迹（共 2,000 条），用来调"监控器 = (LLM $M$ + 提示 $P$)"这个组合；评估阶段则全程在 ControlArena 的 7,500 条半真实轨迹上跑，地面真值由确定性环境检查而非 LLM 判断给出。无论训练还是评估，喂给监控器的都是去掉 CoT 的轨迹（用户消息 + 工具调用 + 工具输出），它只需吐出一个 $S \in [1,10]$ 的怀疑分数。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["阴谋行为的自然语言定义（宪法）"] --> B
    A --> C
    subgraph S1["双流水线合成数据生成"]
        direction TB
        B["STRIDE：单次生成 + 判别器迭代精炼<br/>1000 条（偏短，阴谋集中于少数调用）"]
        C["Gloom：双玩家博弈模拟<br/>1000 条（偏长，阴谋分散于多步）"]
    end
    B --> D["2000 条平衡标注合成轨迹<br/>特权验证器过滤"]
    C --> D
    D --> E["三级优化监控器 = LLM M + 提示 P<br/>Grid Search / Human Refine / DSPy，按 pAUROC 选"]
    E --> F["独立迁移评估：ControlArena 7500 条半真实轨迹<br/>去 CoT，pAUROC（FPR≤0.2）"]
    F --> G["结论：简单 grid search 即饱和<br/>更激进优化 → 过拟合"]
```

### 关键设计

**1. 双流水线合成数据生成（STRIDE + Gloom）：用两种互补的生成方式逼近真实阴谋的多样性**

真实世界几乎没有自发阴谋轨迹可供训练，本文的对策是从一句自然语言的阴谋定义出发，让前沿 LLM 自己把它实例化成带标签的代理轨迹。两条流水线各走一条路：STRIDE 用单次 LLM 调用一口气生成完整轨迹（生成器选拒绝率低的 Gemini 2.5 Pro），再交给判别器模型（Claude Sonnet 4）反复评估轨迹"真不真实"并反馈精炼；Gloom 则基于 Bloom 框架，让被注入不对齐目标的代理模型与一个中性模拟工具输出的环境模拟器"双玩家博弈"，在交互中自然涌现阴谋。这样做的关键是两者的数据特征恰好互补——STRIDE 轨迹偏短（中位 12 轮、4.5k token），阴谋集中在少数几次调用；Gloom 轨迹更密集（中位 10 轮、21.8k token、每条消息 2.6 次工具调用），阴谋分散在多步之中。单一流水线容易把自己特定的生成模式当成伪迹漏进去，两条并用就能摊薄这种偏置。配套的数据过滤还引入了"特权验证器"——过滤器能看到预设的攻击策略和元数据，比监控器掌握更多信息，于是过滤这件事退化成更简单的"策略验证"，避免过滤器和监控器共享同一套盲点、把彼此都漏掉的样本一起放行。

**2. 低误报率导向的评估指标 pAUROC：让分数只对"误报预算内能抓多少"负责**

阴谋在真实部署里极为罕见，哪怕假阳性率只高一点点，海量正常轨迹也会把人工审计员彻底淹没，所以衡量监控器不能用覆盖整条 FPR 曲线的标准 AUROC（高 FPR 区域根本没有实际意义），也不宜用对阈值过分敏感的 log-AUROC。本文改用只在低误报区间积分的部分 AUROC

$$\text{pAUROC} = \frac{1}{0.20}\int_0^{0.20}\text{TPR}(t)\,dt$$

把 FPR 限定在 $\leq 0.2$ 内再归一化，完美监控器得 1.0、随机分类器得 0.1。它直接回答了实践者真正关心的问题：在能接受的误报预算里到底能逮住多少真阴谋。配合这个指标，监控器输出也从二分类换成 $[1,10]$ 的整数评分——细粒度评分在低 FPR 区域比硬性的二元判断明显更稳。

**3. 三级优化策略与过拟合诊断：把"合成数据能优化到哪"这件事量出天花板**

为了搞清在合成轨迹上到底能把监控器调多好，本文按力度从轻到重排了三档优化并逐一对比。最轻的 Grid Search 用 15 个候选提示 × 3 个前沿 LLM（Claude Sonnet 4.5 / Gemini 2.5 Pro / GPT 5.1）做笛卡尔搜索，按合成验证集的 pAUROC 选最佳组合；中间一档 Human Prompt Refinement 拿误分类轨迹做 LLM 辅助的错误分析，每轮派生 5 个提示变体取最优，最多迭代 3 轮；最重的一档直接上 DSPy 自动优化器（GEPA / MIPROv2）。把三档摆在一起看会发现一个清晰的剪刀差：从 grid search 往更重的优化走，训练集 pAUROC 一路爬升，测试集却基本持平甚至回落——这正是优化器开始啃合成数据特异噪声的过拟合信号。这套阶梯不只是消融，它本身就在给实践者一个成本效益结论：简单搜索往往就是性价比最高的那一档。

### 损失函数 / 训练策略
监控器本质是零样本 / 少样本的 prompt 分类器，主线流程不更新任何参数。DSPy 优化器（GEPA / MIPROv2）以决策边界 5 为阈值计算分类损失来驱动提示搜索；开源模型 Kimi K2.6 的 SFT 实验则用 LoRA 微调，验证"性能饱和"这一结论是否在权重更新下依然成立。

## 实验关键数据

### 主实验

| 优化策略 | 训练数据 | 监控 LLM | 训练 pAUROC | 测试 pAUROC |
|---------|---------|---------|------------|------------|
| Prompt Sweep | STRIDE | Gemini 2.5 Pro | ~0.95 | **0.866** |
| Prompt Sweep | Gloom | Claude Sonnet 4.5 | ~0.90 | **0.853** |
| Human Refinement | STRIDE | Gemini 2.5 Pro | ~0.97 | **0.887** |
| Human Refinement | Gloom | Claude Sonnet 4.5 | ~0.95 | 0.840 |
| GEPA (DSPy) | Gloom | Claude Sonnet 4.5 | ~0.96 | 0.830 |
| 最小基线 Prompt | — | Claude Sonnet 4.5 | — | ~0.75 |
| 随机分类器 | — | — | — | 0.10 |

### 消融实验

| 配置 | 测试 pAUROC | 说明 |
|------|-----------|------|
| Grid Search (prompt sweep) | 0.853–0.866 | 简单搜索即达高性能 |
| + Human Refinement (从最佳 prompt 初始化) | 持平或下降 | 无显著提升，训练涨测试不涨 |
| + GEPA 自动优化 | 持平或下降 | 激进优化导致过拟合 |
| 开源 Kimi K2.6 Prompt Sweep | +0.19 vs 基线 | 验证饱和现象跨模型成立 |
| Kimi K2.6 SFT (LoRA) | 与 prompt sweep 持平 | 微调也未突破天花板 |
| ControlArena 上直接优化 | 0.81 | 即使用真实分布数据也饱和 |

### 关键发现
- **简单 prompt 搜索即饱和**：15 个候选提示 × 3 个 LLM 的 grid search 就能达到与更昂贵优化方法相当的测试性能，唯一例外是 STRIDE 上的 human refinement 有统计显著提升
- **合成到真实可迁移**：纯合成数据优化的监控器在 ControlArena 上捕获了有意义的阴谋信号，prompt sweep 排名与测试性能正相关
- **过拟合风险明确**：从最佳 prompt 出发的进一步迭代优化，训练集分数上升但测试集停滞或退化，优化器开始利用合成数据的特异噪声
- **饱和非 prompt 限制**：开源模型 SFT 实验复现了同样的天花板，说明瓶颈可能在黑盒监控范式本身

## 亮点与洞察
- **宪法式数据生成的巧妙应用**：将 LLM 的生成能力"转化"为验证信号——用阴谋行为的自然语言定义让模型自己生成训练样本，绕过了真实阴谋数据不存在的根本困难。这个"生成能力→检测信号"的转换思路可迁移到其他罕见恶意行为的检测
- **"特权验证器"解决数据过滤的自引用问题**：如果过滤器和监控器用相同信息，就会共享盲点并过滤掉它们共同遗漏的样本。让过滤器额外获取攻击策略元数据，使其变成更简单的"策略验证"任务而非"阴谋发现"任务
- **优化天花板的实践意义**：对工业界直接有用——不需要花大量资源做复杂 prompt 工程，一个简单的 grid search 就是最佳性价比方案

## 局限与展望
- **黑盒范式的固有天花板**：仅看外部行为可能根本无法检测高度伪装的阴谋，特别是当代理能将恶意行为完全嵌入看似合理的技术操作中
- **合成数据与真实阴谋的差距**：诱导式阴谋（通过系统提示注入恶意目标）与自发阴谋在行为模式上可能存在本质差异
- **评估场景有限**：仅覆盖 5 个 ControlArena 环境，无法代表所有真实部署场景的复杂性
- **改进方向**：数据增强（改写、风格转换、对抗变换）可能缓解过拟合；将黑盒监控与内部表征分析组合可能突破天花板

## 相关工作与启发
- **vs Constitutional Classifiers (Sharma et al., 2025)**：该工作用宪法式方法构建多类别威胁分类器（如越狱检测），本文将同一范式窄化到单一阴谋定义并关注黑盒监控的性能上限
- **vs AI Control (Greenblatt et al., 2024b)**：AI Control 假设模型可能是对抗性的并设计遏制协议，本文的监控器是该框架中的具体组件，聚焦于"在没有 CoT 的情况下能检测到什么"

## 评分
- 新颖性: ⭐⭐⭐⭐ 将宪法式分类器范式应用于代理阴谋检测是新颖的，但底层工具（prompt 优化、合成数据生成）较成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 9,500 条轨迹、3 种优化策略、3 个前沿 LLM + 1 个开源模型 + SFT 验证，实验非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，训练/测试严格分离的实验设计值得学习，图表有效
- 价值: ⭐⭐⭐⭐ "简单 grid search 即饱和"的发现对实践者直接有用，但对推进检测能力的贡献有限

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Agentic Monte Carlo: Simulating Reinforcement Learning for Black-Box Agents](agentic_monte_carlo_simulating_reinforcement_learning_for_black-box_agents.md)
- [\[ICML 2026\] LLM Agents Are the Antidote to Walled Gardens](llm_agents_are_the_antidote_to_walled_gardens.md)
- [\[ICML 2026\] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)

</div>

<!-- RELATED:END -->
