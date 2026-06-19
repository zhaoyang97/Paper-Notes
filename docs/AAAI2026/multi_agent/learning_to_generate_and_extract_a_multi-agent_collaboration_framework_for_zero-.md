---
title: >-
  [论文解读] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction
description: >-
  [AAAI 2026][多智能体][零样本学习] 提出"提议-评估-修改"多智能体协作框架（生成智能体+评估智能体）解决零样本文档级事件论元提取（ZS-DEAE），通过生成智能体合成未见事件的训练数据，评估智能体评分引导强化学习迭代优化，同时提升合成数据质量和抽取性能。 问题定义 文档级事件论元提取（DEAE）：给定文档 $…
tags:
  - "AAAI 2026"
  - "多智能体"
  - "零样本学习"
  - "文档级事件论元提取"
  - "多智能体协作"
  - "强化学习"
  - "合成数据"
---

# Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction

**会议**: AAAI 2026  
**arXiv**: [2603.02909](https://arxiv.org/abs/2603.02909)  
**代码**: [github](https://github.com/GJZhang2866/GenExtract)  
**领域**: 信息抽取 / 事件论元提取  
**关键词**: 零样本学习, 文档级事件论元提取, 多智能体协作, 强化学习, 合成数据

## 一句话总结

提出"提议-评估-修改"多智能体协作框架（生成智能体+评估智能体）解决零样本文档级事件论元提取（ZS-DEAE），通过生成智能体合成未见事件的训练数据，评估智能体评分引导强化学习迭代优化，同时提升合成数据质量和抽取性能。

## 研究背景与动机

### 问题定义

文档级事件论元提取（DEAE）：给定文档 $d$、事件类型 $e$、触发词 $t$、预定义角色集 $R_e$，从 $d$ 中提取每个角色对应的论元文本跨度。零样本 DEAE（ZS-DEAE）：训练集仅包含已见事件类型 $E_s$ 的标注数据，测试时需在未见事件类型 $E_u$ 上提取论元，且 $E_s \cap E_u = \emptyset$。

### 现有方法局限

**知识迁移方法**：基于共享语义空间或原型网络的迁移学习，在文档级零样本场景中远落后于有监督模型

**LLM 直接应用**：GPT-4o/DeepSeek R1 等在 ZS-DEAE 上效果很差（F1约10-24%），因严格边界匹配（Span-F1）要求

**合成数据质量问题**：
   - 仅用事件类型名称提示 LLM 生成的数据难以捕获未见事件的上下文和结构关系
   - 生成内容往往句式简单、论元集中、语境不丰富（如图1所示 GPT-4o 的生成示例）
   - 缺乏质量评估机制，噪声样本会降低下游性能

### 核心动机 —— 示例分析

"inspect people organization"（审查人/组织）和"physical investigate inspect"（物理观察检查）两个事件类型都涉及"inspect"动作。前者强调审查人或组织，后者强调通过感知或仪器进行物理观察。但 LLM 生成的示例常无法清晰区分这些事件类型，且生成的上下文语言简单、论元密集，缺乏DEAE所需的上下文丰富性和结构复杂性。

## 方法详解

### 整体框架

模拟人类协作认知过程"提议-评估-修改"（Propose-Evaluate-Revise）的三阶段循环：

1. **提议（Propose）**：生成智能体根据未见事件类型和角色集合成文档级上下文、触发词和角色-论元对
2. **评估（Evaluate）**：评估智能体从合成数据中提取论元并评估语义一致性，输出基于对数似然的质量分数
3. **修改（Revise）**：将评估结果转化为奖励信号，通过策略梯度优化两个智能体

### 关键设计

#### 1. **生成智能体**

基于 LLaMA-3.1-8B 或 Qwen-2.5-7B，使用 LoRA 微调（rank=8, scaling=32, dropout=0.05）。

输入提示：$E_{\text{in}}(e, R_e)$ = "Given the event type: $e$ and the following roles: $R_e$, please generate a coherent context..."

输出格式：$E_{\text{out}}(d, t, A_e)$ = "Context: $d$, Trigger: $t$, Role-Arguments: $A_e$"

训练目标（自回归）：

$$\mathcal{L}_g = -\log P(E_{\text{out}} | E_{\text{in}})$$

设计动机：利用 LLM 在已见事件上学到的知识迁移到未见事件类型的数据生成，以提示格式标准化输入输出。

#### 2. **评估智能体**

采用 Bart-Gen（基于 BART-large 的条件生成模型），输入文档 $d$ 和带 `<arg>` 占位符的模板句，输出填充后的模板。

质量评分基于对数似然：

$$\ell_i = \log P(x_i | c_i)$$

归一化分数：

$$\alpha_i = \frac{\ell_i - \mu}{\delta}$$

其中 $\mu, \delta$ 是整个合成数据集 $D_{\text{syn}}$ 上对数似然的均值和标准差。

设计动机：高质量样本的论元填充句应与文档上下文高度语义一致，因此对数似然可作为质量指标。

#### 3. **事件结构约束**

观察到生成智能体倾向产生大量空论元（角色无对应论元标为 None）。评估智能体对正确预测 None 给出高分，形成累积偏差的反馈循环。

解决方案：计算空论元比例 $\rho_i$ 的惩罚项：

$$p_i = \begin{cases} 0 & \text{if } \tau - \varepsilon \leq \rho_i \leq \tau + \varepsilon \\ |\rho_i - \tau| & \text{otherwise} \end{cases}$$

其中 $\tau, \varepsilon$ 是训练集中空论元比例的期望和标准差。最终分数：

$$\alpha_i = \frac{\ell_i - \mu}{\delta} - p_i$$

设计动机：将空论元比例控制在训练数据分布范围内，避免生成结构不完整的事件实例。

#### 4. **强化学习优化（Revise）**

将归一化分数 $\alpha_i$ 作为奖励，通过策略梯度法同时优化两个智能体：

$$\mathcal{G}_{i+1} = \mathcal{G}_i + \gamma_1 \nabla_{\mathcal{G}} \mathbb{E}[\alpha]$$

$$\mathcal{E}_{i+1} = \mathcal{E}_i + \gamma_2 \nabla_{\mathcal{E}} \mathbb{E}[\alpha]$$

梯度计算：

$$\nabla_{\mathcal{G}} \mathbb{E}[\alpha] = \mathbb{E}[\alpha_i \nabla_{\mathcal{G}} \log P(E_{\text{out}} | E_{\text{in}})]$$

$$\nabla_{\mathcal{E}} \mathbb{E}[\alpha] = \mathbb{E}[\alpha_i \nabla_{\mathcal{E}} \log P(x | c)]$$

设计动机：RL 无需显式标注即可学习数据质量偏好，高奖励引导两个智能体更好理解未见事件类型。

### 训练策略

- 生成智能体先在 $D_s$ 上用自回归目标预训练
- 评估智能体先在 $D_s$ 上用标注数据训练
- 五轮智能体交互优化，三个随机种子，报告最佳轮次的平均结果
- 评价指标：Span-F1（精确跨度匹配）

## 实验关键数据

### 实验设置

- 数据集：RAMS 和 WikiEvents，构建三个零样本设置：RAMS2RAMS、RAMS2Wiki、Wiki2Wiki
- 生成智能体：LLaMA-3.1-8B / Qwen-2.5-7B（LoRA微调）
- 评估智能体：Bart-large
- 基线：6个 DEAE 模型、4个零样本模型、7个 LLM（含 CoT）

### 主实验

| 方法 | RAMS2RAMS | RAMS2Wiki | Wiki2Wiki |
|------|-----------|-----------|-----------|
| TabEAE | 36.22 | 26.74 | 30.97 |
| DEEIA | 37.95 | 5.12 | 22.51 |
| Bart-Gen | 38.53 | 28.52 | 40.82 |
| GPT-4o | 19.64 | 10.17 | 11.36 |
| DS-R1 | 24.41 | 12.27 | 12.84 |
| DS-R1+CoT | 23.47 | 9.24 | 10.40 |
| **Ours (LLaMA)** | **45.77** | **32.38** | 46.96 |
| **Ours (Qwen)** | 44.59 | 31.18 | **47.62** |

### 消融实验

| 配置 | R2R | R2W | W2W | 说明 |
|------|-----|-----|-----|------|
| Ours (LLaMA) | 45.28 | 32.38 | 46.96 | 完整模型 |
| -reward | 42.46 | 24.93 | 38.39 | 去除RL奖励，下降显著 |
| -constraint | 44.72 | 29.03 | 40.98 | 去除结构约束，下降明显 |
| Ours (Qwen) | 44.59 | 31.18 | 47.62 | 完整模型 |
| -reward | 40.52 | 27.19 | 43.30 | 去除RL奖励 |
| -constraint | 40.87 | 27.87 | 38.70 | 去除结构约束 |

### 合成数据增强其他模型

| 模型 | 仅已见数据 | +LLaMA合成 | +Ours合成 |
|------|-----------|-----------|----------|
| TabEAE (R2R) | 36.22 | 37.37 | **44.43** |
| Bart-Gen (R2R) | 38.53 | 41.04 | **46.06** |
| TabEAE (W2W) | 30.97 | 12.48 | **33.36** |
| Bart-Gen (W2W) | 40.82 | 44.17 | **46.96** |

### 关键发现

1. **大幅超越现有方法**：在三个零样本设置中，Ours(LLaMA) 比最强基线 Bart-Gen 分别提升 7.53/3.86/6.14 F1
2. **LLM 直接应用效果差**：GPT-4o/DS-R1 等在 Span-F1 上表现远不如经过训练的小模型，因边界匹配困难
3. **RL 奖励是主要因素**：去除 RL 奖励在所有设置上导致最大性能下降
4. **结构约束有效**：去除结构约束增加空论元比例，损害数据质量
5. **合成数据可迁移**：增强其他 DEAE 模型效果显著，LLaMA 直接生成的数据有时反而有害（TabEAE W2W: 12.48）
6. **交互轮次有限**：1-2轮达到最佳，更多轮次导致多样性下降和性能衰退

## 亮点与洞察

1. **"提议-评估-修改"范式**：模拟人类协作认知过程，生成方与评估方在博弈中共同进步
2. **结构约束的巧妙设计**：发现并解决"空论元偏差"问题——评估智能体对 None 预测给高分导致的反馈循环
3. **合成数据的可迁移性**：生成的数据不仅用于自身训练，还能增强其他 DEAE 模型，证明数据质量确实提升
4. **小模型胜过大模型**：8B 模型通过多智能体协作框架显著超越 70B LLaMA、GPT-4o、DeepSeek R1
5. **多样性分析深入**：从词汇、语义、逻辑、句法四维分析合成数据多样性随轮次的变化

## 局限与展望

1. **多样性衰减**：更多交互轮次导致词汇/语义/句法多样性下降，模型收敛到高似然模式
2. **单一评估维度**：评估智能体仅基于对数似然评分，缺乏对事实准确性、逻辑一致性的显式评估
3. **数据集规模有限**：RAMS 和 WikiEvents 规模较小，更大规模数据集上的表现未知
4. **事件类型重叠依赖**：零样本设置允许角色共享以维护事件模式，完全不相交的角色设置未测试
5. **生成智能体与评估智能体的能力不对等**：生成用 8B LLM，评估用 BART-large，规模差异可能影响协作动态

## 相关工作与启发

- **RAMS/WikiEvents**：文档级事件论元提取的标准数据集
- **Bart-Gen (Li 2021)**：条件生成框架用于事件提取，本文的评估智能体基础
- **VIPER/PAIE/TabEAE**：各类事件提取方法的基线
- **RL for NLP**：使用强化学习优化文本生成质量
- 对零样本信息抽取的启发：多智能体协作可能比单模型直接推理更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多智能体协作+RL+结构约束的组合新颖，"空论元偏差"发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个设置、多基线（含7个LLM）、消融、迁移性测试、多样性分析全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，案例分析生动，理论与实践结合好
- 价值: ⭐⭐⭐⭐ — 零样本DEAE的新思路，合成数据可迁移性有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection](../../CVPR2026/multi_agent/agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete.md)
- [\[CVPR 2026\] Visual Document Understanding and Reasoning: A Multi-Agent Collaboration Framework with Agent-Wise Adaptive Test-Time Scaling](../../CVPR2026/multi_agent/visual_document_understanding_and_reasoning_a_multi-agent_collaboration_framewor.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)
- [\[ICML 2025\] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](../../ICML2025/multi_agent/cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)

</div>

<!-- RELATED:END -->
