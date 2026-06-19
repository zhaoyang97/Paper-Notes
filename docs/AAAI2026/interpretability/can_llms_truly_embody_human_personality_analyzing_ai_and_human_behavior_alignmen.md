---
title: >-
  [论文解读] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution
description: >-
  [AAAI 2026][可解释性][LLM人格模拟] 提出首个系统对比框架，在配对的冲突调解场景中直接比较人类与人格提示LLM的策略行为差异，发现LLM在人格-行为映射上与人类存在显著偏差，挑战了"人格提示即可代理人类行为"的假设。 LLM越来越多地被用于模拟高风险社交场景（法律调解、谈判、争端解决），但一个核心问题尚未得到…
tags:
  - "AAAI 2026"
  - "可解释性"
  - "LLM人格模拟"
  - "冲突解决"
  - "大五人格"
  - "行为对齐"
  - "社会仿真"
---

# Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution

**会议**: AAAI 2026  
**arXiv**: [2602.07414](https://arxiv.org/abs/2602.07414)  
**代码**: [有](https://github.com/DSincerity/Personality-LLM-BehavAlign-Dispute)  
**领域**: 可解释性  
**关键词**: LLM人格模拟, 冲突解决, 大五人格, 行为对齐, 社会仿真

## 一句话总结

提出首个系统对比框架，在配对的冲突调解场景中直接比较人类与人格提示LLM的策略行为差异，发现LLM在人格-行为映射上与人类存在显著偏差，挑战了"人格提示即可代理人类行为"的假设。

## 研究背景与动机

LLM越来越多地被用于模拟高风险社交场景（法律调解、谈判、争端解决），但一个核心问题尚未得到验证：当用Big Five Inventory（BFI）人格特质提示LLM时，它们能否重现人类中人格驱动的行为差异？

人类人格（如宜人性、神经质、外向性）已被大量研究证实会系统性地影响冲突解决策略（如合作、竞争、回避），但现有研究存在两个关键空白：(1) 多数人格-冲突研究依赖静态问卷而非动态对话行为观察；(2) 研究要么只看人类、要么只看LLM，缺乏在匹配场景下的直接人机行为对比。本文正是填补这一空白，通过构建平行的人类-LLM对话数据集来系统检验行为对齐程度。

## 方法详解

### 整体框架

本文提出一个**评估框架**（非新模型），包含三个核心组件：

1. **人类基准数据集 KODIS**：来自Prolific平台的众包角色扮演争端解决对话，248组人-人对话，角色为"买家"和"卖家"围绕退款、差评删除、正式道歉三个议题进行谈判
2. **LLM-to-LLM (L2L) 仿真数据集构建**：用匹配的场景和人格特质驱动LLM对话
3. **可解释行为度量体系**：基于IRP（利益-权利-权力）框架的策略行为和冲突结果量化指标

### 关键设计

**LLM人格配置**：每个LLM被分配BFI人格向量 {P_AGR, P_EXT, P_CON, P_NEU, P_OPE}，采用六级极性-程度量表。为保证公平比较，人格分布从人类BFI经验分布中采样。使用70对双极形容词（Goldberg, 1992）生成人格提示，每个特质3个形容词，通过"very/a bit/无修饰词"表达强度。

**议题重要性个性化**：根据人类数据的回归结果（B=2.13, p=.02），道歉议题的重要性与宜人性挂钩。其他议题随机分配重要性值。

**行为度量体系**（基于IRP框架）：

- **最终结果指标**：得分（Score）、是否接受（Accept）、是否留在谈判（Not Walk-Away）
- **策略行为指标**：
    - IRP比率：合作/竞争策略使用频率 $\text{IRP}_{\text{ratio}}^{X} = \frac{N_S^X}{N_S^{\text{all}}}$
    - IRP互惠：对方使用策略X后己方跟随X的比例 $\text{IRP}_{\text{recip}}^{X} = \frac{N_S^{X=X_P}}{N_P^X}$
    - 升级比率：对非竞争性发言的竞争性回应频率
    - 降级比率：对竞争性发言的非竞争性回应频率

**实验模型**：GPT-4o mini（500次仿真）、Claude Sonnet 3.7（250次）、Gemini 2.0 Flash（250次），均用默认参数（temperature=1）。

### 损失函数 / 训练策略

本文为评估框架，不涉及模型训练。分析方法为：对连续型因变量用线性回归，二值型用逻辑回归。自变量包括LLM自身和对手的BFI特质，位置（买家/卖家）作为控制变量，使用效应编码（Buyer=–1, Seller=1）。

## 实验关键数据

### 主实验（人格对结果和策略行为的影响）

**表2：人格特质对谈判结果的回归分析**

| 因变量 | GPT-4 显著IV | Claude 显著IV | Gemini 显著IV | KODIS(人类) |
|--------|-------------|--------------|--------------|-------------|
| Score | S-EXT(B=1.67**), S-AGR(B=-4.38***) | S-AGR(B=-2.50***), P-EXT(B=-1.42*) | S-AGR(B=-4.48***), S-CON(B=1.72*) | 仅位置效应 |
| Accept | POS(B=-0.22*) | S-EXT(B=-0.17**), P-EXT(B=0.17**) | — | S-NEU(B=-0.26*), P-NEU(B=0.27*) |
| Not Walk-Away | S-OPE(B=-0.18*), P-OPE(B=0.18*) | — | S-NEU(B=-0.18*), P-NEU(B=0.18*) | 无显著人格效应 |

**表3：IRP策略行为回归分析（部分）**

| 指标 | GPT-4 | Gemini | Claude | KODIS |
|------|-------|--------|--------|-------|
| 合作比率 | P-EXT(B=0.49*) | 无 | 无 | 无 |
| 竞争比率 | S-EXT, S-NEU, P-EXT, P-NEU多个显著 | S-EXT, P-EXT显著 | 仅POS | S-EXT, P-EXT, P-CON, S-OPE |
| 升级比率 | S-AGR(B=-1.37**), P-EXT(B=1.56**) | 无（与KODIS对齐） | S-AGR(B=-2.45***) | 无 |
| 降级比率 | S-AGR(B=1.83***) | S-EXT, P-EXT显著 | 无（与KODIS对齐） | 无 |

### 消融实验

**IRP策略分布热图**：人类最依赖"事实陈述"且分布最均衡；LLM偏好"提案"和"让步"，呈程式化交易风格。Claude最接近人类（更多事实），Gemini最偏斜（高"权力"+"残余"），GPT-4最平衡但一直偏好"权利"。

**时间动态分析**：人类策略随对话阶段动态演变（早期事实→中期利益和提案→后期让步），LLM呈扁平轨迹，策略组合随时间变化极小。Claude部分模仿了人类的动态模式。

### 关键发现

1. **人格-行为映射根本性分歧**：人类中神经质是策略结果最强预测因子，而LLM中外向性和宜人性效应更强
2. **跨LLM差异**：Claude和Gemini在策略指标上比GPT-4o mini更接近人类
3. **LLM互惠模式极化**：LLM更一致地互惠合作策略，人类则更灵活
4. **GPT-4o mini更易升级冲突**；Claude强烈偏向降级；人类在升级-降级间更平衡

## 亮点与洞察

- 首个在配对冲突场景下直接对比人类和多个LLM的人格-行为关系研究
- 提出了可泛化的评估框架和数据集构建方法论
- 揭示了一个重要的"不能"：人格提示LLM在社会影响场景中不能作为可靠的人类行为代理
- IRP框架在LLM评估中的创新应用

## 局限与展望

- IRP标注由LLM完成，未做全面人工验证
- 未评估提示措辞/指令变体的鲁棒性
- 仅用BFI五因素模型，情商、马基雅维利主义等其他维度可能提供额外解释力
- 单一场景（球衣纠纷），泛化性有待验证
- 未包含开源LLM或更新版本的闭源模型

## 相关工作与启发

- **人格-冲突映射**：Wood & Bell (2008) 证实外向性和宜人性预测合作型冲突风格，本文拓展到动态对话行为
- **LLM社会行为仿真**：Park et al. (2022), Zhou et al. (2023) 的多智能体社会模拟
- **启发**：可将此框架拓展到其他社交心理学构念（如信任、权力动态），或用于LLM微调的心理对齐目标

## 评分

- **新颖性**: ★★★★☆ — 首次系统性对比人类与LLM的人格-冲突行为对齐
- **技术深度**: ★★★☆☆ — 框架设计严谨但无新模型/算法贡献
- **实验充分度**: ★★★★☆ — 3个LLM + 人类基准，多维度回归分析
- **实用价值**: ★★★★☆ — 对LLM社会仿真应用有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[ACL 2026\] Dual Alignment Between Language Model Layers and Human Sentence Processing](../../ACL2026/interpretability/dual_alignment_between_language_model_layers_and_human_sentence_processing.md)
- [\[CVPR 2025\] Towards Human-Understandable Multi-Dimensional Concept Discovery](../../CVPR2025/interpretability/towards_human-understandable_multi-dimensional_concept_discovery.md)
- [\[ACL 2026\] Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs](../../ACL2026/interpretability/flattery_in_motion_benchmarking_and_analyzing_sycophancy_in_video-llms.md)
- [\[ACL 2026\] A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification](../../ACL2026/interpretability/a_systematic_comparison_between_extractive_self-explanations_and_human_rationale.md)

</div>

<!-- RELATED:END -->
