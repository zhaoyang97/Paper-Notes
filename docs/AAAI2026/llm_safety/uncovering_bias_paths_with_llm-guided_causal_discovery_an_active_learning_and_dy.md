---
title: >-
  [论文解读] Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach
description: >-
  [AAAI2026][LLM安全][因果发现] 提出一种融合LLM语义先验与统计信号的混合因果发现框架，通过主动学习（Active Learning）和动态评分机制优先查询信息量最大的变量对，在噪声和混淆条件下有效恢复公平性关键因果路径（如 sex→education→income），显著优于传统CD方法和朴素LLM方法。
tags:
  - "AAAI2026"
  - "LLM安全"
  - "因果发现"
  - "LLM引导"
  - "公平性审计"
  - "主动学习"
  - "动态评分"
  - "偏见路径"
---

# Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach

**会议**: AAAI2026  
**arXiv**: [2506.12227](https://arxiv.org/abs/2506.12227)  
**作者**: Khadija Zanna, Akane Sano (Rice University)  
**代码**: 未公开  
**领域**: AI安全  
**关键词**: 因果发现, LLM引导, 公平性审计, 主动学习, 动态评分, 偏见路径  

## 一句话总结

提出一种融合LLM语义先验与统计信号的混合因果发现框架，通过主动学习（Active Learning）和动态评分机制优先查询信息量最大的变量对，在噪声和混淆条件下有效恢复公平性关键因果路径（如 sex→education→income），显著优于传统CD方法和朴素LLM方法。

## 背景与动机

### 公平性审计的因果视角

机器学习模型在招聘、贷款、教育和医疗等领域的偏见问题日益受到关注。这些偏见往往不是通过敏感属性（如性别、种族）对结果的直接影响产生，而是通过间接的结构性路径——敏感属性经由代理变量或混淆关系影响最终决策。传统的公平性审计依赖统计差异度量（demographic parity等），但这些指标无法揭示偏见的因果传播机制，导致干预措施可能无效甚至误导。因果发现（Causal Discovery, CD）提供了识别此类路径的工具，能够区分真实因果效应与混淆或测量伪影引入的虚假关联。

### 现有方法的不足

经典CD方法（如PC算法、GES）依赖条件独立性检验和强假设（如faithfulness），在噪声数据、潜在混淆或不完整元数据下容易失败。优化方法（NOTEARS、DAGMA）计算成本高且对超参数敏感。近年来LLM作为CD的辅助工具出现，利用语义知识从变量元数据推断因果方向，但存在两个关键问题：(1) 朴素使用LLM可能过度或不足地归因因果性，特别是涉及敏感属性时；(2) 基于BFS的LLM查询策略（Jiralerspong et al. 2024）对所有变量对一视同仁，容易浪费查询预算在非信息性关系上，且早期决策错误会传播。

### 核心动机

在公平性场景中，真实世界数据集缺乏已知ground-truth因果结构，这使得评估CD方法的公平性恢复能力极为困难。本文的动机是：(1) 设计一种自适应查询策略，平衡语义信号和统计信号来高效恢复公平性相关路径；(2) 构建一个包含已知偏见路径的半合成基准，支持系统评估。

## 核心问题

如何在噪声、混淆和数据腐蚀条件下，高效且准确地恢复敏感属性（如性别、种族）到结果（如收入）之间的因果路径？传统CD方法在此场景下精度不足，朴素LLM方法倾向过度预测，如何设计一种混合框架兼顾准确性和查询效率？

## 方法详解

### 整体框架

本文在BFS-based CD框架之上引入主动学习和动态评分两个关键组件。整体流程为：(1) 识别独立根变量；(2) 对每对未查询变量计算动态评分；(3) 优先选择评分最高的变量对查询LLM；(4) 若LLM回答"Yes"且添加边不产生环，则加入因果图；(5) 重复直到达到最大迭代次数或评分低于阈值。

### 关键设计1：动态评分机制

每对未查询变量 $(x, y)$ 接收一个融合统计信号、模型置信度和查询历史的综合评分：

$$S(x,y) = w_{\text{stat}} \cdot \text{StatScore}(x,y) + w_{\text{conf}} \cdot \text{LLMConf}(x,y) + w_{\text{hist}} \cdot \text{HistScore}(x,y)$$

其中各分量定义如下：

**统计评分**：结合互信息（MI）和偏相关（PCorr），捕获线性和非线性依赖关系：

$$\text{StatScore}(x,y) = \frac{\text{MI}(x,y) + \text{PCorr}(x,y)}{2}$$

PCorr 基于当前已发现的父节点集进行条件偏相关计算，遵循PC算法的迭代conditioning过程。

**LLM置信度评分**：基于模型token-level log-probabilities的sigmoid变换，反映LLM对该变量对因果关系判断的确定性：

$$\text{LLMConf}(x,y) = \frac{1}{1 + e^{-\text{confidence}}}$$

**查询历史评分**：惩罚重复查询，鼓励更广泛的探索：

$$\text{HistScore}(x,y) = \frac{1.5}{1 + \text{query\_count}(x,y)}$$

权重 $w_{\text{stat}}, w_{\text{conf}}, w_{\text{hist}}$ 通过贝叶斯优化（Gaussian Process surrogate + gp_minimize）调优。

### 关键设计2：主动学习查询策略

在每一步中，选择评分最高的变量对进行查询：

$$(x^*, y^*) = \arg\max_{(x,y) \in \text{Unqueried}} S(x,y)$$

LLM使用多轮对话模式，保持对先前推理的上下文感知。若返回"Yes"且不产生环则添加有向边，构建邻接矩阵：

$$A(i,j) = \begin{cases} 1 & \text{if } X_i \rightarrow X_j \text{ is predicted} \\ 0 & \text{otherwise} \end{cases}$$

### 关键设计3：公平性路径分析

学到因果图后，枚举所有从敏感属性 $S$ 到结果 $Y$ 的有向路径并分类：

- **直接路径**：$S \rightarrow Y$
- **间接路径**：$S \rightarrow \cdots \rightarrow Y$（经由中介变量）
- **虚假路径**：涉及 $S$ 但不到达 $Y$

通过因果效应分解量化偏见：$TE = DE + IE$，并归一化得到偏见系数：

$$C_{\text{bias}} = \frac{TE}{\text{Var}(Y)}$$

### 关键设计4：半合成基准构建

基于UCI Adult数据集构建包含15个变量的半合成因果图，注入从 race 和 sex 到 income 的直接边和间接路径（如 sex→education→income），并加入噪声、数据腐蚀和潜在混淆变量 $U \sim \mathcal{N}(0,1)$，提供可控的ground-truth评估环境。

## 实验关键数据

### 主实验：三个数据集上的结构恢复性能

| 方法 | 数据集 | Accuracy ↑ | F1 ↑ | Precision | Recall | NHD ↓ | 预测边数 |
|------|--------|-----------|------|-----------|--------|-------|---------|
| PC | Adult (15n, 28e) | 0.239 | 0.382 | 0.352 | 0.420 | 0.193 | 33 |
| GES | Adult | 0.296 | 0.473 | 0.368 | 0.580 | 0.203 | 44 |
| NOTEARS | Adult | 0.021 | 0.039 | 0.035 | 0.045 | 0.260 | 27 |
| DAGMA | Adult | 0.099 | 0.180 | 0.141 | 0.250 | 0.283 | 50 |
| LLM Pairwise | Adult | 0.307 | 0.470 | 0.331 | 0.813 | 0.253 | 69 |
| LLM BFS | Adult | 0.299 | 0.456 | 0.332 | 0.750 | 0.305 | 64 |
| **Proposed** | **Adult** | **0.413** | **0.585** | **0.792** | 0.464 | **0.109** | 17 |
| PC | Child (20n, 25e) | 0.146 | 0.255 | 0.273 | 0.239 | 0.097 | 22 |
| NOTEARS | Child | 0.216 | 0.356 | 0.403 | 0.319 | 0.080 | 20 |
| **Proposed** | **Child** | **0.364** | **0.533** | **0.601** | 0.479 | 0.082 | 20 |
| PC | Neuropathic (221n) | 0.041 | 0.078 | 0.092 | 0.068 | 0.025 | 563 |
| LLM BFS | Neuropathic | 0.000 | 0.000 | 0.000 | 0.000 | 0.903 | 43 |
| **Proposed** | **Neuropathic** | **0.073** | **0.136** | **0.690** | 0.075 | 0.109 | 84 |

### 公平性路径恢复

Ground-truth: 2条直接路径（sex→income, race→income）、25条间接路径，TE=4.89，$C_{\text{bias}}=28.46$。

| 方法 | 直接路径 | 间接路径 | 虚假路径 | $C_{\text{bias}}$ |
|------|---------|---------|---------|-----------|
| PC | 0 | 少量 | 多 | 低 |
| GES | 0 | 中等 | 多 | 低 |
| LLM Pairwise | 3（含虚假age路径） | 过多 | 多 | 膨胀 |
| **Proposed** | **2（仅sex, race）** | 部分恢复 | **极少** | 接近真实 |

本方法是唯一正确恢复两条真实直接路径（sex→income, race→income）同时排除虚假 age→income 路径的方法。

### 超参数敏感性分析

通过Random Forest回归分析贝叶斯优化trials，发现：最大迭代次数对F1影响最大，评分阈值和LLM温度次之。评分权重之间存在反相关，表明存在竞争性权衡——小图中语义引导更重要，大图（如Neuropathic）中MI/PCorr统计信号占主导。

## 亮点

- **自适应查询策略**：动态评分机制根据图结构和数据质量自适应调整语义vs统计信号的权重，在探索早期依赖LLM语义判断，后期转向统计信号验证，体现了从语义探索到经验精化的渐进过程
- **公平性路径精准恢复**：唯一正确识别 sex→income 和 race→income 直接路径，同时避免 age→income 虚假路径的方法，展现了强公平性诊断能力
- **高精度低误报**：在Adult数据集上 Precision 达0.792（远超所有baseline），仅预测17条边（ground-truth 28条），体现保守但可靠的策略
- **跨规模泛化**：从15节点到221节点均保持最优F1，特别在大规模Neuropathic网络上LLM BFS完全失败时仍有效
- **半合成基准**：为公平性敏感的CD评估提供了包含controlled ground-truth的可复现基准

## 局限与展望

- **Recall偏低**：保守策略导致Adult数据集Recall仅0.464，可能遗漏某些公平性相关的间接路径
- **LLM依赖与可复现性**：基于GPT-4的推理存在随机性和版本差异，与Jiralerspong et al.的结果存在不一致，LLM-based CD的标准化评估协议仍缺失
- **半合成数据局限**：因果图将income设为终端节点，忽略了下游反馈效应和时序动态
- **计算成本**：贝叶斯优化+LLM查询在大图上计算开销高，Neuropathic数据集受token限制影响严重
- **LLM社会偏见风险**：LLM自身编码的社会偏见可能导致反映刻板印象而非真实因果机制的推断
- **仅限观测CD**：未涉及干预性或实验性因果发现方法（如do-calculus、工具变量）

## 与相关工作的对比

- **PC / GES**：经典约束/评分方法在噪声下F1仅0.38-0.47，无法恢复公平性直接路径
- **NOTEARS / DAGMA**：连续优化方法在公平性结构上表现最差（F1 < 0.18），对超参数极度敏感
- **LLM Pairwise (Kıcıman et al. 2023)**：高Recall(0.813)但严重过度预测（69条边 vs 28条ground-truth），引入虚假公平性路径导致 $C_{\text{bias}}$ 膨胀
- **LLM BFS (Jiralerspong et al. 2024)**：固定顺序查询，对早期错误敏感，大规模图上完全失败（Neuropathic F1=0）
- **Takayama et al. 2024**：LLM增强统计方法，但单次先验注入，不具备本文的动态自适应能力
- 本文核心改进在于动态评分+主动学习的查询优先级机制，在保持高精度的同时显著降低NHD

## 启发与关联

- **路径级可解释性优于汇总统计**：$C_{\text{bias}}$ 等汇总指标可能被虚假路径膨胀，路径级分析提供更可靠的公平性诊断，这对实际部署中的公平性审计具有重要启示
- **语义先验与统计信号的互补**：在图发现早期，统计信号弱时LLM语义引导尤为重要；随着信息积累，统计信号逐步主导——这种自适应平衡策略可推广到其他LLM-assisted科学发现任务
- **公平性审计工具化**：该框架可作为招聘、贷款等高风险领域的算法审计工具，帮助非技术利益相关者追溯偏见传播路径

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将主动学习和动态评分引入LLM-guided CD是有意义的创新，但整体框架是对BFS方法的增量改进
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集、多baseline对比、超参数敏感性分析和公平性路径评估较为全面，但Recall偏低未深入分析
- 写作质量: ⭐⭐⭐⭐ — 结构完整，动机清晰，公式规范，但部分讨论偏冗长
- 价值: ⭐⭐⭐⭐ — 公平性+因果发现+LLM的交叉方向有实际意义，半合成基准对社区有贡献，但LLM可复现性问题限制了方法的广泛采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach](uncovering_pretraining_code_in_llms_a_syntax-aware_attribution_approach.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](../../ICLR2026/llm_safety/biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[CVPR 2026\] Revisiting Learning with Noisy Labels: Active Forgetting and Noise Suppression](../../CVPR2026/llm_safety/revisiting_learning_with_noisy_labels_active_forgetting_and_noise_suppression.md)
- [\[AAAI 2026\] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans](principles2plan_llm-guided_system_for_operationalising_ethical_principles_into_p.md)
- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)

</div>

<!-- RELATED:END -->
