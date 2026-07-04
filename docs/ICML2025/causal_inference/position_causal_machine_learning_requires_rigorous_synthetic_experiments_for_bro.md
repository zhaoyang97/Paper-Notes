---
title: >-
  [论文解读] Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption
description: >-
  [ICML2025][因果推理][因果推断] 本文是一篇 Position Paper，主张合成实验对因果机器学习 (Causal ML) 方法的严格评估不可或缺：，但当前的合成实验设计存在偏差和复杂度不足，需要遵循一套原则来提高实验质量，从而推动 Causal ML 的广泛采用。 核心矛盾 因果机器学习（Causal ML…
tags:
  - "ICML2025"
  - "因果推理"
  - "因果推断"
  - "合成数据"
  - "实验评估"
  - "基准偏差"
  - "Position Paper"
---

# Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption

**会议**: ICML2025  
**arXiv**: [2508.08883](https://arxiv.org/abs/2508.08883)  
**代码**: [GitHub](https://github.com/panispani/causalml-needs-synth-eval)  
**领域**: 因果推理  
**关键词**: 因果推断, 合成数据, 实验评估, 基准偏差, Position Paper

## 一句话总结

本文是一篇 Position Paper，主张合成实验对因果机器学习 (Causal ML) 方法的严格评估**不可或缺**，但当前的合成实验设计存在偏差和复杂度不足，需要遵循一套原则来提高实验质量，从而推动 Causal ML 的广泛采用。

## 研究背景与动机

### 核心矛盾

因果机器学习（Causal ML）旨在用 ML 算法回答因果问题，在决策领域有巨大潜力，但至今未被主流 ML 社区广泛采用。原因在于：

**因果推断的根本问题**：对同一个体，无法同时观察到"接受处理"和"未接受处理"两种结果（反事实不可观测），这意味着真实数据几乎不可能提供因果查询的 ground truth

**评估实践的缺陷**：现有经验评估无法有效证明因果方法的可靠性和鲁棒性

**社区偏见**：ML 社区习惯于无假设的预测方法，对需要强假设的因果方法持怀疑态度

### 作者立场

针对社区批评 Causal ML 过度依赖合成实验的声音，作者**反向论证**：合成数据本身不是问题，**问题在于合成实验的设计方式**。合成实验是精确评估因果方法的唯一可靠途径。

## 方法详解：三大问题与四项原则

本文并非提出新算法，而是系统诊断当前评估实践的问题，并提出改进原则。

### Problem 1: Ground Truth 数据稀缺

- 预测 ML 可直接观测标签，因果 ML 无法观测反事实结果
- 真实因果数据集来源有限：专家知识（昂贵且主观）、随机对照试验 RCT（代价高、伦理受限）
- LaLonde (1986)、Shadish et al. (2008) 等少数提供观测+实验数据的研究被反复使用，无法支撑通用结论
- **反事实查询（PCH 第三层级）根本不存在真实数据集**

### Problem 2: 合成/半合成数据存在（无意识的）偏差

偏差来源：

1. **研究者设计偏差**：实验通常由方法的作者设计，倾向于展示自己方法的优势
2. **建模局限偏差**：合成数据只能包含研究者知道如何建模的特征，"未知的未知"被排除

半合成数据同样受影响：

- 因果发现数据集从真实数据拟合 CGM，继承了建模假设的偏差
- CATE 估计的半合成数据集中，若数据和假设不满足可识别性条件，方法仍会收敛到一个估计值——就好像唯一解存在一样
- 从 RCT 生成人工观测数据时，采样策略的选择直接引入偏差

### Problem 3: 合成实验复杂度不足

- 过度简化的因果模型：大量工作仍使用加性噪声模型（ANM），甚至限制为二次或广义线性机制
- 缺乏随机性来源：模拟参数（如因果图、混杂水平）通常固定
- 鲁棒性分析被忽视：方法仅在满足自身假设的数据上评估

### 四项原则

**原则 1：合成数据是得出严格结论的必要条件**

合成数据是因果查询 ground truth 的唯一可靠来源，且能完全控制数据生成过程，系统地变化噪声、混杂等参数。作者强调这是**必要**条件但非充分条件。

**原则 2：合成设计选择必须明确声明以减轻无意识偏差**

任何实验至少应明确五个要素：

| 要素 | 说明 |
|------|------|
| (i) 因果模型集合 | 研究的 SCM/CGM 的条件表达式 |
| (ii) 因果查询集合 | 查询的 PCH 层级、变量、取值 |
| (iii) 训练数据集合 | 维度、PCH 层级、可能的扰动（测量误差、选择偏差等）|
| (iv) 生成算法 | 产生合成因果模型、查询和数据集的算法 |
| (v) 诱导分布 | 生成算法在合成样本空间上隐含的分布 |

**原则 3：超越识别域内的聚合准确率，进行全面实验**

- 不仅在方法的识别域内评估，也要在假设被违反的场景下测试
- 评价维度应超越准确率，纳入鲁棒性、可扩展性、稳定性、可解释性
- 关注洞察而非聚合性能指标

**原则 4（隐含）：方法论化的系统评估**

通过定义大型合成实验空间，系统地探索不同条件下的方法表现。

## 实验设置与主要结果

### 实验 1：RealCause 半合成基准的偏差（验证 Problem 2）

**设置**：使用 RealCause（Neal et al., 2020）在 IHDP 数据集上生成半合成数据，评估 ATE 估计误差的稳定性。

**关键发现**：

| 实验条件 | ATE 误差 | 说明 |
|----------|----------|------|
| 原论文单次复现 | 0.17 | 真实 ATE = 4.02 |
| 固定 realization，变 seed（20 seeds）| 0.38 ± 0.39 | 最高 1.77 |
| 固定 seed，变 realization（100 个）| 0.95 ± 1.36 | 最高 9.45 |
| 100 realizations × 20 seeds 聚合 | 极端 case: 6.209 ± 11.318 | 真实 ATE = -0.604 |

**结论**：RealCause 基准具有高误差和极端方差，方法排名在不同 seed/realization 下不稳定。**依赖单次 seed 的基准结果是脆弱且可能误导的**。

### 实验 2：CausalNF 在假设违反下的表现（验证 Problem 3）

**设置**：以 Causal Normalizing Flows（CausalNF）为例，测试其在微分同胚假设被违反时的行为。

**场景 A：假设违反但性能不受影响**
- 对 TriangleNLIN SCM 的噪声施加非微分同胚变换（分段线性函数、正弦函数）
- 结果：反事实预测 RMSE 未显著增加
- 启示：方法友好的合成设计（如最小化噪声-父节点交互）可能掩盖假设违反的影响

**场景 B：假设违反导致性能恶化**
- 在不可识别的反事实例子上测试：两个不同 SCM 共享相同观测分布但反事实分布不同
- 结果：CausalNF 始终收敛到两种结构中的一种，在另一种为真时产生大误差
- 启示：方法存在**系统性偏差**，且仅靠标准实验不会被发现

## 亮点与洞察

1. **反直觉的立场**：不回避合成数据的问题，而是论证"用好合成数据"是推动 Causal ML 的必要路径
2. **RealCause 实验极具说服力**：通过简单的 seed/realization 变化就揭示了广泛使用的基准的不稳定性
3. **五要素分类框架**有操作指导意义：为未来 Causal ML 论文的实验部分提供了清晰的 checklist
4. **CausalNF 的双面实验**巧妙展示了"假设违反不总是灾难性的"和"但有时是致命的"——强调了系统性鲁棒性测试的重要性
5. 将因果推断的可识别性理论与 ML 实验设计的实践问题**桥接**起来

## 局限与展望

1. **Position Paper 的固有局限**：提出原则但未提供完整的自动化工具或标准化框架实现这些原则
2. **实验覆盖窄**：仅以 RealCause 和 CausalNF 为例，其他主流方法（如 DoWhy、EconML）未涉及
3. **原则的可操作性存疑**：原则 2 中"生成算法在合成空间上的诱导分布"在实践中难以精确描述
4. **缺乏对计算成本的讨论**：大规模合成实验空间的生成与评估需要大量算力
5. **未给出"足够好"的标准**：多少实验、多少 seed、多大的合成空间才算符合原则？
6. **Domain-specific 的适配缺失**：医学、经济学等不同领域的因果问题特性差异很大，原则是否普适未讨论

## 相关工作与启发

- **Curth et al. (2021)**：首次展示合成 outcome 实验可因微小设计选择导致方法排名翻转
- **Gentzel et al. (2019)**：提出合成数据的"未知的未知"偏差
- **Herrmann et al. (2024) / Karl et al. (2024)**：预测 ML 中对当前经验实践的反思
- **Pearl & Mackenzie (2018)**：Pearl 因果层级（PCH）三层框架——关联、干预、反事实
- **Javaloy et al. (2023)**：CausalNF，使用正则化流做反事实估计的 SOTA 方法
- **Neal et al. (2020)**：RealCause，广泛使用的半合成因果基准生成方法

## 评分

- 新颖性: ⭐⭐⭐ — Position Paper 本身不含新方法，但立场清晰且论证扎实
- 实验充分度: ⭐⭐⭐ — 两组实验精准说明了问题，但覆盖面偏窄
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰、论证层层递进、问题与原则一一对应
- 价值: ⭐⭐⭐⭐ — 为 Causal ML 社区的实验规范提供了重要参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ICML 2026\] The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents](../../ICML2026/causal_inference/the_synthetic_web_adversarially-curated_mini-internets_for_diagnosing_epistemic_.md)
- [\[NeurIPS 2025\] Characterization and Learning of Causal Graphs from Hard Interventions](../../NeurIPS2025/causal_inference/characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](../../NeurIPS2025/causal_inference/do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[ICML 2025\] RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals](rate_causal_explainability_of_reward_models_with_imperfect_counterfactuals.md)

</div>

<!-- RELATED:END -->
