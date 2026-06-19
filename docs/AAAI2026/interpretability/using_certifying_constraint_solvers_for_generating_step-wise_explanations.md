---
title: >-
  [论文解读] Using Certifying Constraint Solvers for Generating Step-wise Explanations
description: >-
  [AAAI 2026][可解释性][约束求解] 提出利用约束求解器生成的不可满足性证明（proof）作为起点，通过一系列简化和转换技术高效生成面向用户的逐步解释序列（step-wise explanation），相比从零构建解释方法速度提升高达100倍。 1. 领域现状： 声明式问题求解允许用户仅描述问题本身而非求解过程…
tags:
  - "AAAI 2026"
  - "可解释性"
  - "约束求解"
  - "逐步解释"
  - "证明日志"
  - "可解释AI"
  - "不可满足性"
---

# Using Certifying Constraint Solvers for Generating Step-wise Explanations

**会议**: AAAI 2026  
**arXiv**: [2511.10428](https://arxiv.org/abs/2511.10428)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 约束求解, 逐步解释, 证明日志, 可解释AI, 不可满足性

## 一句话总结

提出利用约束求解器生成的不可满足性证明（proof）作为起点，通过一系列简化和转换技术高效生成面向用户的逐步解释序列（step-wise explanation），相比从零构建解释方法速度提升高达100倍。

## 研究背景与动机

1. **领域现状**: 声明式问题求解允许用户仅描述问题本身而非求解过程，通用求解器可以自动找到解。然而当问题不可满足（unsatisfiable）时，向用户解释"为什么没有解"是一个关键挑战。
2. **现有痛点**: 最小不可满足子集（MUS）可以帮助用户聚焦问题子集，但可能仍然很大且难以理解。逐步解释框架（step-wise explanation）通过简单推理步骤的序列来解释，但计算成本极高——每一步都需要多次调用 NP-oracle，整个序列的计算更加昂贵。
3. **核心矛盾**: 高质量的用户级解释需要简单且短小的推理步骤，但这种解释的计算复杂度与求解器内部的高效证明格式之间存在巨大鸿沟。
4. **本文目标**: 大幅降低逐步解释序列的计算开销，使其可应用于更大规模的约束问题。
5. **切入角度**: 利用 certifying solver 生成的证明日志（DRCP proof）作为起点，通过一套转换和简化技术将机器级证明转化为人类可理解的逐步解释。
6. **核心 idea**: 求解器已经在验证不可满足性时生成了完整的推理过程（proof），与其从零构建解释，不如复用并转换这些证明。

## 方法详解

### 整体框架

整体流程分为三阶段：(1) 约束求解器求解并生成 DRCP 证明日志；(2) 通过一系列简化操作（去除辅助变量、简化到域缩减、替换求解器级约束为用户级约束）将证明转换为满足逐步解释格式的抽象证明；(3) 通过原因最小化（reason minimization）进一步精简每步的推理依据，使解释尽可能简单易懂。

### 关键设计

1. **Abstract Proof Framework（抽象证明框架）**:
    - 功能：提供统一的形式化框架，涵盖 DRCP 证明日志和逐步解释两种格式
    - 核心思路：定义抽象证明为一组 $(R_i, C_i)$ 对的序列，其中 $R_i$ 是推理依据集合，$C_i$ 是推导出的约束集合。通过对证明步骤施加不同属性限制（P-proof），可以在同一框架下表达求解器证明和用户解释
    - 设计动机：统一框架使得证明到解释的转换变得系统化——只需逐步refinement属性限制即可

2. **Proof Simplification（证明简化）**:
    - 功能：将 DRCP 证明中不符合解释格式的步骤移除，同时保持证明有效性
    - 核心思路：两个关键简化操作：(a) 辅助变量简化——移除涉及建模系统引入的辅助变量的证明步骤，将其依赖合并到后续步骤中（P: $scope(C) \subseteq \mathcal{X}$）；(b) 域缩减简化——确保每步只推导关于单个变量域的原子约束（P: $|scope(C)| \leq 1$）
    - 设计动机：用户只关心决策变量，不关心求解器内部的辅助变量；逐步解释要求每步推导具体的域值变化

3. **Solver-to-User Level Transformation + Reason Minimization（求解器级到用户级转换 + 原因最小化）**:
    - 功能：将证明步骤中的求解器级约束替换为对应的用户级约束，并最小化每步的依据集合
    - 核心思路：通过追踪用户级约束与求解器级约束的对应关系，利用 Lemma 1（如果用户级约束在决策变量上比求解器级约束更强或等强，则可安全替换）进行替换。原因最小化（Algorithm 1）从证明末尾向前遍历，对每步通过 MUS 计算去冗余依据
    - 设计动机：用户只理解其建模的原始约束；最小化依据减少认知负担

### 损失函数 / 训练策略

本文不涉及深度学习，核心算法是组合优化方法。关键复杂度优化在于：
- Trimming：去除证明中未被后续步骤引用的冗余步骤
- Local vs Global reason minimization：Local 模式只在原有依据中找最小子集，Global 模式可替换为证明历史中的任意可用约束
- 证明简化的关键性质：对于不可满足性证明总是可行的（因为 $\bot$ 没有涉及任何变量）

## 实验关键数据

### 主实验

| 应用场景 | 指标 | 本文方法 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 多种 CSP 基准 | 生成时间 | - | - | 最高 100x 加速 |
| 逐步解释质量 | 序列长度/步骤大小 | 与 SOTA 相当 | - | 质量无明显损失 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整方法 | 最优速度+质量 | 所有简化+最小化 |
| 无 Trimming | 冗余步骤增多 | 证明未修剪 |
| Local vs Global minimization | Global 更短但更慢 | 权衡取舍 |

### 关键发现

- 从证明日志出发可将解释生成速度提升高达 100 倍，而解释质量（序列长度和单步大小）与现有最优方法相当
- 辅助变量简化对于连接求解器内部表示和用户理解至关重要
- 证明简化在极端情况下会退化为单步平凡证明，但在实际场景中很少发生

## 亮点与洞察

- **复用而非重建**：这是一个"站在巨人肩膀上"的思路——求解器内部已经有了完整的推理过程（证明），只需要转换格式而非从零开始
- **统一框架的优雅性**：Abstract Proof 框架巧妙地将机器证明和人类解释统一到同一数学结构中，使转换过程有理论保障
- **实用价值**：可解释约束求解对于调度、规划等实际应用非常重要——当系统说"无解"时，用户需要知道是哪些约束冲突了
- **证明日志的新用途**：传统上 certifying solver 的证明日志主要用于验证正确性，本文展示了它们在可解释性方面的全新价值

## 局限与展望

- 目前仅支持 DRCP 格式的证明日志，其他格式（如 DRAT）需要额外适配
- 解释质量取决于证明质量——低效的求解器可能生成冗长的证明，导致简化后的解释也较长
- Global reason minimization 虽然能生成更短的解释但计算成本更高，需要在质量和速度间权衡
- 目前仅针对不可满足性解释，对最优性解释的支持有限

## 相关工作与启发

- **vs MUS-based 方法**: MUS 只给出一个冲突约束子集，缺乏推理过程；本文给出完整的逐步推理链
- **vs 从零构建的 step-wise 方法 (BGG21, GBG23)**: 这些方法需多次调用 NP-oracle 构建每一步，计算成本极高；本文从现有证明出发，大幅降低计算量
- **vs DRAT (SAT证明格式)**: DRCP 是专为 CP 求解器设计的证明格式，支持更丰富的约束类型

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统地将 certifying solver 的证明日志转化为用户级逐步解释
- 实验充分度: ⭐⭐⭐⭐ 在多种应用域上测试，100x 加速令人印象深刻
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义严谨、示例丰富、组织清晰
- 价值: ⭐⭐⭐⭐ 约束求解可解释性的重要进展，但领域较为小众

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Diffusion-CAM: Faithful Visual Explanations for dMLLMs](../../ACL2026/interpretability/diffusion-cam_faithful_visual_explanations_for_dmllms.md)
- [\[ICML 2025\] On the Effect of Uncertainty on Layer-wise Inference Dynamics](../../ICML2025/interpretability/on_the_effect_of_uncertainty_on_layer-wise_inference_dynamics.md)
- [\[CVPR 2026\] Measuring the (Un)Faithfulness of Concept-Based Explanations](../../CVPR2026/interpretability/measuring_the_unfaithfulness_of_concept-based_explanations.md)
- [\[CVPR 2026\] MedLIME: A Distribution-Aligned and Evidence-Supported Framework for Medical Saliency Explanations](../../CVPR2026/interpretability/medlime_a_distribution-aligned_and_evidence-supported_framework_for_medical_sali.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](../../ACL2026/interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)

</div>

<!-- RELATED:END -->
