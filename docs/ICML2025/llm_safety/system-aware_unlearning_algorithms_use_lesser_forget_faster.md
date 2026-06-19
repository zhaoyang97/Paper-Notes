---
title: >-
  [论文解读] System-Aware Unlearning Algorithms: Use Lesser, Forget Faster
description: >-
  [ICML2025][LLM安全][machine unlearning] 提出系统感知遗忘 (system-aware unlearning) 新定义，将攻击者的能力限制为只能访问系统实际存储的内容而非全部剩余数据，并基于核心集 (core set) + 选择采样 (selective sampling) 设计了线性分类的精确遗忘算法，实现亚线性内存和极低删除时间。
tags:
  - "ICML2025"
  - "LLM安全"
  - "machine unlearning"
  - "system-aware"
  - "selective sampling"
  - "core set"
  - "linear classification"
  - "deletion capacity"
---

# System-Aware Unlearning Algorithms: Use Lesser, Forget Faster

**会议**: ICML2025  
**arXiv**: [2506.06073](https://arxiv.org/abs/2506.06073)  
**代码**: 未开源  
**领域**: 其他/机器遗忘 (Machine Unlearning)  
**关键词**: machine unlearning, system-aware, selective sampling, core set, linear classification, deletion capacity

## 一句话总结

提出系统感知遗忘 (system-aware unlearning) 新定义，将攻击者的能力限制为只能访问系统实际存储的内容而非全部剩余数据，并基于核心集 (core set) + 选择采样 (selective sampling) 设计了线性分类的精确遗忘算法，实现亚线性内存和极低删除时间。

## 研究背景与动机

**现状**：机器遗忘 (machine unlearning) 旨在让训练好的 ML 模型高效"遗忘"被请求删除的数据点，满足 GDPR、CCPA 等隐私法规的"被遗忘权"。当前的标准定义（exact/approximate unlearning）要求遗忘后的模型与从头重训（去掉被删数据后）得到的模型统计不可区分。

**痛点**：
1. 标准定义假定最强攻击者——能获取所有剩余数据 $S \setminus U$，这过于悲观且不现实
2. Cherapanamjeri et al. (2025) 证明在传统定义下，即使是线性分类器的精确遗忘也必须存储**整个数据集**，使遗忘在大规模数据上不可行
3. 现有高效遗忘算法几乎仅限于凸损失函数，无法推广到更一般的模型类

**核心 Idea**：如果一个学习系统只存储和使用训练数据的一小部分（核心集），那么攻击者即使完全攻破系统也只能看到核心集，因此大部分数据点天然具有隐私保护——"用得越少，忘得越快"。

## 方法详解

### 1. System-Aware Unlearning 定义

定义系统状态 (State-of-System) $\textsf{I}_A(S, U)$ 为遗忘算法处理完删除请求后系统中**实际存储**的所有信息（模型、保留的样本、辅助统计量等）。

**Definition 2.3 (System-Aware-$(\varepsilon, \delta)$-Unlearning)**：算法 $A$ 是 system-aware-$(\varepsilon, \delta)$ 遗忘算法，如果对任意训练集 $S$，存在 $S' \subseteq S$，使得对所有删除请求 $U \subseteq S$ 和可测集 $F$：

$$\Pr(\textsf{I}_A(S, U) \in F) \leq e^\varepsilon \cdot \Pr(\textsf{I}_A(S' \setminus U, \emptyset) \in F) + \delta$$

及其反向不等式成立。当 $S' = S$ 时退化为传统定义，因此 system-aware unlearning 严格推广了传统定义。

**关键理论保障 (Theorem 2.4)**：通过互信息论证，$S' \setminus U$ 泄露的关于 $U$ 的信息不多于 $S \setminus U$:

$$\sup_\mu (\mathsf{MI}(U; S' \setminus U) - \mathsf{MI}(U; S \setminus U)) \leq 0$$

### 2. 基于核心集的遗忘框架

**Core Set Algorithm (Definition 2.5)**：如果学习算法 $A$ 存在映射 $\mathfrak{C}$，使得 $\mathfrak{C}(S) \subseteq S$ 且 $A(S) = A(\mathfrak{C}(S))$，则 $\mathfrak{C}(S)$ 为核心集。

**通用框架 (Theorem 3.1)**：组合任意传统 $(\varepsilon, \delta)$-unlearning 算法 $A_{un}$ 和核心集映射 $\mathfrak{C}$，构造 system-aware 遗忘算法 $A_{CS}$：在学习阶段提取核心集并只存储核心集，遗忘时只对核心集中被删除的点执行更新。对核心集外的删除请求无需任何计算。

### 3. 线性分类的精确遗忘算法（Algorithm 1）

**选择采样 (Selective Sampling)**：采用 BBQSampler (Cesa-Bianchi et al., 2009) 作为核心集构造方法。对每个到达的样本 $x_t$，检查查询条件：

$$x_t^\top A_{t-1}^{-1} x_t > T^{-\kappa}$$

若满足则查询标签并加入核心集 $\mathcal{Q}$，更新 $A_t \leftarrow A_{t-1} + x_t x_t^\top$，$b_t \leftarrow b_{t-1} + y_t x_t$，$w_t \leftarrow A_t^{-1} b_t$。

**删除更新 (DeletionUpdate)**：对于核心集中被删的点 $(x, y) \in U \cap \mathcal{Q}$，执行逆更新：$A \leftarrow A - xx^\top$，$b \leftarrow b - yx$，$w \leftarrow A^{-1}b$（Sherman-Morrison 公式，$O(d^2)$ 时间）。

**单调性 (Theorem 4.2)**：BBQSampler 的查询条件仅依赖 $x$（不依赖标签 $y$），因此具有单调性：$\mathfrak{C}(\mathfrak{C}(S) \setminus U) = \mathfrak{C}(S) \setminus U$。这意味着遗忘时无需重新运行选择采样来确定新的核心集。

**理论界 (Theorem 4.3)**：
- 内存：$N_T = O(d \cdot T^\kappa \log T)$，亚线性于 $T$
- $K$ 次核心集删除后 excess risk：$\mathcal{E}(w) = O\left(\frac{N_T \log T + \log(1/\delta)}{T - T_{\bar\varepsilon} - N_T}\right)$
- 核心集删除容量：$K = O\left(\frac{\bar\varepsilon^2 \cdot T^\kappa}{d \log T \cdot \log(1/\delta)}\right)$
- 期望删除时间：$\mathbb{E}[\text{time per deletion}] \leq \frac{d^3 T^\kappa \log T}{T}$（均匀删除分布下）

### 4. 推广至一般函数类（Algorithm 2）

基于 GeneralBBQSampler (Gentile et al., 2022) 将方法推广到一般函数类 $\mathcal{F}$，核心集大小由 eluder-dimension-like 量 $\mathfrak{D}(\mathcal{F}, S)$ 控制。

## 实验关键数据

### 数据集
- **Purchase Dataset**：二分类购买数据，249,215 个样本，$d=600$
- **Margin Dataset**：合成数据，200,000 个样本，$d=100$，硬间隔 $\gamma=0.1$

### 对比方法

| 方法 | 说明 |
|------|------|
| Algorithm 1 (本文) | 基于选择采样的核心集遗忘 |
| SISA (Bourtoule et al., 2021) | 数据分片 + 中间模型存储 |
| Exact Retraining | 每次删除后从头重训 |
| Sekhari et al. (2021) | 线性回归退化为精确重训 |

### 主要结果（80,000 次删除，约 40% 数据）

**Table 1: Purchase Dataset**

| 指标 | Algorithm 1 | SISA | Exact Retraining |
|------|------------|------|------------------|
| 初始训练时间 (s) | 186.3 | 30.2 | 1828.7 |
| 累计删除时间 (s) | **58.3** | 1174.3 | 579.3 |
| 数据存储比例 | **13.1%** | 100% | 100% |

**Table 2: Margin Dataset**

| 指标 | Algorithm 1 | SISA | Exact Retraining |
|------|------------|------|------------------|
| 初始训练时间 (s) | **1.3** | 20.6 | 67.8 |
| 累计删除时间 (s) | **0.5** | 697.1 | 27.1 |
| 数据存储比例 | **0.6%** | 100% | 100% |

### 关键发现
1. Algorithm 1 在两个数据集上都以远低的内存（13.1% / 0.6%）维持与精确重训相当的准确率
2. 面对标签依赖的删除序列（分布偏移），Algorithm 1 的准确率保持稳定，而 SISA 显著退化
3. 数据可压缩性越好（Margin Dataset），Algorithm 1 优势越大——训练/删除时间、内存全面碾压

## 亮点与洞察

1. **定义层面的突破**：将攻击者模型从"知道所有剩余数据"放松为"只能看到系统实际存储的内容"，既合理又强大——突破了"线性分类精确遗忘必须存全部数据"的理论下界
2. **优雅的 reduction**：从选择采样到遗忘的归约非常自然——"不确定的才查询，查询过的才需要遗忘"
3. **单调性是关键**：BBQSampler 查询条件仅依赖输入 $x$（不依赖标签），确保删除不会导致新的点被查询，避免遗忘时重新运行采样器
4. **理论完整**：给出了内存、删除容量、excess risk、期望删除时间的完整 tradeoff 分析

## 局限与展望

1. **仅限分类任务**：当前算法和理论仅针对二分类（线性和一般函数类），未涉及回归、生成等任务
2. **可实现性假设**：要求存在 Bayes optimal 预测器在假设类中（realizability），限制了实际适用性
3. **实验规模有限**：仅在中等规模线性分类上验证，未在深度网络/大模型微调场景中测试
4. **近似版本未展开**：论文主要关注精确 system-aware 遗忘（$\varepsilon=\delta=0$），近似版本的算法设计留为未来工作
5. **核心集外的删除缺乏验证性保障**：虽然核心集外的点"免费删除"，但缺乏对此描述的可审计性

## 相关工作与启发

- **Sekhari et al. (2021)**：提出传统 certified unlearning 的标准定义和删除容量概念
- **Cherapanamjeri et al. (2025)**：证明传统定义下线性分类精确遗忘的内存下界为 $\Omega(T)$，本文正是绕过此下界
- **Bourtoule et al. (2021) SISA**：数据分片方法，实验中的主要对比方法
- **Cesa-Bianchi et al. (2009) BBQSampler**：本文核心采样算法的来源
- **Ghazi et al. (2023)**：ticketed learning-unlearning，建立了遗忘内存复杂度与主动学习查询复杂度的联系

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从定义层面突破传统遗忘的理论壁垒，视角独到
- 实验充分度: ⭐⭐⭐ — 实验仅限线性分类，规模和场景有限
- 写作质量: ⭐⭐⭐⭐ — 理论论述清晰，定义→框架→算法→理论→实验层层递进
- 价值: ⭐⭐⭐⭐ — 为机器遗忘提供了新的理论范式，有望推动更实用的遗忘算法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](../../ACL2025/llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](targeted_unlearning_with_single_layer_unlearning_gradient.md)

</div>

<!-- RELATED:END -->
