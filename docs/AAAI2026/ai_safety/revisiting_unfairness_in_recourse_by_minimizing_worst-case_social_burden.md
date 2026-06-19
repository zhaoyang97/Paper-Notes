---
title: >-
  [论文解读] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden
description: >-
  [AAAI 2026][AI安全][algorithmic recourse] 系统分析了算法追索 (algorithmic recourse) 中公平性度量的三大局限（忽视分类器决策行为、忽略真实标签、差距指标掩盖不公平），提出基于社会负担 (social burden) 的公平性框架 MISOB，通过极小化极大加权训练策略减少所有群体的社会负担，无需访问敏感属性即可在预测和追索阶段同时提升公平性。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "algorithmic recourse"
  - "social burden"
  - "fairness"
  - "minimax optimization"
  - "counterfactual explanation"
---

# Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden

**会议**: AAAI 2026  
**arXiv**: [2509.04128](https://arxiv.org/abs/2509.04128)  
**代码**: [github](https://github.com/abarrainkua/MISOB)  
**领域**: AI Safety / Algorithmic Fairness  
**关键词**: algorithmic recourse, social burden, fairness, minimax optimization, counterfactual explanation

## 一句话总结

系统分析了算法追索 (algorithmic recourse) 中公平性度量的三大局限（忽视分类器决策行为、忽略真实标签、差距指标掩盖不公平），提出基于社会负担 (social burden) 的公平性框架 MISOB，通过极小化极大加权训练策略减少所有群体的社会负担，无需访问敏感属性即可在预测和追索阶段同时提升公平性。

## 研究背景与动机

自动决策系统在信贷审批、公共服务等领域广泛应用，当模型给出负面决策时，应提供可操作的追索建议（如"增加月收入500元"），使个体能够翻转结果。然而，追索过程本身可能存在不公平：不同群体获得相同建议但实际执行成本截然不同。

现有追索公平性研究存在三个核心问题：(1) **忽视分类器的决策行为**——仅对被拒绝个体计算追索成本的均值相等，但如果某群体被拒绝的概率更高，该群体整体承受的追索负担更重；(2) **忽略真实标签**——未区分"本应被接受但被错误拒绝"的个体和"确实应被拒绝"的个体，前者被迫改变特征是系统错误造成的不公；(3) **差距指标可能掩盖不公**——两个群体社会负担的差距为零并不意味着公平，可能是两个群体都遭受了很高的负担。

切入角度：将公平性从"差距最小化"转向 Rawlsian 极小化极大视角，关注最坏群体的社会负担，提出不需要敏感属性的轻量级训练方法 MISOB。

## 方法详解

### 整体框架

MISOB 是一个迭代训练框架：(1) 预训练基础分类器 $f^{(0)}$；(2) 每轮迭代中计算每个训练样本的社会负担，然后用负担加权的损失函数重新训练分类器。高负担样本获得更高权重，引导分类器改善这些样本的决策。

### 关键设计

**1. 社会负担 (Social Burden) 的定义**

传统追索成本关注所有被拒绝个体的平均成本，而社会负担聚焦于**本应被接受但被错误拒绝**的个体。对于敏感群体 $s$：

$$B_{f,g}^s = \underbrace{\mathbb{E}[\delta((X,s), g_f(X))]}_{\text{假阴性的追索成本}} \cdot (1 - \underbrace{P(f(X)=1|S=s, Y=1)}_{\text{真阳性率 TPR}})$$

其中 $\delta$ 是追索成本函数，$g_f$ 是追索算法。社会负担同时考虑了：被错误拒绝的概率（与 TPR 相关）和改变特征的代价。理想情况下社会负担应为零。

论文同时定义了期望追索成本 $C_{f,g}^s$，用接受率 AR 替代 TPR 来衡量：

$$C_{f,g}^s = \mathbb{E}[\delta((X,s), g_f(X))] \cdot (1 - P(f(X)=1|S=s))$$

**2. 负担感知的实例加权**

MISOB 的核心是实例加权方案。每个训练样本 $x^i$ 的权重：

$$\phi(i, \mathcal{Q}, \alpha) = 1 + \alpha N \frac{b_{f,g_f}^i}{\sum_{j} b_{f,g_f}^j} \mathbb{1}\{\beta > 0\}$$

其中 $b_{f,g}^i = \delta(x^i, g_f(x^i)) \cdot \mathbb{1}\{y^i = 1\}$ 是实例级负担（仅正类实例产生负担），$\alpha$ 是权衡公平性与准确率的超参数。高负担样本获得更大权重，推动分类器优先改善这些样本的决策。

**3. 无需敏感属性**

MISOB 完全不需要在训练和推断时访问敏感属性。负担的计算基于真实标签和追索成本，而非群体归属。这意味着：(a) 无需收集敏感信息，避免法律和伦理风险；(b) 自然处理交叉群体公平性（如"年轻少数族裔女性"），因为群体定义可以在评估时事后定义。

### 损失函数 / 训练策略

加权分类损失：$\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^N \phi(i, \mathcal{Q}, \alpha) \cdot \ell(f(x^i), y^i)$。先预训练 $f^{(0)}$，然后进行 $T$ 轮迭代优化。每轮都需要重新计算追索成本。整体计算复杂度 $O(N^3)$，可通过批处理和并行化提升效率。

## 实验关键数据

### 主实验

在 Adult 数据集上，以种族为敏感属性的结果（10次随机划分的平均值）：

| 追索方法 | 策略 | 准确率 ↑ | 最坏负担 ↓ | 负担Δ ↓ | 最坏TPR ↑ | TPR Δ ↓ | 最坏成本 ↓ |
|---------|------|---------|-----------|---------|----------|---------|-----------|
| GS | 无 | 0.81 | 4.56 | 0.03 | 0.27 | 0.08 | 115.69 |
| GS | POSTPRO | 0.80 | 4.96 | 0.61 | 0.37 | 0.00 | 98.40 |
| GS | **MISOB** | **0.82** | **3.01** | **0.85** | **0.52** | **0.11** | **93.06** |
| WT | 无 | 0.81 | 1.28 | 0.01 | 0.27 | 0.08 | 38.27 |
| WT | POSTPRO | 0.80 | 1.55 | 0.01 | 0.37 | 0.00 | 39.71 |
| WT | **MISOB** | **0.82** | **0.79** | **0.16** | **0.59** | **0.02** | **30.77** |
| CCHVAE | 无 | 0.81 | 6.25 | 0.16 | 0.27 | 0.08 | 119.99 |
| CCHVAE | POSTPRO | 0.80 | 11.20 | 3.06 | 0.37 | 0.00 | 120.01 |
| CCHVAE | **MISOB** | **0.81** | **4.03** | **0.42** | **0.48** | **0.19** | **105.10** |

### 消融实验

超参数 $\alpha$ 对公平性-准确率权衡的影响（WT 方法，种族为敏感属性）：

| $\alpha$ 范围 | 准确率趋势 | 最坏负担趋势 | 最坏TPR趋势 |
|--------|-----------|------------|------------|
| 0.1-0.3 | 保持/略升 | 稳步下降 | 稳步上升 |
| 0.3-0.5 | 保持 | 继续下降 | 继续上升 |
| 0.5-1.0 | 开始下降 | 趋于稳定 | 趋于稳定 |

交叉群体（种族×性别）分析：

| 追索方法 | 策略 | 准确率 | 最坏负担 ↓ | 最坏TPR ↑ |
|---------|------|-------|-----------|----------|
| WT | 无 | 0.81 | 1.40 | 0.20 |
| WT | POSTPRO | 0.80 | 1.94 | 0.00 |
| WT | **MISOB** | **0.82** | **0.98** | **0.34** |

### 关键发现

- POSTPRO 虽然在预测层面实现了 TPR 均等，但在追索层面反而增加了负担和成本（CCHVAE 上最坏负担从 6.25 升至 11.20），证实了"预测公平≠追索公平"
- MISOB 在不损失甚至提升整体准确率的同时，系统性降低了所有群体的社会负担
- MISOB 训练一次即可评估任意群体划分（单属性或交叉属性），而 POSTPRO 需要对每种划分分别训练
- 差距指标 Δ 小不等于公平——MISOB 有时 Δ 略大，但每个群体的绝对指标都更好

## 亮点与洞察

- 社会负担的定义将分类器的预测误差（TPR）和追索成本联合考量，揭示了传统"等成本"范式掩盖的结构性不公
- 不需要敏感属性是关键实用优势，符合 GDPR 等隐私法规要求
- 理论贡献扎实：形式化了追索公平性与预测公平性之间的关系，证明了满足机会均等不保证社会负担均等
- 极小化极大视角替代差距最小化，避免了通过降低特权群体性能来"实现平等"的伪公平

## 局限与展望

- 算法收敛性缺乏理论保证，迭代训练的稳定性依赖预训练质量
- 追索成本计算使用 $\ell_2$ 距离，与现实中的真实努力成本可能存在差距
- 当前只在静态设定下验证，数据分布随时间变化时社会负担的动态演化尚未探索
- 计算复杂度 $O(N^3)$ 对大规模数据可能成为瓶颈

## 相关工作与启发

- **vs Equal Cost 范式 (von2022fairness等)**: 传统方法仅关注被拒绝个体的追索成本均等，忽略群体接受率差异和误分类影响；MISOB 定义了包含 TPR 和成本的社会负担指标
- **vs POSTPRO (Hardt et al.)**: POSTPRO 通过后处理实现 TPR 均等但可能恶化追索公平性；MISOB 通过训练阶段的实例加权同时改善预测和追索公平性

## 评分

- 新颖性: ⭐⭐⭐⭐ 社会负担的形式化定义和极小化极大视角是重要理论贡献
- 实验充分度: ⭐⭐⭐ 仅在 Adult 数据集上验证，缺少更多真实数据集的实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，动机层层递进，问题分析深入
- 价值: ⭐⭐⭐⭐ 揭示了追索公平性领域的根本问题，框架通用且实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Minimizing Inequity in Facility Location Games](minimizing_inequity_in_facility_location_games.md)
- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](../../ICML2026/ai_safety/position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[AAAI 2026\] Alternative Fairness and Accuracy Optimization in Criminal Justice](alternative_fairness_and_accuracy_optimization_in_criminal_j.md)
- [\[ACL 2025\] Sandcastles in the Storm: Revisiting Watermarking Impossibility](../../ACL2025/ai_safety/sandcastles_watermarking_impossibility.md)
- [\[CVPR 2026\] Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection](../../CVPR2026/ai_safety/omni-fake_benchmarking_unified_multimodal_social_media_deepfake_detection.md)

</div>

<!-- RELATED:END -->
