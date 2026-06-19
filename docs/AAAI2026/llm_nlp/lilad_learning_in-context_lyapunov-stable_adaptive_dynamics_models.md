---
title: >-
  [论文解读] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models
description: >-
  [AAAI 2026][LLM 其他][Lyapunov稳定性] 提出 LILAD 框架，利用 GPT-2 的 in-context learning 能力同时学习动力学模型和 Lyapunov 函数，在保证全局指数稳定性的同时实现对非平稳参数化动力系统的自适应辨识，在多个基准系统上超越 ICL、MAML 等基线。
tags:
  - "AAAI 2026"
  - "LLM 其他"
  - "Lyapunov稳定性"
  - "上下文学习"
  - "自适应系统辨识"
  - "非平稳动力学"
  - "GPT-2"
---

# LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models

**会议**: AAAI 2026  
**arXiv**: [2511.21846](https://arxiv.org/abs/2511.21846)  
**代码**: [https://github.com/amitjena1992/LILAD](https://github.com/amitjena1992/LILAD)  
**领域**: 其他  
**关键词**: Lyapunov稳定性, 上下文学习, 自适应系统辨识, 非平稳动力学, GPT-2

## 一句话总结

提出 LILAD 框架，利用 GPT-2 的 in-context learning 能力同时学习动力学模型和 Lyapunov 函数，在保证全局指数稳定性的同时实现对非平稳参数化动力系统的自适应辨识，在多个基准系统上超越 ICL、MAML 等基线。

## 研究背景与动机

**领域现状**：系统辨识旨在从轨迹数据中近似动力系统。神经网络因强表达能力被广泛使用，但通常不保证物理属性（如稳定性），且假设系统是平稳的。

**现有痛点**：
   - 稳定性保证方法（如 Lyapunov 约束的神经网络）假设系统参数不变，无法处理非平稳动态
   - 自适应方法（如 meta-learning、ICL）只追求预测精度，不保证学到的模型稳定
   - 当系统参数变化时，重新训练稳定模型代价极高，不适合时间敏感的安全关键应用

**核心矛盾**：稳定性与自适应性此前分别研究，没有统一框架同时保证两者

**切入角度**：利用 ICL 的 prompt 机制实现零样本自适应，同时通过对抗训练联合学习 Lyapunov 函数来保证稳定性

**核心 idea**：基于 GPT-2 的 ICL 框架联合训练动力学模型 $G_\theta$ 和 Lyapunov 函数 $V_\phi$，并通过二分法计算状态相关的衰减因子 $\gamma(x)$ 严格保证稳定性

## 方法详解

### 整体框架

输入：包含 $M$ 个任务的多任务轨迹数据池，每个任务对应不同参数采样的动力系统 $x_{k+1} = f_{\vartheta_i}(x_k)$。训练两个基于 GPT-2 的模型：动力学模型 $G_\theta$ 和 Lyapunov 模型 $V_\phi$。测试时，仅需提供新系统的少量轨迹作为 prompt 即可零样本推理。

### 关键设计

1. **ICL 框架下的双模型架构**:

    - 功能：用相同的 prompt 结构同时训练动力学预测器和 Lyapunov 函数
    - 核心思路：构建 prompt $\mathscr{P}_j^i = \{x_{i,1}, f(x_{i,1}), \ldots, x_{i,j}, f(x_{i,j}), x_{i,j+1}\}$，$G_\theta$ 预测下一状态，$V_\phi$ 输出 Lyapunov 值。两者共享相同的（linear → GPT-2 transformer → linear）架构
    - 设计动机：ICL 天然支持上下文自适应，无需梯度更新即可泛化到新任务

2. **输出变形确保半正定性**:

    - 功能：强制 Lyapunov 函数满足正半定条件 $V(x) > 0, V(0) = 0$
    - 核心思路：$V_\phi(x|\mathscr{C}) = \sigma(c \cdot \tanh(V^{\text{raw}}_\phi(x|\mathscr{C})) - c \cdot \tanh(V^{\text{raw}}_\phi(0|\mathscr{C}))) + \epsilon \|x\|^2$，其中 $\sigma$ 是平滑 ReLU
    - 设计动机：通过架构设计内在满足前两个 Lyapunov 条件，训练只需关注第三个条件（指数衰减）

3. **状态相关衰减因子 $\gamma(x)$**:

    - 功能：在推理时对可能违反 Lyapunov 约束的状态强制稳定性
    - 核心思路：对违约状态求解 $V_\phi(\gamma(x) \cdot G_\theta(x|\mathscr{C})|\mathscr{C}) - \beta V_\phi(x|\mathscr{C}) = 0$，通过中值定理证明 $\gamma \in [0,1]$ 必存在解，用二分法高效求解
    - 设计动机：提供严格的 out-of-distribution 稳定性保证，不依赖于 Lyapunov 模型的凸性假设

### 损失函数 / 训练策略

对抗式交替训练：
- **冻结 $G_\theta$，更新 $V_\phi$**：$\mathcal{L}^{\text{Lyap}} = \frac{1}{M(n+1)} \sum_{i,j} \max\{V_\phi(G_\theta(x|\mathscr{C})|\mathscr{C}) - \beta V_\phi(x|\mathscr{C}), 0\}$
- **冻结 $V_\phi$，更新 $G_\theta$**：$\mathcal{L}^{\text{Dyn}} = \text{MSE} + \lambda \cdot \text{Lyapunov违约惩罚}$

## 实验关键数据

### 主实验（Table 1: MAE 对比）

| 系统 | 维度 | ICL | MAML | CVaR | Stable-Linear | LILAD |
|------|------|-----|------|------|---------------|-------|
| Simple Pendulum | 2 | 0.018 | 0.023 | 0.085 | 0.065 | **0.004** |
| Double Pendulum | 4 | 0.039 | 0.022 | 0.12 | 0.17 | **0.011** |
| Microgrid | 5 | **0.005** | 0.007 | 0.014 | 0.011 | **0.005** |
| SEIR | 8 | 0.022 | 0.032 | 0.077 | 1.049 | **0.017** |
| PDE-SM | 100 | 6.354 | – | – | – | **0.060** |

### 消融：稳定性保证对比

| 方法 | 自适应 | 稳定性保证 | 高维可扩展 |
|------|--------|-----------|-----------|
| ICL | ✅ | ❌ | ✅ |
| MAML | ✅ | ❌ | ❌ |
| CVaR | ❌（鲁棒） | ❌ | ❌ |
| Stable-Linear | ❌ | ✅（线性） | ❌ |
| LILAD | ✅ | ✅ | ✅ |

### 关键发现
- 在高维 PDE 系统（100维）上，LILAD 的 MAE 比 ICL 低约 **106 倍**（0.060 vs 6.354）
- ICL 在某些测试实例上轨迹不收敛到原点，而 LILAD 始终保证收敛
- Microgrid 上 LILAD 与 ICL 接近，因为该系统本身强阻尼，稳定性约束影响小
- MAML 和 CVaR 在高维系统上无法扩展

## 亮点与洞察
- **ICL + Lyapunov 的交叉创新**：首次将 in-context learning 与 Lyapunov 稳定性理论结合。ICL 提供自适应能力，Lyapunov 提供安全保证，二者互补
- **状态相关衰减因子 $\gamma(x)$**：不需要假设 Lyapunov 函数的凸性（与之前工作不同），通过中值定理巧妙证明解的存在性，实际用二分法即可高效求解
- **对抗训练策略**：动力学模型和 Lyapunov 函数交替更新，让两者共同收敛到一致的稳定表示

## 局限与展望
- 仅适用于自治系统（无外部控制输入），未扩展到受控系统
- 假设所有参数化系统共享相同的平衡点（原点），限制了适用范围
- GPT-2 作为 backbone 在高维时需要较大的模型和较长的训练（PDE-SM 需要 2M epochs）
- 二分法求解 $\gamma(x)$ 增加了推理时的计算开销
- 未讨论 Lyapunov 函数的保守性——过度衰减可能导致动力学预测精度下降

## 相关工作与启发
- **vs Kolter & Zico (2019)**：他们联合训练 NN 动力学+Lyapunov，但假设平稳系统；LILAD 通过 ICL 扩展到非平稳场景
- **vs 标准 ICL (Forgione et al.)**：标准 ICL 只追求预测精度，LILAD 在此基础上加入稳定性约束
- **vs MAML**：MAML 需要梯度更新来适应新任务，LILAD 通过 prompt 零样本适应

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次统一自适应+稳定性，ICL+Lyapunov 的结合新颖
- 实验充分度: ⭐⭐⭐⭐ 5个基准系统涵盖不同维度，但缺少真实物理系统验证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但篇幅较长
- 价值: ⭐⭐⭐⭐ 对安全关键系统的自适应建模有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ICML 2026\] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](../../ICML2026/llm_nlp/t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](../../ACL2025/llm_nlp/exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](../../ACL2026/llm_nlp/ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](../../ACL2025/llm_nlp/leveraging_in-context_learning_for_political_bias_testing_of_llms.md)

</div>

<!-- RELATED:END -->
