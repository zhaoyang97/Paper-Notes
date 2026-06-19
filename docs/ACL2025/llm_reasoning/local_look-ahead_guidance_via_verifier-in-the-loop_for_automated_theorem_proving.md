---
title: >-
  [论文解读] Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving
description: >-
  [ACL 2025][LLM推理][自动定理证明] 提出 LeanListener，在自动定理证明(ATP)中引入 verifier-in-the-loop 设计，利用 Lean 验证器在每步提供中间反馈（子目标数变化）而非仅轨迹级奖励，通过在线 GRPO 训练使 ReProver 的 tactic 有效率和证明率均获提升，证明速度快 20%。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "自动定理证明"
  - "验证器反馈"
  - "GRPO"
  - "DPO"
  - "Lean"
---

# Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving

**会议**: ACL 2025  
**arXiv**: [2503.09730](https://arxiv.org/abs/2503.09730)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 自动定理证明, 验证器反馈, GRPO, DPO, Lean

## 一句话总结
提出 LeanListener，在自动定理证明(ATP)中引入 verifier-in-the-loop 设计，利用 Lean 验证器在每步提供中间反馈（子目标数变化）而非仅轨迹级奖励，通过在线 GRPO 训练使 ReProver 的 tactic 有效率和证明率均获提升，证明速度快 20%。

## 研究背景与动机

**领域现状**：LLM 在自动定理证明中常用于生成 tactic（证明步骤），但训练信号稀疏——通常只有整条证明成功或失败的二元反馈，缺乏步级信号。

**现有痛点**：(a) 稀疏奖励使 RL 训练效率低；(b) 合成训练数据生成成本高（需要大量 rollout 和验证）；(c) 模型常生成无效 tactic，浪费证明搜索时间。

**核心矛盾**：形式化验证器（如 Lean）能在每步判断 tactic 是否有效并报告剩余子目标数，但现有方法未充分利用这种步级反馈信号。

**本文目标** 如何将验证器的逐步反馈（子目标数变化）转化为有效的 RL 训练信号，提升 tactic 生成质量和定理证明效率。

**切入角度**：子目标数减少 = 进展，用 $\text{softplus}(\mathcal{G}(s_t) - \mathcal{G}(s_{t+1}))$ 作为步级奖励，通过在线 GRPO 训练。

**核心 idea**：用 Lean 验证器的子目标数变化作为步级奖励信号，在线 GRPO 训练 tactic 生成模型。

## 方法详解

### 整体框架
基于 ReProver（ByT5 encoder-decoder）作为 base 模型。训练时，对每个证明状态 $s_t$，模型采样多个 tactic，Lean 验证器逐步检查每个 tactic 的有效性并返回新状态的子目标数，计算步级奖励后通过 GRPO 更新策略模型。

### 关键设计

1. **子目标数奖励函数**:

    - 功能：为每步 tactic 提供密集奖励信号
    - 核心思路：$R(a_t; s_t) = \text{softplus}(\mathcal{G}(s_t) - \mathcal{G}(s_{t+1}))$，其中 $\mathcal{G}(s)$ 为状态 $s$ 的子目标数。子目标减少则奖励为正，无效 tactic 奖励为 0。softplus 防止负奖励
    - 设计动机：比二元有效性反馈更细粒度——减少 3 个子目标的 tactic 优于仅减少 1 个的

2. **在线 GRPO 训练**:

    - 功能：在线采样 + 验证 + 策略更新的循环训练
    - 核心思路：每 50 步更新模型权重，用组内相对优势 $\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ 做标准化，加 KL 正则防止策略偏移过大
    - 设计动机：离线 DPO 效果差（数据分布与模型分布偏移），在线训练显著更优

3. **前提 Dropout 数据增强**:

    - 功能：训练时随机丢弃部分前提（premises），增加数据多样性
    - 设计动机：防止模型过拟合于特定前提组合

### 训练策略
- 10,000 步微调，batch size 16，lr $2.25 \times 10^{-6}$
- Beam width=8 采样 tactic，每 50 步更新模型
- 约 40 A100 GPU-days（80% 花在 Lean 交互，CPU-bound）

## 实验关键数据

### 主实验（LeanDojo Benchmark, Pass@1 %）

| 模型 | Random Split | Novel Premises |
|------|-------------|----------------|
| ReProver (baseline) | 52.76 | 40.86 |
| **LeanListener (GRPO, 子目标)** | **53.21** | **41.11** |
| LeanListener (DPO, 二元, random) | 35.99 | - |

### Tactic 有效率

| 方法 | Prec@8 ↑ | 0-precision ↓ |
|------|----------|---------------|
| ReProver | 40.77 | 12.74% |
| Offline DPO | 37.75 | 29.10% |
| Online DPO | 44.75 | 11.88% |
| **Online GRPO** | **51.01** | **7.44%** |

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| 离线 vs 在线 | 在线训练关键：Prec@8 离线 37.75 vs 在线 GRPO 51.01 |
| DPO(二元) vs GRPO(子目标) | 二元反馈的 DPO 证明率反而下降到 35.99%（虚假长度偏差），GRPO 提升至 53.21% |
| DPO 配对策略 | random/zero-acc/hard 差异 <0.65%，不敏感 |
| 证明速度 | LeanListener 在相同时间内快 20% |

### 关键发现
- **DPO 的反直觉失败**：DPO 虽提升 tactic 有效率但导致证明率下降，因为学到了"短 tactic 更可能有效"的虚假关联——模型倾向生成过短的 tactic 而非语义正确的
- **在线训练是核心**：离线 DPO 无效甚至有害，但在线 DPO/GRPO 均有大幅提升
- **子目标奖励优于二元奖励**：GRPO+子目标 (53.21%) >> DPO+二元 (35.99%)
- **一步前瞻的局限**：在需要多步证明的定理上性能下降，暗示需要多步前瞻

## 亮点与洞察
- **验证器即奖励信号**：形式化验证器天然提供正确性保证的步级反馈，这在代码生成、数学推理等有外部验证器的场景中都可借鉴
- **DPO 失败的深刻教训**：即使偏好对质量合理，DPO 仍可能学到虚假相关（如长度偏差），这对所有使用 DPO 的工作都是警示
- **密集奖励 > 稀疏奖励**：用 sub-goal 数做密集奖励的思路可以推广到任何有中间状态变化的 RL 场景

## 局限与展望
- **仅限 ReProver 架构**：未验证在其他 ATP 模型上的通用性
- **一步前瞻局限**：长证明需要多步前瞻，当前只看一步
- **计算成本高**：40 A100 GPU-days，80% 是 Lean 交互
- **跨域泛化弱**：在 MiniF2F (36.9% vs 37.7%) 和 ProofNet (14.4% vs 15.3%) 上略低于基线
- 可改进：(a) 引入 n-step 前瞻或蒙特卡洛树搜索；(b) 用更高效的 Lean 交互方式降低 CPU 瓶颈

## 相关工作与启发
- **vs ReProver**: 基线模型，LeanListener 在其基础上加入步级验证反馈
- **vs LLM 数学推理中的 PRM**: PRM 用人工/MCTS 标注过程奖励，LeanListener 用验证器直接获取——成本更低且保证正确
- **vs DeepSeek-Prover**: DeepSeek 用合成数据+RL 训练证明器，LeanListener 关注验证器反馈设计而非数据规模

## 评分
- 新颖性: ⭐⭐⭐⭐ 验证器步级反馈用于ATP训练是新颖的，DPO失败分析有价值
- 实验充分度: ⭐⭐⭐⭐ 多种方法对比+详细消融+速度分析
- 写作质量: ⭐⭐⭐⭐ 方法和实验描述清晰
- 价值: ⭐⭐⭐⭐ 对ATP和密集奖励RL都有借鉴意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] EvolProver: Advancing Automated Theorem Proving by Evolving Formalized Problems via Symmetry and Difficulty](../../ICLR2026/llm_reasoning/evolprover_advancing_automated_theorem_proving_by_evolving_formalized_problems_v.md)
- [\[ICML 2026\] From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates](../../ICML2026/llm_reasoning/from_llm-generated_conjectures_to_lean_formalizations_automated_polynomial_inequ.md)
- [\[ACL 2025\] Marco-o1 v2: Towards Widening The Distillation Bottleneck for Reasoning Models](marco-o1_v2_towards_widening_the_distillation_bottleneck_for_reasoning_models.md)
- [\[ACL 2025\] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)
- [\[ACL 2025\] Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)

</div>

<!-- RELATED:END -->
