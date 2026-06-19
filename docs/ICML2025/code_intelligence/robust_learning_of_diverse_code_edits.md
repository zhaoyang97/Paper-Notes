---
title: >-
  [论文解读] Robust Learning of Diverse Code Edits (NextCoder)
description: >-
  [ICML 2025][代码智能][代码编辑] 提出合成代码编辑数据生成流水线 + 鲁棒自适应算法 SeleKT（Selective Knowledge Transfer），通过在微调过程中周期性地对任务向量做 top-k 稀疏投影，使模型在获得强代码编辑能力的同时保留原始代码生成与通用推理能力，得到的 NextCoder 系列模型在五个代码编辑基准上超越同规模甚至更大模型。
tags:
  - "ICML 2025"
  - "代码智能"
  - "代码编辑"
  - "合成数据"
  - "鲁棒微调"
  - "选择性知识迁移"
  - "灾难性遗忘"
  - "SeleKT"
---

# Robust Learning of Diverse Code Edits (NextCoder)

**会议**: ICML 2025  
**arXiv**: [2503.03656](https://arxiv.org/abs/2503.03656)  
**代码**: [aka.ms/nextcoder](https://aka.ms/nextcoder)  
**领域**: NLP生成 / 代码编辑  
**关键词**: 代码编辑, 合成数据, 鲁棒微调, 选择性知识迁移, 灾难性遗忘, SeleKT

## 一句话总结

提出合成代码编辑数据生成流水线 + 鲁棒自适应算法 SeleKT（Selective Knowledge Transfer），通过在微调过程中周期性地对任务向量做 top-k 稀疏投影，使模型在获得强代码编辑能力的同时保留原始代码生成与通用推理能力，得到的 NextCoder 系列模型在五个代码编辑基准上超越同规模甚至更大模型。

## 研究背景与动机

### 问题定义

代码编辑（Code Editing）是软件工程中最基础的操作——根据自然语言指令修改现有代码（修 bug、优化性能、提升安全性等）。现有代码 LM（尤其 < 16B 的开源模型）在代码编辑任务上表现欠佳，主要受限于：

**训练数据质量差**：现有方法依赖 GitHub commit 数据，但 commit message 往往含义模糊、质量参差不齐，且编辑类型单一

**灾难性遗忘**：在代码编辑数据上微调后，模型的代码生成、指令遵循、数学推理等预训练能力显著下降

**编辑粒度受限**：已有合成方法（如 InstructCoder）仅覆盖函数级 Python 代码片段，无法处理类级/文件级多语言场景

### 核心挑战

如何在代码编辑专项性能和预训练通用能力之间取得最佳平衡：
- 全参数微调（SFT）→ 过拟合 + 灾难性遗忘
- LoRA 等 PEFT 方法 → 事先固定可训练参数，无法根据任务动态调整
- 模型合并（TIES）→ 一次性后处理，缺少训练过程中的动态约束

## 方法详解

### 整体框架

方法由两大模块组成：**(1) 多样化合成数据生成流水线** + **(2) SeleKT 鲁棒自适应算法**。

### 模块一：合成代码编辑数据生成

流水线以种子代码为输入，经 4 个阶段生成高质量代码编辑训练样本：

#### 阶段 1：问题描述 + 源代码生成
- 使用 GPT-4o 或 Llama-3.3-70B 作为生成器
- 输入：种子代码（来自 StarCoder 数据集，筛选 >10 行且包含循环/函数/条件/类的文件）+ 代码粒度要求（函数/类/文件级）+ 待改进方面（bug 修复、延迟优化、安全性等）
- 输出：带有**预设缺陷**的源代码 + 缺陷元数据

#### 阶段 2：目标代码生成
- 基于阶段1的问题描述、源代码和缺陷元数据，生成修复后的目标代码
- 同时输出编辑解释，为后续指令生成提供依据

#### 阶段 3：指令生成
- 基于源代码、目标代码和编辑解释生成自然语言编辑指令
- **四种风格**：简洁（Concise）、详细（Detailed）、人类口语（Human）、对话式（Conversational）
- 每个样本产生四条不同风格的微调数据

#### 阶段 4：质量过滤
- LLM 对样本从 5 个维度打分（0-10）：编辑正确性、指令一致性、代码质量、指令质量、微调价值
- 保留标准：平均分 ≥ 7 且每项 > 5

#### 数据规模

最终生成 **127K 合成样本（229M tokens）**，覆盖 8 种编程语言，结合 127K 条 CommitPackFT 真实 commit 数据。

| 语言 | GPT-4o | Llama-3.3-70B | 总计 | Tokens(M) |
|------|--------|---------------|------|-----------|
| Python | 8,406 | 6,963 | 15,279 | 28.63 |
| C | 7,039 | 10,114 | 17,153 | 33.48 |
| C++ | 6,272 | 11,065 | 17,337 | 30.93 |
| Java | 6,447 | 9,881 | 16,328 | 27.61 |
| JavaScript | 7,367 | 8,663 | 16,030 | 25.92 |
| Rust | 4,701 | 11,737 | 16,438 | 30.43 |
| Go | 4,503 | 10,701 | 15,204 | 28.56 |
| Kotlin | 3,470 | 9,802 | 13,272 | 22.16 |
| **总计** | **48,205** | **78,926** | **127,041** | **227.72** |

### 模块二：SeleKT 鲁棒自适应算法

#### 核心思想

SeleKT 的关键洞见：**哪些参数需要更新不应事先确定，而应在训练过程中根据任务难度动态评估**。

算法交替执行两步操作：
1. **Dense Gradients（稠密梯度）**：对所有参数做全参微调，获得最优更新方向
2. **Sparse Projection（稀疏投影）**：计算任务向量 $\tau = \theta - \theta_{\text{base}}$，保留变化幅度最大的 top-$\alpha N$ 个参数，将其余参数重置回基模型权重

#### 数学形式化

将鲁棒自适应建模为带 $L_0$ 约束的优化问题：

$$\arg\min_{\theta} \mathcal{L}(\theta) \quad \text{s.t.} \quad \|\theta - \theta_{\text{base}}\|_0 \leq c$$

其中 $\mathcal{L}$ 为 next-token prediction 交叉熵损失，$c$ 控制可更新参数数量。

#### 算法流程（Algorithm 1）

```
输入：基模型 θ_base，训练数据 D，总轮次 E，周期 M，稀疏度 α
1. 初始化 θ ← θ_base
2. for epoch e = 1 to E:
3.   for each minibatch D[s]:
4.     θ ← TrainStep(θ, D[s])      # 稠密梯度更新
5.     if s mod M == 0:
6.       τ ← θ - θ_base             # 计算任务向量
7.       γ[i] = 1 if i ∈ top-k(|τ|, ⌊αN⌋) else 0  # 构建掩码
8.       θ ← θ_base + γ ⊙ τ         # 稀疏投影
9. return θ as θ_FT
```

#### 关键设计细节

- **稀疏度 α = 0.05**（per layer）：仅保留每层 5% 参数的更新，实验表明这是最优选择
- **周期 M = 1 epoch**：每轮结束时执行一次投影，频率过高反而有害
- **全局选择**：top-k 选择是全局的，不受限于特定层或结构，优于手动指定层的做法
- **固定基模型**：始终以原始基模型为锚点计算任务向量（而非滑动基准），确保最终模型与预训练权重距离有严格 $L_0$ 上界

### 训练策略

- **基模型**：QwenCoder-2.5-Instruct（3B/7B/14B/32B）、DeepSeekCoder-6.7B-Instruct
- **优化器**：AdamW，学习率 $10^{-5}$，WarmupLR（warmup ratio 0.1）
- **训练轮次**：3 epochs
- **序列长度**：DeepSeekCoder 8192；QwenCoder 16384（使用 sample packing）
- **硬件**：8 × NVIDIA H100 80GB，每 epoch 约 6 小时

## 实验与关键数据

### 评估基准

涵盖代码编辑、代码生成、通用能力三类共 9 个基准：

| 基准 | 任务类型 | 粒度 | 样本数 |
|------|---------|------|--------|
| CanItEdit | Bug 修复 | 类级 | 210 |
| HumanEvalFix | Bug 修复 | 函数级 | 164 |
| NoFunEval | 代码改进 | 文件级 | 397 |
| Aider | Bug 修复（对话式） | 文件级 | 133 |
| Aider Polyglot | Bug 修复（多语言） | 文件级 | 225 |
| HumanEval+ | 代码生成 | 函数级 | 164 |
| MBPP+ | 代码生成 | 函数级 | 378 |
| GSM8K | 数学推理 | - | 1,320 |
| MMLU | 多领域知识 | - | 3,150 |

### 主实验：代码编辑性能（Table 4 精选）

| 模型 | HumanEvalFix | CanItEdit | Aider |
|------|-------------|-----------|-------|
| GPT-4o | 90.2 | 59.5 | 74.4 |
| QwenCoder-2.5-32B | 90.2 | 60.9 | 75.2 |
| Llama-3-70B-Inst | 77.4 | 56.7 | 51.1 |
| DeepSeekCoder-33B | 74.4 | 49.5 | 58.6 |
| QwenCoder-2.5-7B | 73.8 | 48.1 | 59.4 |
| QwenCoder-2.5-7B-SFT | 70.1 | 36.7 | 48.9 |
| QwenCoder-2.5-7B-LoRA | 70.7 | 44.3 | 40.6 |
| QwenCoder-2.5-7B-TIES | 79.5 | 47.0 | 60.2 |
| **NextCoder-7B (SeleKT)** | **81.1** | **50.5** | **65.7** |

**关键发现**：
- NextCoder-7B 在所有 7B 级模型中最优，超越 DeepSeekCoder-V2-16B（2× 参数）
- SFT 和 LoRA 微调后反而比原模型更差，证实灾难性遗忘问題严重
- SeleKT 在 Aider 上比原模型提升 +6.3%，比 LoRA 提升 +25.1%

### 不同尺寸模型效果（Table 5）

| 模型 | HumanEvalFix | CanItEdit | Aider | Aider Polyglot |
|------|-------------|-----------|-------|----------------|
| QwenCoder-2.5-3B | 73.2 | 37.1 | 36.8 | - |
| NextCoder-3B | 75.6 | 42.4 | 37.6 | - |
| QwenCoder-2.5-14B | 87.8 | 58.1 | 66.9 | 9.3 |
| NextCoder-14B | 89.8 | 60.2 | 72.2 | 12.2 |
| QwenCoder-2.5-32B | 90.2 | 61.0 | 72.9 | 16.4 |
| NextCoder-32B | 88.9 | 62.4 | 74.7 | 21.9 |

NextCoder-32B 在 Aider Polyglot 上超越 GPT-4o（18.2%），达到 21.9%。

### 预训练能力保持（Table 6）

| 模型 | HumanEval+ | MBPP+ |
|------|-----------|-------|
| QwenCoder-2.5-7B | 85.4 | 72.5 |
| QwenCoder-2.5-7B-SFT | 79.3 | 67.2 |
| QwenCoder-2.5-7B-LoRA | 81.7 | 70.9 |
| QwenCoder-2.5-7B-TIES | 82.3 | 71.7 |
| **NextCoder-7B** | **84.8** | **72.0** |

SFT 在 HumanEval+ 上下降 6.1%，而 SeleKT 仅下降 0.6%，几乎完全保留代码生成能力。

### 稀疏度消融（Table 9）

| α（稀疏度） | HumanEvalFix | CanItEdit | Aider |
|------------|-------------|-----------|-------|
| 0.05 | 81.1 | **50.5** | **65.7** |
| 0.2 | 76.8 | 45.7 | 53.4 |
| 0.5 | 81.7 | 43.3 | 54.9 |

α = 0.05（最稀疏）整体最优，验证了 $L_0$ 约束越紧越能防止过拟合。

## 亮点与洞察

1. **"先全参更新、再稀疏投影"范式**：与 LoRA 等 PEFT 方法"先固定参数、再训练子集"的思路相反，SeleKT 先用全参梯度找到最优方向，再用稀疏投影截断，兼得高表达力和强正则化
2. **动态参数选择**：top-k 掩码在训练中周期性更新，不同阶段可能选择不同的参数子集，适应性远强于静态方法
3. **理论保证**：SeleKT 确保最终模型与基模型在 $L_0$ 范数下的距离有严格上界（Lemma 1），这是 LoRA/SFT 不具备的
4. **多维度数据多样性**：合成数据在代码粒度（函数/类/文件）、编辑类型（6 类）、指令风格（4 种）、编程语言（8 种）上实现全覆盖
5. **合成 vs. 真实数据互补**：单独使用 CommitPackFT 效果有限（Aider 仅 21.1%），加入合成数据后达到 33.8%（+60%），两者结合效果最佳

## 局限性

1. **计算成本较高**：虽然稀疏投影本身高效，但稠密梯度步要求全参微调，在训练效率上不如 LoRA（需 8×H100 训练约 18 小时）
2. **合成数据依赖强模型**：数据生成依赖 GPT-4o 和 Llama-3.3-70B，质量受限于生成器能力，存在数据同质化风险
3. **评估覆盖有限**：主要聚焦代码编辑，未验证 SeleKT 在其他领域（数学推理、自然语言任务）的鲁棒微调效果
4. **超参数敏感性**：α 和 M 对性能影响显著（α=0.05 vs. 0.2 在 Aider 上相差 12.3%），需要额外调参成本
5. **编辑场景仍有盲区**：未涵盖跨仓库编辑（如 SWE-Bench 类型的 repo-level 修复）和重构类编辑

## 相关工作

- **代码编辑模型**：OctoCoder、EditCoder 基于 commit 数据微调；SWE-Fixer 针对 GitHub issue 做微调；本工作通过合成数据 + 鲁棒算法取得更好的泛化效果
- **代码数据合成**：OSS-Instruct（Magicoder）和 Self-CodeAlign 以种子代码为基础但仅覆盖函数级；InstructCoder 是唯一面向代码编辑的合成方法但局限于 Python 短片段；本工作首次实现多语言、多粒度、多风格的代码编辑合成
- **抗灾难性遗忘**：LoRA 冻结基模型参数添加低秩适配器；TIES 在模型合并中做稀疏剪枝；Nguyen et al. 的稀疏适配先验选择参数并仅训练稀疏梯度；SeleKT 的核心区别在于"稠密梯度 + 周期性稀疏投影"

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | SeleKT 的"全参训练→稀疏投影"范式简洁但有效，与主流 PEFT 方向相反 |
| 技术深度 | 4 | 数据流水线设计完备，算法有理论保证（$L_0$ 上界），实验消融全面 |
| 实验充分度 | 5 | 9 个基准、5 个模型尺寸、4 种微调方法对比、多项消融 |
| 实用价值 | 4 | 开源模型+数据+代码，NextCoder-7B 可直接用于代码编辑场景 |
| 写作质量 | 4 | 结构清晰，图表丰富，动机-方法-实验逻辑顺畅 |
| **总分** | **4.2** | 扎实的系统性工作，在代码编辑微调方向提供了简洁有效的解 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](../../ICLR2026/code_intelligence/shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](../../NeurIPS2025/code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[ACL 2025\] ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation](../../ACL2025/code_intelligence/reflectioncoder_learning_from_reflection_sequence_for_enhanced_one-off_code_gene.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](../../ICLR2026/code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)

</div>

<!-- RELATED:END -->
