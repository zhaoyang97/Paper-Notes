---
title: >-
  [论文解读] DiaBlo: Diagonal Blocks Are Sufficient For Finetuning
description: >-
  [ICLR2026][代码智能][PEFT] 提出 DiaBlo——一种用对角块更新替代低秩分解的参数高效微调方法：将权重矩阵划分为 $N \times N$ 块后只训练对角块 $\mathbf{D}_1, \ldots, \mathbf{D}_N$，彻底绕开 LoRA 中 $\mathbf{AB}$ 乘积带来的非凸优化、初始化敏感与梯度不稳定问题，零初始化即可收敛，PyTorch 一行 `torch.einsum` 实现 batched matmul，理论证明同参数预算下表达力严格优于 LoRA，在常识推理、算术推理、代码生成、安全对齐四大任务及 4-bit/2-bit 量化场景全面领先。
tags:
  - "ICLR2026"
  - "代码智能"
  - "PEFT"
  - "diagonal blocks"
  - "LoRA alternative"
  - "LLM fine-tuning"
  - "parameter efficiency"
---

# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning

**会议**: ICLR2026  
**arXiv**: [2506.03230](https://arxiv.org/abs/2506.03230)  
**代码**: [ziyangjoy/DiaBlo](https://github.com/ziyangjoy/DiaBlo)  
**领域**: 代码智能  
**关键词**: PEFT, diagonal blocks, LoRA alternative, LLM fine-tuning, parameter efficiency

## 一句话总结

提出 DiaBlo——一种用对角块更新替代低秩分解的参数高效微调方法：将权重矩阵划分为 $N \times N$ 块后只训练对角块 $\mathbf{D}_1, \ldots, \mathbf{D}_N$，彻底绕开 LoRA 中 $\mathbf{AB}$ 乘积带来的非凸优化、初始化敏感与梯度不稳定问题，零初始化即可收敛，PyTorch 一行 `torch.einsum` 实现 batched matmul，理论证明同参数预算下表达力严格优于 LoRA，在常识推理、算术推理、代码生成、安全对齐四大任务及 4-bit/2-bit 量化场景全面领先。

## 研究背景与动机

**领域现状**：LoRA 及其变体（DoRA、PiSSA、MiLoRA、LoRA-GA）是当前主流 PEFT 方法。它们在预训练权重旁注入可训练的低秩矩阵乘积 $\Delta\mathbf{W} = \mathbf{AB}$，从而大幅减少可训练参数量。Prompt Tuning / Prefix Tuning 等早期方法虽轻量但表达力有限；Adapter 方法需修改模型结构并引入推理延迟。

**现有痛点**：

1. **非凸优化困难**：LoRA 的 $\mathbf{AB}$ 乘积使目标函数关于 $\mathbf{A}, \mathbf{B}$ 非凸，梯度 $\mathbf{g}_{\mathbf{A}} = \mathbf{g}_{\mathbf{W}} \mathbf{B}^\top$, $\mathbf{g}_{\mathbf{B}} = \mathbf{A}^\top \mathbf{g}_{\mathbf{W}}$ 互相依赖——导致对初始化极度敏感、收敛不稳定
2. **变体复杂度膨胀**：DoRA 解耦幅度/方向、PiSSA 用大奇异值初始化、MiLoRA 用小奇异值、LoRA-GA 对齐首步梯度……实质都是为矩阵乘积结构打补丁，增加了算法与工程复杂度
3. **稀疏方法硬件不友好**：基于 unstructured sparsity 的微调（随机 mask / 重要性选择）虽然避开了低秩分解，但导致不规则内存访问，GPU 利用率低

**核心洞察**：对角块的梯度 $\mathbf{g}_{\mathbf{D}_i} = \mathbf{X}_i^\top \mathbf{g}_{\mathbf{Y}_i}$ 恰好等于全量微调中对应子块 $\mathbf{W}_{ii}$ 的梯度——不经过任何矩阵乘积中间变量，因此零初始化不会梯度消失，优化景观也远比低秩参数化简单。

**核心 idea**：不做低秩分解，直接更新权重矩阵的 $N$ 个对角块 $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$，用 batched matmul 高效实现。

## 方法详解

### 整体框架

DiaBlo 把每个线性层的权重 $\mathbf{W} \in \mathbb{R}^{m_1 \times m_2}$ 切成 $N \times N$ 的块矩阵，冻结预训练权重 $\mathbf{W}_0$ 并在它旁边只训练一条"对角线"上的 $N$ 个块。前向变成 $\mathbf{Y} = \mathbf{X}\mathbf{W}_0 + \mathbf{X}\mathbf{D}$，其中 $\mathbf{D} = \text{diag}(\mathbf{D}_1, \ldots, \mathbf{D}_N)$ 是块对角适配矩阵，每个对角块 $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$（$d_1 = m_1/N$，$d_2 = m_2/N$）实际存成一个张量 $\mathcal{D} \in \mathbb{R}^{N \times d_1 \times d_2}$。整套设计的关键就在于：更新量直接落在原权重的子块上，而不像 LoRA 那样要先经过两个矩阵的乘积。

### 关键设计

**1. 对角块直接更新：用结构化稀疏绕开低秩乘积的非凸优化**

LoRA 把更新写成 $\Delta\mathbf{W} = \mathbf{AB}$，目标函数对 $\mathbf{A}, \mathbf{B}$ 是非凸的，二者的梯度 $\mathbf{g}_{\mathbf{A}} = \mathbf{g}_{\mathbf{W}} \mathbf{B}^\top$ 和 $\mathbf{g}_{\mathbf{B}} = \mathbf{A}^\top \mathbf{g}_{\mathbf{W}}$ 互相缠绕，所以它对初始化极度敏感、训练初期容易出现 $\mathbf{A}$ 矩阵梯度消失，DoRA/PiSSA/MiLoRA 等变体本质都是在给这个乘积结构打补丁。DiaBlo 干脆不做乘积，直接把可训练参数放进权重的对角子块 $\mathbf{D}_i$。此时对角块的梯度 $\mathbf{g}_{\mathbf{D}_i} = \mathbf{X}_i^\top \mathbf{g}_{\mathbf{Y}_i}$ 恰好等于全量微调里对应子块 $\mathbf{W}_{ii}$ 的梯度，不经过任何中间乘积变量，因此从全零初始化出发也不会梯度消失——这正是它无需任何初始化 trick 就能稳定收敛的根源。

**2. batched matmul 实现：让结构化稀疏在 GPU 上真正跑得快**

早期那些靠随机 mask 或重要性选块的稀疏微调虽然也避开了低秩分解，但稀疏模式不规则、内存访问跳来跳去，GPU 利用率很低。DiaBlo 的对角块结构是规整的，前向计算 $\mathbf{X}\mathbf{D}$ 不必真的拼出稀疏矩阵 $\mathbf{D}$，只要把 $\mathbf{X}$ reshape 成 $b \times N \times d_1$，一行 `torch.einsum("bNd1,Nd1d2->bNd2", X, D)` 就完成 batched matmul，反向传播同理。同等参数量下它的计算量 $bNd^2$ 与 LoRA 的 $2bmr$ 持平，因此训练速度和 LoRA 一样快，而比解耦幅度/方向的 DoRA 快约 2.8 倍（170 vs 480 min/epoch）。

**3. 表达力的理论保证：同参数预算下严格强于 LoRA**

在线性最小二乘设定下，DiaBlo 的最优解就是全量微调的解；并且达到这一点所需的参数量为 $Nd_1d_2 = m_1 m_2 / N \geq m_2 r$，而 LoRA 至少要 $(m_1 + m_2)r$ 个参数才能表达秩为 $r$ 的更新，于是在相同参数预算下 DiaBlo 的表达力是严格更强、而非近似更强。把视角推广到非线性网络，当激活 $\mathbf{X}$ 与输出梯度 $\mathbf{g}_{\mathbf{Y}}$ 满足低秩条件（这一条件已有文献的实验支持）时，DiaBlo 的驻点同样落在全量微调的驻点上，说明这套对角块参数化在实践中也不会牺牲掉可达到的解。

## 实验关键数据

### 常识推理（Commonsense Reasoning 170K, 8 个子任务平均）

| 模型 | 方法 | r/N | 可训练参数 | Avg Acc (%) |
|------|------|-----|-----------|-------------|
| LLaMA2-7B | Full FT | — | 100% | 83.5 |
| | LoRA | r=32 | 0.83% | 77.6 |
| | DoRA | r=16 | 0.42% | 80.5 |
| | MiLoRA | r=32 | 0.83% | 79.2 |
| | SMT(Best) | — | 4.91% | 83.4 |
| | **DiaBlo** | **N=128** | **0.52%** | **83.5** |
| LLaMA3-8B | Full FT | — | 100% | 87.5 |
| | LoRA | r=32 | 0.78% | 80.8 |
| | DoRA | r=32 | 0.78% | 85.2 |
| | SMT(Best) | — | 3.01% | 87.2 |
| | **DiaBlo** | **N=64** | **1.04%** | **87.3** |
| LLaMA-13B | DoRA | r=32 | 0.68% | 80.8 |
| | **DiaBlo** | **N=64** | **1.06%** | **84.9** |

### 算术推理（MetaMathQA → GSM8K + MATH, LLaMA2-7B）

| 方法 | r/N | 可训练参数 | GSM8K | MATH | Avg |
|------|-----|-----------|-------|------|-----|
| Full FT | — | 100% | 66.5 | 19.8 | 43.2 |
| LoRA | r=64 | 1.67% | 60.6 | 16.9 | 38.7 |
| PiSSA | r=64 | 1.67% | 58.2 | 15.8 | 37.0 |
| MiLoRA | r=64 | 1.67% | 63.5 | 17.8 | 40.7 |
| **DiaBlo** | **N=32** | **2.09%** | **66.3** | **20.4** | **43.4** |

### 代码生成与安全对齐（LLaMA3-8B）

| 方法 | r/N | 可训练参数 | Pass@1 | Pass@10 | HEx-PHI |
|------|-----|-----------|--------|---------|---------|
| LoRA | r=32 | 1.12% | 34.7 | 50.8 | 91.6 |
| DoRA | r=32 | 1.12% | 33.1 | 48.6 | 93.6 |
| LoRI | r=32 | 0.56% | 43.2 | 63.2 | 92.8 |
| **DiaBlo** | **N=64** | **1.51%** | **43.2** | **63.5** | **97.6** |

### 量化模型微调（Math10K, LLaMA2-7B, 4 任务平均）

| 量化精度 | 方法 | 可训练参数 | Avg Acc (%) |
|---------|------|-----------|-------------|
| 4-bit | QLoRA (r=64) | 112M | 53.7 |
| | ApiQ-bw (r=64) | 112M | 53.5 |
| | **MagR-DiaBlo (N=64)** | **70M** | **54.8** |
| 2-bit | QLoRA (r=64) | 112M | 2.1 |
| | GPTQ-LoRA (r=64) | 112M | 39.9 |
| | ApiQ-bw (r=64) | 112M | 47.3 |
| | **MagR-DiaBlo (N=64)** | **70M** | **48.7** |

### 稀疏模式对比（GSM8K, LLaMA3-8B, Sparsity 1/64）

| 稀疏模式 | 微调 Acc (%) | 训练时间 (min) |
|----------|-------------|---------------|
| **DiaBlo (对角块)** | **67.68** | **17.26** |
| Random Entries | 65.35 | 26.51 |
| Random Block | 64.86 | 29.76 |
| Random Column | 65.19 | 17.01 |
| Random Row | 61.71 | 17.76 |

### 关键发现

- **常识推理**：DiaBlo (N=128, 0.52% 参数) 在 LLaMA2-7B 上达到 83.5%，追平 Full FT，远超 LoRA 的 77.6%；SMT 用 4.91% 参数才勉强持平
- **算术推理**：DiaBlo (N=32) 在 MATH 上取得所有方法最高的 20.4%，超越 Full FT 的 19.8%
- **量化鲁棒性**：2-bit 下 QLoRA 几乎崩溃 (2.1%)，DiaBlo 仍保持 48.7%——差距达 46.6 个百分点
- **训练效率**：DiaBlo 与 LoRA 同等训练速度 (170 min/epoch)，DoRA 需 480 min/epoch（慢 2.8×）
- **结构化稀疏优势**：对角块在所有稀疏模式中准确率最高，且比非结构化方法快 1.5-1.7×
- **梯度稳定性**：DiaBlo 的梯度范数方差始终低于 LoRA，LoRA 在训练初期出现 $\mathbf{A}$ 矩阵梯度消失现象

## 亮点与洞察
- **极度简单但出人意料地有效**：zero init + 对角块更新→无需任何 trick
- **理论严格**：线性问题下严格优于 LoRA（不是近似优于）
- **量化友好**：对角块结构在低比特下比低秩乘积更鲁棒
- **LoRA 的优化困难是根本性的**：低秩矩阵乘积本身就是非凸难题，DiaBlo 完全绕过

## 局限与展望
- 对角块假设不考虑跨块信息——可能在需要全秩更新的任务上受限
- $N$ 的选择需要匹配硬件和参数预算
- 未与 adapter-based 方法系统对比

## 相关工作与启发
- **vs LoRA**：LoRA 用 $\mathbf{AB}$ 低秩近似 $\Delta \mathbf{W}$。DiaBlo 用结构化稀疏（对角块）—— 更稳定、更有表达力
- **vs S²FT**：也是结构化稀疏微调。DiaBlo 的对角块更规整→GPU 效率更高
- **vs QLoRA**：量化+LoRA。DiaBlo+量化的组合可能更好（2-bit 下优势明显）

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路极简但有效，理论支撑充分
- 实验充分度: ⭐⭐⭐⭐⭐ 多任务、多精度、多模型全面验证
- 写作质量: ⭐⭐⭐⭐ 理论和实验叙述清晰，图表直观
- 价值: ⭐⭐⭐⭐⭐ 可能替代 LoRA 成为新的 PEFT 默认选择

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ATGen: Adversarial Reinforcement Learning for Test Case Generation](atgen_adversarial_reinforcement_learning_for_test_case_generation.md)
- [\[ICLR 2026\] SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification](spotit_evaluating_text-to-sql_evaluation_with_formal_verification.md)
- [\[ICLR 2026\] Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning](critique-coder_enhancing_coder_models_by_critique_reinforcement_learning.md)
- [\[ICLR 2026\] BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization](boad_discovering_hierarchical_software_engineering_agents_via_bandit_optimizatio.md)
- [\[ICLR 2026\] AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions](aethercode_evaluating_llms_ability_to_win_in_premier_programming_competitions.md)

</div>

<!-- RELATED:END -->
