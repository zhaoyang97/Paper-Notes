---
title: >-
  [论文解读] MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure
description: >-
  [ICLR 2026][LLM效率][LoRA] MiSS 把 LoRA 的双矩阵 $BA$ 更新换成由单个零初始化小矩阵 $D$ "扩展"出来的分片共享结构，既加快收敛又在显存和算力上同时占优，从而在性能–显存–效率三角中取得更好的折中。 - 领域现状：LoRA 用低秩分解 $\Delta W \approx BA$ 把可…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "LoRA"
  - "参数高效微调"
  - "单矩阵"
  - "分片共享"
  - "收敛性"
  - "Pareto 前沿"
---

# MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=gohmWoUSoS](https://openreview.net/forum?id=gohmWoUSoS)  
**代码**: [https://github.com/Joluck/MiSS](https://github.com/Joluck/MiSS)  
**领域**: LLM 高效微调 (PEFT / LoRA)  
**关键词**: LoRA, 参数高效微调, 单矩阵, 分片共享, 收敛性, Pareto 前沿  

## 一句话总结
MiSS 把 LoRA 的双矩阵 $BA$ 更新换成由单个零初始化小矩阵 $D$ "扩展"出来的分片共享结构，既加快收敛又在显存和算力上同时占优，从而在性能–显存–效率三角中取得更好的折中。

## 研究背景与动机
- **领域现状**：LoRA 用低秩分解 $\Delta W \approx BA$ 把可训练参数压到极小，是当下最主流的 PEFT 方法。围绕它出现了两条改进路线——一条提升**适配性**（用 PiSSA/LoRA-GA/DoRA 等更好的初始化逼近全量微调的收敛速度），一条提升**效率**（用 VeRA/MoS 等共享/压缩参数降低显存与算力）。
- **现有痛点**：两条路线几乎是按下葫芦浮起瓢。PiSSA 这类靠 SVD 初始化的方法适配性好但初始化耗时长、且不一定兼容某些优化器；VeRA/MoS 这类效率方法初始化快、显存省，但进一步分解 LoRA 矩阵会削弱表达能力，性能掉点。更隐蔽的是，AdaLoRA/DoRA/VeRA 虽然把可训练参数数 (TPs) 砍下来了，却仍沿用 $B(Ax)$ 的串行矩阵乘逻辑，**空间复杂度和 FLOPs 依旧被 $O((d+k)\times r)$ 卡死**——"参数变少" ≠ "算得更快、占得更省"。
- **核心矛盾**：性能、显存、效率三者难以同时兼顾，现有变体总是在某一维度受益、另一维度受损。
- **本文目标**：找到一个能在性能、显存、计算效率三个维度同时占优、落在 Pareto 前沿上的 PEFT 结构。
- **核心 idea**（**单矩阵简化优化**）：作者重新审视 LoRA 慢收敛的根源，发现 $B$ 和 $A$ **必须同时更新**显著增加了优化复杂度（呼应 S2FT 固定一个矩阵、LoRA+ 用差异学习率的观察）。由此假设：**只训练单个矩阵即可在不牺牲表达力的前提下简化优化**，并用初始梯度范数实验佐证——MiSS 和其他快收敛变体一样，初始梯度范数明显大于原始 LoRA。

## 方法详解

### 整体框架
MiSS 抛弃 LoRA 的双矩阵结构，只保留一个零初始化的小矩阵 $D$，把权重更新写成 $\Delta W = \text{expand}(D)$——通过"扩展/复制"算子把小矩阵铺成与 $W_0$ 同形的大矩阵。这个结构天然低秩（重复行使有效秩很小），既保住了低秩特性又把更新从"双矩阵乘"简化为"单矩阵扩展"。为避免显式构造大矩阵带来的算力/显存开销，作者再给出等价的高效形式 MiSSe，把"扩展输出维"转写成"聚合输入维"，从而做到训练和初始化都又快又省。

```mermaid
flowchart LR
    subgraph LoRA
        x1[x] --> A[A r×k] --> B[B d×r] --> o1[BAx]
    end
    subgraph MiSS["MiSS (概念形式)"]
        D0[D N×k\nzero-init] --> EX[expand 复制分片] --> dW[ΔW d×k] --> o2[ΔWx]
    end
    subgraph MiSSe["MiSSe (高效形式)"]
        x2[x b×l×k] --> SUM[分块求和 S=Σ x_g] --> DS[D·S, D∈R^{d×r}] --> o3[ΔWx = DS]
    end
```

### 关键设计

**1. 分片共享的单矩阵扩展（MiSS 概念形式）：把"重复"当作低秩。** 作者观察到，一个由小矩阵重复拼出来的大矩阵本身就是低秩的——若每个分片内的行都相同，则整体秩至多为分片数 $N$。据此把输出维 $d$ 切成 $N$ 个分片 $\{s_1,\dots,s_N\}$（$\sum s_i=d$），取小矩阵 $D\in\mathbb{R}^{N\times k}$，让第 $i$ 个分片的更新等于 $D$ 的第 $i$ 行 $D_i$ 纵向重复 $s_i$ 次：
$$(\text{expand}(D))^\top = [(\mathbf{1}_{s_1}D_1)^\top\ (\mathbf{1}_{s_2}D_2)^\top\ \dots\ (\mathbf{1}_{s_N}D_N)^\top]$$
前向变成 $y = W_0x + \text{expand}(D)x$，参数量从 $d\times k$ 降到 $N\times k$。$D$ 零初始化保证训练起点 $\Delta W=0$，不破坏预训练输出。

**2. 输入聚合的等价高效形式（MiSSe）：用分块求和换掉矩阵乘。** 概念形式直接算 $\text{expand}(D)x$ 时间复杂度 $O(bldk)$、显存 $O(dk)$，仍然昂贵。作者把视角从"切输出维"翻转为"切输入维"：重定义 $D\in\mathbb{R}^{d\times r}$，把输入维 $k$ 切成 $r$ 个块、每块大小 $g=\lfloor k/r\rfloor$，对每块沿 $k$ 维求和得到聚合向量
$$S = \sum_{i=1}^{g} x^{(i)} \in \mathbb{R}^{b\times l\times r}, \qquad \Delta W x = DS,\quad y = W_0x + DS.$$
这恰好隐式实现了 $\text{expand}(D)x = DS$，但只需存 $D\in\mathbb{R}^{d\times r}$、不必显式构造 $d\times k$ 大矩阵。实现上极其轻量（伪代码核心只有一行 `result + x @ self.D.expand(...)`），其精髓是"输入分块求和"利用了输入的局部冗余，在保留关键信息的同时压低了计算维度。

**3. 计算分解视角：把效率优势精确定位到输入变换那一步。** 作者把更新拆成两阶段——CStep1 输入变换、CStep2 输出投影——并指出 $D$ 在维度上与 LoRA 的 $B$ 对齐（都对应输出维 $d$、都是输出投影矩阵），所以 CStep2 两者同为 $d\times r$ 的矩阵乘、开销一致；差异**完全集中在 CStep1**：LoRA 要做昂贵的 $Ax$（$O(BLkr)$），MiSSe 只做廉价的分块求和 $\text{sum}(x)$（$O(BLk)$）。由此 MiSSe 的总 FLOPs 从 $O(BL(kr+rd))$ 降到 $O(BL(k+rd))$，空间复杂度 $O(d\times r)$，在 Table 2 的对比中是唯一真正打破 $O((d+k)\times r)$ 下限的方法——这正是"参数少不等于算得快"困局的破解点。

## 实验关键数据

### 主实验表格

NLU（RoBERTa-base，GLUE 子集，可训练比例 0.236%）：

| 方法 | MNLI | SST-2 | CoLA | QNLI | MRPC | Avg |
|------|------|-------|------|------|------|-----|
| LoRA | 85.63 | **94.03** | 62.40 | 91.37 | 87.98 | 84.28 |
| PiSSA | **85.72** | 93.64 | 67.28 | 91.40 | 88.11 | 85.23 |
| **MiSS** | 85.71 | 93.60 | **72.86** | **91.43** | **88.14** | **86.35** |

NLG（多模型，GSM8K/Math/HumanEval/Mbpp 平均，节选）：

| 模型 | 方法 | Trainable | Math | Avg |
|------|------|-----------|------|-----|
| Llama2-7B | LoRA / PiSSA / **MiSS** | 89.9M / 89.9M / **87.0M** | 5.22 / 6.92 / **8.58** | 24.72 / 27.70 / **29.30** |
| Mistral-7B | LoRA / PiSSA / **MiSS** | 94.4M / 94.4M / **87.0M** | 15.82 / 18.13 / **18.85** | 40.12 / 44.45 / **47.79** |
| Llama2-13B | LoRA / PiSSA / **MiSS** | 250M / 250M / 255M | 12.60 / 13.82 / **15.74** | 34.60 / 39.52 / **42.11** |
| Qwen3-4B | LoRA / PiSSA / **MiSS** | 74.3M / 74.3M / **70.1M** | 15.20 / 26.00 / **34.82** | 62.79 / 66.21 / **68.22** |

跨 5 个主流 LLM，MiSS 普遍取得最优/次优平均分，且在复杂推理任务（Math）上提升尤为明显（Qwen3-4B Math 从 PiSSA 的 26.00 跳到 34.82），同时可训练参数往往还更少。

### 消融实验表格

复杂度对比（Table 2，$d$ 输出维 / $k$ 输入维 / $r$ 秩）：

| 方法 | 空间复杂度 | FLOPs | 可训练参数 |
|------|-----------|-------|-----------|
| FT | $O(dk)$ | $O(dk)$ | $d\cdot k$ |
| LoRA | $O((d+k)r)$ | $O((d+k)r)$ | $(d+k)\cdot r$ |
| AdaLoRA | $O((d+k+r)r)$ | $O((d+k+r)r)$ | $(d+k)r+r^2$ |
| LoHA | $O(2r(d+k))$ | $O(2r(d+k))$ | $2(d+k)r$ |
| VeRA | $O((d+k)r+r+d)$ | 同左 | $d+r$ |
| **MiSSe** | $O(d\times r)$ | $O(k+d\times r)$ | $d\cdot r$ |

视觉任务（VTAB-1K）：MiSS 图像平均 88.02、视频 72.96，与 LoRA/DoRA 持平甚至更优，但参数预算只有约 0.4 #TPs（LoRA/DoRA 约 0.8 #TPs），证明效率优势可迁移到多模态。

### 关键发现
- **初始梯度范数**：MiSS 的初始梯度范数明显大于原始 LoRA、接近全量微调，早期收敛更快，验证了"单矩阵优化更简单"的假设（Figure 1）。
- **No Free Lunch 实验**：在受控单层 MLP 上，PiSSA 这类 SVD 方法适配性好但初始化时间随参数量飙升；VeRA/AdaLoRA 初始化快但适配性弱；MiSS 在初始化时间、训练时间、最小验证损失三条曲线上都落在有利位置。
- **Pareto 前沿**：综合性能、显存、效率三维，MiSS 是唯一既保住适配性、又同时把空间复杂度/FLOPs 压到 $O(d\times r)$ 量级的方法。

## 亮点与洞察
- **把"重复=低秩"用到极致**：用一个被复制铺开的小矩阵代替低秩乘积，结构上比 $BA$ 更简单，却保留了同等的低秩约束。
- **诊断切中要害**："参数少 ≠ 算得快"——指出多数 LoRA 变体只压参数量却没改串行矩阵乘逻辑，FLOPs/显存依旧受限，这个观察本身就很有价值。
- **概念形式与高效形式的双重等价**：$\text{expand}(D)x = DS$ 把"切输出维"和"聚合输入维"统一起来，既好初始化又好计算，工程落地几乎零成本（已并入 HuggingFace PEFT）。
- **可零推理开销 + 可扩展服务**：单矩阵结构与 $W_0$ 同形可融合，保留 LoRA 的推理零开销特性。

## 局限与展望
- **与初始化/优化器类方法正交未充分验证**：作者明确说没把 LoRA-GA、LoRA+ 纳入对比，认为 MiSS 与它们正交、可叠加，但缺乏组合实验佐证实际收益。
- **分片粒度的影响**：分片数 $N$/秩 $r$ 与表达能力、收敛速度之间的定量关系讨论较浅，分片划分（等分 vs 非等分）对不同层是否需要自适应仍是开放问题。
- **表达力上界**：分片共享强制同一分片内行相同，理论上限制了 $\Delta W$ 的自由度，在某些需要细粒度方向调整的任务上是否会触顶值得进一步探究。
- **大规模/超长上下文场景**：实验集中在 7B–13B 规模，更大模型与长序列下 blockwise 求和的数值稳定性与收益尚待检验。

## 相关工作与启发
- **适配性路线**：PiSSA（SVD 初始化）、LoRA-GA（梯度对齐初始化）、DoRA（幅度/方向解耦）、OLoRA（QR 正交初始化）——靠改初始化拉大初始梯度逼近全量微调，但引入昂贵预处理。
- **效率路线**：VeRA（冻结随机矩阵 + 共享）、MoS/ProLoRA（参数共享/分块）——省显存但削表达力。MiSS 的启发在于：**与其在 $BA$ 框架内反复打补丁，不如从"单矩阵 + 重复=低秩"重构整个更新算子**，同时拿下两条路线的好处。
- **对 S2FT/LoRA+ 的回应**：S2FT 固定一个矩阵降自由度、LoRA+ 用差异学习率，二者都暗示"双矩阵同更新"是慢收敛根源；MiSS 把这一洞察推到极致——干脆只留一个矩阵。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — "重复矩阵即低秩 + 单矩阵扩展"的视角清新，把输出维分片与输入维聚合的等价转换设计巧妙，是对 LoRA 结构的实质性重构而非微调初始化。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 NLU/NLG/视觉多域、5 个主流 LLM、复杂度理论分析 + Pareto 前沿 + 初始梯度范数验证，证据链完整；扣分在于未与 LoRA-GA/LoRA+ 做组合实验、分片粒度消融偏浅。
- **写作质量**: ⭐⭐⭐⭐ — 动机层层递进（慢收敛根源→单矩阵假设→梯度范数佐证），Table 1 把各变体前向/初始化一图说清，计算分解表格定位效率来源清晰。
- **价值**: ⭐⭐⭐⭐ — 真正在性能–显存–效率三角上同时占优、且已并入 PEFT 生态，对实际部署 LoRA 微调有直接落地价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LoRAGen: Structure-Aware Weight Space Learning for LoRA Generation](loragen_structure-aware_weight_space_learning_for_lora_generation.md)
- [\[ICLR 2026\] PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models](plop_precise_lora_placement_for_efficient_finetuning_of_large_models.md)
- [\[ICLR 2026\] LoRA-S: An Efficient Low Rank Adaptation scheme via Sylvester equation](lora-s_an_efficient_low_rank_adaptation_scheme_via_sylvester_equation.md)
- [\[ICLR 2026\] On-the-Fly Adaptation to Quantization: Configuration-Aware LoRA for Efficient Fine-Tuning of Quantized LLMs](on-the-fly_adaptation_to_quantization_configuration-aware_lora_for_efficient_fin.md)
- [\[ICLR 2026\] Merge before Forget: A Single LoRA Continual Learning via Continual Merging](merge_before_forget_a_single_lora_continual_learning_via_continual_merging.md)

</div>

<!-- RELATED:END -->
