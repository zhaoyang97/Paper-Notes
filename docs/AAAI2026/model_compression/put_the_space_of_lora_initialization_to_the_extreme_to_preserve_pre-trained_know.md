---
title: >-
  [论文解读] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge
description: >-
  [AAAI2026][模型压缩][LoRA] 提出 LoRA-Null，将 LoRA 初始化在预训练知识 input activation 的 null space 中（而非权重的 null space），从信息论角度论证 activation 的 effective rank 远小于权重，因此其 null space 包含更少预训练知识信息，显著减轻微调时的灾难性遗忘。
tags:
  - "AAAI2026"
  - "模型压缩"
  - "LoRA"
  - "catastrophic forgetting"
  - "knowledge preservation"
  - "null space"
  - "activation-aware initialization"
---

# Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge

**会议**: AAAI2026  
**arXiv**: [2503.02659](https://arxiv.org/abs/2503.02659)  
**代码**: [HungerPWAY/LoRA-Null](https://github.com/HungerPWAY/LoRA-Null)  
**领域**: 模型压缩  
**关键词**: LoRA, catastrophic forgetting, knowledge preservation, null space, activation-aware initialization

## 一句话总结
提出 LoRA-Null，将 LoRA 初始化在预训练知识 input activation 的 null space 中（而非权重的 null space），从信息论角度论证 activation 的 effective rank 远小于权重，因此其 null space 包含更少预训练知识信息，显著减轻微调时的灾难性遗忘。

## 研究背景与动机
LoRA 微调虽然参数高效，但仍存在显著的 catastrophic forgetting。现有缓解遗忘的 LoRA 初始化方法沿两条路线：

**使残差权重 $\mathbf{W}_0'$ 接近预训练权重 $\mathbf{W}_0$**：CorDA 和 MiLoRA 都追求此目标

**使 LoRA 初始化空间正交于预训练知识**：MiLoRA 使用 $\mathbf{W}_0$ 的 null space

关键发现：LoRA adapter 微调后相对变化很小（$\|\mathbf{A}^* - \mathbf{A}_0\|_F / \|\mathbf{A}_0\|_F$ 较小），因此 adapter 的初始化空间比残差权重更重要。而 vanilla LoRA 虽冻结完整 $\mathbf{W}_0$，遗忘仍严重，说明"保持残差权重接近"并非关键。

核心洞察：input activation $\mathbf{X}_\text{pre}$ 包含所有前层参数和输入数据的信息（$\mathbf{W}_0$ 仅含当前层），且 $\mathbf{X}_\text{pre}$ 的 effective rank 远小于 $\mathbf{W}_0$（信息更集中），因此其 null space 包含的预训练知识更少。

## 方法详解

### LoRA-Null 流程
1. **采集 calibration 数据**：从代表预训练知识的数据集（如 NQ Open）随机采样 256 条，前向传播获取每层的 input activation $\mathbf{X}_\text{pre} \in \mathbb{R}^{d_\text{in} \times BL}$
2. **提取 null space**：对 $\mathbf{X}_\text{pre}$ 做 SVD，取最小 $r$ 个奇异值对应的左奇异向量 $\mathbf{U}_\text{null} \in \mathbb{R}^{d_\text{in} \times r}$，满足 $\mathbf{U}_\text{null}^\top \mathbf{X}_\text{pre} \approx \mathbf{0}$
3. **投影 + 初始化**：将 $\mathbf{W}_0$ 投影到 null space：$\mathbf{W}_0 \mathbf{U}_\text{null} \mathbf{U}_\text{null}^\top$，再对投影结果做 SVD 初始化 $\mathbf{A}$ 和 $\mathbf{B}$
4. **残差权重**：$\mathbf{W}_0' = \mathbf{W}_0 - \mathbf{BA}$（不追求其接近 $\mathbf{W}_0$）
5. 微调时仅更新 $\mathbf{A}$ 和 $\mathbf{B}$

### 理论分析
- **Theorem 1 & 2**：MiLoRA/CorDA 分别是使 $\|\mathbf{W}_0' - \mathbf{W}_0\|_F$ / $\|\mathbf{W}_0' \mathbf{X}_\text{pre} - \mathbf{W}_0 \mathbf{X}_\text{pre}\|_F$ 最小化的解
- **Theorem 3**：LoRA-Null 不满足上述两个优化目标——它完全放弃残差权重约束，极端化 null space 正交性
- **Effective rank 对比**：$\mathbf{X}_\text{pre}$ 的 eRank 远小于 $\mathbf{W}_0$（如 LLaMA-3.2-3B layer 0 kproj：101 vs 548），信息更集中于主子空间

## 实验关键数据

在 LLaMA-2-7B 和 LLaMA-3.2-3B 上评测 Math、Code、Instruction Following 三个任务。

### LLaMA-2-7B Math 任务

| 方法 | TriviaQA | NQ Open | WebQS | Avg1(Per) | GSM8k | Math | GM |
|---|---|---|---|---|---|---|---|
| LoRA | 45.95 | 1.16 | 6.64 | 64.56% | 42.99 | 6.26 | 21.00 |
| MiLoRA | 47.02 | 3.66 | 6.10 | 69.66% | 41.47 | 6.20 | 21.24 |
| CorDA | 48.99 | 7.15 | 5.76 | 76.24% | 41.47 | 8.22 | 22.64 |
| **LoRA-Null** | **50.02** | **7.98** | **6.55** | **79.21%** | **44.43** | **8.80** | **23.93** |

- 知识保留率（Avg1 Per）较 CorDA 提升约 **3%**，较 MiLoRA 提升约 **10%**
- 下游任务 GSM8k/Math 同时取得最优，实现知识保留与下游性能的双赢
- 跨 LLaMA-3.2-3B 同样一致领先

### 超参分析
- **Calibration 数据量**：LoRA-Null 对数据量敏感度低于 CorDA（64 样本时 CorDA 知识保留率降至 69%，LoRA-Null 仍保持 95%）
- **LoRA rank**：rank 增大时 CorDA 知识保留退化更快，LoRA-Null 更稳定

## 亮点
- **核心洞察清晰**："LoRA 初始化空间的正交性比残差权重接近性更重要"——简洁有力的发现
- **Activation vs Weight null space**：从信息量（effective rank）和信息来源（全层 vs 单层）两个角度论证 activation null space 更优
- **理论-实验一致**：Theorem 1-3 清晰刻画方法定位，Figure 4 的投影分析直观验证了 LoRA-Null adapter 确实只在 null space 中
- **超参鲁棒性**：对 calibration 数据量和 rank 变化的稳定性优于 CorDA

## 局限性
- 需要额外前向传播获取 activation（256 样本 × 1024 长度），有一定计算开销
- Null space 的"近似性"依赖于尾部奇异值足够小，在某些层/模型上可能不成立
- 仅在 LLaMA 系列上验证，未覆盖其他架构（如 Mistral、Qwen）
- 下游任务仅测了 Math/Code/IF 三类，缺少对话、摘要等更多场景
- 未讨论与其他 PEFT 方法（DoRA、AdaLoRA）的兼容性

## 评分
- 新颖性: ⭐⭐⭐⭐ — 从 activation null space 角度切入，洞察力强且理论论证到位
- 实验充分度: ⭐⭐⭐⭐ — 双模型、三任务、多 rank/calibration 消融，但场景可更丰富
- 写作质量: ⭐⭐⭐⭐⭐ — 动机推导清晰，理论与实验衔接紧密，图表设计优秀
- 价值: ⭐⭐⭐⭐ — 对 LoRA 知识保留研究有直接指导意义，方法简单有效易复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](../../ICCV2025/model_compression/efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[ICML 2025\] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics](../../ICML2025/model_compression/beyond_zero_initialization_investigating_the_impact_of_non-zero_initialization_o.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](../../ICML2025/model_compression/random_initialization_of_gated_sparse_adapters.md)

</div>

<!-- RELATED:END -->
