---
title: >-
  [论文解读] Olica: Efficient Structured Pruning of Large Language Models without Retraining
description: >-
  [ICML2025][模型压缩][结构化剪枝] 提出 Olica 框架，通过对 MHA 层矩阵乘积做正交分解（PCA/SVD）并对 FFN 层做线性校准（岭回归闭式解 + 低秩近似），实现 LLM 结构化剪枝无需重训练，仅需 256 样本、3GB 显存、7 分钟即可完成 LLaMA-7B 剪枝且性能优于需要重训练的方法。
tags:
  - "ICML2025"
  - "模型压缩"
  - "结构化剪枝"
  - "LLM压缩"
  - "正交分解"
  - "线性校准"
  - "无需重训练"
  - "PCA"
  - "SVD"
---

# Olica: Efficient Structured Pruning of Large Language Models without Retraining

**会议**: ICML2025  
**arXiv**: [2506.08436](https://arxiv.org/abs/2506.08436)  
**代码**: [BetterTMrR/LLM-Olica](https://github.com/BetterTMrR/LLM-Olica)  
**领域**: 模型压缩  
**关键词**: 结构化剪枝, LLM压缩, 正交分解, 线性校准, 无需重训练, PCA, SVD

## 一句话总结
提出 Olica 框架，通过对 MHA 层矩阵乘积做正交分解（PCA/SVD）并对 FFN 层做线性校准（岭回归闭式解 + 低秩近似），实现 LLM 结构化剪枝无需重训练，仅需 256 样本、3GB 显存、7 分钟即可完成 LLaMA-7B 剪枝且性能优于需要重训练的方法。

## 研究背景与动机

现有 LLM 结构化剪枝方法（如 LLM-Pruner、SlimGPT、LoRAP）普遍依赖大量数据和计算资源进行 **LoRA 重训练** 来恢复被破坏的层间关联：

- DISP-LLM 需要 4× A100 80GB 才能剪枝 LLaMA-13B
- LLM-Pruner / SlimGPT 需要 5 万条标注数据（Alpaca）进行重训练
- 在特定领域做剪枝时，标注大量指令数据成本极高

**核心观察**：MHA 层的计算本质上依赖两类矩阵乘积 $W_q W_k^\top$ 和 $W_v W_o^\top$，可以将这些乘积视为统一实体，直接对其做 PCA 提取最重要信息，从而绕过重训练的需求。

## 方法详解

Olica 由三个核心模块组成：**正交神经元分解（OND）**、**快速 OND**、**线性校准（LC）**。

### 1. 正交神经元分解（OND）——MHA 层压缩

MHA 层依赖矩阵乘积 $W_{qk} = W_q W_k^\top$ 和 $W_{vo} = W_v W_o^\top$。将这些乘积视为统一实体，对 $W_{vo}$ 做 SVD 分解：

$$W_{vo} = U \Sigma V^\top$$

令 $\hat{W}_v \leftarrow U\Sigma$，$\hat{W}_o \leftarrow V$，则 $\hat{W}_v \hat{W}_o^\top = W_v W_o^\top$。由于 $U$、$V$ 的列正交，分解后的神经元携带不同信息，确保在给定维度下尽可能保留信息。

**剪枝策略**：不简单按特征值大小裁剪，而是用基于权重-激活值幅度的重要性评分：

$$\mathcal{I}(\text{neuron}_j) = \sum_i [\mathcal{I}(\hat{W}_{v_{ij}}) + \mathcal{I}(\hat{W}_{o_{ij}})]$$

其中 $\mathcal{I}(W_{ij}) = \|x^{(i)}\|_2 \cdot |W_{ij}|$（Wanda 评分），剪掉最不重要的神经元。

### 2. 快速 OND——降低 SVD 复杂度

对 $W_{vo} \in \mathbb{R}^{d \times d}$ 做 SVD 的复杂度为 $O(d^3)$，$h$ 个头共计 $O(h d^3)$，对 7B 模型需约 1 小时。

**关键发现**：$W_v$ 和 $W_o$ 的奇异值分布高度相似（$W_q$ 和 $W_k$ 也类似）。因此可以只对其中一个矩阵做 SVD 来近似判断统一实体的冗余度。例如对 $W_v \in \mathbb{R}^{d \times d/h}$ 做 SVD，复杂度仅为 $O(d^3/h^2)$，$h$ 个头总计 $O(d^3/h)$，相当于**降低了 $h^2$ 倍**。

### 3. 线性校准（LC）——FFN 层误差补偿

FFN 层剪枝后输出发生变化，逐层累积会放大误差。Olica 通过岭回归闭式解来重建剪枝残差：

$$\hat{W} = \arg\min_{W} \|E - XW\|_2^2 + \lambda \|W\|_F^2$$

其中 $E = f(X) - \hat{f}(X)$ 是剪枝前后的输出残差。闭式解为 $\hat{W} = (X^\top X + \lambda I)^{-1} X E$，**无需迭代训练**。

校准后前向传播变为：$X_{l+1} = \hat{f}_l(X_l) + X_l \hat{W}_l$

**低秩近似**：对 $\hat{W} \in \mathbb{R}^{d \times d}$ 做 SVD 保留前 $r$ 个主成分（$r/d = 0.03$），参数量从 $d^2$ 降至 $2dr$，仅占 FFN 层参数的约 1%。

**层选择准则**：基于多重相关系数 $R_{XE}$ 选择线性可恢复的 FFN 层进行校准：

$$R_{XE} = \frac{1}{d} \sum_{i=1}^{d} R_i$$

其中 $R_i$ 为残差预测与真实残差的 Pearson 相关系数。实验发现浅层 block 的 FFN 残差更适合线性校准。

### 4. RoPE 兼容处理

LLaMA 使用 RoPE 位置编码导致 $W_q$ 和 $W_k$ 之间无直接矩阵乘积。作者对 $W_q$、$W_k$ 分别做加权 SVD（Weighted SVD），目标为：

$$\arg\min_{W_1, W_2} \|(W - W_1 W_2) D\|_2^2$$

其中 $D = \text{diag}(\|x^{(1)}\|_2, \ldots, \|x^{(d)}\|_2)$，利用输入特征幅度加权。

## 实验关键数据

### 资源消耗对比（LLaMA-7B）

| 方法 | 数据量 | 时间 | 显存 | PPL↓ (25%) | Acc↑ (25%) | PPL↓ (33%) | Acc↑ (33%) |
|------|--------|------|------|------------|------------|------------|------------|
| LLM-Pruner | 50K | 3h | 30GB | 20.57 | 58.67 | 24.50 | 55.39 |
| SlimGPT | 50K | 1h | 20GB | 18.45 | 62.45 | 22.43 | 61.41 |
| **Olica** | **256** | **7min** | **3GB** | **16.69** | **63.53** | **19.83** | **61.21** |

### LLaMA-1 系列主要结果

| 模型 | 稀疏率 | 方法 | 需重训 | PPL↓ | 7任务均准↑ |
|------|--------|------|--------|------|-----------|
| LLaMA-7B | 20% | Olica | ✗ | **15.35** | **64.54** |
| LLaMA-7B | 25% | Olica | ✗ | **16.69** | **63.54** |
| LLaMA-7B | 25% | SlimGPT | ✗ | 19.11 | 62.47 |
| LLaMA-7B | 25% | LoRAP | ✗ | 17.40 | 62.57 |
| LLaMA-7B | 33% | Olica | ✗ | **19.83** | **61.21** |
| LLaMA-7B | 33% | SlimGPT | ✗ | 24.55 | 60.37 |
| LLaMA-13B | 20% | Olica | ✗ | **13.68** | **67.67** |
| LLaMA-13B | 20% | LoRAP | ✗ | 13.84 | 66.84 |

### LLaMA-2 / Vicuna 结果

| 模型 | 稀疏率 | 方法 | PPL↓ | 均准↑ |
|------|--------|------|------|-------|
| LLaMA-2-7B | 30% | Olica | **18.54** | **61.14** |
| LLaMA-2-7B | 30% | LoRAP | 19.42 | 58.89 |
| Vicuna-7B | 20% | Olica | **20.23** | **64.88** |
| Vicuna-7B | 20% | LoRAP | 20.74 | 64.39 |

### 推理效率（LLaMA-7B，RTX 4090）

| 稀疏率 | 参数量 | MACs | 显存 | 推理延迟 |
|--------|--------|------|------|----------|
| 0% | 6.74B | 424.02G | 12884 MiB | 46.95s |
| 20% | 5.39B | 373.23G | 10464 MiB | 40.62s |
| 33% | 4.52B | 339.53G | 8718 MiB | 35.78s |

### 消融实验（LLaMA-7B, 33% 稀疏率）

| MHA方法 | 线性校准 | PPL↓ | Acc↑ |
|---------|---------|------|------|
| SVD | ✗ | 71.01 | 47.62 |
| Wanda | ✗ | 20.94 | 59.82 |
| Fast-OND | ✗ | 20.34 | 60.68 |
| Fast-OND | ✓ | **19.83** | **61.21** |

### 运行时间对比（LLaMA-7B, 20%）

| 方法 | 运行时间 | PPL | Acc |
|------|----------|-----|-----|
| OND（标准SVD） | 2910s | 15.17 | 64.32 |
| Fast-OND | **413s** | 15.35 | 64.54 |

Fast-OND 加速约 **7 倍**，性能几乎无损。

## 亮点与洞察

1. **将矩阵乘积视为统一实体** 是核心洞察——直接在参数空间操作避免了数据驱动的重训练需求
2. **奇异值分布对称性** 的发现（$W_q \sim W_k$, $W_v \sim W_o$）使 Fast-OND 成为可能，复杂度降低 $h^2$ 倍
3. **线性校准的闭式解** 巧妙地利用岭回归 + SVD 低秩近似，额外参数仅约 1%
4. 仅需 **256 样本**、对样本数量和序列长度都不敏感（8→2048 变化，PPL 波动仅 2.4）
5. 资源消耗极低（3GB 显存 / 7 分钟），使得边缘设备上也能执行 LLM 剪枝

## 局限与展望

1. **高稀疏率性能骤降**：50% 稀疏率时所有方法性能都大幅下降（LLaMA-7B Olica仅 50.68%），无重训练方法在此区间的天花板较低
2. **仅验证了 LLaMA 家族**：未覆盖 Mistral、Qwen、Phi 等非 LLaMA 架构，泛化性有待验证
3. **RoPE 处理是 workaround**：由于 RoPE 破坏了 $W_q W_k^\top$ 的直接乘积结构，$W_q$/$W_k$ 只能分别做加权 SVD 而非联合分解，MHA 压缩的理论优美性在此打折
4. **线性校准假设限制**：仅用线性模型重建残差，对非线性误差模式的恢复能力有限
5. **缺少与量化方法的联合实验**：剪枝 + 量化联合压缩是工业界主流方案，论文未涉及
6. **层选择准则的理论支撑不足**：MC² 阈值需手动选择（从 {6, 12, 16} 中选），缺乏自适应机制

## 相关工作与启发

- **LLM-Pruner (NeurIPS'23)**: 两阶段（剪枝+LoRA重训练）的开创性工作
- **SliceGPT (ICLR'24)**: 用 PCA 降维隐层表示，但作用在激活空间而非参数空间
- **LoRAP (ICML'24)**: 对 QKV 做低秩近似，但仍需重训练
- **SlimGPT (NeurIPS'24)**: 扩展 OBS 到结构化剪枝，需大量数据
- **Wanda (ICLR'24)**: 提出基于权重×激活值的重要性评分，Olica 直接采用

Olica 的核心启发：**在参数空间做正交分解，可以绕开数据依赖的重训练**，这一思路可能扩展到其他模型压缩场景（如 ViT、扩散模型）。

## 评分
- 新颖性: ⭐⭐⭐⭐ — 矩阵乘积统一实体 + PCA 的视角新颖，Fast-OND 的奇异值对称性观察精妙
- 实验充分度: ⭐⭐⭐⭐ — 多模型多稀疏率，消融完整，但缺少非 LLaMA 模型和量化联合实验
- 写作质量: ⭐⭐⭐⭐ — 行文清晰，公式推导完整，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ — 实用性极强，3GB 显存 + 7 分钟的超低门槛使其成为资源受限场景的首选方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](../../ACL2025/model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[NeurIPS 2025\] Elastic ViTs from Pretrained Models without Retraining](../../NeurIPS2025/model_compression/elastic_vits_from_pretrained_models_without_retraining.md)
- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](../../ACL2026/model_compression/grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)

</div>

<!-- RELATED:END -->
