---
title: >-
  [论文解读] Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models
description: >-
  [ICML2025][LLM安全][联邦学习] 提出 Ferret，首个结合一阶优化与共享随机性的联邦全参数微调方法，通过将本地更新投影到低维空间实现 $10^6\times$ 通信压缩和 $6\times$ 计算加速，同时保持与 FedAvg 相当的模型精度。 - 核心矛盾：联邦学习 (FL) 对 LLM 进行全参数微调时…
tags:
  - "ICML2025"
  - "LLM安全"
  - "联邦学习"
  - "全参数微调"
  - "通信压缩"
  - "共享随机性"
  - "LLM"
  - "投影重建"
---

# Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models

**会议**: ICML2025  
**arXiv**: [2409.06277](https://arxiv.org/abs/2409.06277)  
**代码**: [allen4747/Ferret](https://github.com/allen4747/Ferret)  
**领域**: AI安全  
**关键词**: 联邦学习, 全参数微调, 通信压缩, 共享随机性, LLM, 投影重建

## 一句话总结

提出 Ferret，首个结合一阶优化与共享随机性的联邦全参数微调方法，通过将本地更新投影到低维空间实现 $10^6\times$ 通信压缩和 $6\times$ 计算加速，同时保持与 FedAvg 相当的模型精度。

## 研究背景与动机

- **核心矛盾**：联邦学习 (FL) 对 LLM 进行全参数微调时，需要在数据隐私、通信效率和模型精度之间取得平衡
- **PEFT 局限**：参数高效微调（LoRA 等）虽然降低了通信开销，但只微调部分参数，无法充分捕捉本地数据分布的细微差异，导致精度损失
- **零阶方法瓶颈**：FedKSeed 等零阶优化方法通过传输标量梯度降低通信量至 $\mathcal{O}(K)$，但存在三大问题：
  1. 计算代价高——每轮需要 $K$ 次前向传播估计梯度
  2. 收敛慢——需要更多通信轮数
  3. 梯度估计有偏——误差随本地迭代 $T$ 累积
- **FedAvg 瓶颈**：一阶方法计算高效、收敛快，但通信开销为 $\mathcal{O}(d)$（$d$ 为参数量，动辄数十亿），对 LLM 不可行

**关键问题**：能否设计一种方法，同时兼具一阶方法的计算效率和快速收敛，以及零阶方法的低通信开销？

## 方法详解

### 总体框架

Ferret 在每个通信轮次 $r \in [R]$ 重复三步操作：

**Step ①：全局聚合**

各客户端接收其他客户端的随机种子 $s^{(i)}$ 和投影坐标 $\{\gamma_k^{(i)}\}_{k=1}^K$，利用共享随机性重新生成随机基 $\{\mathbf{v}_k^{(i)}\}_{k=1}^K$，重建本地更新并聚合全局模型：

$$\mathbf{w}_{r-1} \leftarrow \mathbf{w}_{r-2} - \frac{1}{N} \sum_{i \in [N]} \widetilde{\Delta}_{r-1}^{(i)}, \quad \widetilde{\Delta}_{r-1}^{(i)} \triangleq \sum_{k \in [K]} \gamma_k^{(i)} \mathbf{v}_k^{(i)}$$

**Step ②：本地更新（一阶优化）**

每个客户端使用标准梯度下降进行 $T$ 步本地更新：

$$\mathbf{w}_{r,t}^{(j)} \leftarrow \mathbf{w}_{r,t-1}^{(j)} - \eta \nabla \ell(\mathbf{w}_{r,t-1}^{(j)}; \mathbf{x}_{t-1}^{(j)})$$

与零阶方法需要数百步不同，一阶方法使用更精确的梯度信息，仅需很少的迭代步数（$T=10$）即可获得等价的本地更新效果。

**Step ③：投影更新（降维传输）**

计算本地更新 $\Delta_r^{(j)} = \mathbf{w}_{r-1}^{(j)} - \mathbf{w}_r^{(j)}$，投影到 $K$ 维坐标：

$$\boldsymbol{\gamma} \approx (\rho K)^{-1} \mathbf{V}^\top \Delta$$

其中 $\rho$ 为截断正态分布的校正因子，保证重建无偏。仅传输种子 $s^{(j)}$ 和 $K$ 个标量，通信量从 $\mathcal{O}(d)$ 降至 $\mathcal{O}(K)$。

### 关键技术细节

**随机基的选择**：从截断正态分布 $v \sim \mathcal{N}(0,1)$, $v \in [-1/\sqrt{d}, 1/\sqrt{d}]$ 采样，保证 $\|\mathbf{v}_k\| \leq 1$，实现全参数更新的同时保持数值稳定。

**免逆重建**：直接近似 $\mathbf{V}^\top\mathbf{V} \approx \mathbf{I}_K$，避免矩阵求逆的 $\mathcal{O}(K^2d + K^3)$ 计算开销，降至 $\mathcal{O}(Kd)$。

**分块重建**：将 $d$ 维参数分为 $L$ 个块，每块独立投影/重建，计算复杂度进一步降低 $1/L$ 倍，存储复杂度降至 $\mathcal{O}(\max\{K_l, d_l\})$。

### 理论保证

- **无偏重建**（Theorem 1）：$\mathbb{E}[\widetilde{\Delta}] = \Delta$，避免了零阶方法的估计偏差
- **重建误差**（Theorem 2）：误差率为 $\widetilde{\mathcal{O}}(d/K)$，随 $K$ 增大线性降低，且不随本地迭代步 $T$ 累积
- **收敛性**（Theorem 4）：通信轮次复杂度为 $\mathcal{O}(1/\epsilon^2)$，与标准 SGD 渐进等价，且独立于参数维度 $d$

## 实验关键数据

### 精度对比（Rouge-L %）

| 方法 | NI (DataJuicer-1.3B) | NI (LLaMA-3B) | Dolly (DataJuicer-1.3B) | Dolly (LLaMA-3B) |
|------|------|------|------|------|
| FedIT (PEFT) | 22.30 | 28.13 | 30.80 | 33.23 |
| FedZO | 21.74 | 29.46 | 26.99 | 31.67 |
| FedKSeed | 22.33 | 29.77 | **30.91** | **34.56** |
| FedAvg | **23.95** | **32.11** | 29.67 | 30.98 |
| **Ferret** | **24.99** | 30.03 | 30.63 | **34.57** |

### 大模型实验（LLaMA2-7B / 13B）

| 方法 | CodeAlpaca (7B) | CodeAlpaca (13B) | GSM8K (7B) | GSM8K (13B) |
|------|------|------|------|------|
| FedKSeed | 8.33 | 10.70 | 28.26 | 33.67 |
| FedAvg | **15.41** | **14.68** | **38.30** | **39.82** |
| **Ferret** | 12.10 | 11.84 | 36.10 | 34.50 |

### 可扩展性对比（LLaMA-3B 每轮开销）

| 方法 | 本地更新 (s) | 全局聚合 (s) | 总计 (s) | 通信量 (参数数) |
|------|------|------|------|------|
| FedKSeed | 56.9 | 123.8 | 180.7 | 8.2×10³ |
| FedAvg | 1.8 | 0.3 | 2.1 | 6.0×10⁹ |
| **Ferret** | **5.6** (10.2×↓ vs FedKSeed) | **24.7** (5.0×↓) | **30.3** (6.0×↓) | **7.8×10³** (10⁶×↓ vs FedAvg) |

### LLaMA2-7B 每轮开销

| 方法 | 总计 (s) | 通信量 |
|------|------|------|
| FedKSeed | 627.0 | 8.2×10³ |
| FedAvg | 6.5 | 1.4×10¹⁰ |
| **Ferret** | **97.2** (6.5×↓ vs FedKSeed) | **6.4×10³** (10⁶×↓ vs FedAvg) |

Ferret 在 NI 数据集上仅需 12 轮即可收敛（FedKSeed 需 40 轮），收敛轮数减少 3.3×。

## 亮点与洞察

1. **巧妙融合一阶与零阶优势**：用一阶梯度保证计算效率和收敛速度，用随机投影+共享随机性压缩通信，两全其美
2. **无偏重建的理论突破**：证明了近似 $\mathbf{V}^\top\mathbf{V} \approx \mathbf{I}_K$ 后重建仍无偏，误差不随迭代累积——这比零阶方法的有偏估计有本质优势
3. **分块重建策略**：将计算复杂度再降 $1/L$ 倍，使方法能扩展到 7B/13B 规模
4. **实用性强**：兼容任意梯度优化器（AdamW 等），易于集成到现有 LLM 训练流程
5. **隐私增强**：仅传输种子和低维坐标，比传输完整梯度/参数的 FedAvg 更好保护隐私

## 局限与展望

1. **复杂任务精度差距**：在 CodeAlpaca 和 GSM8K 等复杂任务上，Ferret 仍不及 FedAvg（差 3-5%），投影重建的信息损失在难任务上更突出
2. **每轮计算仍高于 FedAvg**：虽然远优于 FedKSeed，但 Ferret 每轮 30s vs FedAvg 2.1s（约 14×），主要开销在全局聚合的重建步骤
3. **理论分析限于齐次设定**：收敛分析仅在 $\mathcal{L}^{(i)} = \mathcal{L}$ 的同分布场景下给出，异构场景缺乏严格保证
4. **超参数 $K$ 的选择**：$K$ 过小则重建误差大影响精度，过大则通信压缩收益减少，需要针对任务调参
5. **大规模客户端场景未验证**：实验中仅使用少量客户端（5% 采样），百/千客户端规模下的表现未知

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将一阶优化与共享随机性投影结合用于联邦全参数微调，方法设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — 多模型（1.3B-13B）多数据集，含可扩展性和消融分析，但大规模客户端实验缺失
- 写作质量: ⭐⭐⭐⭐ — 论文结构清晰，理论与实验互补，图表直观
- 价值: ⭐⭐⭐⭐ — 为联邦 LLM 微调提供了可扩展的全参数方案，平衡了效率、通信和精度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](../../CVPR2025/llm_safety/lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)
- [\[ICML 2025\] Activation Space Interventions Can Be Transferred Between Large Language Models](activation_space_interventions_can_be_transferred_between_large_language_models.md)

</div>

<!-- RELATED:END -->
