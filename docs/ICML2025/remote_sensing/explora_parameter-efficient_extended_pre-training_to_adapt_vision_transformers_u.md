---
title: >-
  [论文解读] ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts
description: >-
  [ICML2025][遥感][参数高效微调] 提出 ExPLoRA，通过解冻 1-2 个 ViT block 并对其余层施加 LoRA，以参数高效的方式在目标域上继续自监督预训练，在遥感等域偏移场景下以 <10% 参数量超越从头全量预训练的 SOTA。 - 问题核心：大型视觉基础模型 (VFM) 如 DinoV2、MAE 在…
tags:
  - "ICML2025"
  - "遥感"
  - "参数高效微调"
  - "LoRA"
  - "域迁移"
  - "视觉基础模型"
  - "自监督预训练"
---

# ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts

**会议**: ICML2025  
**arXiv**: [2406.10973](https://arxiv.org/abs/2406.10973)  
**代码**: [https://samar-khanna.github.io/ExPLoRA/](https://samar-khanna.github.io/ExPLoRA/)  
**领域**: 遥感  
**关键词**: 参数高效微调, LoRA, 域迁移, 视觉基础模型, 遥感, 自监督预训练

## 一句话总结

提出 ExPLoRA，通过解冻 1-2 个 ViT block 并对其余层施加 LoRA，以参数高效的方式在目标域上继续自监督预训练，在遥感等域偏移场景下以 <10% 参数量超越从头全量预训练的 SOTA。

## 研究背景与动机

- **问题核心**：大型视觉基础模型 (VFM) 如 DinoV2、MAE 在自然图像上表现优异，但在遥感、医学等域偏移场景下性能显著下降。现有方案是在新域上从头全量预训练 VFM，计算成本极高（ViT-L 全量预训练需 960+ GPU 小时）。
- **LoRA 局限**：标准 LoRA 假设权重更新 $\Delta W$ 处于低秩子空间，该假设在源域与目标域分布相近时成立，但在自然图像 → 多光谱遥感等大域偏移时往往失效。
- **研究问题**：能否在保留自然图像预训练知识的前提下，仅用一小部分参数高效地将 VFM 适配到新域的无监督预训练中？

## 方法详解

### 核心思想

将最终目标域任务权重分解为三部分：

$$W_T^{(\tau)} \approx W_S + \Delta_T + \Delta^{(\tau)}$$

其中 $W_S$ 为源域预训练权重，$\Delta_T$ 为域适配无监督更新（ExPLoRA 阶段），$\Delta^{(\tau)}$ 为下游任务有监督微调更新。

### ExPLoRA 算法

1. **初始化**：用 DinoV2 或 MAE 的预训练权重 $W_S$ 初始化 ViT。
2. **选择性解冻**：将 $L$ 层 ViT block 划分为两组：
    - $\mathcal{U}$（如 $\{L\}$ 或 $\{1, L\}$）：完全解冻全部参数。
    - $\mathcal{L} \setminus \mathcal{U}$：冻结主体权重，仅在 Q、V 注意力矩阵上施加 LoRA（秩 $r$），同时解冻所有 block 的归一化层。
3. **继续自监督预训练**：在目标域无标签数据 $\mathcal{X}_T$ 上，使用与 $W_S$ 相同的无监督损失 $\mathcal{C}_S$（如 DinoV2 损失或 MAE 重建损失）训练所有解冻参数。

### 优化目标

$$\Delta_T = \arg\min_{\theta \in \Theta(\mathcal{U}, r)} \left( \min_\psi \sum_{\mathbf{x} \in \mathcal{X}_T} \mathcal{C}_S\left(g_\psi\left(f_\theta(\mathbf{x}; W_S)\right), \mathbf{x}\right) \right)$$

其中 $f_\theta$ 为 ViT 编码器，$g_\psi$ 为解码器（如 Dino/MAE head），$\Theta(\mathcal{U}, r)$ 约束可训练参数空间。

### 下游微调

ExPLoRA 后得到 $W_T^* = W_S + \Delta_T$，丢弃解码器 $g_\psi$，将 LoRA 矩阵合并回 ViT 主体，保持原始架构不变。下游任务可灵活使用线性探测、LoRA 微调或全量微调。

### 多光谱扩展

对于 SatMAE 的多光谱 ViT，需额外解冻位置编码和各通道组的 patch embedding 权重，因为 $W_S$ 仅在 RGB 上训练，无法直接初始化多通道输入。

## 实验关键数据

### fMoW-RGB 分类（ViT-L，62 类）

| 方法 | 预训练参数 | 微调参数 | 预训练 GPU h | Top-1 Acc |
|------|-----------|---------|-------------|-----------|
| ScaleMAE (全量) | 303.3M | 303.3M | 960 | 77.80% |
| SatMAE (全量) | 303.3M | 303.3M | 960 | 77.78% |
| DinoV2 + LoRA-r8 | – | 0.8M | – | 78.08% |
| DinoV2 + AdaLoRA-r8 | – | 1.2M | – | 78.87% |
| **D-[L]-r64 + LoRA-r8** | **18.7M** | **0.8M** | **100** | **79.28%** |

### fMoW-RGB 线性探测

| 方法 | Top-1 Acc |
|------|-----------|
| SatMAE (从头预训练) | 65.94% |
| DinoV2 | 69.00% |
| **D-[L]-r64 (ExPLoRA)** | **77.48%** |

ExPLoRA 线性探测精度较 DinoV2 提升 **+8.48%**，优于所有从头全量预训练方法。

### 消融实验要点

| 配置 | 参数量 | GPU h | LP Acc |
|------|--------|-------|--------|
| DinoV2 基线 | – | – | 69.00% |
| 从头全量预训练 | 303.3M | 1200 | 54.29% |
| 解冻 [L] + 无 LoRA | 12.7M | 90 | 74.83% |
| 解冻 [L] + LoRA-r64 Q,V | 18.7M | 100 | 77.48% |
| 解冻 [1,L-1,L] + LoRA-r64 | 43.4M | 180 | 78.04% |

### 多光谱 fMoW-Sentinel

ExPLoRA（M-[1,L]-r32）在仅 29.7M 预训练参数、320 GPU 小时下达到 60.15% top-1 准确率，接近从头全量预训练的 SatMAE（303.3M 参数，1150 GPU 小时，61.48%），计算成本降低约 3.6×。

## 亮点与洞察

1. **参数效率极高**：仅用全模型 6% 的参数（18.7M vs 303.3M）和 8× 更少的计算量就超越全量预训练 SOTA。
2. **知识迁移范式转换**：证明从头在新域预训练不是必需的，从自然图像模型高效迁移是更优路径。
3. **LoRA 发挥在预训练阶段**：首次系统性地将 LoRA 用于无监督预训练的域适配，而非传统的有监督微调。
4. **灵活组合性**：ExPLoRA 与下游 PEFT 方法（LoRA、SA2VP、VPT 等）正交可组合。
5. **关键发现**：LoRA 仅作用于 Q、V 矩阵效果最优；作用于 MLP 或全部矩阵反而大幅降低性能。

## 局限与展望

- **域覆盖有限**：主要在遥感上做了深入 case study，医学/农业等域仅在 WILDS 上做了初步验证。
- **仅限 ViT 架构**：方法绑定 Transformer block 结构，对 CNN 或混合架构的适用性未探索。
- **解冻策略依赖经验**：解冻哪些 block（首/尾）目前靠消融确定，缺乏理论指导或自动选择机制。
- **多光谱效果尚有差距**：在 fMoW-Sentinel 上尚未完全追平从头预训练的 SatMAE，多光谱域偏移更大仍具挑战。
- **预训练目标受限**：ExPLoRA 要求使用与源模型相同的自监督目标函数，无法灵活替换为更适合目标域的预训练策略。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 LoRA 用于无监督预训练域适配的思路新颖，分解 $W_S + \Delta_T + \Delta^{(\tau)}$ 的框架清晰
- 实验充分度: ⭐⭐⭐⭐ — 消融全面（block 选择、秩大小、LoRA 位置），多数据集验证，计算成本对比详尽
- 写作质量: ⭐⭐⭐⭐ — 符号清晰，算法伪代码简洁，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ — 为资源受限场景下的域适配提供了立即可用的高效方案，遥感领域实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning](../../CVPR2025/remote_sensing/meta-learning_hyperparameters_for_parameter_efficient_fine-tuning.md)
- [\[CVPR 2025\] Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking](../../CVPR2025/remote_sensing/learning_occlusion-robust_vision_transformers_for_real-time_uav_tracking.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[CVPR 2026\] CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation](../../CVPR2026/remote_sensing/crossearth-gate_fisher-guided_adaptive_tuning_engine_for_efficient_adaptation_of.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)

</div>

<!-- RELATED:END -->
