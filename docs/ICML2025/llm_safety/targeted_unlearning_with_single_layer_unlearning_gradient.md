---
title: >-
  [论文解读] Targeted Unlearning with Single Layer Unlearning Gradient
description: >-
  [ICML 2025][LLM安全][machine unlearning] 提出 SLUG (Single Layer Unlearning Gradient) 方法，通过层重要性和梯度对齐指标识别最优单层，仅需一次梯度计算和单层参数更新即可实现高效精准的定向遗忘，可应用于 CLIP、Stable Diffusion 和 VLM。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "machine unlearning"
  - "CLIP"
  - "扩散模型"
  - "VLM"
  - "Single Layer Update"
  - "隐私保护"
---

# Targeted Unlearning with Single Layer Unlearning Gradient

**会议**: ICML 2025  
**arXiv**: [2407.11867](https://arxiv.org/abs/2407.11867)  
**代码**: [github.com/CSIPlab/SLUG](https://github.com/CSIPlab/SLUG)  
**领域**: 机器遗忘, 多模态基础模型, 可信AI  
**关键词**: machine unlearning, CLIP, Stable Diffusion, VLM, Single Layer Update, 隐私保护

## 一句话总结

提出 SLUG (Single Layer Unlearning Gradient) 方法，通过层重要性和梯度对齐指标识别最优单层，仅需一次梯度计算和单层参数更新即可实现高效精准的定向遗忘，可应用于 CLIP、Stable Diffusion 和 VLM。

## 研究背景与动机

大型基础模型（LLM、文生图、VLM）在海量数据上训练，不可避免地学习到隐私数据或受版权保护的内容。从头重训成本极高，**机器遗忘 (Machine Unlearning)** 成为必要的替代方案。

现有遗忘方法面临三大挑战：

**计算效率低**：Fine-tuning (FT)、Gradient Ascent (GA) 等需要多次迭代更新整个模型

**副作用大**：全模型参数更新会影响不相关概念的表现

**需大量超参数调优**：学习率、迭代次数、掩码阈值等

核心洞察：**深度网络不同层学习不同特征，只需修改最关键的单层即可实现定向遗忘，同时最小化对模型整体能力的影响。**

## 方法详解

### 整体框架

SLUG 包含三步：
1. 一次性计算 forget 和 retain 的梯度
2. 通过 Pareto 前沿识别最优单层
3. 用二分搜索确定步长进行单步更新

### 损失函数设计

**保留集损失** — 标准对比损失（保持视觉-文本对齐）：

$$\mathcal{L}_{\text{retain}} = \frac{1}{2N_r}\sum_{i=1}^{N_r}(\ell_{i2t}(i) + \ell_{t2i}(i))$$

**遗忘集损失** — 余弦嵌入损失（打破视觉-文本对齐）：

$$\mathcal{L}_{\text{forget}} = \frac{1}{N_f}\sum_{i=1}^{N_f} 1 - \cos(\mathbf{v}_i, \mathbf{t}_j)$$

### 单层识别

**层重要性**（层 $l$ 对遗忘集的敏感度）：

$$\text{Importance}(l) = \frac{\|\nabla_{\theta_l} \mathcal{L}_{\text{forget}}\|_2}{\|\theta_l\|_2}$$

**梯度对齐**（forget 与 retain 梯度的夹角）：

$$\text{Alignment}(l) = \cos(\nabla_{\theta_l}\mathcal{L}_{\text{forget}}, \nabla_{\theta_l}\mathcal{L}_{\text{retain}})$$

目标：**最大化 Importance，最小化 Alignment** → 在所有层上搜索 Pareto 前沿。

### 单梯度方向更新

$$\theta_l^* \leftarrow \theta_l^{(0)} - \lambda^* \nabla_{\theta_l}\mathcal{L}_{\text{forget}}\big|_{\theta=\theta^{(0)}}$$

步长 $\lambda^*$ 通过二分搜索确定（固定 $S=10$ 步），找到 forget accuracy 接近 0 且 test accuracy 保持的平衡点。

### 泛化到 SD 和 VLM

- **Stable Diffusion**：对文本编码器（CLIP）应用 SLUG，实现层级可插拔的遗忘
- **VLM (LLaVA)**：对视觉编码器应用 SLUG，影响下游文本生成

## 实验关键数据

### CLIP 零样本分类

| 方法 | FA@1↓ | TA_IN@1↑ | TA_CA@1↑ | 计算复杂度 |
|------|-------|----------|----------|-----------|
| GA | 0.00 | 35.88 | 24.92 | $O(k \cdot N_f)$ |
| SalUn | 0.00 | 55.45 | 26.11 | $O(N_f) + O(k \cdot (N_f+N_r))$ |
| SSD | 0.00 | 51.84 | 35.96 | $O(N_f+N_r)$ |
| **SLUG** | **0.00** | **59.96** | **58.32** | $O(N_f+N_r)$ |

### UnlearnCanvas 基准

| 方法 | 风格UA↑ | 效率(时间/s)↓ | 存储/GB↓ |
|------|---------|-------------|----------|
| ESD | 98.58 | 6163 | 4.3 |
| SalUn | 86.26 | 667 | 4.0 |
| **SLUG** | **86.29** | **39** | **0.04** |

### VLM 遗忘（LLaVA-1.5-7B）

10 个名人身份的遗忘准确率从 99.50% 降至 2.8%，同时在 VLM benchmark 上保持竞争力。

### 关键发现

- SLUG 在 CLIP 上保持了近 60% 的 ImageNet 准确率和 58.32% 的 CelebA 准确率，**远超其他方法**
- 效率极高：UnlearnCanvas 上仅需 39 秒，存储仅 0.04 GB
- 跨模型泛化：从 ViT-B/32 到 EVA01-g-14 表现一致

## 亮点与洞察

1. **极致简洁**：一次梯度、一层更新、一次二分搜索，无需迭代训练
2. **模块化设计**：只修改单层权重，可作为即插即用的"遗忘补丁"
3. **跨模型通用**：CLIP→SD→VLM 的完整链路验证
4. **理论直觉清晰**：利用 Fisher 信息矩阵和 Pareto 最优进行层选择

## 局限性

- 对复杂纠缠概念的遗忘效果可能有限
- 单层更新的表征能力上界受限
- 二分搜索需要少量验证数据

## 相关工作

- Fine-tuning / Gradient Ascent 遗忘
- SalUn（显著性遗忘，ICLR2024）
- SSD（选择性突触抑制）
- Task Arithmetic（任务算术）

## 评分

⭐⭐⭐⭐⭐ — 方法极简但效果卓越。跨三类基础模型（CLIP、SD、VLM）的全面验证令人信服。39 秒/0.04GB 的效率数据给出了遗忘方法的新标准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/llm_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)
- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](../../NeurIPS2025/llm_safety/trap_targeted_redirecting_of_agentic_preferences.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/llm_safety/munba_machine_unlearning_via_nash_bargaining.md)
- [\[ACL 2025\] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](../../ACL2025/llm_safety/cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)

</div>

<!-- RELATED:END -->
