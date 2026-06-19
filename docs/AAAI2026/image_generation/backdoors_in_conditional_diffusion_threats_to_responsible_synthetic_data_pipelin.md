---
title: >-
  [论文解读] Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines
description: >-
  [AAAI 2026][图像生成][backdoor attack] 揭示了 ControlNet 条件分支的后门攻击漏洞：仅需 1–5% 的投毒数据即可在不修改扩散主干的前提下植入后门，触发时无视文本 prompt 生成攻击者指定内容，并提出 clean fine-tuning (CFT) 作为实用防御。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "backdoor attack"
  - "ControlNet"
  - "扩散模型"
  - "data poisoning"
  - "supply-chain security"
  - "clean fine-tuning"
---

# Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines

**会议**: AAAI 2026  
**arXiv**: [2507.04726](https://arxiv.org/abs/2507.04726)  
**代码**: 未公开（出于安全考虑，作者仅提供无害化复现脚本）  
**领域**: 图像生成  
**关键词**: backdoor attack, ControlNet, diffusion model, data poisoning, supply-chain security, clean fine-tuning

## 一句话总结

揭示了 ControlNet 条件分支的后门攻击漏洞：仅需 1–5% 的投毒数据即可在不修改扩散主干的前提下植入后门，触发时无视文本 prompt 生成攻击者指定内容，并提出 clean fine-tuning (CFT) 作为实用防御。

## 研究背景与动机

**合成数据管线依赖条件扩散**：Text-to-image 扩散模型广泛用于数据增强、领域迁移和隐私保护数据集生成，ControlNet 通过结构化条件（边缘图、深度图、姿态）提供精细控制，是合成数据工作流的核心组件。

**开源生态带来供应链风险**：大量社区微调的 ControlNet checkpoint 在 HuggingFace 等平台无审核分发，用户下载后直接部署，缺乏完整性验证和后门检测。

**现有安全研究的盲区**：先前鲁棒性工作主要针对像素扰动、classifier guidance、prompt 注入等，ControlNet 这一在每个去噪步注入残差的结构化条件路径的安全性几乎未被研究。

**低成本高回报的攻击面**：ControlNet 仅微调辅助分支而非整个扩散主干，参数规模小、训练成本低，使得攻击者可以以极低代价植入后门。

**后门隐蔽性强**：触发信号嵌入在重新计算的 control map 中（如边缘图中的小 patch），在清洁输入下模型表现完全正常，传统数据审查难以发现。

**合成数据的连锁危害**：一旦条件分支被投毒，管线会静默地将有害或违反策略的内容传播到下游合成数据集、审计集或增强语料中，即使 prompt 和基础模型本身是良性的。

## 方法详解

### 整体框架：ControlNet 条件分支投毒

核心思路是只对 ControlNet 分支 $\varepsilon_\phi$ 进行微调投毒，保持扩散主干 $\epsilon_\theta$ 完全冻结。攻击流程分为三步：

1. 对原始图像 $x$ 施加视觉触发器 $\mathcal{T}$ 得到 $x^{\text{trig}}$
2. 重新计算 control map $\tilde{c} = \mathcal{G}(x^{\text{trig}})$（触发器自然编码进 edge/pose map）
3. 将 $\tilde{c}$ 与固定的恶意目标图像 $x_{\text{mal}}$ 配对，构成投毒样本

最终训练集为 $\tilde{\mathcal{D}} = \mathcal{D} \cup \{(x_{\text{mal}}, \tilde{c})\}$，投毒比例仅需 1–5%。

### 关键设计 1：视觉触发器设计

触发器为嵌入在 control stream 中的小型 logo patch，仅占图像面积约 10%，位于右下角。关键在于触发器需经过 control map 提取器 $\mathcal{G}$（如 Canny 边缘检测、OpenPose 姿态估计）的重新计算后仍能保留，确保在实际推理时可被激活。对于姿态条件，使用固定的 RGBA 躺人剪影通过 alpha blending 注入 pose map。

### 关键设计 2：仅微调条件分支

组合去噪器为 $\hat{\epsilon}_{\theta,\phi}(z_t, t, c) = \epsilon_\theta(z_t, t) + \varepsilon_\phi(z_t, t, c)$，仅优化 $\phi$。这种设计使攻击局限于 ControlNet 路径，主干模型未被修改，增加了检测难度——标准的模型完整性检查（针对主干）无法发现异常。

### 关键设计 3：Clean Fine-Tuning (CFT) 防御

冻结扩散主干，仅在可信数据集上以小学习率（$1 \times 10^{-5}$）微调 ControlNet，其他超参不变。通过可信数据的梯度覆盖投毒滤波器来消除后门。

### 关键设计 4：双指标攻击成功率

ASR 同时要求：(i) NSFW 分类器得分 $\mathcal{C}(x) > 0.7$；(ii) CLIP 图像-图像相似度 $S_{\text{CLIP}}(x, x_{\text{ref}}) > 0.7$。双重阈值确保生成图像既包含恶意内容又与攻击者目标高度一致。

## 损失函数与训练

采用标准潜在扩散损失：

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \hat{\epsilon}_{\theta,\phi}(z_t, t, c)\|_2^2\right]$$

其中 $(x, c) \sim \tilde{\mathcal{D}}$。训练使用 AdamW（$\beta_1=0.9, \beta_2=0.999$，weight decay $10^{-2}$，lr $10^{-4}$），batch size 8（SD v1.5）或 4（SD v2/XL），最多 100 epochs，当验证集 ASR 达到 100% 时提前停止。

## 实验

### 实验 1：不同投毒比例下的攻击成功率（Table 1）

| 数据集 | 模型 | 1% | 5% | 10% |
|---------|------|-----|------|------|
| ImageNet | SD v1.5 | 91% | **100%** | 89% |
| ImageNet | SD v2 | 90% | 98% | **100%** |
| ImageNet | SD XL | 8% | 61% | **78%** |
| CelebA-HQ | SD v1.5 | 64% | **96%** | 96% |
| CelebA-HQ | SD v2 | **98%** | 74% | 92% |
| CelebA-HQ | SD XL | 11% | **100%** | 84% |

**结论**：SD v1.5/v2 在 1–5% 投毒时已达 90%+ ASR；SD XL 在低投毒比例下更鲁棒，但 5–10% 时也能达到较高 ASR。10% 时部分设置出现 ASR 下降，表明过拟合。

### 实验 2：姿态条件后门攻击（Table 2）

| 数据集 | 模型 | 投毒比例 | ASR |
|--------|------|----------|------|
| MPII | SD v1.5 | 1% | 80% |
| MPII | SD v1.5 | 5% | **99%** |
| MPII | SD v1.5 | 10% | 74% |

**结论**：后门攻击从边缘条件泛化到姿态条件，5% 投毒时 ASR 达 99%，证明攻击方法的通用性。10% 时 ASR 下降至 74%，进一步印证过拟合现象。

### CFT 防御效果

- CelebA-HQ：ASR 从 96% 降至 **25%**（有效）
- ImageNet：ASR 从 100% 仅降至 **93%**（效果有限）

同质化数据（人脸）提供一致梯度可有效覆盖后门，异质数据（ImageNet）则不然。

### 消融实验

- **触发器强度**：幅度 $\gtrsim 0.4$ 时攻击饱和
- **ControlNet guidance scale**：呈 sigmoid 依赖，$\approx 0.5$ 以上近乎确定性触发
- **采样步数**：对攻击成功率影响较小

## 亮点

- **新颖的攻击面**：首次系统研究 ControlNet 条件分支的后门漏洞，揭示了扩散模型供应链中被忽视的安全风险
- **极低投毒成本**：仅 1% 投毒即可达到 90%+ ASR，实际威胁性高
- **广泛验证**：跨 3 个 SD 版本（v1.5/v2/XL）、3 个数据集（ImageNet/CelebA-HQ/MPII）和 2 种条件类型（edge/pose）
- **攻防并举**：不仅展示攻击，还提出 CFT 防御和一系列实践建议（签名校验、CI 集成探针、运行时监控）
- **负责任披露**：不公开投毒模型和触发器，仅提供无害化脚本

## 局限性

- **CFT 防御在异质数据上效果有限**：ImageNet 上 ASR 仅从 100% 降至 93%，说明 CFT 不是通用解决方案
- **SD XL 分析不够深入**：SD XL 在低投毒率下 ASR 很低（8–11%），但缺乏对其鲁棒性来源的深入分析（如两阶段 refiner 的作用）
- **触发器类型单一**：仅测试了固定 logo patch 和躺人剪影两种触发器，未探索更隐蔽的触发器（如频域触发器）
- **防御基线缺乏**：仅提出 CFT 一种防御，未与现有后门检测方法（如 Neural Cleanse、STRIP）进行对比
- **评估规模有限**：每个设置仅使用 1000 训练 / 100 测试图像，实际大规模训练场景下的表现未验证

## 相关工作

- **数据投毒与后门攻击**：BadNets、clean-label 攻击等经典方法主要针对判别模型；本文将后门攻击扩展到条件生成模型的辅助分支
- **扩散模型安全**：Nightshade 翻转 prompt 语义，Silent Branding 注入 logo 幻觉，BadT2I/BadDiffusion 操纵去噪过程——这些工作均针对基础模型路径，ControlNet 分支未被涉及
- **合成数据治理**：供应链投毒、合成数据偏差放大、信任框架等工作从数据层面分析风险，本文从模型条件分支角度补充了这一方向

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次针对 ControlNet 条件分支的后门研究，攻击面新颖
- 实验充分度: ⭐⭐⭐⭐ — 跨模型/数据集/条件类型验证全面，但防御对比不足
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，threat model 定义严谨，负责任披露值得肯定
- 价值: ⭐⭐⭐⭐ — 对开源扩散模型生态的供应链安全具有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](../../CVPR2026/image_generation/ahs_adaptive_head_synthesis.md)
- [\[CVPR 2026\] Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling](../../CVPR2026/image_generation/accelerating_diffusion_via_hybrid_data-pipeline_parallelism_based_on_conditional.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)

</div>

<!-- RELATED:END -->
