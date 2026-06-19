---
title: >-
  [论文解读] Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning
description: >-
  [CVPR 2025][AI安全][隐私攻击] 首次证明 Adapter-based PEFT 在联邦学习中不是隐私安全的——恶意服务器可以将预训练模型设计为恒等映射使 patch embedding 原样传播到 adapter 层，从 adapter 梯度中解析式恢复训练图像（CIFAR-100 SSIM 0.88）。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "隐私攻击"
  - "梯度反演"
  - "PEFT安全"
  - "联邦学习"
  - "Adapter漏洞"
---

# Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning

**会议**: CVPR 2025  
**arXiv**: [2506.04453](https://arxiv.org/abs/2506.04453)  
**代码**: [https://github.com/info-ucr/PEFTLeak](https://github.com/info-ucr/PEFTLeak)  
**领域**: AI安全  
**关键词**: 隐私攻击、梯度反演、PEFT安全、联邦学习、Adapter漏洞

## 一句话总结
首次证明 Adapter-based PEFT 在联邦学习中不是隐私安全的——恶意服务器可以将预训练模型设计为恒等映射使 patch embedding 原样传播到 adapter 层，从 adapter 梯度中解析式恢复训练图像（CIFAR-100 SSIM 0.88）。

## 研究背景与动机

**领域现状**：Parameter-Efficient Fine-Tuning（PEFT）因只分享少量参数梯度被认为在联邦学习中更安全——攻击者能获取的信息更少。Adapter（如 LoRA）仅训练低秩矩阵，瓶颈维度 $r \ll D$ 进一步限制了可攻击的信息量。

**现有痛点**：之前的工作证明了 LoRA 微调可能泄露文本数据，但视觉领域的 adapter 安全性未被验证。人们普遍假设 PEFT 的少参数特性天然提供隐私保护。

**核心矛盾**：直觉上共享少量参数更安全，但攻击者如果能控制预训练模型的初始化（联邦学习中服务器分发模型），就能将预训练层设计为"透明管道"，使所有信息都流向可观察的 adapter 层。

**本文目标** 证明 adapter-based PEFT 在恶意服务器场景下不安全，并设计具体的攻击算法。

**切入角度**：恶意服务器将 ViT 的 LayerNorm、MSA、MLP 全部设为恒等映射，使 patch embedding 无失真地传播到 adapter 层。设计 adapter 的权重/偏置使特定神经元选择性地"通过"来自特定位置的 patch 信息。

**核心 idea**：将预训练模型"掏空"为恒等映射通道，使图像 patch 信息原样到达 adapter，然后从 adapter 梯度中解析式恢复原始图像。

## 方法详解

### 整体框架
恶意服务器设计预训练 ViT（恒等映射）+ adapter 参数 → 客户端正常在该模型上用 PEFT 训练 → 上传 adapter 梯度 → 服务器从梯度解析恢复训练图像。

### 关键设计

1. **预训练模型恒等化**:

    - 功能：使 patch embedding 无损传播到 adapter 层
    - 核心思路：$\mathbf{E} = 0.5\mathbf{I}_D$（线性嵌入），LayerNorm/MSA/MLP 全设为恒等。位置编码 $\mathbf{E}_{pos}^{(n)} \sim \mathcal{N}(0, 10)$ 使不同位置的 patch 正交（利于后续区分）
    - 设计动机：恒等映射保证 adapter 层接收到的输入就是原始 patch embedding + 位置编码

2. **Adapter 神经元设计**:

    - 功能：每个神经元从特定位置"提取"patch 信息
    - 核心思路：将 adapter 下投影权重设为目标位置的位置编码 $\mathbf{E}_{pos}^{(t)}$，偏置设计使得只有来自目标位置且值在特定区间的 patch 能激活该神经元。类似一个针对特定位置+值域的"选择门"
    - 设计动机：adapter 的瓶颈维度 $r \ll D$ 限制了单次可恢复的 patch 数量，但多个 adapter 层可以恢复不同的 patch

3. **多轮攻击扩展**:

    - 功能：克服小 $r$ 的信息瓶颈
    - 核心思路：每轮训练设计不同的值域区间 → 不同轮次恢复图像的不同 patch/不同值域 → 多轮结果拼合为完整图像
    - 设计动机：$r=8$ 时单轮恢复率低，但 6-8 轮后可恢复完整图像

### 损失函数 / 训练策略
解析式攻击——无需优化，直接从梯度值读取图像信息。恶意服务器有完全的模型初始化控制权。

## 实验关键数据

### 主实验

| 数据集 | LPIPS↓ | SSIM↑ | MSE↓ |
|--------|--------|-------|------|
| CIFAR-10 | 0.10 | 0.74 | 0.21 |
| CIFAR-100 | 0.08 | **0.88** | 0.20 |
| TinyImageNet | 0.12 | 0.76 | 1.06 |
| ImageNet batch=8 | - | - | 90% patch恢复 |

### 消融实验

| 配置 | Patch 恢复率 |
|------|-------------|
| batch=32, r=64 | ~85% |
| batch=128, r=64 | 72.6% |
| r=64 单轮 | ~85% |
| r=8 单轮 | 很低 |
| r=8 多轮(6-8轮) | 完整恢复 |

### 关键发现
- **PEFT 不等于隐私保护**：adapter 的低秩瓶颈可以通过多轮攻击+多 adapter 层绕过
- **batch=128 仍可恢复 72.6% patch**：大批量不能完全掩盖个体隐私
- **解析式攻击无需迭代优化**：比传统梯度反演攻击快得多
- **呼吁差分隐私 PEFT**：需要在 adapter 梯度上加噪声才能真正保护隐私

## 亮点与洞察
- **"将模型掏空为恒等映射"的攻击思路**非常巧妙——利用了 FL 中服务器控制模型初始化的特权
- **对 PEFT 安全性的警示**：论文挑战了"少参数=更安全"的直觉，这对 PEFT+FL 的实际部署有重要影响

## 局限与展望
- 攻击假设恶意服务器完全控制模型初始化——如果客户端验证模型完整性则攻击失败
- 恒等映射的模型在正常任务上性能为零——客户端可以通过验证集检测异常
- 仅攻击 adapter-based PEFT，对 LoRA（在现有层内加低秩而非额外层）的适用性不同

## 相关工作与启发
- **vs 传统梯度反演（GradInversion）**：需要迭代优化且假设诚实服务器。PEFTLeak 是解析式+恶意服务器场景
- **vs TAG / iDLG**：文本领域的梯度泄露攻击。PEFTLeak 首次扩展到视觉 PEFT

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示视觉 PEFT 的梯度隐私漏洞，攻击设计精巧
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多 batch size、多 r 值消融
- 写作质量: ⭐⭐⭐⭐ 攻击流程讲解清楚
- 价值: ⭐⭐⭐⭐⭐ 对 PEFT+FL 的安全实践有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](../../AAAI2026/ai_safety/plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[ICCV 2025\] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](../../ICML2026/ai_safety/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[CVPR 2025\] DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging](deal_data-efficient_adversarial_learning_for_high-quality_infrared_imaging.md)

</div>

<!-- RELATED:END -->
