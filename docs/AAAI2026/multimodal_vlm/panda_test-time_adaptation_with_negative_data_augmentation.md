---
title: >-
  [论文解读] Panda: Test-Time Adaptation with Negative Data Augmentation
description: >-
  [AAAI2026][多模态VLM][test-time adaptation] 提出 Panda，通过负数据增强（patch 打乱重组）生成保留 corruption 但破坏语义的图像，用其特征偏移原始嵌入以抑制 corruption 引起的预测偏差，以极低开销（<10%）即插即用提升各类 TTA 方法的鲁棒性。
tags:
  - "AAAI2026"
  - "多模态VLM"
  - "test-time adaptation"
  - "数据增强"
  - "CLIP"
  - "prediction bias"
  - "corruption robustness"
---

# Panda: Test-Time Adaptation with Negative Data Augmentation

**会议**: AAAI2026  
**arXiv**: [2511.10481](https://arxiv.org/abs/2511.10481)  
**代码**: [ruxideng/Panda](https://github.com/ruxideng/Panda)  
**领域**: 多模态VLM  
**关键词**: test-time adaptation, negative data augmentation, CLIP, prediction bias, corruption robustness

## 一句话总结
提出 Panda，通过负数据增强（patch 打乱重组）生成保留 corruption 但破坏语义的图像，用其特征偏移原始嵌入以抑制 corruption 引起的预测偏差，以极低开销（<10%）即插即用提升各类 TTA 方法的鲁棒性。

## 研究背景与动机
CLIP 等预训练 VLM 在图像 corruption 下性能显著下降，核心原因是 corruption 模式被编码为 spurious features，导致模型对特定类别产生系统性的 **prediction bias**。例如在 CIFAR-10-C 上，corruption 会使预测分布严重偏离真实标签分布（L1 距离显著增大）。

现有 TTA 方法大量使用 positive data augmentation (PDA)，如 AugMix 生成 $K=63$ 个语义保持视图：
- **计算开销大**：每张图独立生成 $K$ 个增强视图，前向传播增至 $K+1$ 倍
- **无法消除偏差**：PDA 保留语义的同时也保留 corruption，平均后偏差甚至可能放大
- Prediction bias 对 entropy-based TTA 方法尤为致命，可导致伪标签偏差累积至 model collapse

## 方法详解

### 负数据增强 (NDA)
1. 对 batch 内 $B$ 张图像切分为 $\frac{H}{H_p} \times \frac{W}{W_p}$ 个 patch（默认 $H_p=W_p=32$，即 $7 \times 7$）
2. 所有 patch 汇入共享 pool，随机打乱重组为 $M$ 张负增强图像（$M = B/10 \ll B$）
3. 负增强图像破坏了物体语义但保留了 corruption 特征

### 特征偏移 (Offset)
- 编码负增强图像得 $\{\mathbf{n}_j\}_{j=1}^M$，计算均值 $\bar{\mathbf{n}} = \frac{1}{M}\sum_j \mathbf{n}_j$
- 对原始嵌入做偏移：$\mathbf{d}_i = \mathbf{v}_i - \beta \cdot \bar{\mathbf{n}}$
- 理论保证（Theorem 4.1）：当负增强与 corruption 相关性 $r>0$ 且与类别信息无关时，offset 策略可将 corruption 分量压缩至 $\sqrt{1-r^2}$ 倍，最优 $\beta=r$

### 与 TTA 方法集成
Panda 仅修改前向传播中的特征表示（$\mathbf{v}_i \to \mathbf{d}_i$），可无缝嵌入 Tent、ETA、SAR、DeYO、TPT、TPS 等任意 TTA 框架。对于 entropy minimization 类方法（如 Tent），用 debiased logits 计算熵可同时提升预测质量和适应稳定性。

### 与 DeYO 的 NDA 策略对比
DeYO 也使用负增强，但仅用于估计预测置信度来指导样本选择和加权。Panda 的 NDA 生成质量更高（Table 4: offset 策略在 CIFAR-100-C 上 43.3% vs DeYO 的 select & weight 38.0%），且两者可组合使用。

## 实验关键数据

在 CIFAR-10-C、CIFAR-100-C、ImageNet-C（severity 5）上评测，覆盖 9 种 TTA baseline。

### Table 1: +Panda 提升（平均准确率 %）

| 数据集 | CLIP | Tent | ETA | SAR | DeYO | 平均提升 |
|---|---|---|---|---|---|---|
| CIFAR-10-C | +2.6 | **+8.3** | +3.4 | +7.4 | +1.7 | +3.3 |
| CIFAR-100-C | +1.6 | +2.7 | +2.5 | +2.6 | **+4.1** | +2.2 |
| ImageNet-C | +1.7 | **+2.9** | +1.4 | +0.6 | +2.2 | +2.0 |

### 效率对比 (Table 3, ViT-B/32 CIFAR-10)
- Panda 额外开销 <10%：Tent 25s→27s (+8.0%), TPT 22min→22min39s (+1.3%)
- 对比 PDA 方法（TPT/Zero/TPS 需 $K=63$ 次增强），Panda 在 CIFAR-10-C 上以 71.1% 大幅超越 TPT 62.2%、TPS 63.7%

### 预测偏差消除
- Tent 在 Gaussian noise 上随适应累积偏差直至 model collapse；Tent+Panda 持续维持低偏差和高准确率
- 在 15 种 corruption 中，PDA 仅在 4 种中减小偏差，Panda 在全部 15 种中均有效减小

## 亮点
- **反直觉设计**：用"破坏语义"的负增强而非"保持语义"的正增强来提升鲁棒性，思路新颖
- **极低开销**：$M=B/10$ 个增强 batch 共享，额外计算 <10%，相比 PDA 的 $63\times$ 代价极具优势
- **即插即用**：仅修改前向传播中的 embedding，兼容所有基于 CLIP 的 TTA 算法
- **理论支持**：提供了 offset 策略准确率提升的理论证明和最优 $\beta$ 的闭式解

## 局限性
- 仅在 CLIP 系列 VLM 上验证，未扩展到 BLIP、SigLIP 等其他 VLM
- 理论分析基于高斯分布假设，真实场景中 corruption 分量可能更复杂
- 默认 patch 大小与 ViT patch 对齐（32×32），对非标准分辨率的适用性需验证
- 仅评测 image classification，未涉及 detection、segmentation 等下游任务
- 超参 $\beta$ 和 $M/B$ 消融显示不敏感，但极端 corruption（如 impulse noise）下是否依然稳健值得进一步验证

### 消融要点
- 单图内 patch 打乱（而非 batch 间共享）效果显著下降 → batch 级信息共享是关键
- 不做负增强特征平均直接逐个减去效果也差 → 平均操作有效抑制个体噪声
- $M/B$ 比例从 1/2 降至 1/100 性能仍稳定 → 少量负增强即可受益
- $\beta$ 在 0.5-2.0 范围内均优于无 Panda 的 baseline

## 评分
- 新颖性: ⭐⭐⭐⭐ — 负增强 + 特征偏移的思路简洁而有效，与正增强形成鲜明对比
- 实验充分度: ⭐⭐⭐⭐ — 3 数据集 × 9 TTA baseline × 15 corruption，消融和敏感性分析全面
- 写作质量: ⭐⭐⭐⭐ — 直觉图示清晰，理论推导严谨
- 价值: ⭐⭐⭐⭐ — 即插即用特性使其对 TTA 社区有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](../../NeurIPS2025/multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[CVPR 2026\] Condensed Test-Time Adaptation of VLMs for Action Recognition](../../CVPR2026/multimodal_vlm/condensed_test-time_adaptation_of_vlms_for_action_recognition.md)
- [\[AAAI 2026\] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](bridging_modalities_via_progressive_re-alignment_for_multimo.md)
- [\[CVPR 2026\] Dynamic Logits Adjustment and Exploration for Test-Time Adaptation in Vision Language Models](../../CVPR2026/multimodal_vlm/dynamic_logits_adjustment_and_exploration_for_test-time_adaptation_in_vision_lan.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](../../CVPR2026/multimodal_vlm/decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
