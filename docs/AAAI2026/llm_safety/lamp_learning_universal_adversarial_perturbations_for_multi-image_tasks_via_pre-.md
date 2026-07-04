---
title: >-
  [论文解读] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models
description: >-
  [AAAI2026][LLM安全][Universal Adversarial Perturbation] 提出 LAMP，一种针对多图 MLLM 的 black-box Universal Adversarial Perturbation 学习方法，通过 attention 约束和"传染式"损失实现仅扰动少量图像即可跨模型/任务迁移攻击。
tags:
  - "AAAI2026"
  - "LLM安全"
  - "Universal Adversarial Perturbation"
  - "Multi-Image MLLM"
  - "Black-box Attack"
  - "注意力机制"
  - "Transferable Attack"
---

# LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models

**会议**: AAAI2026  
**arXiv**: [2601.21220](https://arxiv.org/abs/2601.21220)  
**代码**: 无  
**领域**: AI安全  
**关键词**: Universal Adversarial Perturbation, Multi-Image MLLM, Black-box Attack, Attention Manipulation, Transferable Attack  

## 一句话总结
提出 LAMP，一种针对多图 MLLM 的 black-box Universal Adversarial Perturbation 学习方法，通过 attention 约束和"传染式"损失实现仅扰动少量图像即可跨模型/任务迁移攻击。

## 背景与动机

### 领域现状

**领域现状**：多模态大语言模型 (MLLM) 已支持多图输入（比较、推理、时序理解等），但其对抗鲁棒性几乎未被探索

### 现有痛点

**现有痛点**：现有对抗攻击主要面向**单图**场景，且多为 white-box 设定，不适用于实际 black-box 场景

### 核心矛盾

**核心矛盾**：在真实场景中（如社交媒体图片被模型处理），攻击者无法控制模型接收的图片数量和顺序，现有单图 UAP 方法在多图场景下效果有限

### 解决思路

**本文目标**：如何在 black-box 设定下学习少量固定的 Universal Adversarial Perturbation，使其能在攻击者无法控制推理时图片数量和顺序的条件下，有效攻击多图 MLLM？

## 方法详解

### 整体框架
利用预训练的 surrogate 模型（Mantis-CLIP）学习 UAP，保持 MLLM 参数冻结，仅优化扰动 $\delta_k$（$\|\delta_k\|_\infty \leq \epsilon$）。总损失由五项组成：

$$\mathcal{L}_{adv} = \lambda_1 \mathcal{L}_{adv}^{lm} + \lambda_2 \mathcal{L}_{adv}^{dec} + \lambda_3 \mathcal{L}_{adv}^{h} + \lambda_4 \mathcal{L}_{adv}^{ctg} + \lambda_5 \mathcal{L}_{adv}^{ias}$$

### 关键设计

1. **Adversarial Language Modeling Loss** $\mathcal{L}_{adv}^{lm}$：降低正确 token 的生成概率
$$\mathcal{L}_{adv}^{lm} = -\frac{1}{N}\sum_{i=1}^{N}\log(1 - P_\theta(t_{i+1}|s_{1:i}))$$

2. **Hidden States Divergence Loss** $\mathcal{L}_{adv}^{dec}$：最大化 clean 与 adversarial hidden states 间的 cosine 距离
$$\mathcal{L}_{adv}^{dec} = \frac{1}{L}\sum_{l=1}^{L}\cos(z_l^{adv}, z_l^{clean})$$

3. **Attention via Pompeiu-Hausdorff Distance** $\mathcal{L}_{adv}^{h}$：利用 Hausdorff 距离衡量 clean/adversarial attention 权重的 worst-case 偏差，比 KL 散度更能捕捉局部差异

4. **Contagious Loss** $\mathcal{L}_{adv}^{ctg}$（核心创新）：鼓励 clean token 在 self-attention 中更关注被扰动的 image token，使对抗效果从扰动图像"传染"到干净图像
$$\mathcal{L}_{adv}^{ctg} = -\frac{1}{LH}\sum_{l}\sum_{h}\sum_{i \in \mathcal{C}}\sum_{j \in \mathcal{N}} A^{(l)}_{:,h,i,j}$$

5. **Index-Attention Suppression Loss** $\mathcal{L}_{adv}^{ias}$：抑制 image token 对其位置索引 text token 的注意力，实现 position-invariant 攻击

## 实验关键数据


### 主实验

| 设定 | Avg. Best Baseline | LAMP | Δ (pp) |
|------|-------------------|------|--------|
| 所有模型平均 | 56.3% | 75.8% | **+19.5** |
| Mantis-CLIP | 51.5% | 71.9% | +20.4 |
| VILA-1.5 | 56.1% | 76.2% | +20.1 |
| LLaVA-v1.6 | 58.5% | 78.9% | +20.4 |
| Qwen-2.5 | 62.5% | 79.4% | +16.9 |

- 跨模型 zero-shot 迁移攻击均大幅领先 baseline
- 在防御策略下仍保持 ~70% ASR（vs baseline 20-56%）
- 最优扰动数量 $|\delta|=2$，超过 2 个改善不大（contagious loss 的贡献）
- LPIPS 仅 0.021（baseline 最优 0.068），不可感知性更好

## 亮点与洞察
- **首个多图 MLLM 对抗攻击**：填补了 multi-image 场景 UAP 攻击的空白
- **Contagious Loss 设计精巧**：用固定数量 UAP 即可"感染"clean tokens，解决了推理时图片数量未知的难题
- **Position-invariant 攻击**：通过 index-attention suppression 使攻击不依赖图像位置
- **强迁移性**：在 surrogate 模型上训练的 UAP 可跨 7+ 不同架构的目标模型有效攻击

## 局限与展望
- 仅在开源模型上验证，未测试 GPT-4V、Gemini 等闭源模型
- 扰动预算 $\epsilon=12/255$ 相对较大，对更严格预算下的表现未充分探讨
- 防御仅测试了 query-based defense，未评估更强的对抗训练防御
- 训练需要 A100 GPU 和 17K 样本，计算成本未详细分析

## 相关工作与启发
- vs **CPGC-UAP / UAP-VLP / Doubly-UAP**：这些是单图 encoder/decoder 攻击，LAMP 在多图 ASR 上平均领先 19.5pp
- vs **Jailbreak-MLLM**：后者通过模型集成提升迁移性，但 LAMP 无需集成即可达到更高 ASR
- vs **AnyDoor / MLAI**：这些利用多图能力但非 universal 攻击，LAMP 是首个多图 UAP 方法

## 相关工作与启发
- Contagious loss 的设计思路（让 clean token 关注 noisy token）可推广到其他 attention-based 攻击/防御场景
- Position-invariant attack 的 index suppression 思想对多图模型的安全评估有参考价值
- 揭示了多图 MLLM 的新攻击面：只需污染部分图片即可影响整体推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首个多图 UAP 攻击 + contagious loss + position-invariant)
- 实验充分度: ⭐⭐⭐⭐ (7+ 目标模型、5 benchmark、但缺闭源模型测试)
- 写作质量: ⭐⭐⭐⭐ (结构清晰、公式推导完整)
- 价值: ⭐⭐⭐⭐⭐ (对多图 MLLM 安全性研究有重要意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](../../CVPR2025/llm_safety/dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)
- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](../../ICLR2026/llm_safety/unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[AAAI 2026\] Democratizing LLM Efficiency: From Hyperscale Optimizations to Universal Deployability](democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)

</div>

<!-- RELATED:END -->
