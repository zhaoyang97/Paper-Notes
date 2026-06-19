---
title: >-
  [论文解读] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification
description: >-
  [NeurIPS 2025][AI安全][对抗鲁棒性] 本文系统评估了基于图像压缩的对抗净化防御，发现重建图像的"真实感"（realism）是提升防御鲁棒性的关键因素——高真实感压缩模型在面对强自适应攻击时仍能保持显著鲁棒性，而这并非源于梯度掩蔽。 领域现状：对抗攻击通过微小扰动使分类器产生错误预测。对抗净化（adversa…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "对抗鲁棒性"
  - "图像压缩"
  - "对抗净化"
  - "真实感重建"
  - "自适应攻击"
---

# Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification

**会议**: NeurIPS 2025  
**arXiv**: [2508.05489](https://arxiv.org/abs/2508.05489)  
**代码**: 无  
**领域**: AI Safety  
**关键词**: 对抗鲁棒性, 图像压缩, 对抗净化, 真实感重建, 自适应攻击

## 一句话总结

本文系统评估了基于图像压缩的对抗净化防御，发现重建图像的"真实感"（realism）是提升防御鲁棒性的关键因素——高真实感压缩模型在面对强自适应攻击时仍能保持显著鲁棒性，而这并非源于梯度掩蔽。

## 研究背景与动机

**领域现状**：对抗攻击通过微小扰动使分类器产生错误预测。对抗净化（adversarial purification）通过对输入图像施加变换来消除对抗噪声，是一种不需要重新训练分类器的防御策略。

**现有痛点**：早期研究声称 JPEG 压缩可以有效防御对抗攻击，但后续工作表明许多预处理防御实际上依赖梯度掩蔽（gradient masking），在自适应攻击下会失效。现有评估缺乏全面的自适应攻击分析。

**核心矛盾**：压缩防御的鲁棒性增益是否只是梯度掩蔽的假象？如果不是，什么机制在真正贡献鲁棒性？

**本文目标**：(1) 压缩防御在严格自适应攻击下是否仍有鲁棒性？(2) 如果有，其内在机制是什么？

**切入角度**：聚焦"真实感"（realism）这一被忽视的维度，对比低真实感和高真实感压缩模型在各种攻击下的表现。

**核心idea**：重建图像的真实感——而非简单的失真控制——才是压缩防御有效性的关键，高真实感重建通过保持分布一致性和填补语义合理的细节来对抗对抗噪声。

## 方法详解

### 整体框架

防御流程：输入图像（可能含对抗扰动）→ 压缩-解压缩（编码器-解码器）→ 重建图像 → 分类器 → 类别概率。压缩步骤作为预处理，通过有损压缩来消除对抗扰动，同时保留语义内容。

### 关键设计

1. **真实感的形式化定义**：

    - 失真（Distortion）：$\mathcal{D} = \mathbb{E}[\Delta(x, \hat{x})]$，衡量原图与重建图的逐点距离
    - 真实感（Realism）：$\mathcal{R} = -d(p_{\hat{X}}, p_X)$，衡量重建图分布与自然图像分布的散度
    - 压缩模型训练损失：$\mathcal{L} = \mathcal{L}_{\text{RATE}} + \lambda \mathcal{D} - \beta \mathcal{R}$
    - 关键区别：失真是全参考指标（需要原图），真实感是无参考指标（只需分布匹配）

2. **威胁模型设计**：定义三种对抗者知识水平：

    - **黑盒 (BB)**：不知道防御存在，仅对分类器梯度进行攻击
    - **灰盒 (GB)**：知道防御存在，可在前向传播中使用，但无法计算防御梯度
    - **白盒 (WB)**：完全知道防御机制，可计算完整梯度

3. **四种自适应攻击方法**：

    - **ST BPDA**：直通近似——前向使用压缩防御，反向用恒等函数替代。$\nabla_x h \coloneq \nabla_x f(x)|_{x=g(x)}$
    - **U-Net BPDA**：训练 U-Net 近似压缩防御的前向行为，反向使用 U-Net 梯度。$\nabla_x h \coloneq \nabla_x (f \circ g')(x)$
    - **ACM（攻击压缩模型）**：直接以 $MSE(x, g(x))$ 为目标攻击压缩模型，迫使重建产生大失真
    - **ARA（自适应真实感攻击）**：针对可控真实感模型，用不同 $\beta'$ 的版本梯度来攻击目标 $\beta$ 版本

4. **真实感提升鲁棒性的两个机制**：

    - 避免不自然伪影，防止重建图偏离自然图像分布
    - 通过"幻觉"语义合理的细节（如树叶纹理）来遮蔽对抗噪声

### 排除梯度掩蔽

通过增大攻击预算 $\epsilon$ 验证：如果模型在高 $\epsilon$ 下持续失败，则低 $\epsilon$ 下的鲁棒性是真实的。实验引入 "Hyperprior Noise" 变体（用随机噪声替换梯度），其表现与正常版本相似，确认 Hyperprior 的鲁棒性来自梯度掩蔽，而 CRDR HR 的鲁棒性是真实的。

## 实验关键数据

### 主实验

ImageNet 验证集上 ResNet50 分类器的鲁棒准确率（$\epsilon = 4/255$，最强自适应攻击）：

| 防御模型 | 低真实感 (LR) | 高真实感 (HR) |
|---------|-------------|-------------|
| Hyperprior / HiFiC | 10.98 | 11.83 |
| MRIC | 26.68 | **39.00** |
| CRDR | 16.30 | **34.50** |
| JPEG | 15.19 | — |
| ELIC | 16.43 | — |

高真实感模型在所有设置下均显著优于低真实感版本。

### 消融实验——梯度掩蔽排除（PGD 步数 vs. 鲁棒准确率，$\epsilon = 4/255$）

| PGD 步数 | CRDR LR | CRDR HR |
|----------|---------|---------|
| 10 步 | 26.40 | 46.08 |
| 50 步 | 20.44 | 38.08 |
| 100 步 | 20.14 | 37.12 |
| 400 步 | 19.60 | 36.92 |

CRDR HR 在 400 步攻击下仍保持约 37% 准确率，且随步数增加准确率单调下降（无梯度掩蔽迹象）。

### 全面攻击对比（$\epsilon = 4/255$, CRDR）

| 攻击方式 | CRDR LR | CRDR HR |
|---------|---------|---------|
| BB PGD | 44.92 | 59.80 |
| WB PGD | 16.30 | 35.88 |
| ST BPDA | 39.36 | 56.36 |
| U-Net BPDA | 28.96 | 47.62 |
| ACM | 41.28 | 55.67 |
| ARA | 16.30 | **34.50** |

### 关键发现

- 真实感在所有攻击设置和所有 $\epsilon$ 值下均单调提升鲁棒性
- 失真存在固有权衡（过低保留噪声，过高破坏语义），而真实感无此权衡
- Hyperprior 的鲁棒性来自梯度掩蔽（噪声梯度可达到类似效果）
- WB PGD 和 U-Net BPDA 是最有效的攻击组合

## 亮点与洞察

- **首次系统揭示"真实感"在对抗鲁棒性中的核心作用**：此前工作关注失真和压缩率，真实感作为防御因素完全被忽视
- **严谨的评估方法论**：设计四种自适应攻击并仔细排除梯度掩蔽，符合对抗鲁棒性评估的最佳实践
- **直觉清晰**：高真实感重建将对抗样本"拉回"自然图像流形，类似扩散模型净化的思路但计算成本更低
- **对未来攻击方法的挑战**：指出克服真实感重建是对抗攻击领域的一个重要开放问题

## 局限与展望

- 仅评估了 $l_\infty$ 范数的无目标攻击，未考虑 $l_2$ 攻击或目标攻击
- 高真实感压缩模型本身的标准准确率有所下降（如 CRDR HR 标准准确率仅 62%，低于原始 ResNet 的 ~80%）
- 仅在 ImageNet 分类任务上评估，未扩展到目标检测等其他视觉任务
- 未探索将对抗训练与压缩防御结合的方案
- 依赖 FID 作为真实感代理指标，更精确的真实感度量有待研究

## 相关工作与启发

- **Shin & Song (2017)**：证明使 JPEG 可微后可完全绕过其防御，本文的高真实感模型在类似设置下仍保持鲁棒性
- **DiffPure (Nie et al.)**：基于扩散模型的对抗净化在概念上与高真实感压缩类似，但计算成本高得多
- **Blau & Michaeli (2019)**：提出失真-真实感权衡的理论框架，本文在对抗鲁棒性场景中验证了真实感的重要性
- 启发：对抗鲁棒性可能需要从"消除噪声"转向"恢复自然分布"的范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究真实感与对抗鲁棒性的关系，洞察深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 多种攻击方式、多个压缩模型、梯度掩蔽排除，评估非常严谨
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰流畅，引用 Feynman 格言切题，实验设计逻辑严密
- 价值: ⭐⭐⭐⭐ 对对抗鲁棒性和图像压缩两个社区都有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[AAAI 2026\] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models](../../AAAI2026/ai_safety/toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[NeurIPS 2025\] It's Complicated: The Relationship of Algorithmic Fairness and Non-Discrimination Provisions for High-Risk Systems in the EU AI Act](its_complicated_the_relationship_of_algorithmic_fairness_and_non-discrimination_.md)
- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](../../AAAI2026/ai_safety/breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)

</div>

<!-- RELATED:END -->
