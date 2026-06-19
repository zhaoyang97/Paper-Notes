---
title: >-
  [论文解读] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency
description: >-
  [多模态VLM] 提出SCAN，一种动态自举数据集剪枝方法，通过迭代的剪枝候选识别和数据集突变操作，在CLIP和MoCo对比预训练中以30-35%的数据剪枝率实现平均不到1%的性能下降。 对比预训练（如CLIP、MoCo）是学习通用表示的核心范式，但其数据效率问题：长期被忽视。主要困难在于： 缺乏可靠标签：自监督学习目标没有…
tags:
  - "多模态VLM"
---

# SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2411.09126](https://arxiv.org/abs/2411.09126)
- **代码**: [https://github.com/guoyang9/SCAN](https://github.com/guoyang9/SCAN)
- **领域**: 多模态VLM
- **关键词**: 对比预训练, 数据效率, 数据集剪枝, 动态稀疏训练, CLIP, MoCo

## 一句话总结

提出SCAN，一种动态自举数据集剪枝方法，通过迭代的剪枝候选识别和数据集突变操作，在CLIP和MoCo对比预训练中以30-35%的数据剪枝率实现平均不到1%的性能下降。

## 研究背景与动机

对比预训练（如CLIP、MoCo）是学习通用表示的核心范式，但其**数据效率问题**长期被忽视。主要困难在于：

**缺乏可靠标签**：自监督学习目标没有显式标签，无法像有监督学习那样估计样本的类概率（如EL2N）

**数据规模巨大**：预训练数据集通常包含百万甚至十亿级样本，逐样本计算梯度或Hessian的方法不可行

现有方法主要采用**静态核心集选择**（coreset selection），在训练前预先筛选重要数据。作者类比动态稀疏训练（DST）相对于静态稀疏训练的优势，指出静态剪枝无法动态追踪数据在预训练过程中的有用性变化。

**核心洞察**：数据集剪枝可分解为两个子问题：(1) 指标识别——使用何种proxy metric、(2) 剪枝策略设计——如何决定剪枝哪些数据。SCAN提出动态自举的解决方案。

## 方法详解

### 指标选择

选择**InfoNCE损失值**作为proxy metric，因为它满足：
- 动态适应性：随训练更新
- 快速获取：无需额外计算
- 反映学习状态：低损失=已充分学习，高损失=匹配不良

### 剪枝候选识别

识别两类应剪枝的数据：

**冗余数据（Redundant）**：损失最小的 $\rho$ 比例样本，已被模型充分记忆

$$\mathcal{D}_t^{red} = \mathcal{D}_{t:i}, \quad i \in {}_{\prec\rho}\bar{\mathcal{L}}_{f \rightarrow g}$$

**匹配不良数据（Ill-matched）**：损失最大的 $\rho$ 比例样本，图文语义不匹配

$$\mathcal{D}_t^{ill} = \mathcal{D}_{t:j}, \quad j \in {}_{\succ\rho}\bar{\mathcal{L}}_{f \rightarrow g}$$

最终候选集 $\mathcal{D}' = \mathcal{D}^{red} \cup \mathcal{D}^{ill}$（取两个方向的交集）。

**热身策略**：计算相邻epoch损失差与阈值 $T_{td}$ 比较，当 $(\mathcal{L}'_{pre} - \mathcal{L}'_{cur})/(\mathcal{L}'_{pre} + \epsilon) \geq T_{td}$ 时开始剪枝。

### 数据集突变（Bootstrapping）

不使用固定剪枝比例，而是采用**余弦退火策略**动态调整当前剪枝比例：

$$\rho_{cur} = \frac{1}{2}\left(1 + \cos\left((\tau_{cos} - (\tau_{cur} \bmod (\tau_{cos}+1))) \frac{\pi}{\tau_{cos}}\right)\right)$$

剪枝比例随训练进行周期性增大，从候选集 $\mathcal{D}'$ 中随机选取 $\rho_{cur}|\mathcal{D}'|$ 个样本进行剪枝。每 $(\tau_{cos}+1)$ 个epoch重新生成候选集并恢复到完整数据集，开始新一轮迭代。

## 实验

### CLIP预训练结果（CC12M+, 30%剪枝率）

| 架构 | 方法 | IN Zero-Shot Top-1 | CIFAR10 | CIFAR100 | IN Top-1 | IN-V2 | IN-R |
|------|------|------|---------|----------|----------|-------|------|
| RN101 | CLIP (全数据) | 18.78 | 95.96 | 82.13 | 75.76 | 64.31 | 40.57 |
| RN101 | Random | 14.05 | 95.02 | 78.34 | 73.99 | 60.27 | 36.13 |
| RN101 | SemDeDup | 13.26 | 95.07 | 78.77 | 74.24 | 62.16 | 37.65 |
| RN101 | D-Pruning | 12.59 | 94.94 | 78.89 | 74.07 | 61.30 | 37.07 |
| RN101 | **SCAN** | **23.10** | **96.08** | **82.28** | **75.66** | **63.75** | **40.10** |
| ViT-B/32 | CLIP (全数据) | 24.62 | 95.62 | 82.11 | 63.40 | 49.97 | 31.09 |
| ViT-B/32 | Random | 9.12 | - | - | - | - | - |
| ViT-B/32 | **SCAN** | **23.10+** | **95.5+** | **82.0+** | **63.0+** | **49.5+** | **31.0+** |

SCAN在30%剪枝率下性能几乎无损甚至略有提升（RN101零样本+4.3分），大幅超越静态方法。

### MoCo预训练结果（ImageNet, 35%剪枝率）

| 方法 | IN Linear Probing | IN Fine-tuning |
|------|-------------------|----------------|
| MoCo-v3 (全数据) | 76.2 | 83.2 |
| Random | 74.8 | 82.5 |
| **SCAN** | **75.9** | **83.0** |

在视觉自监督域同样有效，仅0.2-0.3%性能下降。

### 消融实验

| 策略 | IN Zero-Shot (RN101) |
|------|---------------------|
| 静态剪枝（固定比例） | 16.52 |
| 仅剪枝冗余数据 | 19.84 |
| 仅剪枝匹配不良数据 | 18.73 |
| SCAN（动态+两类） | **23.10** |

动态自举优于静态方法6.6分，两类数据的联合剪枝优于单一类型。

### 关键发现

1. 对比预训练中存在大量冗余和匹配不良数据，可安全剪枝30-35%
2. 动态剪枝显著优于静态方法，因为数据的"重要性"随训练进展而变化
3. SCAN的副产品（剪枝后的coreset）作为静态数据集也优于其他coreset选择方法
4. 方法在CLIP和MoCo两种范式、7种架构上普遍有效

## 亮点与洞察

1. **DST类比数据集剪枝**的视角新颖——将稀疏训练中"哪些权重重要"的思路迁移到"哪些数据重要"
2. **双端剪枝**同时去除冗余（低损失）和噪声（高损失）数据
3. **余弦退火式自举**在剪枝稳定性和效率间取得平衡
4. 方法具有强通用性——跨模态（CLIP）和单模态（MoCo）均适用

## 局限性

1. 仍需在完整数据集上运行前几个epoch的热身，不能从一开始就剪枝
2. batch内的损失比较在不同batch间可比性有限
3. 大规模（>10M）数据上的剪枝候选存储和更新有一定开销
4. 未探索更极端的剪枝率（如50%+）

## 相关工作

- **数据集剪枝**：EL2N（梯度）、Forgetting（遗忘次数）、Influence Functions
- **对比预训练**：CLIP、MoCo、SimCLR
- **动态稀疏训练**：RigL、SET、DST
- **VLP数据过滤**：SemDeDup、D-Pruning、DataComp

## 评分

- **创新性**: ★★★★☆ — 动态数据集剪枝+对比预训练的交叉创新
- **实用性**: ★★★★★ — 30%计算节省且几乎无损，直接减少碳排放
- **实验完整度**: ★★★★★ — 16个预训练模型、两种范式、多数据集验证
- **写作质量**: ★★★★☆ — 方法描述清晰，类比恰当

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

</div>

<!-- RELATED:END -->
