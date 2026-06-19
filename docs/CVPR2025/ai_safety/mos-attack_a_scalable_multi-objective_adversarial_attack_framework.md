---
title: >-
  [论文解读] MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework
description: >-
  [CVPR 2025][AI安全][对抗攻击] 提出MOS Attack框架，将对抗攻击建模为多目标集合优化问题，结合smooth max/min近似实现多损失函数联合优化，并自动发现损失函数间的协同模式，在CIFAR-10和ImageNet上超越现有SOTA单目标攻击和集成攻击。 对抗攻击的本质是最大化一个不可微的0-1损…
tags:
  - "CVPR 2025"
  - "AI安全"
  - "对抗攻击"
  - "多目标优化"
  - "损失函数协同"
  - "集合优化"
  - "鲁棒性评估"
---

# MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework

**会议**: CVPR 2025  
**arXiv**: [2501.07251](https://arxiv.org/abs/2501.07251)  
**代码**: [GitHub](https://github.com/pgg3/MOS-Attack)  
**领域**: AI Safety  
**关键词**: 对抗攻击, 多目标优化, 损失函数协同, 集合优化, 鲁棒性评估

## 一句话总结

提出MOS Attack框架，将对抗攻击建模为多目标集合优化问题，结合smooth max/min近似实现多损失函数联合优化，并自动发现损失函数间的协同模式，在CIFAR-10和ImageNet上超越现有SOTA单目标攻击和集成攻击。

## 研究背景与动机

对抗攻击的本质是最大化一个不可微的0-1损失函数，实际中用可微的代理损失函数（如交叉熵CE、DLR等）来近似。现有攻击方法（FGSM、PGD、APGD、ACG等）都是单目标攻击，即每次只优化一个代理损失函数。

**核心问题**：不同代理损失函数对0-1损失的近似能力不同，它们之间存在**协同和冲突**关系。简单地将多个损失函数线性组合并不能充分利用这些关系。已有工作尝试了多损失函数策略（如交替使用、多目标targeted loss），但缺乏**系统性的多目标优化框架**和**对损失函数交互关系的理论理解**。

**具体挑战**：
- 直接为每个损失函数分配独立的对抗样本过于昂贵（样本数=损失函数数）
- 如何用少于损失函数个数的样本同时优化多个目标
- 如何自动发现哪些损失函数"互相帮助"（协同），从而精简攻击

## 方法详解

### 整体框架

MOS Attack将对抗攻击分为两个阶段：(1) **多目标集合优化**：给定m个损失函数，用K个对抗样本（K<m）同时优化所有目标，生成逼近Pareto最优的解集；(2) **协同模式挖掘**：分析主导样本与损失函数的对应关系，自动发现损失函数间的协同模式，构建精简的多目标攻击（如MOS-3*）。

### 关键设计

**1. 平滑集合优化（Smooth Set-based Optimization）**

从Tchebycheff分解出发，但解决其三个问题：(a) 复杂性——用K<m个样本代替需要>m个的分解方案；(b) 权重模糊性——固定权重为全1向量；(c) 不可微——使用smooth max/min算子近似极值操作。

最终优化目标：g(Δ) = -μ log(Σ_i^m (Σ_k^K exp(f_i(δ_k)/μ))^{-1})

其中μ是平滑参数，f_i是第i个损失函数，δ_k是第k个扰动。这个公式优雅地将"每个损失取K个样本中的最大值，再对m个损失取最小值"用可微形式表达。每个样本可以在不同维度"专注于"不同损失函数，形成"虚拟对抗样本"的概念。

**2. 基于APGD的实现**

将smooth集合优化问题嵌入APGD框架：(a) 同时优化X（对抗样本集）和Δ（扰动集），因∇_X g = ∇_Δ g；(b) 集合投影——逐个将样本投影到ℓ∞球内；(c) 继承APGD的动量更新和自适应步长调整策略，包括checkpoint检查和步长减半机制。无需额外超参数。

**3. 自动化协同模式挖掘**

两步流程：(a) **确定主导样本**——提出双目标优化问题：最小化使用子集与全集的优化差距（用smooth算子近似）+ 最小化子集大小（L0→L1松弛），通过梯度方法求解指示向量β；(b) **确定协同模式**——对每个主导样本，检查其在哪些损失函数上的归一化值超过C×最大值的阈值，记录为该样本的"损失协同组合"，在整个数据集上统计频率。

### 损失函数

使用8个代理损失函数：4个经典损失（Cross Entropy、Marginal Loss、DLR、Boosted CE）+ 4个通过自动搜索发现的损失（来自AutoLoss和Tightening工作）。这8个损失函数覆盖了logit空间和概率空间的不同操作。

## 实验关键数据

### 主实验：攻击成功率（Table 3）

**CIFAR-10 (ε=8/255)**：

| 方法 | 平均排名 | ID0 (R-18)↑ | ID2 (R-18)↑ | ID9 (WR-70-16)↑ |
|------|:---:|:---:|:---:|:---:|
| APGD-CE (1 restart) | 5.92 | 39.17 | 41.57 | 31.43 |
| ACG-CW (5 restarts) | 4.00 | 42.45 | 43.10 | 32.54 |
| APGD-All (1×8) | 1.67 | 42.78 | 44.16 | 33.50 |
| **MOS-8 (K=5)** | **1.33** | **42.77** | **44.18** | **33.51** |

**ImageNet (ε=4/255)**：

| 方法 | 平均排名 | ID12 (R-18)↑ | ID13 (R-50)↑ | ID16 (WR-50-2)↑ |
|------|:---:|:---:|:---:|:---:|
| APGD-CE (1 restart) | 6.00 | 70.60 | 61.38 | 59.02 |
| ACG-CW (5 restarts) | 4.00 | 72.94 | 62.74 | 58.92 |
| APGD-All (1×8) | 1.40 | 74.38 | 64.92 | 61.26 |
| **MOS-8 (K=5)** | **1.60** | **74.52** | **64.94** | **61.14** |

MOS-8在CIFAR-10上平均排名1.33（最优），使用5个样本即达到APGD-All（8个独立攻击各取最优）的水平或更优。

### 消融/分析实验

**MOS上界分析（Table 5）**：MOS-8(K=1)与理论上界差距为0.3-0.8%，MOS-8(K=8)差距缩小至0.1-0.3%，说明smooth集合优化的近似效果好。

**协同模式发现（Fig. 2）**：
- CIFAR-10上最频繁的模式是{Loss5, Loss6, Loss7}联合出现（约30%），其次是{Loss5, Loss6}（约15%）
- 搜索发现的Loss 4-7始终表现最好，在APGD-All中这些损失的单独攻击一致获得最高ASR
- 基于协同分析构建的MOS-3*（仅用3个损失）仍优于ACG-CW等5-restart单目标攻击

### 关键发现

- **多目标 > 单目标**：即使是最强的单目标ACG-CW（100步），也只在17个模型中的3个上取得最优
- **效率优势**：MOS-8(K=5)只用5个对抗样本，而APGD-All用8个（效率提升37.5%），性能持平或更优
- 搜索得到的损失函数（ID 4-7）系统性优于经典损失函数（ID 0-3）
- 模型越复杂（WR-70-16），MOS与单目标攻击的差距越小，说明强模型的鲁棒性更均匀

## 亮点与洞察

1. **问题建模的优雅性**：将对抗攻击转化为多目标集合优化，smooth max/min的使用使原本的组合优化问题变得可微且可用梯度方法求解
2. **无参数设计**：框架不引入需要调节的额外超参数，权重向量固定为全1，对抗样本数K是唯一的可配置项
3. **协同模式的自动发现**：不仅提升攻击效果，更重要的是提供了对不同代理损失函数关系的系统性理解——哪些损失"天然搭配"，哪些"各自为战"

## 局限性

- smooth参数μ虽然设为固定值，但其最优选择可能随模型和数据集变化
- 协同模式的发现依赖于初始的8个损失函数选择，更大的损失函数库可能产生不同模式
- 仅在ℓ∞约束下验证，ℓ2等其他范数约束下的效果未知
- 计算成本虽然理论上只增加常数倍，但K个样本的批量前向/反向传播对GPU内存的需求更高
- 主要在分类任务的鲁棒性评估上验证，检测/分割等任务的适用性未探索

## 相关工作与启发

- **APGD/AutoAttack**: MOS的基础框架，MOS在其上增加了多目标优化的能力
- **ACG**: 使用共轭梯度的高级优化，是最强的单目标攻击基线
- **AutoLoss/Tightening**: 自动搜索代理损失函数，提供了Loss 4-7
- **启发**: 多目标优化视角可推广到对抗训练（防御端）；协同模式分析可指导鲁棒性评估标准的设计（如选择哪些损失函数作为benchmark）；smooth集合优化技术可应用于其他需要同时优化多个不可调和目标的AI安全问题

## 评分

⭐⭐⭐⭐ — 从多目标优化理论出发设计对抗攻击框架，数学形式优雅，工程实现简洁（基于APGD改动极小），在17个模型上系统性验证了有效性。协同模式挖掘提供了超越"工具开发"的科学价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2026\] DASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples](../../CVPR2026/ai_safety/dash_a_meta-attack_framework_for_synthesizing_effective_and_stealthy_adversarial.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/ai_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](../../ICML2025/ai_safety/understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [\[CVPR 2025\] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](invisible_backdoor_attack_against_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
