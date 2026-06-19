---
title: >-
  [论文解读] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds
description: >-
  [AAAI 2026 (Neuro for AI & AI for Neuro Workshop, PMLR)][自监督学习][脉冲神经网络] 提出 CATFormer，一种基于脉冲视觉 Transformer 的无数据重放持续学习框架，通过上下文自适应的动态放电阈值实现任务特定的神经元兴奋性调节，在长达 100 个任务序列中不仅不遗忘反而准确率提升（"逆向遗忘"现象）。
tags:
  - "AAAI 2026 (Neuro for AI & AI for Neuro Workshop, PMLR)"
  - "自监督学习"
  - "脉冲神经网络"
  - "持续学习"
  - "动态阈值"
  - "类增量学习"
  - "Transformer"
---

# CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds

**会议**: AAAI 2026 (Neuro for AI & AI for Neuro Workshop, PMLR)  
**arXiv**: [2603.15184](https://arxiv.org/abs/2603.15184)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 脉冲神经网络, 持续学习, 动态阈值, 类增量学习, Vision Transformer

## 一句话总结

提出 CATFormer，一种基于脉冲视觉 Transformer 的无数据重放持续学习框架，通过上下文自适应的动态放电阈值实现任务特定的神经元兴奋性调节，在长达 100 个任务序列中不仅不遗忘反而准确率提升（"逆向遗忘"现象）。

## 研究背景与动机

### 问题定义

深度神经网络在真实世界的持续部署场景中面临**灾难性遗忘**问题：模型在学习新任务时会丧失对先前任务的知识。这在资源受限的边缘设备（如机器人、自主系统）上尤为严重，因为这些平台无法存储过去的数据进行重放（rehearsal），受限于能耗、隐私和存储。

### 现有方法的不足

**正则化方法**（EWC、SI）：通过约束重要参数的更新来防遗忘，但在 SNN 上效果很差（CIFAR-100 50 tasks 下 SI 仅 1.84%）

**重放方法**（iCaRL、DER++）：需要存储过去数据，违反隐私约束，且在硬件受限平台上不可行

**架构方法**（DSD-SNN）：随任务增多性能持续下降，从 10 tasks 60.47% 降至 50 tasks 50.55%

**SNN 持续学习**的研究主要基于 CNN 架构，尚未有人探索脉冲 Vision Transformer 的持续学习

### 生物启发

大脑的抗遗忘能力与**神经调节**密切相关。乙酰胆碱等神经调质通过调节膜兴奋性和突触可塑性来实现新记忆的快速编码，同时降低先前信息的干扰。这种**动态调节神经元放电阈值**的机制启发了 CATFormer 的核心设计。

## 方法详解

### 整体框架

CATFormer 基于 SpikFormer（脉冲视觉 Transformer）骨架，采用**两阶段训练协议**结合**门控动态头选择**进行任务无关推理。核心创新是：冻结骨架权重后，仅通过学习**任务特定的动态放电阈值**来适应新任务。

### 关键设计

#### 1. **动态阈值 LIF 神经元模型（DTLIF）**

在标准 LIF 神经元基础上引入上下文自适应的可学习放电阈值：

- **膜电位更新**：$\tilde{V}_j^{(t)} = (1 - \frac{1}{\tau}) V_j^{(t-1)} + \frac{1}{\tau} I_j^{(t)}$
- **脉冲产生**：$S_j^{(t)} = \Theta(\tilde{V}_j^{(t)} - \phi_j^{(k)})$，其中 $\phi_j^{(k)}$ 是任务 $k$ 特定的阈值
- **软重置**：$V_j^{(t)} = \tilde{V}_j^{(t)} - S_j^{(t)} \phi_j^{(k)}$

阈值通过梯度下降更新：$\phi_j^{(k)} \leftarrow \phi_j^{(k)} - \eta \frac{\partial \mathcal{L}}{\partial \phi_j^{(k)}}$

这使得每个通道可以为不同任务调整放电阈值，实现任务自适应的脉冲动态。

#### 2. **两阶段训练协议**

- **任务 0（基础任务）**：联合训练整个骨架 $\theta$、初始阈值 $\phi^{(0)}$ 和分类头 $W_0$，使用交叉熵损失
- **任务 k > 0（增量任务）**：**冻结所有先前参数**，仅优化新的分类头 $W_k$ 和新的阈值参数 $\phi^{(k)}$

$$\min_{\{\phi^{(k)}, W_k\}} \mathbb{E}_{(x,y) \sim \mathcal{D}^k} [\mathcal{L}_{CE}(W_k \cdot f(x; \theta, \phi^{(k)}), y)]$$

每个任务仅需存储约 16,032 个阈值参数（64.2 KB FP32），极其轻量。

#### 3. **门控动态头选择（G-DHS）**

推理时通过一个两层 MLP 门控网络进行任务预测和分类头路由：

- 使用**基础阈值** $\phi_{init}$ 提取特征 $\mathbf{f}_{base}(x)$
- 门控网络预测任务 ID：$k^* = \arg\max(\mathcal{G}(\mathbf{f}_{base}(x)))$
- 切换到任务特定阈值 $\phi^{(k^*)}$ 重新提取特征并分类

门控网络结构：$\mathcal{G}(\mathbf{f}) = \text{Linear}(\text{ReLU}(\text{Linear}(\mathbf{f})))$，$\mathbb{R}^D \to \mathbb{R}^{D/4} \to \mathbb{R}^k$

### 损失函数 / 训练策略

- 骨架和分类头使用标准交叉熵损失 $\mathcal{L}_{CE}$
- 门控 MLP 在每个任务学完后，使用基础阈值提取的特征单独训练（也用交叉熵），特征仅在当前任务局部使用，不跨任务存储
- 阈值初始化为 $\phi_{init} = 0.5$

## 实验关键数据

### 主实验

| 数据集 | 任务数 | 本文 (CATFormer) | DSD-SNN | EWC | iCaRL | DER++ |
|--------|--------|-----------------|---------|-----|-------|-------|
| CIFAR-100 | 10 | **68.33** ± 4.51 | 60.47 ± 0.72 | 18.81 | 33.46 | 34.99 |
| CIFAR-100 | 25 | **71.34** ± 1.75 | 53.79 ± 2.67 | 15.73 | 22.37 | 24.90 |
| CIFAR-100 | 50 | **75.66** ± 2.72 | 50.55 ± 1.76 | 9.73 | 10.89 | 13.12 |
| CIFAR-10 | 5 | **89.29** ± 2.53 | — | 80.39 (SA-SNN+EWC) | — | — |
| Tiny-ImageNet | 100 | **48.56** ± 0.81 | — | — | — | 40.11* |
| CIFAR10-DVS | 5 | **87.14** ± 2.78 | 76.57 | — | — | — |
| SHD (T=16) | 10 | **87.85** ± 1.20 | 80.47 | — | — | — |

### 消融实验

| 配置 | 准确率 (%) | 任务 0 准确率 (%) | 说明 |
|------|-----------|-----------------|------|
| **CATFormer（完整）** | **89.29** ± 2.53 | 93.87 ± 0.45 | 完整模型 |
| Fixed Threshold | 42.87 ± 1.26 | 72.59 ± 1.86 | 固定阈值→严重遗忘 |
| SpikIdentityFormer | 59.38 ± 0.98 | 70.62 ± 1.75 | 移除注意力 |
| Random Identity Former | 53.17 ± 2.13 | 62.43 ± 0.99 | 随机替换注意力 |
| FFN Frozen | 63.24 ± 1.78 | 72.17 ± 1.59 | 冻结前馈网络 |

### 关键发现

1. **逆向遗忘现象**：CATFormer 随任务数增加准确率反而提升（68.33%→75.66%），这与所有现有方法的下降趋势完全相反
2. **参数效率极高**：第 k 个任务仅更新约 1.4M 参数（vs 其他方法更新 9.32M+全模型），每任务仅需 64.2 KB 存储
3. **Fixed Threshold 消融**证明动态阈值是防遗忘的核心，而非仅靠突触可塑性
4. **神经形态数据集兼容**：在 CIFAR10-DVS 和 SHD 等事件驱动数据集上也表现优异

## 亮点与洞察

1. **逆向遗忘**是本文最令人惊讶的发现：当每个任务包含更少类别时，模型学得更好。这更贴合真实场景（机器人不会一次遇到 50 个新类别）
2. **阈值即记忆**的设计理念新颖：不修改权重，仅通过调节神经元兴奋性来编码新知识，类似于大脑的神经调质机制
3. 将持续学习研究从 CNN 骨架推进到脉冲 Transformer，填补了重要空白

## 局限与展望

1. 门控网络的任务预测准确率未单独报告，这可能是系统瓶颈
2. 仅在分类任务上验证，未涉及检测、分割等更复杂任务
3. 缺乏在真实神经形态芯片（如 Loihi 2）上的部署验证
4. 初始任务的训练需要较大计算量（10.5M 参数），与声称的轻量化有一定矛盾
5. 阈值参数虽少但随任务线性增长，缺乏上限分析

## 相关工作与启发

- **DSD-SNN**（Han et al., 2023）是之前最强的无重放 SNN 持续学习方法，但基于 CNN 且性能持续下降
- **SpikFormer**（Zhou et al., 2023）将 Transformer 引入 SNN，但未探索持续学习
- 阈值调节的思想可推广到其他 SNN 架构和非视觉任务

## 评分

- 新颖性: ⭐⭐⭐⭐ （动态阈值用于持续学习是新颖的，但整体架构相对简单）
- 实验充分度: ⭐⭐⭐⭐ （覆盖静态和神经形态数据集，消融充分）
- 写作质量: ⭐⭐⭐⭐ （条理清晰，生物启发的叙述良好）
- 价值: ⭐⭐⭐⭐ （逆向遗忘现象有重要启发意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Temporal Interaction in Spiking Transformers with Multi-Delay Mixer](../../CVPR2026/self_supervised/temporal_interaction_in_spiking_transformers_with_multi-delay_mixer.md)
- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[CVPR 2026\] Is Parameter Isolation Better for Prompt-Based Continual Learning?](../../CVPR2026/self_supervised/is_parameter_isolation_better_for_prompt-based_continual_learning.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](../../CVPR2026/self_supervised/diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] A Faster Path to Continual Learning](../../CVPR2026/self_supervised/a_faster_path_to_continual_learning.md)

</div>

<!-- RELATED:END -->
