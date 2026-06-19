---
title: >-
  [论文解读] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex
description: >-
  [NeurIPS 2025][可解释性][潜变量模型] 提出TE-ViDS，一种时序潜变量模型，将视觉神经活动分解为与视觉刺激相关的外部表征和反映内部状态的内部表征，通过时间演化结构和对比学习实现最优的自然场景/视频解码性能。 潜变量模型(LVMs)通过构建低维表征揭示神经活动与行为/感官刺激的内在关联…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "潜变量模型"
  - "视觉神经活动"
  - "时间演化动力系统"
  - "对比学习"
  - "小鼠视觉皮层"
---

# Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex

**会议**: NeurIPS 2025  
**arXiv**: [2408.07908](https://arxiv.org/abs/2408.07908)  
**代码**: [有](https://github.com/Grasshlw/Time-Evolving-Visual-Dynamical-System)  
**领域**: 可解释性  
**关键词**: 潜变量模型, 视觉神经活动, 时间演化动力系统, 对比学习, 小鼠视觉皮层

## 一句话总结

提出TE-ViDS，一种时序潜变量模型，将视觉神经活动分解为与视觉刺激相关的外部表征和反映内部状态的内部表征，通过时间演化结构和对比学习实现最优的自然场景/视频解码性能。

## 研究背景与动机

潜变量模型(LVMs)通过构建低维表征揭示神经活动与行为/感官刺激的内在关联，是神经数据分析的核心方法。然而现有研究存在三个重要空白：

**领域偏向运动皮层**: 多数LVM研究聚焦运动脑区的神经数据（如预规划运动），对**视觉皮层**的研究很少

**忽略时间关系**: 自然视觉刺激本质上是高维且时间依赖的，但多数LVM未显式建模神经活动的时间关系

**视觉特性未充分利用**: 视觉神经活动包含刺激相关成分和内部状态成分，现有方法缺乏针对这些特性的设计

**具体挑战**: 在小鼠被动观察自然场景/视频时，其视觉皮层的神经动态受两方面影响：
- **外部视觉刺激**: 场景/电影帧内容
- **内部状态**: 注意力、觉醒程度等，这些内部状态甚至可能比视觉刺激对神经活动的影响更大

如何构建能够解耦这两部分的高质量潜变量表征，是一个关键问题。

## 方法详解

### 整体框架

TE-ViDS是一个序列潜变量模型，核心架构包括：
- **编码器**: 从序列脉冲数据中提取空间特征
- **时间演化系统**: 通过RNN的状态因子条件性地演化潜变量
- **解码器**: 从潜变量映射到推断发放率
- **分离设计**: 外部潜变量(确定性) + 内部潜变量(随机性)

输入为 $\mathbf{x} = (\mathbf{x}_1, ..., \mathbf{x}_T) \in \mathbb{R}^{T \times N}$（T个时间窗口，N个神经元的脉冲计数）。

### 关键设计

#### 1. 外部潜变量（确定性，刺激相关）

**功能**: 捕捉与视觉刺激相关的神经活动成分。

$$\mathbf{z}_t^{(e)} = f_{\text{enc}}^{(e)}(f_x(\mathbf{x}_t), \mathbf{h}_{t-1}^{(e)})$$

**核心思路**: 设计为确定性值（非随机），因为刺激相关成分应当是稳定的，变异性应归属于内部状态。通过对比学习(NT-Xent loss)塑造——将时间偏移的序列作为正样本（因为相邻时间的视觉刺激相似），从训练集随机采样负样本。

**设计动机**: 正样本对覆盖相似视觉刺激的时间段，使外部表征自然地与刺激内容关联。同时采用swap操作——交换正样本对的外部表征但保留内部表征，增强解耦效果。

#### 2. 内部潜变量（随机性，状态相关）

**功能**: 反映动物的内部动态状态（注意力、觉醒度等），包含高变异性和噪声。

近似后验：$\mathbf{z}_t^{(i)} | \mathbf{x}_{1:t}, \mathbf{h}_{1:t-1}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}_{z,t}, \boldsymbol{\sigma}_{z,t}^2 \cdot \mathbf{I})$

先验分布：$\tilde{\mathbf{z}}_t^{(i)} | \mathbf{h}_{1:t-1}^{(i)} \sim \mathcal{N}(\tilde{\boldsymbol{\mu}}_{z,t}, \tilde{\boldsymbol{\sigma}}_{z,t}^2 \cdot \mathbf{I})$

**核心思路**: 设计为随机变量，先验分布仅依赖前一步的状态因子（具有时间自发性），通过KL散度约束后验与先验的差距。

**设计动机**: 内部状态本质上变化大、噪声多，随机建模更合理。时间依赖的先验使模型能捕捉内部状态的缓慢漂移。

#### 3. 时间演化机制（GRU状态因子）

两个独立的GRU分别维护外部和内部状态因子：

$$\mathbf{h}_t^{(e)} = f_{\text{GRU}}^{(e)}(f_x(\mathbf{x}_t), \mathbf{h}_{t-1}^{(e)})$$
$$\mathbf{h}_t^{(i)} = f_{\text{GRU}}^{(i)}(f_x(\mathbf{x}_t), \mathbf{z}_t^{(e)}, \mathbf{z}_t^{(i)}, \mathbf{h}_{t-1}^{(i)})$$

关键区别：内部状态因子的GRU额外接收外部潜变量作为输入，反映了内部状态不可避免地受视觉刺激影响的事实。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{recons}} + \beta \mathcal{L}_{\text{contrastive}} + \gamma \mathcal{L}_{\text{regular}}$$

- $\mathcal{L}_{\text{recons}}$: Poisson负对数似然（重建脉冲计数）
- $\mathcal{L}_{\text{contrastive}}$: NT-Xent对比损失（塑造外部表征）
- $\mathcal{L}_{\text{regular}}$: KL散度 + 先验正则化（约束内部表征）

## 实验关键数据

### 主实验1：自然场景解码（118张场景图片）

| 模型 | Mouse 1 | Mouse 2 | Mouse 3 | Mouse 4 | Mouse 5 |
|------|---------|---------|---------|---------|---------|
| PCA | 0.59% | 1.53% | 1.53% | 0.80% | 0.85% |
| LFADS | 30.76% | 16.46% | 22.20% | 19.69% | 4.69% |
| pi-VAE | 7.49% | 19.42% | 22.92% | 13.71% | 2.22% |
| Swap-VAE | 32.81% | 24.34% | 14.36% | 14.85% | 3.92% |
| CEBRA | 1.53% | 3.42% | 4.86% | 2.81% | 1.08% |
| TE-ViDS-small | 47.08% | 23.95% | 29.08% | 34.95% | 9.93% |
| **TE-ViDS** | **50.86%** | **27.24%** | **29.90%** | **38.05%** | **9.44%** |

TE-ViDS在所有5只小鼠上均取得最高解码分数，比次优模型提升显著（Mouse 1提升18%、Mouse 4提升23%）。

### 主实验2：自然电影帧解码（900帧，1秒窗口）

| 模型 | Mouse 1 | Mouse 2 | Mouse 3 | Mouse 4 | Mouse 5 |
|------|---------|---------|---------|---------|---------|
| PCA | 8.44% | 28.77% | 25.42% | 21.56% | 11.69% |
| LFADS | 8.94% | 26.57% | 26.77% | 24.76% | 12.69% |
| Swap-VAE | 12.19% | 51.31% | 45.96% | 41.53% | 22.70% |
| CEBRA | 10.62% | 52.76% | 61.01% | 42.11% | 22.33% |
| **TE-ViDS** | **13.88%** | **65.38%** | **59.88%** | **54.33%** | **30.18%** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 外部 vs 内部表征 | 外部表征解码分数 >> 内部表征 | 验证了外部表征捕捉刺激相关信息的假设 |
| 时间/非时间合成数据 | 打乱时间维度后性能剧降 | 证明模型对时间关系的敏感性 |
| TE-ViDS vs TE-ViDS-small | 相当或略优 | 小模型也有效，非参数量堆叠 |
| 6个皮层区域对比 | VISp最高，VISrl最低 | 提供视觉皮层功能层级的计算证据 |

### 关键发现

1. **个体差异的内在机制**: 通过RSA分析发现Mouse 1的神经表征在不同场景下分为两个时期（内部状态变化所致），而Mouse 2无此现象。这解释了不同小鼠解码性能的巨大差异。
2. **皮层层级证据**: 初级/中级视觉区域(VISp, VISl, VISal)解码性能高于高级区域(VISpm, VISam)，多感觉区域VISrl最低——为小鼠视觉皮层功能层级提供了新的计算证据。
3. **CEBRA的局限**: CEBRA在自然场景解码上表现极差（~3%），说明其固定卷积核的时间编码方式不适合静态刺激下的时间特征提取。

## 亮点与洞察

1. **刺激相关+状态相关的解耦策略**: 确定性外部+随机性内部的设计精准匹配了视觉神经活动的两种成分
2. **对比学习的自然应用**: 利用时间偏移作为正样本的设计非常自然——相邻时间的视觉刺激确实相似
3. **生物学洞察丰富**: 不仅是方法论贡献，还揭示了内部状态对视觉编码的影响、皮层区域的功能差异等神经科学发现
4. **方法通用性**: 不局限于小鼠视觉皮层，可扩展到其他物种、脑区和模态

## 局限与展望

1. **内部表征缺乏定量评估**: 没有行为或内部状态的录制数据来验证内部潜变量的解释性
2. **个体差异大**: Mouse 1和Mouse 5的解码性能差距悬殊，模型未能完全克服个体间的变异性
3. **仅被动观看范式**: 小鼠没有执行任务，无法建立与任务行为的直接关联
4. **计算成本未详细讨论**: GRU序列处理的时间复杂度可能在超长时间序列上成为瓶颈

## 相关工作与启发

- 与CEBRA(Schneider 2023)对比: CEBRA用固定卷积核编码时间特征，但TE-ViDS用RNN动态演化更适合视觉神经活动
- 与Swap-VAE(Liu 2021)对比: 借鉴了swap操作和split结构，但增加了时间演化机制
- 内部状态对感知的影响与Ashwood(2022)的行为研究一致
- 未来可结合脑机接口应用，利用解耦表征分别处理刺激和状态信息

## 评分
- 新颖性: ⭐⭐⭐⭐ (时间演化+解耦设计合理但非革命性)
- 实验充分度: ⭐⭐⭐⭐⭐ (合成数据+真实神经数据，多动物多区域全面分析)
- 写作质量: ⭐⭐⭐⭐ (方法清晰，生物学讨论深入)
- 价值: ⭐⭐⭐⭐ (填补视觉神经活动LVM的空白，提供有价值的神经科学洞察)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[CVPR 2025\] Learning Visual Composition through Improved Semantic Guidance](../../CVPR2025/interpretability/learning_visual_composition_through_improved_semantic_guidance.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](../../ICLR2026/interpretability/decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[NeurIPS 2025\] H-SPLID: HSIC-based Saliency Preserving Latent Information Decomposition](h-splid_hsic-based_saliency_preserving_latent_information_decomposition.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)

</div>

<!-- RELATED:END -->
