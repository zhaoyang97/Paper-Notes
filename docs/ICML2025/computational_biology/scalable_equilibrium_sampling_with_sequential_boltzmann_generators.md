---
title: >-
  [论文解读] Scalable Equilibrium Sampling with Sequential Boltzmann Generators
description: >-
  [ICML2025][计算生物][Boltzmann生成器] SBG通过Transformer架构规范化流(TarFlow)和退火Langevin动力学的序列蒙特卡洛，首次在笛卡尔坐标系中实现六肽(66原子)系统的高效平衡采样。 现有痛点 现有痛点：分子系统存在多个亚稳定态，能垒导致态间转移极慢。传统MCMC/MD需飞秒级时…
tags:
  - "ICML2025"
  - "计算生物"
  - "Boltzmann生成器"
  - "规范化流"
  - "分子采样"
  - "退火Langevin动力学"
  - "蛋白质肽链"
---

# Scalable Equilibrium Sampling with Sequential Boltzmann Generators

**会议**: ICML2025  
**arXiv**: [2502.18462](https://arxiv.org/abs/2502.18462)  
**代码**: [GitHub](https://github.com/transferable-samplers/transferable-samplers)  
**领域**: 计算生物  
**关键词**: Boltzmann生成器, 规范化流, 分子采样, 退火Langevin动力学, 蛋白质肽链

## 一句话总结
SBG通过Transformer架构规范化流(TarFlow)和退火Langevin动力学的序列蒙特卡洛，首次在笛卡尔坐标系中实现六肽(66原子)系统的高效平衡采样。

## 研究背景与动机

### 现有痛点

**现有痛点**：分子系统存在多个亚稳定态，能垒导致态间转移极慢。传统MCMC/MD需飞秒级时间步长的极长模拟。

### 现有Boltzmann生成器的瓶颈

1. 架构表达力不足：等变连续流不够高效
2. 提议-目标分布重叠差：SNIS方差极大，ESS很小
3. 先前最好的BG方法只能处理二肽(2个氨基酸，22原子)

### 双轴扩展

预训练改进：可扩展非等变架构(TarFlow)替代等变流。推理时改进：退火Langevin渐进运输样本。

## 方法详解

### 关键设计1：软等变规范化流
放弃硬等变，采用TarFlow（Vision Transformer分块掩码自回归流）。

软等变实现：
- 旋转等变：训练中随机旋转数据增强
- 平移等变：质心噪声+推理时Proposition 1补偿

### 关键设计2：退火Langevin动力学
能量插值从提议流能量到目标Boltzmann能量，利用Jarzynski等式追踪重要性权重。计算比先验出发更有信息量。

### 理论保证
Proposition 1证明质心调整后ESS严格提升。

### 损失函数 / 训练策略
模型采用端到端训练，优化目标综合考虑任务损失和正则化项。


## 实验关键数据

### 肽链系统采样能力


### 主实验

| 系统 | 原子数 | SBG-SNIS | SBG-AIS | 连续BG |
|------|-------|---------|---------|--------|
| 二肽 | 22 | 优秀 | 优秀 | 可行 |
| 三肽 | 33 | 良好 | 优秀 | 失败 |
| 四肽 | 44 | 可行 | 良好 | 失败 |
| 六肽 | 66 | - | 可行 | 失败 |

### ESS改进


### 消融实验

| 系统 | SNIS ESS | AIS ESS | 提升 |
|------|---------|---------|------|
| 三肽 | ~0.3 | >0.8 | 2.7x |
| 四肽 | ~0.1 | >0.5 | 5x |
| 六肽 | ~0 | 有统计意义 | 从不可行到可行 |

### 关键发现
1. 退火是关键：加入退火后性能跃升
2. 软等变有效：灵活参数化胜过严格几何约束
3. 从二肽到六肽是质变

## 亮点与洞察

1. 放弃硬等变用Transformer突破扩展性，反映ML领域大趋势。
2. 推理时退火带来指数级采样质量提升。
3. 物理+ML完美融合：Boltzmann分布、Jarzynski等式+TarFlow、SMC。
4. 质心调整有严格理论证明(Proposition 1)。
5. 代码开源。

## 局限与展望

1. 六肽(66原子)是极限，真实蛋白(数千原子)仍遥远。
2. 退火计算成本远高于SNIS。
3. 依赖精确能量梯度，黑盒势能不适用。
4. 质心噪声参数选择未充分讨论。
5. 未对比扩散模型采样方法。

### 进一步展望
- 将TarFlow架构与更高效的等变操作结合，取长补短
- 探索多分辨率退火策略，动态调整步长
- 在全原子（非粗粒化）蛋白上验证
- 与AlphaFold等结构预测模型集成，利用其先验
- 十肽及更大系统的完整验证是必须攻克的下一个里程碑
- 可尝试将SBG用于药物-蛋白结合自由能估计

## 相关工作与启发

- Noe et al. 2019：原始BG框架。NETS：流匹配+非平衡采样。
- TarFlow：本文首次应用于分子。
- 启发：软约束优于硬约束的哲学可推广到科学ML。

## 评分
- 新颖性: 4.5/5
- 实验充分度: 4.0/5
- 写作质量: 4.0/5
- 价值: 4.0/5

## 补充分析

### 方法对比总结表

| 方法 | 使用能量 | 精确似然 | 使用数据 | 退火 |
|------|---------|---------|---------|------|
| DEM | 是 | 否 | 否 | 否 |
| NETS | 是 | 是 | 否 | 是 |
| BG | 是 | 是 | 是 | 否 |
| SBG(本文) | 是 | 是 | 是 | 是 |

SBG是唯一同时具备四个关键特性的方法。

### Alanine dipeptide验证
22原子系统，OM优化生成两条可能的转移通路，将路径作为集合变量做伞形采样，精确估计自由能垒约6 kcal/mol。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching](scalable_generation_of_spatial_transcriptomics_from_histology_images_via_whole-s.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/computational_biology/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/computational_biology/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](../../NeurIPS2025/computational_biology/split_gibbs_discrete_diffusion_posterior_sampling.md)

</div>

<!-- RELATED:END -->
