---
title: >-
  [论文解读] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data
description: >-
  [NeurIPS 2025][计算生物][蛋白质结构预测] CryoBoltz利用冷冻电镜（cryo-EM）密度图通过多尺度引导机制（全局→局部）引导预训练扩散结构预测模型（Boltz-1）的采样轨迹，无需重新训练即可生成与实验数据一致的多构象原子模型。 当前蛋白质结构预测领域面临两大挑战的鸿沟： 结构预测模型的单构象偏差：…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "蛋白质结构预测"
  - "冷冻电镜"
  - "扩散模型引导"
  - "构象多样性"
  - "Boltz-1"
---

# Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data

**会议**: NeurIPS 2025  
**arXiv**: [2506.04490](https://arxiv.org/abs/2506.04490)  
**代码**: [GitHub](https://github.com/ml-struct-bio/cryoboltz)  
**领域**: 计算生物  
**关键词**: 蛋白质结构预测, 冷冻电镜, 扩散模型引导, 构象多样性, Boltz-1

## 一句话总结

CryoBoltz利用冷冻电镜（cryo-EM）密度图通过多尺度引导机制（全局→局部）引导预训练扩散结构预测模型（Boltz-1）的采样轨迹，无需重新训练即可生成与实验数据一致的多构象原子模型。

## 研究背景与动机

当前蛋白质结构预测领域面临两大挑战的鸿沟：

**结构预测模型的单构象偏差**：AlphaFold3、Boltz-1等扩散模型虽能生成结构，但其采样分布高度集中于单一构象——例如Boltz-1只采样到STP10的外向构象，AlphaFold3只采样到内向构象。现有MSA子采样方法虽能略微增加多样性，但效果有限且无法覆盖构象连续体。

**Cryo-EM重建到原子模型的瓶颈**：cryo-EM实验能捕获蛋白质的构象景观，但产出的3D密度图并非原子模型。现有Model Building方法（如ModelAngelo）依赖高分辨率图（<4Å），在低分辨率（>4Å）或异质性复合物面前频繁失败——例如P-glycoprotein的4个构象中ModelAngelo仅建模了2.3%-40.3%的残基。

本文的核心思路是：**将cryo-EM密度图中的实验信息注入到预训练扩散模型的反向采样过程中**，既利用了结构预测模型学到的序列和生物物理先验，又利用了实验数据捕获的真实构象信息。

## 方法详解

### 整体框架

CryoBoltz基于DPS（Diffusion Posterior Sampling）框架，将预训练的Boltz-1扩散模型视为隐式先验$p(\mathbf{x}|\mathbf{s})$，将cryo-EM密度图作为观测$\mathbf{y}$，通过引导反向扩散过程采样后验$p(\mathbf{x}|\mathbf{y},\mathbf{s})$。整个过程分四个阶段，共200步扩散：

**暖机（Warm-up）→ 全局引导 → 局部引导 → 松弛（Relaxation）**

### 关键设计

1. **全局引导（Global Guidance）**：将密度图$\mathbf{y} \in \mathbb{R}^{w \times h \times d}$通过加权k-means聚类转化为3D点云$\mathbf{Y} \in \mathbb{R}^{k \times 3}$，其中$k = \lfloor N/(4r^3) \rfloor$（$N$为原子数，$r$为体素大小）。引导项基于Sinkhorn散度（正则化Wasserstein距离）：

$$\tilde{s}_\theta(\mathbf{y}, \mathbf{x}, \mathbf{s}, t) = -\nabla_\mathbf{x} \mathfrak{D}(\hat{\mathbf{x}}_\theta(\mathbf{x}, \mathbf{s}, t), \mathbf{Y})$$

这一阶段只关心蛋白质的全局形状（如哪些螺旋朝内/朝外），不涉及高分辨率细节，避免了非线性似然函数在早期带来的优化困难。引导强度按余弦退火从0.25降至0.05。

2. **局部引导（Local Guidance）**：使用原始密度图，基于cryo-EM物理正向模型（每个非氢原子贡献一个高斯散射势），引导项直接最小化模拟密度图与实验密度图的L2距离：

$$\tilde{s}_\theta(\mathbf{y}, \mathbf{x}, \mathbf{s}, t) = -\nabla_\mathbf{x} \|\mathbf{y} - \mathcal{B}(\Gamma(\hat{\mathbf{x}}_\theta(\mathbf{x}, \mathbf{s}, t), \mathbf{s}))\|^2$$

其中$\Gamma$是将原子坐标映射为密度的算子，$\mathcal{B}$模拟B因子的模糊效应。引导强度固定为$\lambda=0.5$。

3. **多尺度引导调度**：合成数据用$T_w=125, T_g=25, T_l=25, T_r=25$，实验数据用$T_w=100, T_g=50, T_l=25, T_r=25$（实验数据全局引导更多步以驱动更大构象变化）。暖机和松弛阶段不加引导——暖机建立合理初始化，松弛修正空间位阻冲突等细节。

### 损失函数 / 训练策略

CryoBoltz**无需任何训练**，完全是推理时（inference-time）引导方法。核心优势在于：可随基础模型的持续改进而自动受益。每次实验采样25个结构×3个replicate，选RMSD最低的作为最终结果。计算在单块A100 80GB GPU上完成。

## 实验关键数据

### 主实验（合成数据）

| 系统 | 指标 | CryoBoltz | Boltz-1 | Boltz-1+MSA子采样 | AlphaFold3 |
|---|---|---|---|---|---|
| STP10 (内向) | All-atom RMSD (Å) | **1.057** | 3.815 | 3.768 | 1.263 |
| STP10 (外向) | All-atom RMSD (Å) | **0.888** | 2.656 | 2.542 | 4.478 |
| CH67抗体 | Local RMSD (Å) | **1.269** | 3.120 | 3.270 | 3.191 |
| CH67抗体 | TM score | **0.994** | 0.972 | 0.969 | 0.971 |

### 主实验（实验数据）

| 系统 | 分辨率(Å) | CryoBoltz RMSD | Boltz-1 RMSD | AF3 RMSD | ModelAngelo完整度 |
|---|---|---|---|---|---|
| P-gp (apo) | 4.3 | **1.382** | 6.994 | 3.827 | 40.3% |
| P-gp (内向) | 4.4 | **1.348** | 5.630 | 2.692 | 18.3% |
| P-gp (闭塞) | 4.1 | **1.727** | 2.929 | 3.440 | 2.3% |
| Pma1 (抑制) | 3.52 | **1.999** | 6.140 | 8.017 | 72.8% |
| CYP (闭合) | 4.4 | **2.004** | 8.784 | 3.585 | 18.9% |
| CYP (开放) | 6.5 | **4.167** | 8.532 | 6.490 | 0.0% |

### 消融实验

| 配置 | STP10内向 RMSD | STP10外向 RMSD | 说明 |
|---|---|---|---|
| 完整CryoBoltz | **1.057** | **0.888** | 全局+局部引导 |
| 仅局部引导 | 3.860 | 2.722 | 缺少全局引导，难以驱动大构象变化 |
| 仅全局引导 | 1.287 | 1.164 | 缺少局部引导，高分辨率细节不足 |

### 关键发现

- CryoBoltz在**所有10个实验密度图**上均优于Boltz-1和AlphaFold3
- 未经引导的Boltz-1/AF3通常只采样到一种构象，CryoBoltz成功采样到两种乃至四种构象
- ModelAngelo在低分辨率图（>4Å）上严重失败，6.5Å时完整度为0%
- 全局引导驱动大范围构象变化，局部引导精修高分辨率细节，两者缺一不可

## 亮点与洞察

- **零训练开销**：纯推理时方法，可即插即用到任何扩散结构预测模型
- **多尺度引导的必要性**：密度图到原子模型的正向映射高度非线性，直接优化似然函数会导致多模态问题，点云中间表示优雅地解决了早期引导的稳定性
- **实际应用价值极高**：cryo-EM模型建模通常需数小时人工精修，CryoBoltz可在分钟级完成

## 局限与展望

- 优化稳定性受似然$p(\mathbf{y}|\mathbf{x})$多模态影响，需采样多个结构取最优（增加计算开销）
- 暖机阶段依赖基础模型的合理初始化，部分复杂系统（如DSL1/SNARE复合物）初始化失败
- 各引导阶段的步数划分为启发式设定，尚无自动选择策略
- 多密度图场景下，尚未探索共享的变形模型来替代独立优化

## 相关工作与启发

- **DPS / ScoreALD**：扩散模型后验采样框架，CryoBoltz直接建立在DPS之上
- **ROCKET**：利用AF2作为正则化器，但在AF2的潜空间中优化；CryoBoltz在原子空间直接优化
- **ModelAngelo**：当前最先进的de novo模型建模方法，但受限于高分辨率需求
- 启发：扩散模型的推理时引导是将实验数据与学习先验结合的通用范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将DPS引导框架应用于蛋白质结构预测+cryo-EM数据融合
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+真实数据、6个生物系统、多基线对比、详尽消融和统计检验
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法推导严谨，图表精美
- 价值: ⭐⭐⭐⭐⭐ 解决结构生物学重要痛点，代码开源，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Quantifying the Role of OpenFold Components in Protein Structure Prediction](quantifying_the_role_of_openfold_components_in_protein_structure_prediction.md)
- [\[CVPR 2026\] CryoKRAQEN: Kernel-Regularized Annealing for Quantized Embedding Networks in Cryo-EM Heterogeneous Reconstruction](../../CVPR2026/computational_biology/cryokraqen_kernel-regularized_annealing_for_quantized_embedding_networks_in_cryo.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](../../ICML2026/computational_biology/protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)

</div>

<!-- RELATED:END -->
