---
title: >-
  [论文解读] Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems
description: >-
  [ECCV 2024][LLM评测][电磁逆散射] 提出基于隐式神经表示（INR）的电磁逆散射问题（EISP）求解方案，通过将散射体的相对介电常数建模为连续隐式表示并在前向框架中优化，有效避免了逆估计的困难和离散化导致的低分辨率问题。 电磁逆散射问题（EISP）是计算成像领域的重要课题。通过电磁波穿透物体表面…
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "电磁逆散射"
  - "隐式神经表示"
  - "计算成像"
  - "前向估计"
  - "非侵入式成像"
---

# Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems

**会议**: ECCV 2024  
**arXiv**: [2407.09352](https://arxiv.org/abs/2407.09352)  
**代码**: 有 ([https://luo-ziyuan.github.io/Imaging-Interiors](https://luo-ziyuan.github.io/Imaging-Interiors))  
**领域**: LLM评测  
**关键词**: 电磁逆散射, 隐式神经表示, 计算成像, 前向估计, 非侵入式成像

## 一句话总结

提出基于隐式神经表示（INR）的电磁逆散射问题（EISP）求解方案，通过将散射体的相对介电常数建模为连续隐式表示并在前向框架中优化，有效避免了逆估计的困难和离散化导致的低分辨率问题。

## 研究背景与动机

电磁逆散射问题（EISP）是计算成像领域的重要课题。通过电磁波穿透物体表面，可以非侵入式地确定散射体内部的相对介电常数分布，进而实现内部结构成像。与X射线和MRI相比，电磁波提供了低成本和安全的非侵入式成像手段。

然而，求解EISP面临两大核心困难：

**逆估计的困难**：从测量的散射场反推相对介电常数，多重散射效应使该逆问题高度非线性且病态

**离散化的诅咒**：连续空间必须离散化为有限元素或网格进行数值计算，导致细节丢失和分辨率下降

现有方法的不足：
- **传统迭代方法**（如SOM、CSI）：将介电常数离散化为矩阵形式优化，无法解决离散化引起的低分辨率问题
- **深度学习方法**（如BPS、PGAN）：分为两阶段——先用传统方法获得粗糙图像，再用图像翻译网络细化，但第二阶段忽略了物理散射数据

## 方法详解

### 整体框架

核心思想是将EISP在前向估计过程中求解，避免逆估计的困难。具体做法：

1. 用MLP $F_\theta$ 将空间坐标映射到相对介电常数 $\varepsilon_r$
2. 用另一个MLP $H_\phi$ 将空间坐标和发射器位置映射到感应电流 $J$
3. 在前向过程中优化这两个隐式表示，使计算出的散射场与实测值匹配

两个MLP的使用避免了公式(7)中的复杂矩阵求逆运算，大幅降低了计算成本。

### 关键设计

**1. 相对介电常数的连续表示**

$$\varepsilon_r(\mathbf{x}) = F_\theta(\gamma(\mathbf{x}))$$

其中 $\gamma$ 为位置编码，将低维坐标投射到高维空间以增强拟合能力。

**2. 感应电流的连续表示**

$$J(\mathbf{x}, \mathbf{x}^t) = H_\phi(\gamma(\mathbf{x}), \gamma(\mathbf{x}^t))$$

感应电流同时依赖于空间坐标和发射器位置，这一设计忠实反映了物理关系。

**3. 随机空间采样策略**

将ROI区域划分为 $M \times M$ 网格，每个采样点从高斯分布中随机抽取：
$$x_m^{\text{sample}} \sim \mathcal{N}(x_m, \sigma^2)$$

这种概率采样确保了优化过程中对每个空间位置的全面考虑，避免固定离散位置的采样偏差。

### 损失函数 / 训练策略

**数据损失（Data Loss）**：对比计算散射场与实测散射场
$$\mathcal{L}_{\text{data}} = \sum_{p=1}^{N_t} \|\hat{\mathbf{E}}_p^s - \mathbf{E}_p^s\|^2$$

**状态损失（State Loss）**：对比从 $H_\phi$ 直接查询的感应电流与通过物理关系计算的感应电流
$$\mathcal{L}_{\text{state}} = \sum_{p=1}^{N_t} \|\hat{\mathbf{J}}_p - \mathbf{J}_p\|^2$$

**总损失**：
$$\mathcal{L} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}$$

其中 $\mathcal{L}_{\text{TV}}$ 为全变分正则化项。超参数设置: $\lambda_{\text{data}}=1.0$, $\lambda_{\text{state}}=1.0$, $\lambda_{\text{TV}}=0.01$。

实现细节：两个8层MLP，256通道，ReLU激活，ROI离散化为64×64，Adam优化器，学习率 $5 \times 10^{-4}$，4K次迭代。

## 实验关键数据

### 主实验（表格）

| 方法 | Circular(5%噪声)-RRMSE↓ | Circular(5%)-SSIM↑ | MNIST(5%)-RRMSE↓ | MNIST(5%)-SSIM↑ | Fresnel-RRMSE↓ |
|------|-------------------------|---------------------|-------------------|-----------------|----------------|
| **Proposed** | **0.016** | **0.968** | **0.017** | **0.972** | **0.127** |
| PGAN | 0.021 | 0.957 | 0.090 | 0.918 | 0.167 |
| Physics-Net | 0.024 | 0.945 | 0.079 | 0.938 | 0.168 |
| BPS | 0.027 | 0.964 | 0.098 | 0.912 | 0.166 |
| Gs SOM | 0.034 | 0.926 | 0.101 | 0.853 | 0.135 |
| BP | 0.048 | 0.916 | 0.171 | 0.750 | 0.180 |

### 消融实验（表格）

| 配置 | 参数量↓ | 每次迭代时间↓ | RRMSE↓ | SSIM↑ | PSNR↑ |
|------|---------|--------------|--------|-------|-------|
| 双MLP（完整方法） | 1,019,139 | 117 ms | 0.038 | 0.909 | 30.36 |
| 单MLP | 493,313 | 289 ms | 0.053 | 0.876 | 27.32 |

### 关键发现

1. 在合成数据和真实世界Fresnel数据库上，方法全面超越传统（BP、SOM）和深度学习（BPS、PGAN）基线
2. 在30%噪声水平下仍能准确重建，展示出优异的鲁棒性
3. 训练时使用64×64分辨率，推理时可灵活采样得到更高分辨率图像
4. 在仅使用25%标准测量数据的稀疏测量场景下仍表现优越
5. 方法可自然扩展到3D场景

## 亮点与洞察

- **前向估计代替逆估计**：巧妙地将难以直接求解的逆问题转化为在前向框架中的优化问题
- **双MLP策略**：分别表示介电常数和感应电流，避免矩阵求逆的计算瓶颈
- **INR的天然优势**：连续表示的分辨率灵活性天然解决了离散化的诅咒
- **随机采样**增强了模型对空间位置的全面覆盖

## 局限与展望

- 每个目标需单独优化（case-by-case），无法批量推理
- 优化迭代数仍有一定计算开销（4K次迭代）
- 未在医学诊断等实际应用场景中验证
- 可考虑通过改进INR结构（如TensoRF等加速方法）来缩短优化时间

## 相关工作与启发

- 将INR（NeRF家族的核心技术）应用于电磁成像这一非视觉领域，展示了INR的广泛适用性
- 前向优化+物理约束的范式可推广到其他逆问题
- 双网络分别建模不同物理量的思路值得在其他物理驱动问题中借鉴

## 评分

- **创新性**: ★★★★☆ — INR+前向估计的组合在EISP中属首创
- **实用性**: ★★★☆☆ — case-by-case优化限制了实际部署效率
- **实验完整性**: ★★★★★ — 合成/真实/噪声/稀疏/3D/消融均有覆盖
- **写作质量**: ★★★★☆ — 物理背景和方法阐述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)
- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](../../ICML2025/llm_evaluation/leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ICLR 2026\] ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems](../../ICLR2026/llm_evaluation/acadreason_exploring_the_limits_of_reasoning_models_with_academic_research_probl.md)
- [\[ECCV 2024\] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)
- [\[ECCV 2024\] MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)

</div>

<!-- RELATED:END -->
