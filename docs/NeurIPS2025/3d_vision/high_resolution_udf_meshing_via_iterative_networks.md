---
title: >-
  [论文解读] High Resolution UDF Meshing via Iterative Networks
description: >-
  [NeurIPS 2025][3D视觉][无符号距离场] 本文提出首个针对无符号距离场（UDF）的迭代式网格化方法，通过多轮次前向传播逐步将邻域信息传播到局部体素的伪符号预测中，有效解决了高分辨率下神经 UDF 噪声导致的表面空洞和不连续问题，在多个数据集上显著优于现有单遍方法。 领域现状：隐式神经表示中…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "无符号距离场"
  - "网格化"
  - "迭代网络"
  - "伪符号预测"
  - "高分辨率表面重建"
---

# High Resolution UDF Meshing via Iterative Networks

**会议**: NeurIPS 2025  
**arXiv**: [2509.17212](https://arxiv.org/abs/2509.17212)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 无符号距离场, 网格化, 迭代网络, 伪符号预测, 高分辨率表面重建

## 一句话总结
本文提出首个针对无符号距离场（UDF）的迭代式网格化方法，通过多轮次前向传播逐步将邻域信息传播到局部体素的伪符号预测中，有效解决了高分辨率下神经 UDF 噪声导致的表面空洞和不连续问题，在多个数据集上显著优于现有单遍方法。

## 研究背景与动机

**领域现状**：隐式神经表示中，有符号距离场（SDF）通过符号变化定位表面，Marching Cubes 等经典算法可以高效三角化。无符号距离场（UDF）能表示开放表面，比 SDF 更通用，但三角化困难得多——UDF 在表面处值为零，没有符号变化可以利用。

**现有痛点**：现有 UDF 网格化方法（MeshUDF、NSD-UDF、DCUDF、DualMesh-UDF 等）基本上在单个体素内独立操作，通过预测伪符号或双轮廓顶点来还原表面。但神经 UDF 本身是有噪声的——UDF 值可能无法在表面位置精确达到零，梯度方向也可能不准确。

**核心矛盾**：反直觉地，**提高网格化分辨率反而会恶化问题**，因为高分辨率下体素更小，UDF 噪声对单个体素的影响更大。单遍、单体素的方法缺乏足够的上下文来在噪声区域做出正确判断，导致表面出现空洞和不连续。

**本文目标** 如何在高分辨率下鲁棒地从噪声 UDF 中还原完整、准确的三角网格？

**切入角度**：观察到虽然表面附近的 UDF 值和梯度可能不准确，但远离表面处的梯度仍然可靠，且正确重建的表面区域包含对邻域判断有价值的信息。因此应该利用已提取表面元素的信息来帮助判断相邻的模糊区域。

**核心 idea**：将 UDF 网格化从单遍独立操作变为多遍迭代过程，每轮利用上一轮的输出（邻域伪符号配置）作为额外输入进行空间信息传播。

## 方法详解

### 整体框架
给定神经 UDF $U_\mathcal{S}$，在规则网格上查询 UDF 值和梯度。按体素分组后，每个体素输入到一个逐体素的全连接网络 $f_\theta$。网络的任务是为体素的 8 个角点预测伪符号配置（128 类分类问题）。关键创新在于：网络不仅输入 UDF 值和梯度，还输入目标体素及其邻居的当前伪符号配置。通过多轮迭代，每轮的输出作为下轮的输入，逐步传播空间信息。最终伪符号用 Marching Cubes 或 DualMesh-UDF 三角化。

### 关键设计

1. **迭代式伪符号预测网络**:

    - 功能：逐体素预测伪符号配置，并通过迭代传播邻域信息。
    - 核心思路：网络输入为 UDF 值、梯度、以及目标体素和相邻体素（共享面的 6 个邻居 + 自身）的当前伪符号配置。第 $i$ 次迭代输出为：
      $\mathbf{y}_{\mathcal{S},c}^{(i)} = f_\theta(U_\mathcal{S}(c), \nabla U_\mathcal{S}(c), \sigma(\mathbf{y}_{\mathcal{S},N_c}^{(i-1)}))$
      其中 $\mathbf{y}^{(0)} = [0,0,...,0]$（全零表示无先验信息），$\sigma$ 为 sigmoid 激活。
    - 设计动机：单遍方法只看单个体素内的 UDF 信息，在高噪声区域容易误判。迭代允许正确重建的区域将信息"传播"到邻近的模糊区域。比如，如果一个体素的 UDF 没有达到零但邻居已确认有表面经过，网络可以做出更 informed 的判断。

2. **随机化迭代次数训练**:

    - 功能：训练时随机选择 1-6 次迭代，在每次迭代上都计算交叉熵损失。
    - 核心思路：损失函数 $\mathcal{L}_\theta = \sum_{i=1}^{r} \sum_{c \in \mathcal{S}} CE(softmax(\mathbf{y}_{\mathcal{S},c}^{(i)}), GT_\mathcal{S}(c))$，$r$ 随机采样自 $[1,6]$。
    - 设计动机：如果只用固定迭代次数训练，网络可能学会依赖特定轮数的模式而非鲁棒的传播策略。随机化强制网络在任意迭代深度都能给出合理预测，同时在多轮迭代上叠加损失确保了长链梯度传播的稳定性。

3. **噪声增强与高效过滤**:

    - 功能：增强对 UDF 噪声的鲁棒性，并加速推理。
    - 核心思路：训练时对 UDF 值和梯度加乘性高斯噪声 $U(c) \leftarrow U(c) \cdot (1 + \mathcal{N}(0, \sigma_\mathcal{N}))$。推理时，首先过滤掉 UDF 值 ≥ 截断阈值的体素（~85%），后续迭代中进一步过滤高置信度（>0.999）的体素。
    - 设计动机：噪声增强确保网络不会对精确 UDF 值过拟合。过滤策略将计算成本降到与单遍方法同一量级（$256^3$ 分辨率从 7 分钟降到 30 秒），且不损失精度。

### 损失函数 / 训练策略
使用交叉熵损失在所有迭代的所有体素上求和。UDF 值经过分辨率归一化（除以体素大小），使模型对不同分辨率具有泛化能力。关键发现：sigmoid 激活和随机迭代次数对收敛至关重要。

## 实验关键数据

### 主实验 — Marching Cubes 方法对比（自动解码器 UDF，分辨率 512）

| 数据集 | 方法 | CD(×10⁻⁵)↓ | F1↑ | IC↑ |
|--------|------|-------------|-----|-----|
| ShapeNet cars | MeshUDF | 82.7 | 57.0 | 81.7 |
| ShapeNet cars | NSD-UDF+MC | 56.9 | 58.8 | 83.8 |
| ShapeNet cars | **Ours+MC** | **8.84** | **65.6** | **88.9** |
| ShapeNet chairs | MeshUDF | 378 | 61.5 | 65.7 |
| ShapeNet chairs | NSD-UDF+MC | 295 | 64.7 | 75.7 |
| ShapeNet chairs | **Ours+MC** | **8.76** | **74.5** | **87.2** |
| ShapeNet planes | MeshUDF | 12.6 | 88.1 | 84.6 |
| ShapeNet planes | NSD-UDF+MC | 10.0 | 89.4 | 85.1 |
| ShapeNet planes | **Ours+MC** | **2.37** | **90.9** | **87.1** |

在 512 分辨率下，本文方法在所有数据集上均大幅领先。ShapeNet cars 上 CD 从 NSD-UDF 的 56.9 降到 8.84（6.4 倍改善），chairs 上从 295 降到 8.76（33.7 倍改善）。

### 不同分辨率的表现趋势

| 分辨率 | NSD-UDF+MC (cars CD) | Ours+MC (cars CD) | 说明 |
|--------|---------------------|-------------------|------|
| 128 | 6.79 | 5.64 | 低分辨率差距较小 |
| 256 | 10.2 | 5.23 | 差距开始拉大 |
| 512 | 56.9 | 8.84 | 高分辨率下其他方法严重退化，本文保持稳定 |

### 关键发现
- **分辨率越高，本文方法的优势越明显**：其他方法在高分辨率下性能反而下降（因为噪声影响加剧），而迭代方法保持稳定甚至继续改善。这正是核心卖点。
- 过滤策略将 $256^3$ 的推理时间从 7 分钟降到 30 秒，$512^3$ 从 1 小时降到 2.5 分钟，与基线方法的计算成本相当。
- 迭代通常在 6 轮内收敛，每轮只需处理越来越少的不确定体素。

## 亮点与洞察
- **反直觉洞察的精准把握**：高分辨率 UDF 网格化比低分辨率更难这个观察非常关键，且对方法论设计有直接指导意义。作者将这个问题类比为需要更大感受野的信息传播任务，解决方案自然而然。
- **迭代机制的优雅设计**：输入上一轮的伪符号配置作为"先验知识"，概念简洁但效果显著。类似于图像分割中的 CRF 后处理，但这里是端到端可训练的。
- **跨任务迁移潜力**：这种"从邻域已解决区域传播信息到未解决区域"的思路可以迁移到任何局部决策需要全局上下文的网格处理任务。

## 局限与展望
- 方法仅改进了网格化步骤，不直接提升 UDF 本身的精度。如果 UDF 在大范围内完全错误（而非局部噪声），迭代传播可能也无能为力。
- 迭代次数（最多 6 轮）是经验性设定的，对于特别复杂的拓扑可能需要更多轮次。
- 训练需要 ground truth SDF 来生成伪符号标签，限制了在野外数据上的直接应用。
- 每次迭代仍然是逐体素独立推理，只通过输入编码邻域信息。更强的方式可能是引入图网络在体素间直接传递消息。

## 相关工作与启发
- **vs MeshUDF**: MeshUDF 也尝试考虑邻域（通过定义体素探索顺序），但只是单遍启发式方法，主要针对简单形状如服装。本文的多遍迭代更系统化。
- **vs NSD-UDF**: NSD-UDF 用神经网络从局部 UDF 和梯度预测伪符号，本文在此基础上引入迭代和邻域信息，形成了严格的改进。
- **vs DCUDF**: DCUDF 虽有优化精炼步骤但网格提取仍是单遍的，且需要大量手动调参和切割操作。本文方法更端到端且鲁棒。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将迭代精炼引入 UDF 网格化，解决了高分辨率下的核心痛点
- 实验充分度: ⭐⭐⭐⭐⭐ 5 类数据集、4 种 UDF 架构、MC 和 DC 两种三角化方法的全面评估
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，但公式符号偶有混淆
- 价值: ⭐⭐⭐⭐ 对 UDF 表面重建有实际推动作用，分辨率提升有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[CVPR 2025\] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](../../CVPR2025/3d_vision/a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)
- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](../../CVPR2025/3d_vision/exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2025\] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](../../CVPR2025/3d_vision/event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)
- [\[CVPR 2025\] MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation](../../CVPR2025/3d_vision/mar-3d_progressive_masked_auto-regressor_for_high-resolution_3d_generation.md)

</div>

<!-- RELATED:END -->
