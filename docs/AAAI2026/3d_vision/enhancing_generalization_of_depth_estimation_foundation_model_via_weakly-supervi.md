---
title: >-
  [论文解读] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization
description: >-
  [AAAI 2026][3D视觉][单目深度估计] 提出 WeSTAR 框架，通过语义感知的分层归一化自训练 + 稀疏成对序数弱监督 + LoRA 权重正则化三者协同，以参数高效的方式提升深度估计基础模型（Depth Anything V2）在未见域和损坏数据上的泛化能力，在多个 OOD 基准上达到 SOTA。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "单目深度估计"
  - "域自适应"
  - "弱监督"
  - "LoRA"
  - "自训练"
---

# Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization

**会议**: AAAI 2026  
**arXiv**: [2511.14238](https://arxiv.org/abs/2511.14238)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 单目深度估计, 域自适应, 弱监督, LoRA, 自训练

## 一句话总结
提出 WeSTAR 框架，通过语义感知的分层归一化自训练 + 稀疏成对序数弱监督 + LoRA 权重正则化三者协同，以参数高效的方式提升深度估计基础模型（Depth Anything V2）在未见域和损坏数据上的泛化能力，在多个 OOD 基准上达到 SOTA。

## 研究背景与动机

**领域现状**：Depth Anything 系列等基础模型通过大规模数据训练实现了优秀的零样本单目深度估计泛化能力。但在面对下游任务的分布偏移时（如恶劣天气、传感器噪声、域外场景），性能仍有提升空间。

**现有痛点**：
   - 自训练（Self-Training）在回归任务上面临确认偏差——伪标签不准确时会强化模型错误
   - 当基线模型已经很强时，自训练仅能带来边际增益
   - 激进的适应过程可能导致灾难性遗忘，破坏预训练的泛化知识
   - 全量微调计算代价高且容易过拟合

**核心矛盾**：如何在利用下游数据提升性能的同时，保持模型的泛化能力不被破坏？

**本文目标** 设计一个参数高效且鲁棒的自适应框架，在少量目标域数据（无标注或弱标注）下增强深度基础模型的泛化性。

**切入角度**：三管齐下——自训练提供密集结构监督 + 弱监督提供稀疏但独立的序数约束打破确认偏差 + 权重正则化锚定预训练知识防遗忘。

**核心 idea**：用语义感知的分层归一化自训练 + 低成本成对序数弱监督 + LoRA 正则化三者协同，安全地适应深度基础模型到新域。

## 方法详解

### 整体框架
输入为少量目标域 RGB 图像（无标注或仅有稀疏成对深度序数标注），输出为适应后的深度估计模型。采用 teacher-student 架构：teacher用 EMA 更新，student 通过 LoRA 适配器微调。弱增强图像送入 teacher 生成伪标签，强增强图像送入 student 预测深度。

### 关键设计

1. **语义感知分层深度归一化（SA-HDN）**:

    - 功能：解决自训练中 teacher 伪标签和 student 预测之间的尺度/偏移歧义
    - 核心思路：传统分层归一化（HDN）用固定网格划分图像区域做归一化，但忽略语义信息，可能把同一物体切割。本文用 SAM2 分割模型自动生成实例 mask，构建两级层次：全局上下文 $\mathcal{C}_{global}$（所有像素） + 实例上下文 $\mathcal{C}_{ins}^k$（第 $k$ 个物体的像素）
    - 归一化公式：$\Phi(d_p, \mathcal{C}_p) = \frac{d_p - t(\mathcal{C}_p)}{s(\mathcal{C}_p) + \epsilon}$，其中 $t$ 和 $s$ 分别是中位数和 MAD
    - 设计动机：语义感知的划分确保归一化统计量在物体级别计算，避免跨物体的深度不连续性干扰

2. **弱监督适应（Weakly Supervised Adaptation）**:

    - 功能：用最低成本的成对序数深度标注打破自训练的确认偏差
    - 核心思路：每个弱标签 $w_j = \{p_{jn}^+, p_{jn}^-, l_{jn}\}$ 表示两个像素之间的深度序数关系（更远/相等/更近）。用 margin ranking loss 强制模型的预测满足这些约束
    - 采样策略：每张图像进行 5 次结构化采样，每次先选锚点、再选更远和更近的点，形成满足传递性的成对约束
    - 设计动机：稀疏但独立于模型的标注提供了额外的监督信号，能纠正伪标签无法发现的局部拓扑错误

3. **LoRA 权重正则化**:

    - 功能：约束模型更新幅度，防止过拟合和灾难性遗忘
    - 核心思路：在 encoder 的注意力层注入低秩适配器 $\Theta_a + UV$，仅更新 $U, V$。额外添加正则化损失 $\mathcal{L}_{reg} = \sum \|\frac{\alpha}{r} U_{tk} V_{tk}\|_2^2$ 惩罚大幅偏离初始化
    - 设计动机：LoRA 本身限制了参数空间，但在严重域偏移下仍可能受确认偏差影响。权重正则化确保只有当目标域新证据足够强时才更新参数

### 损失函数 / 训练策略
总损失：$\mathcal{L} = \lambda_{st} \mathcal{L}_{st} + \lambda_w \mathcal{L}_{weak} + \lambda_r \mathcal{L}_{reg}$

权重设置：$\lambda_{st}=1.0, \lambda_w=0.001, \lambda_r=1.0$。使用 AdamW 优化器，余弦退火学习率调度，EMA 衰减因子 0.996，LoRA rank=8, alpha=16。单卡 RTX 3090，batch size=4。

## 实验关键数据

### 主实验
在 9 个未见的真实数据集上评估（NYU, KITTI, Sintel, DIODE, NuScenes, DrivingStereo 等）：

| 方法 | NYU δ₁↑ | KITTI δ₁↑ | Sintel δ₁↑ | NuScenes δ₁↑ | D-Rainy δ₁↑ |
|------|---------|----------|-----------|-------------|-------------|
| Source (零样本) | 97.7 | 93.4 | 74.8 | 74.4 | 84.8 |
| TTAC | 97.7 | 93.4 | 75.0 | 74.4 | 84.5 |
| SGRL | 97.6 | 94.1 | 76.9 | 75.8 | 85.3 |
| **WeSTAR** | **98.2** | **95.1** | **82.2** | **78.1** | **87.4** |

WeSTAR 在所有数据集上均达到最优，Sintel 上 δ₁ 提升 7.4%（74.8→82.2）。

### 消融实验（损坏数据集）

| 方法 | NYU-C δ₁↑ | KITTI-C δ₁↑ | Sintel-C δ₁↑ |
|------|-----------|------------|-------------|
| Source | 87.4 | 83.2 | 60.3 |
| iBOT* | 92.1 | 85.6 | 62.7 |
| SGRL | 92.4 | 87.4 | 66.5 |
| **WeSTAR** | **94.6** | **88.7** | **71.8** |

### 关键发现
- 三个组件协同效果显著：自训练提供全局结构对齐，弱监督纠正局部拓扑错误，正则化防止遗忘
- 在损坏数据上优势更大——NYU-C 上 δ₁ 从 87.4 提升到 94.6（+7.2%）
- SA-HDN 比传统 HDN 显著更好，语义感知的归一化避免了跨物体深度混淆
- 弱监督成本极低（每张图仅需 5 组成对比较），但带来的增益显著

## 亮点与洞察
- **三管齐下的协同设计**很巧妙：密集自训练+稀疏弱监督+正则化，三者各解决一个问题（结构对齐/拓扑纠正/知识保留），设计逻辑清晰。这种"多层防线"的思路可迁移到其他需要安全适应预训练模型的场景
- **用 SAM2 做语义分割来增强深度归一化**：跨任务借力的思路，SAM2 的通用分割能力很好地服务了深度估计的归一化需求
- **弱监督的成本-收益比极高**：仅需少量成对序数标注就能打破确认偏差，这对实际部署非常友好

## 局限与展望
- 弱标注仍需要人工标注成对深度关系，虽然成本低但无法完全自动化
- 仅在相对深度估计上验证，未测试绝对深度估计任务
- SAM2 的分割质量在极端损坏图像上可能下降，影响 SA-HDN 效果
- 实验中仅用 Depth Anything V2 和 MiDaS 两个骨干，更多基础模型的普适性有待验证

## 相关工作与启发
- **vs Depth Anything V2**: WeSTAR 以 DAv2 为基座，通过适应进一步提升其泛化能力，可作为 DAv2 的标准下游适应方案
- **vs TTT++**: TTT++ 基于对比学习做测试时适应，在部分损坏数据上反而性能下降；WeSTAR 通过正则化避免了这一问题
- **vs SGRL**: SGRL 仅用弱监督没有自训练，性能低于 WeSTAR，说明密集自训练的价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 三组件协同设计逻辑清晰，SA-HDN 是有意义的改进
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个数据集 + 损坏基准 + 多种基线比较，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 实用的深度基础模型适应方案，成本低效果好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation](../../CVPR2026/3d_vision/depth_any_panoramas_a_foundation_model_for_panoramic_depth_estimation.md)
- [\[CVPR 2026\] RoSAMDepth: Robust Self-supervised Depth Estimation Leveraging Segment Anything Model](../../CVPR2026/3d_vision/rosamdepth_robust_self-supervised_depth_estimation_leveraging_segment_anything_m.md)
- [\[ECCV 2024\] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](../../ECCV2024/3d_vision/improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [\[CVPR 2026\] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation](../../CVPR2026/3d_vision/rewis3d_reconstruction_improves_weaklysupervised_s.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
