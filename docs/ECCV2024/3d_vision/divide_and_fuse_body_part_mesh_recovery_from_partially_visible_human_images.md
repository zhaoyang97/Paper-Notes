---
title: >-
  [论文解读] Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images
description: >-
  [ECCV 2024][3D视觉][人体网格重建] 提出"分而治之"的自底向上人体网格重建方法，通过独立重建各身体部位后融合，有效解决人体大面积不可见时传统自顶向下方法（如SMPL）失效的问题。 领域现状：主流人体网格重建方法采用自顶向下设计，依赖SMPL等全身参数化模型，从图像中提取全局特征后回归全身参数…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "人体网格重建"
  - "部分可见"
  - "遮挡"
  - "参数化模型"
  - "自底向上"
---

# Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images

**会议**: ECCV 2024  
**arXiv**: [2407.09694](https://arxiv.org/abs/2407.09694)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 人体网格重建, 部分可见, 遮挡, 参数化模型, 自底向上

## 一句话总结

提出"分而治之"的自底向上人体网格重建方法，通过独立重建各身体部位后融合，有效解决人体大面积不可见时传统自顶向下方法（如SMPL）失效的问题。

## 研究背景与动机

**领域现状**：主流人体网格重建方法采用自顶向下设计，依赖SMPL等全身参数化模型，从图像中提取全局特征后回归全身参数。当人体大部分可见时效果良好。

**现有痛点**：当输入图像中人体大面积被遮挡或超出视野（如只露出双腿），这类方法性能严重下降。原因有二：(a) 网络难以从少量可见部位识别人体；(b) SMPL各部位参数相互耦合，不可见部位的信息缺失会干扰可见部位的重建。

**核心矛盾**：自顶向下模型的参数耦合性使得局部重建必须依赖全局信息，而部分可见场景下全局信息不可靠。

**本文目标** 在只有少数身体部位可见时，仍能准确重建可见部位的3D网格。

**切入角度**：如果逐部位独立重建，就能天然避免"信息捕获困难"和"部位间干扰"两个问题。

**核心 idea**：设计独立的身体部位参数化模型（HPPM），逐部位重建后融合，实现对部分可见人体的鲁棒重建。

## 方法详解

### 整体框架

输入单目部分可见人体图像 → Swin Transformer提取特征 → MLP回归各部位HPPM参数（形状、旋转、平移） → 各部位HPPM独立生成部位网格 → 融合模块连接相邻可见部位 → 输出可见部位的人体网格。

### 关键设计

#### 1. Human Part Parametric Models (HPPM)

- **功能**：为人体15个部位各自设计独立的参数化网格模型，无需部位间依赖即可重建单个部位。
- **核心思路**：
    - 从SMPL模板网格出发，根据混合权重矩阵 $W$ 将顶点分配到对应骨骼：$p_i = \arg\max_j W_{ij}$
    - 手动合并相邻近刚体部分（如肩颈与躯干），最终形成15个部位
    - 对每个部位进行膨胀操作，使相邻部位间存在**重叠区域**，方便后续融合
    - 利用PCA对大量SMPL GT数据做降维，训练线性映射矩阵 $\mathcal{U}_p \in \mathbb{R}^{k \times n}$，将 $k$ 维形状参数映射到部位网格
    - 每个部位参数维度可调（16~42维），根据拟合精度自适应设定，最大误差控制在2mm以内
- **设计动机**：与SMPL不同，HPPM各部位完全解耦，可通过简单的全局刚性变换+少量形状参数独立重建，无需显式姿态建模。

#### 2. Divide：部位独立重建网络

- **功能**：从图像特征回归各部位的HPPM参数，独立重建每个部位网格。
- **核心思路**：
    - Swin Transformer提取图像特征，MLP分别预测各部位的形状参数 $\hat{S}_p$、6D旋转 $\hat{R}_p$、平移 $\hat{T}_p$
    - 部位网格生成：$\hat{v}_p = \hat{M}_p(\mathcal{U}_p \hat{S}_p + \mathcal{M}_p)$，其中 $\hat{M}_p$ 为全局变换矩阵
    - 关节位置通过回归器计算：$\hat{J}_p = \mathcal{J}_p \hat{v}_p$
- **设计动机**：所有监督信号都定义在单个部位上，确保不可见部位不影响可见部位的重建。

#### 3. Fuse：相邻部位融合模块

- **功能**：当多个相邻部位同时可见时，将它们无缝连接为一个整体网格。
- **核心思路**：
    - 重叠区域的顶点采用基于拓扑距离的加权平均策略：$v_k^c = \frac{\hat{v}_{p_1 i} d_{2j} + \hat{v}_{p_2 i} d_{1i}}{d_{1i} + d_{2j}}$
    - 其中 $d_{1i}$ 为顶点到最近非重叠顶点的拓扑距离，实现渐变过渡
    - 非重叠区域直接使用对应部位的顶点
- **设计动机**：重叠区域设计+渐变加权避免了融合处的不自然变形。

### 损失函数 / 训练策略

**Divide阶段损失**：
$$\mathcal{L}_{div} = \lambda_v \mathcal{L}_v + \lambda_{j3d} \mathcal{L}_{j3d} + \lambda_{j2d} \mathcal{L}_{j2d} + \lambda_s \mathcal{L}_s + \lambda_r \mathcal{L}_r + \lambda_t \mathcal{L}_t$$

包含部位顶点损失、3D关节损失、2D投影关节损失、形状参数损失、旋转损失和平移损失，均按可见性掩码 $\delta_p$ 加权。

**Fuse阶段损失**：
$$\mathcal{L}_{fu} = \lambda_{ol} \mathcal{L}_{ol} + \lambda_{dc} \mathcal{L}_{dc}$$

- 重叠损失 $\mathcal{L}_{ol}$：约束相邻部位重叠顶点趋近其平均位置
- 深度一致性损失 $\mathcal{L}_{dc}$：约束同图中非直接相邻部位在z方向的一致性

**总损失**：$\mathcal{L} = \mathcal{L}_{div} + \mathcal{L}_{fu}$

**训练数据增强**：在训练时对公开数据集做类似的图像裁剪策略，模拟部分可见输入。评估使用自建的PV-Human3.6M和PV-3DPW基准。

## 实验关键数据

### 主实验

| 方法 | PV-H36M MPVE↓ | PV-H36M MPJPE↓ | PV-3DPW MPVE↓ | PV-3DPW MPJPE↓ |
|------|---------------|-----------------|---------------|-----------------|
| MotionBERT | 196.9 | 169.6 | 185.5 | 155.4 |
| SEFD | 276.0 | 198.9 | 241.1 | 203.0 |
| GLoT | 214.1 | 199.0 | 235.0 | 213.5 |
| CycleAdapt | 249.4 | 231.9 | 189.0 | 137.1 |
| **D&F (Ours)** | **63.3** | **55.9** | **109.9** | **102.7** |

（上述为微调后结果，直接测试本文也显著优于其他方法）

### 消融实验

| 配置 | PV-H36M MPVE/MPJPE | PV-3DPW MPVE/MPJPE | 说明 |
|------|---------------------|---------------------|------|
| w/o 2D投影损失 | 64.2/56.7 | 111.8/104.9 | 轻微下降 |
| w/o 3D关节损失 | 63.9/56.2 | 112.4/105.8 | 轻微下降 |
| w/o 3D顶点损失 | 68.4/63.5 | 120.2/108.3 | 明显下降 |
| w/o 形状参数损失 | 70.1/57.0 | 119.7/103.9 | 明显下降 |
| w/o 6D旋转损失 | 95.6/87.5 | 138.5/127.4 | **下降最大** |
| w/o 平移损失 | 75.1/69.9 | 123.0/115.3 | 显著下降 |
| w/o 重叠损失 | 74.5/64.2 | 125.2/111.8 | 显著下降 |
| 固定参数维度 | 67.5/61.7 | 114.0/105.4 | 可调维度更优 |
| **D&F完整** | **63.3/55.9** | **109.9/102.7** | — |

### 关键发现

- 6D旋转损失对性能影响最大，去掉后MPVE增加超50%
- 可调参数维度优于固定维度，总参数量360即可表达15个部位
- HPPM训练拟合误差平均仅1.11mm（顶点）和1.46mm（关节）
- 在5-10个部位可见时仍优于SOTA（MPVE 117.4 vs CycleAdapt 169.9）

## 亮点与洞察

- **范式创新**：首次提出自底向上的学习型人体网格重建方法，专门面向大面积不可见场景
- **HPPM设计精巧**：重叠区域设计一举两得——既包含关节变形信息，又简化融合
- **可调参数维度**：不同部位形状变化差异大（躯干>手脚），自适应维度分配更高效
- **大幅领先**：在部分可见场景下将MPVE从~200mm降至~60-110mm，提升巨大

## 局限与展望

- 当前仅重建可见部位，未利用部位间先验推断不可见部位
- 融合模块为后处理设计，未端到端联合优化
- 15个部位划分基于经验，可能对某些极端姿态不够精细
- 可考虑加入时序信息或扩散模型先验来补全不可见区域

## 相关工作与启发

- **vs SMPL系列方法**：SMPL参数耦合导致部分可见时全局回归失效，本文通过解耦部位彻底解决
- **vs 遮挡处理方法**（OCHMR等）：这些方法侧重从可见部位推断遮挡部位，对整体识别要求高；本文关注可见部位自身的精确重建
- **vs 自底向上姿态估计**：姿态估计中自底向上思路成熟，但从稀疏关键点推断完整网格是病态问题；本文的部位参数化模型填补了这一空白

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 自底向上人体网格重建的全新范式，HPPM设计独特
- 实验充分度: ⭐⭐⭐⭐ 主实验+消融+可调参数分析，自建基准合理；但缺少真实遮挡场景测试
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 对AR/VR/医疗等场景下部分可见人体重建有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[CVPR 2026\] MetricHMSR: Metric Human Mesh and Scene Recovery from Monocular Images](../../CVPR2026/3d_vision/metrichmsr_metric_human_mesh_and_scene_recovery_from_monocular_images.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](../../CVPR2025/3d_vision/prompthmr_promptable_human_mesh_recovery.md)
- [\[ECCV 2024\] CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images](canonicalfusion_generating_drivable_3d_human_avatars_from_multiple_images.md)

</div>

<!-- RELATED:END -->
