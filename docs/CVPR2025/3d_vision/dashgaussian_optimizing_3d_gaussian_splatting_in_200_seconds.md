---
title: >-
  [论文解读] DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds
description: >-
  [CVPR 2025][3D视觉][3D高斯] 提出 DashGaussian，一种基于频率分析的渲染分辨率和高斯基元数量联合调度方案，将3DGS优化从逐步拟合高频分量的角度进行重新表述，平均加速 45.7% 且不牺牲渲染质量。 - 3DGS 虽然比 NeRF 快很多（数十分钟 vs 数天），但在算力受限设备和大规模场景重建…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯"
  - "训练加速"
  - "频率调度"
  - "分辨率调度"
  - "自适应密度化"
---

# DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds

**会议**: CVPR 2025  
**arXiv**: [2503.18402](https://arxiv.org/abs/2503.18402)  
**代码**: [dashgaussian.github.io](https://dashgaussian.github.io)  
**领域**: 3D视觉 / 3DGS加速  
**关键词**: 3D高斯, 训练加速, 频率调度, 分辨率调度, 自适应密度化

## 一句话总结

提出 DashGaussian，一种基于频率分析的渲染分辨率和高斯基元数量联合调度方案，将3DGS优化从逐步拟合高频分量的角度进行重新表述，平均加速 45.7% 且不牺牲渲染质量。

## 研究背景与动机

- 3DGS 虽然比 NeRF 快很多（数十分钟 vs 数天），但在算力受限设备和大规模场景重建中仍需进一步加速
- 现有加速方法分两类：(1) 工程优化（如高效前向/后向实现）；(2) 算法优化（如裁剪冗余高斯），后者往往损失渲染质量
- 三个关键观察：a) 计算开销主要由渲染分辨率和高斯数量决定；b) 优化初期高斯稀疏，渲染高分辨率图像是浪费；c) 优化后期高斯数量激增带来大量计算但质量提升有限
- 核心洞察：合理分配计算资源比简单裁剪参数更高效

## 方法详解

### 整体框架

将3DGS优化重新表述为渐进式拟合训练视图中更高层频率分量的过程。基于此提出分辨率调度器和基元数量调度器，联合控制优化复杂度。

### 关键设计

1. **频率引导的分辨率调度器**:
    - 功能：自适应地在优化过程中逐步提升渲染分辨率
    - 核心思路：将图像降采样视为去除高频分量，定义分辨率显著性函数 $\mathcal{X}(\mathbf{F}) = \frac{1}{N}\sum_{n=1}^N \sum_{i,j} \|\mathbf{F}^n(i,j)\|_2$；通过频率能量比 $f(\mathbf{F}, \mathbf{F}_r) = \frac{\mathcal{X}(\mathbf{F}) - \mathcal{X}(\mathbf{F}_r)}{\mathcal{X}(\mathbf{F})}$ 分配高/低分辨率的迭代步数
    - 设计动机：低频分量包含主要结构信息且计算量小，应先拟合；高频细节需要更多基元才能有效拟合，应在基元充足后进行
    - 切换时机：第 $s_r = S \cdot \mathcal{X}(\mathbf{F}_r) / \mathcal{X}(\mathbf{F})$ 次迭代从低分辨率切换到高分辨率

2. **分辨率引导的基元调度器**:
    - 功能：控制高斯基元数量随分辨率同步增长
    - 核心思路：$P_i = P_{\text{init}} + (P_{\text{fin}} - P_{\text{init}}) / (r^{(i)})^{2-i/S}$，power factor 从 2 线性降至 1，产生凹形增长曲线（前期抑制、中期鼓励增长）
    - 设计动机：特定分辨率应匹配适当的基元数量——低分辨率阶段过多基元是浪费，且可能导致过密集化；凹形曲线使基元在需要时才快速增长

3. **基于动量的基元预算估计**:
    - 功能：自适应确定最终基元数量 $P_{\text{fin}}$，无需依赖数据集先验
    - 核心思路：将 $P_{\text{fin}}$ 视为动量，每步自然 densify 的数量 $P_{\text{add}}$ 视为力：$P_{\text{fin}}^{(i)} = \max(P_{\text{fin}}^{(i-1)}, \gamma P_{\text{fin}}^{(i-1)} + \eta P_{\text{add}}^{(i)})$
    - 设计动机：避免人为设定上界，根据场景实际需求动态调整基元预算

### 损失函数 / 训练策略

- 使用反锯齿降采样（频域中心裁剪 + DFT/IDFT）避免2D aliasing
- 分辨率因子取 floor 离散化，鼓励基元比连续调度下更快增长
- 超参数：$a=4$（最大降采样对应频率能量的 $1/4$），$\gamma=0.98$，$\eta=1$，$P_{\text{fin}}^{(0)} = 5 \cdot P_{\text{init}}$
- 插入各种3DGS骨干时保持原有超参数不变

## 实验关键数据

### 主实验（与快速优化方法比较）

| 方法 | Mip-NeRF360 PSNR | 时间(min) | Deep Blending PSNR | 时间(min) | T&T PSNR | 时间(min) |
|------|-----------------|-----------|-------------------|-----------|----------|-----------|
| 3DGS | 27.72 | 18.31 | 29.50 | 17.27 | 23.62 | 10.59 |
| Reduced-3DGS | 27.28 | 15.52 | 29.78 | 13.74 | 23.59 | 8.31 |
| Taming-3DGS | 27.61 | 5.51 | 29.69 | 4.52 | 23.62 | 4.02 |
| **DashGaussian** | **27.92** | **3.23** | **30.02** | **2.20** | **23.97** | **2.62** |

### 加速各骨干的效果

| 骨干 | 原始时间 | +DashGaussian | 加速比 | PSNR变化 |
|------|---------|---------------|--------|---------|
| 3DGS | 18.31 min | 10.16 min | 44.5% | +0.09 |
| Mip-Splatting | 25.83 min | 12.60 min | 51.2% | +0.08 |
| Revising-3DGS | 5.73 min | 3.46 min | 39.6% | +0.20 |
| Taming-3DGS | 5.51 min | 3.23 min | 41.4% | +0.31 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅分辨率调度 | 加速显著但质量略降 | 缺少基元配合导致欠拟合 |
| 仅基元调度 | 加速一般 | 高分辨率渲染仍是瓶颈 |
| 联合调度（完整） | **加速最大且质量保持** | 分辨率-基元最佳配合 |
| 固定 $P_{\text{fin}}$ | 部分场景欠/过密 | 动量估计自适应更优 |

### 关键发现

- 平均加速 45.7%，且 PSNR 普遍提升（不是牺牲质量换速度）
- 可无缝插入任意3DGS骨干作为即插即用加速器
- 凹形基元增长曲线（前期少、后期多）优于现有凹形或前半最大的策略
- 在 Mip-NeRF 360 数据集上将 Taming-3DGS 的优化时间压缩到约 3 分钟（约200秒级别）

## 亮点与洞察

- 从频率分析角度统一了分辨率调度的理论基础，将3DGS优化重新表述为渐进式频率拟合
- "不砍参数，而是合理分配计算资源"的思路比硬裁剪更优雅，且不损失质量
- 动量式基元预算是真正自适应的方案，摆脱了人为设定上界的局限
- 实验证明合理的计算资源分配甚至能提升质量（低分辨率阶段相当于隐式正则化）

## 局限与展望

- 离散化分辨率因子（取 floor）是工程近似，连续调度的理论最优性未完全实现
- 频率能量比 $f$ 假设反锯齿降采样是完美的，实际中可能存在残余混叠
- 超参数 $a=4$ 的选择未充分消融
- 未在超大规模场景（如城市级）上充分验证

## 相关工作与启发

- 3DGS → 基础表示；Mip-Splatting → 多尺度渲染与抗锯齿
- Taming-3DGS → 工程加速基线；Mini-Splatting → 算法裁剪基线
- Nyquist 采样定理 + DFT → 分辨率调度的理论基础

## 评分

- 新颖性: ⭐⭐⭐⭐ 频率引导的分辨率调度视角新颖，联合调度方案设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集，四种骨干，充分的消融和对比
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，频率分析角度具有教育意义
- 价值: ⭐⭐⭐⭐⭐ 即插即用的通用加速器，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](../../CVPR2026/3d_vision/fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation](../../AAAI2026/3d_vision/opt3dgs_optimizing_3d_gaussian_splatting_with_adaptive_exploration_and_curvature.md)
- [\[CVPR 2025\] 3D-HGS: 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
