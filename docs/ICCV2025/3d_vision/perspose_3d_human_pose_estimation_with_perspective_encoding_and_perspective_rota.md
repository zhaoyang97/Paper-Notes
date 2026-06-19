---
title: >-
  [论文解读] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation
description: >-
  [ICCV 2025][3D视觉][3D人体姿态估计] 提出PersPose框架，通过透视编码(PE)将裁剪后相机内参编码为2D映射、透视旋转(PR)将人体居中以消除透视畸变，解决了现有方法忽略FOV信息导致深度估计不准确的问题。 核心矛盾 核心矛盾：领域现状：现有3D HPE方法使用裁剪图像作为输入存在两个被忽视的问题：…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D人体姿态估计"
  - "透视编码"
  - "透视旋转"
  - "相机内参"
  - "单目"
---

# PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation

**会议**: ICCV 2025  
**arXiv**: [2508.17239](https://arxiv.org/abs/2508.17239)  
**代码**: [GitHub](https://github.com/KenAdamsJoseph/PersPose)  
**领域**: 3D视觉  
**关键词**: 3D人体姿态估计, 透视编码, 透视旋转, 相机内参, 单目

## 一句话总结

提出PersPose框架，通过透视编码(PE)将裁剪后相机内参编码为2D映射、透视旋转(PR)将人体居中以消除透视畸变，解决了现有方法忽略FOV信息导致深度估计不准确的问题。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：现有3D HPE方法使用裁剪图像作为输入存在两个被忽视的问题：

**裁剪丢失深度信息**：相同裁剪图像可能对应不同相对深度（图2示例：subjects a和b裁剪图像相同但深度不同）；不同裁剪图像可能有相同深度（subjects a和c）

**FOV对深度至关重要**：即使使用全图，缺少FOV信息也会导致错误深度估计（图3示例：两个不同FOV相机拍摄同一人获得视觉差异很大但深度标签相同的图像）

核心洞见：裁剪操作等价于修改相机内参。裁剪后内参 $K^{\text{crop}} = AK$ 包含了裁剪和FOV信息。

## 方法详解

### 透视编码 (Perspective Encoding)

将裁剪内参 $K^{\text{crop}}$ 编码为2D PE映射 $M^{xy}$：将每个像素坐标 $(u_i, v_i)$ 投影到 $z=1$ 平面：

$$(K^{\text{crop}})^{-1} \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

投影区域在 $z=1$ 平面上几何编码了唯一的视锥体。不同焦距对应不同大小的投影区域，偏轴的主点对应偏移的区域。

PE映射与裁剪图像分别通过不同卷积层后逐元素相加。

### 透视旋转 (Perspective Rotation)

人体可出现在图像任意位置，导致主点 $(c_x^{\text{crop}}, c_y^{\text{crop}})$ 变化大，增加拟合难度。PR通过旋转使人体居中：

1. 计算人体边界框中心在 $z=1$ 平面的投影 $(x_c, y_c, 1)$
2. 计算旋转轴和角度：
$$\mathbf{n} = \frac{(x_c, y_c, 1) \times (0,0,d)^\top}{\|(x_c, y_c, 1) \times (0,0,d)^\top\|}$$
$$\phi = \arccos\frac{(x_c, y_c, 1) \cdot (0,0,d)}{\|(x_c, y_c, 1)\| \cdot \|(0,0,d)\|}$$
3. 通过Rodrigues公式得旋转矩阵 $R$，透视变换矩阵 $M = KRK^{-1}$

PR后映射函数从4输入简化为2输入：
$$f_\theta: (I^{\text{crop}}, f^{\text{crop}}, c_x^{\text{crop}}, c_y^{\text{crop}}) \rightarrow P_{\text{XYZ}}$$
简化为：
$$\tilde{f}_\theta: (I^{\text{crop}}, f^{\text{crop}}) \rightarrow P_{\text{XYZ}}$$

### 推理流程

1. 原图经PR得到居中图像 $I'$，从中心裁剪
2. 计算 $K^{\text{crop}}$ 并编码为PE映射
3. 网络预测2D关节坐标+相对深度 $P_{\text{UVD}}$ 和尺度因子 $\hat{s}$
4. 结合内参转为旋转后3D姿态 $P'_{\text{XYZ}}$
5. 逆旋转 $P_{\text{XYZ}} = R^\top P'_{\text{XYZ}}$

## 实验

### 主实验 - 3DPW数据集

| 方法 | PA-MPJPE↓ | MPJPE↓ |
|------|-----------|--------|
| HMR | 81.3 | 130.0 |
| SPIN | 59.2 | 96.9 |
| CLIFF | 43.0 | 69.0 |
| 前SOTA | - | 65.0 |
| **PersPose** | **38.7** | **60.1** |

在野外数据集3DPW上MPJPE降低7.54%，达到SOTA。

### 多数据集对比

| 数据集 | 指标 | PersPose |
|--------|------|----------|
| 3DPW | MPJPE↓ | **60.1** |
| Human3.6M | MPJPE↓ | 竞争性 |
| MPI-INF-3DHP | PCK↑ | SOTA |

PersPose在多个基准上取得一致的SOTA或竞争性结果。

## 亮点与洞察

1. **洞察力强**：清晰论证了裁剪操作等价于修改内参，揭示了长期被忽视的问题
2. **解决方案优雅**：PE和PR模块各自简洁但针对性强
3. **物理原理清晰**：基于相机成像几何原理设计，而非黑盒方法
4. **即插即用**：PE和PR可嵌入任意现有HPE框架

## 局限与展望

- 需要已知或可获取的相机焦距信息
- PR操作需要额外的图像变换计算
- 对极端广角镜头的非线性畸变未建模
- 未探索视频序列中的时序一致性

## 相关工作

- CLIFF: 利用边界框信息矫正全局旋转
- SPEC: 从图像估计焦距
- Ray3D: 利用内参将2D关键点转为3D射线

## 评分

- 新颖性: ⭐⭐⭐⭐ (PE和PR的设计巧妙)
- 技术深度: ⭐⭐⭐⭐ (几何推导完整清晰)
- 实验充分度: ⭐⭐⭐⭐ (三个数据集+消融)
- 实用价值: ⭐⭐⭐⭐⭐ (实际HPE的核心改进)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2025\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](../../CVPR2025/3d_vision/pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)
- [\[CVPR 2026\] Affine Perspective-Three-Point Problem](../../CVPR2026/3d_vision/affine_perspective-three-point_problem.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](../../CVPR2025/3d_vision/extreme_rotation_estimation_in_the_wild.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)

</div>

<!-- RELATED:END -->
