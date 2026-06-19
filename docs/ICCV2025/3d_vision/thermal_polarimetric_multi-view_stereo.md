---
title: >-
  [论文解读] Thermal Polarimetric Multi-view Stereo
description: >-
  [ICCV2025][3D视觉][thermal imaging] 提出利用热偏振（长波红外偏振）线索进行精细三维形状重建的方法，理论证明 LWIR 偏振观测不受光照环境和材质光学属性的影响，从而实现对透明、半透明和异质材料物体的高精度三维重建，显著优于可见光偏振方法。 核心矛盾 核心矛盾：三维形状重建是计算机视觉的基础问题…
tags:
  - "ICCV2025"
  - "3D视觉"
  - "thermal imaging"
  - "polarimetric imaging"
  - "LWIR"
  - "multi-view stereo"
  - "SDF"
  - "shape from polarization"
---

# Thermal Polarimetric Multi-view Stereo

**会议**: ICCV2025  
**arXiv**: [2510.20972](https://arxiv.org/abs/2510.20972)  
**代码**: 无  
**领域**: 三维重建 / 热成像 / 偏振成像  
**关键词**: thermal imaging, polarimetric imaging, LWIR, multi-view stereo, SDF, shape from polarization

## 一句话总结

提出利用热偏振（长波红外偏振）线索进行精细三维形状重建的方法，理论证明 LWIR 偏振观测不受光照环境和材质光学属性的影响，从而实现对透明、半透明和异质材料物体的高精度三维重建，显著优于可见光偏振方法。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：三维形状重建是计算机视觉的基础问题，但现有方法对光照和材质有强假设：

- **多视立体匹配**：依赖表面纹理进行对应点匹配
- **结构光/光度立体**：需要特定光照条件，假设不透明表面
- **可见光偏振（SfP）**：依赖光照条件，受镜面反射和漫反射偏振混合的歧义影响

对于透明物体（玻璃）、半透明物体（塑料）、低反射物体（黑色表面）等，这些方法均存在严重限制。

### 热成像的优势

热成像是一种有吸引力的替代方案：
- **不依赖环境光照**：任何有温度的物体都会发射长波红外（LWIR）光
- **材质不透明性**：绝大多数材料在 LWIR 光谱中是不透明的（除专门为热光学设计的材料）
- **自发光特性**：物体本身就是光源，无需外部照明

### 现有热成像三维重建的局限

- **热多视立体**：仍依赖纹理匹配
- **热光度立体**：需要主动加热/冷却物体，耗时且不实用
- **热 NeRF**：几何精度较低
- **吸收法深度估计**：精度有限

### 核心洞察：LWIR 偏振没有歧义

在可见光谱中，偏振观测是镜面反射、漫反射和透射的混合，各分量的相对强度随材质和光照复杂变化。而在 LWIR 光谱中，大多数物体不透明（透射可忽略），且常温环境中反射分量远小于发射分量，因此 LWIR 偏振观测简化为纯发射偏振，可用 Kirchhoff 定律和 Fresnel 方程解析表达，无歧义。

## 方法详解

### LWIR 偏振理论

#### 偏振物理基础

偏振状态用 Stokes 参数和 Mueller 矩阵描述。完整观测包含四个分量：镜面反射偏振、漫反射偏振、透射偏振、发射偏振。

#### 可见光 vs LWIR 偏振

**可见光**：观测 = 漫反射 + 镜面反射 + 透射，三个分量的相对强度随材质和光照复杂变化，分析困难。实验中可见光 AoLP 因材质不同出现严重不一致（黑石镜面反射污染、玻璃透射污染、半透明塑料散射干扰）。

**LWIR**：观测约等于纯发射偏振。发射偏振由黑体辐射的 Mueller 矩阵和物体温度决定，可用 Kirchhoff 定律（发射率 = 1 - 反射率）和 Fresnel 方程解析表达，无歧义。实验中 LWIR AoLP 在所有材质上保持高度一致。

#### 偏振到表面法线

- **线偏振度（DoLP）**：与天顶角相关，但依赖材质折射率
- **线偏振角（AoLP）**：直接等于表面法线的方位角，**与材质属性完全无关**

这是本文的核心优势：AoLP 提供了稳健的、与材质无关的方位角约束。

### 三维重建方法

#### 形状表示

使用隐式 SDF（由 8 层 MLP + softplus 激活 + 位置编码表示），零等值面定义物体表面。在 IDR（Implicit Differentiable Renderer）框架下优化。

#### 损失函数

总损失 = 切空间一致性损失（TSC Loss） + 轮廓损失 + Eikonal 正则化

**切空间一致性损失**（来自 MVAS 方法）：AoLP 定义了一个投影切向量，表面法线必须与之正交。多视角的切向量集合强约束了法线方向。关键区别：可见光 MVAS 存在法线方向的正负半周期歧义，需要修改 TSC 损失来处理；本文的 LWIR 方法无此歧义，可直接使用标准 TSC 损失。

**轮廓损失**：约束视觉外壳，使用交叉熵损失。

**Eikonal 正则化**：保证 SDF 梯度范数接近 1。

### 实验系统

- **热偏振相机**：FLIR Boson 320 + 15mm 镜头 + 旋转线栅偏振片（0/45/90/135 度采集）
- **可见光偏振相机**：FLIR Blackfly（用于位姿估计和基线对比）
- **标定**：Aruco 标记 + 白色铝板加热器的双相机标定
- **数据集**：7 个不同材质/形状的物体，20-30 视角，结构光扫描真值

## 实验结果

### 定量与定性结果

对比方法：可见光 MVAS、热 IDR

在所有测试物体上，本方法在 Chamfer 距离和平均角度误差上均优于两种对比方法。

关键定性发现：

- **陶瓷猫头鹰（涂层反光）**：可见光 MVAS 和热 IDR 丢失眼睛和喙等细节，本方法成功恢复
- **玻璃容器（含内部餐具）**：浮雕纹理在本方法中清晰可见，其他方法模糊
- **黑色杯子白色盖子**：可见光 AoLP 在盖子上噪声过大导致波浪状表面，LWIR 偏振可合理重建
- **透明花瓶和瓶子**：凹陷细节精确重建，其他方法呈现模糊
- **总体趋势**：可见光 MVAS 有波浪状伪影（AoLP 噪声）；热 IDR 有膨胀的表面（不考虑法线信息）
- 薄部件（瓶颈）因热相机空间分辨率低出现伪影

## 优势与局限

### 优势

- 理论严密：从 Stokes-Mueller 形式推导出 LWIR 偏振的无歧义特性
- 光照无关：完全独立于环境光照
- 材质无关：AoLP 与材质折射率无关，适用于透明、半透明、异质材料
- 无需加热/冷却过程：稳态测量即可
- 在所有测试物体上 Chamfer 距离和角度误差均最优

### 局限

- 当周围存在高温物体或物体温度低于环境时，发射分量主导假设不成立
- 金属表面或粗糙表面的偏振信号不稳定
- 热相机空间分辨率低，薄部件重建困难
- 需旋转偏振片采集四个方向，非单次拍摄
- 定制化采集系统限制了实际应用范围

## 个人思考

1. 理论贡献清晰：通过严格物理推导证明 LWIR 偏振没有可见光偏振的歧义问题，AoLP 直接给出方位角
2. 热偏振相机的发展（未来可能出现单次拍摄的 LWIR 偏振相机）将大大提高实用性
3. 与可见光方法的互补有很大潜力：可见光提供纹理和高分辨率，LWIR 提供材质无关的法线约束
4. 在工业检测（透明/反射物体）、安防（无需照明）等场景有直接应用价值
5. 可以考虑将 DoLP 也纳入约束（虽然依赖折射率，但可联合估计折射率和形状）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](../../CVPR2025/3d_vision/mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/3d_vision/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[CVPR 2025\] MUSt3R: Multi-view Network for Stereo 3D Reconstruction](../../CVPR2025/3d_vision/must3r_multi-view_network_for_stereo_3d_reconstruction.md)
- [\[CVPR 2026\] Thermal is Always Wild: Characterizing and Addressing Challenges in Thermal-Only Novel View Synthesis](../../CVPR2026/3d_vision/thermal_is_always_wild_characterizing_and_addressing_challenges_in_thermal-only_.md)
- [\[ICCV 2025\] Stereo Any Video: Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)

</div>

<!-- RELATED:END -->
