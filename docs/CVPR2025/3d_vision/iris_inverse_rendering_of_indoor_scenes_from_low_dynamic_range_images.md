---
title: >-
  [论文解读] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images
description: >-
  [CVPR 2025][3D视觉][逆渲染] IRIS提出了一个从多视角LDR图像中联合恢复HDR光照、物理材质和相机响应函数的逆渲染框架，通过显式建模色调映射、自动检测发光体和迭代优化策略，在真实和合成室内场景上实现了高质量的材质估计、重光照和虚拟物体插入。 1. 领域现状：基于物理的逆渲染旨在从图像中分解出几何、材质和光…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "逆渲染"
  - "室内场景"
  - "低动态范围"
  - "HDR恢复"
  - "相机响应函数"
---

# IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images

**会议**: CVPR 2025  
**arXiv**: [2401.12977](https://arxiv.org/abs/2401.12977)  
**代码**: [https://irisldr.github.io/](https://irisldr.github.io/)  
**领域**: 3D视觉  
**关键词**: 逆渲染, 室内场景, 低动态范围, HDR恢复, 相机响应函数

## 一句话总结
IRIS提出了一个从多视角LDR图像中联合恢复HDR光照、物理材质和相机响应函数的逆渲染框架，通过显式建模色调映射、自动检测发光体和迭代优化策略，在真实和合成室内场景上实现了高质量的材质估计、重光照和虚拟物体插入。

## 研究背景与动机

1. **领域现状**：基于物理的逆渲染旨在从图像中分解出几何、材质和光照，支持重光照、材质编辑和物体插入等应用。当前主流方法（如FIPT、NeILF++）通常依赖HDR图像作为输入，以捕捉完整的光传输信息。

2. **现有痛点**：HDR捕获需要专用硬件或多曝光合成，对普通用户不友好。现有尝试使用LDR输入的方法要么假设无穷远光源（不适用于室内场景），要么需要额外的发光体mask输入，要么忽略多次弹射光传输，难以处理室内场景的复杂空间变化光照。

3. **核心矛盾**：LDR图像经过动态范围裁剪和非线性CRF映射后，关键的高动态范围光照信息（如窗户、灯光的真实亮度）被不可逆地损失了。如果不恢复HDR光照，材质估计就会受到严重影响。

4. **本文目标** 从"随手拍"的LDR照片中，同时恢复HDR空间变化光照、物理材质（BRDF）和相机响应函数（CRF），使逆渲染技术对普通用户可及。

5. **切入角度**：利用基于物理的渲染方程建立LDR观测与HDR场景参数之间的桥梁——通过可微路径追踪和CRF建模，在优化过程中自然恢复过曝区域的HDR强度。

6. **核心 idea**：显式建模LDR成像管线（裁剪+CRF），通过交替优化HDR发射、材质和CRF来解耦逆渲染中光照-材质-CRF的模糊性。

## 方法详解

### 整体框架
IRIS接受多视角位姿LDR图像和重建的表面mesh作为输入，通过两阶段流程工作：初始化阶段（BRDF初始化+表面光场提取+发光体检测）和迭代优化阶段（HDR辐射恢复→shading烘焙→BRDF&CRF联合优化，循环直到收敛）。输出为空间变化的HDR光照、Cook-Torrance BRDF参数和CRF曲线。

### 关键设计

1. **CRF建模与HDR发射恢复**:

    - 功能：从LDR图像中恢复相机响应函数和高动态范围的发光体辐射
    - 核心思路：使用EMoR模型参数化CRF：$\mathbf{g} = \bar{\mathbf{g}} + \Sigma_b w_b \mathbf{g}_b$，其中 $\bar{\mathbf{g}}$ 是201个真实CRF的均值曲线，$\mathbf{g}_b$ 是PCA基。发光体通过多视角饱和度检测：如果某表面点在所有可见视角的平均LDR值≥0.99则标记为发光体。HDR辐射通过可微路径追踪最小化光度损失恢复：$\min_{L_e} \sum_i \|CRF(\min(L_o \cdot \Delta t_i, 1)) - I_i\|_2$
    - 设计动机：CRF是LDR-HDR转换的关键但通常未知。EMoR模型基于真实CRF数据库，参数空间小且可正则化，比MLP参数化更稳定。多视角饱和度检测可靠区分发光体和反射面

2. **分解光传输与shading烘焙**:

    - 功能：高效计算全局光照效果，避免在材质优化中进行昂贵的路径追踪
    - 核心思路：将光传输分解为漫反射shading $L_d$、粗糙镜面shading $L_s^0$ 和光滑镜面shading $L_s^1$，预计算并存储。射线追踪时，如果二次射线命中发光体则取可学习HDR辐射 $L_e$，否则取预计算的表面光场 $L_{SLF}$ 近似全局光照。这种分解使得BRDF优化可以高效进行
    - 设计动机：直接在优化中做多次弹射路径追踪计算量大且不稳定，分解后可以交替更新shading和BRDF。表面光场用于近似间接光照，避免递归光传输

3. **交替优化策略**:

    - 功能：解决HDR光照、材质和CRF联合估计中的模糊性
    - 核心思路：迭代执行三个步骤：(1) 固定BRDF和CRF，通过可微渲染优化发光体HDR辐射 $L_e$；(2) 用更新的HDR光照烘焙新的shading maps；(3) 固定shading，联合优化BRDF参数和CRF系数。优化目标：$\min_{a,m,\sigma,g} \mathcal{L}_{photo} + \lambda_a\mathcal{L}_{albedo} + \lambda_c\mathcal{L}_{CRF} + \lambda_m\mathcal{L}_{mat}$
    - 设计动机：光照、材质、CRF三者高度耦合，直接联合优化不稳定。交替优化每次只处理部分变量，逐步提升各组件质量

### 损失函数 / 训练策略
- 光度损失 $\mathcal{L}_{photo}$：渲染结果经CRF映射后与LDR观测比较
- Albedo正则 $\mathcal{L}_{albedo}$：使用单目albedo估计方法提供初始化和正则，采用尺度不变损失
- CRF正则 $\mathcal{L}_{CRF}$：L2正则PCA系数+单调性约束
- 材质正则 $\mathcal{L}_{mat}$：同语义实例内的roughness和metallic一致性约束
- 几何重建使用BakedSDF，法线通过off-the-shelf估计方法正则化

## 实验关键数据

### 主实验

| 数据集 | 指标 | IRIS (LDR) | FIPT* (LDR) | NeILF (LDR) | FIPT (HDR) |
|--------|------|------------|-------------|-------------|------------|
| FIPT合成 | $k_d$ PSNR↑ | **22.33** | 15.49 | 16.85 | 29.95 |
| FIPT合成 | $a'$ PSNR↑ | **17.92** | 09.74 | 14.02 | 25.98 |
| FIPT合成 | $\sigma$ PSNR↑ | **21.38** | 04.99 | 16.96 | 26.37 |
| FIPT合成 | $L_e$ IoU↑ | **0.69** | 0.69 | 0.35 | 0.86 |
| FIPT合成 | $L_e$ L2↓ | **0.12** | 0.28 | 2.29 | 0.03 |

### 消融实验
IRIS的定性评估主要通过可视化对比：
- NeILF因单次弹射路径追踪，无法去除漫反射中的阴影，roughness估计将墙面误判为比镜面更光滑
- FIPT*（提供发光体mask的LDR修改版FIPT）因缺少HDR信息，材质估计严重失准，倾向于低估roughness导致不真实的反射
- IRIS产生近乎无阴影的漫反射场，正确识别镜面为低roughness区域，并成功恢复窗户和天花灯的HDR光照

### 关键发现
- LDR→HDR光照恢复是IRIS的核心优势：恢复的HDR光照使光传输更准确，大幅提升材质估计
- 即使用LDR输入，IRIS的材质估计质量也大幅超越其他LDR方法（$k_d$ PSNR提升约6dB）
- 交替优化策略有效解耦了光照-材质-CRF的模糊性
- CRF估计使得IRIS可以处理不同曝光水平的输入图像
- 相比HDR输入的FIPT仍有差距，说明HDR信息的固有价值

## 亮点与洞察
- **LDR成为逆渲染输入**：显式建模CRF和色调映射，使得手机拍照也能做逆渲染。这显著降低了逆渲染的使用门槛，对实际应用意义重大
- **发光体自动检测**：多视角饱和度一致性检测简单有效，能区分发光体和反光面（某视角饱和≠发光体，所有视角都饱和才是）。这个策略可以迁移到任何需要区分自发光和反射的场景
- **EMoR参数化CRF**：基于真实CRF数据库的PCA参数化，比MLP或简单gamma更物理合理且易正则化

## 局限与展望
- 假设主要光源在输入图像中可见，无法处理完全隐藏的光源
- 几何质量依赖BakedSDF的重建精度
- 目前假设所有输入图像使用相同CRF（即同一相机），不支持多相机混合输入
- 定量评估仅在合成场景上进行（因真实场景无GT），真实场景仅有定性结果
- 可改进方向：引入基于扩散模型的HDR先验，或结合3DGS实现更高效的渲染

## 相关工作与启发
- **vs FIPT**: FIPT需要HDR输入且无法处理LDR的CRF问题。IRIS通过CRF建模和HDR恢复将FIPT的能力扩展到LDR领域
- **vs NeILF/NeILF++**: NeILF用神经光场表示光照但只做单次弹射路径追踪，NeILF++用可学习gamma但过于简单。IRIS通过分解光传输+EMoR CRF实现更准确的HDR恢复
- **vs Li et al.**: Li等人从单图估计BRDF和光照（数据驱动），受限于训练数据的domain gap。IRIS基于优化，在真实场景上更鲁棒

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统解决LDR输入的室内逆渲染问题，CRF+HDR恢复+交替优化的组合策略新颖实用
- 实验充分度: ⭐⭐⭐⭐ 合成+真实场景评估，重光照和物体插入demo说服力强，但真实场景缺定量指标
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，成像管线的建模逻辑严密，图示质量高
- 价值: ⭐⭐⭐⭐ 实用价值高，真正让逆渲染从实验室走向消费级设备

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)
- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](../../CVPR2026/3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_extended_dr_3d.md)
- [\[CVPR 2025\] MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)

</div>

<!-- RELATED:END -->
