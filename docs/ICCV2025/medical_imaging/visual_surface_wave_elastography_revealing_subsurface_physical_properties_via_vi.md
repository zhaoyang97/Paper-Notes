---
title: >-
  [论文解读] Visual Surface Wave Elastography: Revealing Subsurface Physical Properties via Visible Surface Waves
description: >-
  [ICCV 2025][医学图像][弹性成像] 本文提出 VSWE（Visual Surface Wave Elastography），仅通过一段表面波传播的视频，提取色散关系并结合基于物理的有限元优化，推断介质的亚表面厚度和刚度参数，在模拟和真实明胶实验中均实现了高精度的参数恢复，为居家健康监测提供了概念验证。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "弹性成像"
  - "表面波"
  - "色散关系"
  - "有限元"
  - "视频分析"
---

# Visual Surface Wave Elastography: Revealing Subsurface Physical Properties via Visible Surface Waves

**会议**: ICCV 2025  
**arXiv**: [2507.09207](https://arxiv.org/abs/2507.09207)  
**代码**: 即将公开  
**领域**: 医学图像 / 材料表征  
**关键词**: 弹性成像, 表面波, 色散关系, 有限元, 视频分析

## 一句话总结
本文提出 VSWE（Visual Surface Wave Elastography），仅通过一段表面波传播的视频，提取色散关系并结合基于物理的有限元优化，推断介质的亚表面厚度和刚度参数，在模拟和真实明胶实验中均实现了高精度的参数恢复，为居家健康监测提供了概念验证。

## 研究背景与动机

**领域现状**：弹性成像（elastography）是测量组织刚度的重要技术，可用于检测肿瘤、肌肉骨骼退化、肝病等。目前的弹性成像方法（瞬态弹性成像、剪切波弹性成像、磁共振弹性成像）都依赖昂贵的专用设备和专业操作人员。

**现有痛点**：(a) 超声和 MRI 弹性成像设备费用高昂，无法在家使用；(b) 需要训练有素的医学专家操作，限制了常规筛查的可行性；(c) 现有视频材料表征方法（如 Visual Vibration Tomography）需要建模整个物体的全局振动模态，对人体等复杂几何体不现实。

**核心矛盾**：人体表面在外部激励下会产生可见的表面波（如按摩枪引起的皮肤涟漪），这些波携带了亚表面物理特性信息，但目前缺乏从视频中提取这些信息的方法。

**本文目标** 从一段表面波视频中推断介质亚表面的厚度 $T$ 和刚度 $E$。

**切入角度**：借鉴地震学中利用表面波推断地下结构的思路，但用密集的视觉数据（视频）替代稀疏的传感器。关键 insight 是色散关系（wavenumber-frequency 关系）完全由厚度和刚度决定（在常见生物力学假设下），因此只需从视频中提取色散关系，再求解与之匹配的物理参数。

**核心 idea**：从表面波视频中提取色散关系，通过有限元仿真+SSIM优化找到最匹配的厚度和刚度，实现纯视频的亚表面物理特性推断。

## 方法详解

### 整体框架
输入是一段拍摄介质表面波传播的视频。管线分两阶段：(1) 从视频中提取色散关系；(2) 求解使理论色散关系最匹配观测色散关系的厚度和刚度参数。

### 关键设计

1. **视频运动提取**:

    - 功能：从视频中提取亚像素级的表面位移场
    - 核心思路：使用基于相位的运动处理（phase-based motion processing），在复数可导金字塔（complex steerable pyramid）中计算局部相位偏移，再转换为像素位移。得到每个像素 $(x, y)$ 在每帧 $t$ 的水平和垂直位移 $\tilde{u}(\tilde{x}, \tilde{y}, t)$, $\tilde{v}(\tilde{x}, \tilde{y}, t)$
    - 设计动机：表面波通常是亚像素级运动，传统光流方法精度不够，而相位方法天然对亚像素位移敏感

2. **色散关系提取**:

    - 功能：将时空域的位移信号转换为频率-波数域的色散关系
    - 核心思路：对位移视频的每一行像素做 2D FFT（空间 $x$ → 波数 $\gamma$，时间 $t$ → 频率 $\omega$），取幅值得到色散图像。对所有行和两个位移方向取平均以提升信噪比：$\mathbf{D}_{\text{obs}} = \frac{1}{2H}\sum_{i=1}^{H}(|\hat{\tilde{u}}(i)| + |\hat{\tilde{v}}(i)|)$
    - 设计动机：色散关系是介质波传播行为的紧凑表示，完全由物理参数决定。FFT 是提取频率信息的标准且高效的方法

3. **物理模型与参数优化**:

    - 功能：找到厚度 $T$ 和刚度 $E$ 使理论色散关系与观测最匹配
    - 核心思路：假设均匀各向同性线弹性介质，软组织层（未知 $T$, $E$，已知 $\rho=1$ g/cm³, $\nu=0.45$）覆盖在硬骨骼层上。对假设的 $(T, E)$ 值，用有限元法（FEM）求解弹性波方程加 Bloch-Floquet 周期边界条件，得到理论色散关系 $\mathfrak{D}(T, E)$。将理论色散曲线通过高斯核转化为图像 $\mathbf{D}_{\text{hyp}}(T, E)$，最大化 $\text{SSIM}(\mathbf{D}_{\text{hyp}}(T, E), \mathbf{D}_{\text{obs}})$。目前使用网格搜索
    - 设计动机：SSIM 对结构相似性的度量比 MSE 和 PSNR 更鲁棒，实验证实 SSIM 给出最尖锐的优化景观和最准确的参数估计

### 损失函数 / 训练策略
本方法是纯物理驱动的优化方法，无需 ML 训练。核心优化目标是 $\text{argmax}_{T,E} \text{SSIM}(\mathbf{D}_{\text{hyp}}, \mathbf{D}_{\text{obs}})$。作者还引入了无量纲特征数 $\pi_1 \sim \pi_6$ 来指导不同参数范围下的实验设计（如调整观测窗口、帧率、频率范围等以保持性能）。

## 实验关键数据

### 主实验：真实明胶样本
三个不同体积（1000/1100/1500 mL）的明胶样本，使用卡尺测量真实厚度、流变仪测量真实刚度：

| 样本 | 真实厚度 (mm) | VSWE估计厚度 | 真实刚度 (kPa) | VSWE估计刚度 | 刚度误差 |
|------|-------------|-------------|-------------|-------------|---------|
| 1000 mL | ~42 | 落在置信区间内 | ~12-16 | 与流变仪吻合 | <1.2% |
| 1100 mL | ~46 | 落在置信区间内 | ~12-16 | 与流变仪吻合 | <1.2% |
| 1500 mL | ~55 | 落在置信区间内 | ~12-16 | 与流变仪吻合 | <1.2% |

每个样本在不同温度下（从冰箱取出后随时间升温）采集约 60 个视频，VSWE 估计的刚度随温度的下降趋势与流变仪一致。

### 模拟实验：灵敏度分析

| 参数变化 | VSWE 检测能力 |
|---------|-------------|
| ±5% 厚度变化 | 可检测 |
| ±10% 厚度变化 | 可检测 |
| ±5% 刚度变化 | 可检测 |
| ±10% 刚度变化 | 可检测 |

在 5 个厚度 × 4 个刚度 × 9 个扰动 = 180 个模拟样本上验证，VSWE 对 5% 级别的参数变化仍敏感。

### 3D 人腿模拟
基于 Visible Human Project 的真实解剖几何，COMSOL 全 3D 物理模拟：
- VSWE 成功恢复了小腿上部滑动窗口中厚度的空间变化趋势
- 在踝关节附近三个厚度差异显著的区域也给出了正确的厚度估计
- 成功恢复了恒定的组织刚度

### 关键发现
- **SSIM 优于其他目标函数**：与曲线匹配、MSE、PSNR 对比，SSIM 在模拟腿数据上是唯一给出正确最优参数的目标函数
- **不完整色散关系仍可工作**：视频中可能只观测到部分波模态，但 SSIM 的结构匹配特性使其仍能从部分匹配中推出正确参数
- **温度-刚度关系**：实验中明胶刚度与温度呈负相关，VSWE 定量地追踪了这一变化

## 亮点与洞察
- **纯物理驱动 + 纯视觉输入**：完全不需要 ML 模型训练，也不需要超声/MRI 等专用设备，仅用高速相机和振动器即可测量组织特性。这种"zero-shot"物理方法在可解释性上远超数据驱动方法
- **色散关系作为中间表示**：将视频中的时空信息压缩为 2D 色散图像，实现了从视频到物理参数的优雅桥梁。这一思路可迁移到其他基于波传播的逆问题
- **特征数指南**：引入 6 个无量纲特征数 $\pi_1 \sim \pi_6$ 来指导实验设计，使方法可适用于参数范围差异极大的场景（如更大或更小的物体、更硬或更软的材料），确保 VSWE 性能不依赖于具体的参数尺度
- **局部分析优势**：不需要建模全身复杂几何（与 Visual Vibration Tomography 相比），仅需局部表面波分析即可推断局部组织特性

## 局限与展望
- **需要高速相机**：当前实验使用 600 FPS 高速相机，普通手机相机可能无法捕捉足够的时间分辨率
- **简化的物理模型**：假设各向同性、线弹性、均匀刚度、已知密度和泊松比，实际人体组织更复杂（粘弹性、各向异性、多层结构）
- **仅验证了 1D 波传播**：当前仅分析单方向波传播，2D 波场分析可能提供更丰富的信息
- **还未在真实人体上验证**：3D 人腿实验仍是模拟的，真实人体上有呼吸、肌肉收缩等额外运动干扰
- **网格搜索效率低**：当前用网格搜索优化，可探索梯度优化或贝叶斯优化加速
- **改进方向**：可引入粘弹性模型；可用手机慢动作（240 FPS）验证；可探索多频率顺序激励提升信噪比

## 相关工作与启发
- **vs Visual Vibration Tomography [CVPR'22]**：VVT 从全局振动模态推断空间变化的刚度和密度，需要已知全局几何。VSWE 只需局部表面波分析，更适合人体等复杂几何
- **vs 传统弹性成像（超声/MRI）**：传统方法精度更高但设备昂贵且需专家操作，VSWE 精度较低但成本极低，适合居家初筛
- **vs 地震学表面波方法**：地震学用稀疏传感器，VSWE 用密集视觉数据，可获得更完整的色散关系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨领域创新，将地震学表面波理论与计算机视觉结合，思路非常新颖
- 实验充分度: ⭐⭐⭐⭐ 模拟+真实实验验证充分，特征数分析深入，但缺少真实人体实验
- 写作质量: ⭐⭐⭐⭐⭐ 从直觉到数学公式的展开非常流畅，物理概念解释清晰
- 价值: ⭐⭐⭐⭐ 概念新颖且具有居家医疗潜力，但距实际应用仍有距离

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](../../NeurIPS2025/medical_imaging/surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](../../CVPR2025/medical_imaging/thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)
- [\[ICCV 2025\] NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction](neurons_emulating_the_human_visual_cortex_improves_fidelity_and_interpretability.md)
- [\[CVPR 2026\] Anatomica: Localized Control over Geometric and Topological Properties for Anatomical Diffusion Models](../../CVPR2026/medical_imaging/anatomica_localized_control_over_geometric_and_topological_properties_for_anatom.md)

</div>

<!-- RELATED:END -->
