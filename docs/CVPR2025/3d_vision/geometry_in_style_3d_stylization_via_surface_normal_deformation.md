---
title: >-
  [论文解读] Geometry in Style: 3D Stylization via Surface Normal Deformation
description: >-
  [CVPR 2025][3D视觉][网格风格化] 提出通过优化三角网格的表面法线方向、结合可微分ARAP（dARAP）层恢复顶点位置的方法，实现文本驱动的网格风格化，能产生表达力强但保持形状身份的几何变形。 领域现状：文本驱动的3D形状编辑近年兴起，主流方法利用扩散模型的得分蒸馏损失来引导形状变形。现有方法通常基于bump…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "网格风格化"
  - "表面法线"
  - "ARAP变形"
  - "文本驱动3D编辑"
  - "可微分几何"
---

# Geometry in Style: 3D Stylization via Surface Normal Deformation

**会议**: CVPR 2025  
**arXiv**: [2503.23241](https://arxiv.org/abs/2503.23241)  
**代码**: [https://threedle.github.io/geometry-in-style](https://threedle.github.io/geometry-in-style)  
**领域**: 3D视觉  
**关键词**: 网格风格化, 表面法线, ARAP变形, 文本驱动3D编辑, 可微分几何

## 一句话总结

提出通过优化三角网格的表面法线方向、结合可微分ARAP（dARAP）层恢复顶点位置的方法，实现文本驱动的网格风格化，能产生表达力强但保持形状身份的几何变形。

## 研究背景与动机

**领域现状**：文本驱动的3D形状编辑近年兴起，主流方法利用扩散模型的得分蒸馏损失来引导形状变形。现有方法通常基于bump map或Jacobian场来表示变形。

**现有痛点**：基于bump map的方法（如Text2Mesh）变形太保守，只能做表面纹理级别的修改；基于Jacobian场的方法（如TextDeformer、MeshUp）变形过于自由，虽然能做大变形，但容易破坏原始形状的身份特征，产生伪影，需要额外的L2正则化来约束。

**核心矛盾**：变形的"表达力"与"身份保持"之间存在根本性的trade-off——bump map太受限、Jacobian太自由，两者都无法同时满足有表达力且保持身份的要求。

**本文目标** 找到一种变形表示，既能做出有语义意义的大变形（如把椅子变成折纸风格），又能保留原始形状的结构特征（如椅子的腿、扶手比例）。

**切入角度**：作者观察到，如果用表面法线作为变形的驱动信号，通过ARAP（As-Rigid-As-Possible）求解恢复顶点位置，由于ARAP强制局部刚性约束，变形天然不会产生缩放和剪切，从而从构造上保证了身份保持。

**核心 idea**：用目标法线驱动可微分ARAP变形层（dARAP），将经典ARAP的多次迭代压缩为单次局部-全局步骤，嵌入梯度下降优化流水线中，配合扩散模型的语义损失实现文本驱动的网格风格化。

## 方法详解

### 整体框架

输入一个三角网格和文本提示，优化每个顶点的目标法线向量。每次优化迭代中，dARAP的局部步骤从目标法线计算每个顶点邻域的最优旋转矩阵，全局步骤通过Poisson方程求解变形后的顶点位置。变形后的网格通过可微分渲染器生成多视角图像，输入级联得分蒸馏（CSD）损失与DeepFloyd IF扩散模型对齐文本语义。

### 关键设计

1. **dARAP可微分变形层**:

    - 功能：从目标法线恢复变形顶点位置的可微分模块
    - 核心思路：将经典ARAP的多次局部-全局交替迭代压缩为单次。局部步骤对每个顶点邻域求解正交Procrustes问题——构建包含边向量和法线的矩阵$X_k$，通过SVD分解得到旋转矩阵$\hat{R}_k = V_k U_k^\top$。全局步骤固定旋转矩阵，求解最小二乘Poisson方程$L\hat{\mathcal{V}} = \text{rhs}$得到变形顶点位置。整个过程可微分，可直接嵌入神经网络。
    - 设计动机：经典ARAP需要迭代到收敛无法高效反向传播，而dARAP在更大的梯度下降循环内运行，每次只需一步即可达到好效果。超参数$\lambda$控制法线匹配强度，默认设为8。

2. **表面法线变形表示**:

    - 功能：用每顶点目标法线向量作为变形的参数化空间
    - 核心思路：直接优化$|\mathcal{V}| \times 3$个实数，将其归一化为单位法线后作为dARAP的输入。局部步骤中旋转矩阵的求解同时考虑保持原始边方向和对齐目标法线，通过cotangent权重加权。这意味着变形被约束为局部刚性旋转，不允许缩放和剪切。
    - 设计动机：与Jacobian表示相比，法线表示天然限制了变形空间——只能旋转不能缩放/剪切，从而从构造上避免了形状身份破坏，无需额外的正则化损失。

3. **级联得分蒸馏视觉损失（CSD）**:

    - 功能：驱动法线优化方向使变形匹配文本描述的风格
    - 核心思路：使用DeepFloyd IF的stage 1和stage 2两阶段扩散模型。每个epoch从多视角渲染变形后的网格，batch大小为8张视角图，计算CSD损失反向传播到目标法线。总共优化2500个epoch，学习率0.002，Adam优化器。
    - 设计动机：级联T2I模型比单阶段模型能生成更高保真的指导信号，配合法线参数化可以产生语义合理且细节丰富的风格化。

### 损失函数 / 训练策略

仅使用CSD视觉损失，不需要额外的身份保持正则化。初始目标法线为原始网格的面积加权顶点法线。优化完成后可以通过改变推理时的$\lambda$值调节风格化强度，$\lambda$增大使风格更强烈，减小则更接近原始形状。

## 实验关键数据

### 主实验

| 方法 | 三角面积比均值↓(理想1.0) | 三角面积比标准差↓(理想0) |
|------|-------------------------|------------------------|
| TextDeformer | 0.827 | 0.360 |
| MeshUp | 1.288 | 0.363 |
| Geometry in Style (Ours) | **1.080** | **0.233** |

在20个mesh-prompt对上评估，本方法的面积比最接近1且标准差最小，说明形状身份保持最好。

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| 推理$\lambda=8$（训练同） | 标准风格化效果 |
| 推理$\lambda > 8$ | 风格更强烈但仍合理，展示了方法的鲁棒性 |
| 推理$\lambda < 8$ | 风格变弱但仍可见 |
| 局部区域风格化（区域外旋转设为identity） | 仅局部变形，无边界伪影，展示dARAP正则化效果 |
| 不同域的同一风格 | 椅子/动物/花瓶均能一致传达乐高风格 |

### 关键发现

- 法线表示在构造上限制了变形空间，使得方法不需要额外的身份正则化损失即可保持形状身份，这比Jacobian方法简洁得多
- 推理时可以使用比训练时更大的$\lambda$值，产生更几何突出但仍合理的风格化，说明优化得到的法线本身具有语义合理性
- 在有机表面（动物、人体）和人造物体（椅子、花瓶）上都表现良好，变形是部件感知的（如热带风格椅子的褶皱出现在座面和靠背而非腿部）

## 亮点与洞察

- **法线作为变形表示的精妙之处**：将变形空间限制为"只能旋转"的子空间，从数学构造上保证了身份保持，避免了Jacobian方法需要额外正则化的笨重设计。这种"通过表示设计来隐式约束"的思路比"通过损失函数来显式正则化"要优雅得多。
- **单步dARAP的发现**：经典ARAP需要多次迭代才能收敛，但在嵌入外层梯度下降的场景中，单步即足够——因为目标法线本身在持续被优化。这个insight可能对其他需要嵌入可微分几何求解器的场景有启发。
- **推理时可调强度**：优化后的法线可以用不同的$\lambda$重新应用，提供了用户友好的风格强度控制，这是MeshUp和TextDeformer不具备的特性。

## 局限与展望

- 依赖cotangent Laplacian，要求输入为流形网格且三角形纵横比良好，需要预处理remesh
- 可能产生自相交（如灯具的杆件旋转后相互穿透），虽然可通过降低$\lambda$缓解但无法根治
- 拓扑保持——不能添加新部件或改变拓扑结构，限制了某些风格化可能
- 优化时间较长（单个A40上约2小时15分钟），实时应用受限

## 相关工作与启发

- **vs TextDeformer/MeshUp**: 这两种方法基于Jacobian场，变形自由度高但容易破坏身份。本文通过法线表示+ARAP约束从根本上解决了这个问题，且CLIP相似度也更好。
- **vs Text2Mesh**: Text2Mesh基于顶点位移做表面纹理，变形受限无法产生大的结构性变化。本文的变形可以产生全局轮廓变化。
- **vs Liu & Jacobson的法线风格化**: 他们用球面法线模板，本文用文本提示，且变形是部件感知的（不同部件可以有不同的目标法线）。

## 评分

- 新颖性: ⭐⭐⭐⭐ 法线+dARAP的组合很精巧，但核心idea是对已有方法的巧妙重组
- 实验充分度: ⭐⭐⭐⭐ 定性结果丰富多样，定量指标（面积比）虽有效但较简单
- 写作质量: ⭐⭐⭐⭐⭐ 论文写作清晰流畅，数学推导严谨且可读性强
- 价值: ⭐⭐⭐⭐ 对3D风格化领域有实际价值，dARAP层可迁移到其他几何任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)
- [\[CVPR 2025\] Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](../../ICCV2025/3d_vision/tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](../../ICCV2025/3d_vision/hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[CVPR 2025\] Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization](morpheus_text-driven_3d_gaussian_splat_shape_and_color_stylization.md)

</div>

<!-- RELATED:END -->
