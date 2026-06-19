---
title: >-
  [论文解读] Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories
description: >-
  [CVPR 2025][3D视觉][3D编辑] Perturb-and-Revise 通过在 NeRF 参数空间中进行自适应扰动使参数跳出局部最小值，然后利用多视图扩散模型的 Score Distillation 沿生成轨迹优化，配合身份保持梯度实现灵活的 3D 编辑，首次支持包括姿态变化和添加新物体在内的大幅几何/外观修改。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D编辑"
  - "NeRF编辑"
  - "Score Distillation"
  - "参数扰动"
  - "多视图一致性"
---

# Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories

**会议**: CVPR 2025  
**arXiv**: [2412.05279](https://arxiv.org/abs/2412.05279)  
**代码**: [https://susunghong.github.io/Perturb-and-Revise](https://susunghong.github.io/Perturb-and-Revise)  
**领域**: 3D视觉  
**关键词**: 3D编辑, NeRF编辑, Score Distillation, 参数扰动, 多视图一致性

## 一句话总结

Perturb-and-Revise 通过在 NeRF 参数空间中进行自适应扰动使参数跳出局部最小值，然后利用多视图扩散模型的 Score Distillation 沿生成轨迹优化，配合身份保持梯度实现灵活的 3D 编辑，首次支持包括姿态变化和添加新物体在内的大幅几何/外观修改。

## 研究背景与动机

**领域现状**：基于文本的 3D 编辑已成为研究热点，Instruct-NeRF2NeRF、Posterior Distillation 等方法通过扩散模型引导 NeRF 参数更新实现文本驱动的 3D 编辑。

**现有痛点**：现有方法在颜色、纹理和风格修改上表现不错，但面对需要大幅几何或外观变化的编辑（如改变姿态、添加新物体、改变物种）时几乎无能为力。根本原因是已优化的 NeRF 参数处于一个能量较低的局部最小值，Score Distillation 的梯度不足以将参数推出当前的吸引盆（basin of attraction），即使更换了编辑 prompt。

**核心矛盾**：NeRF 编辑面临一个两难：(1) 从源 NeRF 出发编辑，参数被困在局部最小值中，无法实现大幅变化；(2) 从随机初始化重新生成，则完全丢失与源物体的关联。

**本文目标**：实现灵活的 3D 编辑，支持从简单的颜色/纹理修改到复杂的几何/姿态变化，同时保持与源物体的相似性。

**切入角度**：将 NeRF 参数视为生成 ODE 中的"粒子"。已优化的 NeRF 是到达了目标分布的粒子，处于低能量状态。通过在参数空间中加入扰动，相当于将粒子回退到优化过程的中间状态，使其可以沿新的生成轨迹（由编辑 prompt 指定）重新收敛。

**核心 idea**：通过参数空间的线性插值扰动使 NeRF 跳出局部最小值，自适应确定扰动量，然后使用多视图 Score Distillation 引导编辑，最后用身份保持梯度拉近编辑结果与源物体的距离。

## 方法详解

### 整体框架

Perturb-and-Revise 包含三个阶段：(1) 参数扰动——将源 NeRF 参数与随机初始化进行线性插值，自适应确定插值比例 $\eta$；(2) 多视图一致编辑——使用 MVDream 等多视图扩散模型通过 Score Distillation 沿生成轨迹优化；(3) 身份保持精调——在后期加入 IPG 梯度，平衡编辑效果与源物体保真度。

### 关键设计

1. **参数扰动（Parameter Perturbation）**:

    - 功能：使 NeRF 参数跳出当前局部最小值，获得足够的"灵活性"以进行大幅编辑
    - 核心思路：将源 NeRF 参数 $\theta_{\text{src}}$ 与随机初始化参数 $\theta_0$ 做线性插值：$\theta_{\text{perturbed}} = (1-\eta) \cdot \theta_{\text{src}} + \eta \cdot \theta_0$。其中 $\eta \in [0,1]$ 控制扰动量——$\eta=0$ 为不扰动，$\eta=1$ 为完全随机初始化。扰动后的参数处于优化过程的"中间状态"，从此处重新开始优化可以沿新 prompt 指引的生成轨迹收敛
    - 设计动机：类比扩散模型中加噪声的过程——加更多噪声意味着撤销更多已完成的生成，从而允许更大的修改。在参数空间中扰动是这种思想在 3D 编辑中的自然推广

2. **自适应 $\eta$ 选择（Loss Landscape Analysis）**:

    - 功能：自动确定合适的扰动量，避免手动搜索
    - 核心思路：利用损失函数作为代理来衡量吸引盆的深度。具体做法是：先用编辑 prompt 模拟若干 Score Distillation 步骤，计算前几步和后几步的平均损失差。如果损失几乎不下降甚至上升，说明参数被困在深的局部最小值中，需要更大的 $\eta$。使用反指数衰减函数将损失差映射为 $\eta$ 值
    - 设计动机：不同类型的编辑需要不同程度的扰动——颜色修改需要很小的 $\eta$，而姿态变化需要较大的 $\eta$。通过分析损失景观可以自动适配，避免昂贵的网格搜索

3. **身份保持梯度（Identity-Preserving Gradient, IPG）**:

    - 功能：在编辑后期拉近结果与源物体的距离，平衡编辑效果和源物体保真度
    - 核心思路：在优化后期阶段额外引入一个梯度项，使编辑后的 NeRF 渲染与源 NeRF 渲染保持相似：$d\theta_\tau^{\text{refine}} = d\theta_\tau + \lambda_d \nabla_\theta d(\theta_\tau, \theta_{\text{src}})$，其中 $d(\cdot, \cdot)$ 使用 L1 + 感知损失的组合。这形成了两个梯度力的"拔河"——Score Distillation 推向编辑目标，IPG 拉回源物体
    - 设计动机：参数扰动+Score Distillation 阶段可能引入扩散模型的估计误差或偏差，IPG 在后期校正这些偏差。同时，从一开始就施加保持约束会与生成 ODE 冲突，所以只在后期引入

### 损失函数 / 训练策略

使用多视图扩散模型（MVDream）同时生成 N 个视角的一致预测进行 Score Distillation 更新。噪声级别采用时间步退火策略（从高到低），先做低频编辑再做精细修改。IPG 使用 L1 + LPIPS 感知损失的组合。

## 实验关键数据

### 主实验

| 方法 | CLIP-Dir-Sim ↑ | LPIPS_vgg ↓ | LPIPS_alex ↓ | 说明 |
|------|---------------|------------|-------------|------|
| SDS (MVDream) | 0.0438 | 0.1273 | 0.1533 | 模糊，无法大幅改变 |
| PDS | 0.0285 | **0.0337** | **0.0215** | 编辑不足，过于保守 |
| IN2N | 0.0557 | 0.1065 | 0.1112 | 只改纹理，无法改几何 |
| **PnR (Ours)** | **0.0565** | 0.1060 | 0.1034 | **编辑与保真的最佳平衡** |

### 消融实验

| 配置 | CLIP-Dir-Sim ↑ | CLIP-Dir-Con ↑ | LPIPS ↓ | 说明 |
|------|---------------|---------------|---------|------|
| 无 IPG 精调 | 0.0624 | 0.7572 | 0.1147 | 编辑更激进但偏离源 |
| 有 IPG 精调 | 0.0565 | 0.7642 | 0.1047 | LPIPS 显著降低 |

### 关键发现

- 参数扰动的效果非常显著：当 $\eta=0$ 时无法实现姿态变化和新物体添加，适当增大 $\eta$ 可以实现越来越大的修改
- 自适应 $\eta$ 选择在所有编辑类型的平均表现上接近或达到最优固定 $\eta$ 的效果，同时实验成功率高且计算开销远低于网格搜索
- 不同编辑类型需要不同的 $\eta$：颜色/纹理编辑需要小 $\eta$，姿态/物体添加需要大 $\eta$
- IPG 在 LPIPS 指标上带来显著改善（0.1147→0.1047），同时 CLIP 方向一致性反而微增，表明 IPG 有效矫正了偏差
- PDS 虽然 LPIPS 最低，但因过于保守几乎未做有效编辑（CLIP-Dir-Sim 最低）

## 亮点与洞察

- **参数空间扰动**的思想非常优雅。将 NeRF 参数视为粒子流的端点，通过与初始分布插值"倒带"优化过程，使其可以沿新轨迹重新演化。这种将扩散中的加噪/去噪思想迁移到参数空间的做法很有启发性
- **损失景观分析自动选择 $\eta$** 是一个实用的算法创新，避免了昂贵的超参数搜索。这种"模拟几步看趋势"的思想可以迁移到其他需要自适应控制的优化场景
- **分阶段策略**（先大幅扰动做粗编辑 → 再 IPG 精调保真）的设计符合直觉且实验验证有效

## 局限与展望

- 扰动后的优化仍需要相当多的迭代步骤，编辑效率可能不如基于单步推理的方法
- 对于需要精确空间控制的编辑（如将物体移到特定位置），文本 prompt 的控制力可能不足
- 方法目前主要验证在物体级编辑上，对复杂场景中的局部编辑（如只修改场景中一个物体）支持有限
- 多视图扩散模型本身的 3D 一致性仍不完美，可能引入 Janus 问题等伪影

## 相关工作与启发

- **vs Instruct-NeRF2NeRF**: IN2N 通过 InstructPix2Pix 迭代更新训练视图进行编辑，擅长颜色/纹理修改但完全无法处理几何变化。PnR 通过参数扰动从根本上解决了这个局限
- **vs PDS (Posterior Distillation)**: PDS 通过匹配随机 latent 保持与源的相似性，但过于保守几乎不做有效编辑。PnR 的"先大胆扰动再精细校正"策略更灵活
- **vs SDS**: SDS 从源 NeRF 直接优化，虽能改变纹理但无法突破局部最小值，且易产生模糊纹理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 参数空间扰动的思想新颖且优雅，将扩散概念迁移到 NeRF 编辑
- 实验充分度: ⭐⭐⭐⭐ 多种编辑类型、多个baseline、定量+定性+消融+用户研究
- 写作质量: ⭐⭐⭐⭐⭐ 动机图解清晰，理论洞察深刻
- 价值: ⭐⭐⭐⭐⭐ 首次实现大幅几何变化的3D编辑，填补了重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)
- [\[ICCV 2025\] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior](../../ICCV2025/3d_vision/driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)
- [\[CVPR 2025\] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)
- [\[CVPR 2025\] PERSE: Personalized 3D Generative Avatars from A Single Portrait](perse_personalized_3d_generative_avatars_from_a_single_portrait.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)

</div>

<!-- RELATED:END -->
