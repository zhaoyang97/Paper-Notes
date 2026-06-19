---
title: >-
  [论文解读] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation
description: >-
  [CVPR 2025][3D视觉][3D场景生成] 提出SceneFactor，通过分解式潜空间扩散（先生成粗语义box布局，再生成精细几何），实现文本引导的大规模3D室内场景生成，并支持通过语义box操作进行直观的局部编辑。 3D场景的可编辑生成对AR/VR、游戏和建筑设计至关重要。内容创作天然是迭代过程…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D场景生成"
  - "分块扩散"
  - "语义引导"
  - "可编辑生成"
  - "VQ-VAE"
---

# SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation

**会议**: CVPR 2025  
**arXiv**: [2412.01801](https://arxiv.org/abs/2412.01801)  
**代码**: 无  
**领域**: 3D Vision  
**关键词**: 3D场景生成, 分块扩散, 语义引导, 可编辑生成, VQ-VAE

## 一句话总结

提出SceneFactor，通过分解式潜空间扩散（先生成粗语义box布局，再生成精细几何），实现文本引导的大规模3D室内场景生成，并支持通过语义box操作进行直观的局部编辑。

## 研究背景与动机

3D场景的可编辑生成对AR/VR、游戏和建筑设计至关重要。内容创作天然是迭代过程，需要用户能在局部区域进行控制和编辑。但现有方法存在明显不足：

- **2D提升方法**（如Score Distillation）：缺乏3D推理，全局结构不连贯
- **物体检索方法**：受限于物体数据库，几何固定无法变化
- **直接场景扩散**（如BlockFusion/XCube）：生成质量好，但不支持局部编辑
- **条件生成方法**：修改输入条件后需要重新合成整个场景

核心需求：一个既能高保真生成3D场景、又能像操作"搭积木"一样轻松局部编辑的方法。

## 方法详解

### 整体框架

SceneFactor采用两阶段分解式生成流程：
1. 文本 → 语义box布局 $S$：通过文本条件的潜空间扩散生成粗糙的3D语义布局
2. 语义布局 $S$ → 几何 $G$：通过语义条件的潜空间扩散生成高保真的unsigned distance field几何
3. 大规模场景通过chunk-by-chunk outpainting生成
4. 编辑通过在语义空间操作box实现，仅需重新生成编辑区域的几何

### 关键设计1：双VQ-VAE潜空间

- **功能**：为语义和几何分别学习高度压缩的潜空间表示
- **核心思路**：几何chunk $G \in \mathbb{R}^{128 \times 64 \times 128}$ 通过3D VQ-VAE编码器压缩4倍为 $f_G \in \mathbb{R}^{32 \times 16 \times 32}$；语义chunk $S \in \mathbb{Z}^{c \times 32 \times 16 \times 32}$（$c=10$类）压缩4倍为 $f_S \in \mathbb{R}^{8 \times 4 \times 8}$。两个VQ-VAE的特征维度均为1维
- **设计动机**：高维3D数据直接生成不可行。VQ-VAE不仅实现了极高的空间压缩率，而且生成smooth流形，有利于后续扩散建模。关键是保持3D网格结构，使潜空间与物理空间有精确对应，从而支持局部编辑。语义空间极高压缩（128→8）使扩散学习简单；几何空间保留足够分辨率以重建细节

### 关键设计2：分解式扩散生成

- **功能**：将复杂的3D场景生成解耦为高层结构+低层几何两个简单任务
- **核心思路**：第一阶段文本→语义扩散 $\Psi_S$ 用3D OpenAI LDM + BERT text attention，目标 $\mathcal{L}_{LDM,sem} = \|\Psi_S(f_{S,t}, t, \tau_i) - v_{S,t}\|_1$。第二阶段语义→几何扩散 $\Psi_G$ 用空间交叉注意力（conv-based，窗口大小3），语义map作为value，几何latent作为query/key，目标 $\mathcal{L}_{LDM,geo} = \|\Psi_G(f_{G,t}, t, f_S) - v_{G,t}\|_2$
- **设计动机**：直接从文本生成精细几何的方程式高度病态。分解后：第一阶段只需学习文本→粗box的对应关系（语义空间极小），第二阶段只需在给定明确空间约束下填充几何细节（任务大大简化）。conv-based attention增大了语义条件的感受野，有效捕获局部语义-几何相关性

### 关键设计3：基于语义box的局部编辑

- **功能**：支持5种直观的场景编辑操作，每次仅需2次鼠标点击
- **核心思路**：编辑在语义map $S$ 上进行box操作（添加/删除/替换/缩放/移动），对应几何区域 $\mathcal{R}$ 填充高斯噪声后用inpainting重新生成。其余区域保持不变
- **设计动机**：由于 $f_G$ 和 $S$ 具有相同分辨率且空间精确对齐，语义空间的局部修改可无缝传播到几何空间。用户只需指定box的两个对角顶点，无需精确的区域边界分割或复杂的prompt engineering

### 损失函数

- VQ-VAE几何：$\mathcal{L}^{geo} = \|G - \mathcal{D}^G(\mathcal{E}^G(G))\|_1 + \mathcal{L}^{quant}(f_G)$
- VQ-VAE语义：$\mathcal{L}^{sem} = \mathcal{L}^{NLL}(S, \mathcal{D}^S(\mathcal{E}^S(S))) + \mathcal{L}^{quant}(f_S)$
- 扩散模型均使用 $v$-parameterization

## 实验关键数据

### 主实验：3D-FRONT场景生成质量

| 方法 | FID ↓ | Scene-FID ↓ | KID(×100) ↓ | Coverage ↑ |
|------|-------|-------------|-------------|------------|
| SDFusion | 较高 | 较高 | 较高 | 较低 |
| BlockFusion* | 中等 | 中等 | 中等 | 中等 |
| **SceneFactor** | **最低** | **最低** | **最低** | **最高** |

*注：BlockFusion为无条件生成。SceneFactor在独立chunk和完整场景生成上均优于baseline。

### 消融实验

| 配置 | 效果 |
|------|------|
| 无语义分解（直接文本→几何） | 几何质量明显下降，文本对齐差 |
| 无conv attention（用标准attention） | 语义-几何局部对应变弱 |
| 单阶段VQ-VAE | 压缩率受限或重建质量差 |

### 关键发现

- 分解式生成比直接端到端显著提升了文本-场景对齐和几何保真度
- 局部编辑保持全局一致性：编辑区域外的场景结构不变
- chunk-based outpainting可以生成任意大小场景

## 亮点与洞察

1. **分解即简化**：将不可解的高维生成问题拆为两个低维子问题的策略值得推广
2. **编辑友好的表征**：语义box作为中间代理层，巧妙桥接了用户意图与复杂的3D几何
3. **空间对齐设计**：$f_G$ 和 $S$ 的等分辨率设计是局部编辑成功的关键前提

## 局限与展望

- 仅支持10个语义类别，限制了场景多样性
- 基于3D-FRONT训练，泛化到真实扫描场景未验证
- 编辑粒度受限于语义类别而非实例级别
- 无外观/纹理生成，仅输出几何距离场
- 未来可扩展到更多类别和实例级编辑

## 相关工作与启发

- **BlockFusion**：通过滑动窗口生成场景，但无条件生成且不支持编辑
- **XCube**：层次化粗到细生成，同样不可编辑
- **RePaint**：inpainting策略的灵感来源，用于chunk间的一致性outpainting

## 评分

⭐⭐⭐⭐ — 分解式生成思路优雅，局部编辑能力是实用性的关键突破。主要局限在于仅限室内场景+固定类别+无纹理，但核心方法论有广泛适用性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LT3SD: Latent Trees for 3D Scene Diffusion](lt3sd_latent_trees_for_3d_scene_diffusion.md)
- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [\[NeurIPS 2025\] From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries](../../NeurIPS2025/3d_vision/from_programs_to_poses_factored_real-world_scene_generation_via_learned_program_.md)
- [\[CVPR 2025\] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion](ouroboros3d_image-to-3d_generation_via_3d-aware_recursive_diffusion.md)

</div>

<!-- RELATED:END -->
