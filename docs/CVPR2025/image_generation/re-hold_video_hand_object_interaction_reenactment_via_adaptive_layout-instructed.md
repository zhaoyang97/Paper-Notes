---
title: >-
  [论文解读] Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model
description: >-
  [CVPR 2025][图像生成][手物交互] 提出 Re-HOLD，首个以人为中心的手物交互(HOI)视频重演框架，通过分离式布局表示解耦手和物体建模，结合交互纹理增强模块和自适应布局调整策略，实现跨物体高保真 HOI 视频生成。 数字人视频技术的发展推动了对手物交互(HOI)场景生成的需求。然而，HOI 视频合成面临三大…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "手物交互"
  - "视频重演"
  - "布局引导"
  - "扩散模型"
  - "纹理增强"
---

# Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model

**会议**: CVPR 2025  
**arXiv**: [2503.16942](https://arxiv.org/abs/2503.16942)  
**代码**: [项目页面](https://fyycs.github.io/Re-HOLD)  
**领域**: 图像生成  
**关键词**: 手物交互, 视频重演, 布局引导, 扩散模型, 纹理增强

## 一句话总结

提出 Re-HOLD，首个以人为中心的手物交互(HOI)视频重演框架，通过分离式布局表示解耦手和物体建模，结合交互纹理增强模块和自适应布局调整策略，实现跨物体高保真 HOI 视频生成。

## 研究背景与动机

数字人视频技术的发展推动了对手物交互(HOI)场景生成的需求。然而，HOI 视频合成面临三大核心挑战：

- **手物遮挡导致的纠缠**：手与物体的物理接触产生复杂遮挡，容易在手物界面处产生伪影
- **细节纹理恢复困难**：手和物体自由度高但在画面中仅占有限像素，精确恢复纹理极具挑战
- **跨物体尺寸差异**：不同物体的形状和尺寸差异影响交互位置，若运动序列不变会导致不自然的抓握
- **现有方法局限**：HOI-Swap 仅支持以物体为中心的单手抓握视频编辑，无法处理双手交互场景

## 方法详解

### 整体框架

Re-HOLD 采用双分支架构：Reference U-Net 处理目标物体图像提取纹理信息；Denoising U-Net 接收噪声潜变量和布局引导进行扩散处理。在此基础上叠加 HOI 复原模块用于手部结构恢复和纹理精炼。采用两阶段训练：第一阶段图像级 HOI 建模，第二阶段时序一致性建模。

### 关键设计1: 分离式布局表示 — 解耦手和物体的位置信息

**功能**: 通过不同特性的边界框表示实现手物解耦，使模型能适应不同物体的跨物体重演。

**核心思路**: 每帧的布局表示由三个检测边界框组成：两个**固定大小的正方形**手部框（仅提供位置信息，姿态无关、尺寸无关）和一个**可变大小**的物体框（根据物体和深度变化）。通过 4 层轻量卷积编码器提取布局特征 $\mathbf{F}_l$，与高斯噪声 $\epsilon_t$ 组合输入 Denoising U-Net。

**设计动机**: 手部框的姿态/尺寸无关性将位置信息与运动信号解耦，允许模型关注手部位置而非绑定特定手势；物体框的可变大小提供了形状和位置的指导。这种稀疏但有效的解耦为后续更精细的 HOI 建模奠定基础。

### 关键设计2: HOI 交互纹理增强模块 — 双记忆库恢复手和物体细节

**功能**: 通过独立的可学习记忆库分别恢复手部姿态和物体纹理的精细细节。

**核心思路**: 构建手部记忆库 $\mathbf{B}_h \in \mathbb{R}^{N_h \times C_h}$ 和物体记忆库 $\mathbf{B}_o \in \mathbb{R}^{N_o \times C_o}$，在 U-Net 架构中集成 Hand-Attention 和 Object-Attention 层。注意力计算使用对应的手/物体掩码 $M$ 限制作用区域：

$$\mathbf{F}^a = \text{Att}(\mathbf{F}, \mathbf{B}, \mathbf{B}) * M + \mathbf{F}$$

同时使用 ControlNet 编码 3D 手部网格 $V^h$（HaMeR 重建）进行结构引导，并在训练中对手部位置进行随机偏移增强以消除过度依赖。

**设计动机**: 仅靠布局和参考图像不足以恢复手指细节和物体纹理。全局记忆库在训练中积累了丰富的手部姿态和物体纹理先验，推理时可有效补充缺失信息。

### 关键设计3: 自适应布局调整策略 — 处理跨物体尺寸差异

**功能**: 在推理阶段自适应调整布局，避免因目标物体与源物体尺寸差异导致的不合理物理接触。

**核心思路**: 四步调整流程：(1) 初始化物体框四边中心为潜在接触点，通过 H2O 距离（手部框中心到最近接触点的距离）判断接触关系；(2) 保持物体框中心不变，按目标物体尺寸调整宽高；(3) 水平调整手部框位置以保持原始 H2O 距离；(4) 移动物体框使底部与原始框底部一致，避免悬浮效果。

**设计动机**: 当目标物体与源物体尺寸差异显著时，直接使用源运动序列会导致手部无法正确接触物体或抓握位置不合理。自适应调整确保了物理交互的合理性。

### 损失函数

基于 Stable Diffusion 的标准噪声预测损失 $\mathbf{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, c, t)\|_2^2]$，训练末期每 10 次迭代仅计算手部和物体区域的 $L_1$ 损失以强调 HOI 区域。

## 实验关键数据

### 主实验: 跨物体重演 (Our dataset)

| 方法 | Hand Fid. ↑ | Subj. Cons. ↑ | Mot. Smth. ↑ |
|------|------------|--------------|-------------|
| AnyV2V | 0.934 | 0.829 | 0.983 |
| VideoSwap | 0.936 | 0.922 | 0.992 |
| HOI-Swap | 0.994 | 0.944 | 0.994 |
| **Re-HOLD** | **0.994** | **0.955** | **0.994** |

### 自重建 (Self-Reenactment)

| 方法 | PSNR ↑ | FID ↓ | Hand Agr. ↑ |
|------|--------|-------|------------|
| HOI-Swap | 31.634 | 30.932 | 0.725 |
| RealisDance | 32.784 | 26.337 | 0.749 |
| **Re-HOLD** | **33.451** | **19.021** | **0.773** |

### 用户评估 (1-5分)

| 方法 | HOI一致性 | 物体一致性 | 时序一致性 |
|------|----------|----------|----------|
| HOI-Swap | - | - | - |
| **Re-HOLD** | **最高** | **最高** | **最高** |

### 关键发现

- Re-HOLD 在 FID 上显著优于所有方法（19.021 vs 次优 26.337），图像质量大幅提升
- 手部保真度和手部一致性均为最优，证明 HOI 纹理增强模块有效
- 自适应布局调整使得即使物体尺寸差异很大也能生成合理的手物交互

## 亮点与洞察

1. **分离式布局表示的简洁优雅**：用三个边界框解耦手和物体，手部框的姿态/尺寸无关性是关键创新
2. **两阶段训练策略务实高效**：先图像级 HOI 建模，再时序一致性，降低了训练难度
3. **自适应布局调整策略解决了跨物体泛化的核心挑战**
4. **首个以人为中心的双手 HOI 视频重演框架**，填补了领域空白

## 局限与展望

- 训练数据仅 9 个主体 14 个物体，泛化到更多样化的场景可能受限
- 依赖 HaMeR 的 3D 手部重建质量，估计失败时会影响生成效果
- 物体分割依赖 LISA 模型，复杂背景下可能不准确
- 未来可探索结合物理引擎约束更真实的抓握力学

## 相关工作与启发

- **HOI-Swap**: 扩散模型的视频物体替换，但仅支持单手以物体为中心的场景
- **AnimateAnyone / RealisDance**: 人体动画方法，但不处理手物交互
- **HaMeR**: 提供 3D 手部网格估计，为手部结构引导提供基础
- **InteractDiffusion**: 布局输入捕获交互关系的启发来源

## 评分

⭐⭐⭐⭐ — 完整解决了 HOI 视频重演的核心技术挑战，布局解耦和自适应调整策略设计巧妙。实验数据全面，在多项指标上显著超越现有方法。但训练数据规模有限，泛化能力有待更大规模验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)
- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)

</div>

<!-- RELATED:END -->
