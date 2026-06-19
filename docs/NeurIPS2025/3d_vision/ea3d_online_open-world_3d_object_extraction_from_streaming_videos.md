---
title: >-
  [论文解读] EA3D: Online Open-World 3D Object Extraction from Streaming Videos
description: >-
  [NeurIPS 2025][3D视觉][在线 3D 重建] 提出 EA3D（ExtractAnything3D），一个在线开放世界 3D 物体提取框架，通过知识集成特征图、在线视觉里程计和循环联合优化，从流式视频中同时进行几何重建和全面场景理解。 自主系统（如机器人）在未知环境中需要"边走边理解"——在线进行 3D 重建和…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "在线 3D 重建"
  - "开放世界场景理解"
  - "Gaussian Splatting"
  - "VLM"
  - "特征高斯"
  - "语义分割"
---

# EA3D: Online Open-World 3D Object Extraction from Streaming Videos

**会议**: NeurIPS 2025  
**arXiv**: [2510.25146](https://arxiv.org/abs/2510.25146)  
**代码**: [VDIGPKU/EA3D](https://github.com/VDIGPKU/EA3D)  
**领域**: 3D视觉  
**关键词**: 在线 3D 重建, 开放世界场景理解, Gaussian Splatting, VLM, 特征高斯, 语义分割  

## 一句话总结

提出 EA3D（ExtractAnything3D），一个在线开放世界 3D 物体提取框架，通过知识集成特征图、在线视觉里程计和循环联合优化，从流式视频中同时进行几何重建和全面场景理解。

## 背景与动机

自主系统（如机器人）在未知环境中需要"边走边理解"——在线进行 3D 重建和语义理解。现有方法存在三大瓶颈：

1. **离线限制**：NeRF/3DGS 方法需要完整多视角图像集和长时间优化，无法在线处理
2. **几何先验依赖**：许多 3D 场景理解方法需要预构建的点云/深度图/mesh
3. **2D-3D 不一致**：VLM 在 2D 表现出色但直接提升到 3D 时存在视角不一致、遮挡处理差等问题

核心思路：像人类感知一样，从流式视频第一帧开始就同时重建和理解，利用历史观测引导当前理解，新观测修正历史认知。

## 核心问题

如何从流式视频（无已知几何、位姿或语义标注）中在线同时进行 3D 几何重建和开放世界语义理解？

## 方法详解

### 知识集成特征图

**VLM 开放世界解释**：用 VLM 识别每帧中所有潜在物体及其语义，动态维护在线语义缓存 $\Omega$。语义嵌入为连续向量 $T \in \mathbb{R}^{1 \times V}$（CLIP 文本编码器）。

**语义特征图**：用 CLIP 视觉编码器和 Grounded-SAM 获取逐像素语义特征。计算类别相似度生成二值掩码，聚合特征后归一化：$\mathbf{S} = T \times \mathbf{f}_{sem}$。

**物理属性**：扩展文本提示从 VLM 提取物体级和部件级物理属性，编码为可学习向量 $\mathbf{Y}$。

**特征图嵌入**：在每个高斯原语上附加知识集成特征。集成特征图：

$$\mathbf{F}_t = \sum_{i,j} \mathbf{X}^{\text{self}}_{i,j} \cdot \mathbf{S}_{i,j}(\mathbf{T}_k; \mathbf{Y}_{i,j}) \cdot \mathbf{C}_t$$

相邻帧的匹配分布用于跨帧传播高斯特征：

$$\mathbf{M}_{t,t-1} = \text{Softmax}\left(\frac{\mathbf{F}_t \mathbf{F}_{t-1}^\top}{\|\mathbf{F}_t\| \|\mathbf{F}_{t-1}^\top\|}\right)$$

### 在线 3D 物体提取

**在线视觉里程计**：用 Cut3R + 深度估计初始化逐帧位姿、点图和置信度图，维护在线关键点图并通过局部 BA 优化修正累积位姿漂移。

**在线高斯更新**：逐帧增量添加特征高斯，精修已有几何并提取新物体。新观测区域从深度反投影初始化新高斯 $\mu_i$，共可见区域内高斯共享平移和旋转，减少冗余。同时应用单步分裂策略基于梯度自适应增长高斯。

### 循环联合优化

**语义感知自适应高斯正则**：

$$\mathcal{L}_\delta = \sum |\delta_i - \bar{\delta}| F_{sem}^q$$

鼓励同一类别的高斯具有相似尺度。

**联合语义-几何优化**：

$$\mathcal{L} = \sum_{t=0}^{t_{\text{now}}} \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_d + \lambda_3 \mathcal{L}_{kw} + \mathcal{L}_\delta$$

其中 $\mathcal{L}_1$ 为光度损失，$\mathcal{L}_d$ 为深度损失，$\mathcal{L}_{kw}$ 为特征图 L2 距离损失。参数设置 $\lambda_1=0.25$，$\lambda_2=0.1$，$\lambda_3=0.15$。

最终通过 alpha-blending 累积渲染特征：$\hat{F} = \sum_{i} F_i \cdot \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$。

## 实验关键数据

### ScanNet 多任务评估

| 方法 | 在线 | 免位姿 | PSNR↑ | SSIM↑ | mIoU↑ | mAcc↑ | AP↑ | mAP↑ |
|------|------|--------|-------|-------|-------|-------|-----|------|
| LangSplat | ✗ | ✗ | 18.4 | 0.69 | 27.5 | 51.3 | - | - |
| GaussianGrouping | ✗ | ✗ | 19.6 | 0.74 | 32.6 | 56.9 | 43.6 | 24.5 |
| FeatureGS | ✗ | ✗ | 23.9 | 0.84 | 41.1 | 66.0 | 51.4 | 32.7 |
| OpenScene | ✗ | ✗ | - | - | 42.8 | 68.6 | 55.7 | 34.8 |
| EmbodiedSAM | ✗ | ✗ | - | - | 44.2 | 71.4 | 58.1 | 39.5 |
| **EA3D** | **✓** | **✓** | **25.8** | **0.89** | **46.3** | **71.8** | **57.9** | **39.9** |

在线+免位姿条件下，重建和理解质量均超越离线方法。

### LERF 稀疏视角与在线稳定性

| 方法 | 在线 | 10 views PSNR | 70 views PSNR | 10 views mIoU | 70 views mIoU |
|------|------|--------------|--------------|---------------|---------------|
| FeatureGS | ✗ | 15.2 | 22.4 | 29.4 | 53.6 |
| OpenGaussian | ✗ | 14.9 | 22.7 | 30.1 | 55.8 |
| **EA3D** | **✓** | **21.9** | **23.2** | **53.8** | **57.4** |

仅 10 帧即获得远超离线方法的结果，展现出极强的稀疏视角鲁棒性。处理速度 0.235 FPS。

### 消融实验

去掉语义感知正则、在线里程计、联合优化任一组件都导致性能下降，验证了各模块的必要性。

## 亮点

1. **统一在线框架**：首个同时在线重建+开放世界理解的 3DGS 框架，无需预构建几何或位姿
2. **多任务能力**：单一框架支持渲染、语义/实例分割、3D 包围盒、语义占用、mesh 生成等多种下游任务
3. **知识传播机制**：通过匹配分布实现跨帧特征传播，确保知识的时间连续性
4. 稀疏视角下仍保持高质量，10 帧 PSNR 即达 21.9

## 局限与展望

- 处理速度仅 0.235 FPS，距实时在线应用仍有距离
- 需要 A100 80GB GPU，部署要求高
- VLM 的开放世界解释质量直接影响最终结果，对小物体可能不够准确
- 在线累积的位姿漂移在长序列中可能恶化
- 仅在室内数据集（ScanNet、LERF）上验证，缺少室外场景评估

## 与相关工作的对比

- **vs FeatureGS/OpenGaussian**：这些方法离线训练需要完整视角和已知位姿，EA3D 在线处理流式视频且免位姿
- **vs MonoGS+VFM**：SLAM 方法+VFM 后处理，关键帧追踪稀疏且需额外后优化，EA3D 一体化联合优化
- **vs EmbodiedOcc**：专为占用预测设计，无法实现照片级渲染；EA3D 多任务统一
- **vs HiCoM**：流式 3DGS 但需预计算位姿和多视角输入，EA3D 自主估计位姿和单视角输入

## 启发与关联

"在线+免位姿+开放世界"的三重约束极具挑战性和实用价值。知识集成特征图的设计——将 VLM 语义、VFM 视觉特征和物理属性统一编码到高斯原语中——为 3D 场景的全面理解提供了有效范式。循环联合优化中"几何引导理解、理解反哺几何"的思路值得推广。

## 评分

- ⭐ 新颖性: 9/10 — 首个在线免位姿开放世界 3D 物体提取框架，系统级创新
- ⭐ 实验充分度: 8/10 — 多数据集多任务评估+增强基线对比+消融，但缺少室外和大场景
- ⭐ 写作质量: 8/10 — 框架图清晰，动机阐述到位，但部分公式符号较密集
- ⭐ 价值: 9/10 — 高实用价值，直接面向机器人自主探索场景，多任务统一令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](towards_3d_objectness_learning_in_an_open_world.md)
- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](../../CVPR2025/3d_vision/depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [\[CVPR 2025\] Open-World Amodal Appearance Completion](../../CVPR2025/3d_vision/open-world_amodal_appearance_completion.md)
- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](../../CVPR2025/3d_vision/odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](online_segment_any_3d_thing_as_instance_tracking.md)

</div>

<!-- RELATED:END -->
