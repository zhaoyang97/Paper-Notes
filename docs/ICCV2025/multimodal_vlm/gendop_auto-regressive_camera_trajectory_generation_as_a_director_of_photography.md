---
title: >-
  [论文解读] GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography
description: >-
  [ICCV 2025][多模态VLM][相机轨迹生成] 提出 DataDoP 数据集（29K 真实电影镜头的自由运动相机轨迹+描述）和 GenDoP 自回归 Transformer 模型，通过文本和/或 RGBD 输入生成艺术化、高质量的相机运动轨迹，在可控性、运动稳定性和复杂度上超越现有方法。 相机运动设计是视频制作的核心…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "相机轨迹生成"
  - "自回归模型"
  - "电影摄影"
  - "多模态条件生成"
  - "数据集构建"
---

# GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography

**会议**: ICCV 2025  
**arXiv**: [2504.07083](https://arxiv.org/abs/2504.07083)  
**代码**: [项目主页](https://kszpxxzmc.github.io/GenDoP/)  
**领域**: 多模态VLM  
**关键词**: 相机轨迹生成, 自回归模型, 电影摄影, 多模态条件生成, 数据集构建

## 一句话总结

提出 DataDoP 数据集（29K 真实电影镜头的自由运动相机轨迹+描述）和 GenDoP 自回归 Transformer 模型，通过文本和/或 RGBD 输入生成艺术化、高质量的相机运动轨迹，在可控性、运动稳定性和复杂度上超越现有方法。

## 研究背景与动机

相机运动设计是视频制作的核心要素，专业摄影指导需要精心设计运镜来传达导演意图。然而现有方法存在明显局限：

**传统方法**：依赖几何优化或手工设计的程序化系统，每种运动都需要单独的几何建模或代价函数工程，限制了创意合成能力

**基于扩散的方法**：CCD 和 E.T. 使用人体追踪数据集训练，继承了以人物为中心的运动偏差；Director3D 虽然引入了物体/场景中心轨迹，但缺少轨迹级别的描述文本，导致生成的路径由几何合理性驱动而非导演意图

**数据集缺失**：现有数据集（RealEstate10K、MVImgNet 等）要么聚焦于简单的物体/场景中心路径，要么局限于追踪运动，缺乏自由运动（free-moving）的艺术化轨迹和对应的导演意图描述

核心洞察：相机轨迹天然具有**序列依赖性**——每一帧的位姿都依赖前一帧，因此自回归模型比扩散模型更适合建模这种时序连续性。

## 方法详解

### 整体框架

GenDoP 采用"数据集+模型"双管齐下的策略：
- **DataDoP 数据集**：从艺术电影中提取 29K 个 free-moving 相机轨迹，配套 RGBD 图像和两种描述文本（Motion Caption + Directorial Caption）
- **GenDoP 模型**：基于 decoder-only Transformer 的自回归生成模型，输入文本描述（可选首帧 RGBD），输出离散化的相机轨迹 token 序列

### 关键设计

1. **DataDoP 数据集构建流程**：

    - **预处理**：从电影视频中通过 PySceneDetect 分割镜头，去除字幕，过滤 10-20 秒的片段，用 GPT-4o 筛除静态和追踪镜头，保留 free-moving 类型
    - **轨迹提取**：使用 MonST3R 估计动态场景几何，提取相机轨迹和深度图，经清洗、平滑、插值为固定长度序列
    - **运动标注**：将轨迹分段标注运动标签，包含 27 种平移组合 × 7 种旋转组合
    - **描述生成**：基于运动标签用 GPT-4o 生成 Motion Caption（纯运动描述）；再结合视频帧 4×4 网格生成 Directorial Caption（运动+场景交互+导演意图）

2. **相机轨迹 Token 化**：将连续相机参数离散化为 token 序列

    - **正则化归一化**：以首帧相机坐标系为世界参考，$\mathbf{R}_0^{\mathrm{norm}} = \mathbf{I}$，$\mathbf{t}_0^{\mathrm{norm}} = \mathbf{0}$，后续位姿做相对变换和尺度归一化
    - **参数离散化**：旋转矩阵转四元数 $(r_1, r_2, r_3, r_4)$，加上平移 $(t_1, t_2, t_3)$、焦距 $(f_1, f_2)$ 和尺度 $s$，共 10 维，归一化到 $[0,1]$ 后乘以 bin 数量 $B=256$ 取整
    - 每条轨迹被编码为长度 $10N$ 的整数序列（$N=60$ 帧），加上 BOS/EOS/PAD 辅助 token

3. **多模态条件编码器**：

    - **文本编码器** $\mathcal{E}_T$：使用 Stable Diffusion 2.1 的可训练文本编码器 + MLP，输出 $\mathbf{Z}_T \in \mathbb{R}^{77 \times 1024}$
    - **RGB 编码器** $\mathcal{E}_I$ 和**深度编码器** $\mathcal{E}_D$：均使用可训练 CLIP Vision Model + MLP，分别输出 $\mathbf{Z}_I, \mathbf{Z}_D \in \mathbb{R}^{257 \times 1024}$
    - 最终拼接：$\mathbf{Z} = [\mathbf{Z}_T; \mathbf{Z}_I; \mathbf{Z}_D]$

4. **自回归解码器**：采用 OPT 架构（12 层，12 头），条件 latent code 拼接在 BOS 前面，通过因果自注意力逐 token 预测轨迹

### 损失函数 / 训练策略

$$L = \mathrm{CrossEntropy}(S[1:], \hat{S}[:,-1]) + \lambda \|\mathbf{Z}\|_2^2$$

- 交叉熵损失：ground truth token 序列与预测 logits 的分类损失
- 正则化项：条件 latent code 的 L2 正则（$\lambda = 10^{-8}$）
- 优化器：AdamW，lr=1e-5，weight decay=0.01，bfloat16 混合精度
- 训练：单卡 A100 约 8 小时收敛，推理 ~3 秒/条轨迹

## 实验关键数据

### 主实验

文本条件轨迹生成（Motion Caption）：

| 方法 | 数据集 | F1-Score↑ | CLaTr-CLIP↑ | Coverage↑ | CLaTr-FID↓ |
|------|--------|-----------|-------------|-----------|------------|
| CCD | Pre-trained | 0.297 | 5.288 | 0.332 | 357.822 |
| E.T. | Pre-trained | 0.330 | 2.450 | 0.020 | 609.906 |
| Director3D | DataDoP | 0.391 | 31.689 | 0.839 | 31.979 |
| **GenDoP** | **DataDoP** | **0.400** | **36.179** | **0.872** | **22.714** |

用户研究（AUR，5 分制）：

| 方法 | Alignment↑ | Quality↑ | Complexity↑ |
|------|-----------|----------|-------------|
| CCD | 3.013 | 3.022 | 3.273 |
| Director3D (DataDoP) | 3.753 | 3.260 | 3.493 |
| **GenDoP** | **4.693** | **4.573** | **4.713** |

### 消融实验

| 配置 | F1-Score | CLaTr-CLIP | Coverage | CLaTr-FID | 说明 |
|------|---------|------------|----------|-----------|------|
| 完整模型 | 0.400 | 36.179 | 0.872 | 22.714 | 可训练编码器+正则归一化 |
| 去掉正则归一化 | 0.322 | 14.917 | 0.766 | 68.590 | CLaTr-CLIP 降 59%，质量严重下降 |
| 冻结编码器 | 0.389 | 31.420 | 0.866 | 22.841 | 对齐能力下降 |

超参数消融（离散 bins 256 最优，轨迹长度 30 最优，base 模型最优）。

### 关键发现

- 相同数据集上，自回归模型 GenDoP 比扩散模型 Director3D 在 CLaTr-CLIP 上高 4.5-9.1 分，验证了 AR 框架对轨迹生成的优势
- RGBD + Text 条件能有效消除文本描述中的空间歧义（如"向左前方移动"的具体方向）
- 正则归一化（将场景中心轨迹转为第一人称路径）对性能提升至关重要
- 可训练编码器优于冻结编码器，因为需要学习运动语义到轨迹的跨模态映射

## 亮点与洞察

- **自回归 vs 扩散**：体现了对任务特性的深入理解——轨迹的序列依赖性使 AR 天然优于扩散模型在连续性和稳定性上的表现
- **三层描述设计**（运动标签→Motion Caption→Directorial Caption）：逐步从底层运动到高层意图，是一个优雅的多粒度标注方案
- **数据集价值**：即使是同一个 Director3D 模型，在 DataDoP 上训练后 CLaTr-CLIP 从 0 跃升至 30+，说明数据集本身的贡献巨大

## 局限与展望

- 目前仅使用首帧 RGBD 作为视觉条件，未利用数据集中提取的 4D 点云信息
- 轨迹生成与视频生成是分离的两步流程，未来可统一为端到端系统
- 正则归一化丢失了绝对尺度信息，不适用于需要精确物理尺度的场景

## 相关工作与启发

- DataDoP 的构建范式（视频→轨迹提取→运动标注→LLM 描述）可迁移到其他结构化序列的数据集构建
- 相机参数的 Token 化方案（正则化→四元数→离散化）值得在 3D 生成领域推广

## 评分

- 新颖性：⭐⭐⭐⭐ （首次将自回归模型用于相机轨迹生成，数据集设计新颖）
- 技术深度：⭐⭐⭐⭐ （Token 化设计精巧，多条件融合合理）
- 实验充分度：⭐⭐⭐⭐ （定量+用户研究+消融+定性分析覆盖全面）
- 实用价值：⭐⭐⭐⭐ （直接可应用于 AI 视频生成的运镜控制）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation](../../CVPR2026/multimodal_vlm/onecat_decoder-only_auto-regressive_model_for_unified_understanding_and_generati.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[ICCV 2025\] Training-free Generation of Temporally Consistent Rewards from VLMs](training-free_generation_of_temporally_consistent_rewards_from_vlms.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)

</div>

<!-- RELATED:END -->
