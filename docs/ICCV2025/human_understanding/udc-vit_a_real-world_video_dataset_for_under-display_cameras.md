---
title: >-
  [论文解读] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras
description: >-
  [ICCV 2025][人体理解][屏下摄像头] 提出首个真实世界屏下摄像头（UDC）视频数据集 UDC-VIT，包含 647 个视频片段共 116,460 帧，通过精心设计的双摄像头-分光器采集系统实现精确的时空对齐，并以人脸识别为核心应用场景，揭示了合成数据集在模拟真实 UDC 退化方面的不足。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "屏下摄像头"
  - "视频数据集"
  - "图像退化"
  - "人脸识别"
  - "视频复原"
---

# UDC-VIT: A Real-World Video Dataset for Under-Display Cameras

**会议**: ICCV 2025  
**arXiv**: [2501.18545](https://arxiv.org/abs/2501.18545)  
**代码**: [GitHub](https://github.com/mcrl/UDC-VIT)  
**领域**: 人体理解  
**关键词**: 屏下摄像头, 视频数据集, 图像退化, 人脸识别, 视频复原

## 一句话总结

提出首个真实世界屏下摄像头（UDC）视频数据集 UDC-VIT，包含 647 个视频片段共 116,460 帧，通过精心设计的双摄像头-分光器采集系统实现精确的时空对齐，并以人脸识别为核心应用场景，揭示了合成数据集在模拟真实 UDC 退化方面的不足。

## 研究背景与动机

屏下摄像头（UDC）将摄像头置于显示屏下方，实现了全面屏设计，已被三星 Galaxy Z-Fold 系列和 ZTE Axon 系列等手机采用。然而，显示面板造成的光衍射会引入严重退化，包括透射率降低、模糊、噪声和光斑（flare）等问题。

**现有数据集的关键不足**：

**合成数据集的局限**：
   - T-OLED/P-OLED 在受控环境中通过显示器显示图片采集，动态范围有限导致几乎没有 flare
   - SYNTH 通过测量的 PSF 卷积生成，但缺少噪声和空间变化的 flare
   - VidUDC33K 通过 PSF 卷积模拟视频退化，但 flare 过于规则且与光源无关，还存在不合理的白色伪影

**缺乏真实视频数据集**：现有真实数据集如 UDC-SIT 仅包含静态图像。视频比图像多了时序维度，涉及运动引起的时间变化 flare，这在合成数据中无法准确模拟。

**缺乏人脸识别数据**：现有数据集中的人物主要是远景或背面，无法支持面部识别研究。

本文构建了首个包含真实 UDC 退化的视频数据集 UDC-VIT，分辨率 1900×1060，60fps，且 64.6% 的视频包含 22 位受试者的正面人体动作。

## 方法详解

### 整体框架

论文的核心贡献是数据集采集系统的设计和数据集的构建，包括硬件设计（双摄像头+分光器）、软件同步、帧对齐和质量评估。

### 关键设计

1. **双摄像头视频采集系统**:

    - 功能：同时采集同一场景的 UDC 退化视频和干净参考视频
    - 核心思路：使用非偏振立方分光器（Thorlabs CCM1-BS013）将入射光以 50:50 比例分成两路，分别送入两个 Arducam Hawk-Eye（IMX686）摄像头模块。其中一路前方放置从 Samsung Galaxy Z-Fold 5 切割的 UDC 区域显示面板，产生退化。两个摄像头通过 Raspberry Pi 5 的双四通道 MIPI 接口连接，使用 MPI barrier 实现帧同步，精度达到 8ms 以内。每个摄像头安装在 Thorlabs K6XS 六轴精动光学支架上，可进行平移、旋转和倾斜调整以对齐视场。
    - 设计动机：使用与 Galaxy Z-Fold 5 相同的 Quad Bayer Coding（QBC）传感器，确保退化特性一致。分光器方案比双摄像头方案（如 Pseudo-real 数据集）具有更好的几何对齐基础。

2. **基于 DFT 的帧对齐**:

    - 功能：修正采集过程中不可避免的像素位置偏差
    - 核心思路：利用离散傅里叶变换（DFT）进行退化鲁棒的对齐。GT 帧中心裁剪至 1900×1060，退化帧通过迭代平移和旋转最小化对齐损失：
    $\mathcal{L} = \lambda_1 \sum_{x,y} (\mathcal{D}(x,y) - \mathcal{G}(x,y))^2 + \lambda_2 \sum_{u,v} \Delta\mathcal{F}_{amp}(u,v) + \lambda_3 \sum_{u,v} \Delta\phi(u,v)$
      其中第一项是空域 MSE，后两项分别是频域振幅和相位的 L1 距离，$\lambda_1 = \lambda_3 = 1, \lambda_2 = 0$
    - 设计动机：传统对齐方法（SIFT、RANSAC）在严重 UDC 退化（尤其是 flare）下性能不佳，DFT 对退化更鲁棒。

3. **数据集特性与真实退化分析**:

    - 功能：系统对比 UDC-VIT 与现有数据集在噪声、透射率、flare 等方面的差异
    - 核心发现：
        - **噪声和透射率**：VidUDC33K 中退化帧的噪声水平反而低于 GT（不合理），而 UDC-VIT 正确呈现了 UDC 区域低透射率导致的信号放大和噪声增加
        - **空间变化 flare**：UDC 退化从镜头中心向外逐渐加重，导致 flare 具有空间变化性。VidUDC33K 对整个图像使用相同 PSF 卷积，无法呈现此特性
        - **光源变化 flare**：不同光源（LED、卤素灯、自然光）产生不同形状的 flare，VidUDC33K 无法模拟
        - **时间变化 flare**：摄像头运动导致 PSF 变化，UDC-VIT 自然捕捉此特性，而 VidUDC33K 模拟几乎无效

### 损失函数 / 训练策略

本文是数据集论文，不涉及新模型训练。在评估使用的六个深度学习模型中，均使用各自原始的训练策略。

## 实验关键数据

### 主实验

六个深度学习模型在 VidUDC33K 和 UDC-VIT 上的复原性能：

| 模型 | VidUDC33K PSNR↑ | VidUDC33K SSIM↑ | UDC-VIT PSNR↑ | UDC-VIT SSIM↑ | UDC-VIT LPIPS↓ |
|------|----------------|----------------|--------------|--------------|---------------|
| Input | 26.22 | 0.8524 | 16.26 | 0.7366 | 0.4117 |
| DISCNet | 28.89 | 0.8405 | 24.70 | 0.8403 | 0.2675 |
| UDC-UNet | 28.37 | 0.8361 | **28.00** | **0.8911** | **0.1779** |
| FastDVDNet | 28.95 | 0.8638 | 23.89 | 0.8439 | 0.2662 |
| EDVR | 28.71 | 0.8531 | 23.55 | 0.8331 | 0.2673 |
| ESTRNN | 29.54 | 0.8744 | 25.38 | 0.8654 | 0.2216 |
| DDRNet | **31.91** | **0.9313** | 24.68 | 0.8539 | 0.2218 |

DDRNet 在合成数据上表现最佳（31.91 dB），但在真实数据上仅 24.68 dB，揭示了合成训练数据的局限性。

### 消融实验

人脸识别准确率随复原质量的变化：

| 条件 | PSNR | 人脸识别准确率 | 说明 |
|------|------|-------------|------|
| Input（退化帧） | 16.31 | 64.5% | 未复原 |
| DISCNet 复原 | ~24.7 | ~75% | 静态图像模型 |
| UDC-UNet 复原 | 27.74 | 82.2% | 最佳复原模型 |
| GT（参考帧） | ∞ | 90.3% | 上界 |

对齐质量（PCK 对比）：

| 数据集 | 对齐方法 | PCK₀.₀₀₃ | PCK₀.₀₁ | PCK₀.₁₀ |
|--------|---------|----------|---------|---------|
| Pseudo-real | AlignFormer | N/A | 58.75 | 99.93 |
| UDC-SIT | DFT | 93.67 | 97.26 | 99.35 |
| **UDC-VIT** | **DFT** | **92.12** | **98.95** | **99.69** |

### 关键发现

- **合成数据训练的模型在真实数据上不可靠**：DDRNet 在合成 VidUDC33K 上最优（31.91 dB），但在真实 UDC-VIT 上排名下降，被 UDC-UNet 大幅超过
- **复原质量与人脸识别强相关**：PSNR 从 16.31 提升到 27.74 时，人脸识别准确率从 64.5% 提升到 82.2%
- **性能排名不一致**：两个数据集上的模型排名不同，说明合成数据无法准确反映真实退化特征
- **残差连接对帧一致性有益**：UDC-UNet 和 ESTRNN 使用残差 CNN，在减少闪烁方面表现更好
- VidUDC33K 存在不合理的场景（如海水上的 flare、鸟嘴上的 flare），以及错误的 PSF 变换导致的黑帧和颜色失真

## 亮点与洞察

1. **硬件-软件协同设计的采集系统**：六轴精动支架 + 分光器 + MPI 同步的方案堪称精巧，8ms 同步精度对于视频采集已非常高
2. **真实 flare 特性的系统分析**：空间变化、光源变化、时间变化三种 flare 特性的对比分析非常有说服力，清楚展示了合成方法的根本缺陷
3. **应用驱动的数据集设计**：专门面向人脸识别场景设计（64.6% 视频含人脸），并量化了复原质量对人脸识别的影响
4. **数据集公开可用**：通过 GitHub 仓库开放下载

## 局限与展望

- 仅针对 Samsung Galaxy Z-Fold 5 的 UDC 面板，其他设备（ZTE Axon 系列或其他 Fold 系列）的退化特性不同，需要迁移学习
- 排除了快速运动物体（如行驶车辆），限制了数据集在高速场景下的适用性
- UDC 复原本质上是设备相关的（取决于光学、传感器和面板设计），通用软件方案的设计仍是开放问题
- 22 位受试者的人脸数据量相对有限，未来可扩大规模

## 相关工作与启发

- 分光器采集系统的设计范式可推广到其他需要匹配退化/干净对的数据集构建
- DFT 对齐方法对严重退化场景的鲁棒性值得在其他图像配准任务中探索
- 复原质量与下游任务（人脸识别）性能的量化关系为端到端优化提供了新的研究方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个真实UDC视频数据集，采集系统设计精巧
- 实验充分度: ⭐⭐⭐⭐ 6个模型+2个数据集对比全面，人脸识别评估有新意
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，现有数据集问题的分析深入直观
- 价值: ⭐⭐⭐⭐ 填补了真实UDC视频数据集的空白，对UDC领域有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios](../../CVPR2025/human_understanding/remote_photoplethysmography_in_real-world_and_extreme_lighting_scenarios.md)
- [\[ICCV 2025\] MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances](magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](../../ECCV2024/human_understanding/worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[NeurIPS 2025\] BEDLAM2.0: Synthetic Humans and Cameras in Motion](../../NeurIPS2025/human_understanding/bedlam20_synthetic_humans_and_cameras_in_motion.md)

</div>

<!-- RELATED:END -->
