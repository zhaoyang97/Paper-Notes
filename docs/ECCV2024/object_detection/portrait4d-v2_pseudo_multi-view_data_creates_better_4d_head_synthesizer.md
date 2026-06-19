---
title: >-
  [论文解读] Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer
description: >-
  [ECCV 2024][目标检测][4D头部合成] 提出一种利用**伪多视角视频**来训练前馈式单图4D头部合成器的新学习范式：先用合成数据学一个3D头部合成器将单目视频转为多视角，再利用伪多视角视频通过**跨视角自重演**学习4D合成器，避免了对3DMM的过度依赖，在重建保真度、几何一致性和运动控制精度上大幅超越先前方法。
tags:
  - "ECCV 2024"
  - "目标检测"
  - "4D头部合成"
  - "单图驱动"
  - "伪多视角"
  - "NeRF"
  - "人脸重演"
---

# Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer

**会议**: ECCV 2024  
**arXiv**: [2403.13570](https://arxiv.org/abs/2403.13570)  
**代码**: 无（项目页面: [https://yudeng.github.io/Portrait4D-v2/](https://yudeng.github.io/Portrait4D-v2/)）  
**领域**: 目标检测  
**关键词**: 4D头部合成, 单图驱动, 伪多视角, NeRF, 人脸重演

## 一句话总结

提出一种利用**伪多视角视频**来训练前馈式单图4D头部合成器的新学习范式：先用合成数据学一个3D头部合成器将单目视频转为多视角，再利用伪多视角视频通过**跨视角自重演**学习4D合成器，避免了对3DMM的过度依赖，在重建保真度、几何一致性和运动控制精度上大幅超越先前方法。

## 研究背景与动机

单图头部化身合成（one-shot head avatar synthesis）是近年来的热门研究方向，旨在给定一张源图像和驱动动作序列，生成逼真的人像视频。技术范式已从2D生成模型（直接生成2D帧）向可动画的3D（即4D）合成器转变，后者能保持更好的几何一致性并支持自由视角渲染。

3D方法学习面临的核心困难：
- **3D数据稀缺**：缺乏大规模多视角视频数据，从单目视频学习3D重建是高度病态的
- **3DMM依赖**：现有方法依赖3D Morphable Models来辅助几何和运动学习，但3DMM重建精度有限、表达力不足
- **合成-真实差距**：Portrait4D通过4D GAN生成合成数据，但表情仍受限于3DMM描述空间

作者的关键洞察：如果先学一个**静态的3D头部合成器**（仅需合成的静态多视角图像），它可以泛化到真实图像，并能将单目视频的每帧转为多视角。这样就能用伪多视角视频来训练4D合成器，同时获得：(1) 单目视频的高扩展性和丰富表情；(2) 多视角数据的几何学习能力；(3) 超越3DMM的运动表示。

## 方法详解

### 整体框架

方法分为两阶段：

**第一阶段**：禁用运动相关层，在合成数据上训练静态3D头部合成器Ψ3d（novel view synthesis）

**第二阶段**：激活运动相关组件，利用真实视频的伪多视角版本通过跨视角自重演学习完整的4D合成器Ψ

### 关键设计

1. **基于ViT的Triplane重建器**：采用Portrait4D的backbone架构，两个CNN编码器分别提取全局和细节外观特征，全局特征送入ViT blocks进行姿态正则化，再与细节特征拼接送入ViT解码器生成triplane T。运动信息通过motion-aware cross-attention层注入：前几个block的cross-attention接收源图像的运动嵌入vs用于中性化，后续block接收驱动图像的运动嵌入vd用于重演。运动嵌入通过预训练的2D运动编码器Emot提取。

2. **3D合成器与多视角视频生成**：将完整重建器中的cross-attention层禁用，得到静态3D合成器Ψ3d。使用GenHead（Portrait4D的4D GAN）生成合成多视角数据训练Ψ3d，学习目标为从输入图像重建同一3D头在任意视角下的图像。训练损失包含7项：
    $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_{LPIPS} + \mathcal{L}_{id} + \mathcal{L}_{adv} + \mathcal{L}_{depth} + \mathcal{L}_{opa} + \mathcal{L}_T$
   
   关键发现：虽然Ψ3d只在静态图像上训练，但它能在视频各帧间保持几何一致性，且比Portrait4D更好地保留真实帧的细节形状和表情。这是因为静态重建利用了更少的3DMM归纳偏置。

3. **跨视角自重演学习**：核心训练策略。每次迭代：

    - 从真实视频采样源帧Îs
    - 从伪多视角驱动视频采样两个不同视角的同一运动帧Id(θp)和Id(θq)
    - 以Îs为外观输入、从Id(θp)提取运动嵌入，让Ψ在θq视角生成重演结果
    - 训练损失：$\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_{LPIPS} + \mathcal{L}_{id} + \mathcal{L}_{adv}$
   
   **替换策略**：由于Ψ3d并不完美，以10%概率将Id(θp)替换为真实驱动帧Îd，以80%概率将Id(θq)替换为Îd。这使得模型主要学习重建真实帧，伪多视角帧更多起几何正则化作用。当使用真实帧作为GT时用全部4项损失，否则仅用LPIPS。

   **跨视角学习的效果**：(1) 将3D合成器的几何和外观先验蒸馏到4D合成器中；(2) 使用非3D感知的运动嵌入成为可能——通过从任意视角的驱动帧提取运动嵌入，姿态信息被挤出，实现了表情和姿态的解耦。

### 损失函数 / 训练策略

- 3D阶段：Adam优化器，lr=1e-4，batch=32，训练至看到10M张图像
- 4D阶段：在VFHQ的10K视频片段上训练，训练至看到6M张图像。运动相关层lr=2.5e-4，其余lr=1e-4
- 4D模型从3D模型权重初始化（除运动相关组件随机初始化）

## 实验关键数据

### 主实验

**自重演与跨身份重演（VFHQ, 512²分辨率）**

| 方法 | Self-LPIPS↓ | Self-FID↓ | Self-ID↑ | Self-AED↓ | Cross-FID↓ | Cross-ID↑ |
|------|-------------|-----------|----------|-----------|------------|-----------|
| Face-vid2vid | 0.289 | 52.0 | 0.826 | 0.028 | 79.2 | 0.606 |
| Real3DPortrait | 0.283 | 44.6 | 0.802 | 0.029 | 61.3 | 0.721 |
| Portrait4D | 0.320 | 43.0 | 0.773 | 0.033 | 54.8 | 0.620 |
| **Ours** | **0.224** | **26.3** | **0.873** | **0.019** | **48.5** | **0.736** |

**用户研究（22名参与者）**

| 方法 | 图像质量偏好↑ | 表情准确度偏好↑ |
|------|-------------|---------------|
| PIRenderer | 0.47% | 1.40% |
| GPAvatar | 6.28% | 14.7% |
| Portrait4D | 11.6% | 6.28% |
| **Ours** | **80.7%** | **71.6%** |

### 消融实验

| 配置 | Init. | Id(θp) | Id(θq) | Self-LPIPS↓ | Cross-ID↑ | Cons.↑ |
|------|-------|--------|--------|-------------|-----------|--------|
| A (随机init, 无跨视角) | Rand. | ✗ | ✗ | 0.216 | 0.776 | 0.822 |
| B (3D init, 无跨视角) | Ψ3d | ✗ | ✗ | 0.213 | 0.718 | 0.813 |
| E (随机init, 双跨视角) | Rand. | ✓ | ✓ | 0.232 | 0.777 | 0.929 |
| F (3DMM运动) | Ψ3d | ✓ | ✓ | 0.221 | 0.704 | 0.811 |
| K (**Ours**) | Ψ3d | ✓ | ✓ | 0.224 | 0.743 | 0.908 |

关键消融发现：
- 3D初始化至关重要（配置A vs K），提供几何先验
- 跨视角驱动帧对几何一致性(Cons.)提升显著（配置B vs K）
- Emot运动编码器优于3DMM（配置F vs K），表情更丰富
- 数据量有显著影响：从1K到10K视频，性能持续提升

### 关键发现

- 推理速度10 FPS（缓存源特征后可达15 FPS）
- 即使在艺术风格人像上也能运作
- 3D合成器虽用静态图训练，但在视频帧间保持了良好的几何一致性
- 跨视角学习有效地将3D先验蒸馏到4D合成器中，同时保持了详细的运动控制

## 亮点与洞察

- **学习范式创新**：不直接生成4D数据或依赖3DMM，而是巧妙地通过3D合成器"升维"单目视频为多视角，再反过来学4D合成器。思路优雅且实用
- **解耦策略精妙**：通过跨视角驱动帧提取运动嵌入来自然解耦姿态和表情
- **替换概率设计**：80%/10%的替换策略在真实数据保真度和几何正则化之间取得了很好的平衡
- **结果质量优异**：用户研究中80.7%的质量偏好率远超所有竞争方法

## 局限与展望

- 推理速度（10-15 FPS）距离实时应用仍有差距
- 3D合成器Ψ3d是在FFHQ上通过GenHead学习的，合成数据与真实数据仍存在域差异
- 颈部以下的身体部分未建模
- 对极端光照和遮挡的鲁棒性未充分评估
- 方法的核心优势来自学习范式而非网络架构，未来可结合更先进的backbone进一步提升

## 相关工作与启发

- 与Portrait4D属于同一技术路线，但通过伪多视角数据避免了4D GAN的3DMM局限
- 思路可泛化到其他需要3D prior的单目重建问题（如全身化身、手部重建等）
- 3D→4D的知识蒸馏范式对生成模型的训练数据获取有启发意义
- DiffusionNet等拓扑无关方法（如本批次另一篇ScanTalk）代表了不同方向的突破

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（学习范式非常新颖）
- 实验充分度: ⭐⭐⭐⭐⭐（量化、定性、用户研究、消融齐全）
- 写作质量: ⭐⭐⭐⭐⭐（思路清晰、motivation充分）
- 价值: ⭐⭐⭐⭐☆（对4D头部合成领域有重要推动）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)
- [\[ECCV 2024\] Plain-Det: A Plain Multi-Dataset Object Detector](plain-det_a_plain_multi-dataset_object_detector.md)
- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [\[ECCV 2024\] Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)

</div>

<!-- RELATED:END -->
