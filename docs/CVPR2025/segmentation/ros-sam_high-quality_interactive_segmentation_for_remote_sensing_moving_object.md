---
title: >-
  [论文解读] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object
description: >-
  [CVPR 2025][语义分割][遥感视频] ROS-SAM通过LoRA微调编码器、改进HQ解码器和重设计数据流水线，使SAM适配遥感视频运动目标的高质量交互分割任务，IoU提升13%并展现强零样本泛化能力。 遥感视频（卫星视频）中的运动目标分割是一个新兴但具有挑战性的研究方向。遥感视频中的目标（飞机、汽车、轮船、火车等）…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "遥感视频"
  - "交互分割"
  - "SAM微调"
  - "高质量分割"
  - "域适配"
---

# ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object

**会议**: CVPR 2025  
**arXiv**: [2503.12006](https://arxiv.org/abs/2503.12006)  
**代码**: [github.com/ShanZard/ROS-SAM](https://github.com/ShanZard/ROS-SAM)  
**领域**: 图像分割  
**关键词**: 遥感视频, 交互分割, SAM微调, 高质量分割, 域适配

## 一句话总结

ROS-SAM通过LoRA微调编码器、改进HQ解码器和重设计数据流水线，使SAM适配遥感视频运动目标的高质量交互分割任务，IoU提升13%并展现强零样本泛化能力。

## 研究背景与动机

遥感视频（卫星视频）中的运动目标分割是一个新兴但具有挑战性的研究方向。遥感视频中的目标（飞机、汽车、轮船、火车等）通常具有小尺寸、模糊特征、高密度和缺乏明确朝向等特点，逐帧标注成本极高。

SAM作为通用分割基础模型拥有强大的零样本能力，但直接应用于遥感数据时存在三个关键问题：
- **域差距**：遥感目标不受重力影响、方向模糊，SAM预测飞机方向时会生成通用的四角星形状
- **边界质量差**：SAM的mask decoder缺乏精细纹理和边缘信息，产生粗糙边界和碎片化mask
- **分辨率问题**：SAM要求输入固定为$1024 \times 1024$，下采样大分辨率遥感图像会导致小目标消失

一个自然的思路是利用现有的遥感目标跟踪数据集（含bounding box标注），通过交互分割将跟踪数据转化为分割数据，以最小成本推动遥感视频分割发展。

## 方法详解

### 整体框架

ROS-SAM基于SAM构建，包含三个核心改进：(1) LoRA微调图像编码器注入遥感领域知识；(2) HQ mask解码器融合多阶段特征生成高质量mask；(3) 重新设计的训练-推理流水线处理多尺度和高分辨率问题。训练时冻结SAM预训练参数，仅更新红色组件。

### 关键设计

**1. LoRA微调图像编码器 + 解冻最后一层**

- **功能**：注入遥感领域知识，同时保持SAM的通用泛化能力
- **核心思路**：对ViT编码器所有Transformer层的Query和Value矩阵注入低秩分解矩阵 $h = W_0 x + W_d W_e x$，其中 $r \ll \min(m,n)$。额外解冻编码器最后一个block以提取更具判别力的全局上下文特征
- **设计动机**：LoRA在不破坏预训练知识的前提下高效适配新域；解冻最后一层增强特征区分能力，使模型能区分相似物体（如飞机vs登机桥）。浅层捕获纹理细节，深层编码语义上下文

**2. HQ Mask解码器 + 交替优化策略**

- **功能**：融合图像编码器早期层的精细纹理特征和深层全局上下文特征，生成高质量分割mask
- **核心思路**：在SAM原始mask decoder基础上添加HQ-SAM的解码器分支，整合多阶段图像特征、prompt token和mask token进行高质量预测。不同于HQ-SAM冻结原始decoder，ROS-SAM**交替更新**SAM原始和HQ两个decoder
- **设计动机**：原始decoder直接更新会破坏预训练知识（消融实验显示IoU下降），而HQ decoder是新引入的轻量组件。交替更新让两个decoder互相学习，最终IoU从47.15提升到48.16

**3. 专用训练-推理数据流水线**

- **功能**：训练时引入多尺度多角度增强，推理时确保单目标高质量预测
- **核心思路**：训练阶段使用LSJ（Large Scale Jittering，尺度0.1~4.0倍）和随机旋转增强。推理阶段基于prompt位置中心裁剪 $N \times 512 \times 512$ 块，双三次插值上采样到 $1024 \times 1024$，单目标逐次推理后还原到原图位置
- **设计动机**：遥感目标不受重力约束无固定方向，需旋转增强；LSJ覆盖多尺度目标。2倍上采样推理最优，过度上采样导致锯齿边缘。仅推理流水线即可为SAM提升6% IoU

### 损失函数

使用Binary Cross-Entropy (BCE) Loss和Dice Loss的组合，交替更新SAM Mask和ROS-SAM Mask两部分权重。训练24个epoch，学习率1e-3。

## 实验关键数据

### 主实验：与SOTA方法对比（SAT-MTB数据集）

| 方法 | IoU | BIoU |
|------|-----|------|
| SAM（原始配置/+推理流水线） | 37.25 / 43.41 | 37.14 / 43.30 |
| SAM2（原始配置/+推理流水线） | 36.80 / 41.75 | 36.67 / 41.55 |
| HQ-SAM（原始配置/+推理流水线） | 43.27 / 47.15 | 43.21 / 47.11 |
| **ROS-SAM** | **50.54** | **50.36** |

### 消融实验：各模块贡献

| 方法 | IoU | BIoU |
|------|-----|------|
| SAM baseline | 43.41 | 43.30 |
| + 直接更新Mask Decoder | 42.82 | 42.69 |
| + HQ Mask Decoder | 47.15 | 47.11 |
| + 交替更新两个Decoder | 48.16 | 48.03 |
| + LoRA + 解冻最后层 | 49.19 | 49.05 |
| + 训练流水线（LSJ+旋转） | **50.54** | **50.36** |

### 跨数据集泛化（静态遥感图像）

| 方法 | iSAID IoU | NPWS VHR-10 IoU |
|------|-----------|-----------------|
| SAM | 53.19 | 65.54 |
| HQ-SAM | 63.96 | 78.44 |
| **ROS-SAM** | **73.22** | **87.46** |

### 关键发现

- 仅推理流水线就为SAM带来6%+ IoU提升，说明遥感场景的分辨率适配至关重要
- 直接微调SAM decoder会降低性能（42.82 < 43.41），验证了保护预训练知识的重要性
- LSJ比随机旋转增益更大（提升~0.8 vs ~0.2），因为多尺度覆盖对遥感更重要
- 在SatSOT、VISO、OOTB等跟踪数据集上零样本生成高质量分割mask，证明强泛化性

## 亮点与洞察

1. **以最小代价将跟踪数据转化为分割数据**的思路极具实用价值，解决了遥感视频分割标注稀缺的问题
2. **交替优化新旧decoder**的策略值得借鉴：既保护预训练知识又引入新能力
3. 推理流水线的中心裁剪+上采样策略简单有效，对所有SAM变体都有即插即用的提升

## 局限与展望

- 仅在SAT-MTB数据集（249个视频）上训练，数据规模有限
- 汽车类目标因像素过小（约10像素）无法有效处理
- 当目标特征高度模糊时（如人工标注基于先验知识推断形状），模型难以达到人工水平
- 未来可考虑结合SAM2的视频传播能力实现半自动视频分割

## 相关工作与启发

- **与SAM/HQ-SAM的关系**：ROS-SAM在HQ-SAM基础上增加了LoRA域适配和交替训练策略
- **遥感+基础模型趋势**：LoRA/Adapter微调大模型适配遥感是当前热点，ROS-SAM验证了LoRA在Q/V矩阵上的有效性
- **启发**：对任何需要保持泛化能力的基础模型微调场景，"冻结主体+LoRA+交替优化"是一个通用模板

## 评分

⭐⭐⭐⭐

工程设计扎实，每个模块的消融实验充分验证。推理流水线和交替优化策略有独立参考价值。主要局限是数据集较小、场景相对受限，且核心思路是对SAM/HQ-SAM微调方法的组合创新。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2026\] F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation](../../CVPR2026/segmentation/f2net_a_frequency-fused_network_for_ultra-high_resolution_remote_sensing_segment.md)

</div>

<!-- RELATED:END -->
