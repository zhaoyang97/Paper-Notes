---
title: >-
  [论文解读] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes
description: >-
  [NeurIPS 2025][目标检测][flicker removal] 提出首个面向多帧闪烁去除（MFFR）的大规模 benchmark 数据集 BurstDeflicker，包含基于 Retinex 的合成数据、真实静态数据和绿幕动态数据三个互补子集，系统解决了动态场景下闪烁-干净图像对难以获取的核心瓶颈。
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "flicker removal"
  - "rolling shutter"
  - "burst imaging"
  - "dataset"
  - "Retinex"
---

# BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes

**会议**: NeurIPS 2025  
**arXiv**: [2510.09996](https://arxiv.org/abs/2510.09996)  
**代码**: [qulishen/BurstDeflicker](https://github.com/qulishen/BurstDeflicker)  
**领域**: 目标检测  
**关键词**: flicker removal, rolling shutter, burst imaging, dataset, Retinex

## 一句话总结

提出首个面向多帧闪烁去除（MFFR）的大规模 benchmark 数据集 BurstDeflicker，包含基于 Retinex 的合成数据、真实静态数据和绿幕动态数据三个互补子集，系统解决了动态场景下闪烁-干净图像对难以获取的核心瓶颈。

## 研究背景与动机

**闪烁伪影普遍存在**：在交流电（AC）供电的人工光源下，短曝光拍摄的图像会因卷帘快门（rolling shutter）的逐行曝光机制与灯光周期性亮度变化的耦合，产生非均匀亮度分布和明显暗带（banding artifact），严重影响图像质量及检测、跟踪等下游任务。

**硬件方案局限性大**：传统方法依赖闪烁检测电路延长曝光时间，但这会引入运动模糊，难以兼顾去闪烁与保清晰。神经形态传感器虽有高时间分辨率优势，但成本高、标定复杂，难以在消费级设备上部署。

**单帧去闪烁（SIFR）有固有歧义**：闪烁伪影是空间局部化且时间动态的，单帧方法难以将闪烁暗区与阴影等相似暗区区分，且缺乏像素上下文导致恢复不可靠。

**多帧方案更有前景**：现代手持设备通常单次拍摄多帧，天然提供丰富的时间线索和帧间相关性，有助于准确定位和去除闪烁，但缺乏大规模高质量的多帧闪烁去除（MFFR）数据集。

**现有合成数据不合理**：此前 Wong 等人和 Lin 等人的闪烁合成方法最初为地理标记设计，将闪烁去除建模为完全消除闪烁照明，而非调整到有效值，导致训练数据缺乏多样性和真实性。

**动态场景配对数据获取几乎不可能**：动态场景不可重复，无法获取对齐的闪烁-干净图像对，导致模型容易将运动引起的像素变化误判为闪烁伪影，产生鬼影。

## 方法详解

### 整体框架

BurstDeflicker 数据集由三个互补子集构成，分别从规模、真实性和动态性三个维度解决数据获取难题：

- **Retinex-based 合成数据**：基于 Retinex 理论建模环境光与闪烁光的交互，支持无限量生成多样闪烁模式，用于预训练
- **BurstDeflicker-S（4,000 张真实静态数据）**：从 369 个真实场景用短曝光捕获闪烁帧 + 长曝光捕获干净参考帧
- **BurstDeflicker-G（3,690 张绿幕动态数据）**：将绿幕前景叠加到真实闪烁背景上，模拟动态场景

训练流程：先用合成数据预训练 → 再用真实数据（S + G）微调，每次随机采样 3 帧（间隔 1-3 帧）拼接输入，输出单张干净图像。

### 关键设计 1：Retinex-based 闪烁合成

- **功能**：重新定义闪烁去除目标，将闪烁照明调整到有效值而非完全消除
- **核心思路**：根据 Retinex 理论，闪烁图像 $I_{flicker} = R \odot (L_a + L_f)$，干净图像 $I_{clean} = R \odot (L_a + \overline{L_f})$，其中 $\overline{L_f}$ 为闪烁光的有效值。通过变换得到 $I_{flicker} = I_{clean} \cdot (1 + \frac{L_f/\overline{L_f} - 1}{k+1})$，其中 $k$ 为环境光与闪烁光强度比
- **设计动机**：此前方法将去闪烁等同于完全去除闪烁照明（$I_{clean} = R \odot L_a$），这与现实不符——闪烁灯实际提供有效照明，只是瞬时值在波动。新公式更准确地反映物理过程

### 关键设计 2：多种整流模式建模

- **功能**：建模全波整流（荧光灯）、半波整流（白炽灯）、PWM 整流（LED）三种主要光源的闪烁模式
- **核心思路**：交流正弦波经不同整流方式产生不同的亮度分布模式，通过参数化公式（Eq. 4）精确建模每种模式下的逐行闪烁强度分布，支持电网频率（50/60 Hz）、行扫描频率、初始相位、占空比等参数控制
- **设计动机**：现实中不同类型灯的闪烁模式差异显著，单一正弦叠加的合成方式远不能覆盖真实世界的多样性

### 关键设计 3：绿幕合成动态数据

- **功能**：解决动态场景无法获取配对数据的核心难题
- **核心思路**：从 VideoMatte240K 数据集选取语义和空间兼容的绿幕前景片段，利用 alpha mask 高质量合成到真实闪烁背景上。对干净 GT 图像复制 10 次并叠加相同前景，确保除背景闪烁外其余一致
- **设计动机**：动态场景不可重复，直接拍摄无法获得对齐配对。模型如果只在静态数据上训练，会把运动像素变化误判为闪烁，导致恢复图中出现鬼影伪影。绿幕方法让模型"看到"带运动的闪烁图像对，学会区分运动与闪烁

### 关键设计 4：真实数据采集管道

- **功能**：构建 4,000 张真实世界闪烁-干净配对数据
- **核心思路**：使用三脚架固定相机 + 遥控快门 + 电子快门消除机械抖动。短曝光（1/1000-1/2000s）捕获闪烁帧，长曝光（1/50 或 1/60s）积分多个闪烁周期得到干净参考帧。每个场景连续拍 10 帧 burst
- **设计动机**：369 个场景覆盖室内（办公室、超市、地铁站）和室外（LED 广告牌、停车场），确保多样性。使用手动模式保证成像条件一致

## 训练策略

- 合成数据预训练 → BurstDeflicker 真实数据微调
- 输入为随机采样的 3 帧（间隔 1-3），数据增强后沿通道维度拼接
- 引入随机旋转（$[-3°, 3°]$）和平移（$[-5, 5]$ 像素）模拟手持抖动
- 训练时 resize 而非裁剪，以保留闪烁伪影沿扫描方向的周期性
- 训练/测试集按 8:2 划分

## 实验关键数据

### 表 1：不同方法在静态和动态测试集上的性能对比

| 方法 | FLOPs (G) | Params (M) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | PIQE ↓ | BRISQUE ↓ |
|------|-----------|------------|--------|--------|---------|---------|--------|-----------|
| Retinexformer* (未用本文数据) | 69.23 | 1.61 | 15.704 | 0.707 | 0.213 | 53.596 | 50.269 | 30.242 |
| Lin et al.* (原始版本) | 509.80 | 92.08 | 20.358 | 0.838 | 0.134 | 55.228 | 43.875 | 25.121 |
| Lin et al. (用本文数据重训) | 509.80 | 92.08 | 26.408 | 0.875 | 0.102 | 58.131 | 35.710 | 22.102 |
| Retinexformer (重训) | 69.23 | 1.61 | 27.212 | 0.885 | 0.081 | 58.249 | 35.942 | 21.648 |
| Burstormer | 141.05 | 0.17 | 29.439 | 0.910 | 0.056 | 58.527 | 37.014 | 20.451 |
| HDRTransformer | 272.12 | 1.04 | 30.031 | 0.914 | 0.054 | 59.069 | 37.292 | 21.588 |
| **Restormer** | **149.01** | **7.92** | **30.634** | **0.918** | **0.045** | **59.097** | **34.896** | **19.324** |

**关键发现**：用 BurstDeflicker 数据集重训后，Lin et al. 的 PSNR 从 20.358 提升到 26.408（+6.05 dB），Retinexformer 从 15.704 到 27.212（+11.51 dB），证明数据集的有效性。Restormer（3 帧输入）达到最佳 30.634 dB。

### 表 2：数据集各子集消融实验（Restormer）

| 合成数据 | BurstDeflicker-S | BurstDeflicker-G | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | PIQE ↓ | BRISQUE ↓ |
|----------|-----------------|-----------------|--------|--------|---------|---------|--------|-----------|
| ✓ | | | 24.483 | 0.862 | 0.122 | 57.096 | 39.726 | 23.755 |
| ✓ | ✓ | | 30.481 | 0.915 | 0.053 | 58.011 | 37.498 | 21.523 |
| ✓ | | ✓ | 30.645 | 0.916 | 0.052 | 58.431 | 36.259 | 20.338 |
| ✓ | ✓ | ✓ | **30.634** | **0.918** | **0.045** | **59.097** | **34.896** | **19.324** |

**关键发现**：三个子集互补——合成数据预训练增强鲁棒性；绿幕动态数据在动态测试集上提升显著（BRISQUE 从 21.523 降到 19.324），有效缓解鬼影伪影。

### 表 3：输入帧数消融（Restormer）

| 输入 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 单帧 | 27.310 | 0.891 | 0.069 |
| 2 帧 | 30.264 (+2.954) | 0.915 | 0.048 |
| 3 帧 | 30.634 (+3.324) | 0.918 | 0.045 |

从 1 帧到 2 帧提升 2.954 dB，从 2 帧到 3 帧仅提升 0.37 dB，体现相邻帧间信息冗余的边际效应。

## 亮点与洞察

1. **重新定义问题**：纠正了此前将去闪烁等同于"完全去除闪烁光照"的错误建模，指出应调整到有效值，这一物理视角的修正对后续工作有指导意义
2. **三阶段数据构建策略非常完整**：合成（解决规模）→ 真实静态（解决真实性）→ 绿幕动态（解决动态场景），每个子集针对一个明确瓶颈，设计清晰
3. **绿幕方法巧妙实用**：借鉴电影工业成熟技术解决学术数据集构建难题，让模型学会区分"运动 vs 闪烁"的像素变化
4. **通用性好**：数据集不绑定特定网络架构，Restormer、Burstormer、HDRTransformer 等均可受益

## 局限性

1. 当闪烁光源为唯一光源时，退化极为严重，恢复结果可能出现明显色偏
2. 帧间严重不对齐（如剧烈手持抖动）时，多帧方法性能退化至单帧水平
3. 绿幕合成本质上仍是半合成数据，与真实动态场景存在域差距
4. 当前未设计专门针对闪烁先验的网络架构，仅用通用 restoration 网络适配

## 相关工作与启发

- **与 DeflickerCycleGAN（Lin et al.）对比**：该单帧方法在严重闪烁和光照不足场景中效果有限，重训后在本文数据集上 PSNR 仅 26.408，远低于多帧方案
- **与 Retinexformer 对比**：低光增强方法全局提亮并引入色偏，不适合闪烁去除任务
- **与 burst SR 的联系**：借鉴了 BurstSR 的多帧预训练+微调策略，以及 resize 替代 crop 的训练技巧
- **启发**：该数据集的三阶段构建思路（合成预训练 → 真实采集 → 半合成动态）可推广到其他难以获取配对数据的 restoration 任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个 MFFR 数据集，Retinex 重新建模闪烁去除目标，绿幕合成方法有创意
- **实验充分度**: ⭐⭐⭐⭐ — 多个 baseline、消融完整（数据子集、帧数），有真实动态测试集；但缺乏与更多最新方法的对比
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，三阶段结构递进，图表丰富
- **价值**: ⭐⭐⭐⭐ — 填补了 MFFR 数据集空白，对闪烁去除研究有实质推动作用，数据集已开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](../../CVPR2025/object_detection/object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[NeurIPS 2025\] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection](dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)

</div>

<!-- RELATED:END -->
