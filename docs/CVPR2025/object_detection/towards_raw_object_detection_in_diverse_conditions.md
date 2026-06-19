---
title: >-
  [论文解读] Towards RAW Object Detection in Diverse Conditions
description: >-
  [CVPR 2025][目标检测][RAW图像检测] 提出 AODRaw 数据集（7,785张高分辨率真实RAW图像，62类，9种光照/天气条件），并通过RAW域预训练+跨域蒸馏方案，无需ISP模块即可在多种恶劣条件下实现优异的RAW目标检测性能。 - 现有目标检测方法主要基于 sRGB 图像，而 sRGB 图像是从 RAW…
tags:
  - "CVPR 2025"
  - "目标检测"
  - "RAW图像检测"
  - "恶劣条件感知"
  - "跨域蒸馏"
  - "数据集"
---

# Towards RAW Object Detection in Diverse Conditions

**会议**: CVPR 2025  
**arXiv**: [2411.15678](https://arxiv.org/abs/2411.15678)  
**代码**: [GitHub](https://github.com/lzyhha/AODRaw)  
**领域**: 目标检测  
**关键词**: RAW图像检测, 恶劣条件感知, 跨域蒸馏, 数据集

## 一句话总结

提出 AODRaw 数据集（7,785张高分辨率真实RAW图像，62类，9种光照/天气条件），并通过RAW域预训练+跨域蒸馏方案，无需ISP模块即可在多种恶劣条件下实现优异的RAW目标检测性能。

## 研究背景与动机

- 现有目标检测方法主要基于 sRGB 图像，而 sRGB 图像是从 RAW 数据经 ISP 压缩而来，在复杂光照和恶劣天气条件下可能丢失关键信息
- RAW 图像保留更高位深（如16-bit），包含更多可分辨信息，尤其在低光照、雾天、雨天等恶劣条件下优势明显
- 现有 RAW 检测数据集存在严重不足：类别少（如 LOD 仅8类，RAOD 仅6类）、条件单一（仅日光/低光），缺乏真实多样的恶劣天气场景
- 传统 RAW 检测方法依赖神经 ISP 将 RAW 转换到 sRGB 域，引入额外计算开销且无法充分利用 RAW 信息
- sRGB 预训练模型直接迁移到 RAW 域时，由于域差距（domain gap）性能受限
- 需要一个规模更大、条件更多样的 RAW 检测基准数据集，以及一种无需 ISP 适配器的高效检测方案

## 方法详解

### 整体框架

本文贡献分为两部分：（1）构建 AODRaw 大规模多条件 RAW 目标检测数据集；（2）提出基于跨域蒸馏的 RAW 域预训练方法，直接在 RAW 域训练骨干网络，消除 sRGB-RAW 之间的域差距。整体流程为：先使用 unprocessing 方法从 ImageNet-1K 合成 RAW 数据（ImageNet-RAW），在其上预训练骨干网络，再在真实 RAW 数据上微调检测器。预训练阶段利用 sRGB 域预训练教师模型进行知识蒸馏，辅助 RAW 模型学习更好的表征。

### 关键设计

**1. AODRaw 数据集构建**

- **功能**：提供大规模、多条件、高分辨率的真实 RAW 检测基准
- **核心思路**：数据集包含 7,785 张 $6000 \times 4000$ 分辨率 RAW 图像和 135,601 个标注实例，涵盖 62 个类别。覆盖 2 种光照条件（日光、低光）和 3 种天气条件（晴天、雨天、雾天），组合形成 9 种不同条件，包括室内和室外场景。平均每张图像 17.4 个实例，采用 COCO 格式标注
- **设计动机**：现有 RAW 检测数据集在类别数、条件多样性和规模上均严重不足，无法支持 RAW 检测在真实恶劣条件下的全面评估

**2. 合成 ImageNet-RAW 预训练**

- **功能**：消除 sRGB 预训练与 RAW 微调之间的域差距
- **核心思路**：使用 unprocessing 方法将 ImageNet-1K 的 sRGB 图像逆转换为 16-bit RAW 格式，并模拟相机噪声。将逆处理操作嵌入数据增强流程，每次迭代随机调整亮度和噪声水平，使模型在不同条件下具有更好的泛化能力
- **设计动机**：收集与 ImageNet 规模相当的真实 RAW 数据集不现实，合成策略可以低成本地生成大量 RAW 训练数据用于预训练

**3. 跨域知识蒸馏**

- **功能**：辅助 RAW 域模型学习更高质量的特征表征
- **核心思路**：利用在 sRGB 域预训练好的现成模型作为教师网络，通过特征蒸馏将知识迁移到 RAW 域学生模型。由于 RAW 图像中相机噪声的存在，直接在 RAW 上预训练较难学到高质量表征，蒸馏可以弥补这一不足
- **设计动机**：实验发现 RAW 预训练比 sRGB 预训练更难学到丰富表征（因噪声干扰），借助已有的 sRGB 预训练知识可以有效缓解此问题

### 损失函数 / 训练策略

- 检测器使用标准检测损失（如 Cascade R-CNN 的多阶段分类+回归损失）
- 蒸馏损失：教师（sRGB 预训练模型）与学生（RAW 预训练模型）之间的特征对齐损失
- 训练48轮（epoch），batch size=16，Deformable DETR 训练100轮
- RAW 图像从 Bayer 格式 $1 \times H \times W$ 去马赛克为 $3 \times H \times W$，再经 gamma 校正加速收敛
- 评估采用下采样至 $2000 \times 1333$ 或裁切为 $1280 \times 1280$ 的 patch（overlap=300）两种设置

## 实验关键数据

### 主实验

| 方法 | 骨干 | 预训练→微调 | AP | AP_normal | AP_low | AP_rain | AP_fog |
|------|------|------------|------|-----------|--------|---------|--------|
| Cascade RCNN | ConvNeXt-T | sRGB→sRGB | 34.0 | 37.0 | 31.5 | 32.9 | 27.2 |
| Cascade RCNN | ConvNeXt-T | sRGB→RAW | 33.7 | 36.8 | 31.3 | 31.3 | 27.2 |
| Cascade RCNN | ConvNeXt-T | RAW→RAW | **34.8** | **37.7** | **32.1** | **36.1** | **28.4** |
| RAOD | ConvNeXt-T | sRGB+ISP→RAW | 34.4 | 37.3 | 32.4 | 37.7 | 29.4 |

### 消融实验

| 训练域 | 评估域 | AP | AP50 | AP75 |
|--------|--------|------|------|------|
| sRGB | sRGB | 34.0 | 52.7 | 36.3 |
| sRGB | RAW | 33.7 | 52.0 | 35.9 |
| RAW | RAW | 34.8 | 53.3 | 36.7 |

跨域测试显示明显性能下降，验证了 sRGB-RAW 域差距的存在。

### 关键发现

1. RAW 域预训练+蒸馏的 Cascade RCNN AP 达到 34.8%，超过 sRGB 基线 34.0%，且无需任何 ISP 模块
2. 恶劣条件下 RAW 检测优势尤为显著：AP_rain 从 32.9% 提升至 36.1%（+3.2%），远超正常条件的提升幅度
3. sRGB 预训练的模型在 RAW 微调时反而略低于 sRGB 微调（33.7 vs 34.0），证实了域差距问题
4. 数据集中小目标占比高、类别分布呈长尾特征，增加了检测难度

## 亮点与洞察

- 首次构建涵盖 9 种光照/天气组合条件的大规模真实 RAW 检测数据集，填补了该领域的数据空白
- 提出了简洁有效的"合成RAW预训练+蒸馏"范式，无需额外 ISP 模块即可超越基于神经 ISP 的方法
- 系统性地揭示了 sRGB 与 RAW 之间存在显著域差距，且这种差距在恶劣条件下更为突出
- 实验设计全面，同时支持 sRGB 和 RAW 两种检测任务的评估

## 局限与展望

- 合成 RAW 数据与真实 RAW 仍有差异，可能限制预训练效果
- 数据集规模对于目标检测而言仍偏小（7,785张），未来可进一步扩展
- 仅验证了分类检测任务，未探索 RAW 在实例分割、全景分割等更细粒度任务上的潜力
- 蒸馏策略较为基础，可探索更先进的知识蒸馏方法（如特征金字塔级蒸馏）
- 未考虑不同相机型号之间的 RAW 域差异

## 相关工作与启发

- **RAW-Adapter**: 使用可训练 ISP 桥接 sRGB 预训练与 RAW 微调，本文证明直接 RAW 预训练可以更简洁地解决此问题
- **Unprocessing**: 将 sRGB 图像逆转换为 RAW 格式的方法，是本文合成数据策略的基础
- **LOD/RAOD**: 早期 RAW 检测数据集，类别和条件覆盖有限，AODRaw 在此基础上大幅扩展
- 启发：在其他视觉任务（如语义分割、深度估计）中，RAW 输入可能同样在恶劣条件下带来显著增益

## 评分

- **新颖性**: ⭐⭐⭐ — 数据集构建有价值，方法设计（合成预训练+蒸馏）属已有技术的组合应用
- **实验充分度**: ⭐⭐⭐⭐ — 基准测试全面，涵盖多种检测器、骨干网络和训练设置
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，分析系统深入
- **价值**: ⭐⭐⭐⭐ — AODRaw 数据集填补领域空白，对恶劣条件检测研究有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements](../../AAAI2026/object_detection/simrod_a_simple_baseline_for_raw_object_detection_with_global_and_local_enhancem.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)
- [\[CVPR 2026\] SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras](../../CVPR2026/object_detection/spiraldiff_spiral_diffusion_with_lora_for_rgb-to-raw_conversion_across_cameras.md)
- [\[CVPR 2026\] See What We Cannot See: A Geo-guided Reasoning Benchmark for Object Counting under Adverse Earth Observation Conditions](../../CVPR2026/object_detection/see_what_we_cannot_see_a_geo-guided_reasoning_benchmark_for_object_counting_unde.md)

</div>

<!-- RELATED:END -->
