---
title: >-
  [论文解读] Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal
description: >-
  [ECCV 2024][图像恢复][雨滴去除] 提出了一个大规模真实世界雨滴去除数据集 Raindrop Clarity，包含15,186组高质量图像对/三元组，首次涵盖雨滴聚焦（清晰雨滴+模糊背景）和夜间雨滴两种现有数据集缺失的场景。 附着在镜头或挡风玻璃上的雨滴会严重降低视觉质量，对监控、自动驾驶、目标检测等应用产生影响…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "雨滴去除"
  - "数据集"
  - "夜间图像"
  - "双焦点"
---

# Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal

**会议**: ECCV 2024  
**arXiv**: [2407.16957](https://arxiv.org/abs/2407.16957)  
**代码**: 有 ([https://github.com/jinyeying/RaindropClarity](https://github.com/jinyeying/RaindropClarity))  
**领域**: 其他  
**关键词**: 雨滴去除, 数据集, 夜间图像, 双焦点, 图像恢复

## 一句话总结

提出了一个大规模真实世界雨滴去除数据集 Raindrop Clarity，包含15,186组高质量图像对/三元组，首次涵盖雨滴聚焦（清晰雨滴+模糊背景）和夜间雨滴两种现有数据集缺失的场景。

## 研究背景与动机

附着在镜头或挡风玻璃上的雨滴会严重降低视觉质量，对监控、自动驾驶、目标检测等应用产生影响。现有雨滴去除数据集存在两个关键缺陷：

**仅包含背景聚焦图像**：现有数据集（如Qian等、RainDS、RobotCar）都是相机对焦在背景上拍摄的，雨滴呈模糊状态。但实际中相机自动对焦可能聚焦到附着的雨滴上，导致雨滴清晰而背景模糊——这类场景完全被忽视。

**缺乏夜间数据**：现有数据集全部为白天拍摄。由于白天与夜间存在显著的域差异（人工照明、低光照），在白天数据上训练的模型难以处理夜间雨滴图像。

这两个缺陷严重制约了雨滴去除算法的泛化能力。

## 方法详解

### 整体框架

本文的核心贡献是数据集而非新算法。Raindrop Clarity 数据集的设计覆盖四种场景：

| 场景 | 时间段 | 焦点 | 雨滴状态 | 背景状态 |
|------|--------|------|----------|----------|
| 1 | 白天 | 背景聚焦 | 模糊 | 清晰 |
| 2 | 白天 | 雨滴聚焦 | 清晰 | 模糊 |
| 3 | 夜间 | 背景聚焦 | 模糊 | 清晰 |
| 4 | 夜间 | 雨滴聚焦 | 清晰 | 模糊 |

数据集提供两种标注格式：
- **图像对** $(\tilde{\mathbf{x}}, \mathbf{b}_0)$：雨滴图像 + 清晰背景（背景聚焦场景）
- **三元组** $(\tilde{\mathbf{x}}, \mathbf{x}_0, \mathbf{b}_0)$：雨滴图像 + 模糊无雨滴背景 + 清晰背景（雨滴聚焦场景）

### 关键设计

**数据采集流程**：
1. 使用球台云台固定相机，保证静止拍摄
2. 拍摄设备：Sony FDR-AX33 4K摄像机、Sony Alpha 7R III、iPhone 14 Pro/15 Pro Max
3. 安装玻璃板，相机与玻璃板距离5-25cm
4. 喷水/自然雨水在玻璃板上形成雨滴，对焦到雨滴拍摄
5. 移除玻璃板，保持相机近平面对焦获取无雨滴模糊背景
6. 调整对焦到远处背景获取清晰背景图像

**差异图**：对于三元组数据，可以计算像素级差异图 $\tilde{\mathbf{m}} = \tilde{\mathbf{x}} - \mathbf{x}_0$ 来精确定位雨滴区域。

**数据特点**：
- 总量约 38,816 张图像，其中 15,186 组高质量配对
- 白天 5,442 组（3,606 三元组 + 1,836 对）
- 夜间 9,744 组（4,838 三元组 + 4,906 对）
- 涵盖城市、乡村、校园、道路、航拍等多种背景
- 雨滴形态多样：椭圆形、水滴流痕、不同密度

### 损失函数 / 训练策略

作者使用标准图像恢复方法进行基准测试。对于背景聚焦的雨滴退化模型：

$$\tilde{\mathbf{x}} = (1 - \mathbf{M}) \odot \mathbf{x}_0 + \mathbf{D}$$

其中 $\mathbf{M}$ 为二值雨滴掩码，$\mathbf{D}$ 为模糊雨滴的光学效果。

## 实验关键数据

### 主实验（表格）

| 方法 | 白天PSNR↑ | 白天SSIM↑ | 白天LPIPS↓ | 夜间PSNR↑ | 夜间SSIM↑ | 夜间LPIPS↓ |
|------|-----------|-----------|------------|-----------|-----------|------------|
| Input | 21.92 | 0.560 | 0.247 | 24.78 | 0.726 | 0.209 |
| AtGAN | 23.62 | 0.658 | 0.200 | 24.38 | 0.773 | 0.185 |
| RDdiff | 26.05 | 0.736 | 0.141 | 26.81 | 0.851 | 0.125 |
| Uformer | **26.08** | **0.748** | 0.131 | **26.87** | **0.848** | 0.123 |
| DiT | 26.03 | 0.752 | **0.106** | 26.23 | 0.826 | **0.111** |
| Restormer | 25.52 | 0.734 | 0.111 | 26.48 | 0.831 | 0.112 |

### 数据集对比（表格）

| 数据集 | 图像数 | 真实/合成 | 白天背景焦 | 夜间背景焦 | 白天雨滴焦 | 夜间雨滴焦 |
|--------|--------|-----------|------------|------------|------------|------------|
| **Raindrop Clarity (Day)** | 14,490 | 真实 | ✓ | ✗ | ✓ | ✗ |
| **Raindrop Clarity (Night)** | 24,326 | 真实 | ✗ | ✓ | ✗ | ✓ |
| Raindrop Qian | 1,838 | 真实 | ✓ | ✗ | ✗ | ✗ |
| RainDS-Real | 992 | 真实 | ✓ | ✗ | ✗ | ✗ |
| RobotCar | 9,636 | 真实 | ✓ | ✗ | ✗ | ✗ |
| Windshield | 3,390 | 真实 | ✓ | ✗ | ✗ | ✗ |

### 关键发现

1. 现有SOTA雨滴去除方法在雨滴聚焦和夜间场景下性能显著下降
2. 即使最好的方法（如Uformer、DiT），在Raindrop Clarity上仍有明显的失败案例
3. 夜间场景中人工光源（街灯、车灯、霓虹灯）增加了雨滴外观的复杂性
4. 雨滴聚焦场景需要同时去除雨滴和恢复模糊背景，难度远高于传统设定
5. 通用图像恢复骨干（Restormer、Uformer）表现可与专用雨滴去除方法媲美

## 亮点与洞察

- **问题定义的前瞻性**：识别出"雨滴聚焦"这一被忽视的实际场景，填补了数据集空白
- **三元组标注**提供的差异图为精确雨滴检测和分割提供了新的监督信号
- **白天+夜间的全面覆盖**使得训练的模型具有更强的全天候工作能力
- 数据采集方法巧妙利用了光学折射模型和焦距切换来生成配对数据

## 局限与展望

- 数据采集使用固定相机+玻璃板的受控环境，与真实行车场景存在差距
- 未提供针对数据集特点的新算法，仅做了基准测试
- 夜间和雨滴聚焦的联合挑战（模糊+雨滴+低光照）尚无有效解决方案
- 可以考虑结合深度信息来辅助区分前景雨滴和背景模糊
- 缺少视频层面的时序雨滴去除数据

## 相关工作与启发

- 与RainDS数据集类似地提供了多种天气退化的组合，但规模和多样性远超前者
- 差异图的思路可借鉴到其他退化检测任务中
- 对自动驾驶等需要全天候工作的系统有直接应用价值
- 数据集的双焦点设计启发我们思考其他"被忽视的退化模式"

## 评分

- **创新性**: ★★★★☆ — 数据集定义新颖，首次覆盖雨滴聚焦+夜间场景
- **实用性**: ★★★★★ — 大规模真实数据集，对社区价值极高
- **实验完整性**: ★★★☆☆ — 仅基准测试，未提出新方法
- **写作质量**: ★★★★☆ — 数据集构建过程描述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[ECCV 2024\] Spatially-Variant Degradation Model for Dataset-free Super-resolution](spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [\[ECCV 2024\] Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising](exploiting_dual-correlation_for_multi-frame_time-of-flight_denoising.md)
- [\[CVPR 2026\] From Events to Clarity: The Event-Guided Diffusion Framework for Dehazing](../../CVPR2026/image_restoration/from_events_to_clarity_the_event-guided_diffusion_framework_for_dehazing.md)
- [\[ECCV 2024\] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)

</div>

<!-- RELATED:END -->
