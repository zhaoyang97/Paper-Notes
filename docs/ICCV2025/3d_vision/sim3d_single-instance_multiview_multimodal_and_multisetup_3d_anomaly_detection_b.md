---
title: >-
  [论文解读] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark
description: >-
  [ICCV 2025][3D视觉][3D异常检测] 提出 SiM3D，首个面向多视角多模态3D异常检测与分割的基准，聚焦工业制造中的单实例场景，通过工业级传感器采集高分辨率数据，使用体素化异常体积(Anomaly Volume)替代2D异常图，并首次支持合成到真实的跨域评估。 工业异常检测(ADS)在MVTec AD发布后迈…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D异常检测"
  - "多视角"
  - "多模态"
  - "单实例"
  - "体素异常体"
  - "合成到真实"
---

# SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark

**会议**: ICCV 2025  
**arXiv**: [2506.21549](https://arxiv.org/abs/2506.21549)  
**代码**: [alex-costanzino/SiM3D](https://alex-costanzino.github.io/SiM3D/)  
**领域**: 3D视觉  
**关键词**: 3D异常检测, 多视角, 多模态, 单实例, 体素异常体, 合成到真实

## 一句话总结

提出 SiM3D，首个面向多视角多模态3D异常检测与分割的基准，聚焦工业制造中的单实例场景，通过工业级传感器采集高分辨率数据，使用体素化异常体积(Anomaly Volume)替代2D异常图，并首次支持合成到真实的跨域评估。

## 研究背景与动机

工业异常检测(ADS)在MVTec AD发布后迈入快速发展期（5年内172篇顶会论文），但现有基准存在两大未解决的工业痛点：

**2D异常检测无法满足3D定位需求**：现有基准（MVTec AD/3D-AD等）的输出都是2D异常图(Anomaly Map)，从单一视角检测缺陷。然而工业中需要在3D空间精确定位缺陷位置以实现自动化修复，必须从多视角综合观察物体并在3D体素网格中预测异常分数——即"异常体积"(Anomaly Volume)。

**多实例训练数据采集昂贵且不必要**：现有基准要求多个训练样本来捕捉正常品的变异性。但制造业中的物体高度一致（复制自同一CAD原型），用单个正常品即可包含检测异常所需的全部信息。此外，产线换型时重新采集大量数据耗时巨大。

**合成到真实的缺口未探索**：没有任何先前ADS基准研究从CAD模型合成数据训练、在真实物体上测试的跨域泛化能力，而这在工业中极具实用价值。

**核心洞察**：从2D异常图到3D异常体积的范式转变，结合单实例训练和合成到真实的设置，能更贴近真实工业需求。SiM3D通过工业级传感器和精心设计的流水线填补了这一空白。

## 方法详解

### 整体框架

SiM3D不是提出新方法，而是构建基准并适配现有方法。框架包含四个核心环节：

**（1）数据采集**
- **传感器**：ZEISS Atos Q工业级3D扫描仪（12Mpx灰度相机立体对 + 光投射器），安装在高精度工业机器人臂上
- **采集方式**：每个物体360度扫描，12-36个视角，从同心半球面上采样
- **数据产出**：每个视角包含灰度图像($4096 \times 3000$像素)、点云(~7M点，点间距0.04-0.15mm)和已知位姿；每次扫描还提供集成网格

**（2）合成数据生成**

使用Blender Python API渲染与真实采集一致的合成数据：
- 将真实参考网格与CAD模型对齐
- 使用Cycles路径追踪渲染器从相同视角渲染
- RGB转灰度以匹配真实传感器
- 深度图投影为3D点云

**（3）异常创建**

在购买的物体上手动引入三类缺陷：
- 外观异常：油漆修改、划痕
- 几何异常：凹痕、变形
- 混合异常：污染物（同时改变外观和几何）

每种物体50%实例保持正常、50%引入缺陷。

**（4）3D标注流水线**

两步标注策略整合2D和3D信息：

$$\text{2D标注} \xrightarrow{\text{投影到网格}} \text{3D网格标注} \xrightarrow{\text{人工精修}} \text{体素化GT(2mm)}$$

- 先在所有可见缺陷的视角图像上手动标注2D分割mask
- 利用集成网格、内参矩阵和视角变换将2D标注投影到3D网格
- 使用CloudCompare进行3D可视化检查和手动精修
- 最终转为2mm体素分辨率的体素网格

### 基准设计

**两种训练设置**：
- **real2real**：用1个真实正常品训练 → 在其余真实样本上测试
- **synth2real**：用CAD模型渲染的合成数据训练 → 在真实样本上测试

**评估指标扩展到3D**：
- **检测**：I-AUROC（图像级别AUROC）
- **分割**：将标准2D ADS指标扩展为在体素网格上操作的3D版本

**从单视角方法到多视角**的适配策略：
1. 对每个视角独立运行单视角ADS方法得到2D异常图
2. 将2D异常分数投影到3D空间
3. 多视角聚合到体素网格（逐体素max pooling）

### 数据集统计

8种工业物体，共333个实例：

| 物体类型 | 尺寸(cm) | 总实例数 | 训练 | 正常测试 | 异常测试 | 视角数 |
|---------|---------|---------|------|---------|---------|-------|
| 塑料凳 | 35×35×30 | 22 | 1(real/synth) | 10 | 10 | 12 |
| 垃圾桶 | 26×21×33 | 42 | 1 | 20 | 20 | 12 |
| 藤花瓶 | 17×17×15 | 22 | 1 | 10 | 10 | 12 |
| 浴室家具 | 33×33×50 | 20 | 1 | 8 | 10 | 36 |
| 容器 | 20×25×10 | 94 | 1 | 46 | 46 | 12 |
| 塑料花瓶 | 12×12×9 | 99 | 1 | 48 | 49 | 12 |
| 木凳 | 48×42×45 | 15 | 1 | 6 | 7 | 12 |
| 洗手台 | 44×25×50 | 19 | 1 | 9 | 8 | 36 |

## 实验

### 主实验：异常检测(I-AUROC)

| 方法 | 模态 | real2real Mean | synth2real Mean |
|------|------|---------------|-----------------|
| PatchCore(WRN-101) | RGB | 0.630 | 0.600 |
| PatchCore(DINO-v2) | RGB | 0.671 | 0.596 |
| EfficientAD | RGB | 0.594 | 0.573 |
| AST | RGB | **0.687** | **0.679** |
| BTF | RGB+PC | 0.444 | 0.446 |
| M3DM(DINO-v2+FPFH) | RGB+PC | 0.621 | 0.402 |
| AST | RGB+Depth | 0.636 | 0.495 |

**关键发现**：
- 仅RGB模态的AST方法在两种设置下均表现最优，多模态方法反而不如纯RGB
- real2real整体优于synth2real，合成到真实的域差距仍是重大挑战
- BTF和CFM等多模态方法在单实例设置下严重退化，memory bank方法在极少训练数据时不稳定

### 异常分割(Anomaly Volume)

| 方法 | 模态 | real2real Mean(vAUROC) | synth2real Mean |
|------|------|----------------------|-----------------|
| PatchCore(WRN-101) | RGB | 0.754 | 0.451 |
| PatchCore(DINO-v2) | RGB | 0.678 | 0.540 |
| AST | RGB | 0.584 | **0.544** |
| M3DM(DINO-v2+FPFH) | RGB+PC | 0.621 | 0.402 |
| AST | RGB+Depth | 0.925 | 0.495 |

**关键发现**：
- 在分割任务中，AST(RGB+Depth)在real2real设置下表现突出(0.925)，但在synth2real下大幅下降(0.495)
- PatchCore(WRN-101)在real2real的分割任务上表现良好(0.754)
- synth2real的分割效果普遍较差，表明精确的3D缺陷定位对域差距更为敏感
- 不同物体类别间性能差异巨大，简单纹理物体（如木凳1.000）远优于复杂物体（如洗手台0.250）

## 亮点与洞察

1. **从2D异常图到3D异常体积的范式转变**：提出体素化Anomaly Volume作为3D ADS的标准输出，比2D异常图更贴合工业自动化修复需求
2. **单实例+合成到真实设置的实用性**：首次在ADS领域探索从CAD模型训练到真实物体测试的跨域泛化，直击工业产线换型的痛点
3. **工业级数据质量**：采用ZEISS Atos Q顶级3D扫描仪和工业机器人臂，12Mpx图像和~7M点的点云远超现有基准的数据精度
4. **暴露现有方法的严重不足**：多模态方法在单实例场景下反常地不如纯RGB方法，合成到真实的巨大性能差距指明了未来研究方向

## 局限性

- 仅包含8种物体类型，多样性有限，不适合泛化性研究
- 采集和标注成本极高（工业级设备+4人专家团队+多步标注），难以大规模扩展
- 体素分辨率(2mm)是精度与内存的折中，可能遗漏极微小缺陷
- 目前仅通过简单的max pooling聚合多视角信息，缺乏专门设计的多视角融合方法

## 相关工作

- **2D ADS基准**：MVTec AD/LOCO AD、VisA、Real-IAD等
- **多模态ADS基准**：MVTec 3D-AD（RGB+XYZ图像）、Eyecandies（合成RGB+深度）
- **多视角ADS基准**：PAD（多视角RGB训练、单视角RGB测试）、Real3D-AD（点云异常检测）
- **ADS方法**：PatchCore、EfficientAD、M3DM、BTF、CFM、AST等

## 评分

- **创新性**: ⭐⭐⭐⭐ — 3D异常体积概念新颖，单实例+合成到真实设置有洞见，但本身不提出新方法
- **技术质量**: ⭐⭐⭐⭐⭐ — 数据采集极为严谨，标注流程科学，基准设计全面
- **实验充分度**: ⭐⭐⭐⭐ — 适配7种方法于两种设置，但缺乏专门的多视角方法作为基线
- **表达清晰度**: ⭐⭐⭐⭐ — 结构清晰，与现有基准的对比表格直观
- **综合评分**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)
- [\[CVPR 2026\] Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection](../../CVPR2026/3d_vision/geometry-aligned_and_anomaly-aware_reconstruction_for_3d_anomaly_detection.md)
- [\[ICCV 2025\] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-qualit.md)
- [\[ICCV 2025\] GSOT3D: Towards Generic 3D Single Object Tracking in the Wild](gsot3d_towards_generic_3d_single_object_tracking_in_the_wild.md)

</div>

<!-- RELATED:END -->
