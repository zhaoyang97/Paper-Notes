---
title: >-
  [论文解读] Towards a Unified Copernicus Foundation Model for Earth Vision
description: >-
  [ICCV 2025][遥感][地球观测基础模型] 构建了涵盖所有主要Copernicus Sentinel任务的统一地球观测基础模型体系，包括1870万对齐图像的Copernicus-Pretrain数据集、支持任意光谱/非光谱传感器的Copernicus-FM模型、以及覆盖15个层级化下游任务的Copernicus-Bench评估基准。
tags:
  - "ICCV 2025"
  - "遥感"
  - "地球观测基础模型"
  - "多模态预训练"
  - "Copernicus Sentinel"
  - "动态超网络"
  - "大气监测"
---

# Towards a Unified Copernicus Foundation Model for Earth Vision

**会议**: ICCV 2025  
**arXiv**: [2503.11849](https://arxiv.org/abs/2503.11849)  
**代码**: [GitHub](https://github.com/zhu-xlab/Copernicus-FM)  
**领域**: Remote Sensing  
**关键词**: 地球观测基础模型, 多模态预训练, Copernicus Sentinel, 动态超网络, 大气监测

## 一句话总结

构建了涵盖所有主要Copernicus Sentinel任务的统一地球观测基础模型体系，包括1870万对齐图像的Copernicus-Pretrain数据集、支持任意光谱/非光谱传感器的Copernicus-FM模型、以及覆盖15个层级化下游任务的Copernicus-Bench评估基准。

## 研究背景与动机

地球观测（EO）基础模型的发展面临三大瓶颈：

**传感器多样性不足**：现有预训练数据集主要聚焦Sentinel-1/2和Landsat等高/中分辨率传感器，忽略了Sentinel-3和Sentinel-5P等低分辨率但具有高时间频率的大气监测任务

**模型灵活性有限**：大多数模型采用针对特定传感器模态的刚性架构，无法动态适应新光谱波段或非光谱输入（如大气成分）

**评估范围狭窄**：现有基准主要关注地表应用的RGB/多光谱/SAR传感器，忽略粗尺度传感器和大气任务

这些限制阻碍了将EO与天气和气候研究相结合的通用多模态基础模型的发展。本文旨在通过数据、模型和基准三方面的贡献来突破这些壁垒。

## 方法详解

### 整体框架

项目包含三个协同组件：**Copernicus-Pretrain**（大规模预训练数据集）→ **Copernicus-FM**（统一基础模型）→ **Copernicus-Bench**（系统评估基准）。模型采用ViT为骨干，通过动态超网络处理多模态输入，以掩码图像建模（MIM）+持续蒸馏为训练目标。

### 关键设计

1. **Copernicus-Pretrain数据集**: 按ERA5再分析数据集的 $0.25° \times 0.25°$ 网格划分全球约310K个网格单元，覆盖8种Sentinel模态：

    - Sentinel-1 GRD（SAR，10m，264×264×2，约420万图像）
    - Sentinel-2 TOA（多光谱，10m，264×264×13，约420万图像）
    - Sentinel-3 OLCI（多光谱，300m，96×96×21，约220万图像）
    - Sentinel-5P（大气变量：CO/NO2/SO2/O3，1km，28×28，约780万图像）
    - Copernicus DEM（高程，30m，960×960，约30万图像）
   
   总计约1870万图像，是目前最大最多样的EO预训练数据集。采用高斯采样策略在全球前10K人口密集城市周围采样S1/S2局部块，同时覆盖极地区域。

2. **动态Patch嵌入的传感器感知超网络**: 解决不同模态输入尺寸和通道数差异的关键模块：

    - **光谱超网络**：将每个通道的中心波长 $\lambda$ 和带宽 $\delta$ 通过Fourier编码映射到 $D$ 维向量，再通过MLP和多头注意力生成卷积核权重 $\mathbf{K}_{\text{conv}} \in \mathbb{R}^{D \times C \times p \times p}$
   
    $\text{FE}(x) = [\cos \frac{2\pi x}{\omega_i}, \sin \frac{2\pi x}{\omega_i}], \quad \omega_i = \exp(\log \omega_{\min} + i \cdot \frac{\log \omega_{\max} - \log \omega_{\min}}{D/2-1})$

    - **变量超网络**（创新点）：非光谱模态（如S5P大气成分、DEM高程）无波长属性。使用冻结的Llama 3.2 LLM编码变量名称为 $D$ 维向量，通过类似MLP管道生成对应的patch嵌入权重。这是零额外推理成本的一次性预处理。
    - **FlexiViT动态patch尺寸**：针对不同GSD（10m到1km）动态调整卷积核的patch尺寸（S1/2用16×16，S3用8×8，S5P用4×4，DEM用64×64）

3. **元数据集成的统一Fourier编码**: 除位置编码外，引入三类可选元数据编码，均用Fourier编码统一处理：

    - **地理位置**：经纬度编码拼接为 $\text{Loc} \in \mathbb{R}^D$
    - **空间覆盖面积**：根据GSD和patch尺寸计算面积编码 $\text{Area} \in \mathbb{R}^D$
    - **时间**：距参考日期的天数编码 $\text{Time} \in \mathbb{R}^D$
    - 训练时以0.7概率随机丢弃元数据，使用可学习token作为缺失替代

### 损失函数 / 训练策略

- **掩码图像建模（MIM）**：MAE式70%掩码率，对每个模态独立重建被掩码patch
- **持续蒸馏**：以DINOv2和SoftCon为教师，分别蒸馏S2-RGB和S1/S2表示，损失权重0.1和0.2
- ViT-Base，100 epochs，220K全模态网格子集
- 数据增强：随机裁剪缩放+水平翻转

## 实验关键数据

### 主实验（Copernicus-Bench）

15个下游任务的代表性结果（frozen encoder评估）：

| 任务 | 指标 | Random | SoftCon | CROMA | DOFA | **Copernicus-FM** |
|------|------|--------|---------|-------|------|-------------------|
| EuroSAT-S1 | OA↑ | 75.4 | 83.6 | 83.9 | 81.7 | **87.2** |
| EuroSAT-S2 | OA↑ | 92.5 | 96.7 | 97.0 | 97.2 | **97.9** |
| BigEarthNet-S1 | mAP↑ | 63.8 | **78.7** | 70.8 | 70.5 | 77.9 |
| LC100Cls-S3 | mAP↑ | 88.9 | - | - | 89.5 | **93.3** |
| LC100Seg-S3 | mIoU↑ | 18.2 | - | - | 16.5 | **24.1** |
| AQ-NO2-S5P | RMSE↓ | 3.4 | - | - | 3.3 | **2.8** |
| AQ-O3-S5P | RMSE↓ | 1741.6 | - | - | 1755.6 | **789.4** |

在S3和S5P任务上提升尤为显著（如O3预测RMSE从1741.6→789.4），11/15任务超过有监督训练。

### 消融实验

逐步添加组件的消融结果：

| 组件 | EuroSAT-S1 | EuroSAT-S2 | EuroSAT-RGB | LC100-S3 | AQ-O3-S5P |
|------|------------|------------|-------------|----------|-----------|
| Baseline (DOFA+动态patch) | 56.3 | 87.6 | 62.2 | 86.7 | 2218.0 |
| + 带宽Fourier编码 | 56.5 | 88.9 | 65.4 | 87.1 | 1710.7 |
| + 变量超网络 | 57.5 | 88.9 | 65.8 | 86.6 | 1598.1 |
| + 元数据编码 | **77.9** | 88.9 | **78.5** | **90.7** | **839.3** |
| + 持续蒸馏 | **81.0** | **89.5** | **78.9** | 90.7 | **811.6** |

元数据编码带来最显著提升（EuroSAT-S1: 57.5→77.9, +20.4），特别是在非光学模态上。这强调了元数据（地理位置等）对遥感应用的关键重要性。

### 关键发现

- 跨模态预训练同时提升地表和大气应用性能
- 元数据编码的贡献远超模型架构改进，位置信息是所有元数据中最重要的
- 元数据丢弃概率0.7最优（越高越好，鼓励模型不依赖元数据）
- 网格嵌入用于气候参数预测时，可补充地理坐标的不足（温度预测RMSE从3.99降至1.98）
- LLM编码的变量名称能够为非光谱模态提供有意义的初始化

## 亮点与洞察

- **打破EO传统壁垒**：首次将地表和大气观测统一到同一预训练框架中
- **LLM辅助的变量超网络**：用语言模型编码非光谱变量名称的思路简洁优雅，零额外成本
- **ERA5对齐的数据组织**：网格化设计天然连接EO和天气/气候数据，为跨领域研究铺路
- **Copernicus-Bench填补空白**：15个任务覆盖三级应用层次（预处理→基础应用→专业应用），6个是新策划数据集
- **全球网格嵌入数据集**：Copernicus-Embed-025deg为气候建模提供了语义丰富的地理表征

## 局限与展望

- 仅限Sentinel系列，未纳入Landsat、MODIS等其他重要卫星数据
- 时间范围约1年（2021年前后），缺乏长时间序列建模能力
- 未原生支持多模态融合和时间序列处理，目前为独立编码
- ViT-Base规模有限，更大模型（Large/Huge）的扩展效果待验证
- Copernicus-Bench中部分新数据集规模较小（如S5P任务仅~1500样本）

## 相关工作与启发

- DOFA的波长条件动态patch嵌入是本文模型的基础，本文将其扩展到带宽和非光谱输入
- SatCLIP提供位置编码器的对比，但Copernicus-FM的网格嵌入在气候预测上表现更好
- MMEarth整合了多源数据但模态有限；本文的Sentinel全覆盖更为系统
- 本文的网格嵌入可直接用于天气预报模型的静态变量扩展

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 统一所有Sentinel任务的框架在EO领域具有开创性意义
- **实验充分度**: ⭐⭐⭐⭐⭐ 数据集1870万、模型全面消融、15任务基准、气候应用探索
- **写作质量**: ⭐⭐⭐⭐ 三组件结构清晰，但信息量极大导致部分细节需看附录
- **价值**: ⭐⭐⭐⭐⭐ 数据集+模型+基准的组合贡献对EO社区有重大推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[CVPR 2026\] GeoDiT: A Diffusion-based Vision-Language Model for Geospatial Understanding](../../CVPR2026/remote_sensing/geodit_a_diffusion-based_vision-language_model_for_geospatial_understanding.md)
- [\[ICML 2025\] MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models](../../ICML2025/remote_sensing/mapeval_a_map-based_evaluation_of_geo-spatial_reasoning_in_foundation_models.md)

</div>

<!-- RELATED:END -->
