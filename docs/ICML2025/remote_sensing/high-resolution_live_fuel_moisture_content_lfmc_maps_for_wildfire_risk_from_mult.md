---
title: >-
  [论文解读] High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data
description: >-
  [ICML 2025][遥感][LFMC] 利用预训练多模态地球观测模型 Galileo 微调生成 10 米分辨率的活体燃料含水量（LFMC）地图，相比随机初始化模型 RMSE 降低 20%+，并通过 2025 年洛杉矶野火案例验证了管线的实用性。 领域现状：活体燃料含水量（Live Fuel Moisture Conten…
tags:
  - "ICML 2025"
  - "遥感"
  - "LFMC"
  - "Wildfire Risk"
  - "foundation model"
  - "Remote Sensing"
  - "Galileo"
---

# High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data

**会议**: ICML 2025  
**arXiv**: [2506.20132](https://arxiv.org/abs/2506.20132)  
**代码**: [github.com/allenai/lfmc](https://github.com/allenai/lfmc)  
**领域**: 遥感  
**关键词**: LFMC, Wildfire Risk, foundation model, Remote Sensing, Galileo

## 一句话总结

利用预训练多模态地球观测模型 Galileo 微调生成 10 米分辨率的活体燃料含水量（LFMC）地图，相比随机初始化模型 RMSE 降低 20%+，并通过 2025 年洛杉矶野火案例验证了管线的实用性。

## 研究背景与动机

**领域现状**：活体燃料含水量（Live Fuel Moisture Content, LFMC）是衡量活体植被含水量的关键指标，直接影响野火的点燃概率、燃料可用性和火灾蔓延。LFMC 计算公式为 $\text{LFMC}[\%] = \frac{W_f - W_d}{W_d} \times 100$，其中 $W_f$ 为新鲜植物重量，$W_d$ 为干燥后重量。LFMC 值越低，野火风险越高。

**现有痛点**：地面 LFMC 采样非常耗时耗力——需要到现场采集植物样本、称重、烘干 1-2 天再称重，单个站点就需要 12 小时到 4 天。受限于此，现有采样点在空间和时间上都很稀疏，无法提供完整覆盖的 LFMC 评估。

**核心矛盾**：现有基于机器学习的 LFMC 估算方法（如 Rao et al. 2020, Miller et al. 2023）采用全监督随机初始化模型，空间分辨率仅 250-500 米，且受限于标签不平衡和异质性问题，泛化性能有限。

**切入角度**：利用在大规模多模态遥感数据上预训练的地球观测基础模型（Galileo），通过微调实现高精度、高分辨率（10 米）的 LFMC 预测，将分辨率提升 25-50 倍。预训练带来的先验知识可提升对缺失输入的鲁棒性和时空泛化能力。

**核心 idea**：预训练的多模态遥感基础模型 + Globe-LFMC 2.0 数据集微调 = 一条自动化的高分辨率 LFMC 制图管线。

## 方法详解

### 整体框架

管线包含三个阶段：
1. **训练数据构建**：从 Globe-LFMC 2.0 数据集筛选 CONUS 2017-2023 年的样本，为每个样本从 Google Earth Engine 导出对应区域的多模态遥感数据
2. **模型微调**：在 Galileo-Tiny 预训练权重基础上，用 MSE 损失微调 LFMC 回归模型
3. **地图生成**：给定时空区域 → 导出遥感数据 → 模型推理 → 输出 10 米分辨率 LFMC 地图

### 关键设计

1. **Globe-LFMC 2.0 数据集处理**：筛选得 41,214 个样本，覆盖 1,031 个站点。同一地点同一天的多个样本取平均。LFMC 值在 302%（99.9 分位数）处截断和归一化以抑制异常值。按 70%/15%/15% 随机划分训练/验证/测试集。数据覆盖 15-3187 米海拔、多种地表覆盖类型和四季。

2. **Galileo 预训练模型**：基于 Vision Transformer 的多模态遥感基础模型（Galileo-Tiny, 5.3M 参数），可处理 10 种遥感产品。输入模态包括：

    - Sentinel-2 多光谱光学数据（可见光、近红外、短波红外 + NDVI）
    - Sentinel-1 SAR 数据（VV/VH 极化）
    - VIIRS 夜间灯光
    - ERA-5 气象数据（降水、温度）
    - TerraClimate 水平衡数据（气候水亏缺、土壤湿度、蒸发蒸腾）
    - SRTM 地形数据（海拔、坡度）
    - 经纬度位置编码
   
   不同模态空间分辨率差异大（10m 到数十 km），时间分辨率也不同（5 天到月度）。Galileo 根据输入是否在空间/时间上变化进行分类处理，时间维度统一到月度尺度。

3. **微调策略**：MSE 损失，最多 100 个 epoch，early stopping（验证集 5 个 epoch 无提升即停止）。单卡 H100 训练约 30-60 分钟。默认输入形状为 32×32 像素、12 个时间步。

### 损失函数 / 训练策略

- 损失函数：均方误差（MSE），直接回归归一化后的 LFMC 值
- 数据预处理：LFMC 截断在 302%，按 99.9 分位数归一化
- 训练设备：单卡 NVIDIA H100，训练时间 30-60 分钟
- Early stopping 策略：验证集连续 5 epoch 无提升则停止

## 实验关键数据

### 主实验

| 模型 | RMSE↓ | MAE↓ | R²↑ | 说明 |
|------|-------|------|-----|------|
| **GalileoLFMC (预训练)** | **18.91** | **12.58** | **0.72** | 本文方法 |
| 随机初始化模型 | 23.61 | 16.33 | 0.57 | 传统全监督方法 |
| 月均值预测 | 33.66 | 25.38 | 0.11 | 简单基线 |

预训练模型相比随机初始化 RMSE 降低 20%（23.61→18.91），R² 提升 26%（0.57→0.72）。

### 消融实验

**输入形状敏感性**：

| 空间 (H×W) | 时间步 T | RMSE | R² | 说明 |
|------------|---------|------|-----|------|
| 32×32 | 12 | 18.91 | 0.72 | 默认配置 |
| 32×32 | 3 | 19.45 | 0.70 | 时间减少影响较小 |
| 1×1 | 12 | 20.25 | 0.68 | 仅点级空间信息 |

**缺失输入鲁棒性**（预训练 vs 随机初始化）：

| 移除的输入 | 预训练 RMSE | 预训练 R² | 随机 RMSE | 随机 R² |
|-----------|------------|----------|----------|---------|
| 无 | 18.91 | 0.72 | 23.61 | 0.57 |
| TerraClimate | 19.51 | 0.70 | 25.57 | **0.49** (-14%) |
| 位置编码 | 20.08 | 0.69 | 23.80 | 0.56 |

预训练模型移除任何单一模态后性能波动小（RMSE 变化 <1.2），而随机初始化模型移除 TerraClimate 后 R² 暴跌 14%。

### 关键发现

- **跨季节泛化**：尽管冬季训练样本最少，模型在冬季的 RMSE（15.31）反而最低，R²（0.77）最高，说明预训练帮助了时间泛化
- **跨地表类型一致性**：不同地表覆盖类型（树木、草地、灌木、建筑区、裸地）的 RMSE 范围为 16.79-20.52，表现一致
- **高海拔退化**：2000 米以下 R² > 0.7，但 3000-3500 米 R² 降至 0.32，与高海拔训练样本不足（仅 444 个）有关
- **空间自相关**：Moran's I = 0.057（p=0.001），残差存在弱正空间自相关，提示随机划分可能有轻微信息泄漏

## 亮点与洞察

- **分辨率飞跃**：从前人的 250-500 米提升到 10 米，实现了 25-50 倍的分辨率提升，这对于小区域精细化火灾风险评估非常有价值
- **预训练的隐性价值**：不仅提升精度，更重要的是提升了对缺失输入的鲁棒性——自监督预训练让模型学会了模态间的互补关系
- **实用导向**：提供了端到端自动化管线（数据导出→推理→制图），直接服务于灾害管理和处方烧除规划
- **2025 洛杉矶野火案例**：Palisades 和 Eaton 大火的 LFMC 预测与专家实地观测一致——2023-2024 年 LFMC 高于 2021-2022 年，反映了连续两个春季降水偏多后的植被增长，可能导致了更高的燃料负荷

## 局限与展望

- 模型仅在美国大陆（CONUS）西部数据上微调，全球泛化性未验证
- 目前仅用于回溯分析，尚未测试 LFMC 预测/预报能力
- 时间分辨率为月度均值，而实际野火管理需要周度甚至日度更新
- 随机划分可能导致空间信息泄漏（邻近站点出现在训练和测试集中），未来应采用空间分区策略
- 高海拔区域（>3000m）性能显著下降，需获取更多高海拔标签数据

## 相关工作与启发

- **Rao et al. (2020)**：用物理辅助的 RNN 在 250 米分辨率映射 LFMC，验证了多模态遥感输入的价值
- **Miller et al. (2023)**：用 tempCNN 在 500 米分辨率预测 LFMC，加入了预报能力（3 个月 lead time）
- **Galileo (Tseng et al., 2025)**：本文核心依赖的预训练地球观测基础模型，能处理 10 种遥感产品
- **Jolly et al. (2024)**：美国国家火灾危险等级系统的物理模型方法，使用气象驱动的 GSI 模型
- 启发：遥感基础模型在下游任务中的核心价值不仅是精度提升，更在于鲁棒性和数据效率的改善

## 评分

- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation](../../CVPR2026/remote_sensing/ramen_resolution-adjustable_multimodal_encoder_for_earth_observation.md)
- [\[CVPR 2026\] OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation](../../CVPR2026/remote_sensing/olmoearth_stable_latent_image_modeling_for_multimodal_earth_observation.md)
- [\[CVPR 2026\] YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction](../../CVPR2026/remote_sensing/yieldsat_a_multimodal_benchmark_dataset_for_high-resolution_crop_yield_predictio.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/remote_sensing/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)
- [\[ICML 2025\] LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth](lighthouse_fast_and_precise_distance_to_shoreline_calculations_from_anywhere_on_.md)

</div>

<!-- RELATED:END -->
