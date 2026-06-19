---
title: >-
  [论文解读] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs
description: >-
  [ICCV 2025][多模态VLM][3D空间理解] Apple 提出 CA-VQA 数据集和 MM-Spatial 模型，利用高质量 3D 场景数据和开放集标注生成涵盖空间关系预测、度量估计和 3D grounding 的训练/评估数据集，训练出一个通用型 MLLM，在 3D 空间理解 benchmark 上达到 SOTA，同时保持其他任务的竞争力。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "3D空间理解"
  - "多模态LLM"
  - "深度估计"
  - "多视图"
  - "空间推理"
---

# MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs

**会议**: ICCV 2025  
**arXiv**: [2503.13111](https://arxiv.org/abs/2503.13111)  
**代码**: [GitHub](https://github.com/apple/ml-cubifyanything)  
**领域**: 多模态VLM  
**关键词**: 3D空间理解, 多模态LLM, 深度估计, 多视图, 空间推理

## 一句话总结

Apple 提出 CA-VQA 数据集和 MM-Spatial 模型，利用高质量 3D 场景数据和开放集标注生成涵盖空间关系预测、度量估计和 3D grounding 的训练/评估数据集，训练出一个通用型 MLLM，在 3D 空间理解 benchmark 上达到 SOTA，同时保持其他任务的竞争力。

## 研究背景与动机

MLLM 在 2D 视觉理解上表现优异，但 3D 空间推理能力仍然有限：

**3D 感知的关键缺口**：现有 MLLM 难以完成以下任务：（1）相对深度判断（"前方"vs"后方"）；（2）度量单位的距离/尺寸估计（"A 距离 2.74m"）；（3）精确 3D 边界框

**现有数据集的不足**：已有的 3D 空间数据集存在多种限制——仅覆盖部分任务、缺乏高质量 3D 真值、不包含深度图、不支持多视图，且未同时提供训练集和评测集

**深度和多视图输入的研究不充分**：很少有工作全面评估不同类型的深度图（传感器 vs. 单目估计 vs. GT）以及多视图输入对 3D 空间理解的影响

作者希望通过一个全面的数据集来系统性地推进 MLLM 的 3D 空间理解研究。

## 方法详解

### 整体框架

1. **数据生成管线**：基于 Cubify Anything 1M（CA-1M）数据集（包含 ARKitScenes 中每个物体的 7-DOF 3D 边界框 + 开放集语义标注），自动生成模板化 QA 对
2. **CA-VQA 数据集**：约 1000 万个 QA 对，覆盖 220K 帧、2K 视频用于训练；约 6.2 万个 QA 对、2.6K 帧用于评测
3. **MM-Spatial 模型**：基于 MM1.5-3B 架构（DFN-CLIP 视觉编码器 + decoder-only LLM），通过 SFT 获得空间理解能力

### 关键设计

1. **CA-VQA 数据集与 Benchmark**

   覆盖六大空间任务类别：
    - **计数**："场景中有几把椅子？"
    - **视角相关关系**："X 在 Y 后面吗？"（依赖相机位姿）
    - **度量回归**："X 到 Y/相机的距离？" "X 有多宽/高？"
    - **2D/3D Referring & Grounding**
    - **二选一 & 多选题**

   独特之处：
    - 基于高精度 FARO 激光扫描仪的 3D 真值（不是伪标注）
    - 每帧提供三种深度图：GT 深度、ARKit 深度（iPad LiDAR）、单目深度（DepthPro）
    - 多视图支持：每个参考帧最多 4 个支持帧，附带相对位姿和相机内参
    - **盲过滤策略**：使用 7 个 MLLM 作为评判者，移除不需要视觉输入即可正确回答的样本，减少语言先验偏差

2. **深度利用方式：CoT / Tool-Use**

   提出两种利用度量深度的策略（非直接编码深度图）：
    - **Tool-Use**：模型预测物体的 2D 边界框和函数调用，工具返回框内中值深度值作为文本，模型据此推理
    - **CoT（Chain-of-Thought）**：训练时提供包含 GT 深度的逐步推理示例，测试时模型自行预测深度值

   设计动机：全图编码深度只能使用归一化的相对深度，而 CoT/Tool-Use 可以利用绝对度量深度。且 CoT 模式不需要额外工具，模型通过 SFT 学会了准确预测深度。

3. **多视图输入**

   利用 MM1.5 的多图像输入能力，将支持帧与参考帧拼接为序列 $I_{t-N}, ..., I_{t-1}, I_t$，并以 JSON 格式提供每帧的相机内参和相对位姿。仅对参考帧应用图像分割（image splitting）。

   设计动机：多视图提供了额外的几何约束信息，有助于消除单视角下的深度模糊。

### 损失函数 / 训练策略

- 采用 MM1.5 的三阶段训练：预训练 → 持续预训练 → SFT
- SFT 阶段在 MM1.5 的基础数据混合（General VQA, Knowledge, Text-rich, 2D Ref./Grounding）中新增 Spatial 类别（CA-VQA 数据）
- 通过调节混合比例确保空间任务提升不损害其他能力
- 图像分辨率 672×672，4 个子图 + 1 个全局图
- 视觉编码器和 LLM 均不冻结

## 实验关键数据

### 主实验

**CA-VQA Benchmark 结果（各任务平均得分）**：

| 模型 | Binary | Count. | 2D AP@50 | 3D AP@15 | Multi-c. | Ego-Dist. | Obj-Size | 平均 |
|------|--------|--------|----------|----------|----------|-----------|----------|------|
| GPT-4o | 44.2 | 69.0 | 0.0 | 0.0 | 36.6 | 11.7 | 11.0 | 22.8 |
| SpatialRGPT-8B | 53.6 | 68.8 | 5.5 | 0.0 | 37.2 | 10.5 | 7.0 | 23.9 |
| MM1.5-3B | 59.1 | 9.1 | 32.6 | 0.0 | 38.6 | 0.6 | 3.4 | 18.2 |
| **MM-Spatial-3B** | **68.8** | **75.8** | **53.2** | **20.7** | **74.2** | **40.0** | **24.4** | **47.0** |
| +CoT | 69.6 | 75.9 | 54.5 | 21.9 | 74.7 | 46.0 | 26.7 | 49.1 |
| +Multi-view+CoT | 69.2 | 76.1 | 55.0 | 23.6 | 75.3 | 46.1 | 28.2 | 49.7 |
| +Multi-view+Tool(GT) | 69.2 | 76.1 | 55.0 | 23.6 | 75.3 | **65.8** | 27.3 | **52.4** |

MM-Spatial-3B（仅 3B 参数）在所有任务上大幅超越 GPT-4o 和 SpatialRGPT-8B。

**跨 Benchmark 类别结果**：

| 模型 | Spatial | General | Knowledge | Text-rich | Ref./Ground | 平均 |
|------|---------|---------|-----------|-----------|-------------|------|
| MM1.5-3B | 39.9 | 64.7 | 46.2 | 62.1 | 77.7 | 58.1 |
| **MM-Spatial-3B** | **70.1** | 65.0 | 46.2 | 62.1 | 79.1 | **64.5** |

空间能力大幅提升（+30.2），其他类别保持不变或微提升。

### 消融实验

**Specialist Model 各配置对比（CA-VQA）**：

| 配置 | Ego-Dist. @10% | Obj-Dist. @10% | Obj-Size @10% | 平均 |
|------|---------------|---------------|---------------|------|
| MM-Spatial | 47.3 | 24.4 | 24.3 | 49.4 |
| +CoT（自行预测深度） | 49.5 | 27.9 | 26.7 | 50.8 |
| +Depth(Tool; Mon.) | 42.1 | 26.1 | 26.1 | 49.5 |
| +Depth(Tool; GT) | **74.0** | **32.4** | 27.4 | **54.5** |
| +Depth(Encoded; GT) | 48.3 | 25.4 | 24.5 | 49.9 |
| +Multi-view | 52.4 | 26.2 | 26.1 | 51.4 |
| +Multi-view+CoT | 55.2 | 29.7 | **28.6** | 52.7 |

CoT 模型自行预测的深度几乎与 GT 深度 tool-use 的效果相当，说明模型成功学会了单目深度估计。

### 关键发现

1. **数据驱动的深度估计**：纯通过 SFT 数据训练，MM-Spatial 就能实现接近专用单目深度估计模型的性能——这是一个令人意外的发现
2. **多视图一致性有帮助**：多视图在所有配置中都有正向提升，尤其在 3D grounding（AP@15: 24.2→27.5）和距离估计上
3. **全图编码深度不如 CoT**：将深度图通过视觉编码器编码（只能用相对深度）效果弱于 CoT 的文本化绝对深度
4. **单目估计深度 < GT 深度**：使用 DepthPro 的单目深度做 tool-use 时效果弱于 GT 深度，说明深度精度是上限因素
5. **盲过滤有效**：移除可被盲模型回答的样本后，benchmark 更具挑战性和可靠性

## 亮点与洞察

- **全面性无出其右**：CA-VQA 是第一个同时提供高质量 3D 真值、三种深度图、多视图、多类任务和训练/评测拆分的空间理解数据集
- **通用模型不损失能力**：仅 3B 参数的 MM-Spatial 在空间任务上大幅超越 GPT-4o，同时在通用/知识/文本类任务上保持性能
- **CoT 深度估计的启示**：模型可以通过数据学会深度感知，不需要显式的深度传感器——这对边缘设备部署意义重大
- **盲过滤策略值得借鉴**：7 个模型联合过滤的方法可以推广到其他 benchmark 构建中，减少语言先验偏差

## 局限与展望

- 数据集限于室内场景（ARKitScenes），户外泛化能力有待验证
- 仅探索了 3B 模型，更大规模模型（7B, 70B）的效果未知
- 3D grounding 的 AP@15 绝对值仍然较低（最高 27.5），有较大改进空间
- 物体间距离估计的 10% 相对误差阈值下准确率仅 ~30%，实际应用需要更高精度
- 多视图的帧选择策略（角度 ≥15° 或平移 ≥30cm）是否最优未做充分消融

## 相关工作与启发

本文在 SpatialRGPT、Cube-LLM、SpatialBot 等先前工作基础上做了全面升级。核心差异在于数据质量（精确 3D 标注 vs. 伪标注）和输入信号的多样性（多视图 + 三种深度）。CoT 深度估计的成功启示了"通过语言学习视觉感知"的可能性。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据集构建管线和 CoT 深度估计是亮点，但模型架构本身没有新设计
- **实验充分度**: ⭐⭐⭐⭐⭐ 极其全面：多种模型变体、多个 benchmark、消融分析、盲过滤验证
- **写作质量**: ⭐⭐⭐⭐⭐ 条理清晰，表格和图示丰富，数据集对比表一目了然
- **价值**: ⭐⭐⭐⭐⭐ 数据集和 benchmark 将推动 MLLM 3D 空间理解的后续研究，且代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](../../CVPR2025/multimodal_vlm/robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)

</div>

<!-- RELATED:END -->
