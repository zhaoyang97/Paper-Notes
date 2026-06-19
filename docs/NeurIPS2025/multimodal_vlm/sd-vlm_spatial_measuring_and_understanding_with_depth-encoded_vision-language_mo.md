---
title: >-
  [论文解读] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][空间推理] 提出MSMU大规模定量空间推理数据集（700K QA对、250万数值标注）和深度位置编码（DPE）方法，使VLM在不引入3D点云的前提下获得强大的定量空间测量和理解能力，在MSMU-Bench上超越GPT-4o达26.91%。 VLM在2D语义理解上表现出色…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "空间推理"
  - "深度编码"
  - "VLM"
  - "定量空间理解"
  - "3D感知"
---

# SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.17664](https://arxiv.org/abs/2509.17664)  
**代码**: [GitHub](https://github.com/cpystan/SD-VLM)  
**领域**: 多模态VLM  
**关键词**: 空间推理, 深度编码, VLM, 定量空间理解, 3D感知

## 一句话总结

提出MSMU大规模定量空间推理数据集（700K QA对、250万数值标注）和深度位置编码（DPE）方法，使VLM在不引入3D点云的前提下获得强大的定量空间测量和理解能力，在MSMU-Bench上超越GPT-4o达26.91%。

## 研究背景与动机

VLM在2D语义理解上表现出色，但在**定量空间推理**方面严重不足——即回答"桌子有多大？""两个物体间距离多少？"等需要精确数值的问题。这一能力对机器人、自动驾驶、增强现实等真实世界应用至关重要。

核心矛盾在于：图像是3D场景到2D平面的投影，丢失了大量3D结构信息。要从2D图像恢复3D空间关系，理论上需要深度图 $d$ 和相机内参 $\mathbf{K}$：

$$\mathbf{P} = d \cdot \mathbf{K}^{-1} \mathbf{p}$$

现有解决方案的两大局限：

**数据匮乏**：现有空间数据集（如SpatialVLM、SpatialRGPT）的定量标注依赖模型估计（检测+分割+深度估计+相机标定），存在系统性误差

**深度融合方式粗糙**：将深度图作为额外图像或拼接token处理，效果有限

本文的核心洞察：如果提供足够多的精确物理测量数据，VLM可以隐式学习"相机内参"，建立从2D到3D的映射。理论上，至少四条线段长度可以约束标定相机内参。因此，作者(1)用真实3D场景数据构建大规模精确标注数据集，(2)设计简洁高效的深度位置编码方式。

## 方法详解

### 整体框架

SD-VLM基于LLaVA-1.5-7B构建，包含：CLIP视觉编码器提取图像特征、深度位置编码模块融入深度信息、LLM处理token序列生成答案。当真实深度图不可用时，使用Depth-Anything-V2估计深度。

### 关键设计

1. **MSMU数据集构建**

   从ScanNet和ScanNet++的真实3D场景出发，构建精确的空间标注数据。数据生成流程：
    - **场景图构建**：从3D点云提取物体的类别和3D包围盒（中心坐标+长宽高）
    - **3D到2D映射**：用官方工具将3D实例光栅化为2D图像mask，建立物体的3D-2D对应
    - **过滤**：排除不常见物体、被遮挡/截断物体、语义歧义物体（用Qwen2.5-VL重标注为"白色桌子""木桌"等）
    - **模板化QA生成**：覆盖尺度估计、物体定位、距离测量、大小比较、参考物推理、计数、存在性判断7类任务
    - **LLM协作CoT增强**：随机选参考物体，用Qwen2.5-VL生成推理路径，再用DeepSeek-V3评估质量

   最终数据集包含2K场景、25K图像、75K物体、**700K QA对、250万数值标注**和10K CoT推理样本。

2. **深度位置编码（Depth Positional Encoding, DPE）**

   核心设计极其简洁。将深度图 $\mathbf{D} \in \mathbb{R}^{H \times W \times 1}$ 按图像patch分割并均值池化得到 $\mathbf{D}' \in \mathbb{R}^{H' \times W' \times 1}$，然后用正弦/余弦函数生成深度位置嵌入：

    $\mathbf{E}^{\text{depth}}(i,j,2t) = \sin\left(\frac{\mathbf{D}'(i,j)}{10000^{2t/d}}\right)$
    $\mathbf{E}^{\text{depth}}(i,j,2t+1) = \cos\left(\frac{\mathbf{D}'(i,j)}{10000^{2t/d}}\right)$

   直接与图像特征**相加**：

    $\mathbf{E}^{\text{vision}} = \mathbf{E}^{\text{image}} + \mathbf{E}^{\text{depth}}$

   设计动机：借鉴Transformer位置编码的成功，DPE将深度信息编码为沿z轴（垂直于图像平面）的第三维度的位置信号，将模型的空间感知从2D升级到3D。这种方式不增加序列长度、不引入额外模块、训练成本最低。

3. **MSMU-Bench评测基准**

   从MSMU中留出约1K来自未见场景的样本，覆盖所有7类空间任务。使用GPT-4进行评判：定性问题打0-1分，定量问题计算比值 $\delta = \max(\hat{d}/d^*, d^*/\hat{d})$，设阈值1.25判断成功率。

### 损失函数 / 训练策略

基于LLaVA-1.5-7B用LoRA在MSMU上微调1 epoch。视觉编码器冻结，LLM和projector分别用2e-4和2e-5学习率。8块V100训练32 GPU小时。

## 实验关键数据

### 主实验：MSMU-Bench结果

| 模型 | 存在性 | 尺度估计 | 绝对距离 | 参考物推理 | 平均 |
|------|--------|---------|---------|-----------|------|
| GPT-4o | 44.68 | 3.86 | 20.00 | 2.09 | 32.28 |
| Gemini-2 | 38.30 | 23.94 | 12.50 | 18.85 | 35.17 |
| InternVL3-78B | 47.62 | 6.47 | 13.33 | 16.46 | 33.63 |
| SpatialRGPT | 10.64 | 20.08 | 15.00 | 9.95 | 28.98 |
| **SD-VLM** | **87.23** | **51.35** | **40.00** | **46.07** | **56.31** |
| **SD-VLM + CoT** | **87.23** | **51.74** | **50.00** | **49.32** | **59.19** |

### 深度融合方式对比

| 方法 | MSMU-Bench成功率 | 说明 |
|------|-----------------|------|
| 无深度（Baseline） | 46.73% | LLaVA-1.5-7B |
| + depth as image | 22.64% | 反而下降，编码器不适合深度图 |
| + depth as prompt | 48.78% | 轻微提升 |
| + depth as token | 35.72% | 序列增长反而有害 |
| + DPE (估计深度) | 55.35% | 显著提升 |
| **+ DPE (sincos)** | **56.31%** | 最优方案 |

### 关键发现

- SD-VLM在Q-Spatial++上达56.2%，SpatialRGPT-Bench定量任务上达33.3%，均为SOTA
- 即使不用空间数据、只在LLaVA-mix665k通用数据上训练，DPE仍能带来25%的相对提升
- 使用不同深度估计器（DepthAnything vs UniDepth），性能几乎一致（48.6% vs 47.6%）
- 加入高斯噪声($\delta$=0.7)后性能仅从56.3%降至51.4%，远优于无深度的46.7%
- 加入空间数据训练不影响通用VQA性能（VQA-v2: 79.1% vs 78.5%）

## 亮点与洞察

- **深度位置编码的极简设计**：仅需一个sincos函数+加法操作，无新参数、无序列增长，效果却最好
- 用真实3D场景（非模型估计）构建数据集，确保了数值标注的精确性，避免了系统性偏差
- MSMU覆盖7类任务非常全面，从基础的尺度估计到需要推理的参考物估计都有覆盖
- 理论分析严谨：4条线段可标定内参的证明为"提供足够物理测量让VLM隐式学习"提供了理论支撑

## 局限与展望

- 仅基于LLaVA-1.5-7B（较旧架构），未在更强VLM（如Qwen2.5-VL、InternVL3）上验证DPE
- MSMU数据集局限于室内场景（ScanNet），室外空间推理的覆盖不足
- 深度估计模型的精度是性能上界，更精确的深度估计可带来进一步提升
- DPE用均值池化压缩深度信息，可能损失局部深度变化

## 相关工作与启发

- SpatialRGPT首先将深度图作为额外图像输入VLM，SpatialBot将深度信息文本化
- Q-Spatial提供了精确的人工标注基准但数据量很小
- Depth-Anything-V2等单目深度估计模型的进步是DPE实际可部署的前提
- 启发：位置编码是融合额外模态信息的轻量级通用方案，可推广到温度场、语义场等

## 评分

- **新颖性**: ⭐⭐⭐⭐ DPE设计简洁而有效，MSMU数据集是重要贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ 多基准测试、多深度融合方式对比、鲁棒性分析、通用性验证
- **写作质量**: ⭐⭐⭐⭐ 理论分析到位，实验组织清晰
- **价值**: ⭐⭐⭐⭐⭐ 数据集和方法均开源，对空间VLM领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](../../CVPR2025/multimodal_vlm/robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[CVPR 2025\] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](../../CVPR2025/multimodal_vlm/florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)

</div>

<!-- RELATED:END -->
