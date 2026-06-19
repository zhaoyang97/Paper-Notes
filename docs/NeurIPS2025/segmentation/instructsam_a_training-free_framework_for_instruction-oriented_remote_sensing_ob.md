---
title: >-
  [论文解读] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition
description: >-
  [NeurIPS 2025][语义分割][遥感] 定义指令导向目标计数/检测/分割(InstructCDS)新任务，构建EarthInstruct遥感基准（覆盖开放词汇/开放端/开放子类三种设置），提出InstructSAM——无需训练的框架：LVLM解析指令+计数、SAM2生成掩码提议、CLIP计算相似度，通过二进制整数规划(BIP)在计数约束下实现掩码-标签最优匹配，推理时间近乎恒定且优于专用基线。
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "遥感"
  - "指令导向"
  - "免训练"
  - "SAM2"
  - "二进制整数规划"
  - "目标计数"
  - "开放词汇检测/分割"
---

# InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition

**会议**: NeurIPS 2025  
**arXiv**: [2505.15818](https://arxiv.org/abs/2505.15818)  
**代码**: [https://VoyagerXvoyagerx.github.io/InstructSAM/](https://VoyagerXvoyagerx.github.io/InstructSAM/)  
**领域**: 遥感  
**关键词**: 遥感, 指令导向, 免训练, SAM2, 二进制整数规划, 目标计数, 开放词汇检测/分割  

## 一句话总结
定义指令导向目标计数/检测/分割(InstructCDS)新任务，构建EarthInstruct遥感基准（覆盖开放词汇/开放端/开放子类三种设置），提出InstructSAM——无需训练的框架：LVLM解析指令+计数、SAM2生成掩码提议、CLIP计算相似度，通过二进制整数规划(BIP)在计数约束下实现掩码-标签最优匹配，推理时间近乎恒定且优于专用基线。

## 研究背景与动机

**领域现状**：遥感目标识别广泛用于可持续发展目标（野生动物监测、贫困估计、灾害救援）。CLIP驱动的开放词汇检测/分割在遥感领域兴起。

**现有痛点**：
   - 现有开放词汇方法**依赖显式类别指令**，无法处理隐式推理（如"交通工具"→自动推断包含哪些子类别）
   - 遥感领域固定类别列表注定不完整——鸟瞰视角下可见目标种类多样
   - 传统检测器依赖**置信度阈值**过滤，在零样本场景下阈值无法获取
   - 直接用LVLM逐个生成检测框时，推理时间随目标数量线性增长

**核心idea**：分解为三步——LVLM计数（O(1)推理时间）+ SAM2掩码提议 + BIP最优匹配，不需要训练、不需要阈值、推理时间近乎恒定。

## InstructCDS任务与EarthInstruct基准

### 三种设置

| 设置 | 描述 | 示例指令 |
|------|------|----------|
| **开放词汇** | 用户指定目标类别 | "检测足球场、停车场" |
| **开放端** | 检测所有可见目标 | "检测图中所有目标" |
| **开放子类** | 检测父类别下所有子类 | "检测所有体育场地" |

### EarthInstruct基准
- 基于NWPU-VHR-10和DIOR两个遥感数据集，覆盖20个类别
- 两个数据集有不同的标注规则和空间分辨率——模型需理解数据集特定指令
- 例如：NWPU-VHR-10低分辨率图像中车辆不标注；DIOR中机场仅标注完整可见的

### 评估指标创新
- **计数指标**：基于TP/FP/FN的Precision/Recall/F1（替代MAE/RMSE，归一化且能区分过/欠计数）
- **无置信度检测指标**：mF1 + mAPnc（不依赖置信度排序），适用于生成式检测器
- **语义匹配**：GeoRSCLIP文本编码器计算余弦相似度>0.95视为类别等价

## 方法详解

### InstructSAM框架（三步流水线）

#### Step 1: LVLM指令解析与计数
用GPT-4o或Qwen2.5-VL-7B作为计数器，输入图像+结构化JSON提示（包含数据集特定指令），输出目标类别 $\{cat_j\}$ 和计数 $\{num_j\}$：

$$\{cat_j, num_j\}_{j=1}^M = \text{LVLM-Counter}(I, P)$$

#### Step 2: SAM2无类别掩码提议
SAM2-hiera-large在规则点网格上自动生成掩码提议 $\{mask_i\}_{i=1}^N$。对图像裁剪区域额外运行掩码生成以提高小目标召回率。

#### Step 3: 计数约束的掩码-标签匹配（核心创新）
计算语义相似度矩阵 $S \in \mathbb{R}^{N \times M}$（GeoRSCLIP的图像-文本余弦相似度），建立二进制整数规划问题：

$$\min_{\mathbf{X}} \sum_{i=1}^N \sum_{j=1}^M (1 - s_{ij}) \cdot x_{ij}$$

约束条件：
- 每个掩码至多分配给一个类别：$\sum_{j=1}^M x_{ij} \leq 1, \; \forall i$
- 每个类别分配掩码数等于计数：$\sum_{i=1}^N x_{ij} = num_j, \; \forall j$（当提议充足时）
- 提议不足时全部分配：$\sum_{i=1}^N \sum_{j=1}^M x_{ij} = N$

通过PuLP求解器高效求解。BIP优雅融合三类信息：视觉（CLIP掩码嵌入）、语义（类别文本嵌入）、定量（LVLM计数约束）。

### 关键优势
- **免训练**：无需遥感任务特定训练数据
- **免阈值**：通过计数约束替代置信度阈值——避免零样本场景下阈值选择难题
- **近乎恒定推理时间**：LVLM只输出计数（少量token），不需要逐个生成bbox；SAM2掩码提议与目标数量无关

## 实验关键数据

### 开放词汇设置（零样本）

| 方法 | NWPU Cnt-F1↑ | Box-F1↑ | Mask-F1↑ | DIOR Cnt-F1↑ | Box-F1↑ | Mask-F1↑ |
|------|-------------|---------|----------|-------------|---------|----------|
| Grounding DINO | 14.9 | 14.0 | - | 10.7 | 6.0 | - |
| OWLv2 | 39.4 | 27.2 | - | 23.4 | 14.3 | - |
| Qwen2.5-VL | 68.0 | 36.4 | - | 52.0 | 27.8 | - |
| InstructSAM-Qwen | 73.2 | 38.9 | 23.7 | 59.3 | 24.7 | 24.0 |
| **InstructSAM-GPT4o** | **83.0** | **41.8** | **26.1** | **79.9** | **29.1** | **28.1** |

InstructSAM-GPT4o在计数F1上大幅领先（83.0 vs 68.0），且同时提供分割结果。

### 开放端设置

| 方法 | NWPU Cnt-F1↑ | Box-F1↑ | DIOR Cnt-F1↑ | Box-F1↑ |
|------|-------------|---------|-------------|---------|
| Qwen2.5-VL | 48.6 | 32.0 | 36.6 | 21.7 |
| GeoPixel | 40.8 | 29.9 | 21.4 | 13.8 |
| LAE-Label | 46.2 | 27.3 | 23.3 | 11.5 |
| **InstructSAM-GPT4o** | **57.4** | **31.3** | **47.9** | **22.1** |

InstructSAM在开放端设置下也持续领先遥感专用模型（如SkySenseGPT、EarthDial、GeoPixel）。

### 开放子类设置

| 方法 | NWPU-S F1↑ | NWPU-T F1↑ | DIOR-S F1↑ | DIOR-T F1↑ |
|------|-----------|-----------|-----------|-----------|
| Qwen2.5-VL | 32.4 | 42.2 | 34.0 | 39.2 |
| GPT4o+OWL | 19.8 | **65.9** | 27.6 | **70.9** |
| **InstructSAM-GPT4o** | **46.9** | 44.2 | **40.9** | 38.3 |

InstructSAM在"体育场地"(S)子类上大幅领先，在"交通工具"(T)子类上通用检测器更强——因为OWLv2训练数据中交通工具类别丰富。

### 效率对比
- InstructSAM输出token数比Qwen2.5-VL减少**89%**
- 总运行时间减少**>32%**
- 推理时间近乎恒定，不随目标数量增长

## 亮点
1. **BIP匹配取代阈值**——将目标识别形式化为约束优化问题，零样本场景下不需要调阈值
2. **计数作为全局约束**——LVLM的全局视角提供准确计数，约束掩码分配
3. **免训练泛化**——组合三个foundation model（LVLM+SAM2+CLIP），无需遥感特定训练
4. **新任务+新基准**——InstructCDS三设置 + EarthInstruct基准 + 无置信度评估指标

## 局限与展望
1. LVLM计数准确性是上限——GPT-4o计数F1仅83%，错误会传播到最终检测
2. 依赖GeoRSCLIP的遥感特定预训练——换到自然图像域需替换CLIP
3. BIP假设每个掩码至多一个类别——无法处理实例重叠严重的场景
4. SAM2对极小目标（<10px）可能漏检，裁剪策略部分缓解但未彻底解决
5. 推理依赖GPT-4o API——成本和网络依赖

## 启发与关联
- BIP匹配思路可推广到任何"已知类别计数+已有检测/分割提议"的场景
- 计数约束替代置信度阈值的范式具有通用性——适用于所有生成式检测模型
- EarthInstruct基准的"数据集特定指令"设计值得其他领域借鉴——不同数据集标注规则不同

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ BIP匹配+计数约束替代阈值，范式创新
- 实验充分度: ⭐⭐⭐⭐ 三设置全面评估，与多种基线对比
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，框架优雅，指标设计思考深入
- 价值: ⭐⭐⭐⭐⭐ 免训练免阈值范式有广泛影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](../../AAAI2026/segmentation/rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)
- [\[CVPR 2026\] ReAttnCLIP: Training-Free Open-Vocabulary Remote Sensing Image Segmentation via Re-defined Attention in CLIP](../../CVPR2026/segmentation/reattnclip_training-free_open-vocabulary_remote_sensing_image_segmentation_via_r.md)
- [\[ACL 2025\] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning](../../ACL2025/segmentation/instructpart_task-oriented_part_segmentation_with_instruction_reasoning.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](../../CVPR2025/segmentation/ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)

</div>

<!-- RELATED:END -->
