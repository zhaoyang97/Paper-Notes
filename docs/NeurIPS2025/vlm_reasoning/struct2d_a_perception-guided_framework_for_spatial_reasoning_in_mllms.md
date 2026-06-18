---
title: >-
  [论文解读] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs
description: >-
  [NeurIPS 2025][多模态VLM][空间推理] 提出 Struct2D，一种感知引导的提示框架，通过将3D感知输出转化为结构化2D表示（BEV图像+对象标记+元数据），使MLLM无需显式3D输入即可完成复杂空间推理任务，并构建了200K QA对的大规模指令微调数据集 Struct2D-Set。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "空间推理"
  - "鸟瞰图"
  - "多模态大语言模型"
  - "指令微调"
  - "3D场景理解"
---

# Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs

**会议**: NeurIPS 2025  
**arXiv**: [2506.04220](https://arxiv.org/abs/2506.04220)  
**代码**: [GitHub](https://github.com/neu-vi/struct2d)  
**领域**: 多模态VLM  
**关键词**: 空间推理, 鸟瞰图, 多模态大语言模型, 指令微调, 3D场景理解

## 一句话总结

提出 Struct2D，一种感知引导的提示框架，通过将3D感知输出转化为结构化2D表示（BEV图像+对象标记+元数据），使MLLM无需显式3D输入即可完成复杂空间推理任务，并构建了200K QA对的大规模指令微调数据集 Struct2D-Set。

## 研究背景与动机

3D空间推理是机器人操控、自主导航和视觉问答等任务的核心能力。传统方法依赖点云等显式3D表示作为输入，但存在两个关键问题：(1) 需要大量标注数据进行训练，泛化能力有限；(2) 难以与语言理解有效桥接，限制了在具身AI中的应用。

近年来MLLM在2D图像和视频感知上取得了巨大进展，部分工作尝试将点云特征对齐到LLM中实现3D理解，但需要点云特征作为输入，灵活性受限。人类实际上是通过连续的2D视觉输入流来推断3D空间关系的，这启发了一个核心问题：**MLLM能否仅通过结构化2D表示来实现空间推理，而无需显式3D特征？**

已有工作如GPT4Scene尝试利用BEV图像作为2D空间线索，但往往忽略了对象外观和详细先验信息（如坐标、类别），不足以支撑全面的3D理解。直接使用视频帧进行空间推理也存在两大局限：感知不完整（稀疏采样导致遗漏关键视觉证据）和缺乏全局上下文（自我中心视角无法捕捉场景整体布局）。

## 方法详解

### 整体框架

Struct2D 将3D感知输出转化为结构化2D输入，包括三个核心组件：(1) 带有过滤对象标记的BEV图像，(2) 对象中心的元数据（类别、3D坐标），(3) 可选的自我中心关键帧。整体流程为：从RGB-D视频中通过感知模块提取点云和检测结果，渲染BEV图像并投影对象标记，构建元数据文本，最终将这些结构化2D输入提供给MLLM进行推理。

形式化表达为：

$$\mathbf{T}^{\text{out}} = \mathcal{F}(\text{Struct2D}(\phi_{\text{percept}}(\mathbf{V}), \mathbf{T}^{\text{meta}}, \mathbf{I}_{\text{keyframe}}), \mathbf{T}^{\text{in}})$$

其中 $\phi_{\text{percept}}$ 是感知模块，$\mathbf{T}^{\text{meta}}$ 是元数据文本，$\mathbf{I}_{\text{keyframe}}$ 是可选关键帧。

### 关键设计

1. **Struct2D Prompting（结构化2D提示策略）**：将3D场景的感知输出转化为MLLM可直接消费的2D表示。关键创新包括：(a) 根据问题过滤对象标记，只保留与查询相关的对象，减少视觉干扰；(b) BEV图像旋转到与智能体朝向对齐，便于相对方向推理；(c) 通过深度感知的3D投影选择关键帧，而非均匀采样，使关键帧更少但更具信息量。相比GPT4Scene的提示策略，训练时间从6小时降至4小时。

2. **Struct2D-Set（大规模指令微调数据集）**：包含从6K+室内3D场景自动生成的200K QA对，覆盖8类空间推理任务。数据生成分两条线：(a) 受VSI-Bench启发的全局空间关系任务（空间关系识别、自我中心导航、比较推理），基于3D几何模板生成初始QA对，再用ChatGPT丰富推理链；(b) 改编自ScanQA、SQA3D等现有基准的场景理解任务（属性识别、计数、验证）。每个QA对包含短答案和带推理过程的长答案。

3. **思考-回答推理机制**：对涉及复杂空间推理的问题（如相对方向、路径规划），插入 `<think>` 和 `</think>` 特殊标记引导模型生成逐步推理过程，最终答案放在 `<answer>` 和 `</answer>` 中。对简单问题则直接生成短答案。

### 损失函数 / 训练策略

基于 Qwen2.5VL 进行监督微调（SFT），学习率 2e-6，余弦退火，在8×H200 GPU上训练约8小时。视觉输入统一缩放到 480×480。评估时使用 BundleFusion 重建点云，Mask3D 和 UniDet 检测3D对象框，投影为BEV图像和2D标记。

## 实验关键数据

### 主实验

**零样本分析（GPT-o3 + Struct2D Prompting，VSI-Bench子集）**

| 设置 | 图像数 | 平均 | 相对距离 | 相对方向 | 路径规划 |
|------|--------|------|----------|----------|----------|
| VSI-Bench 原始 | 16 | 48.6 | 51.0 | 49.4 | 61.9 |
| GPT4Scene | 9 | 50.3 | 50.5 | 47.9 | 58.8 |
| Ours (噪声检测) | 1 | 56.1 | 60.0 | 60.1 | 76.2 |
| Ours (GT对象) | 1 | 83.8 | 96.5 | 94.4 | 80.1 |

**VSI-Bench 全量评估（开源模型对比）**

| 方法 | 平均 | 对象计数 | 绝对距离 | 相对距离 | 相对方向 | 路径规划 |
|------|------|----------|----------|----------|----------|----------|
| LLaVA-NeXT-Video-7B | 36.3 | 48.5 | 14.0 | 43.5 | 42.4 | 34.0 |
| R1-Zero-VSI+SFT | 38.8 | 44.7 | 27.6 | 34.0 | 35.7 | 33.0 |
| Qwen2.5-VL-3B (SFT) | 41.9 | 46.0 | 34.7 | 35.1 | 44.9 | 33.5 |
| **Qwen2.5-VL-7B (SFT)** | **43.6** | 47.1 | 35.1 | 35.1 | **45.9** | **35.8** |

### 消融实验

| 配置 | 平均 | 相对距离 | 相对方向 | 路径规划 | 说明 |
|------|------|----------|----------|----------|------|
| 无增强QA | 31.5 | 21.2 | 14.7 | 31.5 | 无ChatGPT增强 |
| 有增强QA | 38.0 | 33.3 | 42.2 | 33.0 | ChatGPT生成推理链 |
| 无`<think>`标记 | 36.2 | 33.3 | 38.6 | 26.3 | 无显式推理引导 |
| 有`<think>`标记 | 36.1 | 31.5 | 42.2 | 33.0 | 显式推理引导提升复杂任务 |

**提示组件消融（元数据+过滤标记）**

| 元数据 | 过滤标记 | 相对距离 | 相对方向 | 路径规划 |
|--------|----------|----------|----------|----------|
| ✗ | ✗ | 67.5 | 82.1 | 74.3 |
| ✗ | ✓ | 72.1 | 88.3 | 78.3 |
| ✓ | ✗ | 75.3 | 89.5 | 50.6 |
| ✓ | ✓ | **96.5** | **94.4** | **80.1** |

### 关键发现

- 仅需单张BEV图像+轻量元数据即可实现高质量空间推理，成本远低于多帧策略
- 元数据和过滤标记相互补充，缺一不可，尤其路径规划任务需要两者配合才能达到最佳
- ChatGPT生成的推理链对复杂空间任务（相对方向）贡献显著，从14.7提升到42.2
- 在ScanQA和SQA3D上，纯2D输入的方法已能与使用显式3D点云的模型竞争

## 亮点与洞察

- **核心洞察**：结构化2D表示可以有效替代显式3D表示进行空间推理，关键在于信息的组织方式而非表示形式
- 一张BEV图就够了——这极大降低了推理成本（$27 vs. $105）
- 过滤对象标记是个简单但有效的设计，根据问题只显示相关对象，减少视觉噪声
- 数据构建管线完全自动化，可扩展性强

## 局限与展望

- 预处理仍依赖3D感知模块（点云重建、3D检测），在延迟敏感场景可能受限
- 数据集聚焦室内场景，对户外或开放世界场景的泛化尚未探索
- BEV渲染依赖较好的深度估计质量，噪声检测下性能从83.8降到56.1
- 规则基评估指标可能偶尔无法完全反映推理质量

## 相关工作与启发

- GPT4Scene 最先探索了BEV图像用于空间提示，但Struct2D在过滤标记、元数据引导和关键帧选择上做了关键改进
- R1-Zero-VSI 使用GRPO训练方法增强空间推理，但QA对复杂度和任务覆盖较窄
- 思路可启发：利用结构化2D表示替代3D输入的范式可推广到机器人导航、AR/VR等场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 用结构化2D替代3D输入的思路有新意，提示策略设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 零样本分析+SFT+多个基准+详细消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，分析深入
- 价值: ⭐⭐⭐⭐ 为3D空间推理提供了实用且低成本的替代方案，数据集和代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](../../ACL2026/multimodal_vlm/unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)

</div>

<!-- RELATED:END -->
