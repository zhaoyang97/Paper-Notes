---
title: >-
  [论文解读] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding
description: >-
  [NeurIPS 2025][多模态VLM][遥感图像] 提出 DVL-Suite 框架，包含 DVL-Bench 基准和 DVL-Instruct 指令微调数据集，覆盖 42 座美国城市、14,871 张高分辨率多时相遥感影像，系统评估 18 个 MLLM 在长期城市动态理解上的能力，并开发了 DVLChat 基线模型。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "遥感图像"
  - "城市动态理解"
  - "多时相分析"
  - "视觉语言基准"
  - "变化检测"
---

# DynamicVL: Benchmarking MLLMs for Dynamic City Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2505.21076](https://arxiv.org/abs/2505.21076)  
**代码**: [GitHub](https://github.com/weihao1115/dynamicvl)  
**领域**: 多模态VLM  
**关键词**: 遥感图像, 城市动态理解, 多时相分析, 视觉语言基准, 变化检测

## 一句话总结

提出 DVL-Suite 框架，包含 DVL-Bench 基准和 DVL-Instruct 指令微调数据集，覆盖 42 座美国城市、14,871 张高分辨率多时相遥感影像，系统评估 18 个 MLLM 在长期城市动态理解上的能力，并开发了 DVLChat 基线模型。

## 研究背景与动机

遥感技术能通过卫星影像监测城市发展，但现有研究大多局限于双时相（bi-temporal）比较，缺乏覆盖更长时间跨度的视觉语言数据集。虽然 MLLM 在通用视觉理解任务上表现优异，但在多时相遥感分析方面仍面临两大瓶颈：(1) 缺乏长时间序列对齐的视觉语言数据集，(2) 现有多时相遥感 MLLM 仅测试高层语义理解，缺少像素级精确量化分析能力。

现有数据集（如 CDVQA、TEOChatlas、EarthDial）要么仅支持双时相、要么任务单一、要么图像分辨率低（224-512 像素），DVL-Suite 因此应运而生：提供 1024×1024 的高分辨率影像，平均每个场景 6.73-6.94 个时相帧（2005-2023年），覆盖从像素级到场景级的六大任务。

## 方法详解

### 整体框架

DVL-Suite 包含两部分：

1. **DVL-Bench**：评估基准，包含 3,469 张多时相影像，附带 1,391 条指代分割指令、5,854 个 QA 对、1,437 条综合描述
2. **DVL-Instruct**：指令微调数据集，63,771 个文本对、11,402 张多时相图像，用于训练 DVLChat

数据来自 NAIP（国家农业影像计划），GSD 为 1.0m，覆盖 42 座美国主要城市。

### 关键设计

#### 六大任务分类

论文定义了一套层次化的任务体系，覆盖从细粒度到全局的城市动态理解：

- **BCA（基本变化分析）**：识别和比较多时相土地利用变化，涵盖植被、非植被、水体、建筑、游乐场 5 类土地覆盖类型的 20 种变化事件
- **CSE（变化速度估计）**：追踪和量化城市要素的时序趋势（如建筑扩张速率、植被损失）
- **EA（环境评估）**：通过视觉分析评估城市宜居性和经济指标
- **RCD（指代变化检测）**：密集推理+精确空间定位变化区域，需要像素级分割
- **RCC（区域变化描述）**：为用户指定的地理区域生成详细变化描述
- **DTC（密集时序描述）**：生成记录长期时序变化的综合报告

#### 数据标注流水线

采用半自动标注流程：

1. 城市专家进行基础标注（语义变化区域分割、关键帧识别）
2. GPT-4.1 整合专家标注信息生成多样化指令
3. 经过自检、交叉检查、监督审查的三轮质量控制
4. BCA/CSE：从分割掩码计算正确答案，生成干扰选项（±20%、±40%）
5. RCD：领域专家设计事件特定提示 + 手动掩码标注
6. DTC/RCC：标注者识别关键帧 → 撰写阶段描述 → GPT-4.1 润色

#### DVLChat 模型设计

基于 LISA 架构，做了两个关键改进：

1. **双 LoRA 路由机制**：通过前缀 token 路由请求 — `[QA]` 激活 VQA LoRA，`[SE]` 激活变化检测 LoRA，避免任务间互相干扰
2. **多时相图像交错处理**：将多个时相的图像特征交错后再解码，实现跨时间分析
3. **分割能力**：解码 `<SEG>` token 嵌入，通过 SAM 的冻结视觉主干和解冻解码器生成精确分割掩码

底层 MLLM 使用 Qwen2.5-VL，但架构是 MLLM 无关的。

### 训练策略

- 两个独立 LoRA 模块分别训练 VQA 和分割任务
- QA 部分使用 DVL-Instruct 的指令-真值对
- 分割部分使用 RCD 任务的掩码标注
- 在 8 张 H100 GPU 上训练

## 实验关键数据

### 主实验

**表1：QA 任务结果（精度%）**

| 模型 | AVG | BCA-单选 | BCA-多选 | CSE-单选 | CSE-多选 | EA |
|------|-----|---------|---------|---------|---------|-----|
| o4-mini | 34.1 | 62.8 | 36.1 | 33.8 | 12.4 | 25.3 |
| GPT-4.1 | 32.5 | 66.1 | 39.7 | 31.3 | 5.4 | 20.2 |
| Qwen2.5-VL 32B | 31.4 | 62.0 | 33.3 | 36.9 | 3.2 | 21.6 |
| DVLChat 7B | **33.3** | 64.9 | 21.3 | 31.3 | **18.6** | **30.6** |
| TEOChat | 17.2 | 35.1 | 8.7 | 17.0 | 10.8 | 14.6 |

**表2：描述任务结果（0-5分）**

| 模型 | RCC-AVG | DTC-AVG |
|------|---------|---------|
| o4-mini | 4.58 | 4.14 |
| GPT-4.1 | 4.46 | 3.98 |
| DVLChat 7B | 3.98 | 3.40 |
| InternVL3 78B | 3.92 | 3.33 |
| TEOChat | 1.66 | 1.45 |

### 消融实验

- **指代变化检测**：专用模型 ChangeMamba 达 32.41% IoU，DVLChat 达 29.06%（差距仅 3.35%），优于 LISA (13.85%) 和 PSALM (26.93%)
- **模型缩放非单调**：Qwen2.5-VL 系列在 32B 时达到 31.4% 峰值，72B 反降至 29.7%；InternVL3 在 14B 峰值后也下降 — 说明仅增大参数量不足以提升精确变化检测能力

### 关键发现

1. 最强商业模型 o4-mini 在整体 QA 上仅达 34.1%，暴露 MLLM 在长时序理解和量化分析上的严重不足
2. CSE 多选精度峰值仅 13.6%，CRP（变化率精度）始终低于 1.21，说明模型无法捕获细粒度时序变化
3. 7B 的 DVLChat 凭借领域专用数据在多项任务上超越 72B-78B 通用模型，证明领域数据比模型规模更重要
4. 开源模型与商业模型在描述任务上差距显著（DTC 平均分差约 1 分）

## 亮点与洞察

- 首个覆盖像素级到场景级的长时序遥感 VL 基准，填补了多时相分析的空白
- 双 LoRA 路由设计巧妙地在单一模型中融合 QA 和分割能力而不互相干扰
- 模型缩放的非单调现象揭示了一个深刻洞见：通用能力和领域精确分析能力的提升需要不同策略
- 半自动标注（专家 + GPT-4.1）在质量和效率间取得了良好平衡

## 局限与展望

- NAIP 影像包含近红外波段信息，但当前 MLLM 无法有效利用这些光谱数据
- DVLChat 尚未利用像素级分割数据来增强跨任务的数值量化能力
- DVLChat 在整体性能上仍落后于商业模型，需要专用算法和更大规模参数
- 仅覆盖美国城市，缺乏全球多样性数据

## 相关工作与启发

- 与 TEOChat、EarthDial 等现有多时相遥感 VL 数据集相比，DVL-Suite 的时相跨度更长（平均 6.94 帧 vs 2.07 帧）、分辨率更高（1024 vs 224-512）
- 双 LoRA 路由机制可推广到其他需要融合理解和分割的多任务场景
- 模型缩放的非单调现象对 scaling law 研究有启示意义

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个系统性的长时序遥感 VL 基准，任务体系设计完整
- 技术深度：⭐⭐⭐ — DVLChat 架构不复杂但实用，核心贡献在数据和基准
- 实验充分度：⭐⭐⭐⭐⭐ — 评估了 18 个模型，多维度分析透彻
- 实用价值：⭐⭐⭐⭐ — 对城市规划、灾害评估等应用有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[ACL 2025\] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](../../ACL2025/multimodal_vlm/punchbench_mllm_punchline.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](../../ACL2026/multimodal_vlm/cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[CVPR 2026\] RealBirdID: Benchmarking Bird Species Identification in the Era of MLLMs](../../CVPR2026/multimodal_vlm/realbirdid_benchmarking_bird_species_identification_in_the_era_of_mllms.md)

</div>

<!-- RELATED:END -->
