---
title: >-
  [论文解读] Vision-Language Models Struggle to Align Entities across Modalities
description: >-
  [ACL 2025][多模态VLM][跨模态实体链接] 提出 MATE 基准（5,500 个问答实例），通过合成 3D 场景的跨模态属性检索任务系统评估 VLM 的实体链接能力，发现即使最强闭源模型仍落后人类约 15 个百分点，且性能随场景物体数量增加急剧下降——根源在于跨模态特征绑定而非单模态感知。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "跨模态实体链接"
  - "视觉语言模型"
  - "benchmark"
  - "实体属性对齐"
  - "视觉搜索"
---

# Vision-Language Models Struggle to Align Entities across Modalities

**会议**: ACL 2025  
**arXiv**: [2503.03854](https://arxiv.org/abs/2503.03854)  
**代码**: [GitHub](https://github.com/hitz-zentroa/MATE)  
**领域**: 多模态VLM / 跨模态对齐  
**关键词**: 跨模态实体链接, 视觉语言模型, benchmark, 实体属性对齐, 视觉搜索

## 一句话总结

提出 MATE 基准（5,500 个问答实例），通过合成 3D 场景的跨模态属性检索任务系统评估 VLM 的实体链接能力，发现即使最强闭源模型仍落后人类约 15 个百分点，且性能随场景物体数量增加急剧下降——根源在于跨模态特征绑定而非单模态感知。

## 研究背景与动机

**问题定义**：跨模态实体链接（cross-modal entity linking）是指在不同模态（如图像与文本）之间对齐同一实体及其属性的基础能力。这一能力是多模态 AI 系统执行下游任务的前提条件。

**应用场景**：该能力在多个真实场景中至关重要——**自动驾驶**中需要将图像中的车辆与传感器文本数据（速度、轨迹）链接以构建统一表示；**多模态代码生成**中需要对齐 UI 图像与代码实体；**假新闻检测**中需要验证图文信息的一致性；**场景理解**中需要统一视觉与结构化数据。

**现有缺陷**：虽然指代表达理解（REC）、多模态实体链接（MEL）、SIMMC 等任务与跨模态对齐相关，但它们均未直接测试模型从原始多模态输入中对齐实体属性的能力。REC 仅需定位区域而非属性对齐；MEL 聚焦于将提及链接到知识库且通常是单实体场景；SIMMC 通过提供金标准物体 ID 回避了链接挑战。

**核心动机**：缺乏对这一基础能力的**系统性、受控评估**。本文提出的核心问题是：**当前 VLM 能否可靠地将同一实体在视觉和文本模态中的表示进行对齐？**

## 方法详解

### 整体框架

MATE（Multimodal Attribute-based Entity linking）是一个包含 **5,500 个问答实例**的基准测试。每个实例包含一个合成 3D 场景图像及其 JSON 文本表示，场景中有 3-10 个具有不同颜色、形状、材质和尺寸的几何物体。任务核心是：给定一个在某模态中唯一标识物体的**指针属性**（pointer attribute），要求在另一模态中检索该物体的**目标属性**（target attribute）。

| 任务方向 | 指针属性（定位模态） | 目标属性（检索模态） | 示例 |
|---------|-------------------|-------------------|------|
| Image→Text | 图像中的视觉属性（如颜色"红色"） | 文本中的属性（如名称"Object_0"） | 红色物体叫什么名字？ |
| Text→Image | 文本中的属性（如名称"Object_0"） | 图像中的视觉属性（如颜色） | Object_0 是什么颜色？ |

### 关键设计

**1. 合成场景的受控实验方法论**：基于 CLEVR 数据集扩展，生成包含 3-10 个 3D 几何物体的合成场景。核心思路是使用合成图像而非真实图像来控制所有变量，排除物体识别或视觉歧义等混淆因素，确保评估的纯粹性——只测跨模态实体链接能力，而非物体检测能力。数据集在物体数量、任务方向、属性对之间保持均匀分布（每种配置 43±1.5 个样本），消除分布偏差。

**2. 属性信息隔离与跨模态强制机制**：将属性分为三类——仅视觉属性（颜色、形状）、仅文本属性（名称、旋转、尺寸、3D 坐标）、共享属性（材质，不用作指针或目标）。关键设计是**序列化场景中不包含图像中才有的指针/目标属性**，强制模型必须跨模态检索，无法仅靠单一模态完成任务。例如图像中的颜色作为指针时，文本 JSON 中不包含颜色字段。

**3. 三步人类解题过程建模**：通过人类评估识别出跨模态实体链接的三个认知步骤——(a) **视觉搜索**：用指针属性在图像中定位目标物体（如找到红色物体）；(b) **链接属性识别**：找到能区分该物体与场景中其他物体的共享属性（linking attributes），如该物体是唯一的圆柱体；(c) **文本搜索**：用链接属性在另一模态中定位对应物体并检索目标属性。这一分解为后续消融实验提供了分析框架。

### 实验设置

| 设置项 | 详情 |
|-------|------|
| 开源模型 | LLaVA 1.5 (13B), LLaVA 1.6 (34B), Molmo-7B, Llama-3.2-11B, Qwen2-VL-7B, Qwen2.5-VL-7B |
| 闭源模型 | GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Flash |
| 提示策略 | zero-/one-/two-shot，报告 two-shot 结果（最稳定） |
| 文本格式 | JSON（与 YAML/XML/verbalized 无显著差异） |
| 人类评估 | 384 个子集，5 名参与者，保证特征分布代表性 |
| 计算资源 | 4 × NVIDIA A100 80GB，约 300 GPU 小时 |
| 评估指标 | Exact match accuracy |

## 实验结果与分析

### 主实验：跨模态实体链接

| 模型 | Image→Text | Text→Image | 平均 |
|------|-----------|-----------|------|
| **人类** | **97.9** | **97.9** | **97.9** |
| 随机基线 | 25.4 | 18.5 | 22.0 |
| LLaVA 1.5 | 29.3 | 35.7 | 32.5 |
| LLaVA 1.6 | 48.7 | 61.6 | 55.2 |
| Molmo | 18.1 | 20.9 | 19.5 |
| Llama 3.2 | 37.4 | 11.4 | 24.4 |
| Qwen2-VL | 72.1 | 77.2 | 74.7 |
| Qwen2.5-VL | 75.7 | 84.5 | 80.1 |
| Gemini 1.5 | 63.2 | 71.2 | 67.2 |
| GPT-4o | 76.4 | 79.1 | 77.8 |
| **Claude 3.5** | **80.9** | **85.7** | **83.3** |

最佳 VLM（Claude 3.5）仍落后人类 **14.6** 个百分点。所有 VLM 的 Text→Image 方向表现优于 Image→Text（Llama 3.2 除外），表明从文本定位再查图像更简单；人类则两个方向性能一致。随物体数量从 3 增至 10，VLM 性能显著下降——Claude 3.5 在 10 物体场景中落后人类近 30 个百分点，而人类性能保持稳定。

### 核心消融：单模态 vs 跨模态

| 模型 | Image→Image | Text→Text | 平均 |
|------|------------|----------|------|
| 人类 | 100.0 | 99.0 | 99.5 |
| Qwen2.5-VL | 99.7 | 99.4 | 99.5 |
| GPT-4o | 98.4 | 100.0 | 99.2 |
| Claude 3.5 | 97.3 | 100.0 | 98.7 |

**关键对比**：Qwen2.5-VL 在单模态任务中达到 99.5%（与人类持平），但跨模态仅 80.1%，性能骤降 **19.4** 个百分点。这证明困难的根源在于**跨模态链接**而非单模态属性提取，且单模态性能不受物体数量影响。

### CoT 提示与自反思

| 模型 | All（+CoT） | Δ vs 标准 | 10 物体场景 |
|------|-----------|----------|-----------|
| Claude 3.5 | 86.2 | +2.9 | 70.5 |
| GPT-4o | 82.8 | +5.0 | 64.6 |
| Qwen2.5-VL | 78.9 | -1.2 | 62.8 |
| Llama 3.2 | 53.7 | +29.2 | 36.3 |

CoT 对弱模型提升显著（Llama 3.2 +29.2），对强模型帮助有限。即使使用 CoT，所有模型在物体数量增加时仍显著下降。自反思模型 VL-Rethinker-7B 对比其基座 Qwen2.5-VL 也无显著提升（79.6 vs 80.1），表明自反思技术对跨模态实体链接无效。

### 链接属性分析

对 Qwen2.5-VL 在 7 物体场景中的分析显示：需要 **1 个**链接属性时准确率最高，需要组合 **2-3 个**属性时性能递减。当唯一链接属性为 3D 坐标时表现最差，说明模型难以利用空间位置进行跨模态匹配。低 OSE（Out-of-Scene Error）率证实错误主要来自**实体链接错误**而非幻觉。

## 关键发现

1. **VLM 与人类行为差异巨大**：人类接近 100%，最佳 VLM 仍落后约 15 个百分点
2. **困难在于跨模态链接而非单模态搜索**：单模态近乎完美，跨模态性能骤降近 20 个百分点
3. **物体数量是关键难度因素**：VLM 性能随物体数量线性下降（特征干扰增加），人类保持稳定
4. **CoT 和自反思均非根本解法**：CoT 仅带来有限提升，自反思完全无效
5. **与认知科学绑定问题对应**：VLM 的跨模态对齐困难可类比认知科学中的特征绑定问题（binding problem）

## 亮点与不足

**亮点**：(1) 通过精巧的受控实验设计将问题定位到跨模态绑定而非单模态感知；(2) 合成数据排除混淆因素使评估纯粹；(3) 三步分解分析框架（视觉搜索→链接属性识别→文本搜索）清晰且可复用；(4) 基准可轻松扩展到更多物体或多指针/目标属性场景。

**不足**：(1) 合成几何场景与真实世界差距较大；(2) 仅使用简单几何属性，未测试语义更丰富的属性；(3) 发现了问题但未探索解决方案；(4) 仅测试单跳链接，未评估多跳跨模态推理。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 定义了新任务并设计了精巧的受控实验，但任务本身较为基础
- **实验充分度**: ⭐⭐⭐⭐⭐ — 主实验+单模态消融+CoT+自反思+链接属性分析+属性类型分析，极其全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑链条清晰（能力验证→问题定位→原因分析），图表精美
- **实用价值**: ⭐⭐⭐⭐ — 揭示 VLM 基础能力缺陷，为社区提供重要基准和研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models](performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [\[ACL 2025\] Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](con_instruction_universal_jailbreaking_of_multimodal_large_language_models_via_n.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)

</div>

<!-- RELATED:END -->
