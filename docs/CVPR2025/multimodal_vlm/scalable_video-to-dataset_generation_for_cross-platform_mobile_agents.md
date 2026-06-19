---
title: >-
  [论文解读] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents
description: >-
  [CVPR 2025][多模态VLM][移动端Agent] MONDAY 框架从 YouTube 教学视频自动生成移动端导航数据集——通过 OCR 场景转换检测和 GPT-4o 的 3 步动作识别流程，以人工标注 1/17 的成本（$0.34 vs $5.76/视频）构建了覆盖 iOS/Android 双平台的 313K 标注帧，预训练后 agent 在未见的 Windows Mobile 上提升 18.11%。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "移动端Agent"
  - "数据集自动生成"
  - "YouTube视频"
  - "跨平台导航"
  - "OCR"
---

# Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents

**会议**: CVPR 2025  
**arXiv**: [2505.12632](https://arxiv.org/abs/2505.12632)  
**代码**: 待确认  
**领域**: 机器人  
**关键词**: 移动端Agent、数据集自动生成、YouTube视频、跨平台导航、OCR场景检测

## 一句话总结
MONDAY 框架从 YouTube 教学视频自动生成移动端导航数据集——通过 OCR 场景转换检测和 GPT-4o 的 3 步动作识别流程，以人工标注 1/17 的成本（$0.34 vs $5.76/视频）构建了覆盖 iOS/Android 双平台的 313K 标注帧，预训练后 agent 在未见的 Windows Mobile 上提升 18.11%。

## 研究背景与动机

**领域现状**：移动端 GUI agent 的训练数据主要靠人工录制标注（如 AitW、AMEX），成本高、规模小、仅覆盖单一平台。而 YouTube 上有海量移动端教学视频（"如何在 Android 上换壁纸"），但缺乏从视频到结构化动作数据集的自动化管线。

**现有痛点**：(1) 人工标注成本高（$5.76/视频），无法大规模扩展；(2) 现有数据集只覆盖 iOS 或 Android 单平台，agent 泛化能力差；(3) YouTubeUI 截图的场景转换检测很难——暗黑模式切换会导致基于像素的方法失效；(4) 动作识别中 UI 元素的精确定位困难，尤其是复杂界面中的小按钮。

**核心矛盾**：YouTube 视频数据丰富但非结构化（无动作标注），传统标注方法无法 scale。如何自动从视频中提取精确的动作序列？

**本文目标** 设计一个完全自动化的管线，从 YouTube 教学视频生成高质量的移动端导航数据集，并验证该数据对 agent 跨平台泛化的价值。

**切入角度**：利用 OCR 文字比视觉像素更稳定的特性来检测场景变化，用 GPT-4o + Set-of-Mark 的多步推理来识别精确动作，用旁白文字辅助消歧。

**核心 idea**：通过 OCR 驱动的场景转换检测 + GPT-4o 多步动作识别，从 YouTube 视频自动生成大规模跨平台移动端导航数据集。

## 方法详解

### 整体框架
3 阶段流水线：(1) 视频采集与过滤（129K→20K 视频）；(2) OCR 场景转换检测（分割出界面变化帧）；(3) 3 步动作识别（场景摘要→多帧上下文动作识别→放大精修定位）。最终输出帧级别的动作标注。

### 关键设计

1. **OCR 场景转换检测**:

    - 功能：检测移动端截图序列中的界面变化点，将视频分割为独立的操作步骤
    - 核心思路：以 4 FPS 对手机屏幕区域（GroundingDINO 检测 @2FPS，线性插值补帧）做 PaddleOCR 文字提取，追踪相同位置的文字元素，计算 Levenshtein 距离。当 20% 以上的文字发生变化时标记为场景转换。F1 达 95.04%，比 SceneDetect（82.27%）高 12.77%
    - 设计动机：基于 YUV 色差的方法对暗黑模式切换等全局外观变化敏感（F1 仅 70.86%）；而 OCR 文字内容在这些变化中是稳定的，特别适合 UI 场景

2. **3 步动作识别（Scene Summary → Action ID → Refinement）**:

    - 功能：精确识别每帧的用户操作及其对应的 UI 元素坐标
    - 核心思路：第 1 步——场景摘要：无标记的原始帧让 GPT-4o 描述界面布局；第 2 步——动作识别：将当前帧 + 前后 2 帧的场景摘要 + Set-of-Mark（编号的 UI 元素）+ 视频旁白一起喂给 GPT-4o，识别候选动作；第 3 步——精修定位：围绕候选 UI 元素生成放大视图，再次用 GPT-4o + SoM 精确定位。最终坐标取 UI 元素 bounding box 的中心点
    - 设计动机：单步识别准确率仅 70.63%，引入时序上下文（+8.80%）和精修（+1.47%）后提升到 80.90%。旁白信息帮助 GPT-4o 在视觉相似的 UI 元素间消歧（+2.70%）

3. **大规模视频过滤管线**:

    - 功能：从 129K YouTube 视频中筛选出高质量的移动端教学视频
    - 核心思路：多级过滤——GroundingDINO 检测手机屏幕（过滤 Android Watch/MacOS）→ MediaPipe 检测手部遮挡（过滤手持录制视频）→ GPT-4o 采样帧确认 OS 类型。GPT-3.5 从 CommonCrawl 帖子中发现任务名称用于视频搜索
    - 设计动机：YouTube 视频质量参差不齐，需要多级过滤才能确保数据质量。最终保留 20K 视频

### 损失函数 / 训练策略
Agent 预训练/微调用 LoRA，输入为当前截图 + 任务名 + 最近 4 个动作，输出下一步动作预测。选择验证 loss 最低的 checkpoint。评估标准为精确动作匹配 + 触摸/长按的交互区域验证。

## 实验关键数据

### 主实验

| 测试集 | 模型 | 无 MONDAY | 有 MONDAY | 提升 |
|--------|------|------|----------|------|
| AitW (5类平均) | SeeClick | 66.98% | 68.47% | +1.49% |
| AitW (5类平均) | Llama-3.2-11B | 58.96% | 67.38% | +8.42% |
| AMEX | Llama-3.2-11B | 43.74% | 55.96% | +12.22% |
| **Windows Mobile (未见)** | **SeeClick** | **38.54%** | **51.71%** | **+13.17%** |
| **Windows Mobile (未见)** | **Llama-3.2-11B** | **26.83%** | **50.24%** | **+23.41%** |
| MONDAY 自身 | SeeClick | 40.66% | 63.39% | +22.73% |

### 消融实验

| 方法 | 全部动作准确率 | 触摸准确率 |
|------|---------|---------|
| 3 步多帧（完整） | **80.90%** | **91.84%** |
| 2 步（无精修） | 79.43% | 89.97% |
| 1 步（直接识别） | 70.63% | 74.67% |
| 无旁白信息 | 78.20% | 87.64% |
| 单帧（无时序上下文） | 77.22% | 89.30% |

### 关键发现
- **跨平台泛化效果惊人**：在 Windows Mobile（完全未见的平台）上平均提升 18.11%，说明 iOS+Android 双平台数据的多样性让 agent 学到了平台无关的 UI 理解能力
- **OCR 场景检测远优于视觉方法**：F1 95.04% vs SceneDetect 82.27%，证明 UI 中文字是最稳定的信号
- **UI 元素检测几乎完美**：Hit Ratio 99.87% vs OmniParser 91.83%，得益于移动端特定启发式过滤
- **成本效率极高**：$0.34/视频 vs 人工 $5.76/视频，成本降低 17 倍
- **Llama-3.2 的受益大于 SeeClick**：可能因为 LLama 的基础 UI 理解弱，从 MONDAY 的多样数据中获益更多

## 亮点与洞察
- **OCR 驱动的场景转换检测**是一个非常实用的创新——文字内容比像素在 UI 环境中更稳定，可以推广到任何 UI 视频分析任务
- **3 步动作识别的设计思想**（先看全局→上下文推理→放大精修）模仿了人类观看教学视频时的认知过程，每步贡献清晰可量化
- **YouTube 视频是 agent 训练的宝藏**：20K 视频 = 313K 标注帧，成本极低，且自然覆盖极多样的应用和操作场景，这比人工构造数据集有质的优势

## 局限与展望
- 依赖 GPT-4o 做动作识别，成本和 API 限制可能影响更大规模的数据生成
- 多级过滤丢了大量视频（129K→20K），可能遗漏了有价值的数据
- 20% 文字变化阈值是经验设定，对其他语言/脚本的适用性未验证
- 教学视频的动作分布偏向简单操作（Touch 79.83%），复杂手势（Multi-touch、Zoom）样本极少
- 精修步骤需要生成放大视图，增加计算开销

## 相关工作与启发
- **vs AitW/AMEX**: 人工标注的单平台数据集；MONDAY 是自动生成的双平台数据，预训练后在两个数据集上都有提升
- **vs OmniParser**: OmniParser 的 UI 元素检测 Hit Ratio 91.83%，MONDAY 的方法 99.87%，得益于移动端特定的启发式规则
- 为 GUI agent 社区提供了一个低成本 scaling 的范例——不必依赖人工标注

## 评分
- 新颖性: ⭐⭐⭐⭐ OCR 场景检测和 YouTube→数据集的自动化管线都有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 数据集构建各组件的单独评估 + 多 agent/多平台的下游验证 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 管线描述清晰，统计信息完整
- 价值: ⭐⭐⭐⭐ 实用价值高，数据集对移动端 agent 社区有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [\[CVPR 2025\] StarVector: Generating Scalable Vector Graphics Code from Images and Text](starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)

</div>

<!-- RELATED:END -->
