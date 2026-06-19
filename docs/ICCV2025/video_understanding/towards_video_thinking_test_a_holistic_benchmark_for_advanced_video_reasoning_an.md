---
title: >-
  [论文解读] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding
description: >-
  [ICCV 2025][视频理解][Video LLM] 提出 Video Thinking Test (Video-TT)，一个评估视频大语言模型正确性和鲁棒性的基准，包含 1000 个 YouTube Shorts 视频和 5000 个问题，通过视觉/叙事复杂性因子和自然对抗问题揭示了当前最强模型（GPT-4o 36.6%）与人类（84.3%）之间的巨大差距。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "Video LLM"
  - "Adversarial Robustness"
  - "Visual Complexity"
  - "Narrative Complexity"
---

# Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding

**会议**: ICCV 2025  
**arXiv**: [2507.15028](https://arxiv.org/abs/2507.15028)  
**代码**: [https://github.com/zhangyuanhan-ai/video-tt](https://github.com/zhangyuanhan-ai/video-tt)  
**领域**: 视频理解  
**关键词**: Video Understanding Benchmark, Video LLM, Adversarial Robustness, Visual Complexity, Narrative Complexity

## 一句话总结

提出 Video Thinking Test (Video-TT)，一个评估视频大语言模型正确性和鲁棒性的基准，包含 1000 个 YouTube Shorts 视频和 5000 个问题，通过视觉/叙事复杂性因子和自然对抗问题揭示了当前最强模型（GPT-4o 36.6%）与人类（84.3%）之间的巨大差距。

## 研究背景与动机

视频大语言模型（Video LLMs）的能力日益接近人类智能，但现有 benchmark 无法准确反映模型与人类的差距，原因在于两个层面：

**正确性评估的误导**：现有 benchmark（如 VideoMME、MVBench）无法区分"帧采样不足导致的错误"和"真正理解能力不足导致的错误"。Video LLMs 通常先采样有限帧数再理解内容，因此在长视频 benchmark 上的大幅性能差距可能只是反映了采样策略的局限。反过来，在短视频 benchmark（如 VideoMME-Short）上模型性能已接近人类上限，给人"模型已达人类水平"的错误印象——但这只是因为问题不够难。

**鲁棒性评估的失真**：现有鲁棒性研究主要测试人为扰动（如像素修改、拼写错误），这些场景过于刻意，不反映真实世界的复杂性。真正需要评估的是**自然对抗条件**下的鲁棒性——同一问题换个角度问，模型能否保持一致。

Video-TT 的核心设计理念是：(1) 确保每个问题足够复杂以区分人类和模型的理解能力；(2) 确保问题在有限帧数下可回答，分离帧采样问题和理解问题；(3) 通过自然对抗问题评估模型的鲁棒性。

## 方法详解

### 整体框架

Video-TT 由 1000 个 YouTube Shorts 视频组成，每个视频配有 1 个主开放式问题和 4 个对抗变体问题（共 5000 QA 对）。所有问题基于 8 个视觉/叙事复杂性因子设计，确保问题在 80 帧均匀采样下可回答。

### 关键设计

1. **视觉复杂性因子（Visual Complexity Factors）**:

    - 功能：定义使视频内容理解困难的视觉因素
    - 核心思路：借鉴认知科学的视觉复杂性理论，识别四类因子——(a) **不清晰/异常内容**：内容与常见场景不同，或存在噪声/模糊/遮挡；(b) **运动速度**：视频或相机运动过快难以追踪；(c) **时空排列**：场景中物体数量多、交互复杂、认知负荷高；(d) **视觉错觉**：使用创造错觉的技巧
    - 设计动机：问题的复杂性不仅取决于问题类型（如"物体颜色"vs"情节理解"），更取决于**上下文和场景条件**。例如"第二辆车的颜色"在车快速运动且部分遮挡时变得极其困难

2. **叙事复杂性因子（Narrative Complexity Factors）**:

    - 功能：定义超越线性叙事的内容理解挑战
    - 核心思路：四类因子——(a) **复杂情节**：包含转折或意外结局；(b) **叙事剪辑**：蒙太奇等复杂组合镜头；(c) **技术剪辑**：特殊拍摄技巧或后期操作；(d) **世界知识**：需要先验知识才能完全理解
    - 设计动机：这些复杂性需要观众更深度地参与视频内容

3. **自然对抗问题设计（Natural Adversarial Questions）**:

    - 功能：测试模型对同一问题不同表述的鲁棒性
    - 核心思路：每个主问题派生四个变体——(a) **改写开放式问题**：语义微调重述；(b) **正确引导开放式问题**：给出正确线索引导；(c) **错误引导开放式问题**：给出误导线索；(d) **选择题**：混合正确/错误选项
    - 设计动机：如果模型真正理解了视频，就应该能抵抗同一问题的不同表述方式。鲁棒性定义为所有五个问题都回答正确的视频比例除以仅主问题回答正确的比例

4. **数据标注质量控制**:

    - 功能：确保标注质量和一致性
    - 核心思路：多层质量保障——(a) 每个问题须至少涉及一个复杂性因子；(b) 用 GPT-4o、LLaVA-Video-7B、Qwen2.5-VL-7B 验证复杂度（至少一个模型三次尝试中至少一次答错）；(c) 标注者须说明推理过程和模型错误原因；(d) 确保 80 帧采样即可回答；(e) 三人标注一致性检查
    - 设计动机：将帧采样问题与理解问题严格分离

### 评估指标

- **正确性分数**：用 Qwen2.5-72B 评分开放式回答（0-5 分，>3 为正确）
- **鲁棒性分数**：在主问题正确的视频中，所有 5 个问题都正确的比例

## 实验关键数据

### 主实验

| 模型 | Primary | Rephrased | Correctly-Led | Wrongly-Led | Multi-Choice | Avg | 鲁棒性 |
|------|---------|-----------|---------------|-------------|-------------|-----|--------|
| Qwen2.5-VL-7B | 20.9 | 22.5 | 45.3 | 39.3 | 39.9 | 33.6 | 14.4 |
| LLaVA-Video-72B | 24.4 | 25.7 | 57.7 | 32.6 | 47.5 | 37.6 | 19.7 |
| Qwen2.5-VL-72B | 26.6 | 25.7 | 31.1 | 49.8 | 45.6 | 35.8 | 22.2 |
| Gemini Pro | 28.8 | 29.7 | 50.2 | 29.2 | 42.3 | 38.2 | 20.5 |
| **GPT-4o** | **36.6** | **35.4** | **67.5** | **39.8** | **46.6** | **45.2** | **36.0** |
| **Human** | **84.3** | **83.9** | **83.9** | **76.2** | **87.5** | **83.2** | **64.4** |

### 消融实验（GPT-4o 错误类型分析与增强实验）

| 分析维度 | 结果 | 说明 |
|---------|------|------|
| 帧数 8→64 (人类) | 准确率持续上升至接近满分 | 人类理解随帧数稳定提升 |
| 帧数 8→64 (模型) | 约 8 帧后饱和 | 更多帧无法帮助模型理解 |
| CoT on Wrongly-Led | +6.8% 相对提升 | 结构化思考帮助抵抗误导 |
| CoT on Multi-Choice | 无明显提升 | 结构化格式受益有限 |
| 音频转录对鲁棒性 | +15% 相对提升 | 多模态信息增强鲁棒性 |
| 时空混淆错误占比 (元素/事件定位) | 79% / 88% | 模型最大弱点 |
| 世界知识缺失 (角色动机) | 44% 错误 | 缺乏情境知识 |
| 复杂情节混淆 (属性/因果) | 55% 错误 | 因果推理能力不足 |

### 关键发现

- **人机差距巨大**：GPT-4o 正确率 36.6% vs 人类 84.3%；鲁棒性 36.0% vs 64.4%——这在现有 benchmark 上是罕见的差距
- **开源 vs 闭源的鲁棒性鸿沟**：Qwen2.5-VL-72B 鲁棒性 22.2%，比 GPT-4o (36.0%) 低 13.8 分，远超正确性维度的差距
- **帧数增加对模型无效**：模型约 8 帧后性能饱和，证明瓶颈在理解而非采样
- **选择题高估模型能力**：LLaVA-Video-72B 在选择题(47.5%)与 GPT-4o(46.6%)相当，但开放式问题(24.4% vs 36.6%)差距显著

## 亮点与洞察

- **分离帧采样与理解能力**：这是一个被忽视但至关重要的设计——确保所有问题在 80 帧采样下可回答，使评估聚焦于理解能力本身
- **自然对抗问题的价值**：不是人为扰动像素或拼写，而是换个角度问同一个问题。这更接近真实用户交互场景
- **GPT-4o 的错误模式**：时空混淆（无法跟踪多场景中的动作参与者）、世界知识缺失（无法推断隐含意图和社会动态）、复杂情节混淆（无法维持跨事件的因果链）
- **18 类问题的细粒度分析**：从元素属性到情节因果，提供了模型能力的精细画像

## 局限与展望

- 仅使用 YouTube Shorts（<65秒），未涉及长视频理解
- 标注过程依赖人工，成本高昂，难以大规模扩展
- 鲁棒性评估基于二元判断（全对/非全对），中间状态未建模
- 错误分析仅针对 GPT-4o，其他模型的错误模式可能不同
- 未与 Audio/Speech 多模态结合进行系统评估（仅做了初步实验）

## 相关工作与启发

- **VideoMME**：从 YouTube 收集视频，但长视频 track 的性能差距主要反映帧采样不足
- **MVBench**：整合已有 benchmark，但源数据集已被广泛训练，存在数据泄漏风险
- **TemporalBench**：评估细粒度时序理解，但限于特定时序问题类型
- **FunQA**：测试反直觉和幽默内容，但领域受限
- **启发**：benchmark 设计的核心应是控制混淆变量（如帧采样），确保测量目标变量（如理解能力）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](../../NeurIPS2025/video_understanding/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[CVPR 2025\] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs](../../CVPR2025/video_understanding/q-bench-video_benchmark_the_video_quality_understanding_of_lmms.md)

</div>

<!-- RELATED:END -->
