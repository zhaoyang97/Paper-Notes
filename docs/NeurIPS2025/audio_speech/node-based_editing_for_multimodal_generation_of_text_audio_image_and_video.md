---
title: >-
  [论文解读] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video
description: >-
  [NeurIPS 2025][音频/语音][节点图接口] 提出一个节点图式故事编辑系统，允许创作者通过自然语言和节点级操作迭代地生成、编辑和比较多模态内容（文本、音频、图像、视频），支持线性和分支叙事结构。 领域现状：Sora、DALL-E 等生成模型使内容创作变得容易，但当前主要采用单轮提示范式。 现有痛点：(1) 单一提…
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "节点图接口"
  - "叙事生成"
  - "多模态生成"
  - "人-AI 协作"
---

# Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video

**会议**: NeurIPS 2025  
**arXiv**: [2511.03227](https://arxiv.org/abs/2511.03227)  
**代码**: 预计公开  
**领域**: 人机交互 / AI 辅助创作  
**关键词**: 节点图接口, 叙事生成, 多模态生成, 人-AI 协作

## 一句话总结

提出一个节点图式故事编辑系统，允许创作者通过自然语言和节点级操作迭代地生成、编辑和比较多模态内容（文本、音频、图像、视频），支持线性和分支叙事结构。

## 研究背景与动机

**领域现状**：Sora、DALL-E 等生成模型使内容创作变得容易，但当前主要采用单轮提示范式。

**现有痛点**：(1) 单一提示难以充分表达复杂叙事意图；(2) 全部重新生成低效；(3) 线性编辑接口无法表示分支故事；(4) 缺乏机制探索多种叙事方向。

**核心矛盾**：强大的生成能力 vs 贫乏的交互控制——模型能生成好内容，但创作者无法精细控制叙事结构。

**切入角度**：采用图式节点表示使叙事结构显式化，集成对话界面与节点编辑兼顾控制粒度。

## 方法详解

### 整体框架

四层系统架构：(1) 对话接口接收用户输入；(2) 任务路由 agent 解析意图；(3) 各类生成模块（Generator、Reasoner、Diagrammer、Editor、ContextGenerator）；(4) 多模态生成（GPT-Image-1 + Sora + GPT-4o TTS）。

### 关键设计

1. **节点图表示**

    - 功能：将故事拆解为节点（场景/事件）和边（叙事流），支持线性、分支、任意 DAG 结构
    - 核心思路：每个节点包含文本段+多媒体资产，边表示 sequential 或 parallel 依赖
    - 设计动机：显式结构使控制性远超线性接口

2. **LLM 驱动的生成管线**

    - 功能：从用户提示到完整故事图的自动化构建
    - 核心思路：Generator 写故事 → Reasoner 分解为节点图 → Diagrammer 输出 JSON → 图像/视频/音频生成
    - 设计动机：分离叙事逻辑和媒体生成，便于局部修改

3. **编辑与分支能力**

    - 功能：支持手工编辑、AI 辅助编辑、全局风格修改、分支复制与比较
    - 核心思路：复制节点或整个分支创建替代方案，并行渲染多版本进行侧向对比
    - 设计动机：非破坏性迭代——探索替代叙事无需重生成整个序列

4. **跨节点一致性维护**

    - 功能：在多模态内容生成中保持角色、场景等要素的一致性
    - 核心思路：滚动故事上下文（前 5 个节点的累积文本）用于指导后续节点的图像/视频生成
    - 设计动机：防止分段生成导致的视觉和语义不一致

### 损失函数 / 训练策略

本系统基于预训练模型（GPT-4、Sora 等）的推理，无需训练。

## 实验关键数据

### 主实验

| 故事类型 | 成功率 | 节点数 | 说明 |
|---------|-------|--------|------|
| 线性故事 | 80% | 8-12 | 无分支 |
| 分支故事 | 100% | 8-12 | 双路径 |
| JSON 正确解析 | 95% | - | 格式严格 |

### 消融实验（编辑工作流有效性）

| 编辑类型 | 成功率 | 说明 |
|---------|-------|------|
| 手工文本修改 | 100% | 立即反映，效果最佳 |
| AI 高层修改(tone/style) | 95% | 偶尔语义不一致 |
| 全局重写 | 85% | 某些节点失序 |
| 分支复制+比较 | 100% | 清晰对比 |

### 关键发现

- 跨节点媒体生成一致性约 70-80%，是主要瓶颈
- 线性故事失败原因多为过度分支或循环边
- 节点图结构有效支持了迭代式创作流程

## 亮点与洞察

- **节点图中心化**：显式故事结构表示使控制性远超 Sora 等线性接口。这个交互范式可以迁移到教育内容创作、游戏叙事设计等场景。
- **人-AI 协作模式**：结合对话自然性和节点级精确性，既保留高层意图也允许细粒度修改。
- **非破坏性迭代**：分支探索的设计使创作者可以无风险地尝试不同叙事方向。

## 局限与展望

- 当前限于 8-12 节点故事，长篇（50+ 节点）的一致性维护困难
- 滚动上下文（前 5 节点）对大图不足，需要基于嵌入的全局一致性机制
- 缺乏与真实创作者的用户测试
- 依赖 OpenAI API，模型迭代可能破坏工作流

## 相关工作与启发

- **vs ComfyUI**：ComfyUI 对扩散模型管线做了类似的节点图可视化；本文将节点图扩展到叙事层面
- **vs Twine**：Twine 是交互式小说的节点图编辑器，本文加入了多模态生成能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 节点图+多模态组合新颖，各部分并非开创
- 实验充分度: ⭐⭐⭐ 定性案例为主，需用户研究和大规模评估
- 写作质量: ⭐⭐⭐⭐ 系统设计清晰
- 价值: ⭐⭐⭐⭐ 对创意行业有吸引力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](../../CVPR2025/audio_speech/vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[ICML 2025\] Sounding that Object: Interactive Object-Aware Image to Audio Generation](../../ICML2025/audio_speech/sounding_that_object_interactive_object-aware_image_to_audio_generation.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)

</div>

<!-- RELATED:END -->
