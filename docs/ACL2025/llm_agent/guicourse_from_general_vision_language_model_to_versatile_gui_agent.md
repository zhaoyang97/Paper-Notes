---
title: >-
  [论文解读] GUICourse: From General Vision Language Model to Versatile GUI Agent
description: >-
  [ACL 2025][LLM Agent][GUI Agent] 本文提出GUICourse——一套用于从通用视觉语言模型（VLM）训练多功能GUI代理的数据集系列（GUIEnv/GUIAct/GUIChat），通过两阶段训练流程先增强OCR和定位能力、再注入GUI知识，使得3.1B参数的小模型也能在网页和手机GUI导航任务上取得有效表现。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Vision Language Model"
  - "OCR"
  - "数据驱动"
  - "网页导航"
---

# GUICourse: From General Vision Language Model to Versatile GUI Agent

**会议**: ACL 2025  
**arXiv**: [2406.11317](https://arxiv.org/abs/2406.11317)  
**代码**: [https://github.com/RUCBM/GUICourse](https://github.com/RUCBM/GUICourse)  
**领域**: LLM Agent  
**关键词**: GUI Agent, Vision Language Model, OCR与定位, 数据驱动, 网页导航

## 一句话总结
本文提出GUICourse——一套用于从通用视觉语言模型（VLM）训练多功能GUI代理的数据集系列（GUIEnv/GUIAct/GUIChat），通过两阶段训练流程先增强OCR和定位能力、再注入GUI知识，使得3.1B参数的小模型也能在网页和手机GUI导航任务上取得有效表现。

## 研究背景与动机
图形用户界面（GUI）是人机交互的核心媒介，GUI代理旨在自动完成用户在各类GUI系统上的复杂任务。现有VLM（如LLaVA、Qwen-VL）虽在图像描述和视觉问答上表现优异，但在GUI场景下面临两大核心挑战：（1）OCR和定位能力不足——无法准确识别网页截图中不同字体、不同位置的文本并给出精确位置；（2）缺乏GUI相关知识——不了解GUI元素的功能和控制方法，无法完成"点击登录""查找商品"等导航指令。

现有GUI数据集要么环境过于简化（MiniWoB++）、要么领域过窄（WebShop只做购物）、要么数据量太小（RUSS仅80条指令）。在真实网页场景下，同时包含单步和多步指令的大规模数据集十分稀缺。

本文的核心思路是：通过构建系统化的数据-训练流水线，先补齐VLM的基础感知能力（OCR+定位），再通过导航数据注入GUI交互知识，用纯数据驱动的方式将通用VLM转化为实用GUI代理。

## 方法详解

### 整体框架
GUICourse采用两阶段训练流水线：
- **阶段一（预训练）**：使用GUIEnv-global数据集（1000万样本）增强VLM的OCR和定位基础能力
- **阶段二（SFT微调）**：使用GUIEnv-local（70k）+ GUIAct（全部）+ GUIChat（全部）注入GUI元素知识、导航能力和对话交互能力

GUI代理仅接收截图作为输入（纯视觉），输出带位置信息的动作（position-based），定义了11种统一动作空间（click、tap、swipe等）。

### 关键设计

1. **GUIEnv：大规模OCR与定位数据集**

    - GUIEnv-global：从C4语料库收集400万URL，用Playwright渲染获得1000万网页截图-标注对，每个样本包含页面全部可描述内容（文本、定位框、布局序列），用于预训练
    - GUIEnv-local：从中筛选5万截图，裁剪为≤1920×1080的子图，每张随机采样10个带文本+位置的元素，构建"text2bbox"和"bbox2text"双向任务的70万条SFT数据
    - 设计动机：GUI场景的截图通常很大、字体多样、文本密集，通用VLM的OCR在这种场景下严重不足

2. **GUIAct：GUI导航数据集**

    - web-single（67k条）：覆盖50个场景、13k个真实网站，用GPT-4V自动标注单步指令-动作对，再经人工校验（准确率从55%提升至92%）
    - web-multi（5,696条）：8个顶层场景、32个子场景、121个知名网站，众包标注多步导航指令，平均7.9步/任务
    - smartphone（9,157条）：从AITW数据集的General子集转换而来，适配统一动作空间
    - 这是当时真实网页场景下最大的同时包含单步和多步指令的导航数据集

3. **GUIChat：文本丰富的多模态QA数据集**

    - 44k单轮QA + 6k多轮对话，覆盖视觉信息查询、人类相关问题、世界知识和复杂推理四类
    - 对话中包含带定位框的网页截图，用GPT-4基于网页文本表示生成
    - 目的是增强GUI代理的自然语言交互能力

### 训练策略
- 以MiniCPM-V（3.1B）为基础训练MiniCPM-GUI，先将GUIEnv-global融入预训练数据，再用所有SFT数据微调
- 支持高分辨率版本（1344×1344 vs 默认448×448），采用灵活分块切片策略
- 还基于Qwen-VL和Fuyu-8B训练了Qwen-GUI和Fuyu-GUI，但仅使用SFT数据

## 实验关键数据

### 主实验（自建测试集）

| 模型 | Web-Single StepSR | Web-Multi StepSR | Smartphone StepSR | 均值 |
|------|-------------------|-------------------|---------------------|------|
| GPT-4o-mini | 57.0 | 17.0 | 22.0 | 32.0 |
| MiniCPM-GUI (高分辨率+GUIChat) | 70.6 | 47.5 | 53.3 | 57.1 |
| Fuyu-GUI | 63.5 | 47.1 | 40.4 | 50.4 |
| Qwen-GUI | 66.7 | 46.8 | 58.1 | **57.2** |

### 跨任务泛化（Mind2Web）

| 模型 | Cross-Task StepSR | Cross-Website StepSR | Cross-Domain StepSR |
|------|-------------------|----------------------|---------------------|
| Qwen-VL（基线） | 20.3 | 14.0 | 12.3 |
| Qwen-GUI | 24.4 | 15.6 | 17.5 |
| MiniCPM-V（基线） | 8.5 | 6.0 | 5.2 |
| MiniCPM-GUI | 20.8 | 17.3 | 14.6 |

### 消融实验（GUIEnv数据量影响）

| GUIEnv数据量 | Text2Bbox IoU@0.5 | Web-Single StepSR |
|-------------|-------------------|-------------------|
| 0 | 2.15 | 52.84 |
| 2.5M | 25.32 | 64.82 |
| 10M | 47.96 | **70.57** |

### 关键发现
- **小模型也能胜任**：3.1B的MiniCPM-GUI（57.1）与9.6B的Qwen-GUI（57.2）表现相当
- **高分辨率至关重要**：高分辨率使均值从49.0提升至56.1
- **OCR和定位是导航的前置条件**：无GUIEnv时定位IoU@0.5仅2.15，加入10M后升至47.96，导航StepSR也从52.84升至70.57
- **GUIChat数据有助于网页任务**：虽然主要目标是交互能力，但额外贡献了1个点的StepSR提升
- 错误分析显示：50个错误样本中13个是"合理但不匹配金标"的假错误，22个是动作类型错误，15个是位置错误

## 亮点与洞察
- **系统化的数据-训练流水线思路**：明确将问题分解为"基础能力+领域知识"两层，用对应数据分阶段解决，方法论清晰
- **数据构建的工程细节充分**：GPT-4V自动标注+人工校验的流程实用，准确率从55%提升至92%，说明自动标注需要质检
- **统一动作空间设计**：11种动作覆盖网页+手机场景，便于跨平台迁移
- **消融实验揭示了定位能力与导航能力的正相关**：这一发现对后续GUI代理研究有实际指导意义

## 局限与展望
- 仅使用预训练+SFT，未引入RLHF或强化学习进一步提升代理能力
- 仅覆盖网页和智能手机场景，缺少桌面系统、专业软件的训练数据
- 静态评估存在固有缺陷——同一指令可能有多种合理操作路径，但只匹配预定义金标
- 多步导航的评估仅在step级别，缺少end-to-end任务完成率指标
- **研究idea**：可以考虑引入在线交互式评估（模拟环境中实际执行agent动作），以及通过RLHF让agent从执行反馈中持续学习

## 相关工作与启发
- SeeClick（2024）：同样关注GUI定位能力，本文在Mind2Web上与之对比
- CogAgent：视觉语言模型做GUI代理的代表工作
- Mind2Web和AITW：两个主流GUI导航基准
- 本文证明了"先强化基础视觉能力、再注入领域知识"的训练范式的有效性，这一思路可推广到其他需要特定视觉能力的代理场景

## 评分
- 新颖性: ⭐⭐⭐ 方法上是标准的数据驱动+两阶段SFT，新颖性主要体现在数据构建
- 实验充分度: ⭐⭐⭐⭐ 自建测试+跨任务泛化+消融+案例分析+错误分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据集统计表格丰富，pipeline描述明确
- 价值: ⭐⭐⭐⭐ 提供了开源的大规模GUI训练数据集，对社区有实际贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[NeurIPS 2025\] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](../../NeurIPS2025/llm_agent/btlui_blinkthinklink_reasoning_model_for_gui_agent.md)
- [\[ACL 2025\] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model](can_a_single_model_master_both_multi-turn_conversations_and_tool_use_coalm_a_uni.md)
- [\[ICML 2025\] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](../../ICML2025/llm_agent/aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)

</div>

<!-- RELATED:END -->
