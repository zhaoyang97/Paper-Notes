---
title: >-
  [论文解读] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction
description: >-
  [ICML 2025][LLM Agent][GUI Agent] 提出 Aguvis，首个完全基于纯视觉的跨平台自主 GUI Agent 框架，通过统一视觉观察空间、标准化动作空间和内心独白（inner monologue）机制，在离线和在线基准上取得 SOTA，无需依赖闭源模型。 现有 GUI 自动化方法存在三个核心瓶颈…
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Pure Vision"
  - "Cross-Platform"
  - "Inner Monologue"
  - "Two-Stage Training"
---

# Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction

**会议**: ICML 2025

**arXiv**: [2412.04454](https://arxiv.org/abs/2412.04454)

**作者**: Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, Caiming Xiong (HKU & Salesforce Research)

**领域**: LLM Agent

**关键词**: GUI Agent, Pure Vision, Cross-Platform, Inner Monologue, Two-Stage Training

**代码**: [GitHub](https://github.com/xlang-ai/aguvis) | [项目页面](https://aguvis-project.github.io/)

---

## 一句话总结

提出 Aguvis，首个完全基于纯视觉的跨平台自主 GUI Agent 框架，通过统一视觉观察空间、标准化动作空间和内心独白（inner monologue）机制，在离线和在线基准上取得 SOTA，无需依赖闭源模型。

## 研究背景与动机

现有 GUI 自动化方法存在三个核心瓶颈：

**依赖文本表示**：大多数 GUI Agent 需要 HTML DOM 树、accessibility tree 等文本化结构，但这些表示跨平台不通用、信息冗余且难以捕获视觉布局

**动作空间碎片化**：Web、桌面、移动端各自定义了不同的交互操作集合，模型难以跨平台迁移

**推理能力不足**：现有方法缺乏显式的规划和推理链路，在复杂多步 GUI 任务中频繁失败

作者观察到，人类操作 GUI 时主要依靠视觉感知而非底层代码结构，因此从"纯视觉"角度出发，统一所有平台的输入为截图图像，是更自然且更具泛化能力的方案。

## 方法详解

### 整体框架

Aguvis 包含三个关键设计：**统一纯视觉输入**、**标准化跨平台动作空间**和**内心独白推理机制**。

- **输入**：直接接收屏幕截图作为观察，无需任何文本化中间表示
- **输出**：统一的动作序列（click、type、scroll、press 等），通过插件系统适配不同平台
- **推理**：在动作预测前生成自然语言的"内心独白"，显式进行任务分解和当前状态分析

### 数据集构建 — Aguvis Data Collection

构建了大规模多模态 GUI Agent 训练数据集，包含：

- **GUI 定位标注**：截图中 UI 元素的坐标标注，训练模型理解"屏幕上哪里有什么"
- **推理标注**：为每个操作步骤添加规划和推理的自然语言描述
- 涵盖 Web、桌面（Windows/Linux/macOS）、移动端（Android）多个平台

### 两阶段训练管线

**Stage 1 — GUI Grounding（视觉定位）**：

- 目标：让模型学会在截图中精确定位 UI 元素
- 训练任务：给定自然语言描述，预测元素的边界框坐标
- 损失函数为标准的坐标回归损失：

$$\mathcal{L}_{\text{ground}} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{b}_i - b_i \|_2^2$$

其中 $\hat{b}_i$ 为预测坐标，$b_i$ 为 ground truth 坐标

**Stage 2 — Planning & Reasoning（规划与推理）**：

- 目标：在已具备定位能力的基础上，学习多步任务规划和推理
- 引入 inner monologue 机制，模型先生成当前状态分析和下一步计划，再输出具体动作
- 训练数据为完整的任务轨迹，包含 (截图, 内心独白, 动作) 三元组序列

### 关键设计 — Inner Monologue

内心独白包含三个要素：

1. **观察总结**：描述当前屏幕状态（如"当前在文件管理器中，已打开文档"）
2. **任务进度**：评估距离目标还有哪些步骤
3. **下一步计划**：明确下一个操作及其原因

这种显式推理链路显著提升了复杂任务（如跨应用操作）的成功率。

## 实验关键数据

### 离线实验 — ScreenSpot 视觉定位

| 模型 | Mobile Text | Mobile Icon | Desktop Text | Desktop Icon | Web Text | Web Icon | 平均 |
|------|:-----------:|:-----------:|:------------:|:------------:|:--------:|:--------:|:----:|
| GPT-4o | 78.0 | 24.9 | 72.9 | 43.6 | 70.3 | 22.3 | 52.0 |
| CogAgent | 67.0 | 24.9 | 74.2 | 20.0 | 70.4 | 28.6 | 47.5 |
| SeeClick | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| OS-Atlas-7B | 93.0 | 72.2 | 91.8 | 62.7 | 85.7 | 49.0 | 75.7 |
| **Aguvis-7B** | **93.4** | **76.9** | **91.0** | **66.8** | **85.7** | **55.5** | **78.2** |

### 在线实验 — OSWorld 真实桌面环境

| 模型 | 总体成功率 (%) |
|------|:-----------:|
| GPT-4o | 5.03 |
| GPT-4o + SoM | 4.59 |
| SecClick | 9.21 |
| OS-Atlas-Base-4B | 11.65 |
| OS-Atlas-Base-7B | 14.63 |
| **Aguvis-7B** | **14.79** |
| Human | 72.36 |

### 在线实验 — AndroidWorld & Mind2Web-Live

在 AndroidWorld 真实 Android 环境中，Aguvis-72B 达到 **21.8%** 任务成功率，在 MobileMiniWob++ 上达到 **62.5%**，均超越同规模基线。

在 Mind2Web-Live 真实网页交互中，Aguvis-7B 以纯视觉输入在任务成功率上可与使用 HTML 文本输入的方法竞争，同时推理成本更低。

### 消融实验

| 配置 | ScreenSpot Avg | OSWorld SR |
|------|:--------------:|:----------:|
| 完整 Aguvis-7B | 78.2 | 14.79 |
| 去掉 Stage 1（无 grounding 预训练） | 69.5 | 10.2 |
| 去掉 Inner Monologue | 74.8 | 11.6 |
| 仅用 Web 数据训练 | 72.1 | 8.9 |
| 仅用 Mobile 数据训练 | 70.3 | 7.5 |

关键发现：
- 两阶段训练中 Stage 1 对定位能力贡献最大（+8.7 pp）
- Inner Monologue 对复杂在线任务提升显著（+3.2 pp OSWorld SR）
- 跨平台混合训练数据带来明显的迁移增益

## 亮点与洞察

1. **纯视觉路线的可行性验证**：首次证明不依赖 HTML/DOM 的纯视觉 GUI Agent 可以在真实环境中达到 SOTA，这对跨平台部署有重大意义
2. **训练解耦的巧妙设计**：将"看到哪里"（grounding）和"决定做什么"（planning）分离训练，避免两个能力相互干扰
3. **开源生态贡献**：完整开源数据集、模型和训练代码，填补了社区在开源视觉 GUI Agent 方面的空白

## 局限性

1. 与人类 72.36% 的 OSWorld 成功率相比，14.79% 仍有巨大差距
2. 纯视觉方式在文本密集型 UI（如代码编辑器、电子表格）中可能不如 DOM 解析准确
3. 当前 inner monologue 为自然语言形式，缺乏形式化的错误恢复机制
4. 对视觉模型的分辨率和推理速度有较高要求，72B 版本部署成本较高

## 相关工作与启发

- **CogAgent / OS-Atlas**：同样探索视觉定位方法，但未实现完全自主的纯视觉 Agent
- **SeeClick / WebAgent**：依赖 HTML 等文本表示，跨平台能力有限
- **AppAgent / MobileAgent**：专注移动端，缺乏跨平台统一框架

启发：纯视觉 + 内心独白的范式可以推广到机器人操作、游戏 AI 等需要视觉理解+多步规划的场景。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|:----------:|------|
| 创新性 | 4 | 纯视觉路线 + 两阶段训练 + inner monologue 的组合较新颖 |
| 实验充分度 | 5 | 离线+在线、多平台、多基准全面覆盖 |
| 实用价值 | 4 | 完整开源，可直接用于 GUI 自动化产品 |
| 写作清晰度 | 4 | 方法描述清晰，实验组织系统 |
| **总分** | **4.25** | 高质量的系统性工作 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](../../ACL2025/llm_agent/os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)
- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](../../ACL2025/llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[CVPR 2026\] MMBench-GUI: A Unified Hierarchical Evaluation Framework for Multi-Platform GUI Agents](../../CVPR2026/llm_agent/mmbench-gui_a_unified_hierarchical_evaluation_framework_for_multi-platform_gui_a.md)

</div>

<!-- RELATED:END -->
