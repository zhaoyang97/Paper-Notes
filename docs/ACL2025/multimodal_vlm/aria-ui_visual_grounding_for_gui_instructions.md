---
title: >-
  [论文解读] Aria-UI: Visual Grounding for GUI Instructions
description: >-
  [ACL 2025][多模态VLM][GUI Grounding] 提出 Aria-UI，一个专为 GUI 视觉定位设计的纯视觉多模态模型，通过可扩展的指令合成数据管线和文本-图像交错的动作历史机制，在离线和在线 Agent 基准上均达到 SOTA，包括 AndroidWorld 第1名（44.8%）和 OSWorld 第3名（15.2%）。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "GUI Grounding"
  - "visual grounding"
  - "多模态大模型"
  - "GUI Agent"
  - "Action History"
---

# Aria-UI: Visual Grounding for GUI Instructions

**会议**: ACL 2025  
**arXiv**: [2412.16256](https://arxiv.org/abs/2412.16256)  
**代码**: [https://github.com/AriaUI/Aria-UI](https://github.com/AriaUI/Aria-UI)  
**领域**: 多模态 / GUI Agent  
**关键词**: GUI Grounding, visual grounding, 多模态大模型, GUI Agent, Action History

## 一句话总结

提出 Aria-UI，一个专为 GUI 视觉定位设计的纯视觉多模态模型，通过可扩展的指令合成数据管线和文本-图像交错的动作历史机制，在离线和在线 Agent 基准上均达到 SOTA，包括 AndroidWorld 第1名（44.8%）和 OSWorld 第3名（15.2%）。

## 研究背景与动机

数字自动化 Agent 需要根据语言指令在 GUI 界面上定位并操作目标元素（如按钮、输入框等），这一过程称为 **GUI Grounding**。这是 GUI Agent 的核心能力——只有准确地将指令映射到屏幕上的具体元素，Agent 才能执行点击、输入等操作。

现有方法面临三大痛点：

**依赖结构化辅助输入（HTML/AXTree）**：大多数方法依赖 DOM 树或可访问性树（AXTree）作为输入，但这些信息在不同平台（web、mobile、desktop）上获取方式不一致，且某些场景下不可用（如原生应用、游戏界面）。更重要的是，这些结构化输入可能非常冗长，处理成本高。

**指令格式多样性不足**：不同的 planning agent 会生成不同格式的 grounding 指令——有的是简短的元素描述（"search button"），有的是包含推理过程的长指令（"To complete the task, I need to click on..."），现有模型难以适应这种异构性。

**缺乏上下文感知能力**：在多步骤任务执行中，当前步骤的 grounding 需要理解之前的操作历史。例如，"点击下一步"这个指令的目标元素取决于之前做了什么，但现有方法大多忽略历史上下文。

核心idea：**用纯视觉方法替代结构化输入依赖，通过合成多样化指令数据和建模动作历史来增强 GUI grounding 能力**。

## 方法详解

### 整体框架

Aria-UI 基于 Aria 多模态模型（MoE 架构，3.9B 激活参数/token），输入为 GUI 截图 + 文本指令（可选包含历史），输出为目标元素的相对坐标 $(x, y)$，坐标归一化到 $[0, 1000]$ 范围。

Pipeline 分为两阶段：
- **阶段一**：在大规模合成的多样化 grounding 指令数据上进行 SFT
- **阶段二**：在包含动作历史的 episode 数据上进行上下文感知微调

### 关键设计

1. **可扩展的指令合成数据管线 (Scalable Data Pipeline)**:

    - 功能：从现有 GUI 数据集中自动合成多样化和高质量的 grounding 指令样本
    - 核心思路：收集多个来源的 GUI 截图和元素标注数据，然后利用 LLM（如 GPT-4）生成不同格式的指令——包括简短描述型（"the search icon"）、推理型（"To find videos, I should click on the search button"）、和直接坐标型指令。对每个元素生成多种表述，大幅增加训练数据的多样性
    - 设计动机：真实场景中的 planning agent 输出指令格式千差万别，如果训练数据只包含单一格式，模型的泛化能力会受限。通过合成多种格式，Aria-UI 能适配不同上游 agent 的输出

2. **文本-图像交错的动作历史建模 (Action History Modeling)**:

    - 功能：将历史操作以文本和截图交错的方式编码为上下文，辅助当前步骤的 grounding
    - 核心思路：支持两种历史格式：(a) **纯文本历史**——仅用文字描述之前的操作（如 "Step 1: clicked on Settings"）；(b) **文本-图像交错历史**——每一步操作附带当时的 GUI 截图。模型在 SFT 阶段同时在两种格式上训练
    - 设计动机：在动态任务执行中（如 AndroidWorld），当前界面状态是之前操作的结果。不理解历史就无法正确 grounding——例如"点击确认"在不同对话框中的目标完全不同。而图像历史比纯文本提供更丰富的上下文线索

3. **MoE 架构与超分辨率支持**:

    - 功能：高效处理不同尺寸和宽高比的 GUI 截图
    - 核心思路：基于 Aria 模型的 Mixture-of-Experts (MoE) 架构，每个 token 仅激活 3.9B 参数（总参数更大），支持将 GUI 截图切分为多个 patch 进行超分辨率编码（split_image=True, max_image_size=980），能处理高达 ~980px 的图像 patch
    - 设计动机：GUI 截图中的文字和小图标需要高分辨率才能准确识别。MoE 架构在保持轻量推理的同时提供了足够的模型容量

### 训练策略

- 阶段一：在合成的静态 grounding 数据上 SFT，学习基础的指令-元素对齐能力
- 阶段二：在包含动作历史的 episode 数据（~992K instruction-output pairs）上进一步微调，学习上下文感知能力
- 输出格式：模型直接输出坐标文本如 `[523, 187]`，表示目标元素在相对坐标系中的位置

## 实验关键数据

### 主实验（离线 Grounding 基准 - ScreenSpot 系列）

| 基准 | 指标 | Aria-UI | 之前视觉SOTA | AXTree方法 | 说明 |
|------|------|---------|-------------|-----------|------|
| ScreenSpot | Acc | SOTA | - | - | 多平台 GUI 定位准确率 |
| ScreenSpot-V2 | Acc | SOTA | - | - | 扩展版本 |
| ScreenSpot-Pro | Acc | SOTA | - | - | 专业级难度 |

Aria-UI 在 ScreenSpot 系列所有子基准上均超越了纯视觉方法和依赖 AXTree 的方法。

### 在线 Agent 基准

| 基准 | 指标 | Aria-UI | 排名 | 说明 |
|------|------|---------|------|------|
| AndroidWorld | Task SR (%) | 44.8 | 🏆 第1 | 移动端，多步骤真实任务 |
| OSWorld | Task SR (%) | 15.2 | 🥉 第3 | 桌面端，复杂真实任务 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无指令多样化 | 下降显著 | 单一格式指令导致泛化能力差 |
| 无动作历史 | 下降明显 | 多步骤任务依赖上下文 |
| 纯文本历史 vs 文本-图像交错历史 | 交错更优 | 图像提供更丰富的上下文信号 |
| Base模型 vs Context-aware模型 | Context-aware更优 | 二阶段训练的增益 |

### 关键发现

- **纯视觉方法可以超越结构化输入方法**：Aria-UI 在不使用 HTML/AXTree 的情况下超越了依赖这些输入的基线
- **指令多样性是关键**：合成多种格式的训练指令对跨 agent 泛化至关重要
- **动作历史显著提升多步骤任务表现**：尤其在 AndroidWorld 和 OSWorld 等需要多轮交互的在线基准上
- **MoE 架构的效率优势**：3.9B 激活参数即可达到 SOTA，远小于全参数密集模型

## 亮点与洞察

- **纯视觉路线的胜利**：证明了不依赖 DOM/AXTree 的纯视觉方法不仅可行，而且更优，这对跨平台 GUI Agent 的实际部署意义重大
- **数据工程 > 模型工程**：Aria-UI 的核心创新不在模型架构，而在训练数据——这提示了 GUI Agent 领域数据质量和多样性的重要性
- **开源生态完整**：模型权重、训练数据、推理代码全部开源，包括 HuggingFace Space 在线 demo
- **可扩展性强**：数据合成管线和上下文建模方法可以迁移到其他平台和任务

## 局限与展望

- 对于高度动态的界面（如视频流、动画）可能不够鲁棒
- 坐标预测精度依赖截图分辨率，极小元素可能定位困难
- Context-aware 模式需要存储和处理历史截图，增加推理开销
- 目前主要在英文 GUI 上验证，多语言界面的表现未知
- MoE 模型的部署比密集模型更复杂，需要特定推理框架支持

## 相关工作与启发

- **vs CogAgent**: CogAgent 同样采用纯视觉方法，但模型更大（18B），Aria-UI 用 MoE 架构在更少激活参数下达到更好效果
- **vs SeeClick/UGround**: 这些方法通常只处理单一指令格式，Aria-UI 的指令合成管线显著提升了泛化能力
- **vs Set-of-Mark**: SoM 在截图上叠加标注来辅助定位，是一种半结构化方法，而 Aria-UI 完全端到端

## 评分

- 新颖性: ⭐⭐⭐⭐ 纯视觉路线和指令合成管线是清晰的技术贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 离线+在线基准全覆盖，消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ 开源完整，在 GUI Agent 领域有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](r-vlm_region-aware_vision_language_model_for_precise_gui_grounding.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](../../CVPR2026/multimodal_vlm/drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)
- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](../../CVPR2026/multimodal_vlm/widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)

</div>

<!-- RELATED:END -->
