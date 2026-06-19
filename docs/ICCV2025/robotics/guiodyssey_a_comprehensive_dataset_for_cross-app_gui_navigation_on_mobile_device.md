---
title: >-
  [论文解读] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices
description: >-
  [ICCV 2025][机器人][GUI导航] 提出 GUIOdyssey，首个面向移动端跨应用 GUI 导航的综合数据集（8334 episodes、212 apps、1357 app 组合），以及 OdysseyAgent——配备历史重采样模块的多模态导航智能体，在平衡性能与推理效率的同时显著提升跨应用任务表现。
tags:
  - "ICCV 2025"
  - "机器人"
  - "GUI导航"
  - "跨应用程序"
  - "移动端智能体"
  - "多模态大模型"
  - "历史信息建模"
---

# GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices

**会议**: ICCV 2025  
**arXiv**: [2406.08451](https://arxiv.org/abs/2406.08451)  
**代码**: [GitHub](https://github.com/OpenGVLab/GUI-Odyssey)  
**领域**: 机器人  
**关键词**: GUI导航, 跨应用程序, 移动端智能体, 多模态大模型, 历史信息建模

## 一句话总结

提出 GUIOdyssey，首个面向移动端跨应用 GUI 导航的综合数据集（8334 episodes、212 apps、1357 app 组合），以及 OdysseyAgent——配备历史重采样模块的多模态导航智能体，在平衡性能与推理效率的同时显著提升跨应用任务表现。

## 研究背景与动机

智能手机上的 GUI 导航智能体可以自动执行用户指令，对视障人士和日常效率提升都有重要价值。近年来，大型基础模型的发展使得自主 GUI 导航日益可行。然而，现有 GUI 导航数据集和方法面临一个根本性问题：

**几乎所有数据集都局限于单应用导航**。现实场景中，用户经常需要跨多个应用完成任务，例如：从浏览器搜索信息后记录到笔记应用、在音乐应用找到歌曲后分享到社交媒体、协调日历和消息应用安排日程。这类跨应用任务有三大独特挑战：

**更长的操作序列**：跨应用任务平均需要 15.3 步（远超单应用的 5-7 步），错误传播风险成倍增加——一步出错可能导致后续全部失效

**更复杂的工作记忆管理**：关键 UI 元素和上下文信息跨越多个应用，智能体需要在应用切换时保持关键信息的记忆

**更广泛的功能知识**：需要理解不同应用的交互模式（文件分享、邮件撰写、消息发送等）并在它们之间建立工作流

已有研究的评测表明：**当前模型在跨应用任务上的表现远差于单应用任务**。然而，尚无专门的跨应用训练数据集来弥补这一差距。

## 方法详解

### 整体框架

本工作包含两部分：(1) **GUIOdyssey 数据集**——通过人工标注 + GPT-4/4o 增强构建的跨应用导航数据集；(2) **OdysseyAgent**——基于 Qwen-VL 微调的多模态导航智能体，通过历史重采样模块高效处理长序列历史信息。

### 关键设计

1. **数据集构建流程**：采用四阶段流水线保证数据质量和多样性：

   **(a) 跨应用任务提案**：6 大类任务——通用工具、信息管理、网上购物、媒体娱乐、社交分享、多应用复合。91 个高层指令模板由人工参与者 + GPT-4 共同设计。

   **(b) 灵活的指令实例化**：通过三种方式确保多样性——替换模板中的 item（如 "yoga" → "meditation"）、选择不同的 app 来完成同一任务（如 Spotify vs Google Podcast）、GPT-4 改写不同表达方式。

   **(c) 人工标注**：经过训练的标注员在 Android 模拟器上逐步完成指令，记录每一步的截图和操作。覆盖 6 种设备（Pixel Pro、Tablet、Fold 等）。动作集包含 9 种：CLICK、SCROLL、LONG PRESS、TYPE、COMPLETE、IMPOSSIBLE、HOME、BACK、RECENT。

   **(d) 细粒度增强标注**：GPT-4o 为每一步生成三层语义标注——屏幕描述（当前页面内容）、上下文信息（前序步骤摘要）、决策理由（为什么执行下一个动作）。同时生成低层指令作为高层指令的原子化分解。最后进行截图完整性、动作准确性和指令一致性的质量检查。

2. **OdysseyAgent 的历史重采样模块**：跨应用导航的核心挑战是处理大量历史截图和动作序列——需要记住前几个应用中的操作结果以做出当前决策，但直接拼接所有历史截图 token 会严重拖慢推理速度。

   OdysseyAgent 在 Qwen-VL 的基础上引入**历史重采样器**——一个单层交叉注意力模块：
    - Query：可学习嵌入
    - Key/Value：历史截图 token
    - 输出压缩后的历史 token 与当前截图 token、用户指令、前序动作拼接，送入 LLM 预测下一步动作

   预测目标为标准的 next-token prediction：
    $\mathcal{L} = \sum_{i=1}^{N} P_\theta(A_i^t | X^{\{t, t-1, \cdots, t-\delta\}}, I_{user}, A_{<i}^t)$

   其中 $\delta$ 是历史图像窗口大小，$\theta$ 包括 VL adapter、历史重采样器和 LLM 的可训练参数。

3. **多维度评估设计**：数据集划分为 4 种设置以全面评估泛化性：

    - **Train-Random & Test-Random**（域内）
    - **Train-App & Test-App**（未见应用）
    - **Train-Task & Test-Task**（未见任务类型）
    - **Train-Device & Test-Device**（未见设备类型）

### 损失函数 / 训练策略

- 使用标准交叉熵损失训练 next-action prediction
- 基于 Qwen-VL-Chat 微调，保留视觉编码器、LLM 和 VL adapter
- 评估指标 AMS（Action Matching Score）：动作类型匹配 + CLICK/LONG PRESS 需在 14% 屏幕距离内 + SCROLL 方向匹配 + TYPE 使用 ANLS 评估
- Success Rate（SR）：所有步骤正确才算成功，步骤越长越难

## 实验关键数据

### 主实验

**Test-Random（域内）各方法对比**

| 方法 | 高层指令 AMS | 低层指令 AMS |
|------|-------------|-------------|
| GPT-4o (零样本) | 13.19 | 42.71 |
| Claude3.5-Sonnet (零样本) | 15.80 | 34.18 |
| Claude3.5 + OmniParser | 32.88 | 63.91 |
| InternVL2-Pro + OmniParser | 14.69 | 54.31 |
| Qwen-VL (微调) | 74.67 | 86.32 |
| OdysseyAgent (微调) | 75.79 | 86.88 |
| **OdysseyAgent* (+ 语义标注)** | **78.24** | **88.15** |

微调方法远超零样本，OdysseyAgent 加语义标注后达到最佳。

### 消融实验

**历史信息类型的影响（高层指令 AMS）**

| 配置 | 动作 | 截图 | 上下文 | Test-Random | Overall | SR |
|------|------|------|--------|-------------|---------|-----|
| (1) | × | × | × | 66.13 | 55.60 | 1.49 |
| (2) | ✓ | × | × | 74.67 | 63.44 | 5.18 |
| (3) | × | ✓ | × | 71.22 | 60.30 | 4.20 |
| (4) | × | × | ✓ | 75.25 | 64.77 | 5.06 |
| (5) | ✓ | ✓ | × | 75.79 | 63.60 | 4.76 |
| **(6)** | **✓** | **✓** | **✓** | **77.06** | **66.84** | **6.32** |

### 关键发现

- **上下文信息**（对历史步骤的文本摘要）单独使用时效果竟然**优于动作+截图的组合**（实验 4 vs 5）——抽象摘要比原始数据更有助于泛化
- 三种历史信息全部使用时效果最佳，Overall AMS 从 55.60 提升到 66.84（+20.2%）
- 域外性能下降显著：高层指令 AMS 从 78.24 降至 62.90（-19.6%），低层指令仅降 7.8%——说明复杂推理和规划能力仍不足
- CogAgent 和 SphAgent 虽然在其他 GUI 任务上表现不错，但在跨应用场景表现很差（<16% AMS），验证了跨应用与单应用之间的巨大域差距
- OmniParser 的 GUI 定位能力大幅提升了闭源模型表现（Claude3.5：15.80→32.88）

## 亮点与洞察

- **首个跨应用 GUI 数据集**填补了重要空白——验证了跨应用任务需要专门的训练数据，不能简单从单应用能力推导
- **上下文信息 > 原始截图+动作**的发现具有启发性——对历史信息的抽象和总结比原始记忆更重要
- 语义标注（屏幕描述 + 上下文 + 决策理由）的三层设计模拟了人类的认知过程
- 历史重采样器是一种轻量但有效的设计——单层交叉注意力即可实现性能与效率的良好平衡
- 数据集覆盖了折叠屏和平板等新设备类型，前瞻性较好

## 局限与展望

- Success Rate 仍然很低（域内最高 11.61%），15+ 步的长序列任务中错误累积严重
- 仅使用离线评估（AMS），未进行在线实机交互评估，可能高估实际能力
- 现有方法在未见任务类型上表现最差（AMS 58.83），高层推理和规划能力亟需提升
- 坐标式导航本身有脆弱性——依赖精确的屏幕坐标，如果分辨率或布局变化会失效
- Android 模拟器环境与实际手机使用可能存在物理差异（触控响应、动画延迟等）
- 未探索与 accessibility tree 辅助信息的结合

## 相关工作与启发

- **AITW** 是最大的单应用 GUI 数据集（715K episodes），但跨应用能力不足
- **AndroidControl** 提供高/低层指令但限于单应用
- **OmniParser** 的 GUI 定位能力启发——结构化理解可显著提升零样本性能
- 启发：未来 GUI 智能体需要更强的**工作记忆和跨应用推理能力**，可能需要引入检索增强或 scratchpad 机制来管理跨应用上下文

## 评分

- **新颖性**: ⭐⭐⭐⭐ 跨应用导航数据集是首创，但 OdysseyAgent 架构创新有限
- **实验充分度**: ⭐⭐⭐⭐⭐ 域内+域外四种设置、多种基线、历史信息消融详尽
- **写作质量**: ⭐⭐⭐⭐ 统计数据丰富，可视化直观，论述逻辑清晰
- **价值**: ⭐⭐⭐⭐⭐ 数据集对社区价值极高，揭示了跨应用 GUI 导航的核心挑战

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Hoi! - A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation](../../CVPR2026/robotics/hoi_-_a_multimodal_dataset_for_force-grounded_cross-view_articulated_manipulatio.md)
- [\[CVPR 2025\] ShowUI: One Vision-Language-Action Model for GUI Visual Agent](../../CVPR2025/robotics/showui_one_vision-language-action_model_for_gui_visual_agent.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](../../CVPR2025/robotics/gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[CVPR 2025\] SortScrews: A Dataset and Baseline for Real-time Screw Classification](../../CVPR2025/robotics/sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)

</div>

<!-- RELATED:END -->
