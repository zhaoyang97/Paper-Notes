---
title: >-
  [论文解读] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining
description: >-
  [ICCV 2025][多模态VLM][GUI理解] Iris 提出信息敏感裁剪（ISC）和自精炼双重学习（SRDL）两大核心创新，仅用 850K 标注数据即在多个 GUI 理解基准上达到 SOTA，性能匹敌使用 10 倍以上数据的方法，同时将处理时间从 3 秒缩短至 1 秒。 数字化 Agent 需要在网页、软件和操作系统…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "GUI理解"
  - "视觉Agent"
  - "信息敏感裁剪"
  - "自精炼双重学习"
  - "元素定位"
---

# Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining

**会议**: ICCV 2025  
**arXiv**: [2412.10342](https://arxiv.org/abs/2412.10342)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: GUI理解, 视觉Agent, 信息敏感裁剪, 自精炼双重学习, 元素定位

## 一句话总结

Iris 提出信息敏感裁剪（ISC）和自精炼双重学习（SRDL）两大核心创新，仅用 850K 标注数据即在多个 GUI 理解基准上达到 SOTA，性能匹敌使用 10 倍以上数据的方法，同时将处理时间从 3 秒缩短至 1 秒。

## 研究背景与动机

数字化 Agent 需要在网页、软件和操作系统等交互式环境中自动执行任务。基于文本的 Agent 依赖平台特定 API，维护成本高；而基于视觉的 Agent 通过直接与 GUI 交互实现跨平台适配，更具扩展性。然而，视觉 Agent 面临两个核心挑战：

**架构层面的局限**：GUI 界面通常是高分辨率的（如 1920×1080），且信息分布极不均匀——密集的 UI 元素与大面积空白背景共存。现有方法对图像做均匀分割，每个子图分配等量 token，无法根据信息密度自适应地分配计算资源，导致对细粒度区域感知不足或对空白区域浪费算力。

**训练数据的偏差**：标注数据偏向于大型、显眼的 UI 元素（如输入框、"OK"按钮），忽略了小型但交互关键的组件（如侧边栏按钮），导致模型对复杂布局和精细元素理解不足。获取全面标注的成本极高，极大限制了 Agent 的可扩展性。

这两个问题的本质在于：现有方法既无法高效处理 GUI 界面中异质化的视觉信息，也缺乏从困难样本中自主学习的能力。Iris 正是针对这两个瓶颈分别提出了 ISC 和 SRDL。

## 方法详解

### 整体框架

Iris 以 Qwen-VL 为基础模型，聚焦于两个互补任务：**Referring**（给定位置生成 UI 元素描述）和 **Grounding**（给定描述定位 UI 元素位置）。整体分为两个阶段：首先用 ISC 增强视觉训练，然后通过 SRDL 进行自精炼训练。

### 关键设计

1. **信息敏感裁剪（ISC）**

   ISC 的核心思想是根据 GUI 截图中视觉信息的分布进行自适应裁剪，使每个子图包含平衡的信息量。具体分三步：

    - **信息检测**：利用 Canny 边缘检测生成二值信息矩阵 $M \in \{0,1\}^{n \times m}$，其中 $M_{i,j}=1$ 表示该位置存在有意义的视觉信息（GUI 元素通常具有明显边界）。
    - **自适应裁剪**：采用多尺度滑动窗口方法，从最小窗口 $k_{\min}$ 开始，步长 $\text{step}=\max(k/4, 32)$，计算每个窗口的边缘密度。密度阈值随窗口增大而降低：$\rho_k = \rho_{\min} / (k/k_{\min})^2$。当密度超过阈值时提取该区域，并将已处理区域标记为零以避免重叠。窗口大小按因子 $\alpha$ 几何递增。
    - **统一缩放**：所有裁剪子图缩放至统一尺寸（如 $224 \times 224$），确保每个 visual token 都承载有意义的信息。

   ISC 在 CPU 上仅需不到 0.1 秒，可与 GPU 推理并行执行，不增加额外延迟。相比均匀分割，ISC 对简单界面使用极少 token，对复杂界面自动增加 token，实现了 **300% 的效率提升**。

2. **自精炼双重学习（SRDL）**

   SRDL 利用 Referring 和 Grounding 的互补关系构建自我强化的学习循环，核心流程为：

    - **双重学习循环**：对 GUI 图像中的每个 UI 元素，先执行 Grounding 获取位置 $\mathbf{p}$，再执行 Referring 从该位置重新生成描述 $D'$，再对 $D'$ 执行 Grounding 获取新位置。当连续迭代的位置稳定时（IoU 超过阈值 $\tau$），该样本被视为收敛，加入训练集。形式化为 $\text{Sim}(G(R(\mathbf{p})), \mathbf{p}) > \tau$。

    - **视觉困难样本挖掘**：利用 ISC 的信息矩阵 $M$ 计算谱熵 $H = -\sum_k p_k \log(p_k)$，其中 $p_k$ 为频率分量的归一化能量。高谱熵对应复杂的视觉区域，这些图像被优先送入双重学习循环进行额外训练。

    - **功能困难样本挖掘**：基于模型历史表现，收集模型在功能描述理解上表现差的样本 $\mathcal{D}_{\text{hard}}$，利用 LLM 生成描述变体 $\{D_i^{(1)}, D_i^{(2)}, \ldots, D_i^{(n)}\}$，作为合成的功能困难样本送入双重学习循环。

   SRDL 最终自动生成约 **3M 自标注样本**，在不需要额外人工标注的情况下带来 **10% 的准确率提升**。

### 损失函数 / 训练策略

训练遵循 SeeClick 的流程，从 Qwen-VL 初始化，使用 850K GUI 数据 + 150K LLaVA 通用视觉语言指令进行初始训练（ISC 增强视觉感知），随后进行 SRDL 阶段在 ~3M 自标注数据上训练。

## 实验关键数据

### 主实验

**ScreenSpot 基准**（GUI 元素定位准确率）：

| 模型 | GUI标注量 | Mobile Text | Mobile Icon | Desktop Text | Desktop Icon | Web Text | Web Icon | **平均** |
|------|----------|-------------|-------------|--------------|--------------|----------|----------|---------|
| SeeClick | 850K | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| UGround | 10M | 82.8 | 60.3 | 82.5 | 63.6 | 80.4 | 70.4 | 73.3 |
| **Iris** | **850K** | **85.3** | **64.2** | **86.7** | 57.5 | **82.6** | **71.2** | **74.6** |

**GroundUI-1K 基准**：

| 模型 | Web | Desktop | Mobile | 总分 |
|------|-----|---------|--------|------|
| SeeClick | 64.3 | 44.3 | 73.7 | 61.1 |
| **Iris** | **72.2** | **61.3** | **80.2** | **71.3** |

**下游 Agent 任务**（Mind2Web / AITW）：Iris 在 12 个评估类别中的 11 个上取得最佳性能。AITW 整体得分 63.6（SeeClick 59.3，GPT-4V 50.5）。

### 消融实验

| 配置 | 准确率 | 处理时间 | 说明 |
|------|--------|---------|------|
| Baseline (SeeClick) | ~53% | 0.5s | 基线 |
| + ISC only | ~64% | 1.0s | 信息敏感裁剪带来效率提升 |
| + SRDL w/o Visual Mining | 71.4% | - | 仅功能挖掘 |
| + SRDL w/o Functional Mining | 72.1% | - | 仅视觉挖掘 |
| **Full Iris (ISC + SRDL)** | **74.6%** | **1.0s** | 两者互补，最优 |

### 关键发现

- ISC 在低复杂度界面用更少 token 即达到高准确率，在高复杂度界面自动增加 token 保持精度，对比 AnyRes 全面优于后者。
- SRDL 的视觉与功能困难样本挖掘缺一不可，两者结合较单独使用提升 2.5%–3.2%。
- Iris 在 Web 和 Desktop 平台的提升最显著，因为这些平台分辨率更高、布局更复杂，更能体现 ISC 的优势。

## 亮点与洞察

- **数据效率极高**：仅用 850K 标注匹敌 10M 标注的 UGround，核心在于 SRDL 自主发现和学习困难样本，弥补了标注偏差。
- **ISC 设计优雅**：基于边缘检测的信息密度估计简单高效（<0.1s CPU），能在不增加推理延迟的前提下实现自适应裁剪。
- **双重学习的自洽性**：Referring 和 Grounding 互为验证，构成天然的数据质量过滤机制——只有收敛的样本才加入训练。

## 局限与展望

- 对视觉极度相似的 UI 元素（如颜色、形状几乎一致的按钮组）区分能力可能有限。
- SRDL 生成的自标注数据质量依赖初始模型的能力，如果初始模型太弱，可能陷入低质量循环。
- 未探索跨分辨率和跨设备的迁移能力。
- 对动态 GUI（如弹窗、动画）的处理未讨论。

## 相关工作与启发

- SeeClick 首先确立了 GUI Grounding 作为视觉 Agent 基础能力的重要性；Iris 在此基础上以极高的数据效率实现了质的突破。
- ISC 的自适应 token 分配思想可推广到其他高分辨率视觉任务（如遥感、医学影像）。
- SRDL 的双重学习循环类似于半监督学习中的自训练，但通过任务互补性实现了更自然的质量控制。

## 评分

- **新颖性**: ⭐⭐⭐⭐ ISC 和 SRDL 各自并非全新概念，但在 GUI 场景中的结合方式非常巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 GUI Grounding 和下游 Agent 任务，消融清晰
- **写作质量**: ⭐⭐⭐⭐ 论文结构清晰，可视化效果好
- **价值**: ⭐⭐⭐⭐⭐ 对 GUI 视觉 Agent 领域有显著推动，数据效率提升意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GUI-SAGE: Enhancing GUI Automation with Self-Explanatory Learning](../../CVPR2026/multimodal_vlm/gui-sage_enhancing_gui_automation_with_self-explanatory_learning.md)
- [\[ICCV 2025\] Acknowledging Focus Ambiguity in Visual Questions](acknowledging_focus_ambiguity_in_visual_questions.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[ICCV 2025\] VQ-FocusAmbiguity: Acknowledging Focus Ambiguity in Visual Questions](vq_focusambiguity_acknowledging_focus_ambiguity_visual_questions.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)

</div>

<!-- RELATED:END -->
