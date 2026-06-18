---
title: >-
  [论文解读] GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation
description: >-
  [NeurIPS 2025][多模态VLM][GUI Navigation] 提出 GUI-Rise 框架，通过结构化推理（进度估计 + 决策推理）、动作预测和历史摘要三个子任务的联合设计，结合 GRPO 强化学习与历史摘要奖励，显著提升 GUI 导航智能体在跨域场景下的泛化能力。 - GUI 导航是重要方向：多模态大语言模…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "GUI Navigation"
  - "Structured Reasoning"
  - "History Summarization"
  - "GRPO"
  - "Chain-of-Thought"
---

# GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation

**会议**: NeurIPS 2025  
**arXiv**: [2510.27210](https://arxiv.org/abs/2510.27210)  
**代码**: [leon022.github.io/GUI-Rise](https://leon022.github.io/GUI-Rise)  
**领域**: 机器人  
**关键词**: GUI Navigation, Structured Reasoning, History Summarization, GRPO, Chain-of-Thought  

## 一句话总结

提出 GUI-Rise 框架，通过结构化推理（进度估计 + 决策推理）、动作预测和历史摘要三个子任务的联合设计，结合 GRPO 强化学习与历史摘要奖励，显著提升 GUI 导航智能体在跨域场景下的泛化能力。

## 研究背景与动机

- **GUI 导航是重要方向**：多模态大语言模型（MLLM）驱动的 GUI 导航智能体可以将自然语言指令转化为界面操作，但在多步交互中保持一致性仍是挑战
- **现有方法泛化差**：基于 GPT-4 的 prompt engineering 方案受限于冻结的模型能力；基于 SFT 的开源方案容易过拟合静态指令-动作对，跨域表现不佳
- **历史表示方式有缺陷**：现有系统要么仅编码动作序列（丢失视觉状态信息），要么使用完整截图序列（计算开销大、上下文窗口受限严重），均无法像人类一样高效整合历史信息
- **长程序推理不足**：有效的 GUI 决策需要依赖已完成的动作和界面演变，但当前智能体在长期连贯推理方面仍有明显短板
- **SFT 难以学好结构化推理**：消融实验表明仅用 SFT 训练结构化 CoT 反而导致性能下降，需要强化学习来真正提升推理质量
- **历史摘要缺乏监督**：无监督的历史摘要可能质量低下，误导策略学习，需要专门设计奖励函数来保证摘要质量

## 方法详解

### 整体框架

GUI-Rise 在每一步交互中执行三个子任务的循环：(1) 结构化推理——分析当前截图和历史摘要，生成包含进度估计和决策推理的 CoT；(2) 动作预测——基于推理结果输出可执行的 GUI 动作（类型 + 值 + 坐标）；(3) 历史摘要——将新信息压缩为简洁文本摘要，供下一步使用。三个子任务的输出通过 XML 标签序列化，由 MLLM（Qwen-VL 系列）自回归生成。

### 关键设计 1：结构化推理子任务（Structured Reasoning）

- **功能**：将 CoT 推理分解为 Progress Estimation（进度估计）和 Decision Reasoning（决策推理）两个显式阶段
- **核心思路**：智能体先根据当前截图 $\mathbf{o}_t$ 和历史 $\mathbf{h}_{t-1}$ 评估任务完成进度，再结合用户指令 $\mathbf{u}$ 和先前决策确定下一步动作方向
- **设计动机**：模仿人类导航界面时的认知策略——先判断"做到哪了"，再决定"下一步做什么"，实现步步推理的连贯性和可解释性

### 关键设计 2：历史摘要子任务（History Summary）

- **功能**：每一步将当前观测 $\mathbf{o}_t$、上一步摘要 $\mathbf{h}_{t-1}$ 和指令 $\mathbf{u}$ 压缩为简洁的文本记忆 $\mathbf{h}_t$
- **核心思路**：用语义摘要替代原始截图序列或动作列表，以固定长度的文本持续追踪任务进度，不受窗口大小限制
- **设计动机**：原始截图计算开销大且强制截断上下文；纯动作序列丢失视觉状态。语义摘要兼具层次抽象和任务场景锚定，有效支持多步推理

### 关键设计 3：两阶段训练策略（Cold-start + RL）

- **功能**：第一阶段用 GPT-4o-mini 生成伪标签进行 SFT 冷启动；第二阶段在模拟 GUI 环境中用 GRPO 强化学习精调
- **核心思路**：冷启动建立基础推理和摘要能力，避免 RL 初期奖励过于稀疏；RL 阶段通过环境交互发展自适应推理策略
- **设计动机**：直接用 SFT 训练结构化推理效果差（消融实验 Row 3 显示性能大幅下降），必须由 RL 来激发小模型的推理探索能力

### 关键设计 4：三重奖励函数设计

- **功能**：设计格式奖励 $\mathcal{R}^f$、动作奖励 $\mathcal{R}^a$ 和历史摘要奖励 $\mathcal{R}^h$ 三个互补的奖励函数
- **核心思路**：总奖励 $r_{t,i} = r^f_{t,i} + \lambda^a \cdot r^a_{t,i} + \lambda^h \cdot r^h_{t,i}$。其中历史摘要奖励通过额外 $k$ 次 rollout 检验摘要对未来动作的支持质量，将摘要价值与后续动作正确性直接挂钩
- **设计动机**：无监督摘要可能质量低下反而误导策略（消融 Row 5 vs Row 6）；通过"摘要→未来动作正确率"的反馈闭环，驱动模型主动学习提取对任务有用的历史关键线索

## 损失函数与训练策略

- **冷启动阶段**：标准 token 级交叉熵损失 $\mathcal{L}_{\text{CE}}$，在伪标签（CoT + 动作 + 摘要）上进行 SFT
- **RL 阶段**：采用 GRPO 算法，通过组级归一化计算优势 $A_{t,i}$，三重奖励加权求和后用于策略梯度优化，无需价值网络

## 实验

### 表 1：Mind2Web 跨域评估（Step SR）

| 方法 | Backbone | Cross-Task | Cross-Website | Cross-Domain |
|------|----------|-----------|--------------|-------------|
| ShowUI-2B | Qwen2-VL-2B | 37.2 | 35.1 | 35.2 |
| **GUI-Rise** | Qwen2-VL-2B | **38.8** | **35.4** | **39.7** |
| Qwen2.5-VL-3B | Qwen2.5-VL-3B | 48.3 | 43.5 | 44.1 |
| **GUI-Rise** | Qwen2.5-VL-3B | 46.2 | **44.7** | **47.6** |
| ShowUI-2B (ZS) | Qwen2-VL-2B | 18.6 | 16.8 | 21.4 |
| **GUI-Rise (ZS)** | Qwen2-VL-2B | **24.2** | **21.1** | **29.7** |

**关键发现**：零样本设置下 Cross-Domain 上 GUI-Rise 比 ShowUI 提升 38.7%（29.7 vs 21.4）。

### 表 2：AITW 移动端评估（Overall Accuracy）

| 方法 | Backbone | In-Domain | Zero-Shot |
|------|----------|-----------|-----------|
| ShowUI-2B | Qwen2-VL-2B | 70.0 | 35.9 |
| **GUI-Rise** | Qwen2-VL-2B | **71.1** | **54.1** |
| Qwen2.5-VL-3B | Qwen2.5-VL-3B | 72.5 | 38.9 |
| **GUI-Rise** | Qwen2.5-VL-3B | **73.7** | **56.0** |

**关键发现**：零样本设置下 GUI-Rise 相对 ShowUI 提升 50.7%（54.1 vs 35.9），在复杂 WebShop 任务上提升 +15.5 分。

### 消融实验（AITW Overall）

| 配置 | TST | SCoT | HS | HSR | Overall |
|------|-----|------|----|-----|---------|
| Baseline | × | × | × | × | 67.2 |
| + RL only | ✓ | × | × | × | 66.0 |
| + SCoT SFT only | × | ✓ | × | × | 42.6 |
| + RL + SCoT | ✓ | ✓ | × | × | 69.8 |
| + History | ✓ | ✓ | ✓ | × | 70.7 |
| + History Reward | ✓ | ✓ | ✓ | ✓ | **71.1** |

## 亮点

- **三子任务联合框架**设计精巧，推理（进度估计 + 决策分析）→ 动作 → 摘要的循环贴近人类认知范式
- **历史摘要奖励**创新性强，通过 rollout 方式将摘要质量与未来动作正确性直接绑定，形成自改进闭环
- **跨域泛化能力突出**，零样本场景下相对 ShowUI 提升 38.7%（Mind2Web）和 50.7%（AITW），验证了结构化推理对泛化的重要性
- 消融实验清晰展示了每个组件的贡献，特别揭示了"SFT 无法学好结构化推理、必须依赖 RL"的重要发现

## 局限性

- 模型仅在离线数据上训练，无法从在线交互中学习和适应新场景
- 伪标签依赖 GPT-4o-mini，冷启动阶段的标签质量受上限约束
- 仅基于 2B/3B 规模模型验证，更大规模模型上的扩展性尚不清楚
- 历史摘要为纯文本形式，可能丢失细粒度视觉信息（如小按钮状态）

## 相关工作

- **GUI 智能体**：CogAgent、SeeClick、ShowUI、UI-TARs 等通过 SFT 或大规模推理数据提升 GUI 导航，但跨域泛化有限；GUI-Rise 引入结构化 CoT + RL 显著改善
- **GUI 记忆机制**：早期仅用动作序列（SeeClick），后来用截图窗口（ShowUI、UI-Hawk），但信息损失或开销大；GUI-Rise 的语义摘要方案更高效
- **LLM 强化学习**：GRPO 已在代码生成和数学推理中验证，UI-R1 将其扩展到单步 GUI 任务；GUI-Rise 首次将 GRPO + 历史摘要奖励应用于多步 GUI 导航

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三子任务框架和历史摘要奖励设计有较高新颖性
- 实验充分度: ⭐⭐⭐⭐ — 涵盖 Mind2Web/AITW/MiniWob/AndroidWorld/OSWorld，消融完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式化定义严谨
- 价值: ⭐⭐⭐⭐ — 跨域泛化提升显著，对 GUI Agent 领域有实质贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HiconAgent: History Context-aware Policy Optimization for GUI Agents](../../CVPR2026/multimodal_vlm/hiconagent_history_context-aware_policy_optimization_for_gui_agents.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](../../ICML2026/multimodal_vlm/learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] GUI-SAGE: Enhancing GUI Automation with Self-Explanatory Learning](../../CVPR2026/multimodal_vlm/gui-sage_enhancing_gui_automation_with_self-explanatory_learning.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](../../CVPR2026/multimodal_vlm/drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)
- [\[ACL 2025\] Aria-UI: Visual Grounding for GUI Instructions](../../ACL2025/multimodal_vlm/aria-ui_visual_grounding_for_gui_instructions.md)

</div>

<!-- RELATED:END -->
