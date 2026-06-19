---
title: >-
  [论文解读] A Unified Agentic Framework for Evaluating Conditional Image Generation
description: >-
  [ACL 2025][图像生成][LMM Agent] 提出 CIGEval，一个基于大型多模态模型（LMM）的统一 Agent 评估框架，通过工具集成（Grounding、Highlight、Difference、Scene Graph）和分而治之的评估策略，在 7 种条件图像生成任务上达到与人类标注者相当的相关性（0.4625 vs 人类间 0.47），且仅用 2.3K 训练数据微调 7B 模型即超越 GPT-4o 版 SOTA。
tags:
  - "ACL 2025"
  - "图像生成"
  - "LMM Agent"
  - "Evaluation Framework"
  - "Tool Augmentation"
  - "Agent Tuning"
---

# A Unified Agentic Framework for Evaluating Conditional Image Generation

**会议**: ACL 2025  
**arXiv**: [2504.07046](https://arxiv.org/abs/2504.07046)  
**代码**: [https://github.com/HITsz-TMG/Agentic-CIGEval](https://github.com/HITsz-TMG/Agentic-CIGEval)  
**领域**: Image Generation / Evaluation  
**关键词**: Conditional Image Generation, LMM Agent, Evaluation Framework, Tool Augmentation, Agent Tuning

## 一句话总结

提出 CIGEval，一个基于大型多模态模型（LMM）的统一 Agent 评估框架，通过工具集成（Grounding、Highlight、Difference、Scene Graph）和分而治之的评估策略，在 7 种条件图像生成任务上达到与人类标注者相当的相关性（0.4625 vs 人类间 0.47），且仅用 2.3K 训练数据微调 7B 模型即超越 GPT-4o 版 SOTA。

## 研究背景与动机

条件图像生成发展迅速，涵盖文本引导生成/编辑、主体驱动生成/编辑、多概念组合、控制信号引导生成等 7 类任务。但现有评估指标面临三大问题：

**任务特定性**：LPIPS 只衡量感知相似度，CLIP-Score 只衡量文本对齐，无法跨任务通用

**可解释性不足**：仅给出单一分数，缺乏推理过程和多维度细粒度评估

**与人类不对齐**：传统指标（DINO, CLIP）与人类评分差距大；即使 GPT-4o 版 VIEScore 也难以捕捉细微图像差异

作者的核心观察：**GPT-4o 自身的感知能力不足以捕捉高度相似图像间的细微差别**，需要通过外部工具增强。例如图 1 中的 subject-driven image editing 案例，GPT-4o 直接评估给了高分，但通过 Grounding + Highlight 工具聚焦到眼镜区域后，发现形状和设计的差异。

## 方法详解

### 整体框架

CIGEval 将图像评估建模为 Agent 任务：

$$f_{\text{eval}}(I, O, C^*) = (\text{rationale}, \text{score})$$

其中 $I$ 为评估指令，$O$ 为生成图像，$C^*$ 为条件集合（文本、主体图像、控制信号等）。

采用**分而治之**策略：将每个评估任务分解为多个细粒度子问题，对每个子问题选择合适工具，基于工具输出评分，最终取子分数的最小值作为总分。

### 关键设计

**1. 多功能工具箱（Toolbox）**

| 工具 | 输入 | 输出 | 用途 |
|------|------|------|------|
| **Grounding** | 图像 + 目标实体 | 坐标 [x1,y1,x2,y2] | 定位图像中特定对象区域 |
| **Highlight** | 图像 + 区域坐标 | 编辑后图像 | 高亮指定区域（暗化其余区域至 1/4 亮度） |
| **Difference** | 图像1 + 图像2 | 差异区域坐标 | 像素级差异检测 |
| **Scene Graph** | 图像 | 结构化描述 | LMM 分析的对象、属性、关系描述 |

- Grounding 基于 GroundingDINO 实现
- Scene Graph 基于 CCoT prompting（可用 GPT-4o 或开源模型）
- Highlight 常在 Grounding/Difference 之后使用，聚焦关注区域
- Difference 通过像素比较找到两张图的差异位置

**2. 细粒度评估框架**

将每个任务分解为以下子问题的子集：
1. 生成图像是否遵循文本 prompt？
2. 图像编辑是否遵循指令？
3. 是否做了最小编辑且未改变背景？
4. 生成图像中的对象是否与给定主体一致？
5. 图像是否遵循控制信号（如 Canny 边缘、OpenPose）？

每个子问题采用 **ReAct 格式**（Observation → Thought → Action），CIGEval 自主选择工具、分析输出、给出 0-10 分。

**3. 总分聚合**

$$O = \min(\alpha_1, ..., \alpha_i)$$

使用 min 操作而非平均，强调每个条件都必须被满足，任一维度的失败都不可接受。

### 损失函数 / 训练策略

**Agent Tuning**：用 GPT-4o 执行评估流程生成评估轨迹数据，过滤预测分数与人类评分差距 >0.3 的样本，最终获得 **2,274 条**高质量轨迹。

微调策略（在 Qwen2-VL-7B / Qwen2.5-VL-7B 上）：
- 每条轨迹表示为 $\langle o_0, t_1, a_1, ..., o_{n-1}, t_n, a_n, o_n \rangle$
- 仅在 thought $t_i$ 和 action $a_i$ 上计算 cross-entropy loss，前序轨迹 $c_i$ 被 mask
- 学习率 1e-5，batch size 128，序列长度 32768
- AdamW + cosine scheduler + 3% warmup

## 实验关键数据

### 主实验

**ImagenHub 基准上的 Spearman 相关性**（7 个任务）：

| 方法 | 平均相关性 |
|------|-----------|
| Human-to-Human | 0.4700 |
| VIEScore (GPT-4o) | 0.4459 |
| **CIGEval (GPT-4o)** | **0.4625** |
| CLIPScore / LPIPS / DINO | 仅适用于部分任务 |

CIGEval (GPT-4o) 在**所有 7 个任务**上均超越 VIEScore，尤其在多条件任务上提升明显：
- Multi-concept IC：0.4516 → **0.4931**
- Control-guided IG：0.4972 → **0.5402**

**Agent Tuning 后的开源模型**：

| 模型 | 微调前 Avg | 微调后 Avg | 提升 |
|------|----------|----------|------|
| Qwen2-VL-7B | 0.2840 | **0.4997** | +76% |
| Qwen2.5-VL-7B | 0.3455 | **0.4631** | +34% |

微调后的 7B 模型均超越了 VIEScore (GPT-4o) 的 0.4459！

### 消融实验

**工具消融**（CIGEval GPT-4o 版本）：

| 配置 | 平均相关性 |
|------|-----------|
| 完整 CIGEval | **0.7262** |
| 去掉 Grounding | 0.6376 (-8.9%) |
| 去掉 Difference | 0.7020 (-2.4%) |
| 去掉 Scene Graph | 0.6471 (-7.9%) |
| Scene Graph 用 Qwen2.5-VL-7B | 0.7120 (-1.4%) |
| Scene Graph 用 Qwen2.5-VL-70B | 0.7311 (+0.5%) |

每个工具都有贡献，Grounding 和 Scene Graph 影响最大。Scene Graph 替换为开源模型后仅微降，框架具有鲁棒性。

### 关键发现

1. 工具增强是关键：仅靠 LMM 的感知能力不足以区分高度相似图像的细微差异
2. 少量高质量轨迹数据（2.3K）即可大幅提升开源小模型的评估能力
3. 多条件任务（subject-driven editing、multi-concept composition、control-guided generation）是评估的难点，也是 CIGEval 优势最显著的任务
4. GPT-4o 生成的图像在需要多输入图像和控制信号的任务上仍有明显缺陷

## 亮点与洞察

1. **Agent 范式用于评估**：将评估任务建模为 Agent 的 tool-use 过程，使得评估过程可解释、可扩展
2. **工具选择的自主性**：Agent 根据任务类型和子问题自主决定使用哪个工具，而非固定流程
3. **数据效率极高**：仅 2.3K 训练轨迹就让 7B 模型超越 GPT-4o baseline，说明高质量轨迹数据的价值
4. **统一框架**：一个框架覆盖 7 种不同的条件图像生成任务，避免了为每种任务设计专用指标

## 局限与展望

1. 目前只关注语义一致性（Semantic Consistency），未涉及感知质量（Perceptual Quality）
2. 工具箱可进一步扩展（如已在 case study 中添加 OCR 工具）
3. 总分使用 min 聚合较为保守，可能低估在某一维度略差但整体优秀的图像
4. 微调数据的质量依赖 GPT-4o 的评估结果和 0.3 阈值的过滤 → 可能引入 GPT-4o 自身偏见
5. GroundingDINO 在某些细粒度对象上的定位准确性可能成为瓶颈

## 相关工作与启发

- **VIEScore**：GPT-4o 直接 prompting 的指标，CIGEval 的主要对比基准
- **ImagenHub**：标准化的条件图像生成评估基准 + 人类评分数据
- **GroundingDINO**：开放集目标检测模型，CIGEval Grounding 工具的底层实现
- **ReAct**：Observation-Thought-Action 的 Agent 推理框架 → CIGEval 采用此模式
- **CCoT (Mitra et al.)**：Chain-of-Composition prompting → Scene Graph 工具的实现基础

## 评分

- **创新性**: ★★★★☆ — Agent + 工具增强的评估范式在条件图像生成领域较为新颖
- **实用性**: ★★★★★ — 统一框架、开源模型可用、覆盖 7 种主流任务，工程完成度高
- **实验充分度**: ★★★★★ — 7 任务评测 + 详细消融 + GPT-4o 图像生成 case study + agent tuning
- **写作质量**: ★★★★☆ — 结构清晰，案例丰富，图表直观

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](../../ICCV2025/image_generation/a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](../../CVPR2026/image_generation/agentic_retoucher_for_texttoimage_generation.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](../../NeurIPS2025/image_generation/toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](../../ICCV2025/image_generation/unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
