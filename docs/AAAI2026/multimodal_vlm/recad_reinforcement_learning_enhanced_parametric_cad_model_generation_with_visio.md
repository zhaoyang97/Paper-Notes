---
title: >-
  [论文解读] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models
description: >-
  [多模态VLM] 提出 ReCAD 框架，通过将 CAD 脚本重写为参数化代码进行 SFT，再利用 GRPO 强化学习与分层基元课程学习策略，使 VLM 能从文本或图像输入生成高精度、可编辑的参数化 CAD 模型，在分布内和分布外设置上均大幅超越现有方法。 - CAD 建模的现实需求：工业 CAD 建模耗时且需要高精度…
tags:
  - "多模态VLM"
---

# ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models

| 信息 | 内容 |
|------|------|
| **会议** | AAAI 2026 |
| **arXiv** | [2512.06328](https://arxiv.org/abs/2512.06328) |
| **代码** | 未开源 |
| **领域** | multimodal_vlm |
| **关键词** | CAD生成, 强化学习, 视觉语言模型, 参数化代码, 课程学习 |

## 一句话总结

提出 ReCAD 框架，通过将 CAD 脚本重写为参数化代码进行 SFT，再利用 GRPO 强化学习与分层基元课程学习策略，使 VLM 能从文本或图像输入生成高精度、可编辑的参数化 CAD 模型，在分布内和分布外设置上均大幅超越现有方法。

## 背景与动机

- **CAD 建模的现实需求**：工业 CAD 建模耗时且需要高精度，生成式 CAD 建模受到学术界和工业界的广泛关注
- **现有方法的局限性**：
    - 传统编码器-解码器方法（DeepCAD、Text2CAD 等）泛化能力有限，难以生成精确 CAD 模型
    - 基于 PLM 的方法（CAD-LLaMA、CAD-Coder）主要依赖 SFT 注入知识，PLM 仅充当"语义解释器"，未充分利用其生成先验
    - 直接生成低级参数序列（如坐标）缺乏对设计意图的理解，参数调整容易导致无效几何体（如非闭合回路）
- **核心洞察**：参数化 CAD 建模本质上需要精确的数学推理、符号操作和逻辑约束满足，这恰好是 RLVR（强化学习与可验证奖励）在数学和代码生成领域已证明有效的能力方向
- **关键创新**：即使只提供简单的函数接口（仅暴露曲线坐标值），通过 RL 训练可涌现出复杂 CAD 操作（如环形阵列、镜像），这在以前的方法中未曾出现

## 方法详解

### 1. 问题定义与 CAD 层次结构

基于 Sketch-Extrude（SE）范式定义五级层次基元：

$$\mathcal{P} = \{\text{L (Loop)}, \text{F (Face)}, \text{S (Sketch)}, \text{SE (Sketch-Extrude)}, \text{MSE (Multi-SE)}\}$$

从曲线到循环、面、草图、拉伸，逐级封装。设计了一组轻量函数接口，底层仅提供坐标值访问，高层组件通过结构化封装组织，最终形成完整 CAD 模型。

### 2. 参数化代码生成与 SFT 阶段

**硬编码到参数化代码的转换**：直接将 CAD 序列转换为代码会产生不灵活、易过拟合的"硬编码"表示。ReCAD 利用 VLM（GPT-4o）将硬编码代码 $C = f(P)$ 重写为参数化代码：

$$\{\hat{C}^i\}_{i=1}^N = \text{VLM}(I, C)$$

然后用 DINOv2 编码器计算渲染图像的余弦相似度进行质量过滤：

$$\mathcal{C} = \left\{\hat{C}^i \mid \cos(E(\hat{I}_i), E(I)) > \tau_s \right\}$$

其中 $\tau_s = 0.95$ 为相似度阈值。

**文本描述生成**：利用参数化代码天然包含的语义信息（尺度、数量），通过 VLM 生成抽象描述 $T^A$ 和精确描述 $T^D$，避免了传统方法中冗长或不精确的标注。

**SFT 训练**：使用标准因果语言建模目标，在 text-to-CAD 和 image-to-CAD 两个任务上微调 Qwen2.5-VL-7B-Instruct，同时混入 UltraChat（23%）和 OpenCodeReasoning（5%）数据保持通用能力，得到 ReCAD-Base。

### 3. 强化学习阶段（Learn Under Guidance）

采用 GRPO（Group Relative Policy Optimization）进行强化学习，核心创新在于**引导式学习策略**。

**难题识别**：RL 训练前，对每个问题 $q_i$ 采样 $N$ 个解并计算最大奖励。若 $\max\{R(q_i)\} < \tau_h$（$\tau_h = 0.8$），则标记为困难问题。

**引导式目标函数**：对困难问题，将参数化代码 $\mathcal{C}$ 作为离策略引导，在 rollout 时提供互补知识，利用模型的 in-context learning 能力增强推理：

$$\hat{\mathcal{J}}(\theta; \mathcal{C}) = \frac{1}{N-|\mathcal{C}|}\sum_{i=1}^{N-|\mathcal{C}|}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}\text{CLIP}(r_{i,t}, A_i, \epsilon) + \frac{1}{|\mathcal{C}|}\sum_{j=1}^{|\mathcal{C}|}\frac{1}{|\tau_j|}\sum_{t=1}^{|\tau_j|}\text{CLIP}(\hat{r}_{j,t}, A_j, \epsilon) - \beta\mathbb{D}_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}]$$

最终训练目标根据难度自适应切换：

$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}\left[\mathbf{1}_{\text{hard}}(q_i) \cdot \hat{\mathcal{J}}(\theta; \mathcal{C}_i) + (1 - \mathbf{1}_{\text{hard}}(q_i)) \cdot \mathcal{J}(\theta)\right]$$

### 4. 分层基元学习（HPL）

设计课程学习策略，按 CAD 层次结构从简单到复杂逐步学习：

- **学习顺序**：L → F → S → SE → MSE，每一阶段的复杂度递增
- **每层内部排序**：按涉及曲线数量排列，从少到多
- 模拟人类学习过程，先掌握基础技能再应对复合设计

### 5. 奖励函数设计

综合几何精度和语义保真度的统一奖励：

$$R(y_\pi, \Omega) = \lambda_1 \cdot \min\{\text{IOU}_{best}(\hat{\Omega}, \Omega),\ \phi(\text{sim}(\hat{I}, I), \tau)\} + \lambda_2 \cdot R_f(y_\pi)$$

- $\text{IOU}_{best}$：最优对齐下的 IoU（几何一致性）
- $\text{sim}(\hat{I}, I)$：DINOv2 特征余弦相似度（视觉保真度）
- $\phi(s, \tau)$：阈值线性缩放函数，$\tau = 0.55$
- $R_f$：格式奖励（是否包含有效 `<think>` 块）
- $\lambda_1 = 0.1$，$\lambda_2 = 0.9$

对 image-to-CAD 任务，由于输入无绝对尺度信息，用惯性矩阵归一化几何体后再计算奖励。

## 实验结果

### Text-to-CAD 生成

| 方法 | P-F1↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-------|-----------|---------|-----|
| GPT-4o | 50.55 | 107.55 | 165.67 | 15.14 |
| CAD-LLaMA | 60.02 | 41.77 | 98.12 | **0.39** |
| **ReCAD-VL** | **61.48** | **34.31** | **72.47** | 0.81 |

OOD (Fusion 360) 设置：

| 方法 | P-F1↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-------|-----------|---------|-----|
| CAD-LLaMA | 50.47 | 60.36 | 142.48 | 1.29 |
| **ReCAD-VL** | **55.25** | **34.67** | **84.89** | **0.93** |

ReCAD-VL 在分布内外均大幅领先，OOD 设置下 Mean CD 降低 40%，表明强泛化能力。

### Image-to-CAD 生成

| 方法 | IOU_best↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-----------|-----------|---------|-----|
| CAD-Coder | 61.23 | 8.09 | 73.47 | **1.05** |
| **ReCAD-VL** | **63.14** | **7.45** | **29.61** | 1.12 |

OOD (Fusion 360) 设置：

| 方法 | IOU_best↑ | Median CD↓ | Mean CD↓ | IR↓ |
|------|-----------|-----------|---------|-----|
| CAD-Coder | 45.32 | 84.02 | 272.06 | 2.23 |
| **ReCAD-VL** | **54.93** | **17.01** | **80.23** | **0.91** |

Mean CD 从 73.47 降至 29.61（分布内），从 272.06 降至 80.23（OOD），提升幅度极为显著。

### 消融实验

| 配置 | P-F1 | Median CD↓ | Mean CD↓ | IR↓ |
|------|------|-----------|---------|-----|
| SFT only | 53.53 | 84.78 | 155.67 | 3.21 |
| RL only | 55.61 | 107.32 | 179.50 | 4.77 |
| w/o HPL | 59.63 | 44.64 | 90.83 | 2.42 |
| w/o Guidance | 60.03 | 42.85 | 87.34 | 0.93 |
| **完整模型** | **61.48** | **34.31** | **72.47** | **0.81** |

- SFT 和 RL 缺一不可，单独使用效果均不理想
- HPL 去除后重建误差和失败率上升，说明分层课程学习有效
- 引导策略提供互补知识，进一步提升生成质量

## 关键发现

1. **能力涌现**：仅通过简单坐标接口，RL 训练可涌现出环形阵列（circular pattern）和镜像（mirror）等复杂 CAD 操作，这些操作在训练数据中并未显式出现
2. **零样本泛化**：虽然仅在 CAD 生成任务上训练，ReCAD-VL 在 CAD 理解、编辑、调试等多个相关任务上展现出优秀的零样本能力
3. **OOD 鲁棒性**：在 Fusion 360 OOD 数据上，ReCAD 性能下降极小，而之前的方法（如 CAD-LLaMA）性能显著下降，表明参数化代码 + RL 有效缓解过拟合
4. **SFT + RL 互补**：SFT 引入外部知识，RL 通过自我探索强化泛化；两者结合效果远超单独使用

## 亮点

- **参数化代码表示**：将硬编码 CAD 代码重写为参数化代码，既保留语义信息又增强灵活性，是连接 PLM 代码能力与 CAD 领域的桥梁
- **引导式 RL**：针对困难问题将参数化代码作为离策略引导信号，巧妙利用 LLM 的 in-context learning 能力突破 on-policy RL 的局限
- **分层基元课程学习**：将 CAD 结构天然的层次性与课程学习结合，从曲线到完整模型逐步构建能力
- **统一奖励函数**：同时考虑几何精度（IoU）和视觉保真度（DINOv2 特征相似度），确保生成质量

## 局限性

- **依赖 GPT-4o**：参数化代码重写和文本描述生成均依赖 GPT-4o，数据准备成本较高
- **限于 Sketch-Extrude 范式**：仅支持基于草图-拉伸的 CAD 建模，无法处理更复杂的建模操作（如旋转、扫掠、放样等）
- **尺度信息缺失**：image-to-CAD 任务中无法恢复绝对尺度，需要几何归一化
- **失败模式**：存在与输入描述不匹配、空间参数化错误两类失败情况
- **计算资源需求**：需要 8×A800 80GB GPU 训练，资源门槛较高
- **仅 7B 模型**：未探索更大规模模型是否能带来进一步提升

## 相关工作

- **CAD 序列建模**：DeepCAD、CAD-Translator、Text2CAD 等使用编码器-解码器或 Transformer 架构
- **PLM for CAD**：CAD-LLaMA、CAD-Coder 通过 SFT 适配 PLM，但受限于注入知识的范围
- **约束草图生成**：Vitruvion、SketchDNN 聚焦 2D 草图，依赖外部约束求解器
- **RLVR**：DeepSeek-R1 的 GRPO 在数学/代码推理中验证有效，LUFFY 引入离策略推理信号——ReCAD 将此思路引入 CAD 领域

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体评价 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](../../ICLR2026/multimodal_vlm/why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)

</div>

<!-- RELATED:END -->
