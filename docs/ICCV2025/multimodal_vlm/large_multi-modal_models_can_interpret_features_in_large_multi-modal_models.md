---
title: >-
  [论文解读] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models
description: >-
  [多模态VLM] 本文提出了首个面向多模态大模型（LMM）的自动化特征解释框架，使用稀疏自编码器（SAE）分解 LMM 的内部表征为单语义特征，并利用更大的 LMM 对这些特征进行自动解释，还展示了特征引导可修正模型幻觉。 LMM 在各类视觉语言任务中取得了显著成功，但其"黑箱"特性导致不可预测的行为…
tags:
  - "多模态VLM"
---

# Large Multi-modal Models Can Interpret Features in Large Multi-modal Models

## 基本信息

- **会议**: ICCV 2025
- **arXiv**: 2411.14982
- **代码**: [GitHub](https://github.com/EvolvingLMMs-Lab/multimodal-sae)
- **领域**: 多模态视觉语言模型 (Multimodal VLM)
- **关键词**: 可解释性, 稀疏自编码器, 特征解释, 模型引导, 幻觉缓解

## 一句话总结

本文提出了首个面向多模态大模型（LMM）的自动化特征解释框架，使用稀疏自编码器（SAE）分解 LMM 的内部表征为单语义特征，并利用更大的 LMM 对这些特征进行自动解释，还展示了特征引导可修正模型幻觉。

## 研究背景与动机

LMM 在各类视觉语言任务中取得了显著成功，但其"黑箱"特性导致不可预测的行为，如：
- **幻觉问题**：生成图像中不存在的物体和关系
- **越狱攻击脆弱性**：对抗性输入可绕过安全限制

理解和控制 LMM 的神经表征至关重要，但面临两大挑战：

**多语义性**（Polysemanticity）：单个神经元可能编码多个语义（如 Inception v1 的一个神经元同时响应猫脸和车头），LMM 更高的维度加剧了这一问题

**开放语义**：传统视觉模型仅含数百个单语义概念（颜色、物体等），可通过人工标注分析。LMM 包含数十万个开放域概念，人工分析不可行

现有工作（如 GPT-4 解释 GPT-2 的神经元）尚未扩展到多模态领域。

## 方法详解

### 整体框架

三步流水线：解耦（SAE）→ 解释（自动管线）→ 引导（特征值设定）

### 稀疏自编码器 (SAE) 用于表征解耦

采用 OpenAI 的 TopK SAE 架构，嵌入到 LLaVA-NeXT-8B 的第 25 层：

$$\mathbf{z} = \text{TopK}(\text{ReLU}(\mathbf{W}_1(\mathbf{x} - \mathbf{b}_1) + \mathbf{b}_2))$$
$$\hat{\mathbf{x}} = \mathbf{W}_2 \mathbf{z} + \mathbf{b}_3$$

其中 $\mathbf{x} \in \mathbb{R}^{T \times d_l}$ 为 LLaVA 隐藏表征，$\mathbf{z} \in \mathbb{R}^{T \times d_s}$ 为稀疏表征。使用 $d_s = 2^{17}$ 个特征，$k = 256$。

SAE 产生单语义特征的核心原因：$\mathbf{W}_2$ 充当过完备字典，$\mathbf{z}$ 作为稀疏系数。由于稀疏性约束，字典向量趋向近正交（互不相干），每个 $\mathbf{z}$ 的坐标倾向于表示单一语义。

### 零样本特征解释管线

**Step 1: 识别 Top 激活图像和 patch**

缓存约 46,684 张图像的 SAE 激活值。对每个特征，在第一维度上选取 Top-5 激活图像。在 Top-5 图像中定位最激活的 patch。

**Step 2: 自动特征解释**

对 Top 激活图像应用掩码（激活 patch 透明，其余区域变黑），输入 LLaVA-OV-72B 检测共同模式并生成解释。无法识别共同模式时返回"无法解释"。

**Step 3: 参考分数计算**

用小 LLM 精炼描述 → GroundingDINO-SAM 生成分割 mask → 与 SAE 激活区域计算 IoU 作为特征相关性量化。

### 特征引导 (Steering)

设定特定 SAE 特征值来影响模型输出：

$$\mathbf{z}[\mathcal{C}, j] = k$$

其中 $\mathcal{C}$ 为指定 token 集合，$k$ 为引导值，$j$ 为 SAE 特征索引。引导后的 $\hat{\mathbf{x}}$ 替代原始 $\mathbf{x}$ 送入后续层。

### 模型行为因果定位

对输出 token 进行归因分析。设当前输出 token 为 $v_c$, 基线 token 为 $v_b$，通过逐特征消融（设为 0）并使用线性近似计算每个特征的影响：

$$I(i, j, v_c, v_b) \approx \left(\frac{\partial d(\mathbf{u})}{\partial \mathbf{z}}\right)^T (\hat{\mathbf{z}} - \mathbf{z})$$

## 实验关键数据

### 主实验：特征解释质量评估

| 概念 | IoU ↑ (随机) | IoU ↑ (V-Interp) | CLIP Score ↑ (随机) | CLIP Score ↑ (V-Interp) |
|---|---|---|---|---|
| scene | 0.007 | 0.20 | 18.1 | 24.4 |
| object | 0.005 | 0.19 | 18.2 | 24.0 |
| part | 0.007 | 0.21 | 18.1 | 23.5 |
| material | 0.01 | **0.39** | 18.1 | 24.1 |
| texture | 0.007 | 0.21 | 18.4 | 20.9 |
| colour | 0.005 | 0.10 | 19.6 | 20.3 |
| **Total** | 0.005 | **0.20** | 18.2 | **23.6** |

V-Interp 的 IoU 比随机基线高约 40 倍，CLIP Score 高 5.4 分，证明解释与激活区域高度一致。

### 消融实验：跨层分析

| Layer | IoU ↑ | CLIP Score ↑ |
|---|---|---|
| LLaVA (8th) | 0.30 | 22.82 |
| LLaVA (25th) | 0.31 | 24.92 |
| LLaVA (32th) | 0.40 | 26.55 |
| Random | 0.005 | 18.2 |

层深度增加时 IoU 和 CLIP Score 同步提升，证明更深层包含更高级、更可解释的视觉语义。

### 一致性评估

| 概念 | GPT-4o 一致性 | 人类一致性 |
|---|---|---|
| scene | 0.93 | 0.70 |
| object | 0.84 | 0.85 |
| material | 1.00 | 0.95 |
| **Total** | **0.89** | **0.75** |

GPT-4o 与人类评估均确认高一致性（总体 89% / 75%）。

### 幻觉修正案例

在 HallusionBench 示例中，LLaVA 对"图中是否显示 Bolivia"错误回答"Yes"。

通过归因分析发现：
- **图像归因**：模型正确关注了地图关键区域（图例、国家名等）
- **文本归因**：token "Bolivia" 对回答 "yes" 贡献最大 → 模型被预训练知识误导

**修正方法**：钳制 OCR 相关特征（如与 "Barcelona" 文本关联的特征）使模型更依赖图像信息，成功消除幻觉。

## 亮点与洞察

1. **首次在多模态领域实现自动化特征解释**：利用大模型解释小模型的范式从 LLM 扩展到 LMM
2. **情感特征的发现**：LMM 内部存在与"快乐""悲伤""饥饿/贪婪"等概念关联的特征，可通过引导使模型表达情感
3. **跨模态不变特征**：发现视觉动作"吃"与文本概念"贪婪/饥饿"共享统一特征，证明 LMM 内部存在跨模态语义统一表征
4. **幻觉因果分析**：实证发现幻觉并非因为模型"看不到正确区域"，而是文本 token 的预训练先验覆盖了视觉证据
5. **低级视觉特征丰富**：LMM 比纯 LLM 多了大量形状、纹理、颜色等低级视觉特征

## 局限性

- SAE 训练计算成本高（$2^{17}$ 特征），仅在 LLaVA-NeXT-8B 单模型上验证
- 特征解释依赖更大模型（LLaVA-OV-72B），当最大模型本身是分析对象时方法不可用
- 仅分析了 576 个 base image token，未覆盖 Anyres 策略下的高分辨率 token
- 引导效果的量化评估困难，目前主要依赖案例分析
- 幻觉修正仅展示了个例，缺乏系统性评估

## 相关工作与启发

- **Anthropic 的 Dictionary Learning** 和 **Templeton et al.** 在 LLM 上证明了 SAE 能学到单语义特征，本文将其扩展到多模态
- **Network Dissection** 通过预定义概念标注解释神经元，但不适用于 LMM 的开放语义
- 特征引导修正幻觉的思路可能催生新的推理时干预方法（不修改模型参数）
- 情感特征的发现暗示 LMM 可能自然地发展出类人的情感理解机制

## 评分

⭐⭐⭐⭐ — 开创性地将机械可解释性工具引入 LMM 领域，提供了丰富的案例洞察。情感特征和幻觉因果分析给人留下深刻印象。不足在于定量评估较少，可复现性受限于计算资源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](../../ACL2025/multimodal_vlm/spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](../../CVPR2025/multimodal_vlm/multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)

</div>

<!-- RELATED:END -->
