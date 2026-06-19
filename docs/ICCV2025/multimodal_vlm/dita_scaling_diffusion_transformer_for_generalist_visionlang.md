---
title: >-
  [论文解读] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy
description: >-
  [ICCV 2025][多模态VLM][VLA] 提出Dita(Diffusion Transformer Policy)，区别于先前方法用浅层网络在embedding上去噪，采用in-context conditioning让去噪直接条件化于原始视觉token，通过causal Transformer处理语言+图像+timestep+噪声动作的完整token序列，334M参数在SimplerEnv零样本/LIBERO/CALVIN等benchmark上达到SOTA或可比性能。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VLA"
  - "扩散策略"
  - "DiT"
  - "in-context conditioning"
  - "跨embodiment"
---

# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

**会议**: ICCV 2025  
**arXiv**: [2503.19757](https://arxiv.org/abs/2503.19757)  
**代码**: [Project](https://robodita.github.io)  
**领域**: 机器人策略 / VLA模型  
**关键词**: VLA, 扩散策略, DiT, in-context conditioning, 跨embodiment  

## 一句话总结

提出Dita(Diffusion Transformer Policy)，区别于先前方法用浅层网络在embedding上去噪，采用in-context conditioning让去噪直接条件化于原始视觉token，通过causal Transformer处理语言+图像+timestep+噪声动作的完整token序列，334M参数在SimplerEnv零样本/LIBERO/CALVIN等benchmark上达到SOTA或可比性能。

## 研究背景与动机

- **领域现状**：通用机器人策略通过在OXE等大规模跨embodiment数据上预训练取得进展。
- **现有痛点**：(1) 离散化动作(OpenVLA)限制异构动作空间的适应性；(2) 用MLP/小DiT作为扩散head的方法(Octo/π₀)在大规模数据的多样性面前表达能力不足；(3) 在embedding上去噪丢失了历史观测的视觉细节。
- **核心矛盾**：异构的跨embodiment动作空间 vs 需要统一的策略表示。
- **本文目标**：设计表达力强、可扩展的通用机器人策略架构。
- **切入角度**：将动作去噪直接放入causal Transformer中与视觉token交互。
- **核心 idea**：动作去噪不应在压缩的embedding上进行，而应直接与原始视觉patch token做in-context attention。

## 方法详解

### 整体框架

CLIP编码语言→DINOv2+Q-Former提取图像特征→拼接[语言, 图像, timestep, 噪声动作]token序列→causal DiT去噪→输出清洁动作chunk(16步)。

### 关键设计

**设计1：In-Context Conditioning扩散**
- **功能**：将噪声动作token与视觉/语言token在同一causal Transformer中处理。
- **核心思路**：动作token直接参与注意力计算，可以attend到每个图像patch token，捕捉微妙的动作增量和环境细节。
- **设计动机**：先前方法在单个embedding上条件化去噪，丢失了空间细节；in-context conditioning保留了完整的视觉信息。

**设计2：DINOv2端到端微调+Q-Former**
- **功能**：DINOv2提取多尺度特征，Q-Former基于语言指令上下文查询关键视觉特征。
- **核心思路**：DINOv2在网络数据上预训练与机器人数据有gap，端到端微调弥合差距。Q-Former用FiLM conditioning从DINOv2 patch特征中选择任务相关信息，减少计算量。
- **设计动机**：冻结的视觉编码器在机器人领域不够好，但全微调token过多需要Q-Former压缩。

**设计3：轻量可扩展架构**
- **功能**：仅用334M参数实现SOTA性能。
- **核心思路**：LLaMA风格causal Transformer，无需大型VLM(如PaliGemma)。DDPM训练(1000步)+DDIM推理(20步)。
- **设计动机**：提供简洁、轻量、开源的baseline，降低社区入门门槛。

### 损失函数/训练策略

标准DDPM扩散目标：$\min \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2$。AdamW 100K步，batch 8192(32×A100)，2帧观测→16步action chunk。

## 实验关键数据

### 主实验

**SimplerEnv零样本（成功率%）**

| 方法 | coke_can(match/var) | move_near(match/var) |
|------|---------------------|---------------------|
| RT-1-X | 56.7/49.0 | 31.7/32.3 |
| OpenVLA | 16.3/54.5 | 46.2/47.7 |
| **Dita** | **83.7/85.5** | **76.0/73.0** |

**LIBERO微调（成功率%）**

| 方法 | SPATIAL | OBJECT | GOAL | LONG | 平均 |
|------|---------|--------|------|------|------|
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| **Dita** | 84.2 | **96.3** | **85.4** | **63.8** | **82.4** |

### 消融实验

| 配置 | Calvin Avg. Len |
|------|----------------|
| Diffusion head(不是in-context) | 3.16 |
| In-context Dita | **3.53** |
| 无预训练 | 2.38 |

### 关键发现

1. In-context conditioning比diffusion head显著更好，尤其在长任务上(LIBERO-LONG +10%)。
2. 仅第三人称相机+10-shot即可泛化到真实世界新环境。
3. 334M参数超越7B(OpenVLA)和更大模型，说明架构设计比规模更重要。

## 亮点与洞察

1. In-context conditioning的核心insight——动作去噪需要看到原始视觉细节而非压缩embedding。
2. 轻量开源baseline对社区价值大。
3. 跨embodiment预训练+10-shot真实世界微调的范式实用性极强。

## 局限与展望

1. 仅用第三人称视角，加入腕部相机/触觉可进一步提升。
2. Q-Former的查询数量对性能的敏感度未充分分析。
3. 未在双臂操控场景验证。

## 相关工作与启发

- Octo用diffusion head但表达力有限；Dita证明将去噪"内化"到Transformer中更好。
- π₀用更大VLM但Dita用334M达到可比性能。
- 启发：机器人策略学习的关键可能不是模型大小而是动作与观测的交互方式。

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ★★★★☆ |
| 实用性 | ★★★★★ |
| 实验充分性 | ★★★★★ |
| 写作清晰度 | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers](vq-vla_improving_vision-language-action_models_via_scaling_vector-quantized_acti.md)
- [\[ICCV 2025\] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance](coavla_improving_visionlanguageaction_models_via_visualtext.md)
- [\[ICCV 2025\] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models](tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)
- [\[ICCV 2025\] Chimera: Improving Generalist Model with Domain-Specific Experts](chimera_improving_generalist_model_with_domain-specific_experts.md)
- [\[CVPR 2025\] Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy](../../CVPR2025/multimodal_vlm/optimus-2_multimodal_minecraft_agent_with_goal-observation-action_conditioned_po.md)

</div>

<!-- RELATED:END -->
