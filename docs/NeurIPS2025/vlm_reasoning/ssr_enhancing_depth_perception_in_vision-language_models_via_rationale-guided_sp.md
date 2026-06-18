---
title: >-
  [论文解读] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning
description: >-
  [NeurIPS 2025][多模态VLM][空间推理] 提出 SSR 框架，将原始深度信息转化为结构化文本推理 rationale，并通过知识蒸馏压缩为紧凑潜在嵌入，以即插即用方式增强现有 VLM 的空间推理能力。 VLM 在多模态任务上表现出色，但仅依赖 RGB 输入难以准确捕捉空间信息（相对位置、距离等）…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "空间推理"
  - "深度感知"
  - "视觉语言模型"
  - "Chain-of-Thought"
  - "知识蒸馏"
---

# SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2505.12448](https://arxiv.org/abs/2505.12448)  
**代码**: [https://yliu-cs.github.io/SSR](https://yliu-cs.github.io/SSR)  
**领域**: 多模态VLM  
**关键词**: 空间推理, 深度感知, 视觉语言模型, Chain-of-Thought, 知识蒸馏

## 一句话总结

提出 SSR 框架，将原始深度信息转化为结构化文本推理 rationale，并通过知识蒸馏压缩为紧凑潜在嵌入，以即插即用方式增强现有 VLM 的空间推理能力。

## 研究背景与动机

VLM 在多模态任务上表现出色，但仅依赖 RGB 输入难以准确捕捉空间信息（相对位置、距离等）。现有融合空间线索的方法存在两类问题：依赖专用传感器（如 LiDAR 获取点云）不适用于仅有单目 RGB 的场景；或者虽然引入了深度图，但对深度信息的利用停留在表面（作为补充输入），未能发挥其推理价值。

核心洞察是：人类在推理空间关系时，不仅仅"看到"深度，而是将深度作为推理过程的一部分——先分析物体间的空间关系，再利用这种理解来指导后续推理。现有方法缺乏这种隐式的深度推理能力，因此需要一种更sophisticated的深度整合方式。

## 方法详解

### 整体框架

SSR 包含一个核心即插即用模块 MIDI（Mamba-based Image-Depth Interpreter），负责将 RGB 图像和深度图转化为包含空间推理信息的潜在 token 表示。整体流程为：输入图像→单目深度估计→MIDI 模块生成 rationale latent tokens→与原始图像和文本一起输入 VLM 生成答案。训练分两阶段：Stage 1 对齐推理+语义空间，Stage 2（可选）联合训练 MIDI 和 VLM。

### 关键设计

1. **MIDI 模块（Mamba-based Image-Depth Interpreter）**: 使用 CLIP ViT-L/14 编码 RGB 图像特征、SigLIP 编码深度特征，分别通过 MLP 投影到语义空间。然后将图像特征、深度特征和文本 query 输入一个基于 Mamba 的语言模型，生成中间推理 rationale 的潜在 token。关键创新是使用 Mamba（而非 Transformer）作为推理模块，计算效率更高。在 rationale 序列中均匀插入特殊 token，便于知识蒸馏压缩。

2. **Rationale-to-Latent 知识蒸馏**: 不同于传统 CoT 依赖冗长文本推理，SSR 将推理 rationale 压缩为紧凑的潜在嵌入。Stage 1 中，LLM 从 MIDI 生成的潜在 token 中重建文本 rationale，迫使潜在表示编码完整的推理信息。训练完成后，MIDI 可直接插入 VLM 的输入序列，无需修改 VLM 参数。

3. **SSR-CoT 数据集与 SSRBench**: 整合 LLaVA-CoT、Visual-CoT、VoCoT、SpatialQA 四个数据源，使用 Depth Pro 提取深度图、SpatialRGPT 挖掘空间属性、GPT-4o 生成详细推理过程，构建约 120 万级图像-深度-问题-rationale-答案对。SSRBench 包含 6 个任务（3 通用+3 空间），从 SSR-CoT 中抽取并严格去重。

### 损失函数 / 训练策略

- **Stage 1 (Reasoning and Alignment)**: 标准因果语言模型损失，训练 MIDI 生成的潜在 token 使下游 LLM 能重建原始 rationale 文本。仅训练 MIDI 模块，VLM 冻结。
- **Stage 2 (Co-Training, 可选)**: 丢弃中间 rationale，用标准因果损失训练 VLM 直接生成答案。联合训练 MIDI + VLM，并额外引入 LLaVA-Instruct-150K 扩展数据。
- 使用 LoRA + FSDP，单节点 8×H800，Stage 1 约 19h，Stage 2 约 48h。

## 实验关键数据

### 主实验

| Benchmark | 指标 | SSR (3B) | Qwen2.5-VL-3B | 提升 |
|-----------|------|----------|---------------|------|
| SpatialBench | 平均 | 64.8% | 59.3% | +5.4 |
| SSRBench 通用 | 平均 | 79.3% | 62.8% | +16.5 |
| SSRBench 空间 | 平均 | 69.7% | 48.8% | +20.9 |
| CV-Bench | 平均 | 68.9% | 67.0% | +1.9 |
| VSR | 零样本 | 82.9% | 76.4% | +6.5 |

SSR (7B) 进一步提升：SpatialBench 上超越 LLaVA-NeXT-13B、SpatialBot-3B 等基线。

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| 有 rationale vs 无 rationale | SSR-CoT 数据质量验证 | 准确率 +11.62%（67.80→79.42） |
| Stage 1 only (即插即用) | 不联合训练 VLM | 已有显著提升 |
| Stage 1 + Stage 2 | 联合训练 | 进一步提升各 benchmark |
| MIDI 模块大小 130M | 轻量级开销 | 与 3B VLM 配合即获大幅提升 |

### 关键发现

- MIDI 模块仅 130M 参数，就能为 3B VLM 带来空间推理平均 20.9% 的提升，性价比极高。
- Stage 1 训练的即插即用特性使得无需重新训练 VLM，实用性强。
- SSR-CoT 数据中 rationale 质量保证机制（缓存池+采样验证+迭代重标注）有效。
- 不同规模 VLM（3B→7B）均能从 SSR 获益，且在通用任务上也有提升而非仅限于空间任务。

## 亮点与洞察

- **深度信息的正确使用方式**：不是简单拼接深度特征，而是将深度转化为推理 rationale，再压缩为潜在表示——这种"先推理再压缩"的范式让深度信息真正参与了高阶认知过程。
- **即插即用设计**：MIDI 模块训练后可直接插入任意 VLM 的输入序列，无需修改原模型，部署友好。
- **Mamba 作为推理引擎**：选择 Mamba 而非 Transformer 进行推理，充分利用其选择性状态空间特性和线性复杂度优势。
- 数据构建流水线（Depth Pro + SpatialRGPT + GPT-4o）系统化，可复用。- Stage 1 训练 19h、Stage 2 训练 48h 的计算开销可接受，且 Stage 2 是可选的。

## 局限与展望

- 依赖单目深度估计模型（Depth Pro）的质量，深度估计失败时推理也会受影响。
- SSR-CoT 数据集构建依赖 GPT-4o，成本较高且可能引入偏差。
- 轻量 Mamba 推理模块的表达能力是否足以处理极复杂的空间场景有待验证。
- SSRBench 缺乏 3D 场景或视频级别的空间推理评测。
- 深度编码器（SigLIP）与图像编码器（CLIP ViT-L/14）使用不同模型，两者特征空间的对齐质量可能影响下游推理。
- 仅在 Qwen2.5-VL 上验证 Stage 2 联合训练，对其他 VLM 骨干的适配性有待确认。
- Rationale 质量检验仅抽样 10%，可能存在未检出的低质量样本。

## 相关工作与启发

- 与 SpatialVLM、SpatialRGPT 等工作的区别在于引入了 language-based reasoning 能力。
- 知识蒸馏从 rationale 到 latent 的思路受 Coconut (Chain of Continuous Thought) 启发，但在多模态场景下的应用是新颖的。
- 对具身智能中的机器人空间推理有直接应用价值。
- SSR-CoT 的多源数据融合流程（LLaVA-CoT + Visual-CoT + VoCoT + SpatialQA）可作为大规模 CoT 数据构建的参考模板。
- MIDI 模块的设计思路可推广到其他模态增强场景（如触觉、热图、光流等）。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将深度信息转化为推理 rationale 再蒸馏为 latent 的范式新颖
- **实验充分度**: ⭐⭐⭐⭐ 多 benchmark 验证，配备自建 SSRBench，数据质量评估完善
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，流程详尽
- **价值**: ⭐⭐⭐⭐ 即插即用的空间增强方案实用性强，对具身 AI 有意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)

</div>

<!-- RELATED:END -->
