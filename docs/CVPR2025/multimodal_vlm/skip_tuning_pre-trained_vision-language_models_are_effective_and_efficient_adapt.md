---
title: >-
  [论文解读] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves
description: >-
  [CVPR 2025][多模态VLM][VLM微调] 揭示 prompt tuning 冻结 VLM 参数既不促进知识迁移也未显著提升效率（仅减 6% 内存/16% 时间），提出 Skip Tuning 通过层级跳过（LSkip）和类别跳过（CSkip）缩短全微调的梯度传播流，实现 15× 时间效率和 6.4× 内存效率提升的同时精度更优。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "VLM微调"
  - "Skip连接"
  - "高效适配"
  - "提示学习"
  - "CLIP"
---

# Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves

**会议**: CVPR 2025  
**arXiv**: [2412.11509](https://arxiv.org/abs/2412.11509)  
**代码**: [https://github.com/Koorye/SkipTuning](https://github.com/Koorye/SkipTuning)  
**领域**: 多模态VLM  
**关键词**: VLM微调、Skip连接、高效适配、prompt tuning替代、CLIP

## 一句话总结
揭示 prompt tuning 冻结 VLM 参数既不促进知识迁移也未显著提升效率（仅减 6% 内存/16% 时间），提出 Skip Tuning 通过层级跳过（LSkip）和类别跳过（CSkip）缩短全微调的梯度传播流，实现 15× 时间效率和 6.4× 内存效率提升的同时精度更优。

## 研究背景与动机

**领域现状**：Prompt tuning（CoOp、MaPLe、PromptSRC 等）被认为是适配 CLIP 等 VLM 到下游任务的有效+高效范式——学习少量 context vector，冻结 VLM 参数。

**现有痛点**：对比全微调（FT）baseline，prompt tuning 虽然参数量降至 1/51200，但内存仅减 6.3%、时间仅减 15.8%——因为冻结参数仍需前向传播占用内存。更关键的是，FT 的分类精度比 CoOp 高 3.49%（base）和 4.49%（new），说明冻结参数反而限制了知识迁移。

**核心矛盾**：PT 追求的"参数效率"（少参数可训练）在实际部署中不如"内存/时间效率"重要，而PT 的高参数效率并未转化为高内存/时间效率。

**本文目标** 不引入额外 prompt 或 adapter，直接优化 FT baseline 的内存和时间效率，使其在效率和效果上都超越 PT 方法。

**切入角度**：分析 Feature-Gradient Propagation Flows（FGPF）发现大部分浅层对任务特定知识贡献极小（Feature Sensitivity 接近零），大部分类别 token 对特定训练图片的梯度也很小（Gradient Dependence 低）。跳过这些部分可以大幅减少计算量。

**核心 idea**：缓存浅层特征跳过前向/反传（LSkip）+ 过滤无关类别 token 减少宽度（CSkip），双管齐下让 FT 变得比 PT 更快更省。

## 方法详解

### 整体框架
FT 开始前，对 CLIP 的前 $\omega$ 层进行一次前向传播并缓存中间特征 → 训练时只从第 $\omega+1$ 层开始前向/反传（LSkip）→ 对文本编码器，每个训练样本只保留 top-$r \times M$ 个最相关类别 token 和指数采样的额外类别（CSkip）→ 标准 ITM 损失训练。

### 关键设计

1. **Layer-wise Skipping（LSkip）**:

    - 功能：缩短 FGPF 的长度
    - 核心思路：用 Feature Sensitivity 度量每层对 FT 的贡献——计算 FT 前后每层输出的欧氏距离。发现浅层 FS 接近零，深层 FS 显著。只需微调 FS 高的后 $N-\omega$ 层，前 $\omega$ 层缓存特征即可
    - 设计动机：ViT-B/16 的前 9 层 FS ~0，仅后 3 层 FS 显著。缓存前 9 层节省了 75% 的层计算

2. **Class-wise Skipping（CSkip）**:

    - 功能：缩短 FGPF 的宽度
    - 核心思路：用 Gradient Dependence 度量每个类别 token 对每张训练图片的贡献。发现大部分类别的梯度极小。保留 top-$r \times M$ 个最相关类（$r=0.05$），其余类按指数衰减采样保留少量以维持泛化
    - 设计动机：1000 个类中通常只有 50 个对当前训练图有意义。去掉其余 950 个不仅降低计算量，还减少了无关梯度的噪声干扰

### 损失函数 / 训练策略
标准 ITM 损失，全参数微调（仅后 $N-\omega$ 层）。无额外 prompt/adapter 参数。

## 实验关键数据

### 主实验

| 方法 | Base-New H | 时间效率 | 内存效率 |
|------|-----------|---------|---------|
| CoOp | ~72 | 1× | 1× |
| PromptSRC | ~74 | 1× | 1× |
| **Skip Tuning** | **最优 H** | **×15 快** | **×6.4 省** |
| LoRA | ~73 | 1× | 1× |
| **Skip Tuning vs LoRA** | **+3.59% H** | **×3.8 快** | **×3.9 省** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| FT baseline | 好精度但慢 | 全部计算 |
| +LSkip (ω=9) | 精度不降，3× 加速 | 浅层冻结+缓存 |
| +CSkip (r=0.05) | 精度略升，2× 加速 | 去掉无关类 |
| LSkip+CSkip | **最优效率+精度** | 双重加速 |

### 关键发现
- **PT 的效率优势被高估**：CoOp 只比 FT 省 6.3% 内存，因为冻结的大量参数仍需前向传播
- **浅层几乎不贡献任务知识**：前 9/12 层的 Feature Sensitivity 接近零
- **CSkip 实际上提升泛化**：去掉无关类 token 减少了梯度噪声，对 new class 性能有正面影响

## 亮点与洞察
- **颠覆了"PT 比 FT 更高效"的认知**——当正确优化 FT 时，它在效率和效果上都可以超越 PT
- **不需要任何额外参数/模块**就超越了 prompt tuning 和 adapter 方法，方法极致简洁
- **FGPF 分析框架**可推广到其他需要高效微调的场景

## 局限与展望
- ω 的选择依赖于 Feature Sensitivity 分析，不同 backbone 可能需要重新确定
- CSkip 的指数采样策略是启发式的，可以探索自适应采样
- 仅在 CLIP 上验证，对 BLIP/SigLIP 等其他 VLM 的效果未知

## 相关工作与启发
- **vs CoOp/MaPLe/PromptSRC**：这些方法引入额外 prompt 参数但效率提升有限。Skip Tuning 无额外参数且更快
- **vs LoRA/Adapter**：这些方法引入可训练低秩矩阵/模块。Skip Tuning 更简单且效率更高（3.8× 时间，3.9× 内存）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 颠覆 PT 认知的实证分析和极简解决方案都非常出色
- 实验充分度: ⭐⭐⭐⭐⭐ base-to-new/cross-dataset/domain/few-shot 四大基准全面超越
- 写作质量: ⭐⭐⭐⭐⭐ 从挑战认知到提出方案，逻辑严密且有说服力
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 高效适配领域有范式级意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)

</div>

<!-- RELATED:END -->
