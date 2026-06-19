---
title: >-
  [论文解读] Spatial Preference Rewarding for MLLMs Spatial Understanding
description: >-
  [ICCV 2025][多模态VLM][MLLM] 提出 SPR（Spatial Preference Rewarding）框架，通过语义分数和定位分数自动构建偏好数据对，利用 DPO 训练 MLLM 区分高精度定位（正样本）和模糊/错误定位（负样本），大幅提升细粒度空间理解能力，尤其在高 IoU 阈值下效果显著。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "MLLM"
  - "空间理解"
  - "偏好优化"
  - "DPO"
  - "目标定位"
---

# Spatial Preference Rewarding for MLLMs Spatial Understanding

**会议**: ICCV 2025  
**arXiv**: [2510.14374](https://arxiv.org/abs/2510.14374)  
**代码**: [SPR](https://github.com/hanqiu-hq/SPR)  
**领域**: 多模态大模型 / 空间理解  
**关键词**: MLLM, 空间理解, 偏好优化, DPO, 目标定位

## 一句话总结

提出 SPR（Spatial Preference Rewarding）框架，通过语义分数和定位分数自动构建偏好数据对，利用 DPO 训练 MLLM 区分高精度定位（正样本）和模糊/错误定位（负样本），大幅提升细粒度空间理解能力，尤其在高 IoU 阈值下效果显著。

## 研究背景与动机

多模态大语言模型（MLLM）在空间理解任务上取得了显著进展，如引用对话（referential dialogue）、定位描述（grounding captioning）等。然而 MLLM 在细粒度空间感知上仍存在明显不足：

**区域描述模糊**：模型生成的 grounded 区域描述经常不够详细，对象定位不准确

**注意力偏移**：模型可能被查询区域外的对象"分心"，未能聚焦于用户指定的区域

**缺乏正负样本反馈**：现有指令微调（SFT）仅优化模型模仿正样本（ground truth），无法惩罚推理时产生的错误定位。这与传统目标检测中正负样本训练机制的缺失形成对比

**核心矛盾**：SFT 训练范式缺少对 MLLM 实际输出质量的直接监督——模型知道什么是"对的"，但不知道什么是"错的"。

**切入角度**：将偏好优化（DPO）引入空间理解领域，让模型学会区分精确定位和错误定位，而非仅仅模仿标注。现有偏好优化工作主要聚焦于减少幻觉，细粒度空间对齐的偏好优化几乎空白。

## 方法详解

### 整体框架

SPR 采用三步式 DPO 流水线：
1. **收集 MLLM 原始响应**：构建随机区域查询 → 多样化提示生成 grounded 区域描述
2. **评估与排序**：语义分数 + 定位分数综合评估 → 最优/最差配对 → 精化最优描述
3. **偏好优化训练**：DPO + LoRA 微调

### 关键设计

#### 1. 随机区域查询构建

现有区域描述数据集过于简单（如"停在街上的车"等短语），难以产生有足够差异性的偏好数据对。因此作者设计了从头构建查询区域的方案：

- 从 Objects365 数据集中筛选对象丰富的图像
- 随机选一个标注框作为起始区域
- 迭代扩展到最近的邻近对象，直到包含 4+ 个对象
- 构建多样化提示（含裁剪区域图像 + 对象引用），引导 MLLM 生成多个候选区域描述

#### 2. 语义分数（Semantic Score）

评估描述与查询区域的语义匹配度：

$$S_{sem} = \frac{1}{2} \left( S(I_{crop}, T) + S_{local}(I, T) \right)$$

- $S(I_{crop}, T)$：裁剪区域图像与描述文本的 CLIP 余弦相似度
- $S_{local}(I, T)$：完整图像通过局部注意力层提取区域嵌入的相似度

**设计动机**：仅用裁剪图像会忽略周围上下文；结合局部注意力的完整图像相似度弥补了这一不足。

#### 3. 定位分数（Localization Score）

评估描述中对象的定位准确度和详细程度：

- 用 Grounding DINO 从描述文本提取对象框
- 结合原始标注框形成 ground truth 集合
- 合并 MLLM 描述中的定位框和 Grounding DINO 结果作为预测集合
- 计算预测框与 GT 框的平均 IoU（阈值 0.5）

$$S_{loc} = \frac{1}{n} \sum_i^n \max_j \mathbf{p}[i,j]$$

鼓励模型描述更多对象并给出准确的定位框。

#### 4. 综合评分与描述精化

$$S = \lambda S_{sem} + (1-\lambda) S_{loc}, \quad \lambda = 0.8$$

- 最高分和最低分的描述配对为偏好/拒绝数据
- **精化步骤**：对偏好描述进一步提升定位质量——保留 IoU>0.5 的预测，将其定位框替换为匹配的 GT 框，增大偏好/拒绝对之间的定位差异

### 损失函数 / 训练策略

DPO 损失：

$$\mathcal{L} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log\sigma\left(\beta \log\frac{\pi_*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_*(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

- 基础模型作为冻结的参考策略 $\pi_{ref}$
- 策略模型 $\pi_*$ 通过 LoRA 更新权重
- 训练在 10k images（Objects365）上构建偏好数据
- 1×A100，Ferret-7B 约 3 小时，13B 约 5 小时

## 实验关键数据

### 主实验（Referring Expression Comprehension，Acc@0.5）

| 模型 | RefCOCO val | RefCOCO+ val | RefCOCOg val | Flickr30k val |
|------|------------|-------------|-------------|---------------|
| Ferret-7B | 87.49 | 80.78 | 83.93 | 80.39 |
| + SPR | **88.39** | **82.07** | **85.58** | **81.53** |
| Ferret-13B | 89.48 | 82.81 | 85.83 | 81.13 |
| + SPR | **89.94** | **83.29** | **86.46** | **81.82** |
| CogVLM-17B | 92.76 | 88.68 | 89.75 | - |
| + SPR | **92.95** | **88.83** | **90.01** | - |

SPR 在三个不同基线 MLLM 上一致性地提升了性能。

### 消融实验（不同 IoU 阈值下的 REC）

| IoU 阈值 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---------|-----|-----|-----|-----|-----|
| Ferret-7B | 83.91 | 81.28 | 76.72 | 67.02 | 43.25 |
| + SPR | 84.93 | 82.36 | 78.42 | 70.09 | **52.21** |
| 提升 | +1.02 | +1.08 | +1.70 | +3.07 | **+8.96** |

SPR 的提升随 IoU 阈值升高而显著增大——**在 0.9 阈值下 7B 模型提升 8.96 个点**，表明 SPR 真正提升了定位精度而非仅仅增加了检测数量。

### 关键发现

1. **高 IoU 阈值下效果放大**：SPR 在严格定位要求下优势更加明显（IoU=0.9 时 +8.96 点）
2. **DPO > SFT**：仅用接受数据做 SFT 只获得约一半的提升（REC +0.44 vs DPO +1.02）
3. **语义分数和定位分数缺一不可**：λ=0 或 λ=1 都明显弱于 λ=0.8 的综合配置
4. **描述精化对多目标定位尤为关键**：在 Phrase Grounding 中提升更为显著
5. **空间能力提升带来通用能力增益**：GQA、TextVQA 等通用 benchmark 也有提升，同时 POPE 幻觉指标也有改善

## 亮点与洞察

1. **将传统检测的正负样本思想迁移到 MLLM**：填补了 MLLM 空间理解领域缺乏负样本监督的空白
2. **全自动流水线**：无需外部 MLLM 或人工标注，可扩展性强
3. **极低的训练成本**：1 个 A100，3-5 小时即可完成
4. **模型无关性**：在 Ferret、LLaVA-OV、CogVLM 三个不同 MLLM 上均有效
5. **高 IoU 阈值的实验设计**：区别于常规只看 0.5 阈值，提供了更深入的定位精度分析

## 局限与展望

- CLIP 作为语义评分器可能对某些细粒度差异不够敏感
- 偏好数据仅基于 Objects365 构建，领域泛化性有待验证
- 未探索 SFT 和 DPO 的融合训练策略
- 定位分数依赖 Grounding DINO 的检测质量
- 仅验证了 2D 图像场景，未扩展到视频或 3D 空间理解

## 相关工作与启发

- RLHF-V、POVID 等工作将偏好优化引入 MLLM 以减少幻觉，本工作将其扩展到空间理解
- Ferret、Shikra、Kosmos-2 等空间感知 MLLM 提供了基础架构支持
- CLIP-DPO 避免了昂贵的人工标注，与本工作的自动化评分思路一致
- 传统检测中 Focal Loss 等正负样本平衡策略启发了 SPR 的设计理念

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将偏好优化聚焦于空间定位精度，填补了重要空白
- **技术深度**: ⭐⭐⭐ — 方法简洁有效，但技术复杂度不高
- **实用价值**: ⭐⭐⭐⭐⭐ — 低成本、即插即用，对 MLLM 空间理解有直接提升
- **写作质量**: ⭐⭐⭐⭐ — 清晰系统，实验设计出色

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/multimodal_vlm/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)

</div>

<!-- RELATED:END -->
