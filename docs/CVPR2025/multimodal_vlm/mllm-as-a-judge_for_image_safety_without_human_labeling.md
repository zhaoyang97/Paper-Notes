---
title: >-
  [论文解读] MLLM-as-a-Judge for Image Safety without Human Labeling
description: >-
  [CVPR 2025][多模态VLM][image safety] 提出 CLUE 框架，通过规则客观化、CLIP 相关性扫描、前置条件链分解和去偏 token 概率分析，实现无需人工标注的零样本图像安全判定，在多个 MLLM 上大幅超越基线。 领域现状： 在线平台和 AIGC 时代，图像内容安全审核至关重要…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "image safety"
  - "MLLM"
  - "zero-shot judgment"
  - "safety constitution"
  - "debiased token probability"
  - "content moderation"
---

# MLLM-as-a-Judge for Image Safety without Human Labeling

**会议**: CVPR 2025  
**arXiv**: [2501.00192](https://arxiv.org/abs/2501.00192)  
**代码**: 未开源  
**领域**: 多模态VLM  
**关键词**: image safety, MLLM, zero-shot judgment, safety constitution, debiased token probability, content moderation

## 一句话总结

提出 CLUE 框架，通过规则客观化、CLIP 相关性扫描、前置条件链分解和去偏 token 概率分析，实现无需人工标注的零样本图像安全判定，在多个 MLLM 上大幅超越基线。

## 研究背景与动机

**领域现状**: 在线平台和 AIGC 时代，图像内容安全审核至关重要。现有方案主要依赖两类方法：(1) 传统分类器（Q16、NSFW Detector），(2) 微调 MLLM（LLaVA Guard）。两者都依赖人工标注数据。

**核心矛盾**: 人工标注成本高昂且难以扩展；安全规则可能频繁更新，而基于微调的方法每次规则变更都需要重新标注和训练。这引出核心问题：**能否在零样本设置下，仅凭预定义的安全准则（constitution）让预训练 MLLM 进行图像安全判定？**

**直接查询 MLLM 的三大挑战**:

1. **规则主观性**: 模糊规则如"不应包含色情内容"让人类专家都难以判断边界案例
2. **长规则推理困难**: 当前 MLLM 难以对复杂、冗长的安全规则进行准确推理
3. **模型固有偏见**: 包括语言先验偏见（模型倾向性回答）和图像非中心区域偏见（裸上身→模型偏向认为下半身也裸露）

## 方法详解

### 整体框架

CLUE（Constitutional MLLM JUdgE）是一个多阶段推理框架：

输入图像 → CLIP 相关性扫描 → 逐条规则检查 → 前置条件分解 → 去偏 token 概率判定 → (低置信度时) 级联 CoT 推理 → 输出安全标签 + 违规规则列表

### 关键设计

#### 1. 规则客观化（Rules Objectification）

将主观/模糊的安全规则转化为客观可操作的规则：

- 使用 LLM-as-an-Optimizer 评估每条规则的客观性（1-10 分）
- 低于 9 分的规则反复修订直到达标
- 例如：「不应包含色情内容」→ 细化为多条具体规则，如「女性胸部区域中未被不透明衣物完全覆盖的部分不被允许」
- 允许用户调整关键参数（如角度阈值 90°）

#### 2. 相关性扫描（Relevance Scanning）

利用 CLIP 的文本-图像相似度快速过滤与当前图像明显无关的规则：

$$\text{relevant if } \cos(\mathbf{I}(x), \mathbf{T}(r)) > t$$

阈值 $t=0.22$。CLIP 编码器参数量远小于 MLLM，大幅提升了整体推理效率。

#### 3. 前置条件链分解（Precondition Extraction）

将复杂规则自动分解为简化的前置条件链，仅当所有前置条件都满足时才判定违规：

**示例**: 规则「不应有遭受可见血腥伤害并导致即将死亡的人或动物」
→ 前置条件链: [[人可见] OR [动物可见]] AND [身体有可见血腥伤害] AND [伤害严重到导致即将死亡]

这种分解：(1) 降低了单次 MLLM 查询的推理复杂度，(2) 允许早期退出（某个前置条件不满足则跳过后续检查）

### 损失函数 / 判定机制

**去偏 Token 概率判定**:

对每个前置条件查询"Yes/No"，计算前置条件分数（Yes 概率 / (Yes + No 概率)）。

**Strategy 1 — 去语言先验偏见**:
比较有图像和无图像时的 token 概率差：
- $\mathcal{M}(x, c) - \mathcal{M}(\text{None}, c) < \alpha_1$ → 前置条件不满足
- $\mathcal{M}(x, c) - \mathcal{M}(\text{None}, c) > \alpha_2$ → 前置条件满足

**Strategy 2 — 去图像非中心区域偏见**:
使用 OWLv2 检测中心物体，比较原图和移除中心区域后的概率差：
- $\mathcal{M}(x, c) - \mathcal{M}(x \ominus i, c) > \beta$ → 前置条件满足

两策略结合使用。低置信度样本进入级联 CoT 推理阶段。

## 实验关键数据

### 主实验表（零样本基线对比）

| 方法 | 模型 | Recall | Accuracy | F-1 |
|------|------|--------|----------|-----|
| Prior Knowledge + Yes/No | InternVL2-76B | 62.6% | 71.8% | 0.691 |
| Entire Constitution + Yes/No | InternVL2-76B | 79.7% | 85.5% | 0.846 |
| Entire Constitution + CoT | InternVL2-76B | 75.3% | 82.2% | 0.809 |
| **CLUE (Ours)** | **InternVL2-76B** | **95.9%** | **94.8%** | **0.949** |
| **CLUE (Ours)** | InternVL2-8B-AWQ | 91.2% | 87.4% | 0.879 |
| **CLUE (Ours)** | Qwen2-VL-7B | 88.9% | 86.3% | 0.866 |

### 与微调方法对比

| 方法 | 类型 | 泛化性 |
|------|------|--------|
| Q16, SD Filter, NSFW Detector, LLaVA Guard | 微调 | 差（仅在训练规则上有效） |
| **CLUE** | **零样本** | **强（无需重新标注/训练即可更新规则）** |

CLUE 在零样本设置下大幅超越所有微调基线，验证了微调方法在规则泛化上的固有局限。

### 关键发现

1. **规则客观化是基础**: 将原始主观规则提升到客观性评分 ≥9 后，MLLM 的判定能力显著提升
2. **去偏机制至关重要**: 去除语言先验和图像非中心区域偏见后，token 概率判定准确性大幅提升
3. **前置条件分解优于直接推理**: 即使 GPT-4o 也无法对复杂规则直接推理，但能正确判断分解后的前置条件
4. **CLIP 相关性过滤高效**: 以极低计算成本过滤大量无关规则，推理速度提升数倍
5. **跨模型泛化**: 超参数（$\alpha_1, \alpha_2, \beta$）在不同 MLLM 上无需调整

## 亮点与洞察

1. **完全零样本**: 无需任何人工标注数据，规则更新只需修改文本，极大降低部署和维护成本
2. **系统性解决 MLLM 偏见**: 从语言先验和视觉注意力两个维度进行去偏，思路新颖且通用
3. **多阶段级联设计**: token 概率快速判定 + CoT 深度推理的级联策略，兼顾效率和准确性
4. **构建了 OS Bench**: 首个基于客观规则标注的图像安全评测基准，填补评测空白

## 局限性

1. **安全准则需人工定义**: 虽然免去了标注图像的人力，但仍需专家编写详细的安全规则
2. **依赖 CLIP 的感知能力**: 相关性扫描受限于 CLIP 对安全相关概念的理解能力
3. **推理成本仍较高**: 需对每条相关规则的每个前置条件查询 MLLM，多次前向传播
4. **OS Bench 使用 AI 生成图像**: 测试集由文生图模型生成，与真实用户上传内容分布可能有偏差
5. **阈值超参数**: 虽然声称跨模型鲁棒，但仍存在多个需要设定的阈值

## 相关工作与启发

- **LLaVA Guard**: 微调方式的 MLLM 安全判断，本文证明零样本可以超越
- **Constitutional AI (Bai et al.)**: 安全准则的概念来源，本文将其扩展到视觉域
- **VCD (Visual Contrastive Decoding)**: 去偏思想的灵感来源
- **启发**: 将复杂判断任务分解为简单前置条件链 + 去偏 token 概率的思路具有很强的通用性，可推广到其他需要基于规则判定的场景（如内容合规、版权检测）

## 评分 ⭐

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles](../../ICCV2025/multimodal_vlm/calibrating_mllm-as-a-judge_via_multimodal_bayesian_prompt_ensembles.md)
- [\[CVPR 2025\] HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator](heie_mllm-based_hierarchical_explainable_aigc_image_implausibility_evaluator.md)
- [\[CVPR 2025\] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)
- [\[CVPR 2025\] Locality-Aware Zero-Shot Human-Object Interaction Detection](locality-aware_zero-shot_human-object_interaction_detection.md)

</div>

<!-- RELATED:END -->
