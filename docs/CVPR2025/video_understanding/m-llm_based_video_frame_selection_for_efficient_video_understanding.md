---
title: >-
  [论文解读] M-LLM Based Video Frame Selection for Efficient Video Understanding
description: >-
  [CVPR 2025][视频理解][视频帧选择] 提出一个轻量级 M-LLM 帧选择器，通过空间和时序伪标签训练，自适应地为下游视频 LLM 选取与问题最相关的帧，无需微调下游模型即可提升多个视频 QA 基准性能。 当前视频 M-LLM 普遍采用均匀采样策略从视频中抽取固定数量帧送入模型。这种"一刀切"的方式存在明显问题：…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频帧选择"
  - "多模态大语言模型"
  - "视频问答"
  - "伪标签"
  - "即插即用"
---

# M-LLM Based Video Frame Selection for Efficient Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2502.19680](https://arxiv.org/abs/2502.19680)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频帧选择, 多模态大语言模型, 视频问答, 伪标签, 即插即用

## 一句话总结

提出一个轻量级 M-LLM 帧选择器，通过空间和时序伪标签训练，自适应地为下游视频 LLM 选取与问题最相关的帧，无需微调下游模型即可提升多个视频 QA 基准性能。

## 研究背景与动机

当前视频 M-LLM 普遍采用均匀采样策略从视频中抽取固定数量帧送入模型。这种"一刀切"的方式存在明显问题：

1. **信息损失**：均匀采样可能遗漏关键事件帧，尤其在长视频中，每隔数秒采一帧极易错过短时间动作
2. **冗余帧干扰**：采样到的帧可能相互冗余或与问题无关，浪费宝贵的上下文窗口
3. **效率瓶颈**：密集均匀采样虽能覆盖更多时间点，但输入帧数 $n$ 增大会显著增加推理开销

核心洞察：大多数视频 QA 问题只需少量关键帧就能回答。如果能根据问题自适应选帧，就可以用更少的帧达到甚至超过密集采样的效果。

## 方法详解

### 整体框架

系统采用两阶段架构：先用轻量帧选择器从密集采样的 $n=128$ 帧中挑选 $k$ 个关键帧，再将选出的帧送入冻结的下游视频 M-LLM 进行问答。帧选择器以即插即用方式工作，只需训练一次即可增强多个不同的下游模型。

### 关键设计

1. **M-LLM 帧选择器架构**: 基于 Qwen2.5-1.5B 小型 LLM 微调而成。输入 $n$ 帧视频和问题文本，在输入序列末尾附加一个可学习的 score query $q \in \mathbb{R}^{1 \times d}$。利用因果注意力机制，$q$ 能聚合所有视觉和文本 token 的信息。从倒数第二个 Transformer block 提取 $q$ 的隐层表示 $e^q$，经 MLP 映射为 $n$ 维重要性向量 $s = \text{MLP}(e^q) \in \mathbb{R}^n$。关键效率设计：对每帧视觉 token 做激进空间池化，从 $12 \times 12 = 144$ 压缩到 $3 \times 3 = 9$ 个 token，因为判断帧重要性不需要精细视觉细节。

2. **空间-时序伪标签生成**: 由于缺乏帧级重要性标注数据，设计两种伪标签自动生成策略：
    - **空间伪标签**：用 Qwen2-VL-7B 对每帧独立评分。采用 CoT prompting 让模型先解释再输出 True/False，重要性分数 $s = p_{\text{True}} / (p_{\text{True}} + p_{\text{False}})$
    - **时序伪标签**：先用 M-LLM 为所有帧生成描述 caption，再将全部 caption 和问题送入 GPT-4o mini，让 LLM 进行跨帧时序推理，输出最相关帧的索引列表
    - 最终伪标签取二者平均，兼顾单帧空间信息和多帧时序关系

3. **Greedy NMS 帧采样**: 获得重要性分数后，不直接取 top-k（邻近帧分数相近会导致冗余），而是用贪心 + 非极大抑制策略：每次选最高分帧后，抑制其邻居帧（距离 $\leq n/4k$），确保选出的帧在时间轴上分布合理。

### 损失函数 / 训练策略

采用两阶段训练：

- **Stage 1**：冻结视觉编码器和 LLM backbone，训练对齐投影器 $g_a$、score query $q$ 和 score projector $g_s$。交替优化两个任务：(1) 视觉指令跟随（交叉熵损失），训练投影器对齐特征空间；(2) 重要性分数预测（二元交叉熵损失），初始化评分模块
- **Stage 2**：加入 LLM 的 LoRA 权重，仅训练重要性分数预测任务，使 LLM 适应帧选择任务。学习率 $10^{-5}$，cosine scheduler，5 个 epoch

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (PLLaVA 34B + Selector) | PLLaVA 34B baseline | 提升 |
|--------|------|------|----------|------|
| ActivityNet-QA | Acc/Score | 62.3/3.6 | 60.9/3.7 | +1.4 |
| NExT-QA | Acc | 69.3 (LLaVA-NeXT 34B) | 68.1 | +1.2 |
| EgoSchema | Acc | 65.9 (Qwen2-VL 7B) | 64.6 | +1.3 |
| VideoMME | Avg Acc | 58.7 (Qwen2-VL 7B) | 58.1 | +0.6 |
| LongVideoBench | Acc | 57.0 (Qwen2-VL 7B, 32帧) | 53.3 | +3.7 |

在所有测试的下游模型（PLLaVA、LLaVA-NeXT-Video、Idefics2、Qwen2-VL）上均获得一致提升。

### 消融实验

| 配置 | ActivityNet-QA | NExT-QA | 说明 |
|------|---------|------|------|
| 均匀采样 | 53.5 | 62.4 | baseline |
| CLIP 相似度选帧 | 53.7 | 62.2 | 简单文图匹配不足 |
| SeViLA 伪标签 | 54.0 | 63.2 | 单帧评估缺乏时序 |
| 仅空间伪标签 | 54.2 | 63.6 | CoT 改进评估质量 |
| 空间+时序伪标签 | 55.5 | 63.9 | 时序推理显著有效 |
| 训练后选择器 | 55.1 | 63.4 | 轻量选择器接近伪标签上限 |

### 关键发现

- **帧选择器可用一半帧数达到相同性能**：128→8 帧选择的性能 ≈ 16 帧均匀采样，推理速度快约 1.5x
- **每帧仅需 9 个 token**：从 1 到 25 token 差异不大，验证了"判断帧重要性不需要精细视觉信息"的假设
- **1.5B backbone 已足够**：0.5B→1.5B 提升明显，1.5B→7B 提升有限，体现轻量设计的有效性

## 亮点与洞察

- **即插即用设计**：帧选择器不修改下游模型参数，训练一次可服务多个不同的视频 LLM，实用性极强
- **空间+时序伪标签互补**：空间标签捕捉单帧内容相关性，时序标签通过 caption 进行跨帧推理，二者融合效果最佳
- **激进 token 压缩**：将每帧 token 压缩到 9 个的设计直觉非常好——选帧只需粗略轮廓，不需要细节

## 局限与展望

- 在已经很强的模型（如 Qwen2-VL）上提升有限（+0.6~1.3%），可能因为强模型本身就对输入帧有一定鲁棒性
- 伪标签生成成本高（需要对每帧 prompt M-LLM），虽然只在训练时使用，但数据标注开销大
- 在 Video Grounding (QVHighlights) 上弱于 SeViLA，说明帧选择和时刻定位仍有差距
- 选择器与下游模型分离训练，无法端到端优化，存在次优风险

## 相关工作与启发

- SeViLA 通过 M-LLM 对每帧独立打分选帧，但缺乏时序推理且推理开销大（每帧需单独推理）
- 本文的 score query 设计类似于 DETR 的 object query，用可学习 token 聚合全局信息做预测
- Token 压缩思路可迁移到其他视频理解场景，如视频摘要、长视频检索

## 评分

- 新颖性: ⭐⭐⭐⭐ 帧选择器整体架构设计新颖但非颠覆性，伪标签策略有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个基准、4 个下游模型、丰富消融，非常完整
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详细，但部分LaTeX公式排版不够规范
- 价值: ⭐⭐⭐⭐ 即插即用的实用设计，对工业界视频 QA 系统有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient Frame Selection for Long Video Understanding via Reinforcement Learning](../../CVPR2026/video_understanding/efficient_frame_selection_for_long_video_understanding_via_reinforcement_learnin.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](../../ICCV2025/video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)

</div>

<!-- RELATED:END -->
