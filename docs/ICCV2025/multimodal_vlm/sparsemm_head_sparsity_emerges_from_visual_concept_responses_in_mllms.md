---
title: >-
  [论文解读] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs
description: >-
  [ICCV 2025][多模态VLM][Visual Head] 揭示了多模态大语言模型(MLLM)中仅约5%的注意力头实际参与视觉理解的"visual head sparsity"现象，提出基于OCR任务的免训练visual head识别框架，并设计SparseMM——一种按视觉分数对不同head分配不对称KV-Cache预算的加速策略，实现1.38×实时加速和52%显存降低，同时保持性能不降。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "Visual Head"
  - "MLLM"
  - "KV-Cache Compression"
  - "Head Sparsity"
  - "注意力机制"
  - "Inference Acceleration"
---

# SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

**会议**: ICCV 2025  
**arXiv**: [2506.05344](https://arxiv.org/abs/2506.05344)  
**代码**: [https://github.com/CR400AF-A/SparseMM](https://github.com/CR400AF-A/SparseMM)  
**领域**: 多模态大模型 / 模型加速 / KV Cache优化  
**关键词**: Visual Head, MLLM, KV-Cache Compression, Head Sparsity, Attention Analysis, Inference Acceleration  

## 一句话总结
揭示了多模态大语言模型(MLLM)中仅约5%的注意力头实际参与视觉理解的"visual head sparsity"现象，提出基于OCR任务的免训练visual head识别框架，并设计SparseMM——一种按视觉分数对不同head分配不对称KV-Cache预算的加速策略，实现1.38×实时加速和52%显存降低，同时保持性能不降。

## 研究背景与动机

### 问题定义
MLLM（如LLaVA、Qwen2-VL）通过在预训练LLM上接入视觉编码器来处理多模态输入。随着多模态输入的复杂度增长（高分辨率图像、长视频、多轮对话），维护完整KV-Cache的计算和显存开销变得难以承受。

### 现有方法的局限

**通用KV-Cache压缩（SnapKV/PyramidKV/AdaKV）**：专为纯文本设计，均匀对待所有attention head，忽略了视觉token在MLLM中的特殊角色

**视觉token剪枝（FastV）**：按层级剪枝冗余视觉token，但未从head层面考虑模态特异性

**根本性知识缺口**：LLM是如何在visual instruction tuning中获得视觉理解能力的，这一问题尚未被充分研究

### 核心发现
通过分析MLLM的注意力机制，发现两个关键性质：

**稀疏性**：即使经过大量多模态数据训练，各层中仅不到5%的attention head是"视觉活跃"的

**普遍性**：visual head在不同LLM架构（Vicuna/Qwen2）和注意力机制（MHA/GQA）中一致存在

## 方法详解

### 整体框架
SparseMM分两步：
1. **Visual Head识别**：基于OCR任务，利用文本-图像精确对应关系，免训练量化每个head的视觉相关性
2. **KV-Cache不对称分配**：根据视觉分数，为visual head分配更多cache预算，非视觉head激进压缩

### 关键设计1：Visual Head识别算法
利用OCR作为锚点任务（文字→图像区域有精确映射）：
- 对每个输出token $y_i$，通过(text, bbox)配对找到对应图像区域
- 确定该区域的image tokens集合 $I_{y_i}$
- 遍历所有attention head $h$：若head $h$的最高注意力指向image tokens内，则记录hit
- Visual Score定义：

$$\text{Visual Score}(h) = \frac{1}{N}\sum_{i=1}^{N}\frac{\mathbb{I}_{hit(y_i, A_h)}}{\#\text{image\_tokens}}$$

关键：区域越小（越精确）得分越高。统计1000张Synthdog OCR图像的累计分数并归一化。

### 关键设计2：三部分KV-Cache分配

给定总预算 $B$、$L$ 层每层 $H$ 个head：

1. **Local Window Cache**：每个head固定分配最近邻窗口 $w=32$
2. **Uniform-Based Cache**：从剩余预算中取比例 $\rho=0.1$ 均匀分配给所有head
   $$r = \frac{\rho \cdot (B - N \cdot w)}{N}$$
3. **Score-Preferred Cache**：剩余预算按visual score比例分配
   $$b_{ij}^{score} = B_{remain2} \cdot \frac{s_{ij}}{\sum_{i,j} s_{ij}}$$

最终每个head (i,j) 的预算：$b_{ij} = w + r + b_{ij}^{score}$

### KV-Cache选择
借鉴SnapKV，使用末尾观察窗口(32 tokens)计算注意力分数，选top-K个KV Cache保留。将复杂度从 $O(N^2)$ 降为 $O(N \times L)$。

## 实验关键数据

### 实验设置
- **模型**：LLaVA-NeXT-Vicuna-7B (MHA, 32层×32头)、LLaVA-NeXT-Mistral-7B (GQA, 32层×8 KV头)、Qwen2-VL-7B (GQA, 28层×4 KV头)
- **基线**：SnapKV、PyramidKV、AdaKV + Random Head
- **评测**：DocVQA、OCRBench、TextVQA、ChartQA、TextCaps、MMBench、VQAv2

### 主要结果（精度-速度权衡）

| 方法 | DocVQA | OCRBench | TextVQA | ChartQA | TextCaps | 延迟(ms) |
|------|--------|----------|---------|---------|----------|----------|
| FullKV | 0.68 | 0.52 | 0.65 | 0.55 | 0.73 | 52.9 |
| **SparseMM** | **0.68** | **0.52** | **0.65** | **0.54** | **0.73** | **37.1 (-30%)** |
| SnapKV | 0.64 | 0.46 | 0.62 | 0.50 | 0.65 | 35.3 |
| PyramidKV | 0.65 | 0.48 | 0.62 | 0.53 | 0.65 | 34.9 |
| AdaKV | 0.65 | 0.48 | 0.62 | 0.49 | 0.66 | 37.3 |

KV Cache预算256，16K input tokens。SparseMM在几乎无精度损失下降低30%延迟。

### 效率评估（LLaVA-NeXT-Vicuna-7B，预算256）

| 输入长度 | 加速比 | 显存节省 |
|----------|--------|----------|
| 8K tokens | 1.16× | ~2GB |
| 16K tokens | ~1.5× | ~4GB |
| 32K tokens | 1.87× | ~15.5GB (32.87→17.38GB, 约50%) |

### 消融实验：Cache分配策略

| Local Window | Uniform | Score-Preferred | MMBench (512/256/128/96/64/48) |
|:---:|:---:|:---:|------|
| ✓ | ✗ | ✗ | 81.3/80.5/77.3/73.6/70.5/67.2 |
| ✓ | ✓ | ✗ | 81.5/81.4/79.3/77.6/74.6/73.9 |
| ✓ | ✓ | ✓ | 81.5/81.4/81.5/81.4/80.3/77.9 |

关键发现：三部分缺一不可。特别是ρ=0时（完全靠score分配），Mistral模型性能从0.519骤降到0.145，说明确保每个head最低预算很重要。

### Visual Head遮蔽实验
- 遮蔽top 5% visual head → OCRBench下降~20%、TextVQA下降~15%
- 随机遮蔽同等比例head → 影响极小（<3%）
- 证实visual head虽稀疏但不可或缺

### Visual Head的跨数据集鲁棒性
OCR数据集（MLT、CTW）识别的visual head分布高度一致，且优于检测任务（COCO），因为OCR提供更精确的一对一映射。

## 亮点与洞察

1. **从认知层面理解MLLM**：首次系统揭示MLLM中visual head的稀疏分布现象——LLM在visual instruction tuning后仅有极少数head学会了"看"
2. **OCR锚点任务的巧妙性**：利用OCR的精确text-bbox-image对应关系来量化visual relevance，比用检测任务噪声更小
3. **免训练、架构无关**：visual score可离线计算一次，推理时零额外开销。支持MHA和GQA架构
4. **Random Head=SnapKV**：当head score随机初始化时，score-based分配退化为均匀分配，等价于SnapKV——这解释了SnapKV的局限性
5. **可视化说服力强**：visual head确实聚焦图像中的文字/物体区域，而非视觉head随机分散

## 局限性

1. 仅支持解码阶段的KV Cache压缩，未考虑prefill阶段的token剪枝
2. visual head识别依赖OCR数据，对于非文字密集型任务（如3D理解）的适用性需更多验证
3. 当前仅在7B级模型验证，更大模型（如70B+）上的稀疏比例和效果未知
4. 单图评测居多，多图交错输入和长视频场景的适用性需进一步探索
5. GQA模型中将多个query head的score求和作为KV head的总分，可能损失细粒度信息

## 相关工作与启发

- 与FastV的互补：FastV做层级visual token剪枝，SparseMM做head级KV分配，两者可结合
- head sparsity现象可能推动MLLM架构设计——是否可以在训练时就引导更多head关注视觉？
- 类似attention sink的思想（如StreamingLLM），但SparseMM发现的是模态特异性的"visual sink heads"

## 评分 ⭐⭐⭐⭐
- 创新性：⭐⭐⭐⭐（visual head sparsity现象的发现有原创意义，OCR识别框架简洁有效）
- 实验：⭐⭐⭐⭐⭐（覆盖7个benchmark + 3个模型架构 + 效率评测 + 可视化 + 消融全面）
- 写作：⭐⭐⭐⭐（结构清晰，图表丰富，发现和方法的逻辑链条紧密）
- 实用性：⭐⭐⭐⭐⭐（免训练+即插即用，实测50%显存降低、30%延迟降低，实际部署价值高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](../../ICLR2026/multimodal_vlm/sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICCV 2025\] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)

</div>

<!-- RELATED:END -->
