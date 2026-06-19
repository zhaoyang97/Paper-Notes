---
title: >-
  [论文解读] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models
description: >-
  [CVPR 2025][视频理解][视觉token剪枝] 将视觉token剪枝问题重新建模为**Max-Min Diversity Problem (MMDP)**，通过精确求解使保留token集合的**最小pair-wise距离最大化**，实现无需训练/校准的即插即用剪枝方案，在16个多模态基准上实现SOTA，特别是在≥80%极端剪枝率下显著优于所有基线。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视觉token剪枝"
  - "大规模多模态模型"
  - "多样性最大化"
  - "Max-Min Diversity Problem"
  - "免微调"
---

# DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models

**会议**: CVPR 2025  
**arXiv**: [2503.02175](https://arxiv.org/abs/2503.02175)  
**代码**: [https://github.com/vbdi/divprune](https://github.com/vbdi/divprune)  
**领域**: 视频理解  
**关键词**: 视觉token剪枝, 大规模多模态模型, 多样性最大化, Max-Min Diversity Problem, 免微调

## 一句话总结

将视觉token剪枝问题重新建模为**Max-Min Diversity Problem (MMDP)**，通过精确求解使保留token集合的**最小pair-wise距离最大化**，实现无需训练/校准的即插即用剪枝方案，在16个多模态基准上实现SOTA，特别是在≥80%极端剪枝率下显著优于所有基线。

## 研究背景与动机

大规模多模态模型（LMM）将文本和视觉信息编码为token后送入LLM处理。视觉token数量通常远超文本token（如LLaVA 1.5使用576个视觉token），导致LLM的推理延迟和内存消耗急剧增加，因为注意力机制的复杂度与输入长度呈二次关系。

**现有痛点**：
- **基于注意力的方法**（FastV、PruMerge）：依赖注意力分数度量token重要性，但已被证明不够optimal——某些重要token被忽略，且保留的token之间**高度冗余**（高相似度），在高剪枝率时性能急剧下降
- **需要校准/微调的方法**（FitPrune、VTW、M³）：准确度较高但需要额外的校准数据集或模型微调，**部署成本高**且需针对每个模型定制

**核心矛盾**：视觉token中存在大量信息冗余（已有研究表明可剪枝50%-95%），但现有方法在高剪枝率时无法保留足够多样化的token来代表原始信息。

**核心idea**：与其关注单个token的"重要性"，不如关注保留集合的**多样性**——选择一组彼此最不相似的token，自然能最好地代表原始token分布。

## 方法详解

### 整体框架

DivPrune在LMM的视觉编码器输出（或LLM中间层）之后应用：计算所有视觉token的pair-wise余弦距离矩阵，求解MMDP问题选出 $\tilde{M}$ 个最多样化的token子集，丢弃其余token，将选中的token连同文本token送入LLM。整个过程无需训练、无需校准数据、即插即用。

### 关键设计

1. **Max-Min Diversity Problem (MMDP) 建模**
    - **功能**：将token剪枝转化为组合优化问题，从数学上保证保留集合的最大多样性
    - **核心思路**：寻找子集 $\tilde{\mathbf{E}}_v$ 使得其中任意两个元素间的最小距离最大化：$\tilde{\mathbf{E}}_v = \text{arg max}[\min_{\gamma,\omega \in S} d(\gamma,\omega)]$，距离度量采用余弦距离 $d(\gamma,\omega) = 1 - \frac{\gamma \cdot \omega}{\|\gamma\|\|\omega\|}$
    - **设计动机**：最大化多样性等价于最小化保留集合中的冗余。基于注意力的方法倾向于保留与query高度相关但彼此相似的token，导致信息覆盖面窄；而多样性目标确保token能更好地"覆盖"原始token的分布空间

2. **两阶段贪心求解算法**
    - **功能**：高效且精确地求解MMDP
    - **核心思路**：
     - **第一阶段**（首个token选择）：遍历所有token，计算每个token到其他token的最小距离，选择最小距离最大的token作为种子
     - **第二阶段**（迭代添加）：在每次迭代中，对候选列表中每个token计算其到已选集合的最小距离，选择该值最大的token加入已选集合，重复直至达到目标数量
    - **设计动机**：由于token数量有限（如576个），可通过预计算距离矩阵避免重复计算，GPU上的额外开销相比LLM计算几乎可忽略

3. **灵活的应用位置**
    - **功能**：支持在视觉编码器输出层或LLM中间解码层应用剪枝
    - **核心思路**：既可作用于投影后的视觉token（before LLM），也可作用于LLM某层输出的hidden states中对应视觉token的部分
    - **设计动机**：不同模型和任务可能在不同位置剪枝效果不同，灵活性使DivPrune适配各种LMM架构

## 实验关键数据

### 主实验（LLaVA 1.5-7B，TFLOP ratio≈15.6%，即剪枝约84%视觉token）

| 方法 | COCO(CIDEr) | GQA(EM) | MMBench(Acc) | OKVQA(EM) | POPE(F1) | SeedB(Acc) |
|------|-------------|---------|-------------|-----------|----------|-----------|
| 原始模型 | 1.10 | 61.96 | 64.09 | 53.39 | 85.84 | 66.17 |
| VTW | 0.05 | 38.94 | 21.31 | 18.64 | 25.35 | 36.13 |
| FastV | 0.06 | 38.73 | 20.62 | 18.32 | 32.84 | 35.69 |
| FitPrune(需校准) | 0.90 | 52.39 | 57.65 | 42.53 | 60.89 | 54.84 |
| **DivPrune** | **0.96** | **56.85** | **59.19** | **46.98** | **86.02** | **59.47** |

- 在POPE上DivPrune甚至超过未剪枝的原始模型（86.02 vs 85.84）
- 在约85%剪枝率下，DivPrune相比FastV/VTW性能提升巨大（OKVQA: 47 vs 18）

### 视频理解实验（LLaVA-NeXT-Video-7B）

DivPrune在MVBench、VideoMME、MLVU等5个视频基准上均优于FastV，平均提升约2-5个百分点

### 效率提升

| 指标 | 原始模型 | DivPrune(50%剪枝) | DivPrune(75%剪枝) |
|------|---------|-------------------|-------------------|
| 延迟(s) | 基准 | 降低~30% | 降低~50% |
| GPU内存 | 基准 | 降低~20% | 降低~40% |

### 消融实验

- 距离度量：余弦距离 > 欧氏距离 > L1距离
- 剪枝位置：在第2层LLM之后剪枝效果最佳（LLaVA 1.5-7B）
- 最小距离分布：DivPrune保留token集合的最小pair-wise距离显著高于FastV（直方图分析），验证了多样性假设

### 关键发现

- t-SNE可视化显示：FastV保留的token聚集在特征空间的局部区域，而DivPrune保留的token分布均匀覆盖整个特征空间
- 剪枝率越高（≥80%），DivPrune相对于基线的优势越明显
- 在LLaVA 1.5-13B上也表现一致，说明方法的模型无关性

## 亮点与洞察

- **问题建模的优雅性**：将token剪枝建模为经典的组合优化问题MMDP，视角新颖且数学上有保证
- **即插即用的工程价值**：无需训练、无需校准数据、适用于任何LMM架构、兼容KV cache等推理优化技术
- **反直觉发现**：token的"重要性"（注意力分数）不如token的"多样性"（互不相似程度）重要——高注意力token之间往往高度冗余

## 局限性

- 求解MMDP虽然在当前token数量（~576）下开销可忽略，但若token数量大幅增加（如高分辨率LMM的数千个token），计算成本可能变得不可忽略
- 多样性目标完全忽略了token的语义重要性，在某些需要聚焦特定区域的任务（如细粒度识别）上可能不如注意力方法
- 仅在CLIP视觉编码器的LMM上验证，对其他视觉编码器（如SigLIP、InternViT）的效果有待确认

## 相关工作与启发

- **FastV**和**PruMerge**代表了基于注意力的主流范式，DivPrune揭示了其"关注重要性而忽视冗余"的根本缺陷
- **FitPrune**和**M³**虽效果好但需额外训练/校准，DivPrune提供了免训练的替代方案
- 多样性优化思想可推广到：**KV cache压缩**（保留多样化的key-value对）、**视频帧采样**（选择多样化的关键帧）等场景

## 评分

⭐⭐⭐⭐ — 将token剪枝问题优雅地建模为MMDP是本文的核心亮点，方法简洁高效且实验全面（16个数据集×4个模型）。在高剪枝率下的显著优势和即插即用的特性使其具有很强的实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)

</div>

<!-- RELATED:END -->
