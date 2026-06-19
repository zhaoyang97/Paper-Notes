---
title: >-
  [论文解读] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs
description: >-
  [NeurIPS 2025][模型压缩][流式视频理解] 提出 rLiVS（Recurrent LLM-informed Visual Selection），一种无需训练的通用流式视频理解方法，通过LLM注意力权重选择关键视觉token（仅保留~6%）、循环复用历史token、基于字幕的检索问答三重设计，在流式视频基准上取得SOTA。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "流式视频理解"
  - "视觉token压缩"
  - "注意力选择"
  - "长视频"
  - "Video-LLM"
---

# Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2510.17364](https://arxiv.org/abs/2510.17364)  
**代码**: 暂无  
**领域**: 模型压缩  
**关键词**: 流式视频理解, 视觉token压缩, 注意力选择, 长视频, Video-LLM

## 一句话总结

提出 rLiVS（Recurrent LLM-informed Visual Selection），一种无需训练的通用流式视频理解方法，通过LLM注意力权重选择关键视觉token（仅保留~6%）、循环复用历史token、基于字幕的检索问答三重设计，在流式视频基准上取得SOTA。

## 研究背景与动机

Video-LLM在短视频理解上表现优异，但在流式场景（小时级视频在线处理、问题需实时回答）中面临严峻挑战。视觉token数量随帧数线性增长，暴力处理全部帧在长视频中计算不可承受且超出上下文长度限制。

现有长视频理解方案各有局限：
- **训练型方案**（如VideoStreaming、Flash-VStream）：需要额外训练，在任意长度视频上有外推问题，训练成本高
- **KV-cache方案**（如ReKV）：存储完整解码器KV-cache，内存消耗大（18.8GB/小时），存在冗余
- **纯字幕方案**（如Goldfish）：独立处理各短片段，缺乏时间连续性，难以跟踪实体

作者从认知神经科学获得启发：注意力是有限记忆容量下选择性编码的关键，而过去经验塑造当前注意力。由此提出利用LLM自身注意力做视觉token选择、循环传递历史上下文、文本检索回答三者结合的方案。

## 方法详解

### 整体框架

将长视频切为短片段（如16帧），逐片段流式处理。每个片段经过：(1) 拼接历史选中token + 当前片段token → 输入LLM生成字幕；(2) 基于注意力权重从当前片段选出少量关键token → 加入FIFO历史队列；(3) 字幕存入长期文本记忆。问答时，从文本记忆中检索最相关字幕，送入LLM生成答案。

### 关键设计

1. **基于注意力的视觉Token选择**

   在生成字幕后，利用已计算的注意力矩阵来衡量每个视觉token的重要性。从第 $l$ 层第 $h$ 个注意力头提取字幕token对视觉token的注意力系数:

    $\mathbf{A}^{l,h}_V = \mathbf{A}^{l,h}[TN_V+N_I : TN_V+N_I+N_C, \; 0:TN_V]$

   对每个视觉token $j$，跨所有字幕token、注意力头和层求平均得到全局重要性分数：

    $a_j = \frac{1}{L}\sum_{l=1}^{L}\frac{1}{H}\sum_{h=1}^{H}\left(\frac{1}{N_C}\sum_{i=1}^{N_C}\mathbf{A}^{l,h}_{V_{ij}}\right)$

   选择分数最高的 $N_S$ 个token保留（$N_S \ll N_V$），实践中仅保留6.25%（从3136个token中选196个）。为效率只需从$L$层中均匀采样4层即可获得稳健结果。

   设计动机：注意力分数是LLM在字幕生成过程中已经计算好的信号，不引入额外开销；且自然反映了哪些视觉token对当前语言理解最重要。

2. **循环式长视频处理**

   维护FIFO队列存储历史选中token $[\mathbf{S}^{(0)}, \mathbf{S}^{(1)}, \ldots, \mathbf{S}^{(t)}]$，在处理下一个短片段时作为上下文前缀输入LLM。当超出上下文窗口限制$W$时，丢弃最早的选中token。

   循环设计的双重作用：(1) 增强短片段间的视觉连续性和一致性；(2) 引导LLM注意力关注与历史信息一致的内容，强化选择效果。

3. **基于字幕的检索问答**

   存储所有短片段字幕的嵌入 $\{\mathbf{X}_C^{(t)}\}$。给定问题 $q$，计算问题token $\mathbf{X}_q$ 与字幕token的平均余弦相似度，用 MMR（Maximal Marginal Relevance）平衡相关性与多样性来检索top-K字幕。仅将检索到的字幕（非视觉token）输入LLM回答。

   选择字幕而非视觉token的原因：实验发现视觉token与问题的相似度集中在[-0.02, 0.06]附近，几乎无区分度；而字幕相似度分布在[0.4, 0.9]，区分度好。且LLM在长文本推理上能力成熟，将问题转化为文本QA更有效。

### 损失函数 / 训练策略

完全无需训练（training-free），直接基于预训练的Video-LLM推理。适用于任何短视频预训练的Video-LLM，无需调整架构。

## 实验关键数据

### 主实验

流式基准 (RVS-Ego / RVS-Movie)：

| 方法 | 骨干 | RVS-Ego Acc | RVS-Movie Acc | 延迟 | VRAM |
|------|------|:---:|:---:|:---:|:---:|
| Flash-VStream-7B | 专用 | 57.3 | 53.1 | 2.1s | 19GB |
| ReKV | LLaVA-OV 7B | 63.7 | 54.4 | 2.7s | 36GB |
| **rLiVS** | LLaVA-OV 7B | **65.3** | **57.7** | **1.9s** | **25GB** |
| **rLiVS** | Qwen2.5-VL 7B | **68.1** | 56.1 | 2.7s | 19GB |
| ReKV | LLaVA-OV 0.5B | 54.7 | 44.6 | 1.6s | 19GB |
| rLiVS | LLaVA-OV 0.5B | 57.6 | 51.3 | 1.5s | 11GB |

离线基准：

| 方法 | VS-Ego Acc | VS-Movie Acc | MovieChat Acc | CG-Bench Acc |
|------|:---:|:---:|:---:|:---:|
| Flash-VStream-7B | 59.0 | 56.1 | - | - |
| Goldfish | - | - | 67.6 | - |
| **rLiVS** | **61.0** | **59.3** | **78.0** | **33.1** |

### 消融实验

Token选择方法对比 (NextQA, 保留6% token)：

| 选择方法 | 准确率 |
|----------|:---:|
| 完整模型 (100%) | 78.6 |
| 均匀采样 (6%) | 75.5 |
| Mean Pooling (6%) | 70.7 |
| K-Means (6%) | 76.8 |
| **注意力选择 (6%)** | **77.0** |
| 注意力选择 (12%) | 78.4 |

设计选择消融 (流式基准)：

| 配置 | RVS-Ego Acc | RVS-Movie Acc | 说明 |
|------|:---:|:---:|------|
| rLiVS (完整) | 65.3 | 57.7 | 包含循环+注意力选择+字幕问答 |
| 去除循环 | 62.5 | 53.7 | 循环贡献3-4%提升 |
| 用视觉token检索回答 | 58.2 | 48.4 | 字幕远优于视觉token |
| 均匀采样替代注意力选择 | 64.2 | 56.0 | 注意力选择优1-2% |

### 关键发现

- 仅保留6%视觉token，性能损失仅1.6%(NextQA)；12%时几乎无损
- 循环传递历史token对长视频理解提升3-4个百分点
- 字幕显著优于视觉token作为检索和问答的信息载体
- 0.5B模型+rLiVS 超过了需要7B的多数竞争方法
- 10K上下文长度是效率与效果的最佳平衡点

## 亮点与洞察

- 方法极为简洁优雅：利用LLM已计算的注意力做选择，零额外开销
- 模型无关设计：可即插即用到LLaVA-OV、Qwen2.5-VL等任意Video-LLM
- KV-cache零存储：不需要像ReKV那样存储完整KV-cache（节省18.8GB/h）
- 认知科学启发的设计：注意力→选择性记忆→循环处理，模仿人类视觉信息处理机制

## 局限与展望

- 仅关注被选中内容，可能遗漏细粒度细节
- FIFO记忆缓冲区基于时间而非语义优先，关键但早期的信息可能被丢弃
- 循环字幕生成可能引入跨段冗余
- 完全依赖预训练骨干的能力，继承其局限
- 可探索自适应压缩率（根据场景复杂度动态调整保留比例）

## 相关工作与启发

- ReKV存储完整KV-Cache做流式理解 → rLiVS用极少token实现更好效果
- Goldfish独立处理短片段 → rLiVS通过循环增加时间连续性
- 注意力作为token重要性指标 → 可推广到其他多模态长上下文场景
- "将视频问答转化为文本QA"的洞察值得在其他长视频系统中借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ 整合已知概念（注意力选择、循环处理、字幕QA）为优雅的统一方案
- 实验充分度: ⭐⭐⭐⭐⭐ 流式+离线多基准、充分消融、效率对比
- 写作质量: ⭐⭐⭐⭐⭐ 刻画清晰、算法伪代码完整
- 价值: ⭐⭐⭐⭐⭐ 无训练、即插即用、高效实用，为流式视频理解设立了强基线

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[NeurIPS 2025\] Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video](eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](../../ICML2026/model_compression/token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)
- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](../../ICML2025/model_compression/orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)

</div>

<!-- RELATED:END -->
