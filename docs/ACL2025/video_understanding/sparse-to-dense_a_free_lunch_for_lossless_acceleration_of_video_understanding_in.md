---
title: >-
  [论文解读] Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs
description: >-
  [ACL 2025][视频理解][视频大语言模型] 基于Video-LLM中注意力分数的稀疏性观察，提出Sparse-to-Dense (StD)解码策略，用top-K稀疏注意力模型作为draft model快速生成候选token，再用全注意力模型并行验证，实现最高1.94倍的无损加速，且无需额外训练或架构修改。
tags:
  - "ACL 2025"
  - "视频理解"
  - "视频大语言模型"
  - "推测解码"
  - "稀疏注意力"
  - "KV缓存"
  - "无损加速"
---

# Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs

**会议**: ACL 2025  
**arXiv**: [2505.19155](https://arxiv.org/abs/2505.19155)  
**代码**: 无  
**领域**: 视频理解 / LLM推理加速  
**关键词**: 视频大语言模型, 推测解码, 稀疏注意力, KV缓存, 无损加速

## 一句话总结

基于Video-LLM中注意力分数的稀疏性观察，提出Sparse-to-Dense (StD)解码策略，用top-K稀疏注意力模型作为draft model快速生成候选token，再用全注意力模型并行验证，实现最高1.94倍的无损加速，且无需额外训练或架构修改。

## 研究背景与动机

**领域现状**：视频大语言模型(Video-LLM)通过将视频表示为图像帧序列来处理视频理解任务，在视频问答和描述等任务上取得了优异性能。然而，一小时视频在5秒采样间隔下产生720帧，在VILA中可转化为141,120个视觉token，导致推理延迟极高。

**现有痛点**：Video-LLM的自回归解码机制要求每个新token都要attend所有前序token，随着KV缓存不断增长，频繁的内存访问对带宽造成了巨大压力。现有解决方案包括KV缓存压缩和量化，但这些方法都引入了训练和推理之间的分布偏差，会降低模型性能。

**核心矛盾**：要实现无损加速，不能修改模型输出分布。推测解码(speculative decoding)可以满足这一要求，但通常需要额外的draft model，对Video-LLM来说成本过高。

**本文目标** 设计一种针对Video-LLM的无损加速方法，不需要额外训练模型，即插即用。

**切入角度**：作者观察到Video-LLM解码时注意力分数呈现显著稀疏性——仅保留top-K的KV缓存就能保持约95%的next-token预测准确率。这意味着稀疏注意力模型可以作为天然的draft model，与原始模型共享参数，无需额外GPU内存。

**核心 idea**：利用Video-LLM自身的注意力稀疏性构建免训练的draft model，通过"稀疏提案+密集验证"实现无损推测解码加速。

## 方法详解

### 整体框架

StD包含两个模块：(1) 稀疏模型(Sparse Model)使用top-K注意力快速自回归生成γ个候选token，(2) 密集模型(Dense Model)即原始Video-LLM使用完整KV缓存并行验证这些候选token。两个模型共享相同的架构和参数，仅在注意力计算方式上不同——稀疏模型加载精选的KV缓存子集，密集模型使用完整缓存。验证通过后将匹配的token加上一个bonus token追加到序列中，进入下一轮提案-验证循环。

### 关键设计

1. **基于文本引导的视觉KV缓存选择**:

    - 功能：从大量视觉token中选出最关键的K个KV缓存对
    - 核心思路：由于视觉token数($m_v$)远大于文本token数($m_t$)，重点压缩视觉KV缓存。在prefill阶段，分析文本token $X_t$ 对视觉token $X_v$ 的平均注意力分数，为每层$l$选择注意力得分最高的top-K个视觉KV对：$\text{Cache}_s[l] = \text{argTopK}_{x \in X_v}\big(\frac{1}{m_t}\sum_{\hat{x} \in X_t} A_l(\hat{x}, x)\big)$。对于GQA架构，直接在每个group内求和注意力分数来选择。
    - 设计动机：top-K选择仅在prefill阶段进行一次，避免了解码阶段动态选择带来的额外计算开销；以文本token的注意力作为信号，因为文本token代表了用户查询的意图

2. **稀疏-密集协同解码流程**:

    - 功能：实现无损加速的核心推理循环
    - 核心思路：稀疏模型自回归生成γ个候选token（仅访问K+$m_t$个KV缓存），密集模型一次性并行验证所有γ个候选（读取完整$m_v + m_t$个KV缓存但只需一次I/O）。验证找到前n个匹配token后，连同额外的bonus token $\hat{x}_{n+m}$ 一起追加到序列中，形成下一轮上下文
    - 设计动机：稀疏模型速度快但可能出错，密集模型慢但准确；通过推测解码框架将两者结合，每轮最少接受1个token（bonus token），最多接受γ+1个token

3. **I/O复杂度优化分析**:

    - 功能：理论证明StD在何种条件下能带来加速
    - 核心思路：StD每轮平均每token的I/O为 $\frac{\gamma \times (K + m_t) + m_v + m_t}{\alpha \times \gamma}$，其中$\alpha$为接受率。相比原始解码的 $m_v + m_t$，当$\alpha > (K + m_t)/(m_v + m_t) + \gamma^{-1}$时StD更优。由于注意力的集中特性，K可以远小于$m_v$，使得这一条件容易满足
    - 设计动机：提供了理论保证，说明在视觉token远多于文本token的典型Video-LLM场景下（$m_v > 10000$, $m_t \approx 100$），StD几乎必然能加速

### 损失函数 / 训练策略

StD是完全无训练的方法。超参数设置：KV缓存总量 $K + m_t = 1024$，候选token数 $\gamma = 9$，batch size = 8。仅需约20行代码即可将原始Video-LLM转化为稀疏版本。

## 实验关键数据

### 主实验

| 模型/方法 | MLVU Acc/Speedup | VideoMME-s Acc/Speedup | VideoMME-m Acc/Speedup | VideoMME-l Acc/Speedup |
|----------|-----------------|----------------------|----------------------|----------------------|
| LLaVA-OV-7B + LayerSkip | 10.0/0.47× | 5.6/0.33× | 8.1/0.46× | 4.8/0.44× |
| LLaVA-OV-7B + Streaming | 34.7/1.34× | 36.4/1.38× | 41.0/1.51× | 36.2/1.45× |
| LLaVA-OV-7B + **StD** | **47.8/1.72×** | **51.8/1.82×** | **52.1/1.83×** | **52.9/1.59×** |
| Qwen2-VL-7B + LayerSkip | 5.2/0.63× | 3.7/0.59× | 4.9/0.55× | 5.7/0.55× |
| Qwen2-VL-7B + Streaming | 53.9/1.61× | 52.9/1.32× | 59.2/1.36× | 59.6/1.36× |
| Qwen2-VL-7B + **StD** | **66.1/1.94×** | **71.8/1.71×** | **73.4/1.62×** | **81.8/1.70×** |

### 消融实验

| 配置 | 关键观测 | 说明 |
|------|---------|------|
| Top-K ratio变化 | K=512时接受率~92%,K=256时~88% | 保留少量KV缓存即可保持高接受率 |
| γ=5 vs γ=9 vs γ=13 | γ=9是最佳平衡点 | 过大的γ导致低接受率，过小浪费并行验证优势 |
| LayerSkip作draft | 接受率极低(5-10%) | 层跳过导致分布偏移太大 |
| Streaming作draft | 接受率中等(35-60%) | 流式注意力保留效果不如top-K选择 |

### 关键发现

- **稀疏注意力draft模型大幅优于其他draft策略**：LayerSkip因分布偏移过大导致接受率极低甚至减速；Streaming基于固定窗口选择，不如StD的文本引导top-K选择
- **StD在长视频上加速更明显**：视觉token越多，稀疏模型相对全注意力模型的速度优势越大
- **无损保证是核心卖点**：由于验证阶段使用完整KV缓存，输出分布与原始模型完全一致，不存在性能损失
- **平均接受率62.2%**：远高于LayerSkip，与Streaming相比也有显著优势

## 亮点与洞察

- **自身即draft model的优雅设计**：利用模型固有的注意力稀疏性，无需训练额外模型，共享参数不增加GPU内存，是推测解码在多模态LLM中的最佳实践
- **20行代码即插即用**：极低的工程门槛使其可以立即部署到任何基于Transformers的Video-LLM
- **文本引导视觉缓存选择**：利用文本query的注意力模式来挑选关键视觉token，比随机选择或基于位置的选择更有效，这个思路可以迁移到其他多模态场景

## 局限与展望

- **KV缓存仍需全部存储在GPU内存中**：虽然减少了I/O访问，但没有减少内存占用，这在处理超长视频时仍然是瓶颈
- **单一视频粒度**：仅在帧级别的视觉token上做稀疏化，没有考虑跨帧的时序冗余
- **仅支持greedy/sampling解码**：没有讨论对beam search等复杂解码策略的支持
- **未来方向**：将KV缓存部分卸载到CPU内存（offloading），利用CPU更大的容量来突破HBM瓶颈；扩展到长思维链Video-LLM如QvQ

## 相关工作与启发

- **vs MagicDec (Chen et al., 2024a)**：MagicDec将流式注意力作为draft model用于文本LLM，本文将稀疏注意力方案扩展到Video-LLM，利用视觉token的稀疏性获得更大加速比
- **vs EAGLE / Medusa**：这些方法需要训练额外的draft head，成本高且不一定适用于多模态模型；StD完全免训练
- **vs FastV / VidCompress**：这些方法通过token剪枝/压缩减少视觉token数量，但会引入信息损失；StD保持完全无损

## 评分

- 新颖性: ⭐⭐⭐⭐ 将注意力稀疏性作为免费draft model的思路简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 两个主流Video-LLM、两个benchmark、三个baseline，实验完整
- 写作质量: ⭐⭐⭐⭐ 观察→理论→方法→实验的逻辑链清晰
- 价值: ⭐⭐⭐⭐ 即插即用、无损加速的实用价值高，研究思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](../../ICCV2025/video_understanding/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](../../CVPR2025/video_understanding/ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](../../CVPR2025/video_understanding/dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)

</div>

<!-- RELATED:END -->
