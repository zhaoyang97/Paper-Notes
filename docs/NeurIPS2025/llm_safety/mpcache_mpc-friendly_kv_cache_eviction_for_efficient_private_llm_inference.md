---
title: >-
  [论文解读] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference
description: >-
  [NeurIPS 2025][LLM安全][隐私推理] 本文提出MPCache，一个面向安全多方计算（MPC）的KV缓存淘汰框架，结合一次性静态淘汰和查询感知的动态选择，配合层次化聚类、线性化相似度近似和跨层索引共享等优化，在不牺牲LLM性能的前提下实现最高2.01倍延迟降低和8.37倍通信量削减。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "隐私推理"
  - "安全多方计算"
  - "KV缓存淘汰"
  - "LLM效率"
  - "稀疏注意力"
---

# MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference

**会议**: NeurIPS 2025  
**arXiv**: [2501.06807](https://arxiv.org/abs/2501.06807)  
**代码**: [GitHub](https://github.com/zengwenxuan/MPCache)  
**领域**: AI安全  
**关键词**: 隐私推理, 安全多方计算, KV缓存淘汰, LLM效率, 稀疏注意力

## 一句话总结

本文提出MPCache，一个面向安全多方计算（MPC）的KV缓存淘汰框架，结合一次性静态淘汰和查询感知的动态选择，配合层次化聚类、线性化相似度近似和跨层索引共享等优化，在不牺牲LLM性能的前提下实现最高2.01倍延迟降低和8.37倍通信量削减。

## 研究背景与动机

LLM即服务（MLaaS）中，用户需上传含敏感信息的prompt到云端，而服务商不愿公开模型权重。MPC能在保护双方隐私的同时完成推理，但面临严重的效率挑战，**尤其是长序列的注意力计算**。

作者对GPT-2在Secretflow框架上的MPC推理进行了性能分析，发现：
- 注意力计算主导了2PC和3PC协议的延迟和通信开销
- Softmax因包含max、指数、除法等复杂运算占注意力成本的主体
- 随序列长度增加，Softmax成本急剧增长

虽然明文推理中的KV缓存淘汰和稀疏注意力已被广泛研究，但**直接应用到MPC会适得其反**。作者在图1(d)中展示，直接应用动态KV淘汰算法反而增加了通信和延迟，因为它引入了MPC中代价高昂的操作：top-k排序（频繁比较协议）、余弦相似度（大量乘法）、token聚集（需将索引转为one-hot向量再与KV缓存相乘）。

因此，核心挑战是：**如何设计对MPC友好的KV缓存淘汰算法？**

## 方法详解

### 整体框架

MPCache基于三个关键观察设计：(1) 长序列的注意力图整体稀疏，可静态淘汰不重要token；(2) 注意力具有token级局部性，可通过聚类构建高效动态选择；(3) 相邻层的top-k排序相似，可跨层共享索引。整体流程分为prefill阶段的一次性静态淘汰和decoding阶段的查询感知动态选择。

### 关键设计

1. **一次性静态KV缓存淘汰**: 在prefill阶段对token进行三类划分——IA（对所有token重要，始终保留）、UIA（对所有token不重要，直接丢弃）、IC（对部分token重要，动态选择）。通过累加最后20%prompt token的注意力分数来衡量token重要性，排序后淘汰UIA token。实验显示，LLaMA-2-7B上约60%的token可以被静态淘汰而不影响性能。该操作仅需一次，成本可被整个生成过程摊销。

2. **MPC友好的动态KV缓存选择**: 将相邻token分组为聚类，将选择粒度从token级提升到聚类级。核心问题是如何准确高效地度量聚类的相似度。

   **相似度近似**: 使用最大点积上界而非均值或余弦相似度。给定查询 $\mathbf{q}$ 和聚类键缓存 $\mathbf{K}_c$，推导上界：
    $\mathrm{Sim}(\mathbf{q}, \mathbf{K}_c) \leq \sum_{i=0}^{d-1} \max(\mathbf{q}_i \mathbf{r}_i^{\max}, \mathbf{q}_i \mathbf{r}_i^{\min})$
   其中 $\mathbf{r}_i^{\max}$ 和 $\mathbf{r}_i^{\min}$ 分别为聚类中键缓存在第 $i$ 维的极值（仅需计算一次）。

   **线性化与重排序**: 为消除MPC中代价高的max操作，进一步线性化近似：
    $\mathrm{Sim}(\mathbf{q}, \mathbf{K}_c) \approx \sum_{i=0}^{d-1} \mathbf{q}_i \cdot (\alpha \mathbf{r}_i^{\max} + (1-\alpha) \mathbf{r}_i^{\min})$
   其中 $\alpha=0.6$。重排序后先求和再乘法，乘法量减少2倍，max操作减为0。

3. **层次化KV缓存聚类**: 将KV缓存组织为层次结构，从粗粒度（大聚类）到细粒度（小聚类）逐级选择。粗粒度级别大幅缩小搜索空间，细粒度级别仅在已选聚类内精细筛选，平衡了精度和效率。

4. **跨层索引共享策略**: 利用相邻层注意力模式的相似性（commonality score高达80%+），让相邻两层共享选中的token索引，将动态选择的额外开销直接减半。第一两层因commonality较低不参与共享。

### Token聚集协议优化

聚类设计直接优化了MPC开销最大的token聚集协议：
- 等价协议次数从 $T$ 降为 $C$（聚类数）
- one-hot向量位宽从 $\log T$ 降为 $\log C$
- 例如 $T=1024, C=64$ 时，延迟降低73.5倍，通信降低369.8倍

## 实验关键数据

### 与动态方法的对比（LongChat-7B，LongBench）

| 数据集 | 缓存预算 | InfLLM性能/延迟 | LongCache性能/延迟 | MPCache性能/延迟 |
|--------|---------|----------------|-------------------|-----------------|
| HotpotQA | 5% | 28.20/51.64s | 24.31/89.46s | **30.27/39.85s** |
| TriviaQA | 5% | 75.65/51.64s | 59.85/89.46s | **75.61/37.37s** |
| NarrQA | 5% | 12.80/47.74s | 14.65/86.42s | **17.23/36.13s** |
| PassageR | 10% | 8.87/68.04s | 24.92/123.1s | 27.75/58.47s |

### 延迟与通信改进

| 序列长度 | 解码延迟降低 | 通信量降低 |
|---------|------------|----------|
| 短序列 | 1.8× | 3.39× |
| 长序列 | 2.01× | **8.37×** |

### 扩展到LLaMA-3.1-8B-Instruct

| 方法 | NarrQA | Qasper | HotpotQA | TriviaQA | 平均 |
|------|--------|--------|----------|----------|------|
| Full Cache | 30.21 | 45.52 | 55.53 | 91.74 | — |
| StreamingLLM (2048) | 27.40 | 30.77 | 49.23 | 90.98 | 差距大 |
| SnapKV (2048) | 28.53 | 39.13 | 54.32 | 92.04 | 中等 |
| MPCache (2048) | **30.17** | **46.11** | **55.21** | **91.53** | 接近Full |

### XSUM摘要任务（ROUGE-L）

| 预算 | 模型 | Full Cache | H2O | MPCache |
|------|------|-----------|-----|---------|
| 10% | 7B | 11.90 | 10.50 | **11.10** |
| 5% | 7B | 11.90 | 4.886 | **10.08** |
| 10% | 13B | 13.60 | 13.24 | **13.44** |
| 5% | 13B | 13.60 | 9.081 | **13.08** |

### 关键发现

- MPCache在所有基准任务上一致超越现有KV淘汰方法，尤其在激进压缩率（5%预算）下优势明显
- 线性化近似（$\alpha=0.6$）在消除max操作的同时几乎不损失性能
- 聚类策略使token聚集协议的延迟和通信分别降低73.5倍和369.8倍
- 跨层索引共享几乎不影响性能（共享两层的commonality > 80%）

## 亮点与洞察

- **问题定义精准**: 不是简单地将明文KV淘汰搬到MPC，而是深入分析了MPC中每个操作的成本，识别出三大MPC不友好操作并逐一解决
- **从"密码学协议"到"算法优化"**: 不修改底层MPC协议，而是设计对MPC友好的算法层面优化，兼容性更强
- **层次化设计思想**: 聚类、层次选择、跨层共享三者配合，形成了系统性的效率优化方案

## 局限与展望

- 静态淘汰的60%阈值可能在不同任务/模型上需要调整
- 线性化近似中 $\alpha=0.6$ 是经验选择，缺乏自适应机制
- 仅在GPT-2和LLaMA系列上验证，未测试Transformer变体（如MoE模型）
- 安全分析依赖已有协议的安全性，未考虑恶意对手模型

## 相关工作与启发

- 相比MPC协议优化（PUMA, Bolt）类工作，MPCache在算法层面优化，两者是互补的
- 与H2O、StreamingLLM等静态淘汰方法相比，MPCache的静态+动态混合策略在MPC场景下取得了更好的精度-效率平衡
- 启发：MPC友好的算法设计应成为隐私AI中的一个独立研究方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将KV缓存淘汰专门适配MPC的思路新颖，聚类近似设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ 多个模型（7B/8B/13B）、多个任务（QA/摘要/Needle-in-Haystack）、完整的效率数据
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰，观察驱动的设计思路，图表丰富且信息量大
- **价值**: ⭐⭐⭐⭐ 对隐私LLM推理的实际部署有重要意义，但受限于MPC的应用范围

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Cascade: Token-Sharded Private LLM Inference](../../ICML2025/llm_safety/cascade_token-sharded_private_llm_inference.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)

</div>

<!-- RELATED:END -->
