---
title: >-
  [论文解读] Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective
description: >-
  [ICLR 2026][LLM效率][KV缓存淘汰策略] 提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。 KV缓存是LLM推理加速的核心技术：通过复用先前查询的key-v…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "KV缓存淘汰策略"
  - "随机化算法"
  - "负载均衡路由"
  - "多LLM服务"
  - "竞争比分析"
---

# Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective

**会议**: ICLR 2026  
**arXiv**: [2601.18999](https://arxiv.org/abs/2601.18999)  
**代码**: [GitHub](https://github.com/fzwark/KVRouting)  
**领域**: LLM Serving / KV Cache / Load Balancing  
**关键词**: KV缓存淘汰策略, 随机化算法, 负载均衡路由, 多LLM服务, 竞争比分析  

## 一句话总结
提出首个KV缓存感知负载均衡统一数学模型，设计随机化叶节点淘汰算法RLT(O(log n)竞争比)和基于学习的贪心路由LBGR，在多LLM服务场景下将延迟降低最高11.96×、TTFT降低14.06×。

## 研究背景与动机
**KV缓存是LLM推理加速的核心技术**：通过复用先前查询的key-value对避免重复计算，但在内存受限时其效果高度依赖淘汰策略

**LRU淘汰策略存在根本缺陷**：传统最近最少使用(LRU)策略在动态查询到达模式下表现脆弱，最坏情况下恰好淘汰下一个查询需要的token，导致缓存命中率骤降

**多LLM服务场景的固有矛盾**：最大化单个LLM的缓存命中率（将相似查询发往同一LLM）与全局负载均衡（避免队列延迟）是相互冲突的目标

**现有方法依赖启发式**：SGLang用固定阈值切换路由策略，NVIDIA/llm-d用静态线性评分，缺乏形式化建模，无法适应动态查询模式

**缺乏理论分析**：实际系统设计和理论理解之间存在显著差距，没有统一模型捕捉缓存淘汰与负载均衡之间的耦合关系

**队列负载建模过于粗糙**：现有方法仅用待处理查询数量衡量拥塞，未考虑各查询实际服务时间的差异

## 方法详解

### 整体框架
论文把"缓存淘汰"和"查询路由"这两条原本各自为政的线放进同一个数学模型里：$M$ 个 worker（LLM）各带一块大小为 $B_i$ 的 KV 缓存，在线接收查询序列 $Q=\{q_j\}$，每个查询既要选一个 worker 落地、又会改变该 worker 的缓存内容和队列负载。系统优化目标是最小化所有 worker 的 makespan（最大累计负载）。这套统一模型不只是记账工具——它直接把 SGLang 沿用的叶节点 LRU（L-LRU）算出 $O(n)$ 的最坏竞争比，把现有系统的脆弱性量化出来；由此拆出两个相互耦合的算法：单 worker 内"该淘汰谁"的随机化淘汰 RLT，把竞争比压到 $O(\log n)$；跨 worker"该发给谁"的学习路由 LBGR，用各 worker 的缓存与负载状态贪心分流。两者再被观测到的真实延迟在线校准，形成闭环。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    Q["在线查询序列 q_j"] --> MODEL["统一代价模型<br/>服务时间 Cost_ij + 队列负载 P_i<br/>目标：最小化 makespan"]
    MODEL -->|揭示 L-LRU 竞争比退化到 O(n)| RLT["RLT 随机化叶节点淘汰<br/>标记集满 B_i+1 即清空<br/>未标记叶节点均匀随机淘汰 → O(log n)（已达下界）"]
    RLT -->|每个 worker 缓存命中更稳健| LBGR["LBGR 学习贪心路由<br/>估 Ê_ij = Cost + 衰减负载 + 回归残差<br/>选预测延迟最低的 worker"]
    LBGR --> OUT["分配查询 → 服务 → 写回 Radix 树<br/>逼近最小 makespan / 端到端延迟"]
    OUT -.观测真实延迟 E_ij.-> UPD["在线更新回归参数 θ_i<br/>+ 队列负载指数衰减 ρ"]
    UPD -.持续校准估计.-> LBGR
```

### 关键设计

**1. 统一形式化模型：把淘汰与路由耦合进一个可分析的代价函数**

现有系统（SGLang 阈值切换、NVIDIA/llm-d 静态线性评分）都靠启发式拼凑，没人能说清"把相似查询塞给同一 LLM 提命中率"和"摊平负载避免排队"这对矛盾该如何权衡。论文先把一次查询的服务时间写成可命中部分与未命中部分的加权和：$\text{Cost}_{ij} = \alpha_{\text{CACHED}}\cdot h_{ij} + \alpha_{\text{MISS}}\cdot(|q_j| - h_{ij}) + O_{ij}$，其中 $h_{ij}$ 是查询 $q_j$ 在 worker $i$ 上的命中 token 数，$O_{ij}$ 为固定开销；路由决策 $x_{ij}\in\{0,1\}$ 配合 `UpdateCache` 更新缓存状态、队列负载 $P_i$ 按服务时间累加。这套形式化的直接收益是把 LRU 的脆弱性算成了数字——SGLang 用的叶节点 LRU（L-LRU）竞争比为 $O(n)$，一旦查询长度高度不平衡，最坏情况会恰好淘汰下一个查询要用的 token，命中率崩塌。

**2. RLT 随机化叶节点淘汰：用"标记+均匀随机"把竞争比从 $O(n)$ 砍到 $O(\log n)$**

确定性的 LRU 之所以脆弱，是因为对抗性的到达顺序总能精准踩中它的淘汰规则。RLT 借鉴在线算法里的 marking 思想引入随机性来打散这种最坏情况：维护一个标记集合 $T$ 记录已访问 token，标记数攒到 $B_i+1$ 时清空（只保留最新 token）；当缓存满、又要加载新 token 时，不按时间挑，而是从**未标记的叶节点**里**均匀随机**抽一个淘汰。这一点随机性让对抗者无法预测，把竞争比压到 $\Theta(\log(B_i - L))$，相比 L-LRU 的 $O(n)$ 是指数级改进；论文进一步证明这是所有随机化淘汰算法的下界，也就是信息论意义上不可再改进的最优。

**3. LBGR 学习贪心路由：估计每个 worker 的真实延迟，再贪心选最低的那个**

光把单 worker 淘汰做到最优还不够，跨 worker 怎么分才是命中率与负载的真正战场。LBGR 不再用待处理查询"数量"这种粗糙指标，而是逐项估出把 $q_j$ 发给 worker $i$ 的预期延迟 $\hat{E}_{ij} = \text{Cost}_{ij} + \tilde{P}_i + \theta_i^{\top}\varphi_{ij}$，发给预测值最低的 worker。三个分量各管一件事：$\text{Cost}_{ij}$ 借全局 Radix Tree 追踪各 worker 缓存、估出命中 token 数 $\tilde{h}_{ij}$；$\tilde{P}_i$ 追踪队列负载，并用指数衰减 $\tilde{P}_i \leftarrow \rho\cdot\tilde{P}_i$ 模拟负载随时间自然消退、查询完成后释放残余负载（由后台线程每 $\Delta t$ 执行一次，避免估计累积失真）；$\theta_i^{\top}\varphi_{ij}$ 是一个在线线性回归残差项，特征 $\varphi$ 含命中数、未命中数与当前负载，专门吸收环境波动带来的偏差。每次观测到真实延迟后，模型按 $(E_{ij} - \hat{E}_{ij})^2$ 在线更新回归参数，让估计随 workload 漂移持续校准。这套"解析代价 + 衰减负载 + 回归修正"的组合避开了训练 RL 或重型预测模型的开销，却能动态贴合查询模式。

## 实验

### 主实验结果

| 对比方法 | 中位延迟降低 | 中位TTFT降低 | 缓存命中率提升 | 吞吐提升 |
|---------|------------|------------|-------------|---------|
| vs Random+LRU | 30.9× | 44.49× | - | - |
| vs Cache-Aware+LRU (SOTA) | 11.96× | 14.06× | 36.45% | 36.51% |
| vs Cache-Aware+LRU (P95) | 2.03× | 2.62× | - | - |

### 消融实验

| 方法 | P50延迟(ms) | P50 TTFT(ms) | 命中率 | 额外开销 |
|------|-----------|------------|-------|---------|
| Cache-Aware+LRU | 26680 | 25022 | 23.89% | baseline |
| Cache-Aware+RLT | 19191 | 14332 | 26.36% | +0.58ms淘汰 |
| LBGR+LRU | 6025 | 2958 | 33.33% | +0.56ms路由 |
| LBGR+RLT | 2263 | 1088 | 37.31% | +2ms总开销 |

### 模型规模与架构泛化

| 模型 | 硬件 | vs Cache-Aware+LRU 延迟降低 | vs Cache-Aware+LRU TTFT降低 |
|------|------|---------------------------|---------------------------|
| Llama-3.1-8B | 10×L40 | 11.96× | 14.06× |
| Llama-3.1-70B | 4×H200 | 5.46× | 7.19× |
| Mixtral-8×7B (MoE) | 4×H200 | 显著优于所有baseline | 显著优于所有baseline |

### 关键发现
- 在Llama-3.1-8B(L40 GPU)和70B(H200 GPU)、Mixtral-8×7B上均有效，跨稠密/稀疏架构泛化
- 最坏情况轮询到达顺序下，LBGR+RLT依然大幅领先(22.8×延迟降低、15.5× TTFT降低)
- 缓存大小从50%到90%、worker数从2到10、请求率4-20 req/s均鲁棒
- 单worker实验也验证RLT优于L-LRU：缓存命中率最高提升6.92×，吞吐提升77.4%
- 随着缓存容量增大，LBGR+RLT与baseline的性能差距**进一步拉大**
- worker数≥6时各方法吞吐趋于饱和（请求率固定12 req/s时系统容量充足）

## 亮点
- 首次为KV缓存感知负载均衡建立统一数学模型，填补了理论与实践的空白
- RLT的O(log n)竞争比证明是信息论最优（不可再改进），这是经典在线算法理论在LLM系统中的优雅应用
- 两个算法的额外运行时开销极低（每查询仅~2ms），高度实用，可即插即用到现有系统
- LBGR的指数衰减+在线回归设计简洁有效，避免了复杂RL或预测模型的训练开销
- 实验覆盖4种基准(GSP/ShareGPT/UltraChat/Loogle)、3种前缀共享场景、多种模型规模/架构，非常全面

## 局限性
- 仅评估文本推理，未涉及多模态KV缓存（图像token的缓存特性可能不同）
- 受限于10个worker的单域部署，未验证大规模或异地分布式场景下的网络延迟影响
- 理论分析假设在线对抗到达，实际workload的分布特性（如Zipf分布）可进一步利用以获得更紧的界
- LBGR的线性回归残差模型可能在高度非线性的延迟场景中不够表达力
- 输出token长度实验仅测试到128，更长生成（如4K+）场景未探索

## 相关工作
- **KV缓存压缩**：H2O(高注意力token保留)、StreamingLLM(注意力锚定token)——模型层面压缩，与本文系统层面淘汰策略互补
- **KV缓存系统**：vLLM(分页虚拟内存减碎片)、SGLang RadixAttention(Radix树前缀共享)——本文在SGLang基础上替换淘汰与路由模块
- **负载均衡路由**：SGLang(阈值切换命中率/最少负载路由)、NVIDIA TensorRT-LLM/llm-d(静态线性评分)——均为启发式，缺乏理论保证
- **在线算法理论**：经典marking algorithm(Fiat et al., 1991)为RLT的理论基础；本文将其推广到Radix树结构的叶节点淘汰
- **竞争比分析**：连接了在线算法竞争分析与LLM系统优化，属于"理论指导系统"的研究范式
- 本文首次将缓存淘汰与路由两条线统一建模，是理论驱动的系统优化典范

## 评分 ⭐⭐⭐⭐⭐
- **新颖性**: 5/5 — 首个统一形式化+最优竞争比证明，理论贡献扎实
- **实验充分度**: 5/5 — 4基准×3设置×多模型×多参数扫描，消融彻底
- **写作质量**: 5/5 — 理论与实验衔接清晰，问题动机阐述充分
- **实用性**: 5/5 — 额外开销仅2ms，可直接集成到SGLang等系统

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] QuoKA: Query-Oriented KV Selection for Efficient LLM Prefill](quoka_query-oriented_kv_selection_for_efficient_llm_prefill.md)
- [\[ICLR 2026\] FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](flashdlm_accelerating_diffusion_language_model_inference_via_efficient_kv_cachin.md)
- [\[ICML 2026\] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](../../ICML2026/llm_efficiency/criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)
- [\[ICLR 2026\] Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](overcoming_joint_intractability_with_lossless_hierarchical_speculative_decoding.md)
- [\[ICLR 2026\] Scaling Large Vision-Language Model RL Training via Efficient Load Balancing](scaling_large_vision-language_model_rl_training_via_efficient_load_balancing.md)

</div>

<!-- RELATED:END -->
