---
title: >-
  [论文解读] Quantize What Counts: More for Keys, Less for Values
description: >-
  [ACL 2026][模型压缩][KV缓存量化] 本文从线性代数角度证明 Transformer 中 Key 权重的谱范数和 Frobenius 范数系统性大于 Value 权重，据此提出 Key 优先的混合精度 KV 缓存量化策略（如 K4V2），在减少 25% 内存的同时保持 98.3% 的全精度准确率。
tags:
  - "ACL 2026"
  - "模型压缩"
  - "KV缓存量化"
  - "混合精度"
  - "谱范数"
  - "Key-Value不对称"
  - "LLM推理优化"
---

# Quantize What Counts: More for Keys, Less for Values

**会议**: ACL 2026  
**arXiv**: [2502.15075](https://arxiv.org/abs/2502.15075)  
**代码**: [https://github.com/mohsenhariri/spectral-kv](https://github.com/mohsenhariri/spectral-kv)  
**领域**: 模型压缩  
**关键词**: KV缓存量化, 混合精度, 谱范数, Key-Value不对称, LLM推理优化

## 一句话总结
本文从线性代数角度证明 Transformer 中 Key 权重的谱范数和 Frobenius 范数系统性大于 Value 权重，据此提出 Key 优先的混合精度 KV 缓存量化策略（如 K4V2），在减少 25% 内存的同时保持 98.3% 的全精度准确率。

## 研究背景与动机
**领域现状**: LLM 推理时 KV 缓存是主要内存瓶颈，随着上下文长度（可达 1000 万 token）和模型规模增长，KV 缓存量化成为刚需。

**现有痛点**: 现有 KV 量化方法在 Key 和 Value 之间的比特分配要么固定相同精度（忽略二者差异），要么通过启发式网格搜索调参（缺乏理论基础，不可泛化）。

**核心矛盾**: Key 和 Value 在注意力机制中扮演根本不同的角色，但现有方法缺乏理论指导来决定如何不对称地分配量化比特。

**本文目标**: 从模型权重的内在几何性质出发，建立有理论根据的 KV 混合精度量化策略。

**切入角度**: 分析 Key/Value 投影权重矩阵的谱范数和 Frobenius 范数差异，推导量化误差与范数的关系。

**核心 idea**: Key 权重因同时参与注意力图计算和缓存存储而在训练中累积更大范数，因此量化误差更敏感——应优先为 Key 分配更高精度。

## 方法详解

### 整体框架
本文走的是一条「先证范数差距、再推比特分配、最后说明可叠加」的纯理论路线，不引入任何训练或推理时开销。第一步证明 **Key-Value 范数不均等定理**：训练后 Key 投影权重的 Frobenius 范数系统性大于 Value（$\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$）。第二步把这个差距代入均匀量化的误差界，得到 **Key 优先量化定理**：在固定比特预算下，给 Key 高精度、Value 低精度（如 K4V2）严格最小化总量化误差。第三步证明这套混合精度分配与 QuaRot 等旋转类异常值方法**正交**，可即插即用地叠加进现有量化框架。整套策略只需一次性分析模型权重范数，随后在 11 个模型、6 个数据集、2 种量化后端上验证其有效性。

### 关键设计

**1. Key-Value 范数不均等定理（Theorem 3.1）：Key 权重在训练后系统性地比 Value 更"重"**

以往方法之所以靠网格搜索来分配比特，是因为没人说得清 Key 和 Value 到底差在哪。本文把这件事钉成一个可证的命题：$\mathbb{E}[\|W^K\|_F^2] > \mathbb{E}[\|W^V\|_F^2]$ 在训练后普遍成立。证明从 Xavier 初始化出发追踪 SGD 的梯度更新——关键在于 $W^K$ 一身二职，既要塑造注意力图、又要决定缓存内容，于是 $W^Q$ 在训练中的增长会通过链式法则放大反传到 $W^K$ 的梯度信号，让它的范数不断累积；而 $W^V$ 只影响后注意力表示，没有这种乘法放大。

正是这个不对称的梯度路径，给"Key 更信息密集、更值得保精度"提供了几何层面的依据，而不再是经验观察。

**2. Key 优先量化定理（Theorem 3.2）：在固定比特预算下，给 Key 高精度、Value 低精度严格最小化总量化误差**

有了范数差距，比特该怎么分就能直接算出来。均匀标量量化在 $b$ 比特下的期望 MSE 为 $\Theta(\|M\|_F^2 \cdot 2^{-2b})$，也就是说一个矩阵的范数越大，它对量化精度就越敏感。把这条代入 $\|K\|_F \gg \|V\|_F$ 的事实，等精度分配（如 K2V2）下 Key 的量化误差会主导整个总误差；反过来把省下来的比特从 Value 挪给 Key（如 K4V2），就能严格压低误差上界。

这一步把"给 Key 多分比特"从一句经验直觉，升级成了在固定内存预算下可证最优的几何驱动设计原则。

**3. 与旋转方法的正交组合：混合精度可以直接叠在 QuaRot 这类异常值重分布方法上**

实践中一个现实顾虑是：这套比特分配会不会和已有的 KV 量化技巧冲突。本文证明它和旋转类方法是正交的——QuaRot 对激活施加 Hadamard 旋转来分散异常值，处理的是"值的分布"，而比特分配处理的是"给谁多少位"，两者作用在不同维度。实验据此把比特宽度 × 分组大小 × 旋转策略当成三维设计空间一起扫，验证混合精度能作为即插即用组件叠加进现有量化框架，而不是另起炉灶。

### 损失函数 / 训练策略
本方法为纯后训练量化（PTQ），无需额外训练。量化误差分析基于均匀标量量化的理论界，实验中使用 Optimum Quanto（token-wise）和 HQQ（channel-wise）两种量化后端。

## 实验关键数据

### 主实验（GSM8K 下游准确率，Optimum Quanto）

| 模型 | K2V2 | K2V4 | **K4V2** | K4V4 |
|------|------|------|----------|------|
| Llama-3.2-1B (1-shot) | 0.033 | 0.035 | **0.338** | 0.357 |
| Llama-3.1-8B (1-shot) | 0.511 | 0.547 | **0.752** | 0.754 |
| Phi-4-14B (1-shot) | 0.759 | 0.783 | **0.913** | 0.923 |
| DeepSeek-R1Q-14B (1-shot) | 0.772 | 0.775 | **0.865** | 0.867 |

### 量化误差对比（MMLU，2-bit，MSE ↓）

| 模型 | K₂ 误差 | V₂ 误差 | K/V 误差比 |
|------|---------|---------|-----------|
| Llama-3.2-1B | 4.851 | 0.127 | 38.2× |
| Llama-3.1-8B | 6.003 | 0.187 | 32.1× |
| Llama-3.3-70B | 4.883 | 0.112 | 43.6× |
| Phi-4-14B | 5.929 | 0.657 | 9.0× |
| Mistral-0.3-7B | 4.718 | 0.398 | 11.9× |

### 关键发现
- K4V2 在 1-shot GSM8K 上平均恢复全精度 K4V4 基线的 ~98.3% 准确率，同时减少 25% KV 缓存内存
- K4V2 比 K2V4 在 Llama-3.2-1B 上高出 30 个百分点，在 Phi-4-14B 上高出 16 个百分点
- Key 缓存在相同比特宽度下的量化重建误差是 Value 的 9-44 倍，验证了范数不均等定理
- 将 K4V2 与 QuaRot 的 Key-only 旋转结合，可在 CoQA/GSM8K/EQ-Bench 上超越 K4V4 基线 4.4-18%

## 亮点与洞察
- 首次为 KV 不对称量化提供严格的理论基础（两个定理），将 ad hoc 调参提升为几何驱动原则
- 方法极轻量：仅需一次性分析模型权重范数，无需推理时内省或额外训练
- Key 优先策略在 Llama/Phi/Mistral/Qwen/DeepSeek 五个模型家族上一致有效，泛化性强
- 证明了与旋转方法的正交性，可作为现有量化框架的即插即用模块

## 局限与展望
- 理论分析基于 SGD + Xavier 初始化的假设，对 AdamW 等优化器的严格推导尚未完成
- 实验未覆盖更极端的长上下文场景（如 100K+ token）
- 当前仅验证了 2/4 bit 组合，更细粒度的混合精度（如 3-bit Key + 1.5-bit Value）有待探索
- 对 MoE 架构（如 Mixtral）的适用性未讨论

## 相关工作与启发
- **KIVI / FlashDecoding**: 固定精度 KV 量化方法，本文的混合精度策略可与之叠加
- **KVTuner / SKVQ / QAQ**: 观察到 Key 应分配更多比特，但均缺乏理论解释
- **QuaRot**: 旋转类异常值重分布方法，与本文的比特分配策略正交互补
- 启发：LLM 推理优化中，理解模型内在几何结构比纯经验调参更有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从谱分析角度给出 KV 量化的理论基础，视角新颖且深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 11 个模型、6 个数据集、2 个量化后端，三维消融全面
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但部分符号较密集
- 价值: ⭐⭐⭐⭐⭐ 提供了可直接落地的量化策略，且具理论通用性

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)
- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](../../NeurIPS2025/model_compression/homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)

</div>

<!-- RELATED:END -->
