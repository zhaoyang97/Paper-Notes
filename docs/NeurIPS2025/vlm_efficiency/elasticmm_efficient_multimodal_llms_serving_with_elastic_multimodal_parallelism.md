---
title: >-
  [论文解读] ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism
description: >-
  [NeurIPS 2025 Oral][多模态VLM][MLLM推理服务] 提出弹性多模态并行（EMP）范式和 ElasticMM 系统，通过模态感知负载均衡和弹性分区调度将多模态推理的不同阶段解耦到独立实例，相比 vLLM TTFT 降低最高 4.2 倍、吞吐量提升 3.2-4.5 倍。 MLLM 推理比纯文本 LLM 复…
tags:
  - "NeurIPS 2025 Oral"
  - "多模态VLM"
  - "MLLM推理服务"
  - "弹性并行"
  - "模态感知调度"
  - "推理解耦"
  - "资源分配"
---

# ElasticMM: Efficient MLLM Serving with Elastic Multimodal Parallelism

**会议**: NeurIPS 2025 Oral  
**arXiv**: [2507.10069](https://arxiv.org/abs/2507.10069)  
**代码**: 无（即将开源）  
**领域**: 多模态VLM  
**关键词**: MLLM推理服务, 弹性并行, 模态感知调度, 推理解耦, 资源分配

## 一句话总结

提出弹性多模态并行（EMP）范式和 ElasticMM 系统，通过模态感知负载均衡和弹性分区调度将多模态推理的不同阶段解耦到独立实例，相比 vLLM TTFT 降低最高 4.2 倍、吞吐量提升 3.2-4.5 倍。

## 研究背景与动机

MLLM 推理比纯文本 LLM 复杂得多：(1) 额外组件（视觉编码器、跨注意力层）增加架构复杂度；(2) 多模态数据编码后拼接文本提示，大幅增加上下文长度（论文展示 multimodal 请求平均 prompt 长度远超 text-only）。

现有推理系统（vLLM、SGLang、DeepSpeed）采用紧耦合架构，存在两层问题：

- **服务层耦合**：将文本请求和多模态请求统一处理，但它们资源需求差异巨大
- **基础设施层耦合**：预处理器、视觉编码器、LLM 后端共置于同一硬件，共享计算和内存资源

这导致：多模态重负载下 TTFT 急剧增加，编码器-解码器架构（如 LLaMA3.2-Vision）的混合批处理效率低下，静态资源分配无法适应突发流量。

## 方法详解

### 整体框架

ElasticMM 采用两级层次化调度框架：

1. **模态级**：实例按服务模型的模态分组（text-only vs multimodal），通过模态感知负载均衡动态分配资源
2. **阶段级**：推理流水线进一步拆分为编码（encoding）、预填充（prefill）、解码（decode）三个独立阶段，通过弹性分区调度实现每个阶段的独立并行度调整

每一级都提供**解耦**和**弹性**两个核心能力。

### 关键设计

#### 模态感知负载均衡

结合主动和反应两种机制：

**主动机制**：利用长期负载的可预测性（夜间低、白天高），将空闲弹性实例预分配给各模态组。目标是最大化所有组的最小突发容忍度（burst tolerance）：

$$bt(i) = \frac{N_i^{\text{peak}}}{N_i^{\text{avg}}}$$

采用贪心策略迭代将每个实例分配给当前突发容忍度最低的组。

**反应式扩缩**：应对不可预测的短期突发（如突然涌入的图像请求）。系统评估组内并行度调整 vs 组间反应式扩缩的收益-成本权衡，选择最优策略。当一个实例被抢占时，其工作负载迁移到同阶段的其他实例。

#### 弹性分区调度

针对组内请求调度和并行度调整，解决三个子问题：

**请求分派**：FCFS 策略从队列中选择预填充请求集 $R_p$，受 GPU 内存和计算吞吐量两类约束。当系统从内存受限转为计算受限时，继续添加请求会降低性能。

**阶段分配**：为 $R_p$ 分配弹性实例集 $E_p$，优先分配空闲实例，不够时可抢占解码阶段实例。使用增益-成本模型评估抢占收益：

$$\text{Gain} = \sum_{r \in R_p} \frac{T(R_p, E_p) - T(R_p, E_p \cup e_{\max})}{r.\text{input\_len}}$$
$$\text{Cost} = \sum_{r \in B_d} \frac{M(e_{\max}) + w \cdot L(B_d, E_d - e_{\max})}{r.\text{output\_len}}$$

可调惩罚因子 $w$ 控制抢占激进程度。阶段内优先使用数据并行（DP），因为弹性扩缩时只需迁移 KV 缓存，避免昂贵的权重传输。

**弹性自动扩缩**：监控解码阶段触发扩缩。由于解码阶段可扩展性差，先缩减到最小并行度；资源不足时从组内预填充实例或组间实例中选择抢占候选，使用类似的增益-成本模型选择最优实例。

#### 统一多模态前缀缓存

针对真实场景中的请求冗余（如相同系统提示、重复图像），构建统一缓存方案：

- 缓存池 1：多模态输入编码后的 token
- 缓存池 2：统一序列（多模态+文本 token）的前缀 token

通过哈希匹配跳过重复编码，通过前缀树查找最长匹配前缀跳过重复预填充。两个缓存池都使用 LRU 动态淘汰策略。

#### 非阻塞编码

将图像预处理和编码隔离到独立进程/实例异步执行，打破编码和预填充之间的阻塞依赖，降低 TTFT 并提升整体吞吐。

### 训练策略

ElasticMM 是推理系统，不涉及训练。它基于 vLLM 构建，兼容 decoder-only（如 Qwen2.5-VL）和 encoder-decoder（如 LLaMA3.2-Vision）两类架构。

## 实验关键数据

### 主实验

实验在 8 张 NVIDIA A800 80GB GPU 上进行，使用 LLaMA3.2-Vision-11B 和 Qwen2.5-VL-7B 两个模型，ShareGPT-4o 和 VisualWebInstruct 两个数据集。

**表1：TTFT 降低倍数（相对 vLLM）**

| 数据集 | Qwen2.5-VL (DecOnly) | LLaMA3.2-Vision (EncDec) |
|--------|---------------------|-------------------------|
| ShareGPT-4o | 4.2× | 3.5× |
| VisualWebInstruct | 3.7× | 2.9× |

**表2：最大吞吐量提升（SLO 约束下，相对 vLLM）**

| 数据集 | Qwen2.5-VL | LLaMA3.2-Vision |
|--------|-----------|----------------|
| ShareGPT-4o | 4.5× | 3.2× |
| VisualWebInstruct | 约 3.5× | 约 2.8× |

ElasticMM 相比 DistServe（静态解耦基线）也有 2.3 倍吞吐量优势。

### 消融实验

**EMP 有效性**：三种静态资源分配策略（文本优先、均分、多模态优先）都不如 EMP 动态调度。ElasticMM 在 Qwen-2.5-VL 和 LLaMA-3.2-Vision 上分别比最佳静态策略提升 1.8 倍和 2.3 倍吞吐量。

**推理优化有效性**：
- 仅 EMP → 对 TTFT 改善有限
- +统一多模态前缀缓存 → 显著降低冗余计算和数据传输的延迟
- +非阻塞编码 → 进一步消除编码阶段对后续阶段的阻塞，延迟再降

两项优化在大多数请求上提供一致的性能提升。

### 关键发现

1. Decoder-only 模型（Qwen2.5-VL）比 encoder-decoder 模型获益更大，因为其预填充计算更重，与编码的冲突更严重
2. 视觉密集型数据集（ShareGPT-4o 更高分辨率图像）上 ElasticMM 优势更明显
3. 静态资源分配无论偏向哪种模态都是次优的，弹性调度是唯一可行的解
4. 解码阶段可扩展性差，应缩减到最小并行度后再按需扩展

## 亮点与洞察

- 两级解耦+弹性的设计思路优雅：模态级隔离文本和多模态请求，阶段级分离编码/预填充/解码，每级都支持动态资源调整
- 增益-成本模型为抢占决策提供了量化框架，避免启发式规则
- 统一前缀缓存将文本缓存和多模态缓存整合，用哈希+前缀树实现高效匹配
- 非阻塞编码虽然概念简单，但在实际推理中效果显著（编码延迟通常是预填充的 5 倍以上）

## 局限与展望

- 目前仅在单节点（8 GPU）上验证，多节点分布式场景的通信延迟和并行策略搜索空间是开放问题
- 弹性扩缩的增益-成本模型依赖离线 profiling 确定阈值，不同硬件/模型需重新标定
- 未评估在超大模型（如 72B）上的表现
- 反应式扩缩的惩罚因子 $w$ 需要手动调参
- 未与 ModServe 等最新多模态推理系统做直接对比

## 相关工作与启发

- DistServe/Splitwise 提出预填充-解码分离，但采用静态分配；ElasticMM 在此基础上增加了弹性调度
- LoongServe 引入弹性序列并行，ElasticMM 将弹性思想扩展到多模态维度
- FlashAttention/Flash-Decoding 等算子级优化与 ElasticMM 正交，可叠加使用
- 模态组隔离 + 阶段分离的双层架构对未来 MLLM 服务设计有参考价值

## 评分

- 新颖性：⭐⭐⭐⭐ — 两级弹性范式是对 MLLM 推理架构的系统性创新
- 技术深度：⭐⭐⭐⭐ — 增益-成本模型、弹性调度算法设计扎实
- 实验充分度：⭐⭐⭐⭐ — 多模型、多数据集、多消融，但受限于单节点
- 实用价值：⭐⭐⭐⭐⭐ — 4.2 倍 TTFT 降低对在线服务有巨大意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)
- [\[ECCV 2024\] Efficient Inference of Vision Instruction-Following Models with Elastic Cache](../../ECCV2024/multimodal_vlm/efficient_inference_of_vision_instruction-following_models_with_elastic_cache.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](../../CVPR2025/multimodal_vlm/efficient_motion-aware_video_mllm.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ICCV 2025\] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting](../../ICCV2025/multimodal_vlm/physsplat_efficient_physics_simulation_for_3d_scenes_via_mllm-guided_gaussian_sp.md)

</div>

<!-- RELATED:END -->
