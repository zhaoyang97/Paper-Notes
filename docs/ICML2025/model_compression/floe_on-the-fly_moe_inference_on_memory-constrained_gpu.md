---
title: >-
  [论文解读] FloE: On-the-Fly MoE Inference on Memory-constrained GPU
description: >-
  [ICML2025][模型压缩][MoE推理] 提出 FloE，一个面向消费级 GPU 的 MoE 即时推理系统，通过专家内部混合压缩（上下文稀疏化 + 超低比特量化）和双预测器实现计算-传输流水线化，在 RTX 3090 上仅 11GB 显存即可部署 Mixtral-8×7B，相比 DeepSpeed-MII 加速 48.7 倍，性能仅下降 4.4%~7.6%。
tags:
  - "ICML2025"
  - "模型压缩"
  - "MoE推理"
  - "专家卸载"
  - "激活稀疏性"
  - "超低比特量化"
  - "GPU内存受限"
  - "推理加速"
---

# FloE: On-the-Fly MoE Inference on Memory-constrained GPU

**会议**: ICML2025  
**arXiv**: [2505.05950](https://arxiv.org/abs/2505.05950)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: MoE推理, 专家卸载, 激活稀疏性, 超低比特量化, GPU内存受限, 推理加速

## 一句话总结
提出 FloE，一个面向消费级 GPU 的 MoE 即时推理系统，通过专家内部混合压缩（上下文稀疏化 + 超低比特量化）和双预测器实现计算-传输流水线化，在 RTX 3090 上仅 11GB 显存即可部署 Mixtral-8×7B，相比 DeepSpeed-MII 加速 48.7 倍，性能仅下降 4.4%~7.6%。

## 研究背景与动机
MoE 模型（如 DeepSeek-R1、Mixtral）通过稀疏激活降低推理计算量，但大量未激活专家占用巨额显存。Mixtral-8×7B FP16 需 94GB 显存，其中 70% 是未激活专家的"浪费"。

**专家卸载（Expert Offloading）** 将专家参数存储在 CPU 内存、按需加载到 GPU，是自然的解决方案。但核心瓶颈在于：
- PCIe 4.0 带宽仅 32GB/s，远低于 GPU 内部 HBM 带宽（300GB/s）
- 单个 Mixtral 专家 300MB FP16 参数，传输需 ~15ms，计算仅需 ~5ms
- 已有工作采用统一超低比特量化减少传输，但严重损害生成质量

**核心问题**：如何在不显著损害性能的前提下，将专家 I/O 开销隐藏在模型计算中，实现即时推理？

## 方法详解

### 整体框架
FloE 由三个核心模块组成：专家混合压缩、稀疏性预测器、系统级协同优化。

### 1. 专家混合压缩（Hybrid Compression）
**核心发现**：MoE 专家的三个投影矩阵（gate/up/down）对压缩的敏感度不同。

#### 上下文稀疏化（Contextual Sparsification）—— 用于 gate 和 down 投影
- **观察**：MoE 专家内部存在大量激活稀疏性（up 投影输出的绝对值集中在零附近）
- **理论证明**：按 down 投影输入的幅度剪枝误差最小，up 投影次之，gate 投影最敏感
- **设计选择**：基于 up 投影输出幅度生成 mask，对 gate 和 down 投影进行通道级稀疏化
- 90% 稀疏度下 Mixtral 困惑度仅增加约 0.5%

稀疏化前向传播：$\mathbf{a}^S(x) = (\text{SiLU}(\mathbf{x}\mathbf{W}^{\text{gate}}) \odot S_t(\mathbf{x}\mathbf{W}^{\text{up}})) \mathbf{W}^{\text{down}}$

#### 超低比特量化（Ultra-low-bit Quantization）—— 用于 up 投影
- **观察**：up 投影对量化最不敏感（INT2 时困惑度仅为 gate 的 46%、down 的 27%）
- **分析**：MLP 可视为 key-value 记忆模型，up/gate 为 key 选择激活的 value（down），up 作为 key 容忍更大量化噪声
- 使用 HQQ INT2 量化 up 投影

**最终压缩效果**：每个专家参数压缩 9.3 倍。

### 2. 双稀疏性预测器

#### 专家间预测器（Inter-expert, learning-based）
- 利用当前层隐状态预测下一层应激活的专家
- 相邻层隐状态余弦相似度 > 0.95（首层除外）
- 浅层用单层 MLP（32K 参数），深层用两层 MLP（2M 参数）
- 平均精度 0.88

#### 专家内预测器（Intra-expert, reuse-based）
- 用当前层隐状态乘以下一层的 up 投影矩阵（复用），估计稀疏分布
- 无额外参数，无额外内存开销
- 平均召回率 0.95

### 3. 系统协同优化
- **高效稀疏 GEMV Kernel**：基于 Triton 实现列优先存储 + 选择性列加载
- **紧凑异步传输**：gate 和 down 投影列对齐存储、AVX-512 指令 + 多线程打包 + 多流异步传输
- 传输带宽利用率达峰值的 88%，比 PyTorch 原生实现快 12.6 倍

## 实验关键数据

### 端到端推理速度（RTX 3090, 12GB VRAM）

| 方法 | 相对 Mixtral-GPU 速度 | 加速比 |
|------|----------------------|--------|
| DeepSpeed-MII | 极慢（FP16 卸载） | 1× |
| Mixtral-Offloading | - | 18.7× |
| Fiddler | - | 15.5× |
| **FloE** | **91% Mixtral-GPU** | **48.7×** |

### 部署能力

| 指标 | 数值 |
|------|------|
| 最低 VRAM 需求 | 11GB |
| 内存占用压缩 | 8.5× |
| 每专家参数压缩 | 9.3× |
| 性能下降 | 4.4%~7.6% |

### 下游任务（7 个 zero/few-shot 任务平均准确率）
- FloE-W^up（仅稀疏化）在 80% 稀疏度下比 CATS 高 2.8%，90% 稀疏度下高 9.8%
- FloE（稀疏化+量化）性能仍高于 HQQ INT3 和 CHESS
- 稀疏性和量化误差基本独立可加

### 稀疏 Kernel 加速（RTX 3090 单专家）

| 稀疏度 | 加速比 |
|--------|--------|
| 50% | 1.43× |
| 70% | 1.72× |
| 90% | 1.92× |

## 亮点与洞察
1. **矩阵级差异化压缩策略**是核心创新：up 投影量化 + gate/down 投影稀疏化，比统一量化大幅保质
2. **双预测器设计**精巧：inter-expert 有参数但轻量，intra-expert 无参数零开销，合力实现计算-传输流水线
3. **理论与实践配合**：定理证明 down 输入剪枝误差最优，实验验证 up 输出次优但兼顾传输可预测性
4. **系统工程完整**：从算法到 Kernel 到传输协议全栈优化，端到端加速真实可感知
5. **消费级 GPU 可部署**：11GB VRAM 即可运行 Mixtral-8×7B，极大降低 MoE 使用门槛

## 局限与展望
1. 仅在 Mixtral-8×7B 上做了完整评估，对 DeepSeek-V2/V3 等更新 MoE 架构的泛化性有待验证
2. 当前仅支持 single-batch 推理（延迟敏感场景），高吞吐批量推理未覆盖
3. inter-expert 预测器需要训练，扩展到新模型时有额外成本
4. 稀疏度-性能权衡在 MMLU 等知识密集型任务上下降更明显

## 相关工作与启发
- **Mixtral-Offloading (Eliseev & Mazur, 2023)**：统一量化+预测+缓存的先驱方案
- **CATS (Lee et al., 2024a)**：密集 LLM 的激活稀疏化方法
- **Fiddler (Kamahori et al., 2024)**：CPU-GPU 协同执行方案
- 启示：MoE 推理优化应从"专家内部"挖掘冗余，而非仅关注专家间路由

## 评分
- 新颖性: ⭐⭐⭐⭐ （混合压缩+双预测器的系统级创新）
- 实验充分度: ⭐⭐⭐⭐ （效率+效果双维度，多 GPU、多任务验证）
- 写作质量: ⭐⭐⭐⭐ （图表丰富，动机-观察-设计逻辑链清晰）
- 价值: ⭐⭐⭐⭐⭐ （让消费级 GPU 可即时运行 MoE，实用价值极高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds](../../ICML2026/model_compression/memory-efficient_partitioned_dnn_inference_on_resource-constrained_android_crowd.md)
- [\[CVPR 2026\] Otil: Accelerating Diffusion Model Inference via Communication-Efficient Multi-GPU Parallelism](../../CVPR2026/model_compression/otil_accelerating_diffusion_model_inference_via_communication-efficient_multi-gp.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](../../NeurIPS2025/model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[ICML 2025\] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams](lego_sketch_a_scalable_memory-augmented_neural_network_for_sketching_data_stream.md)

</div>

<!-- RELATED:END -->
