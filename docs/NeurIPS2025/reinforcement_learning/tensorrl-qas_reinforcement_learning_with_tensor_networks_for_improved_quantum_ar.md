---
title: >-
  [论文解读] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search
description: >-
  [NeurIPS 2025][强化学习][Quantum Architecture Search] 提出 TensorRL-QAS 框架，通过用张量网络（MPS/DMRG）对强化学习量子架构搜索进行 warm-start，显著降低电路深度和 CNOT 门数量（最高 10 倍），同时加速训练（最高 98%），有效解决了 RL-QAS 在大规模量子系统上的可扩展性瓶颈。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "Quantum Architecture Search"
  - "Tensor Networks"
  - "DMRG"
  - "Variational Quantum Algorithms"
---

# TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search

**会议**: NeurIPS 2025  
**arXiv**: [2505.09371](https://arxiv.org/abs/2505.09371)  
**代码**: [https://github.com/Aqasch/TensorRL-QAS](https://github.com/Aqasch/TensorRL-QAS)  
**领域**: 量子计算 / 强化学习  
**关键词**: Quantum Architecture Search, Reinforcement Learning, Tensor Networks, DMRG, Variational Quantum Algorithms

## 一句话总结

提出 TensorRL-QAS 框架，通过用张量网络（MPS/DMRG）对强化学习量子架构搜索进行 warm-start，显著降低电路深度和 CNOT 门数量（最高 10 倍），同时加速训练（最高 98%），有效解决了 RL-QAS 在大规模量子系统上的可扩展性瓶颈。

## 研究背景与动机

变分量子算法（VQA）是 NISQ 时代的主流范式，其核心是设计参数化量子线路（PQC）来最小化目标哈密顿量的期望值。然而，PQC 的架构设计面临两难：线路太浅不够表达，太深则噪声影响严重。

量子架构搜索（QAS）旨在自动发现最优 PQC 结构，其中 RL-based QAS 展现了良好前景——RL agent 通过选择门和位置逐步构建线路。但 **RL-QAS 面临严重的可扩展性问题**：到目前为止，多数方法仅在无噪声场景下验证到 8 qubit，有噪声场景仅 4 qubit。根本原因是：(1) action space 随 qubit 数爆炸增长；(2) episode 需要更多步骤和更大量子模拟开销。

**核心 idea**：利用张量网络方法（DMRG）获取目标态的近似解作为 warm-start，将搜索空间收缩至物理有意义的线路区域，大幅加速 RL-QAS 的收敛。

## 方法详解

### 整体框架

TensorRL-QAS 由三个步骤组成：

1. **DMRG 求解**：给定目标哈密顿量 $H$，用 DMRG 以最大 bond dimension $\chi$ 找到 MPS 近似基态 $|\Psi\rangle$
2. **MPS → PQC 映射**：通过 Stiefel 流形上的 Riemannian 优化，将 MPS 转化为 brickwork 结构的量子线路 $U|0\rangle \approx |\Psi\rangle$
3. **RL-QAS 精化**：在 warm-start 线路基础上，RL agent 继续添加门 $VU|0\rangle$ 以进一步降低能量

### 关键设计

1. **MPS 到量子线路的映射**：将 MPS 的 overlap $|\langle\Psi|U|0\rangle|$ 表示为张量网络的缩并，通过自动微分计算梯度，使用 Cayley 变换实现 Stiefel 流形上的 Riemannian Adam 优化。这保证了 2-qubit 酉矩阵 $U_k \in U(4)$ 在优化过程中始终满足酉性约束。brickwork 结构天然适配线性连接的量子硬件。

2. **TensorRL (trainable TN-init)**：将 warm-start 线路的结构和参数通过 binary encoding 编码为 RL state，agent 可以在训练中修改 TN 部分的参数。RL state 大小为 $(D_{\text{MPS}} + D) \times N \times (N + N_{\text{1-qubit}})$。优点是完整信息可见，但 state 较大导致训练较慢。

3. **TensorRL (fixed TN-init)**：不将 warm-start 线路编码到 RL state，而是作为固定的初始 statevector。RL state 大小缩减为 $D \times N \times (N + N_{\text{1-qubit}})$。三重加速效果：(i) statevector 模拟更快；(ii) 神经网络输入更小；(iii) 可训练参数更少，经典优化器函数调用减少。

### 损失函数 / 训练策略

- 使用 DDQN（Double Deep Q-Network）+ 5-step trajectory roll-outs
- 折扣因子 $\gamma = 0.88$，$\epsilon$-greedy 探索（$\epsilon$ 从 1 衰减到 0.05）
- 奖励函数与目标能量的化学精度相关
- Action space: $\{RX, RY, RZ, CNOT\}$
- 默认 bond dimension $\chi = 2$
- 每个 episode 最大步数仅为 baseline 的一半

## 实验关键数据

### 主实验（无噪声场景，Table 1）

| 分子系统 | 方法 | Error | CNOT | Depth |
|----------|------|-------|------|-------|
| 8-H₂O | TensorRL (fixed) | 8.9×10⁻⁴ | **9** | **6** |
| 8-H₂O | CRLQAS | 1.8×10⁻⁴ | 105 | 75 |
| 8-H₂O | Vanilla RL | 1.7×10⁻⁴ | 117 | 96 |
| 10-H₂O | TensorRL (fixed) | 4.1×10⁻⁴ | **15** | **17** |
| 10-H₂O | Vanilla RL | 2.5×10⁻⁴ | 96 | 73 |
| 12-LiH | TensorRL (trainable) | **1.0×10⁻²** | **37** | **31** |
| 12-LiH | Vanilla RL | 2.2×10⁻² | 321 | 140 |

### 消融实验（噪声场景，Table 2，8-H₂O）

| 噪声类型 | 方法 | Error | CNOT | 成功率 |
|----------|------|-------|------|--------|
| Depolarizing | TensorRL (fixed) | **9.0×10⁻⁴** | 5 | **100%** |
| Depolarizing | CRLQAS (rerun) | 1.3×10⁻³ | 11 | 30% |
| Shot noise | TensorRL (trainable) | **8.7×10⁻⁵** | 28 | 100% |

### 关键发现

- TensorRL (fixed) 将函数评估次数减少 **100 倍**，每 episode 执行时间减少 **98%**
- 随系统规模增长，CNOT 门削减倍数从 6-qubit 的 5-9 倍增长到 12-qubit 的 10-13 倍
- TensorRL 在所有种子上 100% 达到化学精度，而 Vanilla RL 在 10-H₂O 上仅 70%
- 在 20-qubit TFIM 模型上也展示了有效性（附录实验）

## 亮点与洞察

- **TN + RL 的互补性**：DMRG 在低 bond dimension 时便能提供足够好的初始化，RL 则在此基础上发现更紧凑的线路，两者单独都无法达到的效果
- **Fixed TN-init 的意外优势**：虽然信息更少，但由于 state 更小、参数更少，反而在实践中表现最好，且产生最紧凑的线路
- **CPU-only 可行性**：TensorRL (fixed) 在 8-qubit 以内可在 CPU 上高效训练，降低了量子架构搜索的硬件门槛

## 局限与展望

- Action space 限为 $\{RX, RY, RZ, CNOT\}$，更大系统可能需要更丰富的门集
- 仅使用 brickwork warm-start 结构，其他张量网络拓扑（如 MERA）未探索
- 实际量子硬件上的训练尚未进行，目前依赖噪声模拟
- Bond dimension $\chi=2$ 是默认选择，对更复杂分子可能需要更大的 $\chi$

## 相关工作与启发

- 继承了 Rudolph et al. (2023) 的 TN-VQA 协同思想，但首次将其与 RL-QAS 结合
- 与 CRLQAS (curriculum learning) 和 SA-QAS (模拟退火) 等非 RL 方法形成互补
- TN warm-start 思路可推广到其他 RL-based 量子优化问题

## 评分

- 新颖性: ⭐⭐⭐⭐ TN + RL 的结合思路自然但实现细节丰富，fixed/trainable 两种模式的对比有洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 6-12 qubit 多个分子、有噪声/无噪声场景、多个 baseline、成功率统计
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示丰富，但正文略长
- 价值: ⭐⭐⭐⭐ 显著推进了 RL-QAS 的可扩展性，是量子线路设计的 SOTA 框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[ICML 2025\] Benchmarking Quantum Reinforcement Learning](../../ICML2025/reinforcement_learning/benchmarking_quantum_reinforcement_learning.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](a_theory_of_multi-agent_generative_flow_networks.md)
- [\[NeurIPS 2025\] Meta-World+: An Improved, Standardized, RL Benchmark](meta-world_an_improved_standardized_rl_benchmark.md)
- [\[NeurIPS 2025\] Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation](decoderhybriddecoder_architecture_for_efficient_reasoning_wi.md)

</div>

<!-- RELATED:END -->
