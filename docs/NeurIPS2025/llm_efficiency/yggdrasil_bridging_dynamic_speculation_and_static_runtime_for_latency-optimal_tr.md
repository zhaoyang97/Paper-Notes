---
title: >-
  [论文解读] Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding
description: >-
  [NeurIPS 2025][LLM效率][推测解码] 提出 Yggdrasil，一个延迟最优的推测解码系统，通过 Equal-Growth Tree (EGT) 结构实现编译友好的动态草稿、延迟感知优化目标替代传统 AAL 指标、以及阶段调度运行时减少 CPU-GPU 协调开销，在 A100/A40 上实现了最高 3.98× 的端到端加速。
tags:
  - "NeurIPS 2025"
  - "LLM效率"
  - "推测解码"
  - "树结构草稿"
  - "编译器优化"
  - "延迟优化"
  - "LLM 推理加速"
---

# Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2512.23858](https://arxiv.org/abs/2512.23858)  
**代码**: 无  
**领域**: LLM效率 / 推测解码  
**关键词**: 推测解码、树结构草稿、编译器优化、延迟优化、LLM 推理加速

## 一句话总结

提出 Yggdrasil，一个延迟最优的推测解码系统，通过 Equal-Growth Tree (EGT) 结构实现编译友好的动态草稿、延迟感知优化目标替代传统 AAL 指标、以及阶段调度运行时减少 CPU-GPU 协调开销，在 A100/A40 上实现了最高 3.98× 的端到端加速。

## 研究背景与动机

**领域现状**：推测解码（Speculative Decoding）通过让小模型（drafter）生成候选 token 序列、大模型（verifier）并行验证来加速 LLM 推理，是不改变模型输出分布的无损加速手段。基于树的推测解码进一步扩展了探索空间，可以在一次验证中并行检查多条路径。

**现有痛点**：现有系统存在一个根本性矛盾——**动态草稿与静态运行时的 mismatch**。(1) 动态草稿算法（如 DISCO）每步动态调整树结构和算子形状以最大化 token 接受率，但这与深度学习编译器（如 TorchInductor）依赖的静态计算图假设冲突，无法享受图融合、kernel 调优等编译优化带来的 2-3× 加速。(2) 现有方法通常优化 AAL（平均接受长度），但 AAL 高不等于端到端速度快——当验证 token 数增多时，验证延迟非线性增长，实际的 per-token 加速可能不升反降。

**核心矛盾**：动态性带来的高 AAL 和编译器需要的静态形状之间存在根本 trade-off，没有现有框架能同时实现两者。

**本文目标** (1) 设计一种既能动态适应上下文又能保持静态算子形状的树结构；(2) 用真实延迟而非 AAL 作为优化目标；(3) 消除推测解码中 CPU 逻辑导致的 GPU 空闲气泡。

**切入角度**：将推测解码视为算法-系统协同设计问题，从三个层面同时优化。

**核心 idea**：用等宽增长树 (EGT) 兼顾动态适应性和编译友好性，从算法到运行时全栈优化延迟。

## 方法详解

### 整体框架

Yggdrasil 从三个层面协同优化：(1) **算法层**：EGT 树结构 + 延迟感知目标函数；(2) **编译层**：利用 EGT 的静态形状特性实现 TorchInductor 编译优化；(3) **运行时层**：阶段调度消除 CPU-GPU 协调开销。输入是任意未修改的 LLM drafter-verifier 对，输出是延迟最优的推测解码系统。

### 关键设计

1. **延迟感知优化目标**:

    - 功能：替代传统 AAL 作为树结构选择的优化目标，直接最大化 wall-clock 加速比
    - 核心思路：加速比 $= \frac{AAL(W_d, D_d, W_v) \cdot T_{verifier}(1)}{\sum_{D_d} T_{drafter}(W_d) + T_{verifier}(W_v)}$，其中 $W_d, D_d, W_v$ 分别是草稿宽度、深度和验证宽度。通过硬件 profiling 获取 $T_{drafter}$ 和 $T_{verifier}$ 的延迟曲线，在运行时做出延迟最优的树结构选择
    - 设计动机：AAL 忽略了验证延迟随 token 数非线性增长的事实。实验表明，在验证 token 较多时 AAL 仍在增长但实际加速比已经饱和甚至下降。使用延迟感知目标额外带来 8% 的性能提升

2. **Equal-Growth Tree (EGT)**:

    - 功能：在保持编译器兼容的静态算子形状的同时，实现上下文自适应的树结构
    - 核心思路：EGT 将树结构分解为三个贪心子决策——(a) **深度预测**：轻量级 MLP 预测器从目标模型的 last-token embedding 预测最优草稿深度 $D_d$，预构建所有深度的图消除条件分支；(b) **宽度选择**：在预测深度下选择最大化加速比的宽度 $W_d$，每步等宽增长保证形状固定；(c) **验证剪枝**：草稿完成后用动态规划从树中提取最大化加速比的子树作为验证集。由于每步新增的叶子数固定为 $W_d$，算子形状在编译时确定
    - 设计动机：DISCO 等动态方法虽然 AAL 高但无法编译，Sequoia 等静态方法虽然可编译但对所有输入用同一树结构。EGT 通过"形状静态+结构动态"解决这个矛盾

3. **阶段调度运行时 (Stage-Based Scheduling)**:

    - 功能：通过提前执行和重叠调度减少推测解码中的 CPU-GPU 气泡
    - 核心思路：(a) **Ahead-of-Time Tail Draft**：不等接受结果出来就预先草稿整个候选序列，接受后直接复用；(b) **Ahead-of-Time Head Draft**：将 head draft 提前到上一轮 bonus draft 之后，与接受阶段重叠执行。然后通过 profile-guided 搜索在所有可能的阶段重叠方案中找到最优执行计划
    - 设计动机：推测解码的 CPU 逻辑（接受判断、token 管理）造成 GPU 空闲。通过打破数据依赖并预执行下游阶段，将 CPU 逻辑从关键路径上移除

### 系统实现

Yggdrasil 基于 PyTorch 2.0 和 TorchInductor 编译器，核心抽象是 TokenTree 类，封装 EGT 的语义。系统在编译时完成 drafter/verifier 的图优化和 kernel 调优，运行时通过 TokenTree 管理 KV cache、attention mask 和叶子位置。不需要修改原始 LLM 架构。

## 实验关键数据

### 主实验（端到端 per-token 延迟）

| 硬件 | 模型 | Yggdrasil 加速比 | vs SpecInfer | vs Sequoia | vs vLLM-Spec |
|------|------|-----------------|-------------|------------|-------------|
| A100 | Llama-2-7B + 68M | **3.98×** | +3.37× | +1.54× | +1.32× |
| A100 | Llama-2-13B + 160M | 3.45× | +2.89× | +1.41× | +1.23× |
| A40 | Llama-2-7B + 68M | **2.76×** | +2.31× | +1.13× | +1.04× |

### 消融实验（优化分解，Llama-2-7B）

| 优化阶段 | 累积加速比 | 增量贡献 | 说明 |
|----------|-----------|---------|------|
| O1: EGT 树结构 | 1.0× (baseline) | — | 算法层基线 |
| O2: + 图编译 | 2.775× | **2.775×** | 贡献最大 |
| O3: + 验证剪枝 | 2.970× | 1.07× | 自适应验证数量 |
| O4: + 阶段调度 | 3.593× | 1.21× | 减少 GPU 气泡 |
| O5: + 深度预测 | 3.952× | 1.10× | 上下文自适应深度 |

### 关键发现

- **编译优化是最大贡献者**：单独的图编译就带来 2.775× 加速，说明静态形状兼容性对推测解码至关重要。使用简单但编译友好的 sequence 结构也能超越 Sequoia 等复杂但不可编译的方案
- **延迟目标 vs AAL 目标**：在 C4 数据集上延迟目标额外提供 8% 加速，因为它能更好地捕获验证延迟的非线性特征
- **温度鲁棒性**：Yggdrasil 在不同采样温度下始终优于 Sequoia，平均额外加速 1.49×
- **EGT 参数敏感性**：最优组合为 $D_d=8, W_d=8, W_v=64$，说明树结构的宽度-深度-验证数量三者需要联合优化

## 亮点与洞察

- **算法-系统协同设计思路精辟**：现有工作要么从算法端优化（提高 AAL）要么从系统端优化（编译加速），Yggdrasil 发现这两个方向存在根本矛盾并提出统一解决方案。这种协同设计思路可以迁移到其他需要动态性与编译效率兼顾的场景
- **EGT 的巧妙约束**：通过限制每步增长的叶子数为常数，将树结构参数化为三个可解耦的子问题，既保持了编译时的静态性又允许运行时的动态性。这是一个优雅的工程-算法折中
- **从 branch prediction 类比到推测解码**：论文将推测解码类比为 CPU 的分支预测和 cache prefetching，ahead-of-time 执行的思路直接借鉴了推测执行

## 局限与展望

- **仅适用于单请求延迟优化**：Yggdrasil 假设单个请求独占 GPU，不适用于 batch serving 场景。与 batch scheduler 的联合优化是重要的开放问题
- **需要 offline profiling**：深度预测器和阶段调度策略需要针对每个 drafter-verifier pair 和硬件做 offline 训练/搜索
- **不支持 model-invasive 方法**：Medusa、Eagle 等修改目标模型的方法不在支持范围内
- 未来可以探索将 EGT 与 self-speculative decoding（如 LayerSkip）结合

## 相关工作与启发

- **vs Sequoia**：Sequoia 通过 dataset-adaptive 树结构平衡 AAL 和开销，但对所有输入用同一树，且不可编译；Yggdrasil 的 EGT 逐请求动态选择且保持编译兼容
- **vs vLLM-Spec**：vLLM 利用编译优化但只支持 sequence 结构（非树），AAL 受限；Yggdrasil 兼具树结构的高 AAL 和编译优化的低延迟
- **vs DISCO**：DISCO 用完全动态树获得最高 AAL，但运行时开销极大无法编译；EGT 在动态性和编译性之间找到了最佳平衡点

## 评分

- 新颖性: ⭐⭐⭐⭐ 算法-系统协同设计角度新颖，EGT 和阶段调度都有独到之处
- 实验充分度: ⭐⭐⭐⭐ 多硬件、多模型、详细的优化分解，但只测了 Llama-2 系列偏少
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，图表设计好，逻辑链完整
- 价值: ⭐⭐⭐⭐ 推测解码系统优化的重要工作，3.98× 加速实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)
- [\[ICLR 2026\] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding](../../ICLR2026/llm_efficiency/bridging_draft_policy_misalignment_group_tree_optimization_for_speculative_decod.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](../../ACL2025/llm_efficiency/tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)

</div>

<!-- RELATED:END -->
