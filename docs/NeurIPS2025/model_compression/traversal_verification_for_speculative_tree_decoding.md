---
title: >-
  [论文解读] Traversal Verification for Speculative Tree Decoding
description: >-
  [NeurIPS 2025][模型压缩][推测解码] 提出 Traversal Verification，一种从叶节点到根节点的自底向上验证算法，通过考虑整条路径的序列级概率而非单 token 概率来决定接受/拒绝，理论证明无损性和单链最优性，在多种树结构和任务上一致提升接受长度 2.2%-5.7%。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "推测解码"
  - "树解码"
  - "验证算法"
  - "无损推理加速"
  - "序列级接受"
---

# Traversal Verification for Speculative Tree Decoding

**会议**: NeurIPS 2025  
**arXiv**: [2505.12398](https://arxiv.org/abs/2505.12398)  
**代码**: 暂无  
**领域**: 模型压缩  
**关键词**: 推测解码, 树解码, 验证算法, 无损推理加速, 序列级接受

## 一句话总结

提出 Traversal Verification，一种从叶节点到根节点的自底向上验证算法，通过考虑整条路径的序列级概率而非单 token 概率来决定接受/拒绝，理论证明无损性和单链最优性，在多种树结构和任务上一致提升接受长度 2.2%-5.7%。

## 研究背景与动机

推测解码通过草稿模型生成候选 token 树、由目标模型并行验证来加速 LLM 推理。现有树解码验证方法（如 SpecInfer 的 RRS）继承了 vanilla 推测解码的逐 token 验证机制，存在两个关键缺陷：

**序列概率 vs. 单 token 概率不一致**：vanilla 推测解码基于单个 token 的概率比 $\min(1, p(y_i)/q(y_i))$ 决定接受。但序列的联合接受概率应当考虑整条路径的联合分布，逐 token 决策牺牲了全局最优性。例如，父节点 $X_1$ 接受率为 0.5，子节点 $X_3$ 的比率为 $4/3$，token 级接受概率为 $0.5 \times 1 = 0.5$，而序列级概率为 $\min(0.5 \times 4/3, 1) \approx 0.667$——序列级更高。

**自顶向下验证浪费候选**：现有方法从根节点逐层向下验证，一旦父节点被拒绝，所有子节点全部丢弃，即使子节点可能形成更好的序列。

**核心矛盾**：单 token 验证是局部最优的，但在树结构中并非全局最优。

**切入角度**：反转验证方向——从叶节点开始，如果接受则整条路径都被接受；如果拒绝则回溯到兄弟节点或父节点，最大化利用所有候选 token。

## 方法详解

### 整体框架

Traversal Verification 是一个即插即用的验证模块，替换现有推测解码中的验证步骤，不改变草稿生成或其他流程。核心变化：由自顶向下的逐层验证改为自底向上的遍历验证。

### 关键设计

1. **序列级接受率初始化**：对树 $T$ 中的每条链 $\alpha = (X_0, X_1, \ldots, X_{\gamma_\alpha})$，递归计算每个节点的序列级接受率：

    $p_\alpha^{ini}(X_i) = \min\left\{p_\alpha^{ini}(X_{i-1}) \cdot \frac{\mathcal{M}_b(X_i | X^{i-1})}{\mathcal{M}_s(X_i | X^{i-1})}, 1\right\}$

   其中 $p_\alpha^{ini}(X_0) = 1$。这意味着接受率不仅取决于当前 token，还取决于从根到当前节点整条路径的累积概率比。设计动机：序列级概率 $\prod p_i/q_i$ 可能超过单 token 截断后的乘积 $\prod \min(p_i/q_i, 1)$，因为高概率子节点可以"补偿"低概率父节点。

2. **自底向上遍历顺序**：从第一个叶节点开始验证（即最深层最左侧），采样 $\eta \sim U(0,1)$：

    - 若 $\eta < p_\alpha(X_{\gamma_\alpha})$：接受整条路径 $(X_0, \ldots, X_{\gamma_\alpha})$
    - 若拒绝：删除当前叶节点，更新残差分布和草稿分布，转到兄弟节点或回溯到父节点

   遍历顺序示例（图 1 的树）：$X_3 \to X_4 \to X_1 \to X_5 \to X_2$。关键特性：父节点仅在所有子节点都被拒绝后才被验证，最大化了候选利用率。

3. **残差分布更新**：当节点 $X_{\gamma_\alpha}$ 被拒绝时：

    $\mathcal{M}_b'(x | X^{\gamma_\alpha - 1}) = \text{norm}([p_\alpha(X_{\gamma_\alpha-1}) \cdot \mathcal{M}_b(x | X^{\gamma_\alpha-1}) - \mathcal{M}_s(x | X^{\gamma_\alpha-1})]_+)$

   并更新父节点的残余接受率 $p'_\alpha(X_{\gamma_\alpha-1})$，传播到所有共享前缀的兄弟链。这确保了概率质量的正确重分配。

### 理论保证

- **定理 3.3（无损性）**：Traversal Verification 输出的序列分布与目标模型的分布完全相同。证明核心：利用算法的自相似性——每个子树都是整体验证机制的缩小版本，可通过对后代节点数的数学归纳法证明。

- **定理 3.4（单链最优性）**：当树退化为单链时，Traversal Verification 的期望接受长度等于 Block Verification（已知最优），且 $\geq$ 任何其他有效验证算法。

## 实验关键数据

### 主实验：Llama3.2-1B + Llama3.1-8B (Temperature=1)

| 树结构 | Token-level Avg Accept | Traversal Avg Accept | 提升 |
|---|---|---|---|
| Chain (depth=5) | 3.88 | 3.99 | **2.8%** |
| Binary Tree (depth=5) | 4.63 | 4.73 | **2.2%** |
| EAGLE Sparse Tree | 4.49 | 4.60 | **2.4%** |
| Chain Avg Token/s | 51.2 | 52.5 | 2.5% |
| Binary Avg Token/s | 54.0 | 54.9 | 1.7% |
| EAGLE Avg Token/s | 57.3 | 58.5 | 2.1% |

### 消融实验：链深度和树大小的影响

| 链深度 | Token-level Accept | Traversal Accept | 提升 |
|---|---|---|---|
| Depth 2 | 2.55 | 2.58 | 1.2% |
| Depth 4 | 3.47 | 3.56 | 2.6% |
| Depth 6 | 4.10 | 4.23 | 3.2% |
| Depth 8 | 4.50 | 4.67 | **3.8%** |

| 二叉树深度 (节点数) | Token-level Accept | Traversal Accept | 提升 |
|---|---|---|---|
| Depth 2 (7 nodes) | 3.12 | 3.17 | 1.6% |
| Depth 3 (15 nodes) | 3.94 | 4.03 | 2.3% |
| Depth 4 (31 nodes) | 4.63 | 4.73 | 2.2% |
| Depth 5 (63 nodes) | 5.12 | 5.27 | **2.9%** |

### 关键发现

- 树越深越大，Traversal Verification 提升越显著，因为序列级概率补偿效应在长链中更明显
- 高温度下提升更大（temp=1.0 时 2.8%，temp=0.2 时 1.0%），因为概率分布越分散，序列级和 token 级的差异越大
- 在 Llama-68M + Llama2-7B 的较弱草稿模型组合上，提升更显著（平均 3.8%-5.7%）
- 吞吐量提升略低于接受长度提升，因为自底向上遍历引入少量额外计算开销

## 亮点与洞察

- 核心洞察极为优雅：序列联合概率 $\min(\prod r_i, 1)$ 可以大于各 token 截断概率之积 $\prod \min(r_i, 1)$
- 理论证明严谨完整：无损性证明利用自相似性和数学归纳法，单链最优性与 Block Verification 等价
- 即插即用设计使得所有现有推测解码系统都能受益，无需改变草稿生成或模型架构
- 随着推测解码系统趋向更大更深的树（如 Sequoia 用 768 节点），Traversal Verification 的优势将更加显著

## 局限与展望

- 自底向上遍历可能引入额外计算开销（需要维护和更新所有链的接受率），目前吞吐量提升低于理论接受长度提升
- 目前仅用于 temperature sampling，greedy decoding 下效果未知
- 论文未讨论与 EAGLE/Medusa 等学习型草稿方法的协同效果
- 更优化的实现（如并行化遍历计算）可进一步释放性能

## 相关工作与启发

- 与 Block Verification 在单链上等价，但扩展到了任意树结构
- 与 SpecInfer、EAGLE 等正交——它们关注更好的草稿生成，本文关注更好的验证
- 启发：在推测解码的验证环节仍有优化空间，不仅限于改进草稿模型

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 反转验证方向的核心洞察极为巧妙，序列级接受率是本质性改进
- 实验充分度: ⭐⭐⭐⭐ 多模型多任务多树结构，消融覆盖深度/大小/温度
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，直觉示例清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 改进幅度中等（2-6%），但零成本即插即用特性使其实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding](reject_only_critical_tokens_pivot-aware_speculative_decoding.md)
- [\[ACL 2026\] SSSD: Simply-Scalable Speculative Decoding](../../ACL2026/model_compression/sssd_simply-scalable_speculative_decoding.md)
- [\[ICML 2025\] Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding](../../ICML2025/model_compression/gumiho_a_hybrid_architecture_to_prioritize_early_tokens_in_speculative_decoding.md)
- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[AAAI 2026\] Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)

</div>

<!-- RELATED:END -->
