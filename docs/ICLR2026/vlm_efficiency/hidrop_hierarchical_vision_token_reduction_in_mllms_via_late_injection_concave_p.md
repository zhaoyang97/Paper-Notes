---
title: >-
  [论文解读] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit
description: >-
  [ICLR 2026][多模态VLM][视觉token压缩] 提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "视觉token压缩"
  - "MLLM加速"
  - "渐进式剪枝"
  - "Late Injection"
  - "扩散注意力"
---

# HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit

**会议**: ICLR 2026  
**arXiv**: [2602.23699](https://arxiv.org/abs/2602.23699)  
**代码**: [https://github.com/EIT-NLP/HiDrop](https://github.com/EIT-NLP/HiDrop)  
**领域**: 多模态VLM  
**关键词**: 视觉token压缩, MLLM加速, 渐进式剪枝, Late Injection, 扩散注意力

## 一句话总结

提出 HiDrop 框架，通过对 MLLM 不同层的功能进行深入分析（浅层=传播器、中层=融合中心、深层=语言推理），设计了 Late Injection（跳过浅层）+ Concave Pyramid Pruning（凹金字塔中层剪枝）+ Early Exit（深层退出）三阶段策略，压缩约 90% 视觉 token 且几乎不损失性能，训练加速 1.72×。

## 研究背景与动机

**领域现状**：MLLM（如 LLaVA）处理视觉 token 的计算开销随 token 数量二次增长。视觉编码器产生的 token 远多于文本（如 576 个 patch token），成为推理和训练的主要瓶颈。

**现有痛点**：已有视觉 token 剪枝方法存在两个核心误解：(a) 错误认为浅层是关键的多模态融合层，必须保留密集视觉 token；实际上浅层对视觉 token 几乎不做处理，只是被动传播。(b) 采用固定比例的金字塔/线性剪枝调度（如 FastV、PDrop），忽略了不同层信息流的非均匀性。

**核心矛盾**：如何让 token 剪枝策略与模型内部层级处理动态真正匹配？

**本文目标**：设计与 MLLM 层级功能对齐的 token 管理策略——浅层不需要处理视觉 token（直接跳过）、中层是融合冗余最高的地方（激进剪枝）、深层已完成融合（直接丢弃）。

**切入角度**：先做系统的层级行为分析（intra-modal similarity + cross-modal influence），用数据驱动发现取代启发式假设。

**核心 idea**：根据 MLLM 层级功能分工（传播/融合/推理），在正确的位置做正确的事——晚注入、猛剪枝、早退出。

## 方法详解

### 整体框架

HiDrop 的出发点是：视觉 token 在 LLM 的不同深度根本不是"一直需要"的，所以管理策略应该跟着层级功能走，而不是从第一层到最后一层都密集保留。它先做了一轮层级行为分析，把 LLaVA 的 32 层切成三段功能——浅层只是被动传播视觉信息、中层是真正做跨模态融合的地方、深层已经退化成纯语言推理——然后在每一段做不同的事。

具体来说，视觉 token 在浅层（约 Layer 1~8）压根不进入序列（Late Injection），到中层（约 Layer 9~24）才注入并在几个选定的 filtering layer 上用可微 Top-K 前快后慢地渐进剪枝（Concave Pyramid Pruning），进入深层（约 Layer 25~32）后剩余视觉 token 被一次性全部丢弃（Early Exit）。三段拼起来等于给视觉 token 开了一个只覆盖大约一半层数的"处理窗口"，窗口外的层完全不为视觉信息买单。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    IMG["输入图像"] --> ENC["视觉编码器<br/>576 个视觉 token"]
    TXT["输入文本"] --> SHALLOW["浅层 L1~8<br/>纯文本前向<br/>视觉 token 不注入"]
    ENC -.->|"并行编码<br/>(工程优化)"| INJ
    SHALLOW --> INJ["Late Injection<br/>L_inj=9 注入全部视觉 token"]
    INJ --> PRUNE["Concave Pyramid Pruning + ILVAS<br/>中层 L9~24 前快后慢渐进剪枝<br/>ILVAS 选 filtering layer + DTop-K"]
    PRUNE --> EXIT["Early Exit<br/>L_exit=25 丢弃全部视觉 token"]
    EXIT --> DEEP["深层 L25~32<br/>纯语言推理"]
    DEEP --> OUT["输出答案"]
```

### 关键设计

**1. Late Injection：浅层根本不该处理视觉 token，所以干脆不注入**

层级分析里有两个直接证据：浅层视觉 token 的 intra-modal cosine similarity 极高，说明它们经过这些层几乎不变化；同时 cross-modal influence 近零，说明文本表示在浅层基本不受图像影响。两者合起来意味着浅层对视觉信息只是被动搬运。既然如此，HiDrop 干脆让视觉 token 在第 $L_{inj}=9$ 层才注入序列，前面 8 层只跑文本。这和"先把视觉 token 全注入、再想办法剪"的传统路线根本不同——别人是注入后做减法，HiDrop 是从源头就不让视觉 token 经过浅层，计算从一开始就省下来了。

**2. Concave Pyramid Pruning + ILVAS：在中层融合区前快后慢地渐进剪枝**

视觉 token 注入后落在中层这个融合最密集、冗余也最高的区间，关键是两个问题：在哪几层剪、剪掉谁。在哪剪由 ILVAS（Inter-Layer Visual Attention Similarity）决定，它衡量相邻两层之间视觉 token 注意力分布的稳定性——ILVAS 高说明注意力分配已经稳定下来，是适合做过滤的层，所以选 ILVAS 曲线的局部极大值点作为 filtering layer（如 layer {10,14,16,18}）。剪掉谁则用 Differentiable Top-K（DTop-K）做可微选择：先把重要性分数做归一化排序得到 $c'_i$，再用 sigmoid 配一个可学习阈值 $a$ 生成软掩码

$$\text{Mask}(c,a) = \sigma(\lambda(c'_i - a))$$

前向时按硬阈值做离散的保留/丢弃，反向时梯度仍能沿软掩码回流到重要性估计。整个剪枝量按凹形调度安排——前期剪得猛、后期放慢，因为融合初期大量 token 冗余、可以放心快删，越往后剩下的 token 越关键、要慢慢删，这正好贴合中层"融合稀疏性递增"的规律。

**3. Early Exit：深层进入纯语言推理后，视觉 token 只是负担**

HiDrop 用一组 training-free 实验在不同层一次性移除全部视觉 token，发现 layer 24 之后再移除几乎不影响性能。原因是深层已经完成跨模态融合、转入纯语言推理阶段，视觉 token 留在序列里只消耗算力却不再贡献信息。于是在第 $L_{exit}=25$ 层把剩余视觉 token 全部丢掉，后面几层只跑文本。

**4. 工程优化：让动态剪枝在真实加速框架里跑得通**

三阶段策略要落到实际加速，还需要几处工程配合。Persistent Position Encoding 给每个视觉 token 固定的位置标识符，避免动态剪枝把序列打乱后 RoPE 位置错乱。FlashAttention 兼容是靠一条轻量辅助注意力来完成 token 选择，主注意力计算不被改动，从而保留 FlashAttention 的加速。并行解耦则利用 Late Injection 带来的空档——浅层的文本前向和视觉编码可以并行执行，因为这时视觉 token 还没进序列，两条路径互不依赖。

### 损失函数 / 训练策略

- 遵循 LLaVA 标准两阶段训练（预训练 + 指令微调）
- DTop-K 的温度系数 $\lambda = N_v$（即视觉 token 数量）
- 在 8× A100 40GB 上训练

## 实验关键数据

### 主实验

LLaVA-1.5-7B 上 11 个 benchmark 比较（保留约 64 tokens，压缩 88.9%）：

| 方法 | 类型 | MMEP | GQA | VQAv2 | POPE | MMStar | Avg(%) |
|------|------|------|-----|-------|------|--------|--------|
| LLaVA-1.5-7B | 上界(576 tokens) | 1506.5 | 61.9 | 78.5 | 86.8 | 33.7 | 100.0 |
| FastV | 训练free | 1086.6 | 48.8 | 61.6 | 67.7 | 29.6 | 82.8 |
| PDrop | 训练based | 1350.7 | 56.6 | 71.8 | 82.6 | 32.7 | 94.2 |
| TwigVLM | 训练based | 1404.0 | 58.8 | 75.6 | 82.7 | 33.1 | 95.3 |
| **HiDrop** | **训练based** | **1473.3** | **60.5** | **76.5** | **86.4** | **32.0** | **98.3** |

在最极端的 48 token（压缩 91.7%）下，HiDrop 仍达 97.1% 原始性能。

### 消融实验

| 配置 | Avg(%) | 说明 |
|------|--------|------|
| Full HiDrop | 98.3 | 完整框架 |
| w/o Late Injection | 96.8 | 去掉晚注入，浅层也处理视觉token |
| w/o Early Exit | 97.5 | 深层保留视觉token |
| w/o Concave (用线性调度) | 96.9 | 用均匀剪枝替代凹金字塔 |
| Hard Top-K (替代 DTop-K) | 97.1 | 不可微的硬选择 |

训练效率：HiDrop 训练加速 1.72×（vs 原始 LLaVA-1.5-7B）。

### 关键发现

- Late Injection 贡献最大——约 1.5% 的性能保持提升，说明避免浅层处理视觉 token 不仅省计算，还减少了无意义的浅层干扰
- 凹金字塔 > 线性 > 凸金字塔：融合初期激进剪枝的策略最优，与中层融合动态分析的结论完全一致
- DTop-K 比 Hard Top-K 好约 1.2%：可微选择使训练能反向传播到 token 重要性估计
- 即使压缩到仅 48 个视觉 token（每张图像 576→48，压缩 12 倍），POPE 指标只从 86.8 降到 86.6，几乎无损

## 亮点与洞察

- **先分析后设计的范式**：不是先设计方法再找实验支撑，而是先做系统的层级行为分析（intra-modal similarity, cross-modal influence, early exit 实验），用数据发现驱动算法设计。这种研究范式本身就值得学习
- **"延迟注入"是认知突破**：之前所有方法都默认视觉 token 从第一层就参与计算，HiDrop 首次提出"浅层根本不需要视觉信息"——这是对 MLLM 工作机制的深刻洞察，可推广到其他模态（如音频 token）
- **三阶段与层级功能一一对应**：Late Injection-传播层、Concave Pruning-融合层、Early Exit-推理层，设计美感强

## 局限与展望

- **仅在 LLaVA-1.5 验证**：层级行为分析的结论可能不适用于所有 MLLM（如 Qwen-VL、InternVL 的层级功能可能不同），需要更多架构的验证
- **注入层和退出层固定**：$L_{inj}=9$, $L_{exit}=25$ 是硬编码的，不同输入（简单 vs 复杂图像）可能需要不同的处理窗口
- **未考虑多图输入**：视频理解或多图 QA 场景下，视觉 token 数量更多，层级行为可能不同
- **DTop-K 训练开销**：可微 Top-K 引入额外参数和计算，在更大模型上的开销-收益比需要验证

## 相关工作与启发

- **vs FastV**: FastV 在早期单层一次性剪枝，过于粗暴且位置选择不当（浅层即剪）。HiDrop 证明浅层根本不该有视觉 token
- **vs PDrop**: PDrop 用均匀间隔和均匀比例渐进剪枝，忽略了中层融合的非均匀性。HiDrop 的 ILVAS 指标和凹金字塔调度更精准
- **vs TwigVLM**: TwigVLM 在浅层剪枝+深层移除，但浅层剪枝是多余的。HiDrop 用 Late Injection 替代浅层剪枝更高效
- **对视频/多图 MLLM 的启发**：可以分析视频 token 在不同层的行为，可能也存在"浅层冗余"现象，可以用类似策略大幅压缩

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Late Injection 是全新视角，层级功能分析驱动设计是很好的研究范式
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个 benchmark、3 个模型规模、详细消融、效率分析、层级行为可视化
- 写作质量: ⭐⭐⭐⭐⭐ 分析→洞察→设计的叙事流畅，图表精美
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高——直接可用于任何 LLaVA 架构的 MLLM 加速

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[AAAI 2026\] Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration](../../AAAI2026/multimodal_vlm/filter_correlate_compress_training-free_token_reduction_for_.md)
- [\[ICLR 2026\] IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning](ivc-prune_revealing_the_implicit_visual_coordinates_in_lvlms_for_vision_token_pr.md)
- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)
- [\[NeurIPS 2025\] Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](../../NeurIPS2025/multimodal_vlm/beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)

</div>

<!-- RELATED:END -->
