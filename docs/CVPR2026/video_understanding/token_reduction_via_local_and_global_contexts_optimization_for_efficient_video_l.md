---
title: >-
  [论文解读] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models
description: >-
  [CVPR2026][视频理解][Video LLM] 提出 AOT 框架，通过建立局部-全局 token anchors 并利用最优传输（Optimal Transport）在帧内和帧间两级聚合被裁剪/合并 token 的语义信息，实现 training-free 的视频 token 压缩，在裁剪 90% token 的情况下仍保留 97.6% 的原始性能。
tags:
  - "CVPR2026"
  - "视频理解"
  - "Video LLM"
  - "Token Reduction"
  - "optimal transport"
  - "training-free"
  - "Spatiotemporal Compression"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models

**会议**: CVPR2026  
**arXiv**: [2603.01400](https://arxiv.org/abs/2603.01400)  
**代码**: [AOT Project](https://github.com/) (待确认)  
**领域**: 视频理解  
**关键词**: Video LLM, Token Reduction, optimal transport, training-free, Spatiotemporal Compression

## 一句话总结

提出 AOT 框架，通过建立局部-全局 token anchors 并利用最优传输（Optimal Transport）在帧内和帧间两级聚合被裁剪/合并 token 的语义信息，实现 training-free 的视频 token 压缩，在裁剪 90% token 的情况下仍保留 97.6% 的原始性能。

## 研究背景与动机

**视频 LLM 的计算瓶颈**：Video LLM 处理视频时，视觉编码器将采样帧转化为大量 token（长视频可达百万级），prefilling 阶段占据了绝大部分 FLOPs（约 98%），推理成本极高。

**现有训练式压缩的局限**：部分方法通过可训练模块压缩 token，但需大量训练资源和 GPU 开销，难以广泛部署。

**空间压缩忽略时序依赖**：VisionZip、LLaVA-PruMerge 等方法主要做帧内空间冗余去除，在低保留率下性能骤降（如 10% 保留时性能下降 8.4%），因为未利用帧间时序冗余。

**LLM 内部剪枝效率有限**：FastV、PDrop 等在 LLM 层内做 token 剪枝，但浅层开销仍在，且难以有效利用长上下文的可压缩性。

**简单合并/丢弃丢失关键信息**：现有方法要么直接丢弃低重要性 token，要么简单合并相似 token，忽略了这些 token 中包含的细微但有用的语义和上下文信息。

**缺乏全局最优的信息聚合视角**：已有方法缺少一种系统性的框架来度量被剪枝 token 与保留 token 之间的关系，并将有用信息最优地汇聚到保留 token 上。

## 方法详解

### 整体框架

AOT（Anchors + Optimal Transport）要解决的是 Video LLM 里 prefill 占了约 98% FLOPs、而现有压缩要么直接丢 token 丢信息、要么需要训练的难题。它的思路是“不丢弃而聚合”：先在每帧里挑出语义重要且空间多样的 token 作锚点（Local-Global Token Anchors），再用最优传输把被裁掉的 token 的信息分两级——帧内 OT 聚合（Phase I）和帧间 OT 聚合（Phase II）——最优地汇聚回锚点，全程 training-free。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["采样视频帧 → 视觉编码器"] --> ANCHOR
    subgraph ANCHOR["Local-Global Token Anchors"]
        direction TB
        B["全局锚点<br/>末层 [CLS] 注意力 Top-K"]
        C["局部锚点<br/>网格窗口内浅层 [CLS] Top-K"]
    end
    ANCHOR --> D["帧内 OT 聚合（Phase I）<br/>非锚点经 Sinkhorn 传给锚点"]
    D --> E["帧间 OT 聚合（Phase II）<br/>clip 首帧为时序锚点"]
    E -->|"max 分配概率 q < τ：含独特时序信息"| F["保留 token"]
    E -->|"q ≥ τ：时序冗余"| G["聚合到锚点"]
    F --> H["压缩后 token → LLM prefill"]
    G --> H
```

### 关键设计

**1. Local-Global Token Anchors：用全局 + 局部双配额挑出该保留的锚点**

简单丢弃会丢关键信息，而要聚合信息先得有一组好的保留点。AOT 同时取两类锚点：全局锚点用视觉编码器最后一层 [CLS] token 的多头注意力得分取 Top-K 个高注意力 token $\mathbf{x}_V^g$；局部锚点把图像特征划成 $W$ 个非重叠网格窗口，在浅层每个窗口内按 [CLS] 注意力取 $K_w = K/W$ 个 token $\mathbf{x}_V^l$。最终锚点 $\mathbf{X}_V^{\text{anchors}} = \mathbf{x}_V^g \cup \mathbf{x}_V^l$，全局与局部配额相等以兼顾“重要”和“覆盖”。

**2. 帧内 OT 聚合：把帧内被裁 token 的语义最优地传给锚点**

针对“空间冗余但信息别丢”，把锚点 $\mathbf{X}_V^a$ 和非锚点 $\mathbf{X}_V^u$ 看成两个离散分布，以逆余弦相似度构代价矩阵 $\bm{C} = \bm{1} - (\mathbf{X}_V^a)^\top \mathbf{X}_V^u$，用 Sinkhorn-Knopp 迭代求最优传输计划 $\bm{T}^*_{intra}$，再按传输质量加权聚合 $\tilde{\mathbf{x}}_j^a = \frac{\mathbf{x}_j^a + \lambda_{intra} \sum_i T^*_{ij} \mathbf{x}_i^u}{1 + \lambda_{intra} m_j}$。这样被丢的 token 不是凭空消失，而是把上下文“运”到了保留点上。

**3. 帧间 OT 聚合：消时序冗余又不抹掉变化帧**

针对“空间压缩忽略帧间时序冗余、低保留率下性能骤降”，把帧序列切成 clip、每个 clip 以首帧 token 为时序锚点，对后续帧逐帧算 OT 计划；行归一化后看每个 token 的最大分配概率 $q_i^{(\ell)} = \max_j p_{ij}^{(\ell)}$，若 $q_i < \tau$ 说明该 token 含时序独特信息、保留不合并，否则按 OT 权重聚合到锚点上逐步更新。一保一聚，既压时序冗余又不丢动态信息。

### 损失函数 / 训练策略

- 不涉及训练损失，完全 training-free
- 核心优化目标是最小化 OT 距离 $d_{\text{OT}}(\bm{u}, \bm{v} | \bm{C})$
- 通过 Sinkhorn-Knopp 迭代（默认 100 次）快速求解，entropic 正则化系数 $\lambda$ 控制平滑度
- 权重系数 $\lambda_{intra}$ 和 $\lambda_{inter}$ 均默认 1.0，控制聚合贡献强度

## 实验

### 主要结果

在 LLaVA-OneVision-7B 上的对比（32 帧输入，4 个视频理解 benchmark）：

| 方法 | FLOPs (T) | 保留率 | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. | Score% |
|------|-----------|--------|---------|-----------|---------------|----------|------|--------|
| LLaVA-OV-7B (vanilla) | 40.8 | 100% | 58.3 | 60.4 | 56.4 | 58.6 | 58.4 | 100 |
| VisionZip | 3.4 | 10% | 53.5 | 58.0 | 49.3 | 53.4 | 53.5 | 91.6 |
| PruneVid | 3.4 | 10% | 56.2 | 59.8 | 54.5 | 56.0 | 56.6 | 96.9 |
| FastVID | 3.4 | 10% | 55.9 | - | 56.3 | 57.3 | - | - |
| **AOT** | **3.4** | **10%** | **57.2** | **60.3** | **53.8** | **56.6** | **57.0** | **97.6** |

在 LLaVA-Video-7B 上（64 帧输入，25% 保留率）：

| 方法 | FLOPs (T) | 保留率 | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. | Score% |
|------|-----------|--------|---------|-----------|---------------|----------|------|--------|
| LLaVA-Video-7B (vanilla) | 80.2 | 100% | 60.4 | 57.2 | 58.9 | 64.3 | 60.2 | 100 |
| VisionZip | 9.3 | 25% | 56.7 | 54.7 | 54.7 | 60.7 | 56.7 | 94.2 |
| **AOT** | **9.3** | **25%** | **59.2** | **55.6** | **55.9** | **62.4** | **58.3** | **96.8** |

### 消融实验

Token Anchors 各组件贡献（10% 保留率）：

| 配置 | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. | Score% |
|------|---------|-----------|---------------|----------|------|--------|
| w/o Local Anchors | 56.5 | 60.1 | 54.0 | 55.7 | 56.6 | 96.9 |
| w/o Global Anchors | 55.5 | 59.4 | 53.4 | 53.1 | 55.4 | 94.9 |
| w/o OT | 56.1 | 60.2 | 53.5 | 55.8 | 56.4 | 96.6 |
| OT w/o Intra-frame | 57.1 | 60.2 | 53.6 | 54.6 | 56.3 | 96.6 |
| OT w/o Inter-frame | 56.1 | 60.0 | 53.6 | 55.9 | 56.4 | 96.6 |
| **AOT (Full)** | **57.2** | **60.3** | **53.8** | **56.6** | **57.0** | **97.6** |

聚合策略对比：No Merging (56.4) vs Cosine Merging (52.4) vs **AOT (57.0)**，验证 OT 聚合远优于简单余弦合并。

### 关键发现

- **Sinkhorn 计算开销极低**：100 次迭代的帧内+帧间 OT 仅需 2.11ms，不到总推理时间的 1%
- **超越 vanilla 模型的现象**：在部分 benchmark 上 AOT 压缩后性能反而优于原模型，说明大量冗余/无关 token 实际上充当了噪声，干扰了 LLM 对关键信息的聚焦
- **帧数 scaling 优势**：从 16 到 128 帧，AOT 持续优于其他压缩方法；在 128 帧时 vanilla 模型受上下文长度限制，而 AOT 仍可正常工作
- **Global Anchors 比 Local Anchors 更关键**：移除全局锚点的性能下降（-3.0 Avg）大于移除局部锚点（-0.4 Avg）

## 亮点

- **新视角**：首次系统研究如何将被裁剪/合并 token 的信息"最优地"聚合回保留 token，而非简单丢弃
- **OT 的巧妙应用**：将 token 压缩建模为离散最优传输问题，供应方（被裁 token）向需求方（锚点）传递上下文，Sinkhorn 求解快速高效
- **双层优化架构**：帧内消除空间冗余 + 帧间消除时序冗余，两级 OT 互补，覆盖完整的时空压缩需求
- **完全 training-free**：无需微调即可作为即插即用模块应用于各种 Video LLM
- **极端压缩下表现突出**：10% 保留率下仍保持 97.6% 原始性能，大幅领先 VisionZip 的 91.6%

## 局限性

- 仅在 LLaVA 系列 7B 模型上验证，缺乏对更大模型（如 72B）或其他架构（如 Qwen-VL、InternVL）的泛化验证
- 帧间 clip 分割采用均匀采样或简单聚类，未探索自适应的帧级重要性感知分组
- 阈值 $\tau$ 和权重 $\lambda$ 需手动设定，不同视频场景可能需要不同超参
- OT 求解虽然快速但仍引入额外内存（代价矩阵 $M \times N$），对超高帧率或超高分辨率场景的可扩展性待验证
- 当前评估局限于多选题 benchmark，缺乏开放式生成任务（如 video captioning、video grounding）的评测

## 相关工作

- **帧内空间压缩**：VisionZip（CLS注意力选择+token合并）、LLaVA-PruMerge（自适应空间冗余检测与token裁减）、ToMe（bipartite matching token合并）
- **LLM内部剪枝**：FastV（prefilling注意力引导token选择）、PDrop（层级渐进剪枝）、SparseVLM（文本引导的视觉token排序）
- **视频专用压缩**：DyCoke（跨帧合并+动态KV cache裁减）、PruneVid（空间+时序联合聚类裁减）、FastVID（时序分段+时空token合并）、TempMe（渐进空间裁减+相邻clip合并）、FrameFusion（LLM浅层交叉合并+剪枝）
- **最优传输基础**：Sinkhorn距离、Wasserstein距离在分布比较中的广泛应用

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 OT 引入 video token 压缩是新颖视角，"不丢弃而聚合"的思路有启发性
- 实验充分度: ⭐⭐⭐⭐ — 4个benchmark、2个模型、多保留率对比、充分消融，但缺乏大模型和开放式任务评测
- 写作质量: ⭐⭐⭐⭐ — 框架清晰、公式规范、图示直观，但部分段落略冗长
- 价值: ⭐⭐⭐⭐ — training-free且效果出色，实用价值高，对 Video LLM 推理加速有即时意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](../../ICLR2026/video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](../../ICML2026/video_understanding/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](../../NeurIPS2025/video_understanding/self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)

</div>

<!-- RELATED:END -->
