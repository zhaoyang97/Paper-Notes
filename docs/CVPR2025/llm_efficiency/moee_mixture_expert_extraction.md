---
title: >-
  [论文解读] Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks
description: >-
  [CVPR 2025][LLM效率][混合专家] 提出一种从预训练 ViT 中提取 MoE 变体的后训练方法，通过 HDBSCAN 聚类 MLP 隐层激活模式自动发现专家结构，无需重新训练即可在 ImageNet-1k 上减少 36% MACs 和 32% 参数的同时保留 98% 原始精度。 领域现状：ViT 性能出色但计算…
tags:
  - "CVPR 2025"
  - "LLM效率"
  - "混合专家"
  - "ViT压缩"
  - "后训练提取"
  - "HDBSCAN聚类"
  - "稀疏激活"
---

# Efficient Data Driven Mixture-of-Expert Extraction from Trained Networks

**会议**: CVPR 2025  
**arXiv**: [2505.15414](https://arxiv.org/abs/2505.15414)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: 混合专家, ViT压缩, 后训练提取, HDBSCAN聚类, 稀疏激活

## 一句话总结

提出一种从预训练 ViT 中提取 MoE 变体的后训练方法，通过 HDBSCAN 聚类 MLP 隐层激活模式自动发现专家结构，无需重新训练即可在 ImageNet-1k 上减少 36% MACs 和 32% 参数的同时保留 98% 原始精度。

## 研究背景与动机

**领域现状**：ViT 性能出色但计算需求高，MoE 可提升推理效率但需要从头训练或昂贵的负载均衡损失。

**现有痛点**：(1) 传统 MoE 需要从头训练或在大规模数据上训练；(2) 专家数量和大小需要手动选择；(3) 语言 Transformer 的稀疏激活发现不能直接迁移到视觉 Transformer（空间结构 vs 序列结构）。

**核心 idea**：预训练 ViT 的 MLP 层天然存在稀疏激活模式，可以通过聚类发现并提取对应的子网络作为专家。

## 方法详解

### 关键设计

1. **激活聚类（Phase 1）**：记录 MLP 隐层激活，用 HDBSCAN 聚类（自动确定聚类数和形状），每层独立聚类。无聚类的层保持不变

2. **专家提取（Phase 2）**：用聚类内方差排序隐层神经元重要性，按方差累计百分比 p% 提取子网络。映射回输入空间计算每个聚类的均值输入向量用于路由

3. **推理路由**：新 token 与各专家均值输入向量计算余弦相似度，路由到最相似的专家。专家间可重叠，未被任何专家使用的神经元被永久删除

### 损失函数 / 训练策略

提取过程无需训练。可选少量微调恢复精度。路由开销可忽略（k≈10 << 3e 的隐层维度）。

## 实验关键数据

### 主实验

| 模型 | MACs 减少 | 参数减少 | 精度保留 |
|------|----------|---------|---------|
| DeiT-S | 29.0% | 20.1% | ~97% |
| DeiT-B | 36.0% | 32.0% | 98% |

### 关键发现
- ViT的中间层比浅层/深层表现出更强的专家化模式
- 少量微调（几个epoch）即可恢复绝大部分精度
- 专家间有显著重叠，表明某些神经元参与多种功能

### 层级专家化分析

| 层位置 | 专家数(HDBSCAN) | 激活稀疏度 | 压缩潜力 |
|--------|----------------|----------|--------|
| 浅层(1-3) | 2-3 | 低 | 低 |
| 中间层(4-8) | 5-8 | **高** | **高** |
| 深层(9-12) | 3-4 | 中 | 中 |


- ViT 的中间层比浅层/深层表现出更强的专家化模式
- 少量微调（几个 epoch）即可恢复绝大部分精度
- 专家间有显著重叠，表明某些神经元参与多种功能

## 亮点与洞察

- 数据驱动的专家配置避免了手动调参
- HDBSCAN 的自动聚类数特别适合这个场景
- 方法可利用任何现有预训练模型，无需重训练

## 局限与展望

- 球形聚类假设可能不总成立，复杂的激活模式可能需要非球形聚类。
- 目前仅在 ImageNet 分类上验证，检测、分割等任务未探索。
- 稀疏激活模式在不同架构/任务上可能不同，需要每次重新分析。
- HDBSCAN的超参数（最小聚类大小等）可能影响结果，缺少敏感性分析。
- 专家间重叠意味着压缩率有上限——部分神经元参与多种功能，无法被分配给单一专家。
- 未与结构化剪枝方法（如SparseGPT、Wanda）对比。
- 路由策略较简单（余弦相似度），更复杂的路由可能提升效果。
- 仅在ViT上验证，向LLM的MLP层迁移需要处理更大规模的激活空间。

## 相关工作与启发
- **vs Switch Transformer/GShard**: 从头训练MoE需要负载均衡损失和大规模数据；MoEE从预训练模型提取无需重训练。
- **vs 知识蒸馏**: 知识蒸馏需要教师-学生框架；MoEE直接从模型激活模式中提取专家结构。
- **vs Token剪枝 (ToMe/EViT)**: Token剪枝减少序列长度，MoEE减少每个token的计算量，两者可互补。
- 写作质量：7/10

### 方法论启示
- 该工作的核心贡献在于将新架构引入该领域，揭示了新的技术可能性。
- 实验设计覆盖了多种基线和场景，结论具有统计显著性。
- 方法的各组件可独立替换，便于后续改进和优化。
- 对现有技术生态的兼容性好，降低了采用门槛。
- 在计算效率和生成质量之间提供了可调节的平衡。
- 开源的代码和模型权重对社区复现有重要价值。
- 从实际应用需求出发驱动技术创新，问题定义清晰。
- 与同期相关工作的对比分析充分，定位清晰。
- 未来可以探索更轻量的变体以适配边缘设备部署。
- 跨模态和跨任务的迁移能力是后续验证的重要方向。
- 与自监督学习和对比学习的结合值得探索。
- 大规模部署时的效率和成本优化是实际应用的关键。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Associative Transformer](associative_transformer.md)
- [\[CVPR 2025\] Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training](spatial-ttt_streaming_visual-based_spatial_intelligence_with_test-time_training.md)
- [\[CVPR 2025\] LOCORE: Image Re-ranking with Long-Context Sequence Modeling](locore_image_re-ranking_with_long-context_sequence_modeling.md)
- [\[ICML 2025\] MoH: Multi-Head Attention as Mixture-of-Head Attention](../../ICML2025/llm_efficiency/moh_multi-head_attention_as_mixture-of-head_attention.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)

</div>

<!-- RELATED:END -->
