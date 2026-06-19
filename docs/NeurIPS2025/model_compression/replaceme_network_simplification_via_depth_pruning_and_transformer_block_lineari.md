---
title: >-
  [论文解读] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization
description: >-
  [NeurIPS 2025][模型压缩][深度剪枝] 提出 ReplaceMe，一种无训练的深度剪枝方法：用少量校准数据估计线性变换来近似被剪枝的 Transformer 块组，该变换可融合到相邻层权重中不增加参数，在 LLaMA-2-7B 上实现 25% 剪枝率并保留约 90% 性能。 领域现状：结构化剪枝是 LLM 压缩…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "深度剪枝"
  - "Transformer"
  - "无训练压缩"
  - "LLM加速"
  - "层选择"
---

# ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization

**会议**: NeurIPS 2025  
**arXiv**: [2505.02819](https://arxiv.org/abs/2505.02819)  
**代码**: [https://github.com/mts-ai/ReplaceMe](https://github.com/mts-ai/ReplaceMe)  
**领域**: 模型压缩  
**关键词**: 深度剪枝, Transformer线性化, 无训练压缩, LLM加速, 层选择

## 一句话总结
提出 ReplaceMe，一种无训练的深度剪枝方法：用少量校准数据估计线性变换来近似被剪枝的 Transformer 块组，该变换可融合到相邻层权重中不增加参数，在 LLaMA-2-7B 上实现 25% 剪枝率并保留约 90% 性能。

## 研究背景与动机

**领域现状**：结构化剪枝是 LLM 压缩的主要方法之一，但大多数方法需要后剪枝重训练（"healing"过程），计算开销大

**现有痛点**：
   - 直接删除 Transformer 块会造成不可恢复的信息损失（因为删除层等价于恒等映射假设，而实际层不完全是恒等映射）
   - 现有无训练方法（如 LaCo、SliceGPT）性能不足
   - 需要 healing 的方法（如 UIDL、LLMPruner）虽然效果好，但训练开销约几小时+几十GPU小时

**核心矛盾**：删除层 = 假设层是恒等映射，但实际上层做了非平凡变换

**切入角度**：用线性变换**近似**被删除层的变换，比恒等映射精确得多

**核心 idea**：选择影响最小的连续 Transformer 块组，用线性变换 $\mathbf{T}$ 近似其输入-输出映射，然后将 $\mathbf{T}$ 融合到相邻 MLP 的 down-projection 权重中，最终模型结构不变、参数不增

## 方法详解

### 整体框架
ReplaceMe 分四步：(1) **层选择**——用余弦距离找对输出影响最小的连续 n 层；(2) **线性变换估计**——在校准数据上求解最优线性映射 $\mathbf{T}^*$ 使得 $\mathbf{M}_i \cdot \mathbf{T} + \mathbf{Y}_i \approx \mathbf{L}_{i+n}$；(3) **权重融合**——将 $\mathbf{T}^*$ 乘入前一层 MLP 的 down-projection 矩阵；(4) 删除 n 个 Transformer 块。全程**零训练，零额外参数**。

### 关键设计

1. **层选择策略**：

    - 功能：找到最适合剪枝的连续 n 层起始位置 $i^*$
    - 核心思路：$i^* = \arg\min_i h(\mathbf{L}_i, \mathbf{L}_{i+n})$，用余弦距离衡量第 i 层和第 i+n 层隐藏状态的差异。校准数据上平均
    - 设计动机：余弦距离在穷举验证中被证明能找到最优或近最优的剪枝位置，优于 L2 距离
    - 与暴力搜索的对比：在 Appendix 中穷举验证了所有可能的剪枝位置，余弦距离定位准确

2. **线性变换估计——L2 目标**：

    - 功能：求解 $\mathbf{T}^* = \arg\min_\mathbf{T} \|\mathbf{M}_i \cdot \mathbf{T} + \mathbf{Y}_i - \mathbf{L}_{i+n}\|_2^2$
    - 核心思路：经典最小二乘解 $\mathbf{T}^* = (\mathbf{M}_i^T \mathbf{M}_i)^{-1} \mathbf{M}_i^T (\mathbf{L}_{i+n} - \mathbf{Y}_i)$，有解析解，无需迭代优化
    - 其中 $\mathbf{M}_i$ 是第 i 层 MLP 的输出，$\mathbf{Y}_i$ 是注意力子块输出，$\mathbf{L}_{i+n}$ 是第 i+n 层输出

3. **线性变换估计——余弦目标**：

    - 功能：$\mathbf{T}^* = \arg\min_\mathbf{T} \sum_k (1 - \frac{(\mathbf{M}_{i,k} \cdot \mathbf{T} + \mathbf{Y}_{i,k})^\top \mathbf{L}_{i+n,k}}{\|\cdot\| \|\cdot\|})$
    - 核心思路：无解析解，用 Adam 优化（lr=1e-4, 10 epochs）。实际用简化版 $\cos(\mathbf{M}_i \mathbf{T}, \mathbf{L}_{i+n} - \mathbf{Y}_i)$ 减少内存占用
    - 设计动机：余弦距离关注方向而非幅度，对 Transformer 的残差流更合适

4. **权重融合**：

    - 功能：$\mathbf{T}^*$ 与第 i 层 MLP 的 down-projection 矩阵相乘，得到新的 down-projection 矩阵
    - 设计动机：MLP 输出 × 线性变换 = 两个连续线性操作 = 一个等效线性操作，可以合并为单一矩阵。融合后模型结构完全不变，推理代码无需修改

5. **正则化**：

    - L1/L2 正则化改善基准测试精度但增加困惑度——存在精度-困惑度权衡
    - 可选的多组变换（Multi-LT）：非重叠的多组块各自估计独立的线性变换

### 损失函数 / 训练策略
- **完全无训练**——只需一个小校准数据集（如 512 样本的 C4/RedPajama）
- 校准数据选择影响有限（实验证明不同来源的校准数据结果差异 <1%）
- 解析法（L2目标）几乎瞬间完成，数值法（余弦目标）需约 10 epoch Adam 优化

## 实验关键数据

### 主实验 — LLaMA-2-7B (25% 剪枝 = 删除 8/32 层)

| 方法 | 需要训练？ | C3 | HellaSwag | PIQA | MMLU | 平均 | 保留率 |
|------|----------|-----|-----------|------|------|------|--------|
| 未压缩 | - | 43.8 | 71.3 | 78.1 | 46.8 | 45.3 | 100% |
| LLMPruner | ✓ | 29.7 | 54.6 | 72.0 | 25.3 | 35.4 | 78.2% |
| SliceGPT | ✓ | 31.5 | 47.5 | 68.3 | 28.8 | 35.1 | 77.5% |
| LaCo | ✓ | 39.7 | 55.7 | 69.8 | 26.5 | 37.4 | 82.7% |
| UIDL | ✓ | 40.2 | 59.7 | 69.0 | 44.6 | 40.9 | 90.3% |
| **ReplaceMe (余弦)** | **✗** | **42.4** | **64.7** | **73.5** | **45.1** | **41.9** | **92.5%** |

### 消融实验

| 配置 | 平均准确率 | 困惑度 | 说明 |
|------|----------|--------|------|
| 直接删除（恒等映射） | 37.4 | 高 | 无补偿 |
| + L2 线性变换 | 40.8 | 中 | 解析解 |
| + 余弦线性变换 | **41.9** | 中 | 数值优化 |
| + 余弦 + L1 正则 | 42.1 | 较高 | 精度↑ 困惑度↑ |

### 跨模型验证

| 模型 | 剪枝率 | ReplaceMe 保留率 | 最佳竞品 |
|------|--------|----------------|---------|
| LLaMA-2-7B | 25% | **92.5%** | UIDL 90.3% |
| LLaMA-3-8B-Instruct | 25% | **91.8%** | UIDL 89.5% |
| Qwen2.5-7B | 25% | ~90% | - |
| Falcon-11B | 25% | ~89% | - |
| ViT (视觉) | 25% | **93%** | - |

### 关键发现
- **无训练 > 有训练（对大多数方法）**：ReplaceMe 不训练就超越了需要 LoRA healing 的 UIDL（92.5% vs 90.3%），说明线性变换补偿比重训练更高效
- 余弦目标 > L2 目标（+1%），因为方向比幅度更重要——与层选择时余弦距离更优一致
- 校准数据量影响很小：512 样本就够，增加到 2K 仅提升 <0.5%
- ReplaceMe 的 CO2 排放仅是 UIDL 的 1/50——真正的绿色压缩

## 亮点与洞察
- **"连续 Transformer 块 ≈ 线性变换"**的假设被实验充分验证——这本身是一个关于 Transformer 表征结构的有趣发现。说明深层 Transformer 的很多中间层确实只做了"微调方向"的变换
- **融合技巧极为优雅**：线性变换 × MLP down-projection = 新的 down-projection，零额外参数，零推理开销。这是本文最实用的工程贡献
- 余弦距离在层选择和变换估计中**双重使用**——方向性度量在 Transformer 的残差流中确实比幅度度量更合适
- 方法对 ViT 同样有效——泛化到视觉 Transformer

## 局限与展望
- 超过 25% 剪枝率后性能快速下降——线性假设在大比例剪枝下不成立
- 当前只能剪枝**连续**的块——非连续的最优剪枝位置可能更好（Multi_LT_NC 部分探索了这个方向，但效果有限）
- 线性变换不捕获非线性映射——考虑用低秩非线性映射或轻量级 adapter 替代可能更好
- 对指令微调模型（Instruct）的效果不如基础模型——可能因为 Instruct 模型层间依赖更强

## 相关工作与启发
- **vs ShortGPT/LaCo**：这些方法直接删除层（等价于恒等映射假设），ReplaceMe 的线性映射补偿是核心改进
- **vs UIDL**：UIDL 需要 LoRA healing 训练，ReplaceMe 完全无训练但结果更好——说明关键在于初始补偿质量而非后续训练
- **vs SliceGPT**：SliceGPT 做宽度剪枝，ReplaceMe 做深度剪枝——两者正交可以结合
- 启发：如果线性变换就能很好地近似，那么是否可以在**训练时**就约束中间层接近线性，从而设计天然可压缩的架构？

## 评分
- 新颖性: ⭐⭐⭐⭐ 线性变换替代层剪枝是简洁有效的新思路
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多基准、多配置消融、ViT 扩展、CO2 对比
- 写作质量: ⭐⭐⭐⭐ 方法部分数学推导清晰，实验全面
- 价值: ⭐⭐⭐⭐⭐ 无训练+无额外参数+性能保持好=立即可用的LLM压缩方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](../../ICML2026/model_compression/flattengpt_depth_compression_for_transformer_with_layer_flattening.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)
- [\[NeurIPS 2025\] The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis](the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)
- [\[CVPR 2025\] Layered Image Vectorization via Semantic Simplification](../../CVPR2025/model_compression/layered_image_vectorization_via_semantic_simplification.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)

</div>

<!-- RELATED:END -->
