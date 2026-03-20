# Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2505.22038](https://arxiv.org/abs/2505.22038)  
**代码**: [https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning](https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning)  
**领域**: 多模态VLM / 模型压缩  
**关键词**: token pruning, vision-language model, attention-diversity balance, inference acceleration, local-global optimization  

## 一句话总结
提出 Balanced Token Pruning (BTP)，通过在浅层优先多样性剪枝、深层优先注意力剪枝的分阶段策略，联合优化局部输出一致性和全局表示质量，在仅保留 22% 视觉 token 的情况下保持原模型 98% 的性能。

## 背景与动机
大型视觉-语言模型（LVLM）将图像编码为上千个视觉 token，高分辨率输入下（如 LLaVA-NeXT 每张图最多 2880 token）计算开销巨大，严重限制边缘部署。现有 token 剪枝方法分两派：
- **注意力方法**（FastV、PyramidDrop）：根据文本对图像 token 的注意力分数选择重要 token，但只优化当前层输出，忽略对后续层的影响（局部最优）
- **多样性方法**（DivPrune）：最大化保留 token 的语义多样性，考虑了全局影响，但忽略当前层输出的一致性

这两类方法各有盲点，且剪枝层的选择依赖手动调参，缺乏模型内在属性的指导。

## 核心问题
如何在 token 剪枝时**同时兼顾局部（当前层输出一致性）和全局（后续层表示质量）目标**，实现整体最优？同时，如何自动确定哪些层适合执行剪枝？

## 方法详解

### 整体框架
BTP 是一个 plug-and-play 的推理时视觉 token 剪枝方法。输入图像经视觉编码器产生视觉 token 序列，与文本 token 拼接后送入 LLM backbone。BTP 在 backbone 的多个层执行分阶段剪枝：
1. 用小型校准集（64 样本）确定剪枝层
2. 在浅层强调多样性目标（保护下游层的表示质量）
3. 在深层强调注意力目标（保持当前层输出一致性）
4. 每阶段保留前一阶段 50% 的 token，最终阶段丢弃所有视觉 token

### 关键设计
1. **Local-Global 目标函数**: 将剪枝目标形式化为加权组合——注意力项（局部）保证当前层输出与原模型接近，多样性项（全局）用 token 间的距离和来近似后续层的 token 偏好。权重系数 λ 随层数递增：浅层 λ 小（偏多样性），深层 λ 大（偏注意力）。公式：$\mathcal{L} = -\sum_i (\lambda_i \cdot \text{Attn}(P_i) + (1-\lambda_i) \cdot F_{dis}(P_i))$

2. **注意力重平衡（Rebalanced Attention）**: 位置编码导致注意力偏向序列后部的 token。BTP 先 over-select top-k' 个 token，然后优先保留前半段位置的 token，再从后半段补充至目标数量 k，缓解位置偏差。

3. **空间多样性初始化**: 解 MMDP（Max-Min Diversity Problem）需要 O(n²) 复杂度，GPU 无法高效加速。BTP 利用图像 patch 的 2D 空间位置，用曼哈顿距离进行初始化——空间上分散的 patch 通常语义差异大，以此加速多样性集合的选择。

4. **基于校准的剪枝层选择**: 计算每层图像 token 隐状态的余弦相似度变化，找到图像 token 表示发生剧烈变化的层，在变化前后执行剪枝。这是任务无关的，仅需 64 个样本即可稳定确定。

### 损失函数 / 训练策略
BTP 是无训练的推理时方法，不涉及模型参数更新。核心是 local-global 目标的加权平衡，λ 设置例如 LLaVA-v1.5 为 (0.6, 0.8, 1.0)，Qwen2.5-VL 为 (0.2, 0.5, 0.8, 1.0)。

## 实验关键数据

| 模型 | 方法 | Token数 | GQA | MME | POPE | SQA | Avg性能保持 |
|------|------|---------|-----|-----|------|-----|-----------|
| LLaVA-1.5-7B | Original | 576 | 62.0 | 1510.7 | 85.8 | 69.4 | 100% |
| LLaVA-1.5-7B | FastV | 172 | 57.6 | 1465.0 | 81.0 | 68.9 | 96% |
| LLaVA-1.5-7B | DivPrune | 128 | 58.8 | 1405.4 | 85.1 | 68.4 | 96% |
| LLaVA-1.5-7B | **BTP** | **128** | **59.0** | **1487.0** | **85.6** | **69.1** | **98%** |
| LLaVA-1.5-13B | **BTP** | 128 | 62.2 | 1519.7 | 86.9 | 72.7 | 98% |
| Qwen2.5-VL-7B | **BTP** | 25% | 57.2 | 1651.5 | 86.2 | 74.1 | 97% |

效率对比（LLaVA-1.5-7B, 128 tokens）：BTP 延迟 0.134s（加速 7%），DivPrune 0.224s（反而慢 54%）。

### 消融实验要点
- **λ 平衡因子**：浅层过度偏向多样性会掉点，深层必须偏向注意力；中间层保持适度多样性效果最好
- **注意力重平衡**：去掉后 GQA 从 59.0 降至 57.4，因位置编码偏差导致 token 选择次优
- **空间初始化**：去掉后延迟从 0.134s 升至 0.232s，甚至超过原模型，证明空间初始化对实际加速至关重要
- **校准层选择 vs 均匀划分**：校准方法在 Qwen2.5-VL 上优势尤为明显（1641.5 vs 1551.6 MME）

## 亮点
- 核心洞察非常直觉：浅层 token 多，应保多样性（为深层留余地）；深层 token 少，应保注意力一致性（精确选择）
- Plug-and-play，无需微调，兼容 FlashAttention
- 空间位置初始化是巧妙的工程 trick，将 O(n²) 的 MMDP 加速且在 GPU 上高效运行
- 仅需 64 个校准样本即可确定剪枝层，跨数据集稳定

## 局限性 / 可改进方向
- 仅处理图像 token 剪枝，未考虑视频场景下的时序冗余
- λ 的层级设置仍需手动指定（虽然跨数据集稳定），可探索自适应学习
- 在极端压缩率下（如仅保留 64 token）性能下降较明显
- 未探索与 token merge 方法的结合（当前只做 prune 不做 merge）

## 与相关工作的对比
- **vs FastV**：FastV 只在一层做注意力剪枝，BTP 多阶段且兼顾全局，同压缩率下 BTP 大幅领先
- **vs PyramidDrop**：PyramidDrop 也分阶段但纯注意力策略，BTP 引入多样性平衡后效果更好
- **vs DivPrune**：DivPrune 纯多样性方法在局部一致性上差，且 MMDP 求解慢导致实际推理反而变慢；BTP 的空间初始化解决了效率问题

## 启发与关联
- 与 [ideas/model_compression/20260316_task_aware_token_compression.md](../../../ideas/model_compression/20260316_task_aware_token_compression.md) 高度相关：可以将 BTP 的 local-global 平衡思想引入任务感知压缩
- 浅层保多样性、深层保精确性的分阶段策略可迁移到 3D 场景（参考 [ideas/3d_vision/20260317_adaptive_token_pruning_from_4dgs.md](../../../ideas/3d_vision/20260317_adaptive_token_pruning_from_4dgs.md)）
- 空间初始化的 trick 可用于检测任务的 token 压缩（参考 [ideas/object_detection/20260317_token_compress_det_head.md](../../../ideas/object_detection/20260317_token_compress_det_head.md)）

## 评分
- 新颖性: ⭐⭐⭐⭐ local-global 平衡视角清晰但本质是注意力和多样性的加权组合
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多基准多消融，效率分析全面
- 写作质量: ⭐⭐⭐⭐ 动机分析清楚，可视化直觉性强
- 价值: ⭐⭐⭐⭐ 实用性强，plug-and-play 且效果好，但领域较成熟
