# FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers

**会议**: ICCV 2025  
**arXiv**: [2501.16297](https://arxiv.org/abs/2501.16297)  
**代码**: 无（未提及）  
**领域**: 多模态VLM / 高分辨率理解  
**关键词**: visual registers, token compression, MLLM, 高分辨率, 子图分裂, 冗余消除  

## 一句话总结
针对高分辨率MLLM中裁切子图导致的视觉编码分裂和token冗余问题，提出可学习的Visual Registers在encoder内部自适应聚合关键信息（ReCompact）并跨子图交互（ReAtten），实现9倍视觉token压缩且性能更优。

## 背景与动机
高分辨率MLLM（如LLaVA-NeXT）通常将大图裁切为多个子图分别编码，导致两个问题：(1) **冗余**——每个子图的ViT输出包含大量低信息量的背景token，子图越多token越爆炸；(2) **分裂**——裁切割断了物体的空间连续性（一个物体可能跨2-3个子图），各子图独立编码无法获取全局上下文。现有方法要么用独立的压缩模块后处理（如PruMerge），要么用Qformer重采样（如BLIP-2），但都是视觉编码之后的补救。

## 核心问题
如何在ViT编码器内部就解决高分辨率裁切带来的冗余和分裂问题，而非事后修补？

## 方法详解

### 整体框架
FALCON在ViT的每个子图编码中插入一组可学习的Visual Register token（类似DINOv2的register概念但功能不同）。这些register通过两种机制工作：ReCompact在encoder内部聚合重要信息产生少量紧凑输出，ReAtten在编码过程中实现跨子图的register交互。

### 关键设计
1. **ReCompact（Register-based Representation Compacting）**：在ViT的attention层中引入一组learnable register tokens。这些register通过自注意力与图像patch tokens交互，自适应地从大量patch中聚合关键视觉信息。编码完成后，只需取出register tokens的表示作为该子图的输出（丢弃原始patch tokens），实现极大的压缩比。与后处理式压缩不同，信息聚合是在encoder内部逐层完成的——更早、更彻底。

2. **ReAtten（Register Interactive Attention）**：不同子图的register之间可以交互（cross-attend），使得每个子图的register能"看到"其他子图的全局信息。这巧妙地解决了裁切导致的编码分裂——被切成两半的物体，其两个子图的register可以通过交互恢复完整的语义理解。关键在于register数量很少，所以跨子图交互的计算开销极小。

3. **端到端训练**：Register tokens和ReAtten模块与ViT联合训练（或微调），不需要额外的独立压缩模块。最终输出到LLM的视觉token数量等于register数量×子图数，相比原始patch tokens数量实现9倍压缩。

### 损失函数 / 训练策略
标准的MLLM instruction tuning损失。Register tokens作为learnable参数随ViT微调更新。

## 实验关键数据
- 在多个高分辨率benchmark上**9倍token压缩**后性能不降反升
- 对比LLaVA-NeXT等高分辨率方法在TextVQA、DocVQA、ChartQA等需要细节理解的任务上展现出竞争力或更好的性能
- 相比PruMerge等后处理压缩方法，FALCON在高压缩比下保持更稳定的性能
- ReAtten使子图间信息流通，在物体跨越多个子图的场景中尤其有效

### 消融实验要点
- ReCompact单独贡献最大——在encoder内部压缩比事后压缩质量更高
- ReAtten在需要跨区域推理的任务上有显著提升（如文档理解、大场景理解）
- Register数量的选择：少量register就能聚合足够信息，数量增加收益递减
- 9x压缩比在性能-效率trade-off上最优

## 亮点
- **在encoder内部做压缩**vs后处理压缩，是理念上的创新——让encoder"学会"只输出精华
- **Register的双重用途**：既做信息聚合（ReCompact），又做跨子图通信（ReAtten），优雅统一
- **9x压缩比非常实用**：对部署高分辨率MLLM到资源受限设备有直接价值
- **解决了"分裂"这个被忽视的问题**：裁切子图的独立编码是高分辨率MLLM的固有缺陷，通过register交互巧妙修复

## 局限性 / 可改进方向
- 需要对ViT进行微调（非training-free），不能直接即插即用
- Register数量是超参数，可能需要针对不同任务调优
- ReAtten增加了一些跨子图的通信开销（虽然因register少而很小）
- 未与最新的动态分辨率方案（如InternVL的动态裁切）对比

## 与相关工作的对比
- **vs. LLaVA-PruMerge**：PruMerge在encoder输出后基于CLS注意力剪枝+合并；FALCON在encoder内部用register聚合——更早更彻底
- **vs. Feather the Throttle**：Feather在LLM内部剪枝视觉token；FALCON在ViT阶段就压缩——两者作用在不同位置，理论上可以叠加
- **vs. DINOv2 registers**：DINOv2的register主要解决attention map中的artifact问题；FALCON的register被设计为信息聚合器+跨子图通信器

## 启发与关联
- Visual register作为"信息容器"的概念可以迁移到其他需要信息压缩的场景（如视频理解中的temporal register）
- ReAtten的跨子图通信思路对任何使用tiling/cropping策略的VFM都有参考价值
- 与Scaling Language-Free Visual Repr论文结合：在大模型encoder中使用register是否能进一步改善scaling behavior

## 评分
- 新颖性: ⭐⭐⭐⭐ Visual register用于压缩和跨子图通信的设计新颖，但register token概念本身不新
- 实验充分度: ⭐⭐⭐⭐ 多个高分辨率benchmark验证，消融详尽
- 写作质量: ⭐⭐⭐⭐ 问题定义（冗余+分裂）清晰精准
- 价值: ⭐⭐⭐⭐⭐ 9x压缩对高分辨率MLLM部署有直接实用价值
