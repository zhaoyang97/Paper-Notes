# Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation

**会议**: ICCV 2025  
**arXiv**: [2510.09094](https://arxiv.org/abs/2510.09094)  
**代码**: 无（未提及）  
**领域**: 图像生成 / 模型压缩 / MoE  
**关键词**: Diffusion Transformer, Mixture of Experts, 结构化稀疏, FLUX, dense-to-sparse转换  

## 一句话总结
首次将预训练的dense DiT（如FLUX.1）转换为Mixture-of-Experts结构实现结构化稀疏推理，通过Taylor度量专家初始化+知识蒸馏+Mixture-of-Blocks进一步稀疏化，在激活参数减少60%的同时保持原始生成质量，全面超越剪枝方法。

## 背景与动机
DiT在文生图任务上表现出色，但参数量巨大（如FLUX.1有12B参数）导致推理开销很高。现有压缩方法主要是剪枝（pruning），但激进剪枝会导致模型容量下降和严重性能退化。MoE天然适合这个场景——保留总模型容量的同时只激活部分参数——但直接从头训练MoE扩散模型成本极高。如何将已有的dense DiT高效转换为MoE结构是关键问题。

## 核心问题
如何将预训练的dense DiT有效转换为MoE架构，在大幅减少激活参数的同时保持生成质量？

## 方法详解

### 整体框架
Dense2MoE通过三步pipeline将dense DiT转换为稀疏MoE模型：(1) 用Taylor度量将每层FFN拆分为多个专家并初始化；(2) 通过知识蒸馏+load balance loss训练MoE router和微调专家；(3) 进一步引入Mixture-of-Blocks(MoB)实现block级别的稀疏。

### 关键设计
1. **Taylor度量专家初始化**：将每个FFN层的权重矩阵拆分为多个专家时，不是随机分配而是基于Taylor展开的重要性度量来分组——使得每个专家保留最相关的参数子集，减少初始化时的性能损失。FFN的激活参数减少62.5%（8个专家只激活3个）。

2. **知识蒸馏训练**：用原始dense模型作为teacher，MoE模型作为student，通过特征级蒸馏损失+load balancing loss训练。蒸馏确保MoE输出与dense模型对齐，load balance确保专家被均匀使用避免退化。

3. **Mixture-of-Blocks (MoB)**：在block级别进一步稀疏化——不是所有DiT block都需要对每个token完整执行。通过group feature loss训练MoB router，让模型学会跳过不必要的block，进一步降低计算量。

### 损失函数 / 训练策略
- 多步蒸馏：Taylor初始化→MoE layer蒸馏→MoB蒸馏
- 损失：特征对齐loss + load balance loss + group feature loss

## 实验关键数据
- 在**FLUX.1 [dev]**上验证：激活参数减少**60%**
- 生成质量保持与原dense FLUX.1相当（FID/CLIP/GenEval等指标几乎不降）
- 全面超越基于剪枝的压缩方法（pruning在相同压缩比下质量明显下降）
- MoB进一步提升稀疏度而不显著影响质量

### 消融实验要点
- Taylor度量初始化 >> 随机初始化（性能差距显著）
- 知识蒸馏对保持生成质量至关重要
- MoB在MoE基础上可以进一步提升10-15%的效率
- 62.5% FFN激活参数减少是性能-效率最优平衡点

## 亮点
- **首次在DiT上做dense-to-MoE转换**：开创了扩散模型从dense到sparse的新范式，区别于传统pruning
- **容量保留**：MoE保留了全部模型容量（所有参数仍在），只是推理时选择性激活——理论上比pruning永久删除参数更优
- **在FLUX.1上验证**：在目前最强的开源T2I模型之一上实际验证，实用价值高
- **与Dynamic-DINO的MoE思路异曲同工**：都是将dense模型转为MoE以支持稀疏推理，但Dense2MoE面向生成模型，Dynamic-DINO面向检测模型

## 局限性 / 可改进方向
- MoE的并行推理需要工程优化（顺序循环各专家较慢）
- 蒸馏训练仍需一定计算资源
- 未探索与步骤蒸馏（如SANA-Sprint）的结合——MoE减参数×步骤蒸馏减步数可能实现更极致加速
- 仅在FLUX.1上验证，SD3/SANA等其他架构未测试

## 与相关工作的对比
- **vs. Dynamic-DINO**：Dynamic-DINO在OV检测器的decoder用MoE；Dense2MoE在DiT的FFN用MoE——方向类似但应用不同
- **vs. SANA-Sprint**：SANA-Sprint通过蒸馏减少推理步数；Dense2MoE通过MoE减少每步的计算量——两者正交可叠加
- **vs. 传统pruning**：Dense2MoE保留模型容量只减激活参数，pruning直接删除参数——在高压缩比下Dense2MoE显著更优

## 启发与关联
- Dense-to-MoE转换范式可以推广到视频DiT——视频模型参数更大，稀疏化需求更迫切
- 与REPA-E结合：端到端训练的VAE+MoE DiT可能进一步提升效率

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在DiT上做dense-to-MoE，Taylor初始化+MoB设计有创意
- 实验充分度: ⭐⭐⭐⭐ FLUX.1实际验证，与pruning全面对比
- 写作质量: ⭐⭐⭐⭐ 方法pipeline清晰
- 价值: ⭐⭐⭐⭐ 为大型DiT的高效部署提供了新的技术路径
