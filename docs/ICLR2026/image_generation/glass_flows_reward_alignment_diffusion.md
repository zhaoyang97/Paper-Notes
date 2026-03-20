# GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models

**会议**: ICLR 2026 Oral  
**OpenReview**: [vH7OAPZ2dR](https://openreview.net/forum?id=vH7OAPZ2dR)  
**代码**: 有  
**领域**: 图像生成 / 扩散模型  
**关键词**: flow matching, diffusion models, reward alignment, Feynman-Kac, stochastic transitions, GLASS  

## 一句话总结
提出 GLASS (Gaussian Latent Sufficient Statistic) Flows——一种在流/扩散模型的去噪过程中实现高效随机转移的新采样范式，通过充分统计量重参数化将随机转移重铸为内部 ODE 求解问题，在无需重训的条件下结合 ODE 效率和 SDE 随机性，使 Feynman-Kac Steering 在 FLUX 文生图模型上一致超越 Best-of-N 基线。

## 研究背景与动机
1. **领域现状**：奖励对齐扩散/流模型的推理方法（如 SMC、Feynman-Kac Steering）需要随机转移（SDE）来探索分布，但 SDE 采样远慢于 ODE 且降低生成质量。
2. **现有痛点**：标准 FKS 使用 SDE 转移时甚至无法超越简单的 Best-of-N ODE 基线——效率与随机性之间存在根本矛盾。
3. **核心idea一句话**：将随机转移 $p_{t'|t}$ 重铸为内部流匹配问题并用 ODE 求解，通过充分统计量 $S(\mathbf{x}) = \frac{\mu^\top \Sigma^{-1}}{\mu^\top \Sigma^{-1}\mu}\mathbf{x}$ 复用预训练模型，实现"ODE 速度 + SDE 多样性"。

## 方法详解
### 关键设计
1. **充分统计量重参数化**：利用高斯转移核的充分统计量，将去噪器输出映射为内部 ODE 的向量场，无需训练新模型
2. **相关系数 $\rho$**：控制随机转移的强度（默认 $\rho=0.4$），$\rho=1$ 退化为确定性 ODE
3. **即插即用**：作为训练无关的 drop-in replacement，可应用于任何预训练流/扩散模型

## 实验关键数据
- 在 FLUX (768×1360) 上测试，4 个奖励模型（CLIP, PickScore, HPSv2, ImageReward）
- **标准 FKS (SDE)**：无法超越 Best-of-N ODE 基线
- **FKS-GLASS**：一致超越 Best-of-N ODE，同时保持生成质量
- GenEval 和 PartiPrompts 两个基准上均有提升

## 亮点与洞察
- **解决了效率-随机性的根本矛盾**：不再需要在 ODE 速度和 SDE 多样性之间二选一
- **数学优雅**：充分统计量构造将复杂的采样问题简化为标准 ODE 求解，理论基础扎实

## 局限性 / 可改进方向
- 依赖高斯转移核假设，对非高斯架构的适用性未验证
- 超参数 $\rho$ 的最优值可能随任务变化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "流中流"的概念和充分统计量构造极其优雅
- 实验充分度: ⭐⭐⭐⭐ FLUX 验证有说服力，但缺少更多架构对比
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，直觉解释到位
- 价值: ⭐⭐⭐⭐⭐ 为扩散模型推理时奖励对齐提供了实用工具
