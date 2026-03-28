# Attention (as Discrete-Time Markov) Chains

**会议**: NeurIPS 2025  
**arXiv**: [2507.17657](https://arxiv.org/abs/2507.17657)  
**代码**: https://yoterel.github.io/attention_chains/  
**领域**: 注意力分析 / 可视化  
**关键词**: 注意力Markov链, TokenRank, PageRank, 多跳注意力, 图像分割

## 一句话总结
将 softmax 归一化后的注意力矩阵重新解读为离散时间 Markov 链（DTMC）的转移概率矩阵，提出多跳注意力（Multi-Bounce）和 TokenRank（稳态分布，类似 PageRank）来捕获间接注意力路径和全局 token 重要性，在 ImageNet 分割上达 94.29% mAP，并增强 Self-Attention Guidance 的图像生成质量。

## 研究背景与动机

1. **领域现状**：注意力分析依赖直接操作——行选择（token 关注谁）、列选择（谁关注 token）、求和（全局聚合）。这些只捕获一阶/直接注意力效应。
2. **现有痛点**：像 PageRank 在网页链接中的启示——直接超链接计数不如传播式重要性评估。注意力也有间接影响路径：token A 关注 B，B 关注 C，但直接操作看不到 A→C 的间接关系。
3. **核心矛盾**：注意力矩阵的行列操作是一阶的，忽略了高阶间接路径。需要一种数学框架统一直接和间接注意力效应。
4. **本文要解决什么？** 提供基于 Markov 链的注意力分析框架，捕获多阶间接依赖和全局 token 重要性。
5. **切入角度**：softmax 归一化的注意力矩阵天然满足 Markov 链转移概率的定义（行和为 1，非负），可以直接复用 Markov 链/PageRank 的理论工具。
6. **核心 idea 一句话**：注意力矩阵 = DTMC 转移矩阵 → 多跳传播 = k 次矩阵幂 → 稳态向量 = TokenRank（类比 PageRank）→ $\lambda_2$ 加权混合多头。

## 方法详解

### 整体框架
注意力矩阵 $A$（softmax 输出）→ **Multi-Bounce**: $\mathbf{v}_{i,n+1}^T = \mathbf{v}_{i,n}^T A$（n=1 即标准行选择）→ **TokenRank**: 求 $A$ 的稳态向量 $\pi$（满足 $\pi^T A = \pi^T$）→ **$\lambda_2$ 加权**: 用第二大特征值大小加权不同注意力头的贡献 → 应用于分割/生成/token masking

核心洞察：softmax 归一化后的注意力矩阵天然满足 Markov 链转移概率定义（行和为 1，非负），PageRank 的所有理论工具可直接迁移。

### 关键设计

1. **多跳注意力（Multi-Bounce Attention）**:
   - 做什么：通过矩阵幂传播捕获间接注意力路径
   - 核心思路：对 token $i$，初始化 one-hot 向量 $\mathbf{v}_{i,0} = \mathbf{e}_i$，迭代 $\mathbf{v}_{i,n+1}^T = \mathbf{v}_{i,n}^T A$。$n=1$ 是标准行选择（直接关注），$n=2$ 加入二阶间接路径，$n \to \infty$ 收敛到稳态
   - 设计动机：图像分割中一个像素可能不直接关注其所属物体的中心，但通过中间像素间接关联——多跳捕获这种间接关系

2. **TokenRank（稳态分布）**:
   - 做什么：计算每个 token 的全局重要性（类比 PageRank）
   - 核心思路：用幂法或 PageRank 修正（$P' = \alpha P + (1-\alpha)\frac{1}{n}\mathbf{e}\mathbf{e}^T$ 保证遍历性和原始性）求稳态向量 $\pi$。$\pi(i)$ 越高表示 token $i$ 在注意力传播中越"重要"
   - 设计动机：关注于 token 的全局影响力而非局部关系——"哪些 token 在整个注意力图中最核心？"

3. **$\lambda_2$ 加权多头混合**:
   - 做什么：用第二大特征值 $\lambda_2$ 加权不同注意力头的贡献
   - 核心思路：$\lambda_2$ 越大表示 Markov 链收敛越慢（更多亚稳态），对应注意力头有更丰富的多尺度结构。用 $\lambda_2$ 加权给这些头更大权重
   - 设计动机：不是所有注意力头对下游任务同等重要——$\lambda_2$ 提供了无监督的头重要性估计

### 损失函数 / 训练策略
- 纯分析方法，无训练
- 通过 10-20 次幂迭代计算 TokenRank

## 实验关键数据

### 主实验

| 任务 | 方法 | Accuracy | mIoU | mAP |
|------|------|----------|------|-----|
| ImageNet分割 | Concept Attention | 83.07% | 71.04% | — |
| | **Ours (FLUX DiT)** | **84.12%** | 70.20% | **94.29%** |
| SAG图像生成 | SD1.5 base | IS 16.32 | — | — |
| | **TokenRank** | IS **18.37** | — | — |
| DiffSeg | Uniform sampling | 72.50 mACC | 43.60 mIoU | — |
| | **TokenRank grid** | **84.97** mACC | **44.87** mIoU | — |

### 消融实验

| 配置 | 结果 |
|------|------|
| $\lambda_2$ 加权 vs 均匀 | 统计显著改善（E.1 检验） |
| n=1 (标准) vs n=2 (双跳) | n=2 在分割上最优 |
| 结构化特征(DINOv2) vs 非结构化(ViT) | DINOv2 + TokenRank 收益更大 |
| Token Masking | TokenRank 移除 → AUC 0.26-0.64（比 baseline 0.27-0.79 更快降低准确率）——证明 TokenRank 确实找到了重要 token |

### 关键发现
- 多跳注意力（n=2）在分割任务上比直接注意力（n=1）好——间接路径确实包含有用信息
- TokenRank 在 SAG 中比随机采样种子显著改善生成质量——找到"重要 token"作为引导更有效
- 结构化注意力（DINOv2 + registers）从 TokenRank 获益更大——说明 Markov 链分析需要注意力有一定结构
- $\lambda_2$ 加权提供了微小但统计显著的改善——自动头选择有价值

## 亮点与洞察
- **PageRank 到 TokenRank 的类比非常自然**：注意力矩阵的行随机性恰好满足 Markov 链定义，PageRank 的所有理论工具可以直接迁移
- **多跳注意力揭示了"注意力的注意力"**：一个 token 的真正影响不仅是它直接关注的，还包括它间接通过其他 token 到达的——这对理解 Transformer 的信息流动有深刻意义
- **统一框架涵盖多个应用**：分割（Multi-Bounce）、生成引导（TokenRank + SAG）、token 重要性分析（masking）——一个理论框架多个应用

## 局限性 / 可改进方向
- 仅适用于方阵（自注意力/混合注意力），交叉注意力有不可达状态问题——需要扩展理论
- $\lambda_2$ 计算对大矩阵较贵——但多跳和 TokenRank 效率高（10-20 次迭代）
- 在无结构注意力（如普通 ViT 无 registers）上收益有限——暗示 Markov 分析需要注意力有一定结构先验
- 未探索多跳注意力的最优跳数自动选择——当前手动设 n=2
- 未分析跨层注意力的 Markov 链特性——当前仅在单层内分析
- 稳态向量可能不唯一（周期链或不连通注意力图）——需要 PageRank 修正保证

## 相关工作与启发
- **vs Attention Rollout**: Rollout 做层间注意力传播但只沿层叠加，不做 Markov 链分析
- **vs GradCAM**: GradCAM 需要梯度，TokenRank 完全免梯度
- **vs Concept Attention**: 针对特定概念的注意力分析，TokenRank 是通用的全局重要性评估
- **可迁移性**: TokenRank 可应用于任何 Transformer——不限于视觉，文本/音频 Transformer 也适用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Markov 链视角完全新颖，PageRank→TokenRank 的迁移优雅
- 实验充分度: ⭐⭐⭐⭐ 分割+生成+masking 三个应用 + 消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，类比直观
- 价值: ⭐⭐⭐⭐ 为注意力分析提供了新的数学框架，对理解 Transformer 信息流动有深层启发