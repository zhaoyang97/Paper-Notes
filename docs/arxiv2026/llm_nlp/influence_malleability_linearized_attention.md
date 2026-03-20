# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**会议**: arXiv 2026  
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)  
**作者**: Jose Marie Antonio Miñoza, Paulo Mario P. Medina, Sebastian C. Ibañez
**代码**: 待确认  
**领域**: LLM/NLP  
**关键词**: influence, malleability, linearized, attention, dual  

## 一句话总结
由于注意力机制复杂的非线性动力学，理解注意力机制的理论基础仍然具有挑战性。
## 背景与动机
Understanding the theoretical foundations of attention mechanisms remains challenging due to their complex, non-linear dynamics.. This work reveals a fundamental trade-off in the learning dynamics of linearized attention.

## 核心问题
由于注意力机制复杂的非线性动力学，理解注意力机制的理论基础仍然具有挑战性。
## 方法详解

### 整体框架
- This work reveals a fundamental trade-off in the learning dynamics of linearized attention.
- Using a linearized attention mechanism with exact correspondence to a data-dependent Gram-induced kernel, both empirical and theoretical analysis through the Neural Tangent Kernel (NTK) framework shows that linearized attention does not converge to its infinite-width NTK limit, even at large widths.
- A spectral amplification result establishes this formally: the attention transformation cubes the Gram matrix's condition number, requiring width $m = Ω(κ^6)$ for convergence, a threshold that exceeds any practical width for natural image datasets.

### 关键设计
1. **关键组件1**: Understanding the theoretical foundations of attention mechanisms remains challenging due to their complex, non-linear dynamics.
2. **关键组件2**: Using a linearized attention mechanism with exact correspondence to a data-dependent Gram-induced kernel, both empirical and theoretical analysis through the Neural Tangent Kernel (NTK) framework shows that linearized attention does not converge to its infinite-width NTK limit, even at large widths.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 待深读后补充

## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐
