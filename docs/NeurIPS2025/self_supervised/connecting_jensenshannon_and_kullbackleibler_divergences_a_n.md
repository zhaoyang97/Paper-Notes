# Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.20644](https://arxiv.org/abs/2510.20644)  
**代码**: 有（GitHub，详见论文）  
**领域**: 自监督学习 / 信息论  
**关键词**: mutual information, Jensen-Shannon divergence, KL divergence, variational bound, representation learning, information bottleneck  

## 一句话总结
推导了一般情况下 KL 散度关于 JS 散度的新的紧致可计算下界，证明最大化 JSD 目标等价于最大化互信息的一个下界，为判别式学习在 MI 基础表示学习中的使用提供了理论基础，并在 MI 估计和 Information Bottleneck 中验证了其紧致性和实用性。

## 背景与动机
互信息（MI）是表示学习中衡量统计依赖性的基础度量。但直接优化 MI（通过 KL 散度定义）通常不可计算。大量方法转而最大化 JSD（Jensen-Shannon 散度）作为替代目标（如通过判别器区分联合分布与边际分布的乘积）。但关键理论问题未解决：**最大化 JSD 是否真的在最大化 MI？两者之间的定量关系是什么？**

先前已知 JSD ≤ MI（JSD 是 MI 的下界），但此界不紧致。Pinsker 不等式给出了 $\text{KLD} \geq 2 \cdot \text{JSD}$ 但仅在低 MI 时紧致。缺乏一个一般性的、紧致的 KLD(JSD) 下界。

## 核心问题
如何建立 KL 散度与 JS 散度之间的**紧致定量关系**，从而理论上证明 JSD 最大化对 MI 最大化的有效性？

## 方法详解

### 核心理论贡献
1. **新的 KLD 下界**: 推导了一般情况下 $\text{KLD}(P \| Q) \geq f(\text{JSD}(P \| Q))$ 的紧致下界。该下界：
   - 对一般分布 P, Q 成立（不限于联合/边际分布）
   - 在高/低 MI 区域均保持紧致性
   - 使用变分公式，可通过训练二分类器实际计算

2. **MI 下界**: 将上述结论特化到联合分布和边际乘积时，得到 $\text{MI}(X;Y) \geq f(\text{JSD}(P_{XY} \| P_X P_Y))$，证明最大化 JSD 必然增加 MI 的一个保证下界。

3. **二分类器连接**: 证明训练一个区分联合样本 $(x,y) \sim P_{XY}$ 和边际样本 $(x,y) \sim P_X P_Y$ 的二分类器（最小化交叉熵损失）等价于求 JSD 的已知变分下界，从而将判别式训练与 MI 最大化正式联系起来。

### 实验验证
1. **MI 估计**: 在多维高斯分布等已知 MI 的 benchmark 场景下，与 MINE、NWJ、InfoNCE 等 SOTA 神经 MI 估计器对比。新下界提供了稳定、低方差的 MI 下界估计，在高维场景下尤其优势明显。

2. **Information Bottleneck**: 在 IB 框架中使用新下界替代传统 MI 估计，展示了实际应用价值——更稳定的优化和更好的表示质量。

## 实验关键数据
- 在多通道高斯场景中，新下界在所有 MI 水平上均紧致，且方差远低于 MINE 和 NWJ
- 与 InfoNCE 相比：InfoNCE 受 batch size 限制（$\text{MI} \leq \log N$），新下界无此限制
- IB 框架中使用新下界获得了更好的压缩-预测权衡曲线

## 亮点
- 理论贡献扎实——首次给出一般性的 KLD(JSD) 紧致下界
- 为"为什么判别式 SSL 方法有效"提供了严格理论解释
- 新估计器稳定低方差，无需对抗训练，实用性强
- 覆盖了理论推导 + MI 估计 + IB 应用三个层次的验证

## 局限性 / 可改进方向
- 理论贡献为主，缺乏在大规模视觉 SSL（如 SimCLR/BYOL）上的直接应用实验
- 下界虽紧致但仍是下界，在某些场景下可能低估 MI
- JSD 估计器本身需要足够好的判别器

## 与相关工作的对比
- **vs MINE（Belghazi et al.）**: MINE 直接估计 KLD，方差大且需对抗训练；新方法通过 JSD 间接下界 KLD，更稳定
- **vs InfoNCE（van den Oord et al.）**: InfoNCE 受 $\log N$ 约束（N 为 batch size）；新下界无此限制
- **vs Pinsker 不等式**: Pinsker 给出 $\text{KLD} \geq 2 \cdot \text{JSD}$ 但仅在小散度时紧致；新下界在所有范围紧致

## 启发与关联
- 为 Adv-SSL（同批次笔记）的对比学习目标提供了理论支撑——JSD 最大化确实在做 MI 最大化
- 可用于分析 CLIP 对比训练的理论性质
- Information Bottleneck 的改进可用于 VLM 的特征压缩

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次建立一般性 KLD-JSD 紧致下界，理论贡献突出
- 实验充分度: ⭐⭐⭐ MI 估计 benchmark 全面，但缺乏大规模视觉实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰严谨
- 价值: ⭐⭐⭐⭐ 为判别式 SSL 方法提供了缺失的理论基础
