# Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets

**会议**: ICLR 2026  
**arXiv**: [2510.02818](https://arxiv.org/abs/2510.02818)  
**代码**: https://github.com/Sung-Ho-Jo/hierarchical-dro  
**领域**: AI安全 / 鲁棒学习  
**关键词**: spurious correlation, Group DRO, Wasserstein, hierarchical ambiguity, robustness

## 一句话总结
提出层次化歧义集的分布鲁棒优化方法，同时建模组间比例变化和组内分布偏移（Wasserstein 球），在少数群组内部分布偏移场景下显著优于 Group DRO（CelebA shifted: 56.3%→72.1% worst-group accuracy）。

## 研究背景与动机

1. **领域现状**：Group DRO 最大化最差组性能缓解虚假相关性，但假设组内分布不变。
2. **现有痛点**：真实场景中少数群组的训练/测试分布可能不同（如训练中水鸟=水禽，测试出现海鸟），Group DRO 严重退化。
3. **核心idea一句话**：两层层次化歧义集——外层组间比例变化 ($\beta$)，内层每组 Wasserstein 球偏移 ($\epsilon_g = \epsilon/\sqrt{n_g}$)。

## 方法详解

### 关键设计
1. **层次化歧义集**: $\mathcal{Q} = \{\sum_g \beta_g Q_g : \beta \in \Delta^{m-1}, W_\infty(Q_g, P_g) \leq \epsilon_g\}$
2. **三步坐标下降**: z' 潜空间投影梯度上升 → β 指数梯度上升 → θ 加权 SGD
3. **自适应扰动半径**: $\epsilon_g \propto 1/\sqrt{n_g}$——样本少的组允许更大偏移

## 实验关键数据

### 主实验

| 数据集 | 方法 | Worst Acc | Avg Acc |
|--------|------|----------|--------|
| Shifted CelebA | PDE | 56.3±11.2 | 91.6 |
| Shifted CelebA | **Ours** | **72.1±2.0** | 91.3 |
| Shifted Waterbirds | GroupDRO | 91.7±0.3 | 94.9 |
| Shifted Waterbirds | **Ours** | **93.7±0.2** | 94.6 |
| Shifted CMNIST | GroupDRO | 65.9±8.2 | 74.0 |
| Shifted CMNIST | **Ours** | **71.8±2.8** | 75.0 |

### 关键发现
- 标准设置下也优于 Group DRO（Waterbirds 95.1 vs 94.1）
- Grad-CAM：Group DRO 聚焦躯干/背景（虚假），本方法分散到翅膀/喙/脚（真实特征）
- LISA 在偏移下暴跌 32%

## 亮点与洞察
- **组内偏移是被忽视的现实问题**——本文填补这一空白
- $\epsilon_g \propto 1/\sqrt{n_g}$ 精妙——样本少不确定性大，允许更大偏移

## 局限性 / 可改进方向
- 主要在图像数据集验证，文本场景的组定义更模糊
- 理论上界依赖 pushforward 密度假设

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化歧义集自然扩展 Group DRO
- 实验充分度: ⭐⭐⭐⭐ 标准+偏移设置、Grad-CAM 分析
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 填补 DRO 在组内偏移场景的空白
