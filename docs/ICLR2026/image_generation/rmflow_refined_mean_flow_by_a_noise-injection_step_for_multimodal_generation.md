# RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation

**会议**: ICLR 2026  
**arXiv**: [2602.00849](https://arxiv.org/abs/2602.00849)  
**代码**: 无  
**领域**: 扩散模型 / 单步生成 / Mean Flow 改进  
**关键词**: mean flow, noise injection refinement, 1-NFE, likelihood maximization, multimodal generation  

## 一句话总结
提出 RMFlow，在 1-NFE MeanFlow 传输后加入一步噪声注入精炼来弥补单步传输的误差，同时在训练中加入最大似然目标来最小化学习分布与目标分布间的 KL 散度，在 T2I、分子生成、时间序列生成上实现接近 SOTA 的 1-NFE 结果。

## 研究背景与动机

1. **领域现状**：MeanFlow 通过学习平均速度场实现少步生成，无需预训练或蒸馏。但 1-NFE 时性能显著下降——单步传输不够精确，生成样本偏离目标分布。

2. **现有痛点**：1-NFE MeanFlow 在高斯混合分布上偏差大、在分子生成上生成无效结构（断裂分子）。多步（8/32 NFE）效果好但失去了效率优势。

3. **核心矛盾**：1-NFE 传输的确定性输出偏离真实分布（因为平均速度近似带误差），但不能增加 NFE。

4. **本文要解决什么？** 在保持 1-NFE 的前提下改善 MeanFlow 的生成质量。

5. **切入角度**：将 1-NFE 看作"粗传输"，然后加一步噪声注入来"精炼"——本质上是将 MeanFlow 的确定性输出转化为概率性输出，用噪声来补偿传输误差。训练时额外加入最大似然目标以最小化 KL 散度。

6. **核心 idea 一句话**：1-NFE MeanFlow 的确定性输出 + 高斯噪声注入 ≈ 更好的目标分布近似。

## 方法详解

### 整体框架
$x_{\text{gen}} = x_0 + \hat{u}_{0,1}(x_0; \theta) + \sigma \epsilon$（一步传输+噪声注入），训练损失 = MeanFlow 损失 + $\lambda_1$ 负对数似然损失 + $\lambda_2$ guidance 正则化。

### 关键设计

1. **噪声注入精炼**：1-NFE 传输后加 $\sigma \epsilon$（$\epsilon \sim \mathcal{N}(0, I)$），将确定性点估计转为带方差的分布，弥补传输误差
2. **最大似然训练目标**：$\mathcal{L}_{\text{NLL}} = \mathbb{E}[\|x_{\text{tgt}} - (x_0 + \hat{u}_{0,1})\|^2]$，理论证明该损失下界最大化目标分布的期望对数似然
3. **联合损失**：MeanFlow 损失控制概率路径的 Wasserstein 距离，NLL 损失控制终端分布的 KL 散度

### 损失函数 / 训练策略
- 大规模任务先训 MeanFlow，再用 LoRA 微调加入 NLL 损失
- T2I: COCO 数据集，480M U-Net，SD-VAE 潜空间
- 分子生成: QM9 + Geom-Drugs

## 实验关键数据

### 主实验

| 方法 | NFE | Mixture Gaussian TV ↓ | QM9 分子稳定性 ↑ |
|------|-----|---------------------|----------------|
| MeanFlow | 1 | 1.44 | 低（生成断裂） |
| MeanFlow | 8 | 0.80 | 中等 |
| MeanFlow | 32 | 0.67 | 高 |
| **RMFlow** | **1** | **0.76** | **接近 32-NFE** |

T2I (COCO FID-30K): RMFlow 达到与 Distilled SD、StyleGAN-T 可比的 FID，且不需要辅助模型。

### 关键发现
- 1-NFE RMFlow 超越 8-NFE MeanFlow（TV 0.76 vs 0.80），接近 32-NFE
- 分子生成中噪声注入有效避免了结构断裂
- 训练成本与 MeanFlow 相当（噪声注入几乎零开销）

## 亮点与洞察
- **极简改进**：只加一步 $\sigma\epsilon$ 就显著改善 1-NFE 质量——本质上是将点估计问题转为分布估计，用噪声补偿模型误差。
- **多模态通用**：同一框架处理图像、分子、时间序列，说明噪声注入精炼是模态无关的通用技巧。
- 理论上证明了 NLL 损失与 KL 散度的联系，为噪声注入提供了原理性解释。

## 局限性 / 可改进方向
- 噪声注入的 $\sigma$ 是超参数需要调优
- T2I 实验用 COCO + 小 U-Net，缺少 ImageNet/大模型验证
- 与 SoFlow、TwinFlow 等最新单步方法缺少直接对比
- 噪声注入是否在所有任务上都有益？高维图像中可能引入模糊

## 相关工作与启发
- **vs MeanFlow**: 1-NFE RMFlow > 8-NFE MeanFlow，核心改进是噪声注入+NLL 损失
- **vs SoFlow**: 不同的改进思路——SoFlow 学解函数，RMFlow 学平均速度+精炼

## 评分
- 新颖性: ⭐⭐⭐ 噪声注入思路简单，但在 MeanFlow 上是首次
- 实验充分度: ⭐⭐⭐⭐ 多模态验证（图像/分子/时间序列），但图像实验规模小
- 写作质量: ⭐⭐⭐⭐ 清晰有条理，理论推导严谨
- 价值: ⭐⭐⭐⭐ 提供了一个简单有效的 MeanFlow 改善方案
