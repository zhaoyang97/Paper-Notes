# EB-JDAT: Energy-based Joint Distribution Adversarial Training

**会议**: CVPR 2026  
**arXiv**: [2505.19459](https://arxiv.org/abs/2505.19459)  
**代码**: [GitHub](https://github.com/yujkc/EB-JDAT)  
**领域**: AI安全 / 对抗鲁棒性 / 能量模型  
**关键词**: adversarial training, energy-based model, JEM, robustness, generation  

## 一句话总结
通过能量景观分析揭示AT和JEM的互补性(AT缩小clean-adv能量差→鲁棒性；JEM缩小clean-generated能量差→生成+精度)，提出EB-JDAT建模联合分布p(x,x̃,y)，用min-max能量优化对齐三种数据的能量分布——CIFAR-10上鲁棒性68.76%(AutoAttack, 超SOTA AT +10.78%)，同时保持90.39%清洁精度和竞争力的生成质量(FID=27.42)。

## 背景与动机
分类器面临精度-鲁棒性-生成能力的三难困境。AT(PGD/TRADES)鲁棒但牺牲精度、无生成能力；JEM将softmax分类器重新解释为EBM实现分类+生成但鲁棒性不足。能量景观分析揭示根本原因：AT让clean-adv能量分布重叠；JEM让clean-generated能量分布重叠。如果三者能量都对齐→统一三种能力。

## 核心问题
能否用单个模型同时实现高精度、对抗鲁棒性和生成能力？

## 方法详解

### 整体框架
将JEM的联合分布p(x,y)扩展为p(x,x̃,y)=p(y|x̃,x)·p(x̃|x)·p(x)。p(y|x̃,x)=鲁棒分类(CE on adversarial)；p(x)=数据分布(SGLD采样+能量最大似然)；p(x̃|x)=对抗分布(min-max能量优化)。

### 关键设计
1. **Min-Max能量优化for p(x̃|x)**: Inner max: 沿能量上升方向(反向SGLD)采样对抗样本—把它们推到高能量区域(类似对抗攻击但能量视角)。Outer min: 最小化clean-adv能量差—把对抗样本拉回低能量区域。这捕获了完整的对抗数据分布而非仅优化分类损失p(y|x̃)。

2. **梯度组合**: h_θ = w₁·∂log p(x)/∂θ + w₂·∂log p(x̃|x)/∂θ + w₃·∂log p(y|x̃,x)/∂θ。三项分别处理生成能力、能量对齐和鲁棒分类。消融表明w₂(能量对齐)是关键——没有它会崩溃。

3. **兼容多种JEM**: EB-JDAT是通用优化框架，可即插即用到JEM++或SADAJEM——利用其更快更稳定的训练。

### 训练策略
WRN28-10, SGLD采样, 对抗采样5步, lr=0.01, perturbation 8/255 ℓ∞, 100 epochs, 3090 GPU。

## 实验关键数据

### 对比SOTA AT方法(Tab.2, CIFAR-10, WRN28-10)
| 方法 | Clean↑ | PGD-20↑ | AutoAttack↑ |
|------|--------|---------|-------------|
| TRADES | ~84 | ~56 | ~53 |
| LAS-AWP | 87.74 | 60.16 | 55.52 |
| DHAT-CFA | 84.49 | 62.38 | 54.05 |
| **EB-JDAT-SADAJEM** | **90.37** | **68.76** | **66.12** |

### 对比使用生成数据的AT(Tab.3)
| 方法 | 额外数据 | Epochs | Clean | AA | GPU时间 |
|------|---------|--------|-------|-----|---------|
| Better DM | 1M | 400 | 91.12 | 63.35 | 1438h |
| [Gowal'21] | 100M | 2000 | 87.50 | 63.38 | 719460h |
| **EB-JDAT-SADAJEM** | **0** | **100** | 90.39 | **66.30** | **66.64h** |

### 三方面统一(Tab.5, CIFAR-10)
| 方法 | Acc↑ | AA↑ | FID↓ | IS↑ |
|------|------|------|------|------|
| JEM | 92.90 | 4.28 | 38.40 | 8.76 |
| SADAJEM | 96.03 | 29.63 | 17.38 | 8.07 |
| WEAT(IF) | 83.36 | 49.02 | 177.92 | 3.50 |
| **EB-JDAT-SADAJEM** | 90.39 | **66.30** | **27.42** | 8.05 |

### 消融要点
- **w₂=0(无能量对齐)**: 退化为标准AT(88.95/62.96/FID 173)且epoch 41崩溃
- **w₁=0(无生成)**: 仍可训练(89.84/64.69)但FID较差(42.57)
- **对抗采样步数**: K=5最优平衡，K=1鲁棒性不足，K≥10训练不稳定

## 亮点 / 我学到了什么
- **能量景观分析作为统一视角**: 用能量分布重叠度量来解释AT(缩小clean-adv)和JEM(缩小clean-generated)的本质——这是一个非常优雅的诊断框架
- **p(x̃|x)的min-max能量优化**: 不是传统AT的max-CE(找最误导的样本)，而是max-energy gap(找能量最高的样本)后min-energy gap(拉回)。在能量视角下AT变成了一个条件EBM的训练问题
- **无额外数据超越使用百万级生成数据的AT**: 核心原因是EB-JDAT作为生成模型能更准确估计clean/adv数据密度的梯度
- **干打雷不下雨的发现**: w₂(能量对齐)是防止崩溃的关键——更强调了将AT和EBM本质统一的必要性

## 局限性 / 可改进方向
- 在高维复杂数据(full ImageNet)上训练仍不稳定——EBM的固有问题
- 推荐只在JEM++/SADAJEM上使用(原始JEM太不稳定)
- 未与最新的diffusion-based AT方法在full-scale上对比
- 生成质量与专门的生成模型仍有差距(FID 27 vs DDPM<10)

## 与相关工作的对比
- **vs JEAT**: 只建模p(x̃,y)忽略clean-adv关系。EB-JDAT建模完整p(x,x̃,y)
- **vs WEAT**: 重新解释TRADES为EBM但仍是判别模型p(y|x̃,x)。EB-JDAT同时建模生成分布
- **vs Better DM/SCORE**: 用diffusion模型生成额外训练数据做AT。EB-JDAT不需要额外数据→计算效率高20x+

## 与我的研究方向的关联
- 能量景观分析可能用于理解VLM中不同模态数据的分布关系
- "统一判别-生成-鲁棒"的思路在多模态领域也有潜力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ p(x,x̃,y)的联合建模和min-max能量优化是新的理论贡献，能量景观分析提供深刻洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集、与10+种AT和JEM方法对比、生成数据AT对比、detailed消融、梯度遮蔽分析
- 写作质量: ⭐⭐⭐⭐ 动机论证(Fig.2/Tab.1)有说服力，但部分符号较重
- 对我的价值: ⭐⭐⭐ 对抗鲁棒性非核心方向，但能量视角和统一框架思路有理论价值
