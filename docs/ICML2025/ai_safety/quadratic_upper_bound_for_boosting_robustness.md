---
title: >-
  [论文解读] Quadratic Upper Bound for Boosting Robustness
description: >-
  [ICML2025][AI安全][对抗训练] 利用交叉熵损失关于 logit 的凸性，推导出对抗训练损失的二次上界 (QUB)，作为即插即用的损失函数替换应用于现有快速对抗训练方法，显著提升鲁棒性。 对抗训练 (AT) 是防御对抗攻击的主流方法，其核心是求解 min-max 优化： $$\min_\theta \max_{\…
tags:
  - "ICML2025"
  - "AI安全"
  - "对抗训练"
  - "快速对抗训练"
  - "二次上界"
  - "损失函数平滑"
  - "鲁棒性"
---

# Quadratic Upper Bound for Boosting Robustness

**会议**: ICML2025  
**arXiv**: [2601.13645](https://arxiv.org/abs/2601.13645)  
**代码**: 待确认  
**领域**: 对抗鲁棒 / 快速对抗训练  
**关键词**: 对抗训练, 快速对抗训练, 二次上界, 损失函数平滑, 鲁棒性

## 一句话总结

利用交叉熵损失关于 logit 的凸性，推导出对抗训练损失的二次上界 (QUB)，作为即插即用的损失函数替换应用于现有快速对抗训练方法，显著提升鲁棒性。

## 研究背景与动机

对抗训练 (AT) 是防御对抗攻击的主流方法，其核心是求解 min-max 优化：

$$\min_\theta \max_{\|\delta\|_p \le \epsilon} \mathcal{L}(f_\theta(x+\delta), y)$$

多步攻击 (如 PGD) 能生成强对抗样本但计算代价高。**快速对抗训练 (FAT)** 使用单步攻击 (如 FGSM) 降低训练时间，但因对抗空间探索不足，常遭遇**灾难性过拟合**——模型对训练时的攻击过度鲁棒，对未见攻击失去防御力。

**核心动机**：现有 FAT 方法多聚焦于设计更好的扰动生成策略或正则项，本文另辟蹊径——从损失函数本身出发，推导 AT 损失的上界来替代原始损失函数，无需加强内层极大化即可提升鲁棒性。

## 方法详解

### 二次上界推导

交叉熵损失关于 logit 向量 $f(x)$ 是凸函数。利用这一性质，对 AT 损失进行 Taylor 展开并结合 Hessian 的上界，得到 **Lemma 1**：

$$\mathcal{L}(f(x+\delta)) \le \mathcal{L}(f(x)) + (f(x+\delta)-f(x))^T \nabla_f \mathcal{L}(f(x)) + \frac{\|\boldsymbol{H}\|_2}{2}\|f(x+\delta)-f(x)\|_2^2$$

其中 $\|\boldsymbol{H}\|_2$ 是损失 Hessian 的谱范数。**Lemma 2** 进一步证明 $\|\boldsymbol{H}\|_2 \le \frac{1}{2}$。代入后得到 **QUB 损失**：

$$\mathcal{L}_{\text{QUB}} = \underbrace{\mathcal{L}(f(x))}_{\text{干净样本损失}} + \underbrace{(f(x+\delta)-f(x))^T \nabla_f \mathcal{L}(f(x))}_{\text{扰动导致的损失增量}} + \underbrace{\frac{1}{4}\|f(x+\delta)-f(x)\|_2^2}_{\text{logit 变化的二次惩罚}}$$

### 三项的直觉解释

| 项 | 作用 | 联系 |
|---|---|---|
| 第一项 $\mathcal{L}(f(x))$ | 驱动模型提升标准精度 (SA) | 关注干净样本 |
| 第二项 | 使扰动方向远离梯度方向，平滑损失面 | 类似输入梯度正则 |
| 第三项 $\|f(x+\delta)-f(x)\|_2^2$ | 限制扰动对 logit 的影响 | 类似 TRADES 的思想 |

**计算优势**：梯度 $\nabla_f \mathcal{L}$ 有闭式解 $\hat{y} - y$（softmax 减 one-hot），无需额外反向传播；所有项在 $\mathbb{R}^C$ 空间操作（$C$ 为类别数），远小于输入空间 $\mathbb{R}^{c \times H \times W}$。

### 训练策略

- **QUB-static**：全程使用 QUB 损失替换 AT 损失
- **QUB-decreasing**：早期用 QUB 损失，随训练推进线性过渡到 AT 损失

$$\mathcal{L}_{\text{total}} = (1-\lambda_t) \cdot \mathcal{L}_{\text{QUB}} + \lambda_t \cdot \mathcal{L}_{\text{AT}}, \quad \lambda_t = t/T$$

动机：QUB 的上界性质早期可快速提升鲁棒性，但后期梯度过强会过度正则化，牺牲标准精度。渐进过渡兼顾鲁棒与泛化。

## 实验关键数据

### CIFAR-10 + ResNet18 鲁棒精度 (%)

| 方法 | Step | SA | PGD-50/10 | AA | 时间(h) |
|---|---|---|---|---|---|
| FGSM-CKPT | 1 | 90.02 | 37.42 | 37.22 | 1.05 |
| + QUB-static | 1 | 87.63 | 42.54 | **41.53** | 1.35 |
| + QUB-decreasing | 1 | 88.56 | 40.70 | 39.85 | 1.35 |
| FGSM-GA | 1 | 82.93 | 47.74 | 45.75 | 3.02 |
| + QUB-static | 1 | 79.75 | **50.82** | **47.33** | 3.27 |
| N-FGSM | 1 | 81.21 | 47.36 | 45.17 | 0.58 |
| + QUB-static | 1 | 80.76 | **49.60** | **47.00** | 0.70 |
| FGSM-PGI(MEP) | 1 | 81.48 | 51.75 | 48.41 | 0.89 |
| + QUB-decreasing | 1 | 81.56 | 52.24 | **48.58** | 1.19 |
| PGD-AT | 10 | 81.53 | 51.82 | 48.33 | 2.34 |
| + QUB-static | 10 | 80.24 | **53.39** | **49.91** | 2.64 |
| TRADES | 10 | 82.11 | 52.77 | 50.16 | 3.50 |

**关键发现**：

- 除 FGSM-RS 外，所有基线方法 +QUB 后鲁棒精度均有提升
- FGSM-CKPT +QUB-static 在 AA 上从 37.22% 提升到 **41.53%**（+4.31%），提升最显著
- PGD-AT +QUB-static 达到 49.91% AA，接近 TRADES 的 50.16%，但训练时间更短
- QUB-static 鲁棒性更优但牺牲标准精度，QUB-decreasing 更均衡
- 训练时间增加约 20-30%，远低于升级到多步攻击的代价

### 损失面可视化

QUB 训练后的模型损失面显著更平坦，表明模型对输入周围更大区域的扰动保持稳定预测，印证了 QUB 通过平滑损失面增强鲁棒性的机制。

## 亮点与洞察

1. **理论优雅**：从交叉熵凸性出发的推导干净简洁，QUB 三项各有明确物理意义
2. **即插即用**：仅需替换损失函数，与现有 FAT 方法正交兼容，实现门槛低
3. **计算高效**：$\nabla_f \mathcal{L}$ 有闭式解，所有中间量在类别维度 $\mathbb{R}^C$ 上，内存和计算开销小
4. **普适性强**：在 9 种 FAT 基线中 8 种有效（除 FGSM-RS），且多步 PGD-AT 也受益
5. **渐进策略**：QUB-decreasing 通过简单线性调度平衡鲁棒性与泛化，无需额外调参

## 局限与展望

1. **FGSM-RS 失效**：当基线方法本身对抗样本质量低时，QUB 反而在非信息区域平滑，放大缺陷
2. **SA 下降**：QUB-static 在多数方法上牺牲 2-5% 标准精度换鲁棒性，QUB-decreasing 缓解有限
3. **大规模验证不足**：仅在 CIFAR-10/100 和 Tiny ImageNet 上实验，未验证 ImageNet 等大规模数据
4. **仅限 $\ell_\infty$ 攻击**：未探索 $\ell_2$、$\ell_1$ 等其他攻击范数下的表现
5. **上界紧度**：Hessian 谱范数取全局上界 $1/2$，可能不够紧，自适应估计或许更优
6. **与 TRADES 的关系**：第三项与 TRADES 的 KL 散度正则有相似思路，直接与 TRADES 结合的效果未探索

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从损失函数凸性推导上界的视角新颖，与现有方法在优化目标层面形成互补
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 9 种基线、3 个数据集、多种攻击和模型，消融详尽
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，三项解释直观
- 价值: ⭐⭐⭐⭐ — 即插即用的损失替换具有很强的工程实用性，但大规模场景验证是瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[CVPR 2026\] A Provable Energy-Guided Test-Time Defense Boosting Adversarial Robustness of Large Vision-Language Models](../../CVPR2026/ai_safety/a_provable_energy-guided_test-time_defense_boosting_adversarial_robustness_of_la.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](../../NeurIPS2025/ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](../../NeurIPS2025/ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)

</div>

<!-- RELATED:END -->
