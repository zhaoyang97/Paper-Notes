---
title: >-
  [论文解读] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference
description: >-
  [AAAI 2026 Oral][AI安全][对抗鲁棒性] 提出方向正交反攻击（DOC）方法，通过在反攻击优化中引入正交梯度分量和动量更新扩展搜索空间，结合基于余弦相似度的方向敏感度评分自适应调控反攻击强度，在 16 个数据集上显著提升 CLIP 的测试时对抗鲁棒性。 CLIP 等视觉语言预训练模型具有强大的零样本泛化能力…
tags:
  - "AAAI 2026 Oral"
  - "AI安全"
  - "对抗鲁棒性"
  - "CLIP防御"
  - "测试时防御"
  - "正交反攻击"
  - "视觉语言模型"
---

# Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.09064](https://arxiv.org/abs/2511.09064)  
**代码**: [有](https://github.com/bookman233/DOC)  
**领域**: AI安全  
**关键词**: 对抗鲁棒性, CLIP防御, 测试时防御, 正交反攻击, 视觉语言模型

## 一句话总结

提出方向正交反攻击（DOC）方法，通过在反攻击优化中引入正交梯度分量和动量更新扩展搜索空间，结合基于余弦相似度的方向敏感度评分自适应调控反攻击强度，在 16 个数据集上显著提升 CLIP 的测试时对抗鲁棒性。

## 研究背景与动机

CLIP 等视觉语言预训练模型具有强大的零样本泛化能力，但对对抗样本极其脆弱。现有防御方法主要有三类：

**对抗微调**（如 TeCoA、PMG-AFT、FARE）：用对抗样本微调 CLIP，但计算开销大且可能损害泛化能力

**对抗 Prompt 调优**：在嵌入空间调整 Prompt，但丧失语义可解释性

**测试时反攻击（TTC）**：最新的无参数防御方法，生成反攻击扰动最大化对抗输入与其变体的嵌入距离

**TTC 的核心问题**：对抗攻击和反攻击之间存在**根本性的优化目标不匹配**：

- 对抗攻击目标：最大化分类损失
- 反攻击目标：最大化嵌入距离

TTC 使用 PGD 沿梯度方向生成反攻击，但由于目标不匹配，搜索空间被限制在狭窄区域，导致反攻击容易过拟合于有限的对抗模式，缺乏多样性来中和广泛的扰动分布。

## 方法详解

### 整体框架

DOC（Directional Orthogonal Counterattack）包含两个核心组件：

1. **正交梯度增强（OGA）**：在反攻击优化的每一步添加正交于主梯度方向的随机分量 + 动量更新
2. **方向敏感度评分（DSS）**：基于余弦相似度判断输入是否为对抗样本，自适应调控反攻击强度

### 关键设计

**正交梯度增强（OGA）**：

1. 计算归一化梯度 g（对反攻击损失关于对抗输入求梯度并归一化）
2. 从标准正态分布采样随机向量 r，通过 Gram-Schmidt 正交化得到与梯度正交的分量：r_perp = (r - <r,g>g) / ||r - <r,g>g||
3. 组合更新方向：d = g + lambda * r_perp（lambda 控制正交注入强度）
4. 动量更新：m_t = mu * m_{t-1} + (1-mu) * d
5. 反攻击扰动迭代：delta_{t+1} = Proj(delta_t + alpha * sign(m_t))

设计直觉：正交分量使反攻击探索梯度方向之外的区域，动量帮助逃离狭窄局部最优，生成更多样化的反攻击扰动。t-SNE 可视化确认 DOC 的反攻击分布比 TTC 更分散。

**方向敏感度评分（DSS）**：

TTC 使用 l2 距离判断输入是否为对抗样本，存在两个问题：(a) 方向相似但尺度不同的嵌入会导致 l2 虚高；(b) 单个噪声样本引入不稳定性。

DOC 改用余弦相似度 + 多次采样：

- tau_hat(x) = 1 - (1/M) * Sum cos(I_theta(x_m), I_theta(x))
- 低 tau_hat：扰动后嵌入方向不变，说明是干净样本
- 高 tau_hat：方向不一致，可能是对抗样本

通过软门控函数自适应调控反攻击强度：

- w = sigmoid(gamma * (tau - tau_hat(x)))
- 最终：delta_ca = w * delta_ca + (1-w) * delta_ca_0

干净样本 w 接近 0（几乎不施加反攻击），对抗样本 w 接近 1（全力反攻击）。

### 损失函数 / 训练策略

DOC 是**无训练（training-free）**的测试时防御方法：

- 不修改模型参数、不需要训练数据、不依赖标签监督
- 反攻击预算 epsilon_ca = 4/255
- 默认 4 步反攻击，步长 alpha = 3/255
- batch size 256，仅需单张 NVIDIA 4090 GPU

## 实验关键数据

### 主实验

**PGD-10 攻击下 16 个数据集的平均结果**（epsilon_atk = 4/255）：

| 方法 | 类型 | 平均鲁棒准确率 | 平均干净准确率 |
|---|---|---|---|
| CLIP（原始） | - | 0.06% | 61.51% |
| HD | 测试时防御 | 0.56% | 54.85% |
| TeCoA4 | 对抗微调 | 10.95% | 37.58% |
| FARE4 | 对抗微调 | 1.38% | 56.62% |
| TTC | 测试时防御 | 21.22% | 55.63% |
| **DOC** | **测试时防御** | **31.02%** | **58.26%** |

DOC 比 TTC 鲁棒准确率提升 **9.80%**，同时干净准确率更高（+2.63%）。

**逐数据集关键结果**（鲁棒准确率 PGD-10）：

| 数据集 | CLIP | TTC | DOC | 提升 |
|---|---|---|---|---|
| CIFAR-10 | 0.00% | 30.25% | 38.14% | +7.89% |
| STL-10 | 0.04% | 51.89% | 69.16% | +17.27% |
| ImageNet | 0.00% | 13.07% | 24.64% | +11.57% |
| OxfordPets | 0.00% | 25.89% | 46.52% | +20.63% |
| Caltech-256 | 0.13% | 26.38% | 43.08% | +16.70% |

### 消融实验

| DSS | OGA | 干净准确率 | PGD 鲁棒 | CW 鲁棒 | AutoAttack |
|---|---|---|---|---|---|
| 无 | 无 | 55.66% | 21.43% | 20.70% | 21.97% |
| 有 | 无 | 58.23% | 23.37% | 22.27% | 22.66% |
| 无 | 有 | 55.38% | 31.83% | 29.02% | 26.07% |
| 有 | 有 | **58.27%** | **31.04%** | **28.15%** | **25.89%** |

- **DSS 单独使用**：主要提升干净准确率（+2.57%），抑制对干净样本的不必要扰动
- **OGA 单独使用**：鲁棒准确率大幅提升（+10.4%），验证了多样化反攻击的有效性
- **两者结合**：同时兼顾鲁棒性和干净准确率

CW 攻击下平均鲁棒准确率：DOC 28.18% vs TTC 20.61%（+7.58%）。AutoAttack 下 DOC 较 TTC 提升约 4.1%。

### 关键发现

- DOC 在几乎所有 16 个数据集上均超越 TTC，唯一例外是 EuroSAT
- DOC 可作为**即插即用模块**与对抗微调结合：与 FARE 结合后平均鲁棒准确率超原始 CLIP 18%
- 反攻击步数仅需 N=3-4 即可饱和，计算开销极低
- 干净准确率在增加步数时保持稳定，鲁棒性提升不以干净性能为代价

## 亮点与洞察

1. **问题定位精准**：揭示了对抗攻击与反攻击之间的优化目标不匹配问题
2. **正交梯度增强设计直觉清晰**：通过正交化引入探索噪声，既数学优雅又实践有效
3. **余弦相似度替代 l2 距离**用于对抗样本识别，在高维空间中更合理（尺度不变性）
4. **完全无训练**：不需数据、不改参数、单 GPU 即可运行，部署门槛极低
5. t-SNE 可视化直观展示了 DOC 将对抗样本推向干净分布的效果

## 局限与展望

1. 反攻击预算与攻击预算设为相同值，实际场景中攻击预算未知
2. 正交分量是随机采样的，每次推理结果可能不同（虽然实验中方差较小）
3. ImageNet 上干净准确率下降（-3.25%），在细粒度分类数据集上也有波动
4. 仅在 CLIP 上验证，未扩展到其他 VLP（如 BLIP-2、LLaVA）
5. 对自适应攻击的鲁棒性未充分讨论

## 相关工作与启发

- **TTC**（Xing et al. 2025）：测试时反攻击的开创性工作，DOC 直接改进对象
- **TeCoA**（Mao et al.）：对抗微调代表方法
- **PMG-AFT**（Wang et al. 2024）：加入 CLIP 引导正则化的对抗微调
- **FARE**（Schlarmann et al. 2024）：较大预算下的对抗微调
- **Hedge Defense**（Wu et al. 2021）：最大化所有类别损失的测试时防御
- 启发：**在无监督测试时防御中，多样性比精度更重要**，正交探索思路可推广到其他鲁棒优化场景

## 评分

- 新颖性: 4/5 - 正交梯度增强和方向敏感度评分是有意义的新贡献
- 技术深度: 4/5 - 方法设计有清晰的理论动机和数学推导
- 实验充分度: 5/5 - 16 个数据集 x 3 种攻击 x 消融 x 组合实验 + 可视化
- 写作质量: 4/5 - 问题动机阐述清晰，图表丰富
- 综合: 4.0/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] When CLIP Sees More, It Fights Back Harder: Multi-View Guided Adaptive Counterattacks for Test-Time Adversarial Robustness](../../CVPR2026/ai_safety/when_clip_sees_more_it_fights_back_harder_multi-view_guided_adaptive_counteratta.md)
- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] FedMOP: Achieving Enhanced Privacy and Performance in Federated Learning via Momentum Orthogonal Projection](../../CVPR2026/ai_safety/fedmop_achieving_enhanced_privacy_and_performance_in_federated_learning_via_mome.md)
- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)

</div>

<!-- RELATED:END -->
