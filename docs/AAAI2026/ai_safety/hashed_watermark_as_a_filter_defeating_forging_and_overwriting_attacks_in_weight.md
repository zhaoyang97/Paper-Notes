---
title: >-
  [论文解读] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking
description: >-
  [AAAI 2026][AI安全][神经网络水印] 提出 NeuralMark——一种基于哈希水印过滤器的权重水印方法，利用哈希函数从秘钥生成不可逆二值水印作为私有过滤器选择嵌入参数，借助雪崩效应阻断伪造攻击的梯度反推，通过多轮过滤减少参数重叠抵御覆写攻击，在13种CNN/Transformer架构、5个图像分类和1个文本生成任务上验证了有效性和鲁棒性。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "神经网络水印"
  - "模型知识产权保护"
  - "伪造攻击"
  - "覆写攻击"
  - "哈希函数"
  - "白盒水印"
---

# Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking

**会议**: AAAI 2026  
**arXiv**: [2507.11137](https://arxiv.org/abs/2507.11137)  
**代码**: [GitHub](https://github.com/AIResearch-Group/NeuralMark)  
**领域**: AI安全/模型水印  
**关键词**: 神经网络水印, 模型知识产权保护, 伪造攻击, 覆写攻击, 哈希函数, 白盒水印

## 一句话总结

提出 NeuralMark——一种基于哈希水印过滤器的权重水印方法，利用哈希函数从秘钥生成不可逆二值水印作为私有过滤器选择嵌入参数，借助雪崩效应阻断伪造攻击的梯度反推，通过多轮过滤减少参数重叠抵御覆写攻击，在13种CNN/Transformer架构、5个图像分类和1个文本生成任务上验证了有效性和鲁棒性。

## 研究背景与动机

- **领域现状**：深度神经网络训练成本巨大（GPT-4约4000万美元），模型作为数字资产需要版权保护。神经网络水印（NNW）分为白盒（访问参数）、黑盒（查询输入输出）和无盒（仅模型输出）三类。白盒方法中，权重水印直接将水印嵌入模型参数，无需修改网络结构，兼容性最好。
- **核心痛点**：现有权重水印方法面临两类关键攻击：  
  (1) **伪造攻击**：攻击者冻结模型参数，通过梯度反传逆向推导出一对伪造的秘钥-水印，声称模型所有权——VanillaMark 和 VoteMark 都100%可被伪造  
  (2) **覆写攻击**：攻击者嵌入自己的水印覆盖原始水印——当攻击者的嵌入强度 $\lambda$ 为原始1000倍时，GreedyMark 检测率降至49.60%
- **核心矛盾**：抗伪造需要阻断梯度计算（不可逆性），抗覆写需要使不同水印选择不同参数（参数隔离性），已有方法最多只能防御其中一种攻击。
- **切入角度**：利用哈希函数的**雪崩效应**（输入微小变化→输出巨大变化）实现不可逆性，利用不同哈希水印作为**私有过滤器**实现参数隔离，一个机制同时解决两个问题。

## 方法详解

### 整体框架

NeuralMark 包含三个阶段：(1) **哈希水印生成**——从秘钥矩阵经SHAKE-256哈希函数生成二值水印；(2) **水印嵌入**——用哈希水印作为过滤器多轮筛选参数 → 平均池化 → 二元交叉熵嵌入；(3) **水印验证**——提取水印与原始水印比对检测率。

### 关键设计

1. **哈希水印过滤器（核心创新）**

    - 功能：利用哈希水印 $\mathbf{b} = \mathcal{H}(\mathbf{K})$ 作为私有过滤器，多轮筛选模型参数
    - 过滤过程：初始参数向量 $\mathbf{w}^{(0)} = \mathbf{w}$，在第 $r$ 轮，将 $\mathbf{b}$ 重复匹配参数长度得到 $\mathbf{b}^{(r)}$，仅保留 $\mathbf{b}^{(r)} = 1$ 位置的参数得到 $\mathbf{w}^{(r)}$
    - **抗伪造机理**：SHAKE-256 的雪崩效应使得秘钥微小变化导致水印巨变，梯度无法有效传播，逆向工程不可行
    - **抗覆写机理**：不同水印产生不同过滤器，多轮过滤后参数重叠率趋近0%。以 $\mathbf{b}_1 = [1,0,1,0]$ vs $\mathbf{b}_2 = [0,1,1,0]$ 为例：无过滤100%重叠 → 一轮后50%重叠 → 两轮后0%重叠

2. **平均池化增强参数鲁棒性**

    - 功能：对过滤后的参数 $\mathbf{w}^{(R)}$ 进行平均池化得到 $\widetilde{\mathbf{w}} = \text{AVG}(\mathbf{w}^{(R)})$
    - 设计动机：平均池化聚合更广区域的参数信息，使水印对微调和剪枝引起的参数扰动具有鲁棒性——单个参数被修改/置零时，平均值变化较小

3. **水印嵌入与验证**

    - 嵌入损失：$\min_\theta \mathcal{L}_m + \lambda \mathcal{L}_e(\widetilde{\mathbf{b}}, \mathbf{b})$，其中 $\widetilde{\mathbf{b}} = \delta(\widetilde{\mathbf{w}} \mathbf{K})$ 为提取水印
    - 验证条件：检测率 $\rho = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[b_i = \mathcal{T}(\tilde{b}_i)]$ 需超过安全边界**88.29%**（$n=256$ 时伪造概率 $< 1/2^{128}$）且哈希一致性 $\mathcal{H}(\mathbf{K}) = \mathbf{b}$

### 安全边界理论分析

**Proposition 1**：假设哈希函数输出均匀分布，伪造水印检测率 $\geq \rho$ 的概率上界为 $\frac{1}{2^n}\sum_{i=0}^{n-\lceil\rho n\rceil}\binom{n}{i}$。$n=256, \rho=88.29\%$ 时此概率 $< 1/2^{128}$，可忽略不计。

## 实验

### 保真度评估（分类精度/水印检测率均100%）

| 数据集 | Clean (AlexNet/ResNet-18) | NeuralMark | VanillaMark | GreedyMark | VoteMark |
|:--|:--|:--|:--|:--|:--|
| CIFAR-10 | 91.05 / 94.76 | 90.93 / 94.50 | 91.01 / 94.87 | 90.88 / 94.69 | 90.86 / 94.79 |
| CIFAR-100 | 68.24 / 76.23 | 68.57 / 76.34 | 68.43 / 76.22 | 68.31 / 76.14 | 68.53 / 76.74 |
| TinyImageNet | 42.42 / 53.48 | 42.31 / 53.22 | 42.50 / 53.36 | 42.94 / 53.31 | 42.50 / 53.47 |

架构泛化：ViT-B/16 (39.22%), Swin-V2-B (53.57%), VGG-16 (72.61%), ResNet-34 (77.03%), GPT-2-S/M（文本生成），均100%检测率且性能损失可忽略。

### 鲁棒性评估

| 攻击类型 | 设置 | NeuralMark检测率 | VanillaMark | GreedyMark | VoteMark |
|:--|:--|:--|:--|:--|:--|
| 伪造攻击 | CIFAR-10, ResNet-18 | **48.56%**（≈随机） | 100%（完全被伪造） | 50.70% | 100%（完全被伪造） |
| 覆写 $\lambda$=1000 | CIFAR-100→10 | **100%** | 53.90% | 49.60% | 59.37% |
| 覆写 $\eta$=0.01 | CIFAR-100→10 | **92.18%** | 62.10% | 49.60% | 60.15% |
| 微调 | CIFAR-100→10 | **100%** | 85.93% | 94.14% | 85.54% |
| 微调 | CIFAR-10→100 | **100%** | 70.31% | 82.42% | 71.87% |
| 剪枝 50% | CIFAR-10, AlexNet | ≈**100%** | 略低 | — | — |

### 关键发现

- **伪造攻击**：NeuralMark 的伪造水印检测率仅48.56%（≈随机猜测），VanillaMark 和 VoteMark 被完全伪造（100%），GreedyMark 也能抗伪造（50.70%）但不能抗覆写
- **覆写攻击**：即使攻击者的嵌入强度 $\lambda$ 是原始的1000倍，NeuralMark 原始水印检测率仍维持100%——因为参数重叠为0%。学习率 $\eta$ 增大到0.1时，虽然检测率下降，但模型性能也完全崩溃（10%准确率），攻击无效
- **微调/剪枝**：平均池化机制使NeuralMark在所有微调和剪枝场景中保持100%检测率
- **参数分布与收敛**：水印嵌入对参数分布和训练收敛几乎无影响，隐蔽性好
- **过滤轮数分析**：4轮过滤后参数重叠率已接近0%，增加到6/8轮对鲁棒性无显著提升

## 亮点

- ⭐ **一个机制同时解决两个难题**：哈希水印过滤器巧妙地将雪崩效应（抗伪造）和参数隔离（抗覆写）统一在一个设计中
- ⭐ **理论安全边界**：Proposition 1 给出了严格的伪造概率上界，$n=256$ 时安全边界为88.29%，伪造概率 $< 1/2^{128}$
- ⭐ **极广的架构覆盖**：在13种架构（8种CNN + 3种Transformer + 2种GPT-2）、5个视觉任务+1个文本生成任务上验证，是目前覆盖最全的权重水印工作之一
- ⭐ **极端覆写场景下的鲁棒性**：1000倍嵌入强度仍100%检测，远超所有基线

## 局限性

- **仅关注白盒权重水印**：未涉及黑盒和无盒场景，也未与passport-based和activation-based方法组合
- **假设可信的第三方验证者**：实际中第三方验证机制的建立存在挑战
- **假设攻击者计算资源有限**：若攻击者有能力从头训练模型，水印保护失效
- **哈希函数选择固定为SHAKE-256**：未探讨不同哈希函数对安全性的影响
- **过滤轮数增加可能减少可用参数**：8轮过滤时参数大幅减少，可能限制水印容量

## 相关工作

- **权重水印**：VanillaMark (Uchida et al. 2017)——首个权重水印, 不抗伪造/覆写; GreedyMark (Liu et al. 2021)——贪心选参数, 抗伪造但不抗覆写; VoteMark (Li et al. 2024)——多轮投票, 两种攻击均不抗
- **Passport水印**：Fan et al. (2019, 2021)——用passport样例生成归一化层参数; Liu et al. (2023)——哈希映射passport到水印, 也利用了哈希函数
- **激活水印**：DeepSigns (Rouhani et al. 2019)——在激活图均值中嵌入; Li et al. (2021)——直接嵌入激活图
- **知识产权保护综述**：Li et al. (2021), Sun et al. (2023), Lukas et al. (2022)

## 评分

⭐⭐⭐⭐ — 设计优雅（一个机制解决两个问题），理论分析充分，实验覆盖面极广；但仅限白盒场景且假设可信第三方。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[AAAI 2026\] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates](fairgse_fairness-aware_graph_neural_network_without_high_false_positive_rates.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[CVPR 2026\] Verifying Neural Network Robustness with Dual Perturbations](../../CVPR2026/ai_safety/verifying_neural_network_robustness_with_dual_perturbations.md)
- [\[AAAI 2026\] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service](regionmarker_a_region-triggered_semantic_watermarking_framework_for_embedding-as.md)

</div>

<!-- RELATED:END -->
