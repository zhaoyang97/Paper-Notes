---
title: >-
  [论文解读] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models
description: >-
  [AAAI 2026][AI安全][对抗防御] 提出 TopoReformer，一种基于拓扑自编码器的模型无关对抗纯化管线，利用持久同调（persistent homology）在潜空间中强制拓扑一致性，无需对抗训练即可过滤对抗扰动，有效保护 OCR 系统免受经典攻击、自适应攻击和 OCR 专用水印攻击。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "对抗防御"
  - "拓扑自编码器"
  - "OCR"
  - "持久同调"
  - "流形纯化"
---

# TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models

**会议**: AAAI 2026  
**arXiv**: [2511.15807](https://arxiv.org/abs/2511.15807)  
**代码**: [github](https://github.com/invi-bhagyesh/TopoReformer)  
**领域**: AI安全  
**关键词**: 对抗防御, 拓扑自编码器, OCR安全, 持久同调, 流形纯化

## 一句话总结

提出 TopoReformer，一种基于拓扑自编码器的模型无关对抗纯化管线，利用持久同调（persistent homology）在潜空间中强制拓扑一致性，无需对抗训练即可过滤对抗扰动，有效保护 OCR 系统免受经典攻击、自适应攻击和 OCR 专用水印攻击。

## 研究背景与动机

OCR 系统在文档自动化、车牌识别、合规审计等高风险场景中被广泛使用，但深度学习驱动的 OCR 模型继承了 DNN 的对抗脆弱性——人眼不可见的微小扰动即可导致严重的转录错误，甚至能在物理世界（打印-扫描、拍照）中存活。

现有防御策略主要有四类：

**预处理/去噪纯化**（如 MagNet、PuVAE）：将输入映射回学习到的数据流形，但往往降低无扰动输入的性能

**异常检测**：基于自编码器重建误差或分布检验，但对自适应攻击脆弱

**对抗训练**：计算开销大、依赖特定攻击类型

**后处理修正**：仅对文本输出做修补，治标不治本

这些方法的共同局限在于：**模型特定、计算昂贵、无法抵御未知或自适应攻击**。更关键的是，许多防御被证明只是提供了"虚假安全感"（gradient obfuscation），在 BPDA/EOT 等自适应攻击下彻底失效。

**核心洞察**：对抗扰动往往只改变局部像素关系，而数据的全局拓扑结构（连通性、环、孔洞）保持不变。如果能在编码过程中保持拓扑不变性，就可以自然过滤掉拓扑无关的扰动——这是一种"纯化"而非"去噪"，是本文的核心创新点。

## 方法详解

### 整体框架

TopoReformer 是一个三阶段级联管线：

1. **拓扑自编码器（TopoAE）**：以持久同调损失约束潜空间，对输入进行拓扑级纯化
2. **Reformer（VAE）**：将 TopoAE 输出对齐到下游分类器期望的流形
3. **辅助模块（Auxiliary Module）**：将 TopoAE 的潜向量注入 Reformer 的瓶颈层，提供拓扑信息补充

整个管线是 **模型无关** 的——作为即插即用的前处理模块，可置于任意 OCR 模型之前，无需修改或重新训练下游模型。

### 关键设计

#### 1. **拓扑自编码器（TopoAE）与持久同调损失**

核心思路：计算输入空间 $X$ 和潜空间 $Z$ 的持久同调图（persistence diagram），惩罚两者之间的拓扑差异。

持久同调追踪拓扑特征（连通分量、环、空洞）随尺度参数变化的出生与消亡，生成稳定的、对微小扰动鲁棒的数据结构摘要。

拓扑损失定义为双向匹配：

$$L_t = L_{X \to Z} + L_{Z \to X}$$

其中：
- $L_{X \to Z} = \frac{1}{2} \|A_X^{\pi_X} - A_Z^{\pi_X}\|^2$：要求潜空间在输入空间拓扑配对下是一致的
- $L_{Z \to X} = \frac{1}{2} \|A_Z^{\pi_Z} - A_X^{\pi_Z}\|^2$：要求输入空间在潜空间拓扑配对下是一致的

总损失为：

$$L = L_{rec}(X, \hat{X}) + \lambda L_t$$

**设计动机**：与像素级去噪不同，拓扑损失关注全局结构一致性。数据流形上拓扑不相关的变异（如对抗扰动）在编码过程中被自然丢弃。TopoAE 仅在无扰动数据上训练，无需见过任何攻击样本。

#### 2. **Reformer（VAE）与分类器对齐**

TopoAE 输出虽然拓扑纯化了，但其流形与下游 OCR 分类器的输入分布可能不匹配。Reformer 是一个轻量 VAE，负责将纯化后的图像对齐到分类器期望的流形。

Reformer 训练目标：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{KL}$$

- $\mathcal{L}_{MSE}$：TopoAE 输出与 VAE 重建输出的像素级重建损失
- $\mathcal{L}_{CE}$：VAE 重建输出经分类器后的交叉熵损失，确保分类可用性
- $\mathcal{L}_{KL}$：KL 散度正则化

**设计动机**：纯拓扑纯化可能保留了正确的结构但偏离了分类器的输入分布，Reformer 弥补了这一流形不匹配。

#### 3. **Freeze-Flow 训练范式与辅助模块**

辅助模块接收 TopoAE 的潜向量（通过学习到的投影网络），利用拓扑感知的潜空间信息辅助 Reformer 做更准确的预测。

然而直接联合训练时，模型倾向于仅依赖纯化图像路径，忽略辅助路径，导致辅助模块梯度不足。

Freeze-Flow 训练范式：
1. **冻结阶段**：冻结 Reformer VAE 的编码器，强制梯度流向辅助模块
2. **解冻阶段**：经过预热期后解冻 VAE 解码器，两条路径联合训练

**设计动机**：确保辅助路径在主路径主导之前建立有意义的潜表示，平衡两条路径的优化。实验表明，Freeze-Flow 在 Carlini-Wagner 攻击下可额外提升约 5% 的分类性能。

### 损失函数 / 训练策略

- TopoAE 单独在无扰动样本上预训练至收敛，权重冻结后仅做推理
- Reformer + 辅助模块在 TopoAE 输出上训练，使用目标函数 $\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{CE} + \lambda_3 \mathcal{L}_{KL}$
- 训练超参：$\lambda_1=1, \lambda_2=0.5, \lambda_3=0.5$，Adam 优化器，lr=0.001
- 分类器权重在整个过程中保持冻结

## 实验关键数据

### 主实验

**经典攻击下的消融实验（MNIST/EMNIST, F1-score %）**

| 攻击类型 | 防御配置 | MNIST (弱/强) | EMNIST (弱/强) |
|---------|---------|--------------|---------------|
| Carlini (c=1e-2/1e+1) | No Defense | 30.41 / 4.30 | 36.54 / 33.85 |
| | + TopoAE | 53.92 / 48.51 | 30.87 / 27.71 |
| | + Reformer | 65.38 / 67.93 | 50.64 / 49.16 |
| | + Aux + Warmup | **65.86 / 75.15** | **69.66 / 68.82** |
| PGD (ε=0.005/0.01) | No Defense | 96.74 / 96.62 | 84.87 / 72.66 |
| | 完整管线 | **97.70 / 97.62** | **84.53 / 83.79** |
| FGSM (ε=0.005/0.01) | No Defense | 96.87 / 96.61 | 90.83 / 72.59 |
| | 完整管线 | **97.69 / 97.51** | **90.48 / 84.42** |

**自适应攻击结果（攻击成功率 ASR↓ / F1↑）**

| 攻击 | MNIST ASR↓ / F1↑ | EMNIST ASR↓ / F1↑ |
|------|------------------|-------------------|
| EOT | 9.19 / 90.73 | 28.32 / 73.28 |
| EOT+BPDA | 36.59 / 64.71 | 44.26 / 58.92 |
| BPDA | 81.14 / 15.65 | 84.46 / 12.77 |

**OCR 专用攻击（FAWA 水印攻击）**

| OCR 模型 | 无防御 ASR / Acc | 有防御 ASR / Acc |
|---------|----------------|----------------|
| CRNN | 100 / 48.13 | 78.83 / 71.00 |
| Rosetta | 99.83 / 69.66 | **44.08 / 85.98** |
| TRBA | 99.83 / 46.68 | 60.75 / 80.26 |

### 消融实验

| 组件 | C&W (MNIST, 强) F1 | 提升 |
|------|-------------------|------|
| No Defense | 4.30% | — |
| + TopoAE | 48.51% | +44.21 |
| + Reformer | 67.93% | +19.42 |
| + Auxiliary | 72.41% | +4.48 |
| + Freeze-Flow Warmup | **75.15%** | +2.74 |

逐步消融验证了每个组件的贡献，Freeze-Flow 训练范式在强攻击下提供了显著增益。

### 关键发现

1. **对 C&W 攻击效果最显著**：C&W 生成的扰动精细且低幅度，恰好是拓扑纯化最擅长过滤的类型，F1 从 4.30% 提升至 75.15%
2. **对 EOT 自适应攻击鲁棒**：ASR 从 99.05% 降至 9.19%，表明防御机制不是简单的梯度遮蔽
3. **不牺牲干净样本性能**：无扰动情况下 MNIST ~98%、EMNIST ~94% 准确率，几乎无损
4. **跨 OCR 架构通用**：对 CTC-based（CRNN, Rosetta, STAR-Net）和 Attention-based（RARE, TRBA）模型均有效

## 亮点与洞察

- **范式创新**：首次将拓扑自编码器用于对抗防御/输入纯化，不同于传统去噪——它执行的是"拓扑纯化"，丢弃拓扑不相关的变异
- **无需对抗样本参与训练**：完全在干净数据上训练，对未知攻击天然泛化
- **Freeze-Flow 训练技巧精巧**：通过控制梯度流向解决了多路径模型中辅助分支欠训练的常见问题
- **隐式 Lipschitz 平滑性**：拓扑约束隐式地限制了潜表示对输入扰动的敏感度，提供了一种不依赖显式梯度正则化的鲁棒性来源

## 局限与展望

1. **对 BPDA 攻击仍脆弱**：BPDA 可利用局部曲率绕过全局拓扑平滑性，ASR 仍高达 81%
2. **评估数据集局限**：仅在 MNIST/EMNIST 等简单数据上验证，未在 CIFAR-10/ImageNet 等自然图像上测试
3. **OCR 评估中未使用 Reformer**：为保持部署效率刻意省略了 Reformer，但这限制了 OCR 场景的防御上限
4. **计算开销分析缺失**：未报告持久同调计算的时间开销，这在大规模部署时可能是瓶颈
5. **未与 diffusion-based 纯化方法对比**：近年来基于扩散模型的对抗纯化方法（如 DiffPure）是强力基线

## 相关工作与启发

- 拓扑数据分析（TDA）在对抗鲁棒性中的应用是一个新兴方向，本文是 TopoAE 用于对抗纯化的首次尝试
- Freeze-Flow 训练范式对多分支/多路径模型的训练平衡有通用参考价值
- "拓扑纯化 vs 去噪"的思想区分值得深入研究：去噪是试图恢复原始信号，纯化是映射到正确流形
- 可以考虑将拓扑约束整合到 OCR 模型的训练目标中（作者提出的 Future Work）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 拓扑自编码器 + 对抗纯化的组合很新颖，Freeze-Flow 训练也有创意
- 实验充分度: ⭐⭐⭐ — 攻击覆盖面广但数据集偏简单，缺少自然图像验证
- 写作质量: ⭐⭐⭐⭐ — 清晰，拓扑背景介绍到位，但部分表述冗长
- 价值: ⭐⭐⭐⭐ — 开辟了 TDA 用于对抗防御的新路线，但实用性需在更复杂场景验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[CVPR 2026\] Mitigating Error Amplification in Fast Adversarial Training](../../CVPR2026/ai_safety/mitigating_error_amplification_in_fast_adversarial_training.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](../../NeurIPS2025/ai_safety/keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)

</div>

<!-- RELATED:END -->
