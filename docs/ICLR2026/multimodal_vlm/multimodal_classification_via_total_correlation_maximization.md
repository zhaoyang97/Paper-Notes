---
title: >-
  [论文解读] Multimodal Classification via Total Correlation Maximization
description: >-
  [ICLR 2026][多模态VLM][多模态学习] 从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。 领域现状：多模态学习通过融…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "多模态学习"
  - "模态竞争"
  - "总相关性"
  - "信息论"
  - "损失函数设计"
---

# Multimodal Classification via Total Correlation Maximization

**会议**: ICLR 2026  
**arXiv**: [2602.13015](https://arxiv.org/abs/2602.13015)  
**代码**: [https://github.com/hubaak/TCMax](https://github.com/hubaak/TCMax)  
**领域**: 多模态VLM  
**关键词**: 多模态学习, 模态竞争, 总相关性, 信息论, 损失函数设计

## 一句话总结
从信息论角度分析多模态分类中的模态竞争问题，提出 TCMax 损失函数通过最大化多模态特征与标签之间的总相关性（Total Correlation），同时兼顾联合学习、单模态学习和跨模态对齐三重目标，在多个音视频/图文分类基准上超越 SOTA。

## 研究背景与动机

**领域现状**：多模态学习通过融合不同模态（如音频+视觉、文本+图像）来获取更鲁棒的表征。主流做法是联合学习（joint learning），即用一个共享预测头对所有模态特征做分类。

**现有痛点**：联合学习存在"模态竞争"（modality competition）问题——某些模态收敛更快（如音频），导致其他模态（如视觉）被抑制，最终多模态模型甚至不如最好的单模态模型。现有平衡方法如 OGM-GE（梯度调制）、AGM（自适应梯度）虽然能缓解，但无法根本解决"模态懒惰"问题。

**核心矛盾**：联合学习最大化的是 $I(y; z^{(a)}, z^{(v)})$，当音频编码器已经学到足够信息（$I(y; z^{(a)}) \approx H(y)$）后，视觉编码器能学到的条件互信息 $I(y; z^{(v)}|z^{(a)})$ 的上界趋近于零，本质上是优化目标导致强模态"挤占"弱模态的学习空间。

**本文目标** 如何设计一个损失函数，既能避免模态竞争（让每个模态独立学到充分信息），又能利用跨模态交互（不像纯单模态学习那样完全割裂），同时不需要额外超参数或结构修改？

**切入角度**：作者从信息论出发，发现总相关性（Total Correlation）可以天然地分解为"联合学习 + 单模态学习 + 跨模态对齐"三项——这恰好覆盖了现有方法各自的优势。

**核心 idea**：用总相关性替代互信息作为优化目标，通过最大化 $\text{TC}(z^{(a)}, z^{(v)}, y)$ 同时实现联合学习、单模态学习和模态对齐，且无需额外超参数。

## 方法详解

### 整体框架
TCMax 要解决的是多模态分类里"强模态把弱模态的学习空间挤占掉"的模态竞争（modality competition）问题，而它的思路是**不动网络结构、只换训练目标**。整体数据流和普通多模态模型完全一样：输入多模态数据 $(x^{(1)}, \dots, x^{(M)})$，每个模态各有一个编码器 $\psi^{(m)}$ 把它映射到特征 $z^{(m)}$，再一起送进一个共享预测头 $f_\theta$ 输出标签概率。

区别全在损失函数上。作者先从信息论上看清模态竞争的病根（联合学习最大化的互信息目标会让强模态饿死弱模态），于是把优化目标从"特征与标签的互信息"换成"两者的总相关性（Total Correlation, TC）"——因为 TC 的数学分解里天然同时装着联合学习、单模态学习、跨模态对齐三件事。但 TC 在高维特征上无法直接计算，作者借 MINE（互信息神经估计）的思路推广出一个可优化的下界，并巧妙地让**预测头自己充当 TC 估计器**：训练时正样本是真实的多模态配对、负样本是把不同样本的各模态特征随机重组得到的"伪配对"，预测头一边做分类、一边把这两类样本在输出上拉开，就等价于在抬高 TC 下界。推理时不需要任何额外操作，直接走 Softmax 输出。

### 关键设计

**1. 信息论视角的模态竞争分析：把"模态懒惰"归因到优化目标本身**

现有平衡方法（OGM-GE、AGM）只调梯度，没回答模态竞争究竟从哪来。作者把联合学习的目标拆开看：它最大化的是 $I(y; z^{(a)}, z^{(v)}) = I(y; z^{(a)}) + I(y; z^{(v)}|z^{(a)})$。当音频这种快收敛的模态先把标签信息学得差不多、$I(y; z^{(a)}) \approx H(y)$ 时，留给视觉的条件互信息 $I(y; z^{(v)}|z^{(a)})$ 上界就被压到接近零——视觉编码器再怎么训也没有可学的空间。这说明模态竞争不是优化技巧的问题，而是联合学习目标天生就会让强模态挤占弱模态，调梯度只能缓解"症状"。

**2. 总相关性分解：用一个量同时装下联合学习、单模态学习和对齐**

既然单一互信息目标有结构缺陷，作者改用总相关性（Total Correlation）。它的妙处在于有两种等价分解。对两模态情况，$\text{TC}(z^{(a)}, z^{(v)}, y) = I(y; z^{(a)}, z^{(v)}) + I(z^{(a)}; z^{(v)})$，这一式含联合学习项 + 跨模态对齐项；同时它又等于 $I(y; z^{(a)}) + I(y; z^{(v)}) + I(z^{(a)}; z^{(v)}|y)$，这一式含两个独立的单模态学习项 + 条件对齐项。也就是说，最大化同一个 TC 就等于同时优化"联合学习 + 单模态学习 + 模态对齐"——现有方法各自靠多个损失和超参数才凑齐的三件事，在 TC 里是数学上自然合一、互不冲突的。

**3. 总相关性神经估计（TCNE）：给不可计算的 TC 一个可优化的下界**

直接算 TC 要知道联合分布与各边际分布的密度比，在高维特征空间里做不到。作者把 MINE（Mutual Information Neural Estimation）从双变量推广到多变量来绕开这一点：借 Donsker-Varadhan 表示定理，得到

$$\text{TC} \geq \sup_\theta \mathbb{E}_{\mathbb{P}_{joint}}[T_\theta] - \log\big(\mathbb{E}_{\mathbb{P}_{product}}[e^{T_\theta}]\big)$$

其中 $T_\theta$ 是一个神经网络统计量。这样最大化 TC 就转成训练 $T_\theta$ 去拉大联合分布样本与独立边际乘积样本之间的差距，完全用采样估计，不再依赖显式密度。

**4. TCMax 损失函数：让预测头自己充当 TC 估计器，零额外参数**

如果再单独搭一个 $T_\theta$ 网络，就违背了"不改结构"的初衷。作者的关键复用是直接令 $T_\theta(z^{(1)}, \dots, z^{(M)}, y) = f_\theta(z^{(1)}, \dots, z^{(M)})_y$——即把分类预测头在真实标签维度上的输出当作统计量。代入下界并取负，就得到训练损失

$$\mathcal{L}_{\text{TCMax}} = -\mathbb{E}_{\mathbb{P}_{joint}}[F_\Theta] + \log\big(\mathbb{E}_{\mathbb{P}_{product}}[e^{F_\Theta}]\big)$$

训练时正样本是来自联合分布的真实多模态样本，负样本是把不同样本的各模态特征随机重组得到的"伪配对"。由此 TCMax 只换损失、不加任何网络参数，预测头本身既做分类又做 TC 估计；推理时直接走 Softmax，不需要任何额外操作。

### 损失函数 / 训练策略

TCMax 的直接实现需要 $|B|^M$ 次前向传播（分母需要枚举所有模态特征的组合），计算开销大。两个优化策略：

- **负样本采样**：从 $\mathcal{B} \times \mathcal{B}$ 中随机采样 $\mathcal{N}$ 个负样本对，将 $O(|B|^M)$ 降为 $O(|\mathcal{N}|)$
- **线性融合解耦**：如果预测头是 $f_\theta(z^{(a)}, z^{(v)}) = f^{(a)}(z^{(a)}) + f^{(v)}(z^{(v)})$，则分母可分解为各模态独立求和，复杂度降为 $O(|B|)$

理论保证：(1) 最小化 $\mathcal{L}_{\text{TCMax}}$ 等价于提升 TC 下界；(2) 当 TC 估计器达到最优时，模型能精确估计联合分布（Proposition 2-3）；(3) 推理时无需任何额外操作。

## 实验关键数据

### 主实验

在 5 个音视频/图文数据集上与 10+ 方法对比，使用 ResNet-18 从头训练：

| 数据集 | 指标 | TCMax (Share Head) | 之前SOTA (MMPareto) | 提升 |
|--------|------|------|----------|------|
| CREMA-D | Acc | **82.7** | 74.4 | +8.3% |
| Kinetics-Sounds | Acc | **63.5** | 62.7 | +0.8% |
| AVE | Acc | **64.5** | 63.1 | +1.4% |
| VGGSound | Acc | **47.6** | 46.2 | +1.4% |
| UCF101 | Acc | **56.0** | 55.9 | +0.1% |

### 消融实验

| 配置 | 说明 |
|------|------|
| TCMax (Concat) | 使用拼接融合，也能获得竞争力结果 |
| TCMax (Share Head) | 使用共享头融合，整体最优 |
| 负样本数影响 | CREMA-D 在 1024 负样本时最优，UCF101 在 256 时最优 |
| JS 散度分析 | TCMax 在所有数据集上模态间预测一致性最高（JS散度最低） |
| 预测熵均衡 | TCMax 的强弱模态预测熵比 $\rho$ 最接近 1（CREMA-D: 1.549 vs Concat: 2.913） |

### 关键发现
- TCMax 的多模态增益主要来自跨模态协同而非单模态提升：单模态性能与 unimodal 方法持平，但多模态融合显著更优
- JS 散度实验证实 TCMax 确实学到了跨模态对齐：两个模态的预测分布最一致
- 训练曲线显示 TCMax 损失值始终高于联合/单模态学习，有效防止过拟合
- 使用 CLIP 冻结编码器时 TCMax 仍有效（MVSA 上 ViT-B/32: 84.05 vs Joint: 82.83）

## 亮点与洞察
- **统一框架**：通过一个 TC 量，自然统一了联合学习、单模态学习和对齐三个通常需要多个损失和超参数平衡的目标。巧妙之处在于这不是人为拼凑，而是 TC 的数学分解自然包含这三项
- **零超参数**：TCMax 不引入任何额外超参数（不像 QMF 需要正则化权重、MMPareto 需要 Pareto 方向调整），直接替换交叉熵即可使用。这大幅降低了实际应用的调参成本
- **TCNE→MINE 推广**：将 MINE 从二变量推广到多变量是一个自然但有价值的理论贡献，可迁移到任何需要衡量多变量依赖性的场景（如多任务学习、多视角表示学习）
- **线性融合解耦技巧**：利用 $\exp(a+b) = \exp(a)\exp(b)$，当预测头是线性融合时将负样本计算从 $O(|B|^2)$ 降到 $O(|B|)$，这个技巧可迁移到其他对比学习场景

## 局限与展望
- 作者承认 TCMax 目前仅适用于分类任务，无法直接扩展到检测、生成等任务——需要重新定义输入输出的概率分布
- 实验仅使用 ResNet-18 从头训练，缺乏在大规模预训练模型（如 ViT-L、大型多模态模型）上的验证（CLIP 实验只冻结了编码器）
- 负样本采样数的选择依赖数据集（CREMA-D 需要 1024，UCF101 只需 256），缺乏自适应选择机制
- 非线性融合头的计算开销仍为 $O(|\mathcal{N}|)$，在大 batch 或多模态场景下可能成为瓶颈
- 所有实验数据集规模较小（最大 VGGSound ~15万），未在百万级数据集上验证可扩展性

## 相关工作与启发
- **vs OGM-GE/AGM**: 这些方法通过梯度调制平衡模态贡献，但只解决"症状"（梯度不均衡）不解决"病因"（优化目标本身的缺陷）。TCMax 从目标函数层面重新设计，更根本
- **vs QMF/MLA**: 这些方法显式引入单模态损失+正则化项，需要超参数平衡。TCMax 通过 TC 分解天然包含单模态目标，无需额外项
- **vs MMPareto**: MMPareto 用 Pareto 优化平衡多目标方向，但仍是在多个独立目标之间做权衡。TCMax 用单一目标统一替代，更简洁
- **vs 对比学习 (InfoNCE)**: InfoNCE 是 MINE 的一个特例（固定函数形式），而 TCMax 可以看作 InfoNCE 从 pair-wise 到 multi-variable 的自然推广

## 评分
- 新颖性: ⭐⭐⭐⭐ 信息论视角不是全新的，但 TC 统一三重目标的见解很深刻
- 实验充分度: ⭐⭐⭐⭐ 5个数据集+多种分析，但缺少大规模验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰严谨，motivation 层层递进
- 价值: ⭐⭐⭐⭐ 实用的无超参数损失函数，但受限于分类任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/multimodal_vlm/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](../../NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/multimodal_vlm/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
