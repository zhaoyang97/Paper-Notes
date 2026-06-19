---
title: >-
  [论文解读] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment
description: >-
  [ICCV 2025][AI安全][自监督学习] 提出Noisy Alignment（NA）方法，通过显式压缩投毒图像中的噪声成分来增强自监督对比学习的后门攻击效果，将攻击建模为二维图像布局优化问题，并推导出理论最优参数，在ImageNet-100上ASR提升最高达45.9%。 自监督对比学习（CL）如CLIP、DINOv2…
tags:
  - "ICCV 2025"
  - "AI安全"
  - "自监督学习"
  - "对比学习"
  - "backdoor attack"
  - "数据投毒"
  - "噪声对齐"
---

# Backdooring Self-Supervised Contrastive Learning by Noisy Alignment

**会议**: ICCV 2025  
**arXiv**: [2508.14015](https://arxiv.org/abs/2508.14015)  
**代码**: [https://github.com/jsrdcht/Noisy-Alignment](https://github.com/jsrdcht/Noisy-Alignment)  
**领域**: AI安全 / 对比学习后门攻击  
**关键词**: 自监督学习, 对比学习, backdoor attack, 数据投毒, 噪声对齐

## 一句话总结
提出Noisy Alignment（NA）方法，通过显式压缩投毒图像中的噪声成分来增强自监督对比学习的后门攻击效果，将攻击建模为二维图像布局优化问题，并推导出理论最优参数，在ImageNet-100上ASR提升最高达45.9%。

## 研究背景与动机
自监督对比学习（CL）如CLIP、DINOv2等模型利用海量未标注数据学习通用表征，但由于训练数据通常来自未经审查的互联网爬取，存在数据投毒的安全风险。已有研究表明，仅污染百万分之一的预训练数据就可能操控CL模型的行为。

现有的数据投毒型对比学习后门攻击（DPCL）方法存在两个核心问题：

**隐式共现依赖脆弱**：现有方法（如SSLBKD、CorruptEncoder）依赖后门触发器与目标对象在随机增强视图中的共现来建立关联，但这种隐式关联不够可靠

**缺乏噪声压缩**：投毒图像中原始语义特征（如"熊猫""树木"等非触发器特征）会主导表征空间，干扰后门触发器的有效性

作者从训练可控型后门攻击（Oracle Attack）中提取了关键目标——"噪声对齐"（Noisy Alignment），发现其隐含两个子目标：参考对齐（将投毒特征与目标类对齐）和噪声压缩（压缩正交于目标方向的特征），而现有DPCL方法只考虑了前者，忽略了后者。

## 方法详解

### 整体框架
NA方法的核心流程：（1）收集少量影子图像和参考图像；（2）将触发器嵌入影子图像生成含噪投毒图像；（3）通过理论推导的最优布局参数，将含噪投毒图像与参考图像组合成复合投毒样本；（4）将复合样本注入预训练数据集，利用CL的随机裁剪增强自然实现噪声对齐。

### 关键设计

1. **噪声对齐目标的理论分解**:

    - 功能：将Oracle攻击的目标 $\mathcal{L}_{\text{align}} = \mathbb{E}[1 - \cos(f(\mathbf{x}_s \oplus \mathbf{p}), f(\mathbf{x}_r))]$ 分解为两个正交成分
    - 核心思路：对投毒特征 $\mathbf{v} = f(\mathbf{x}_s \oplus \mathbf{p})$ 沿参考特征 $\mathbf{u}$ 方向分解为：
    $\mathbf{v} = \underbrace{(\mathbf{v}^\top \mathbf{u})\mathbf{u}}_{\text{对齐分量}} + \underbrace{\mathbf{v}_\perp}_{\text{压缩分量}}$
      优化余弦相似度时梯度同时推动 $\alpha \to +\infty$（完美对齐）和 $\mathbf{v}_\perp \to \mathbf{0}$（维度坍缩）
    - 设计动机：揭示了为什么Oracle攻击远优于现有DPCL——它隐式完成了噪声压缩，而现有方法缺少这一关键机制

2. **Oracle投毒变体**:

    - 功能：将Oracle攻击的训练控制目标转化为数据投毒场景
    - 核心思路：构造恶意正对（影子投毒图像，参考图像），使CL自然将两者的增强视图视为正对来训练
    - 关键公式：$\mathcal{L}_{\text{oracle-poisoning}} = \mathbb{E}[\mathcal{L}_{cl}] + \mathbb{E}[\mathcal{L}_{cl}(f(T_1(\mathbf{x}_s \oplus \mathbf{p})), f(T_2(\mathbf{x}_r)))]$
    - 设计动机：验证了noise compression是Oracle攻击高效的核心原因

3. **离线布局优化（核心创新）**:

    - 功能：不需要控制训练过程，通过预先优化投毒样本的空间布局来模拟噪声对齐
    - 核心思路：将问题转化为二维布局优化——在画布上放置参考图像和含触发器的影子图像，最大化CL随机裁剪同时满足三个条件的概率：
    $P(\underbrace{\mathbf{p} \subseteq \mathcal{V}_1 \subseteq \mathbf{x}_s \oplus \mathbf{p}}_{\text{触发器保留}} \wedge \underbrace{\mathcal{V}_2 \subseteq \mathbf{x}_r}_{\text{参考匹配}} \wedge \underbrace{\mathcal{V}_1 \cap \mathcal{V}_2 = \emptyset}_{\text{视图不相交}})$
    - **定理1**（最优位置）：左右布局下，参考图像放置在 $(0,0)$，影子图像放置在 $(c_w/2, 0)$，触发器放在影子图像中心
    - **定理2**（最优画布尺寸）：$c_h^* = r_l$，$c_w^* = 2r_l$（即画布高度等于图像边长，宽度为两倍）
    - 设计动机：从理论上推导出最优参数，避免了启发式搜索的低效

4. **投毒样本制作流程**:

    - 随机选择参考图像和影子图像
    - 将触发器嵌入影子图像
    - 随机选择四种布局方向（左右/右左/上下/下上）之一
    - 按定理1和定理2确定的最优参数拼接复合图像
    - 仅需约650张投毒图像（ImageNet-100的0.5%）

### 损失函数 / 训练策略
攻击者不修改训练损失——投毒样本被注入后，CL的标准训练过程（如InfoNCE损失）会自然地将含触发器的裁剪与参考图像裁剪视为正对进行学习，从而隐式实现噪声对齐目标。这是数据投毒攻击的核心优势：无需介入训练过程。

## 实验关键数据

### 主实验（ImageNet-100 各CL框架 ASR%）

| 攻击方法 | MoCo v2 | BYOL | SimSiam | SimCLR |
|----------|---------|------|---------|--------|
| SSLBKD | 50.9 | 70.2 | 51.2 | 33.9 |
| CTRL | 1.1 | 4.7 | 0.1 | 0.1 |
| CorruptEncoder | 55.1 | 20.4 | 26.1 | 42.1 |
| BLTO | 45.1 | 77.6 | 31.6 | 51.0 |
| **NA (本文)** | **84.8** | **71.4** | **97.1** | **64.8** |
| Oracle-Poisoning (上界) | 97.3 | 98.5 | 96.1 | 97.7 |

### 消融实验（影响因素分析）

| 变量 | 设置 | ASR | 说明 |
|------|------|-----|------|
| 投毒比例 | 0.2% | >50% | 极少量投毒即有效 |
| 投毒比例 | 0.5% | 84.8% | 默认设置 |
| 触发器大小 | 30×30 | >50% | 较小触发器仍有效 |
| 触发器大小 | 50×50 | 84.8% | 默认设置 |
| 布局方式 | 固定布局 | 更高 | 但泛化性差 |
| 布局方式 | 随机布局 | 84.8% | 泛化性更好（采用） |
| 影子图像数量 | ~200 | 饱和 | 200张即可饱和 |

### 关键发现
- NA在SimSiam上达到97.1% ASR，甚至超过了需要控制训练的Oracle BadEncoder
- 在CLIP等图文对比模型上也有效，ASR达到100%（含噪图像+参考文本）
- CTRL和BLTO使用隐形触发器，对CL增强（尤其高斯模糊）极其敏感，在ImageNet-100上几乎失效
- 多目标攻击场景下，即使同时攻击4个类别，ASR仍保持92.7%
- 常见检测方法在高维空间（ImageNet-100）上检测性能显著下降
- 自适应防御（修改裁剪策略）可有效防御NA，但会大幅损害模型性能

## 亮点与洞察
- 理论分析深入：从Oracle攻击目标的数学分解中提炼出噪声压缩的关键机制，为后续防御研究提供了理论基础
- 方法简洁优雅：仅通过图像空间布局即可实现强大的攻击效果，不需要频域操作或生成模型
- 最优参数有解析解：定理1和定理2给出了闭式最优解，避免了搜索开销
- 可自然扩展到图文对比学习（CLIP），实用威胁性强

## 局限与展望
- 自适应防御（修改或移除随机裁剪）可有效防御，但代价是模型性能下降
- 投毒样本的视觉形态（拼接两张图片）在人工审查下较易被发现
- 主要在ResNet-18上验证，更大backbone上的效果有待充分验证
- 未深入探讨防御方的可行对策和攻防均衡

## 相关工作与启发
- **vs SSLBKD**: SSLBKD简单地在目标类样本上叠加触发器，缺少噪声压缩，ASR较低
- **vs CorruptEncoder**: CorruptEncoder优化共现概率但不考虑压缩，在大规模数据上效果有限
- **vs BadEncoder (Oracle)**: BadEncoder需要控制训练过程，NA在数据投毒约束下逼近其效果
- **vs CTRL**: CTRL使用频域触发器追求隐蔽性，但对CL增强极其敏感，ASR极低

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从Oracle攻击中提取噪声压缩机制并建模为布局优化，理论推导扎实
- 实验充分度: ⭐⭐⭐⭐ 涵盖4种CL框架、2个数据集、多种防御方法，消融全面
- 写作质量: ⭐⭐⭐⭐ 从理论分析到方法设计到实验验证逻辑链清晰
- 价值: ⭐⭐⭐⭐ 揭示了CL的重要安全漏洞，为防御研究提供了方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](../../CVPR2025/ai_safety/invisible_backdoor_attack_against_self-supervised_learning.md)
- [\[NeurIPS 2025\] FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation](../../NeurIPS2025/ai_safety/faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)
- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](../../CVPR2025/ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[ICLR 2026\] Less is More: Towards Simple Graph Contrastive Learning](../../ICLR2026/ai_safety/less_is_more_towards_simple_graph_contrastive_learning.md)

</div>

<!-- RELATED:END -->
