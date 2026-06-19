---
title: >-
  [论文解读] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis
description: >-
  [CVPR 2025][AI安全][对抗攻击检测] 本文提出了一种基于查询更新模式(而非输入模式)的黑盒对抗攻击检测框架 GWAD，引入 Delta Similarity 指标来捕获基于查询的攻击中零阶优化的固有模式，在8种SOTA攻击(包括自适应攻击OARS)上实现了接近100%的检测率且误报率极低，显著优于现有的状态化防御方法。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "对抗攻击检测"
  - "黑盒攻击"
  - "查询分析"
  - "Delta相似度"
  - "状态化防御"
---

# Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis

**会议**: CVPR 2025  
**arXiv**: [2503.02986](https://arxiv.org/abs/2503.02986)  
**代码**: [https://github.com/jpark04-qub/GWAD](https://github.com/jpark04-qub/GWAD)  
**领域**: AI安全  
**关键词**: 对抗攻击检测, 黑盒攻击, 查询分析, Delta相似度, 状态化防御

## 一句话总结

本文提出了一种基于查询更新模式(而非输入模式)的黑盒对抗攻击检测框架 GWAD，引入 Delta Similarity 指标来捕获基于查询的攻击中零阶优化的固有模式，在8种SOTA攻击(包括自适应攻击OARS)上实现了接近100%的检测率且误报率极低，显著优于现有的状态化防御方法。

## 研究背景与动机

**领域现状**：基于查询的黑盒对抗攻击是最具现实威胁的攻击场景，攻击者只需通过API查询模型并分析输出即可生成对抗样本。现有防御主要分为两类：一类是"事后"检测已生成的对抗样本，另一类是"状态化防御"在攻击生成过程中检测异常查询。

**现有痛点**：现有的状态化防御如 Blacklight、PIHA 和 Stateful Detection 都在输入空间分析查询模式，通过监控输入间的相似性来检测攻击。但这些方法存在两个关键问题：(1) 容易被自适应攻击 OARS 绕过，OARS通过在输入空间精心构造查询方向来规避检测；(2) 在高相似度场景(如连续视频帧)中误报率较高。

**核心矛盾**：输入空间的相似性模式对攻击者而言是可操控的——攻击者可以在不影响梯度估计的前提下改变输入空间的查询分布来逃避检测。防御者需要一种攻击者无法轻易规避的检测信号。

**本文目标** 找到一种比输入空间模式更鲁棒、更本质的信号来区分正常查询和攻击查询，同时降低误报率。

**切入角度**：所有基于查询的黑盒攻击都依赖零阶优化来估计梯度，这个过程需要用随机向量进行结构化的输入更新。这些更新之间的模式(差分关系)是攻击的固有属性，无法通过输入空间的变换来消除。

**核心 idea**：分析连续查询之间的更新差分的相似性(Delta Similarity)，而非查询本身的相似性，来检测正在进行中的黑盒攻击。

## 方法详解

### 整体框架

GWAD 框架持续监控发送到被保护模型的查询序列。对于每三个连续查询，计算一个 Delta Similarity 值。收集最近256个 DS 值后生成直方图特征(HoDS)，输入到预训练的分类器中判断是否存在攻击。整个流程：查询流 → DS计算 → HoDS特征提取 → 攻击分类/检测。

### 关键设计

1. **Delta Similarity (DS) 指标**:

    - 功能：捕获查询序列中更新模式的特征，区分正常和攻击行为
    - 核心思路：对于三个连续查询 $q_1, q_2, q_3$，计算差分 $\delta_1 = q_2 - q_1$, $\delta_2 = q_3 - q_2$，然后 DS 就是 $\delta_1$ 和 $\delta_2$ 的余弦相似度。在零阶优化过程中，查询更新由随机向量 $u$ 的线性组合构成，由于高维空间中随机向量的集中度现象，这些向量近似正交且等长，导致攻击的 DS 分布呈现显著的峰值模式（例如 NES 在 $-0.7071$ 处、HSJA 在 $-0.5$ 和 $\pm 1.0$ 处），而正常查询的 DS 分布是高方差的。
    - 设计动机：DS 操作的是"更新空间"而非"输入空间"。OARS等自适应攻击通过改变输入空间的方向来逃避检测，但无法改变零阶优化本身的更新结构。

2. **HoDS 特征表示**:

    - 功能：将 DS 序列转换为固定维度的特征向量用于分类
    - 核心思路：收集最近256个 DS 值，在 $[-1, 1]$ 范围内用200个等距bin加一个 $DS=1.0$ 的额外bin构建直方图，然后做 min-max 归一化。最终生成 $1 \times 201$ 的特征向量。不同攻击方法有各自独特的HoDS签名。
    - 设计动机：直方图表示天然对序列顺序不敏感但对分布模式敏感，且维度固定便于分类器处理。256的窗口大小经实验验证为最佳。

3. **轻量级攻击分类器**:

    - 功能：判断是否存在攻击，并识别具体攻击类型
    - 核心思路：6层全连接网络+ReLU激活+Log-Softmax输出，用SGD训练100 epoch，batch size 128。支持多分类(benign + 各种攻击)和二分类(benign/attack)两种模式。
    - 设计动机：HoDS特征已经充分编码了攻击模式，简单的分类器即可高效判别。轻量级设计确保不会显著增加服务器端延迟。

### GWAD+ 增强：Screener预筛

针对GWAD在大量良性注入攻击下的弱点，提出 GWAD+。Screener在输入空间做轻量预筛：将查询缩放到32×32、Canny边缘检测转二进制、压缩为128字节向量，然后与FIFO队列中历史查询比较。低相似度的查询被认为是良性直接放行，高相似度的才送入GWAD进行DS分析。这样即使攻击者注入10倍良性查询(rb=1000%)，GWAD+仍保持99.2%+检测率。

## 实验关键数据

### 主实验

| 攻击方法 | Blacklight检测率 | PIHA检测率 | GWAD检测率 |
|----------|------------------|------------|------------|
| BA | 23.96% | 38.08% | **99.98%** |
| HSJA | 97.86% | 98.75% | **100.00%** |
| NES | 99.96% | 94.66% | **100.00%** |
| QEBA(未知攻击) | 96.51% | 96.78% | **100.00%** |
| Surfree(未知攻击) | 98.77% | 70.96% | **100.00%** |

| OARS自适应攻击 | Blacklight ASR | PIHA ASR | GWAD ASR |
|---------------|----------------|----------|----------|
| OARS-NES | 98% | 82% | **0%** |
| OARS-HSJA | 75% | 71% | **0%** |
| OARS-QEBA | 98% | 95% | **0%** |

### 消融/误报率实验

| 数据集/场景 | Blacklight FPR | PIHA FPR | GWAD FPR |
|------------|----------------|----------|----------|
| CIFAR-10 | 0.00% | 0.00% | **0.00%** |
| ImageNet | 0.16% | 0.14% | **0.03%** |
| FLIR ADAS(汽车) | 2.17% | 1.43% | **0.00%** |
| BIRDSAI(高相似度) | 16.92% | N/A | **1.29%** |
| Hollywood高相似度 | 25.47% | 26.19% | **17.99%** |

### 关键发现

- GWAD 对 OARS 自适应攻击完全免疫(ASR=0%)，而 Blacklight 和 PIHA 被完全击穿。这验证了"更新空间"比"输入空间"本质上更难被攻击者操控的洞察。
- GWAD 具有跨数据集和跨模型的泛化能力：在CIFAR-10上训练的GWAD-CIFAR10在ImageNet上也达到97.32%分类准确率。
- 在高相似度场景(连续视频帧)中 GWAD 的误报率远低于现有方法，FLIR自动驾驶场景中达到0% FPR。
- 良性注入攻击需要消耗2.5倍以上的查询预算才能开始影响GWAD检测，而配合Screener后(GWAD+)即使10倍注入也能保持99%+检测率。

## 亮点与洞察

- **从输入空间到更新空间的范式转移**：这是对抗攻击检测领域的一个根本性视角转换。零阶优化的数学结构是攻击无法绕过的"指纹"，就像密码学中的侧信道——攻击者可以隐藏意图但无法隐藏计算过程的固有模式。
- **DS指标的通用性**：不同攻击有不同的DS分布特征（由其梯度估计策略决定），这意味着GWAD不仅能检测还能识别攻击类型，为后续的针对性防御提供情报。
- **GWAD+的互补设计**：输入空间筛选和更新空间分析的结合形成了纵深防御，各自覆盖对方的盲区。

## 局限与展望

- 在极高相似度场景(连续电影帧)中FPR仍有17.99%，虽然优于现有方法但还需改进
- 目前只在图像分类任务上测试，其他模态(文本、语音)的黑盒攻击是否呈现类似的DS模式需要验证
- 攻击者若改变零阶优化的随机分布参数(补充材料中讨论了这种情况)，可能部分影响DS模式
- 分类器训练需要已知攻击的样本，面对全新类型的零阶优化策略的泛化能力需进一步研究

## 相关工作与启发

- **vs Blacklight**: Blacklight在输入空间做量化哈希检测相似查询，在常规攻击上有效但被OARS完全击穿(ASR 98%)。GWAD因工作在更新空间而对OARS免疫，是本质上的维度升级。
- **vs PIHA**: PIHA用感知哈希做统计分析，但灰度图像不适用(需要色调信息)且高相似度场景FPR高。GWAD无此限制。
- **vs Stateful Detection**: 最早的状态化防御，用预训练网络降维后计算l2相似度，计算开销大且准确率低于Blacklight。GWAD更轻量且更准确。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从输入空间到更新空间的范式转移是关键突破
- 实验充分度: ⭐⭐⭐⭐⭐ 8种攻击+自适应攻击+泛化+误报率+多场景全面评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰、理论与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 解决了状态化防御被自适应攻击击穿的核心痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[ACL 2025\] Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](../../ACL2025/ai_safety/multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)
- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](../../ICML2026/ai_safety/mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[CVPR 2026\] SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning](../../CVPR2026/ai_safety/seba_sample-efficient_black-box_attacks_on_visual_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
