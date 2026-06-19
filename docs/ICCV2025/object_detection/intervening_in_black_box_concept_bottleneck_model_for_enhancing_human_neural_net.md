---
title: >-
  [论文解读] Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding
description: >-
  [ICCV 2025][目标检测][可解释性] 提出 CBM-HNMU 框架，通过概念瓶颈模型（CBM）逼近黑盒模型的推理过程，自动识别并修正有害概念，再将修正后的知识蒸馏回黑盒模型，实现超越样本级别的系统性模型干预与准确率提升。 深度学习模型的决策过程日趋不透明，现有可解释性方法主要分为两类： 特征归因方法：（Grad-C…
tags:
  - "ICCV 2025"
  - "目标检测"
  - "可解释性"
  - "概念瓶颈模型"
  - "黑盒干预"
  - "知识蒸馏"
  - "模型修正"
---

# Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding

**会议**: ICCV 2025  
**arXiv**: [2506.22803](https://arxiv.org/abs/2506.22803)  
**代码**: [https://github.com/XiGuaBo/CBM-HNMU](https://github.com/XiGuaBo/CBM-HNMU)  
**领域**: 目标检测  
**关键词**: 可解释性, 概念瓶颈模型, 黑盒干预, 知识蒸馏, 模型修正

## 一句话总结

提出 CBM-HNMU 框架，通过概念瓶颈模型（CBM）逼近黑盒模型的推理过程，自动识别并修正有害概念，再将修正后的知识蒸馏回黑盒模型，实现超越样本级别的系统性模型干预与准确率提升。

## 研究背景与动机

深度学习模型的决策过程日趋不透明，现有可解释性方法主要分为两类：

**特征归因方法**（Grad-CAM、Saliency Maps 等）：能解释模型关注"哪里"，但无法回答"关注了什么特征"，且缺乏干预能力

**概念解释方法**（ACE、CRAFT 等）：能提取人类可理解的视觉概念，但仍停留在被动解释层面

概念瓶颈模型（CBM）引入了可干预的框架，但存在两大问题：(1) 表示能力受限，分类精度通常低于同backbone黑盒模型；(2) 干预需要人类先验知识且局限于样本级别。Post-hoc CBM 通过残差连接部分缓解了精度问题，但削弱了可解释性且干预效果有限。

CBM-HNMU 的核心动机在于：能否构建一个闭环框架，让人类"理解"神经网络的推理（解释），同时让神经网络"理解"人类的知识（干预修正）？

## 方法详解

### 整体框架

CBM-HNMU 包含三个阶段：(a) 局部逼近：用 CBM 逼近黑盒模型在混淆类上的推理；(b) 概念干预：基于梯度贡献自动识别有害概念并修正；(c) 知识转移：将修正后的 CBM 知识蒸馏回黑盒模型。

### 关键设计

1. **混淆类选择**：在验证集上评估黑盒分类器，统计类间错分频率，提取错分频率最高的类对，形成混淆类集合 $\Gamma \subseteq \{1, 2, \ldots, N_{class}\}$。这样做的动机是限制概念瓶颈的规模，使 CBM 能用有限概念集精确逼近黑盒在这些类上的推理。

2. **概念通信**：利用 CRAFT 从黑盒中间层提取视觉概念 $C^{|}$，用 ChatGPT-3.5-Turbo 生成自然语言概念瓶颈 $C^t$，再通过 CLIP 的图像/文本编码器将两者映射到共享空间 $\mathbb{R}^{1 \times d}$（$d=512$）。概念分数计算为：

$$S_i = \frac{1}{n} \sum_{k=1}^{n} E_{img}(C_k^{|}(i)) \times E_{text}(C^t)^T$$

3. **局部逼近**：构建概念权重矩阵 $W \in \mathbb{R}^{N_\Gamma \times N_c}$，令 CBM 输出 $P_{cbm}(x_i) = S_i \times W^T$，通过 L2 范数与黑盒在混淆类上的输出 $P_{org}^\Gamma$ 对齐。

4. **基于梯度的概念干预**：核心创新。定义两类有害概念：

    - $S_{nT}$：对正确类有负面影响的概念
    - $S_{pF}$：对错误预测类有正面影响的概念
   
   通过梯度归因 $G(w_k, P_k(x_i)) = \frac{\partial P_k(x_i)}{\partial w_k} \odot w_k$ 计算概念贡献，全局累积后排序，选择 Top-$\bar{q}$ 个最有害概念进行删除/替换。

5. **知识转移**：将干预后 CBM 预测（混淆类）与原始黑盒的冻结预测（非混淆类）拼接为教师信号 $P_t$，通过概率残差系数 $pr = 1 - |softmax(P_{org})^{\Gamma^\complement}|_1$ 确保非干预类的预测分布不变，用交叉熵蒸馏回黑盒。

### 损失函数 / 训练策略

- 局部逼近：L2 损失对齐 CBM 与黑盒，学习率 1e-4，200 epochs
- 知识转移：交叉熵蒸馏损失，教师温度 $T_1=2.0$，学生温度 $T_2=1.5$，学习率 3e-7，10 epochs
- 整个流程无需额外标注，CLIP + ChatGPT 实现全自动化

## 实验关键数据

### 主实验

| 模型 | Flower-102 | CIFAR-10 | CIFAR-100 | CUB-200 | FGVC-Aircraft | 平均提升 |
|------|-----------|---------|----------|---------|--------------|---------|
| NFResNet50 w/o → w/ | 94.28→95.36 | 80.92→83.56 | 73.58→73.92 | 62.41→62.67 | 64.63→64.94 | ↑0.93% |
| ViT_Small w/o → w/ | 97.24→98.20 | 91.51→92.84 | 81.26→82.43 | 74.75→75.39 | 69.88→70.94 | ↑1.03% |
| GCVit w/o → w/ | 93.58→95.20 | 80.20→80.97 | 72.38→72.55 | 76.99→77.64 | 69.81→71.41 | ↑0.96% |

*所有模型在所有数据集上均获得正向提升，最大单项提升 2.64%（NFResNet50 on CIFAR-10）。*

### 消融实验

| 对比方法 | Flower-102 | CUB-200 | FGVC-Aircraft |
|---------|-----------|---------|--------------|
| CBM-HNMU (ours) | 95.36 | 62.67 | 64.94 |
| 随机概念干预 | 94.72 (↓0.64) | 62.51 (↓0.16) | 64.75 (↓0.19) |
| CBM (same backbone) | 92.57 | 54.53 | 56.31 |
| Post-hoc CBM | 93.58 | 58.20 | 64.45 |
| 黑盒基线 | 94.28 | 62.41 | 64.63 |

*CBM-HNMU 超越 CBM 和 Post-hoc CBM，且是唯一能超越黑盒基线精度的概念瓶颈方法。*

### 关键发现

- 干预类比例 < 25% 时效果最稳定，全类干预会导致逼近偏差显著增大
- 概念替换（从外部搜索集寻找替代概念）可进一步提升效果
- 用户研究（30 人）确认干预概念与人类视觉感知一致，置信度均 > 0.5
- 非干预类在干预前后精度基本不变，验证了 $pr$ 系数的有效性

## 亮点与洞察

- 创新性地构建了"理解→干预→修正"的闭环框架，突破了现有概念解释方法只能被动解释的局限
- 全流程无需人类标注（CLIP + ChatGPT），极大降低了实际部署成本
- 干预机制不局限于样本级别，而是通过修改黑盒参数实现全局性修正

## 局限与展望

- 概念提取依赖 CRAFT，该方法并非模型导向，概念与样本的关联可能不够精确
- ChatGPT 生成的概念瓶颈可能包含抽象概念（幻觉），不利于人类理解
- 干预效果依赖混淆类选择的质量

## 相关工作与启发

- 与 Beyond CBM 相比，HNMU 不仅诊断偏差，还通过蒸馏修改黑盒参数
- 概念替换策略（附录 A）为未来研究提供了在概念空间中进行更精细调控的思路
- 框架可扩展到检测/分割等任务的可解释性干预

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting](../../CVPR2025/object_detection/t2icount_enhancing_cross-modal_understanding_for_zero-shot_counting.md)
- [\[CVPR 2025\] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](../../CVPR2025/object_detection/efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [\[CVPR 2026\] Black-Box Domain Adaptation for Object Detection with Retention-Driven Knowledge Compression](../../CVPR2026/object_detection/black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)

</div>

<!-- RELATED:END -->
