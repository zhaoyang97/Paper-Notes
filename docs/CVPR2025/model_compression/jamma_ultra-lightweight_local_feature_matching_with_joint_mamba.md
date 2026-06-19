---
title: >-
  [论文解读] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba
description: >-
  [CVPR 2025][模型压缩][特征匹配] JamMa提出了基于Joint Mamba的超轻量级半密集特征匹配器，通过JEGO扫描-合并策略实现跨视角联合扫描、高效四方向扫描、全局感受野和全方向特征表示，以不到50%的参数和FLOPs实现了优于Transformer-based匹配器的性能-效率平衡。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "特征匹配"
  - "Mamba"
  - "状态空间模型"
  - "轻量级"
  - "半密集匹配"
---

# JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba

**会议**: CVPR 2025  
**arXiv**: [2503.03437](https://arxiv.org/abs/2503.03437)  
**代码**: [https://leoluxxx.github.io/JamMa-page/](https://leoluxxx.github.io/JamMa-page/)  
**领域**: 模型压缩  
**关键词**: 特征匹配, Mamba, 状态空间模型, 轻量级, 半密集匹配

## 一句话总结
JamMa提出了基于Joint Mamba的超轻量级半密集特征匹配器，通过JEGO扫描-合并策略实现跨视角联合扫描、高效四方向扫描、全局感受野和全方向特征表示，以不到50%的参数和FLOPs实现了优于Transformer-based匹配器的性能-效率平衡。

## 研究背景与动机

1. **领域现状**：特征匹配是SfM、SLAM等任务的基础。当前最先进的匹配器分为稀疏方法（SuperGlue、LightGlue等，依赖关键点检测器）和半密集/密集方法（LoFTR、ASpanFormer等，在网格点间建立对应关系）。后者通过Transformer建模长距离依赖实现了在纹理缺失场景的鲁棒匹配。

2. **现有痛点**：Transformer的 $O(N^2)$ 复杂度在处理高分辨率图像时导致训练成本高、推理延迟大。即使使用线性注意力，参数量和计算量仍然偏高。

3. **核心矛盾**：长距离依赖建模能力（对匹配至关重要）与计算效率之间的trade-off。

4. **本文目标** 利用Mamba的线性复杂度 $O(N)$ 替代Transformer，构建超轻量级半密集匹配器。但Mamba是单序列因果模型，用于双图像特征匹配面临三个挑战：(1)缺乏互交互；(2)单向性；(3)因果性导致感受野不均衡。

5. **切入角度**：针对Mamba的三个挑战设计全新的扫描策略——联合扫描实现跨视角交互，跳步扫描+四方向保持效率和全方向性，局部聚合器补偿感受野不均衡。

6. **核心 idea**：Joint Mamba通过JEGO策略（Joint联合扫描+Efficient高效跳步+Global全局感受野+Omnidirectional全方向）在线性复杂度下实现类Transformer的特征匹配能力。

## 方法详解

### 整体框架
输入为图像对 $(I_A, I_B)$，经CNN编码器（ConvNeXt V2，0.65M参数）提取粗特征 $F^c$（1/8分辨率）和细特征 $F^f$（1/2分辨率）。粗特征经JEGO Scan → 4个独立Mamba块 → JEGO Merge处理后得到增强的跨视角特征。最后通过粗到细匹配(C2F)模块生成最终匹配结果（粗匹配→细匹配→亚像素精化）。

### 关键设计

1. **联合扫描 (Joint Scan)**:

    - 功能：实现两幅图像特征的高频互交互
    - 核心思路：将两幅图像的粗特征分别进行水平拼接 $X^h = [F_A^c | F_B^c]$ 和垂直拼接 $X^v = [F_A^c; F_B^c]$，然后在拼接后的特征图上进行行/列扫描。关键在于扫描方向使得序列中两图像的特征交替出现（"joint"），而非先扫完一张图再扫另一张（"sequential"）。例如水平扫描时，每一行包含A和B的特征，扫描过程中A和B的特征高频交替进入Mamba的状态空间
    - 设计动机：实验证明联合扫描比顺序扫描在位姿估计AUC上高出约2.5个百分点。直觉上，特征匹配需要两图之间的密切交互，联合扫描让Mamba的隐状态同时携带两图信息

2. **JEGO四方向高效扫描与合并**:

    - 功能：以总序列长度 $N$（而非 $2N$ 或 $4N$）实现全方向扫描和全局感受野
    - 核心思路：采用EVMamba的跳步扫描策略（步长 $p=2$）减少每个方向的序列长度为 $N/4$，同时将四个序列的起始点 $(m,n)$ 安排在不同位置以覆盖四个方向（右、左、上、下）。四个序列由独立Mamba块处理后，通过JEGO Merge恢复到2D特征图，分离两图特征，并使用门控卷积聚合器（3×3 Conv）融合四个方向的信息。聚合器公式：$\sigma = \text{GELU}(\text{Conv}_3(\tilde{F}^c))$，$\hat{F}^c = \text{Conv}_3(\sigma \cdot \text{Conv}_3(\tilde{F}^c))$
    - 设计动机：VMamba的四方向扫描虽然全面但总序列长度为 $4N$，EVMamba的跳步扫描效率高但只有前向扫描（感受野限于右下角且非全方向）。JEGO策略通过精心安排四个方向的起止点，使得感受野在空间上互补——小感受野的特征总与大感受野的特征相邻，简单的3×3卷积聚合就能让每个特征都获得全局全方向信息

3. **粗到细匹配模块 (C2F)**:

    - 功能：从增强的特征中生成最终的亚像素级匹配
    - 核心思路：粗匹配阶段计算双向概率矩阵 $P_{A\to B}$ 和 $P_{B\to A}$，取union获得many-to-one匹配（比Dual-Softmax的one-to-one更鲁棒）。细匹配阶段在5×5窗口内用MLP-Mixer交互，通过Dual-Softmax建立一对一细匹配。亚像素精化阶段用回归预测偏移量
    - 设计动机：采用XoFTR的匹配框架，粗匹配的双向many-to-one策略比传统Dual-Softmax在无纹理区域更鲁棒

### 损失函数 / 训练策略
总损失：$\mathcal{L} = \mathcal{L}_c + \mathcal{L}_f + \mathcal{L}_s$。粗匹配损失 $\mathcal{L}_c$ 为双向focal loss；细匹配损失 $\mathcal{L}_f$ 为focal loss；亚像素损失 $\mathcal{L}_s$ 为对称极线距离。训练在MegaDepth上30 epochs，batch size 2，AdamW优化器，初始学习率0.0002+cosine衰减。单张4090 GPU训练约50小时。

## 实验关键数据

### 主实验：MegaDepth相对位姿估计

| 类别 | 方法 | Params(M) | FLOPs(G) | Time(ms) | AUC@5° | AUC@10° | AUC@20° |
|------|------|-----------|----------|----------|--------|---------|---------|
| 稀疏 | SP+LG | 13.2 | 459.9 | 84.2 | 58.8 | 73.6 | 84.1 |
| 半密集 | LoFTR | 11.6 | 815.4 | 117.5 | 62.1 | 75.5 | 84.9 |
| 半密集 | ASpanFormer | 15.8 | 882.3 | 155.7 | 62.6 | 76.1 | 85.7 |
| 半密集 | ELoFTR | 16.0 | 968.8 | 69.6 | 63.7 | 77.0 | 86.4 |
| **半密集** | **JamMa** | **5.7** | **202.9** | **59.9** | **64.1** | **77.4** | **86.5** |
| 密集 | RoMa | 111.3 | 2014.3 | 824.9 | 68.5 | 80.6 | 88.8 |

### 消融实验

| 配置 | Time(ms) | AUC@5° | AUC@10° | AUC@20° | 说明 |
|------|----------|--------|---------|---------|------|
| JamMa | 3.2 | 64.5 | 77.3 | 86.3 | 完整模型 |
| 顺序扫描替代联合扫描 | 3.2 | 62.2 | 74.7 | 83.7 | AUC@5°掉2.3% |
| 去掉聚合器 | 3.0 | 62.3 | 75.1 | 84.3 | AUC@5°掉2.2% |
| 用EVMamba扫描 | 3.0 | 61.9 | 74.8 | 84.1 | 无全方向+无全局 |
| 用VMamba扫描 | 9.7 | 64.1 | 77.1 | 86.2 | 效果相近但慢3× |
| 用线性注意力 | 24.3 | 64.2 | 77.0 | 86.1 | 效果相近但慢7.6× |
| 无交互层 | 0 | 60.1 | 73.0 | 82.6 | 基线 |

### 关键发现
- JamMa在半密集方法中综合排名第一（平均排名3.0 vs 期望7.5），性能-效率平衡优势明显
- 联合扫描vs顺序扫描：联合扫描AUC@5°高2.3%，说明高频互交互对特征匹配至关重要
- 聚合器虽然只是简单3×3 Conv，但去掉后AUC@5°掉2.2%，说明局部信息聚合对弥补感受野不均衡效果显著
- JamMa粗层仅3.2ms vs 线性注意力24.3ms（7.6×加速），总推理仅59.9ms
- 仅5.7M参数，是ASpanFormer的36%、LoFTR的49%

## 亮点与洞察
- **联合扫描+高频互交互**：将两图特征交错排列而非顺序排列，使Mamba的隐状态持续携带跨视角信息。这个insight对任何需要双序列交互建模的Mamba应用都有启发（如文档比较、立体匹配）
- **平衡感受野+简单聚合器=全局全方向**：精心安排四个方向的起止点使得感受野在空间上互补，然后仅用一个3×3 Conv就能让每个特征获得全局信息。相比VMamba的4×序列长度和Vim的2×，JEGO保持总序列长度 $N$ 不变
- **Mamba在视觉匹配中的首次成功**：证明线性复杂度的SSM可以替代二次复杂度的Transformer用于特征匹配，且性能更好。为其他计算密集型视觉任务提供了轻量化路径

## 局限与展望
- 仅在MegaDepth一个数据集上训练，未在其他任务上微调，泛化性可能受限
- 密集匹配器（DKM、RoMa）在纯精度上仍然领先，JamMa的优势主要在效率端
- Mamba的因果性虽然通过四方向扫描+聚合器缓解，但仍不如真正的全局注意力灵活
- 可改进方向：(1) 引入Mamba2的更高效并行计算；(2) 将JEGO策略应用到更多视觉Mamba任务；(3) 探索动态跳步步长以适应不同分辨率

## 相关工作与启发
- **vs LoFTR/ASpanFormer**: 基于Transformer的半密集匹配器，JEGO Mamba以约1/3的参数和1/4的FLOPs实现了更好的性能。核心优势来自Mamba的线性复杂度和JEGO的高效扫描
- **vs ELoFTR**: ELoFTR使用高效注意力加速LoFTR，JamMa进一步用Mamba替代注意力，参数更少（5.7M vs 16.0M）且速度更快（59.9ms vs 69.6ms）
- **vs VMamba/EVMamba**: 视觉Mamba模型的扫描策略设计。VMamba四方向全面但4×序列长度，EVMamba高效但牺牲了全方向性和全局感受野。JEGO策略兼顾二者优势

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ JEGO策略的设计巧妙且系统（联合+高效+全局+全方向），Joint Mamba概念新颖
- 实验充分度: ⭐⭐⭐⭐ 位姿估计+单应矩阵估计+详细消融，缺少更多下游任务评估
- 写作质量: ⭐⭐⭐⭐⭐ 图示直观清晰（尤其是感受野可视化），逻辑层层递进
- 价值: ⭐⭐⭐⭐⭐ 开创了Mamba用于视觉匹配的新方向，超轻量且性能出色的实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)
- [\[CVPR 2025\] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](binarized_mamba-transformer_for_lightweight_quad_bayer_hybridevs_demosaicing.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/model_compression/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[ICML 2025\] Joker: Joint Optimization Framework for Lightweight Kernel Machines](../../ICML2025/model_compression/joker_joint_optimization_framework_for_lightweight_kernel_machines.md)

</div>

<!-- RELATED:END -->
