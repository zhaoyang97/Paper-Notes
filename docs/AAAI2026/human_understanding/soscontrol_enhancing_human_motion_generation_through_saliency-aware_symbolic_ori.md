---
title: >-
  [论文解读] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control
description: >-
  [AAAI2026][人体理解][人体运动生成] 提出Salient Orientation Symbolic (SOS) script——基于Labanotation启发的可编程符号化运动表示框架，通过时序约束的凝聚聚类提取关键帧显著性，结合SMS数据增强和梯度优化的SOSControl框架实现对身体部位朝向和运动时序的精确控制，在HumanML3D上SOS-Acc达0.988且FID仅3.892。
tags:
  - "AAAI2026"
  - "人体理解"
  - "人体运动生成"
  - "符号化控制"
  - "Labanotation"
  - "扩散模型"
  - "显著性检测"
  - "ControlNet"
---

# SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control

**会议**: AAAI2026  
**arXiv**: [2601.14258](https://arxiv.org/abs/2601.14258)  
**作者**: Ho Yin Au, Junkun Jiang, Jie Chen (香港理工大学)  
**代码**: [GitHub](https://github.com/asdryau/SOSControl)  
**领域**: 人体理解  
**关键词**: 人体运动生成, 符号化控制, Labanotation, 扩散模型, 显著性检测, ControlNet

## 一句话总结

提出Salient Orientation Symbolic (SOS) script——基于Labanotation启发的可编程符号化运动表示框架，通过时序约束的凝聚聚类提取关键帧显著性，结合SMS数据增强和梯度优化的SOSControl框架实现对身体部位朝向和运动时序的精确控制，在HumanML3D上SOS-Acc达0.988且FID仅3.892。

## 背景与动机

文本驱动的人体运动生成因其在影视内容创作、机器人和人机协作中的应用前景而受到广泛关注。然而，文本描述天然具有主观性和歧义性——当用户要求"蹲下后向前出拳"时，传统text-to-motion框架无法精确控制手臂朝向（是正拳还是下勾拳）和动作时序（出拳时机是何时到达峰值）。

现有增强控制性的研究探索使用**关节关键帧位置**作为额外条件信号（如OmniControl、GMD）。但关节位置控制存在根本性局限：(1) 仅提供位置引导，无法指定身体部位**朝向**；(2) 模型可能将终点位置误解为中间路径点，导致运动"过冲"破坏时序；(3) 在3D空间中精确定义合理的关键帧位置需要大量手动调整和对运动动力学的深入理解，在工业动画流水线中极不实用。

Labanotation是舞蹈领域广泛使用的符号化运动记录系统，使用方向符号在五线谱上标注身体部位运动。本文受其启发，设计了面向方向和时序控制的SOS script，将运动控制从"精确3D坐标"抽象到"方向符号+时间位置"的层次，同时引入基于显著性的自动提取管线，使符号稀疏、可解释且可编程。

## 核心问题

如何设计一种直观、可编程的运动控制接口，使用户能以符号化方式精确指定身体部位朝向和运动时序，同时确保生成的运动自然流畅且与控制信号高度对齐？

## 方法详解

### 整体框架

SOSControl包含两大部分：(1) SOS提取管线：从运动数据自动提取显著性感知的符号脚本；(2) SOSControl生成框架：以SOS script为条件进行扩散式运动生成。

### 关键设计1：SOS Script与自动提取

**朝向特征提取**：使用PRPP (Pairwise Relative Position Phrase)计算每个身体部位的朝向特征：

$$\mathbf{o}_t^J = \text{PRPP}(e^J, a^J)_t = (\mathbf{l}_t(e^J) - \mathbf{l}_t(a^J)) \cdot \mathbf{r}_t$$

其中 $e^J$ 为末端关节，$a^J$ 为锚定关节，$\mathbf{r}_t$ 为自我中心参考方向。

**空间特征量化**：定义26个单位方向向量 $\mathbf{u} \in \mathbb{R}^{26 \times 3}$，通过可微softmax量化得到离散方向符号：

$$\mathbf{q} = \text{softmax}\left(\frac{\mathbf{o}}{\|\mathbf{o}\|} \cdot \mathbf{u}^T\right) \cdot \mathbf{u}$$

**时序显著性检测**：对每个身体部位的朝向特征变化率进行时序约束的凝聚聚类（connectivity矩阵限制仅相邻时序段可合并），构建自底向上的分割树。每个节点的合并距离作为对应帧的显著性值——合并距离越大，表示该帧前后运动变化越显著。

**显著性掩码方案(SMS)**：设定阈值过滤低显著性帧，仅保留高显著性关键帧的方向符号，生成稀疏、可解释的SOS staff。

### 关键设计2：周期潜空间扩散

采用ACTOR-PAE将运动编码为周期参数（频率 $\mathbf{f}$、振幅 $\mathbf{a}$、偏置 $\mathbf{b}$、相移 $\mathbf{s}$），生成周期信号：

$$\mathbf{p} = \mathbf{a}\sin(\mathbf{f} \cdot (N - \mathbf{s})) + \mathbf{b}$$

在此周期潜空间中训练MDM式扩散模型 $\mathcal{D}^-$，损失为：

$$\mathcal{L}_{\mathcal{D}^-} = \|\mathbf{p}^0 - \mathcal{D}^-(k, \mathbf{c}, \mathbf{p}^k)\|_2$$

### 关键设计3：SOS引导注入

**ControlNet适配**：对扩散模型和ACTOR-PAE解码器均应用ControlNet架构，冻结原模型参数并训练可学习副本，使SOS条件 $\mathbf{d}$ 注入生成过程：

$$\mathcal{L}_{\mathcal{D}^+} = \|\mathbf{p}^0 - \mathcal{D}^+(k, \mathbf{c}, \mathbf{d}, \mathbf{p}^k)\|_2$$

**梯度迭代优化**：通过可微的朝向特征提取管线，在测试时对周期潜变量进行梯度下降：

$$\mathbf{p}^* = \mathbf{p}^* - \nabla_{\mathbf{P}}\|\mathcal{M}_{\mathbf{d}}(\hat{\mathbf{q}}) - \mathbf{d}\|_2$$

其中 $\mathcal{M}_{\mathbf{d}}$ 为SOS掩码，仅在可见显著区域计算朝向差异。

**SMS数据增强**：训练时对每个身体部位随机采样显著性阈值 $m^{J} \sim \mathcal{U}(0,1)$，生成不同粒度的SOS script，使模型适应用户可能提供的各种稀疏程度的控制信号。

## 实验关键数据

### 主实验：HumanML3D上SOS条件运动生成

| 方法 | SOS-Acc↑ | L2-Rot6D↓ | FID↓ | MMD↓ |
|------|---------|----------|------|------|
| MDM (baseline, 无SOS) | 0.151 | 0.351 | 2.592 | 6.001 |
| GMD 1-stage | 0.113 | 0.427 | 25.669 | 7.835 |
| GMD 2-stage | 0.120 | 0.402 | 21.278 | 7.823 |
| OmniControl | 0.873 | 0.325 | 3.975 | 6.095 |
| TLControl | 0.982 | 0.341 | 11.132 | 7.066 |
| **SOSControl (Ours)** | **0.988** | **0.325** | **3.892** | **6.199** |

SOSControl在控制信号对齐(SOS-Acc)和运动质量(FID)上同时达到最优，GMD类方法因无法直接impute SOS到运动信号而效果极差。

### 消融实验：各模块贡献

| 配置 | SOS-Acc↑ | L2-Rot6D↓ | FID↓ | MMD↓ |
|------|---------|----------|------|------|
| w/o SMS data proc. | 0.991 | 0.499 | 13.494 | 6.893 |
| w/o ACTOR-PAE | 0.956 | 0.323 | 3.025 | 5.988 |
| w/o ControlNet (both) | 0.956 | 0.333 | 4.611 | 6.258 |
| w/o Iter. Opt. (both) | 0.531 | 0.329 | 5.570 | 6.382 |
| **Full** | **0.988** | **0.325** | **3.892** | **6.199** |

关键发现：(1) 去掉SMS数据增强导致FID从3.892暴涨至13.494——模型无法适应不同粒度的SOS输入；(2) 去掉迭代优化导致SOS-Acc从0.988骤降至0.531——优化是保证控制精度的核心。

### 迭代优化消融

| 方法 | 无优化 | Diff-time优化 | Test-time优化 | 两者同时 |
|------|--------|-------------|-------------|---------|
| OmniControl SOS-Acc | 0.674 | 0.873 | 0.956 | 0.956 |
| Ours SOS-Acc | 0.531 | 0.535 | 0.988 | 0.988 |
| Ours FID | 5.570 | 5.187 | 4.209 | 3.892 |

Test-time优化贡献最大，diffusion-time优化效果有限（其调整可能被后续扩散步骤覆盖）。

## 亮点

- **创新的运动控制范式**：从"精确3D坐标"抽象到"方向符号+时间位置"，大幅降低用户指定运动约束的门槛，更符合工业动画流水线的实际需求
- **显著性感知的稀疏表示**：通过凝聚聚类自动检测关键帧，生成稀疏可解释的SOS staff，避免逐帧密集标注的繁琐
- **SMS数据增强的关键贡献**：随机采样显著性阈值使模型适应各种稀疏度的SOS输入，去掉后FID暴涨3.5倍
- **ACTOR-PAE解码器的稳定性保证**：周期潜空间的正则化特性使稀疏引导自然传播到相邻帧，避免了OmniControl中test-time优化仅影响单帧导致的运动不一致问题

## 局限与展望

- **26个方向符号的表达力有限**：离散化不可避免丢失精细朝向信息，对要求高精度方向控制的应用（如手术机器人运动规划）可能不够
- **推理速度较慢**：100步扩散推理 + 100步test-time迭代优化导致总推理时间约17秒/batch，其中前向运动学计算是主要瓶颈
- **仅在HumanML3D上评估**：未在其他数据集（如BABEL、KIT-ML）上验证泛化性，SOS提取中的聚类超参数可能需要对不同数据集重新调优

## 与相关工作的对比

- **OmniControl (Xie et al. 2024)**：使用关节关键帧位置+ControlNet+diffusion-time优化，SOS-Acc仅0.873且test-time优化导致运动不一致（仅影响单帧）
- **TLControl (Wan et al. 2024)**：Transformer+VQ-VAE编码+test-time优化，SOS-Acc高达0.982但FID=11.132——VQ-VAE的有限codebook影响运动表达力和平滑度
- **GMD (Karunratanakul et al. 2023)**：依赖直接impute控制信号到运动信号，不适用于SOS这种非直接映射的抽象控制信号
- **PriorMDM (Shafir et al. 2024)**：同样依赖imputation和噪声零化，不适用于SOS
- **KP/PoseScript等运动描述符**：提取朝向特征但不处理运动显著性检测，表述逐帧密集，编程成本高

## 启发与关联

- **符号化控制的通用性**：SOS script将运动控制抽象为"什么部位+什么方向+什么时间"的简洁三元组，这种范式可扩展到机器人任务规划、舞蹈编排等
- **显著性提取的思路**：凝聚聚类检测运动显著性的方法可迁移到视频摘要、运动压缩等需要关键帧提取的任务
- **SMS数据增强的启示**：在稀疏控制信号场景下，训练时随机掩码使模型适应不同稀疏度是一个通用策略，类似于MAE/BERT的masking思想

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从Labanotation到可编程SOS script的创新控制范式，首次将显著性聚类引入运动生成控制
- 实验充分度: ⭐⭐⭐⭐ — 全面的对比、消融和优化策略分析，但仅在HumanML3D单数据集验证
- 写作质量: ⭐⭐⭐⭐ — 系统完整、模块清晰，但整体pipeline较复杂，读者学习成本较高
- 价值: ⭐⭐⭐⭐ — 提出了实用的运动控制新范式，代码开源，对动画和人机交互领域有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](../../CVPR2026/human_understanding/lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[CVPR 2026\] FrankenMotion: Part-level Human Motion Generation and Composition](../../CVPR2026/human_understanding/frankenmotion_part-level_human_motion_generation_and_composition.md)
- [\[AAAI 2026\] mmPred: Radar-based Human Motion Prediction in the Dark](mmpred_radar-based_human_motion_prediction_in_the_dark.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
