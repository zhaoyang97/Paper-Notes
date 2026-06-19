---
title: >-
  [论文解读] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise
description: >-
  [AAAI 2026][事故预测] 提出统一框架，将基于扩散模型的双层去噪模块与时间感知的Actor-Critic强化学习模型结合，在传感器噪声条件下实现鲁棒的长期交通事故预测，在三个基准数据集上取得了准确率（AP）和平均事故前预警时间（mTTA）的最优性能。 交通事故预测——在碰撞发生前预测其可能性——是自动驾驶的关键能力…
tags:
  - "AAAI 2026"
  - "事故预测"
  - "扩散去噪"
  - "Actor-Critic强化学习"
  - "传感器噪声"
  - "长期时间推理"
---

# Predict and Resist: Long-Term Accident Anticipation under Sensor Noise

**会议**: AAAI 2026  
**arXiv**: [2511.08640](https://arxiv.org/abs/2511.08640)  
**代码**: 无  
**领域**: 其他  
**关键词**: 事故预测, 扩散去噪, Actor-Critic强化学习, 传感器噪声, 长期时间推理

## 一句话总结

提出统一框架，将基于扩散模型的双层去噪模块与时间感知的Actor-Critic强化学习模型结合，在传感器噪声条件下实现鲁棒的长期交通事故预测，在三个基准数据集上取得了准确率（AP）和平均事故前预警时间（mTTA）的最优性能。

## 研究背景与动机

交通事故预测——在碰撞发生前预测其可能性——是自动驾驶的关键能力。与传统感知系统仅在事故发生后进行检测不同，预测能力使得主动安全干预成为可能，如及时刹车或规避操作，从而将安全模式从"事后反应"转变为"事前预防"。

然而，实现可靠的事故预测面临两个**相互耦合**的核心挑战：

**不完美感知下的鲁棒性**：自动驾驶车辆的传感器不可避免地受到雨水、眩光、灰尘、镜头损伤和运动模糊等因素影响。在这些噪声条件下，基于单帧的短期预测变得不可靠。讽刺的是，这些正是最需要更长时间推理的场景——通过跨时间累积微弱信号，模型可以在单帧被破坏时仍提取有意义的模式。

**"何时预警"问题**：现有方法大多聚焦于帧级分类或短期预测，能指示是否可能发生事故，但很少优化**何时**发出预警。在安全关键场景中，时机与准确性同样重要：预警过晚毫无用处，过早或过于频繁则会侵蚀信任，甚至诱发不安全反应。这本质上是一个**长期信用分配问题**：模型必须识别微妙的早期线索，维持时间推理，并确定最佳行动时刻。

**关键洞察**：这两个挑战相互放大。传感器退化增加了即时观测的不确定性，进而放大了长期时间推理以稳定预测的需求；反之，没有有效时间信用分配的模型无法利用跨帧冗余来克服噪声感知。因此，一个真正可部署的预测系统必须将时机和鲁棒性作为**单一耦合问题**来处理。

## 方法详解

### 整体框架

本文将事故预测重新建模为**不确定性下的序列决策问题**。框架由五个核心组件构成：目标检测器、特征提取器、自适应目标感知模块、图像/目标双层扩散去噪模块、以及Actor-Critic决策模块。

给定视频 $V = \{V_t\}_{t=1}^T$，可学习函数 $f_\theta$ 预测逐帧事故概率：

$$p_t = f_\theta(V_{1:t}), \quad t = 1, \ldots, T$$

事故前预警时间（TTA）定义为首次置信预测到实际事故帧的间隔：

$$\Delta t = \tau - t_o \quad \text{其中} \quad t_o = \min\{t \in \{1,\ldots,T\} \mid p_t \geq p_{th}\}$$

### 关键设计

1. **目标检测与特征提取**：每帧通过Cascade R-CNN检测，选取Top-K动态智能体，用VGG-16编码为特征向量 $\mathbf{F}_{obj}$。全局特征 $\mathbf{F}_{img}$ 由VGG-16+MLP提取，编码场景上下文。

2. **自适应目标感知模块（Self-Adaptive Object-Aware Module）**：根据时间上下文和目标间交互，动态关注最具信息量的交通参与者。注意力机制计算如下：

    $\mathbf{e}_t = \tanh(\mathbf{W}_{wa}\mathbf{h}_{t-1} + \mathbf{W}_{ua}\mathbf{F}_{obj} + \mathbf{b}_a)$
    $\alpha_t = \text{softmax}(\mathbf{W}_w \mathbf{e}_t)$
    $\bar{\mathbf{F}}_{obj} = \alpha_t \odot \mathbf{F}_{obj}$

   该机制自适应地优先关注高风险目标，同时编码关键的时间交互信息。

3. **基于扩散的层次化特征增强（Diffusion-Based Hierarchical Feature Enhancement）**：这是本文的核心创新之一，在图像级和目标级分别进行扩散去噪。

   **前向扩散过程**采用方差保持的马尔可夫链：
    $\mathbf{F}_{img}^{noisy} = \sqrt{\bar{\alpha}_t}\mathbf{F}_{img} + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$

   线性噪声调度从 $\beta_{start} = 0.001$ 到 $\beta_{end} = 0.02$。

   **去噪网络**为轻量级前馈网络：
    $p_\theta(\mathbf{F}_{img}^{noisy}, t) = W_2(\text{ReLU}(W_1\mathbf{F}_{img}^{noisy} + b_1)) + b_2$

   **残差融合策略**保持语义保真度：
    $\mathbf{F}_{img}^{enhanced} = \mathbf{F}_{img} + \lambda \cdot p_\theta(\mathbf{F}_{img}^{noisy}, t), \quad \lambda = 0.15$

   对目标级特征 $\bar{\mathbf{F}}_{obj}$ 也应用相同的残差增强。该设计的本质是概率特征稳定器，类似于贝叶斯证据累积——通过迭代细化噪声输入为结构忠实且时间一致的表示，减少抖动和虚假激活。

4. **时间融合与GRU推理**：增强后的图像和目标特征拼接后送入GRU捕获序列依赖：
    $\mathbf{X}_t, \mathbf{h}_t = \text{GRU}(\text{concat}(\mathbf{F}_{img}^{enhanced}, \mathbf{F}_{obj}^{enhanced}))$
   MLP预测帧级事故概率 $p_t = \text{MLP}(\mathbf{X}_t)$，时间权重层计算时间权重损失 $w_t = \text{fc}(\mathbf{h}_t)$。

5. **Actor-Critic长期决策模块**：采用滚动缓冲区存储最近 $W$ 个隐藏状态，通过均值池化获得摘要向量 $\bar{\mathbf{h}}_t$。

   **Actor**将历史状态映射为离散动作分布：
    $\pi_t = \text{softmax}(\mathbf{W}_p \bar{\mathbf{h}}_t + \mathbf{b}_p)$

   **Critic**预测期望累积奖励：
    $V_t = \mathbf{w}_v^\top \bar{\mathbf{h}}_t + b_v$

   **奖励函数**平衡预测正确性和时间紧迫性：
    $r_t = \mathbb{I}(a_t = y_t) \cdot e^{-t/\tau} + \mathbb{I}(a_t \neq y_t) \cdot \gamma$
   其中正确预测的奖励随时间指数衰减（鼓励早期预测），错误预测给予固定惩罚（$\tau=5, \gamma=-0.5$）。

### 损失函数 / 训练策略

总损失函数由三部分组成：

$$\mathcal{L}_{total} = \mathcal{L}_{an} + \alpha(\mathcal{L}_{actor} + \beta \mathcal{L}_{critic})$$

其中 $\alpha = \beta = 0.5$。

- **预测损失 $\mathcal{L}_{an}$**：对正样本使用时间惩罚 $p = -\max(0, (t_{accident} - t_{current} - 1)/\text{fps})$ 和自适应时间权重 $\omega_t = 1 + \sigma(h_t)$，鼓励更早的预测
- **策略梯度损失**：$\mathcal{L}_{actor} = -\mathbb{E}[\log \pi_t(a_t) \cdot A_t] - \lambda_e \mathcal{H}(\pi_t)$，其中 $A_t = \tilde{r}_t - V_t$ 为优势函数，$\lambda_e = 0.1$ 控制熵正则化
- **价值损失**：$\mathcal{L}_{critic} = \frac{1}{2}(\tilde{r}_t - V_t)^2$

训练采用PyTorch 2.0，在NVIDIA RTX 3050上训练30个epoch，batch size 10，Adam优化器（初始学习率 $3 \times 10^{-4}$），ReduceLROnPlateau调度器。每帧包含至多19个目标，使用4096维VGG-16特征，256单元GRU建模时间动态。

## 实验关键数据

### 主实验

在三个基准数据集上与SOTA方法的对比：

| 方法 | 会议 | DAD AP(%) | DAD mTTA(s) | CCD AP(%) | CCD mTTA(s) | A3D AP(%) | A3D mTTA(s) |
|------|------|-----------|-------------|-----------|-------------|-----------|-------------|
| DSA | ACCV | 48.1 | 1.34 | 98.7 | 3.08 | 92.3 | 2.95 |
| ACRA | CVPR | 51.4 | 3.01 | 98.9 | 3.32 | - | - |
| AdaLEA | CVPR | 52.3 | 3.44 | 99.2 | 3.45 | 92.9 | 3.16 |
| UString | TIV | 53.7 | 3.53 | 99.5 | 3.74 | 93.2 | 3.24 |
| AccNet | AAP | 60.8 | 3.58 | 99.5 | 3.78 | 95.1 | 3.26 |
| LATTE | IF | 89.7 | 4.49 | 98.8 | 4.53 | 92.5 | 4.52 |
| **本文** | - | **91.2** | **4.59** | **99.8** | **4.29** | **95.7** | **4.60** |

在DAD上实现91.2% AP和4.59s mTTA，在CCD和A3D上也取得一致性提升。

### 消融实验

核心模块消融（CCD数据集）：

| 配置 | AP(%) | mTTA(s) | 说明 |
|------|-------|---------|------|
| 完整模型 | 99.8 | 4.29 | 基线 |
| w/o 目标感知模块 | 99.3 | 4.61 | mTTA略升但AP下降 |
| w/o 时间权重层 | 99.5 | 4.47 | 两项指标略降 |
| w/o 预测损失 | 33.3 | 5.00 | AP暴跌，说明预测损失是核心 |
| w/o 策略梯度损失 | 99.6 | 4.47 | 影响较小 |
| w/o 价值损失 | 92.8 | 3.03 | AP和mTTA均显著下降 |

噪声鲁棒性（CCD数据集，高斯噪声）：

| 噪声 $\sigma$ | 完整模型AP(%) | 完整模型mTTA(s) | w/o图像扩散AP | w/o目标扩散AP | w/o全部扩散AP |
|---------------|-------------|--------------|-------------|-------------|-------------|
| 原始 | 99.8 | 4.29 | 99.6 | 99.6 | 99.6 |
| 0.5 | 99.6 | 4.00 | 99.4 | 99.5 | 99.4 |
| 1.0 | 99.6 | 4.04 | 99.5 | 99.5 | 99.4 |
| 5.0 | 99.6 | 4.35 | 99.5 | 99.3 | 99.0 |
| 10.0 | 98.0 | 3.43 | 98.6 | 98.8 | 98.2 |
| 20.0 | 91.6 | 3.05 | 92.8 | 91.4 | 91.0 |

奖励-惩罚权衡实验（A3D数据集）：

| 奖励倍率 | 惩罚倍率 | AP(%) | mTTA(s) |
|----------|----------|-------|---------|
| ×1 (5.0) | ×1 (-0.5) | 95.7 | 4.60 |
| ×10 | ×1 | 93.6 | 4.77 |
| ×50 | ×1 | 92.7 | 4.70 |
| ×0.1 | ×1 | 96.2 | 4.47 |
| ×1 | ×10 | 91.2 | 4.92 |
| ×1 | ×0.1 | 92.1 | 4.71 |

### 关键发现

- **预测损失是不可或缺的核心**：移除后AP从99.8%暴跌至33.3%，同时也说明单纯追求更高mTTA不一定更好
- **价值损失对稳定长期决策至关重要**：移除后AP降至92.8%，mTTA从4.29s降至3.03s，说明价值估计在稳定长期决策中的关键作用
- **双层扩散模块在中等噪声下效果最佳**：$\sigma \leq 5$ 时完整模型保持99.6% AP；极端噪声下（$\sigma = 20$）所有变体均退化，且移除图像扩散有时反而提升AP，暗示过度去噪可能损害严重损坏的输入
- **奖励-惩罚的精妙权衡**：增大奖励权重会降低AP但提升mTTA（鼓励但过度冒进），增大惩罚权重使模型过于保守（最高mTTA=4.92s但最低AP=91.2%）
- **长期训练（history window=10）vs 帧级基线（window=0）**：长期模型在复杂多智能体雨天场景中产生更短、更少的虚假警报；在典型碰撞中提前近1秒预测

## 亮点与洞察

1. **耦合问题的统一解决**：将噪声鲁棒性和预警时机作为单一耦合问题处理，而非分别解决，这是本文的核心设计哲学
2. **扩散模型的特征空间应用**：不同于传统的图像级去噪，本文在特征空间中应用扩散去噪，更轻量且直接服务于下游任务
3. **残差融合的保守设计**：$\lambda=0.15$ 的小系数确保去噪输出作为原始特征的微调增量，避免破坏已有的语义信息
4. **RL框架的巧妙应用**：将"何时预警"建模为序列决策问题，time-weighted reward自然编码了"越早越好"的安全需求
5. **实用性导向**：在RTX 3050上即可训练，模型轻量（256单元GRU），具有很强的实际部署潜力

## 局限与展望

- **去噪过度问题**：极端噪声下（$\sigma \geq 10$），移除图像扩散有时反而更好，说明当前固定的残差系数 $\lambda$ 和扩散步数可能需要自适应调整
- **脉冲噪声50%时性能下降明显**：AP降至91.6%（A3D），在密集城市场景中可能不够可靠
- **目标检测器依赖**：使用Cascade R-CNN作为目标检测器，本身在噪声条件下可能漏检关键目标，形成级联错误
- **单一视角限制**：仅使用行车记录仪（dashcam）单一视角，未利用多传感器融合（LiDAR、雷达等）
- **数据集局限**：三个数据集（DAD、CCD、A3D）均为特定地区的城市场景，对高速公路、农村道路等场景的泛化能力未验证
- **固定时间窗口 $W$**：滚动缓冲区大小固定，可能需要根据场景复杂度动态调整
- **未考虑多模态输入**：仅使用视觉特征，未融合驾驶行为、车辆动态等信息

## 相关工作与启发

本文处于视觉事故预测领域的演进脉络中：

- **从帧级感知到时间推理**：CNN → RNN/LSTM/GRU 的发展路线
- **从孤立目标到交互建模**：GNN显式编码多智能体关系，Transformer捕获长程依赖
- **稀缺性和可解释性**：GAN/VAE合成稀缺事故场景，注意力机制提升可解释性

与扩散模型在感知任务中的应用趋势一致（如去噪、数据增强），本文创新性地将扩散过程应用于特征空间而非图像空间。强化学习的引入则参考了序列决策领域的经典方法，将安全预警的时机优化问题转化为信用分配问题。

**对后续研究的启发**：该框架的双层去噪+RL决策模式可推广到其他需要在噪声条件下做时序决策的安全关键场景，如工业设备故障预测、医疗异常检测等。

## 评分

- **创新性**: ⭐⭐⭐⭐ — 扩散去噪+Actor-Critic的组合设计新颖，问题建模视角独特
- **实验完整度**: ⭐⭐⭐⭐⭐ — 三个数据集、多种噪声条件、详尽消融和超参数分析
- **实用价值**: ⭐⭐⭐⭐ — 轻量级设计、噪声鲁棒、实际部署潜力大
- **写作质量**: ⭐⭐⭐⭐ — 问题动机阐述清晰，挑战耦合关系分析深入
- **综合评分**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Non-parametric Sensor Noise Modeling and Synthesis](../../ECCV2024/others/non-parametric_sensor_noise_modeling_and_synthesis.md)
- [\[CVPR 2026\] Learning Long-term Motion Embeddings for Efficient Kinematics Generation](../../CVPR2026/others/learning_long-term_motion_embeddings_for_efficient_kinematics_generation.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](../../NeurIPS2025/others/coresets_for_clustering_under_stochastic_noise.md)
- [\[CVPR 2026\] CHIRP dataset: towards long-term, individual-level, behavioral monitoring of bird populations in the wild](../../CVPR2026/others/chirp_dataset_towards_long-term_individual-level_behavioral_monitoring_of_bird_p.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](../../CVPR2025/others/effortless_active_labeling_for_long-term_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
