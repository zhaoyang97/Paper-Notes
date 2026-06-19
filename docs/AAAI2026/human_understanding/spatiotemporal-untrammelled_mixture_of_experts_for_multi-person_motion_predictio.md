---
title: >-
  [论文解读] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction
description: >-
  [AAAI 2026 Oral][人体理解][多人运动预测] 提出ST-MoE框架，首次将混合专家模型（MoE）与双向时空Mamba相结合用于多人运动预测，通过四种异构时空专家灵活捕获复杂时空依赖，实现SOTA精度的同时减少41.38%参数量，训练加速3.6倍。 多人运动预测（Multi-Person Motion Pred…
tags:
  - "AAAI 2026 Oral"
  - "人体理解"
  - "多人运动预测"
  - "混合专家模型"
  - "Mamba"
  - "时空建模"
  - "高效推理"
---

# Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction

**会议**: AAAI 2026 Oral  
**arXiv**: [2512.21707](https://arxiv.org/abs/2512.21707)  
**代码**: [https://github.com/alanyz106/ST-MoE](https://github.com/alanyz106/ST-MoE)  
**领域**: 人体理解  
**关键词**: 多人运动预测, 混合专家模型, Mamba, 时空建模, 高效推理

## 一句话总结

提出ST-MoE框架，首次将混合专家模型（MoE）与双向时空Mamba相结合用于多人运动预测，通过四种异构时空专家灵活捕获复杂时空依赖，实现SOTA精度的同时减少41.38%参数量，训练加速3.6倍。

## 研究背景与动机

多人运动预测（Multi-Person Motion Prediction, MPMP）旨在根据历史运动序列预测未来多人关节位置，在人机交互、自动驾驶、监控系统等领域有重要应用。

现有方法存在两个核心限制：

**时空表示不灵活**：
   - MRT使用固定模式的时空位置编码，缺乏灵活性
   - TBIFormer虽引入了轨迹感知相对位置编码增强空间感知，但身体部位连接操作增加了序列长度
   - IAFormer利用自注意力探索交互信息中的时空特征，性能好但效率差

**计算成本高**：
   - 基于Transformer的方法因自注意力的二次复杂度导致计算开销大
   - 随着人数增多，计算量急剧增长

核心动机：**能否设计一种既灵活又高效的新范式，全面捕获人体运动中的时空依赖？**

作者的洞察是：(1) MoE的动态激活机制可以实现灵活的子网络选择；(2) Mamba的线性复杂度可以替代二次复杂度的注意力机制；(3) 将两者结合可同时解决灵活性和效率问题。

## 方法详解

### 整体框架

输入运动序列 → DCT + Multi-Pose Encoder编码 → 门控路由器将特征分配给4种时空专家 → 专家输出经加权聚合 → Multi-Pose Decoder + iDCT解码 → 预测未来运动。

### 关键设计

#### 1. 问题定义与输入处理

定义第 $i$ 个人的历史运动序列 $\textbf{P}_{1:t}^i \in \mathbb{R}^{D \times t}$，预测未来运动 $\textbf{P}_{t+1:T}^i \in \mathbb{R}^{D \times (T-t)}$，其中 $D = J \times 3$（$J$个关节的3D坐标）。

**输入填充**：复制最后观测帧 $T-t$ 次并拼接到观测序列后，形成 $\textbf{P}_{\text{input}}^i \in \mathbb{R}^{D \times T}$。

**Multi-Pose Encoder**：采用IAFormer的3层GCN编码器，先做DCT变换增强表示紧凑性：

$$\textbf{F}_{\text{input}}^i = \text{ME}(\text{DCT}(\textbf{P}_{\text{input}}^i))$$

#### 2. 混合时空Mamba专家（MoSTME）

这是论文的核心创新。编码特征 $\textbf{F}_{\text{input}} \in \mathbb{R}^{B \times D \times T}$ 同时输入专家池和路由器。

**门控路由机制**：

$$\textbf{E}_{\text{output}} = \sum_{e=1}^{N} \textbf{f}_e(\textbf{F}_{\text{input}}) \textbf{p}_e$$

$$\textbf{p}_e = \text{softmax}(\text{TopK}(g(\textbf{F}_{\text{input}}), k))_e$$

其中 $g(\cdot)$ 是基于MLP的门控函数，TopK保留前k个条目的原始值，其余设为 $-\infty$，经softmax后接近零实现稀疏激活。实验验证全部激活（k=4）效果最佳。

**四种异构专家**——每种由双向空间Mamba和双向时间Mamba的不同组合构成：

| 专家类型 | 处理顺序 | 捕获模式 |
|----------|----------|----------|
| Spatial-Temporal (ST) | 空间→时间 | 先空间后时间的依赖 |
| Temporal-Temporal (TT) | 时间→时间 | 强时间依赖 |
| Temporal-Spatial (TS) | 时间→空间 | 先时间后空间的依赖 |
| Spatial-Spatial (SS) | 空间→空间 | 强空间依赖 |

以ST专家为例：

$$\textbf{F}'' = \text{rearrange}(\text{Bi-SMamba}(\textbf{F}_{\text{input}}))$$

$$\textbf{F}_{\text{output}_1} = \text{rearrange}(\text{Bi-TMamba}(\textbf{F}''))$$

**关键设计**：所有专家共享同一组双向时间Mamba和双向空间Mamba参数，仅组合顺序不同，大幅减少参数量。

#### 3. 双向时空Mamba

原始Mamba的单向建模限制了全局依赖捕获。引入双向扫描机制：

$$\textbf{f}_o^s = \text{SMamba}(\overrightarrow{\textbf{f}_s}) + \text{SMamba}(\overleftarrow{\textbf{f}_s}) + \overrightarrow{\textbf{f}_s}$$

$$\textbf{f}_o^t = \text{TMamba}(\overrightarrow{\textbf{f}_t}) + \text{TMamba}(\overleftarrow{\textbf{f}_t}) + \overrightarrow{\textbf{f}_t}$$

然后通过LayerNorm + FFN + 残差连接增强特征表示：

$$\textbf{F}_o^\star = \text{LN}(\text{LN}(\textbf{f}_o^\star) + \text{FFN}(\text{LN}(\textbf{f}_o^\star)))$$

**空间Mamba**沿姿态维度 $D$ 扫描，**时间Mamba**沿时间维度 $T$ 扫描，各自的参数化遵循标准Selective SSM架构（含离散化、输入依赖的A、B、C矩阵）。

### 损失函数 / 训练策略

**空间损失** $L_s$：约束历史和未来关节位置

$$L_s = \frac{\lambda}{J \cdot M \cdot t}\sum_{m,j,i=1}^{t}\|\hat{\textbf{P}}_{i,j}^m - \textbf{P}_{i,j}^m\|^2 + \frac{1}{J \cdot M \cdot (T-t)}\sum_{m,j,i=t+1}^{T}\|\hat{\textbf{P}}_{i,j}^m - \textbf{P}_{i,j}^m\|^2$$

**时间一致性损失** $L_t$：减轻预测运动中的时间抖动

$$L_t = \text{MSE}(\text{Conv}(\textbf{P}_{\text{pred}}), \text{Conv}(\textbf{P}_{\text{gt}}))$$

**总损失**：$L = \alpha L_s + \beta L_t$，$\alpha=1, \beta=1, \lambda=0.1$

**训练配置**：batch size=96，Adam优化器，初始学习率0.01，指数衰减（$0.1^{1/50}$/epoch），单张RTX 3090 GPU。

## 实验关键数据

### 主实验

**CMU-Mocap（UMPM）数据集 — JPE（mm）**：

| 方法 | 0.2s | 0.6s | 1.0s | 平均 |
|------|------|------|------|------|
| MRT | 36 | 115 | 193 | 114 |
| TBIFormer | 30 | 109 | 182 | 107 |
| JRFormer | 32 | 104 | 161 | 99 |
| IAFormer | 32 | 96 | 159 | 96 |
| **ST-MoE (Ours)** | **31** | **95** | **158** | **95** |

**CHI3D数据集 — JPE（mm）**：

| 方法 | 0.2s | 0.4s | 0.6s | 0.8s | 1.0s | 平均 |
|------|------|------|------|------|------|------|
| TBIFormer | 45 | 95 | 145 | 192 | 233 | 142 |
| IAFormer | 39 | 83 | 129 | 176 | 218 | 129 |
| **ST-MoE (Ours)** | 44 | **79** | **123** | **161** | **200** | **121** |

比IAFormer平均JPE降低8mm，比TBIFormer降低21mm。

**效率对比**：参数量减少41.38%，训练加速3.6×（vs IAFormer）。

### 消融实验

**异构专家有效性（CMU-Mocap UMPM）**：

| 配置 | JPE平均(↓) | APE平均(↓) | 说明 |
|------|----------|----------|------|
| Baseline（仅Encoder/Decoder） | 111.1 | 73.3 | 无专家 |
| +ST专家×4 | 104.5 | 70.7 | 仅空间-时间专家 |
| +TT专家×4 | 98.1 | 66.4 | 仅时间-时间专家 |
| +TS专家×4 | 100.1 | 68.7 | 仅时间-空间专家 |
| +SS专家×4 | 98.3 | 68.2 | 仅空间-空间专家 |
| **+All（4种各1个）** | **95.0** | **65.4** | 异构组合最优 |

**双向扫描有效性**：

| 扫描策略 | JPE平均 | APE平均 |
|----------|---------|---------|
| 仅前向 | 99.3 | 67.5 |
| 仅后向 | 98.9 | 67.0 |
| **双向** | **95.0** | **65.4** |

双向比单向分别降低4.3mm和3.9mm JPE。

### 关键发现

1. **全部激活最优**：实验表明激活所有4个专家性能最优，随着激活数增加，JPE/APE持续下降
2. **单层MoE最佳**：堆叠更多MoE层反而导致过拟合
3. **异构优于同构**：4种不同专家的组合显著优于使用任何单一类型的4个专家
4. **t-SNE可视化**证实4种专家学到了不同的特征分布，形成明显分离的聚类
5. **自适应门控权重可视化**：TT/ST专家偏向捕获近似静态的运动，SS/TS专家偏向捕获空间动态模式

## 亮点与洞察

1. **MoE+Mamba的巧妙结合**：利用MoE实现灵活的专家选择，用Mamba的线性复杂度替代注意力的二次复杂度，两个正交的改进相辅相成
2. **参数共享设计**精妙——4种专家共享同一组Mamba参数，仅改变组合顺序，实现了"少量参数、多样功能"
3. **定性分析有说服力**：通过t-SNE和门控权重可视化直观展示了不同专家捕获不同运动模式的机理（静态vs动态、空间vs时间）
4. 框架具有通用性，可扩展到其他需要时空建模的序列预测任务

## 局限与展望

1. **仅支持确定性预测**：当前方法输出单一确定性轨迹，未来需扩展到随机多人运动预测
2. **场景限制**：实验数据主要是少量人的实验室环境（2-10人），密集人群场景的表现未验证
3. **单层MoE的局限**：作者发现多层MoE会过拟合，可能需要更好的正则化策略
4. **专家数量固定为4**：专家类型的设计空间可以进一步探索（如引入跨人交互专家）

## 相关工作与启发

- **IAFormer**是直接比较对象，使用注意力机制学习时空交互信息，性能好但效率差
- **Mamba**提出选择性扫描机制，实现线性推理时间的长程依赖建模
- **MoE-Mamba**将MoE与Mamba交替堆叠，本文的方案（将Mamba嵌入专家内部）更轻量
- 将异构专家用于运动预测的思路可启发其他时空建模任务（如交通流预测、动作识别）

## 评分

- 新颖性: ⭐⭐⭐⭐ — MoE+Mamba的异构专家设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 4个数据集+丰富消融+可视化分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表设计优秀
- 价值: ⭐⭐⭐⭐ — 效率-精度权衡的标杆性工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Gaussian-Mixture Latent Flow for Stochastic 3D Human Motion Prediction](../../CVPR2026/human_understanding/gaussian-mixture_latent_flow_for_stochastic_3d_human_motion_prediction.md)
- [\[CVPR 2025\] MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](../../CVPR2025/human_understanding/moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)
- [\[AAAI 2026\] mmPred: Radar-based Human Motion Prediction in the Dark](mmpred_radar-based_human_motion_prediction_in_the_dark.md)
- [\[CVPR 2026\] MAMMA: Markerless Accurate Multi-person Motion Acquisition](../../CVPR2026/human_understanding/mamma_markerless_accurate_multi-person_motion_acquisition.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)

</div>

<!-- RELATED:END -->
