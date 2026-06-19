---
title: >-
  [论文解读] ScanTalk: 3D Talking Heads from Unregistered Scans
description: >-
  [ECCV 2024][人体理解][3D Talking Heads] 提出 ScanTalk，首个能够对任意拓扑：（包括未配准的3D扫描数据）的3D人脸进行语音驱动动画生成的深度学习框架，核心依赖于 DiffusionNet 的离散化无关特性来突破固定拓扑约束。 语音驱动的3D说话人头部生成（3D Talking Head…
tags:
  - "ECCV 2024"
  - "人体理解"
  - "3D Talking Heads"
  - "扩散模型"
  - "语音驱动动画"
  - "拓扑无关"
  - "3D 人脸动画"
---

# ScanTalk: 3D Talking Heads from Unregistered Scans

**会议**: ECCV 2024  
**arXiv**: [2403.10942](https://arxiv.org/abs/2403.10942)  
**代码**: [https://github.com/miccunifi/ScanTalk](https://github.com/miccunifi/ScanTalk)  
**领域**: 人体理解  
**关键词**: 3D Talking Heads, DiffusionNet, 语音驱动动画, 拓扑无关, 3D 人脸动画

## 一句话总结

提出 ScanTalk，首个能够对**任意拓扑**（包括未配准的3D扫描数据）的3D人脸进行语音驱动动画生成的深度学习框架，核心依赖于 DiffusionNet 的离散化无关特性来突破固定拓扑约束。

## 研究背景与动机

语音驱动的3D说话人头部生成（3D Talking Heads）是计算机视觉和图形学中的重要研究方向，广泛应用于虚拟现实、游戏等领域。然而，现有方法（VOCA、FaceFormer、CodeTalker、SelfTalk、FaceDiffuser 等）均受限于**固定拓扑**约束——模型只能对与训练时相同顶点数量和连接关系的网格进行动画化。

这一限制导致：
1. 新获取的3D数据必须先配准到特定拓扑才能使用，增加了预处理成本
2. 无法直接对原始3D扫描数据进行动画化
3. 单一模型无法跨不同拓扑的数据集进行训练
4. 阻碍了实时在线应用（配准步骤耗时）

ScanTalk 旨在彻底解决拓扑依赖问题，使任意3D人脸（包括原始扫描）都可以被语音驱动动画化。

## 方法详解

### 整体框架

ScanTalk 采用 Encoder-Decoder 架构，输入为中性人脸网格 $m_i^n$ 和音频片段 $A_i$，输出逐顶点形变场序列。核心公式：

$$\text{ScanTalk}(A_i, m_i^n) \approx M_i^{gt}$$

框架包含两个编码器（人脸网格编码器 + 音频编码器）和一个 DiffusionNet 解码器。

### 关键设计

1. **DiffusionNet 人脸编码器**：采用 DiffusionNet 作为3D人脸编码器，它通过预计算的表面算子（余切拉普拉斯算子、特征基、质量矩阵、空间梯度矩阵）来计算内在描述子，天然支持不同拓扑的网格。编码过程为：

$$f_i^n = DN_e(m_i^n, P_i^n) \in \mathbb{R}^{V_i \times h}$$

其中 $P_i^n = OP(m_i^n)$ 是预计算的表面特征。DiffusionNet 集成了 MLP、学习的扩散和空间梯度特征，无需显式表面卷积或池化层次结构，从而实现对任意拓扑的适应。

2. **HuBERT 音频编码器 + BiLSTM**：使用预训练的 HuBERT 模型提取语音表示，再通过多层双向 LSTM 增强时间一致性：

$$a_i = \text{SpeechEncoder}(A_i) \in \mathbb{R}^{T_i \times (h/2)}$$
$$v_i = \text{BiLSTM}(a_i) \in \mathbb{R}^{T_i \times h}$$

3. **DiffusionNet 解码器**：将逐顶点描述子 $f_i^n$ 与音频时间特征 $v_i$ 拼接，形成联合表示 $F_i^j \in \mathbb{R}^{V_i \times 2h}$，再通过 DiffusionNet 解码器预测形变场：

$$(F_i^j)_k = (f_i^n)_k \oplus v_i^j$$
$$\hat{m_i}^j = DN_d(F_i^j, P_i^n) + m_i^n$$

模型预测的是中性人脸的**形变**而非完整人脸，降低了学习难度。

4. **多数据集联合训练**：由于拓扑无关性，ScanTalk 可以同时在不同拓扑的数据集上训练（VOCAset、BIWI6、Multiface），这在之前的方法中是不可能的。

### 损失函数 / 训练策略

训练使用逐顶点的均方误差（MSE）损失：

$$\mathcal{L}_{MSE} = \frac{1}{T_i} \sum_{j=0}^{T_i-1} \frac{1}{V_i} \sum_{k=0}^{V_i-1} \|(m_i^j)_k - (\hat{m_i}^j)_k\|_2^2$$

消融实验表明，在多数据集训练场景下，简单的 $L_2$ 损失优于添加 Mask Loss 或 Velocity Loss 的组合，这归因于不同数据集之间显著的几何差异。

训练细节：DiffusionNet 编码/解码器各4个块，隐藏维度 $h=32$，BiLSTM 3层，Adam 优化器，学习率 $10^{-4}$，200 epochs。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ScanTalk (s-d) | ScanTalk (m-d) | CodeTalker | FaceDiffuser | SelfTalk |
|--------|------|---------------|----------------|------------|-------------|----------|
| VOCAset | LVE↓ | **3.012** | 6.375 | 3.549 | 4.350 | 5.618 |
| VOCAset | MVE↓ | **0.861** | 0.987 | 0.888 | 0.901 | 0.918 |
| BIWI6 | LVE↓ | 4.651 | **4.044** | 5.190 | 4.022 | 3.628 |
| BIWI6 | MVE↓ | 2.148 | **2.057** | 2.641 | 2.128 | 2.062 |
| Multiface | LVE↓ | 2.653 | **2.435** | 4.091 | 3.555 | 2.281 |
| Multiface | MVE↓ | 1.871 | **1.678** | 2.382 | 2.388 | 1.901 |

### 消融实验

| 配置 | LVE↓ | MVE↓ | FDD↓ | 说明 |
|------|------|------|------|------|
| ScanTalk (HuBERT) | **3.012** | 0.861 | 2.400 | 最佳音频编码器 |
| w/ Wav2Vec2 | 3.309 | **0.860** | 2.244 | 次优 |
| w/ WavLM | 3.674 | 0.937 | 2.413 | 最差 |
| w/ BiLSTM | **3.012** | 0.861 | 2.400 | 最佳时序建模 |
| w/ BiGRU | 3.036 | **0.835** | 2.358 | 接近 |
| w/ TD (Transformer) | 3.291 | 0.859 | 2.406 | 弱于 RNN |
| w/o temporal | 3.361 | 0.870 | 2.365 | 无时序模块 |

### 关键发现

- **多数据集 vs 单数据集**：在 VOCAset（仅唇部运动）上单数据集训练更优；在 BIWI6/Multiface（含头部运动）上多数据集训练更优——因为多数据集模型见过更多头部运动样本
- **用户研究**：ScanTalk 在与 FaceFormer（82.67%）和 FaceDiffuser（90.67%）的对比中被更多用户偏好，但略逊于 SelfTalk（44%）和 GT（44%）
- **GPU 内存与顶点数**呈线性关系，可扩展性强

## 亮点与洞察

1. **首次实现拓扑无关的3D说话人头部生成**，打破了该领域长期存在的固定拓扑限制
2. **DiffusionNet 的巧妙应用**：利用其离散化无关特性从静态几何分析扩展到多模态4D动态设置
3. **多数据集联合训练能力**：单一模型可在多个不同拓扑的数据集上训练，这是其他方法无法实现的
4. 预测形变而非完整人脸的设计简化了学习目标

## 局限与展望

1. 没有嘴部开口的网格难以生成张嘴动画（尽管唇部同步仍然准确）
2. FDD 指标在某些配置下不是最优，上面部表情生成仍有改进空间
3. 模型相对简单（隐藏维度仅32），可探索更大容量的架构
4. 仅使用 MSE 损失，未利用感知损失或对抗训练
5. 推理前需要将输入网格与训练数据进行刚性对齐

## 相关工作与启发

- **DiffusionNet** [Sharp et al.]：离散化无关的表面学习架构，是本文方法的基石
- **Neural Face Rigging (NFR)**：最接近的工作，使用 DiffusionNet + Neural Jacobian Fields 进行动画迁移，但不支持语音驱动
- **HuBERT**：自监督语音表示学习，在语音-人脸跨模态映射中优于 Wav2Vec2
- 启发：DiffusionNet 的拓扑无关特性可以推广到其他3D任务（如表情迁移、手势生成）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次在3D说话人头部生成中实现拓扑无关，问题定义有突破性
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多种消融、用户研究，但缺少与 NFR 的直接对比
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，图表丰富
- **价值**: ⭐⭐⭐⭐ — 实用价值高，直接解决了领域痛点，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Talking Together: Synthesizing Co-Located 3D Conversations from Audio](../../CVPR2026/human_understanding/talking_together_synthesizing_co-located_3d_conversations_from_audio.md)
- [\[ECCV 2024\] Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)
- [\[ECCV 2024\] Diffusion Model is a Good Pose Estimator from 3D RF-Vision](diffusion_model_is_a_good_pose_estimator_from_3d_rf-vision.md)
- [\[ECCV 2024\] EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)
- [\[ECCV 2024\] Audio-Driven Talking Face Generation with Stabilized Synchronization Loss](audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)

</div>

<!-- RELATED:END -->
