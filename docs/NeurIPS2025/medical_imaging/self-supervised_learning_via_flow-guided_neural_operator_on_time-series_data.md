---
title: >-
  [论文解读] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data
description: >-
  [NeurIPS 2025][医学图像][自监督学习] 提出 FGNO（Flow-Guided Neural Operator），将 Flow Matching 与算子学习结合用于时间序列自监督预训练，通过 STFT 实现分辨率不变的函数空间学习，并将流时间（flow time）和网络层作为控制特征粒度的"旋钮"，在生物医学任务上显著优于 MAE 等基线。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "自监督学习"
  - "Flow Matching"
  - "神经算子"
  - "时间序列"
  - "生物医学信号"
---

# Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data

**会议**: NeurIPS 2025  
**arXiv**: [2602.12267](https://arxiv.org/abs/2602.12267)  
**代码**: 暂无  
**领域**: 医学图像 / 时间序列自监督学习  
**关键词**: 自监督学习, Flow Matching, 神经算子, 时间序列, 生物医学信号

## 一句话总结

提出 FGNO（Flow-Guided Neural Operator），将 Flow Matching 与算子学习结合用于时间序列自监督预训练，通过 STFT 实现分辨率不变的函数空间学习，并将流时间（flow time）和网络层作为控制特征粒度的"旋钮"，在生物医学任务上显著优于 MAE 等基线。

## 研究背景与动机

时间序列自监督学习（SSL）面临三个核心挑战：

**分辨率异质性**：真实信号采集频率各异（如可穿戴设备 4Hz-200Hz），上/下采样会扭曲信号内在特性。传统方法处理固定尺寸输入，难以泛化到不同采样率。

**多尺度需求**：不同下游任务需要不同粒度的表征——睡眠阶段分类依赖秒级局部模式，而呼吸暂停指数回归需要全夜跨度信息。现有 SSL 通常只产出单一潜在表征。

**固定预训练目标**：MAE 使用固定掩码比例重建，缺乏灵活性。扩散/流模型在图像上展示了不同噪声水平产出多尺度特征层次的能力，但在时间序列 SSL 中尚未被充分探索。

FGNO 的核心洞察是：将**腐蚀程度（噪声水平/流时间）视为表征学习的新自由度**，而非 MAE 中的固定超参。结合 STFT 的分辨率不变性和神经算子的函数空间学习能力，构建统一的多尺度 SSL 框架。

## 方法详解

### 整体框架

FGNO 分为两个阶段：
1. **自监督预训练**：在无标签数据上，用 Flow Matching 目标训练 Transformer 模型
2. **下游探测**：冻结预训练骨干，选择最优的（层 $l$，流时间 $s$）组合，训练轻量级探测头

### 关键设计

1. **STFT 数据嵌入**：将 1D 时间序列 $x \in \mathbb{R}^T$ 通过短时傅里叶变换转为时频谱图 $f \in \mathbb{C}^{N_f \times N_t}$，使用幅度谱 $\phi = |\Phi|$ 作为模型输入。STFT 的关键优势是**分辨率不变性**——不同采样率的信号可以直接变换而无需重采样，避免插值失真。这与 Fourier Neural Operator (FNO) 使用全局 FFT 不同，STFT 分析局部时间窗口内的频率信息，能捕获时变特征。

2. **Flow Matching 预训练**：在谱图空间上训练时间条件网络 $u_\theta(s, g)$，将简单先验分布（高斯噪声）映射到复杂数据分布。对于时间步 $s \sim \mathcal{U}[0,1]$，构建噪声插值 $g = s\phi + \sigma_s \epsilon$，训练目标为：

$$J(\theta) = \mathbb{E}_{s, \phi, g} \left[\|v_s^\phi(g) - u_\theta(s, g)\|^2\right]$$

其中目标速度场 $v_s^\phi(g) = \frac{(\sigma_s)'}{\sigma_s}(g - s\phi) + \phi$。骨干网络 $u_\theta$ 使用 Transformer 架构，流时间 $s$ 通过正弦位置编码作为条件输入。

3. **干净输入特征提取**：预训练后，使用**干净谱图**（而非带噪输入）提取特征——给定干净输入 $\phi$ 和指定流时间 $s$，提取第 $l$ 层激活 $z_{l,s}(\phi) = u_\theta^{(l)}(s, \phi)$。虽然训练时输入是带噪的、推理时用干净数据存在分布偏移，但实验表明轻量级探测头能有效弥合这一gap。干净输入消除了噪声引起的随机性，提供确定性且稳定的特征。

### 损失函数 / 训练策略

- 预训练：标准 Flow Matching 目标（$L_2$ 速度场回归），单一模型同时学习多尺度表征
- 探测阶段：网格搜索 $(l^*, s^*) = \arg\min_{l,s} \mathcal{L}_{\text{val}}(l, s)$
- 浅层+低 $s$（高腐蚀）→ 粗粒度全局特征；深层+高 $s$（低腐蚀）→ 精细时间细节
- 模型仅 370K 参数，远小于基线（BrainBERT 43M, PopT 20M）

## 实验关键数据

### 主实验：DREAMT 数据集（可穿戴设备数据）

| 任务 | 指标 | FGNO | MAE | Chronos |
|------|------|------|-----|---------|
| 睡眠/清醒分类 | AUROC (%) ↑ | **96.5** | 95.8 | 96.3 |
| 皮肤温度回归 | RMSE (°C) ↓ | **0.600** | 0.735 | 0.954 |

### 主实验：BrainTreeBank（神经信号解码）

| 方法 | 参数量 | Speech AUROC | Volume | Pitch |
|------|--------|-------------|--------|-------|
| FGNO | **370K** | **最优** | **最优** | **最优** |
| BrainBERT | 43M | 次优 | 次优 | 次优 |
| PopT | 20M | - | - | - |

FGNO 在 3/4 个任务上超越所有基线，参数量仅为基线的 1/50。

### 数据稀缺鲁棒性（仅 5% 标注数据）

| 方法 | SleepEDF ACC | SleepEDF MF1 | Epilepsy ACC | Epilepsy MF1 |
|------|-------------|-------------|-------------|-------------|
| FGNO (5%) | **93.5** | **89.0** | **94.1** | **90.3** |
| TS-TCC (5%) | 77.0 | 70.9 | 93.1 | 93.7 |
| Supervised (5%) | 60.5 | 54.8 | 83.4 | 80.4 |
| FGNO (100%) | 93.9 | 89.1 | 94.8 | 90.3 |

FGNO 在仅 5% 标注数据时几乎保持了全量数据的性能水平（SleepEDF: 93.5% vs 93.9%）。

### 消融实验

| 消融项 | 结果 | 说明 |
|--------|------|------|
| 干净 vs 带噪输入 | AUROC 96.40% vs 95.86% | 干净输入更优且确定性，无需噪声采样 |
| 带噪输入方差 | std=0.0039（10次运行） | 噪声引入不必要的随机性 |
| 跨分辨率泛化 (48× 降采样) | FGNO 74%+ vs MAE ~52% | 函数空间学习天然支持分辨率不变 |
| 计算效率 | 探测时间降低 60% | 370K 参数 + 冻结骨干 + 轻量探头 |

### 关键发现

- 不同任务需要不同的 $(l, s)$ 组合：分类任务偏好高 $s$（低腐蚀，保留局部模式），回归任务偏好中等 $s$（需要全局特征）
- 最优 $(l, s)$ 对在超参空间中形成连续的结构化区域，便于实际选择
- FGNO 在 48× 降采样下仍保持 74%+ AUROC，而 MAE 降至 ~52%
- 预训练时间与 MAE 相当（~21h），但下游适配时间减少 60%

## 亮点与洞察

- **流时间作为控制旋钮**：将扩散/流模型的噪声水平从"训练超参"提升为"可调特征粒度"，提供了统一框架产出多尺度表征的优雅方式
- **干净输入推理的务实选择**：理论上存在训练-推理分布偏移，但实践证明轻量探测头能弥合gap，同时带来确定性和效率的双重优势
- **STFT + 算子学习的结合**：STFT 提供分辨率不变的时频表示，算子学习在函数空间操作，两者配合天然支持多分辨率泛化
- **极致参数效率**：370K 参数的模型超越 43M 参数的基线，说明自监督目标的质量比模型大小更重要

## 局限与展望

- $(l, s)$ 选择需要验证集网格搜索，增加了下游适配的计算成本
- 主要评估在生物医学信号领域，未验证其他时间序列域（如金融、工业IoT）
- Flow Matching 预训练阶段与 MAE 相比并无显著效率优势
- 未与更新的时间序列 Foundation Model（如 TimesFM）比较
- 理论上的函数空间学习声称需要更严格的验证

## 相关工作与启发

- **MAE / BrainBERT**: 掩码自编码器在时间序列 SSL 中的应用
- **Chronos**: 时间序列基础模型，基于 T5 的自回归方法
- **FNO (Fourier Neural Operator)**: 用全局 FFT 建模，本文改用局部 STFT
- **REPA / CleanDIFT**: 图像领域中从扩散模型提取干净输入表征的先驱工作
- 启发：Flow Matching 在 SSL 中的潜力值得更多领域探索，$(l, s)$ 选择机制可推广到其他生成式 SSL 方法

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ Flow Matching + 算子学习 + STFT 的原创组合，流时间作为特征控制旋钮的思路新颖
- **实验充分度**: ⭐⭐⭐⭐ 多数据集多任务评估，丰富的消融实验
- **写作质量**: ⭐⭐⭐⭐ 图示清晰，方法动机阐述充分
- **价值**: ⭐⭐⭐⭐ 对生物医学时间序列分析有实际价值，特别在数据稀缺场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[CVPR 2025\] NOIR: Neural Operator Mapping for Implicit Representations](../../CVPR2025/medical_imaging/noir_neural_operator_mapping_for_implicit_representations.md)
- [\[CVPR 2025\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](../../CVPR2025/medical_imaging/addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)

</div>

<!-- RELATED:END -->
