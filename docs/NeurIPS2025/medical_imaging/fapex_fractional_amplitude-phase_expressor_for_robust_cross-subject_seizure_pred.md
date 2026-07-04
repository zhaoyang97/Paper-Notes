---
title: >-
  [论文解读] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction
description: >-
  [NEURIPS2025 Spotlight][医学图像][seizure prediction] 提出 FAPEX 框架，通过可学习的分数阶神经帧算子 (FrNFO) 实现自适应时频分解，结合幅度-相位交叉编码和空间相关性聚合，在 12 个跨物种、跨模态的癫痫预测基准上全面超越 33 个基线方法。
tags:
  - "NEURIPS2025 Spotlight"
  - "医学图像"
  - "seizure prediction"
  - "EEG"
  - "fractional Fourier transform"
  - "state-space model"
  - "phase-amplitude coupling"
---

# FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction

**会议**: NEURIPS2025 Spotlight  
**arXiv**: [2511.03263](https://arxiv.org/abs/2511.03263)  
**代码**: 待确认  
**领域**: 医学图像  
**关键词**: seizure prediction, EEG, fractional Fourier transform, state-space model, phase-amplitude coupling  

## 一句话总结
提出 FAPEX 框架，通过可学习的分数阶神经帧算子 (FrNFO) 实现自适应时频分解，结合幅度-相位交叉编码和空间相关性聚合，在 12 个跨物种、跨模态的癫痫预测基准上全面超越 33 个基线方法。

## 背景与动机
- 癫痫影响全球超过 5000 万人，准确预测癫痫发作对临床干预至关重要
- 现有方法多为**受试者特异性** (subject-specific) 模型，需要为每个新患者收集大量标注数据，无法跨患者泛化，严重阻碍临床大规模部署
- **受试者无关的癫痫预测** (Subject-Agnostic Seizure Prediction, SASP) 是更具临床价值的设定，但面临三大核心挑战：
    1. 传统 CNN/Transformer 存在**频谱偏置**，倾向于低频成分，难以捕捉高频振荡 (HFO) 等关键生物标志物
    2. 癫痫发作涉及异常的**相位-幅度耦合** (phase-amplitude coupling)，但现有模型通常分别处理时域和频域的幅度信息，忽略了相位与幅度的交互
    3. 不同患者的电极排布、植入策略和脑区覆盖差异极大，导致**通道异质性**问题

## 核心问题
如何设计一个统一的、受试者无关的癫痫预测模型，能够跨物种 (人/鼠/犬/猕猴)、跨采集模态 (Scalp-EEG/SEEG/ECoG/LFP) 稳健泛化，同时有效捕捉高低频生物标志物和相位-幅度耦合关系？

## 方法详解

### 整体架构
FAPEX 由三个核心模块组成：FrNFO 骨干编码器 → 幅度-相位交叉编码 (APCE) → 空间相关性聚合 (SCA)。

### 1. 输入 Patch 化
- 多通道神经信号 $\boldsymbol{X} \in \mathbb{R}^{C \times T}$ 被切分为固定长度 $\tau$ 的非重叠 patch
- 每个 patch 经共享线性嵌入投影到 $d_{\text{model}}$ 维特征空间
- 这一设计使模型不依赖于电极数量和空间排列

### 2. 分数阶神经帧算子 (FrNFO)
- **核心创新**：将分数阶 Fourier 变换 (FrFT) 与可学习的 Weyl-Heisenberg 帧结合
- FrFT 通过分数阶参数 $\theta$ 在时域 ($\theta=0$) 和频域 ($\theta=\pi/2$) 之间连续插值，但传统 FrFT 受限于固定 chirp 函数且对形变敏感
- FrNFO 的解决方案：
    - 用隐式 MLP 生成**自适应窗函数**，基于 Hermite 多项式和正弦激活函数参数化
    - 引入**可学习的分数阶** $\boldsymbol{\theta} \in (0, \pi)^{d_{\text{model}}}$，每个特征通道独立控制时频分辨率
    - 通过分数阶卷积定理在分数域进行滤波：$\hat{\boldsymbol{X}}_{:,k} = \exp(-\pi i \omega^2 \cot\theta_k) \odot \mathcal{F}_{\theta_k}(\boldsymbol{X}_{:,k}) \odot \mathcal{F}_{\theta_k}(\boldsymbol{\Psi}_{:,k})$
- 输出自然分解为复数表示的**幅度**和**相位**两部分
- 理论保证：从散射变换角度证明了 FrNFO 的幅度表示具有可证明的鲁棒性

### 3. 幅度-相位交叉编码 (APCE)
- 采用双向状态空间模型 (Bidirectional SSM) 构建交叉注意力机制
- **Phase BSSM**：以相位嵌入为输入、幅度嵌入提供状态空间参数 ($\boldsymbol{B}, \boldsymbol{C}$)，捕捉相位对幅度的调制关系
- **Amplitude BSSM**：角色互换，幅度作为查询、相位提供上下文
- 最终通过残差连接融合，得到编码了相位-幅度耦合信息的特征

### 4. 空间相关性聚合 (SCA)
- 使用线性注意力建模全局跨电极依赖关系，复杂度为 $O(C)$（线性于通道数）
- 特征映射 $\phi$ 用单层 MLP 实现：$\phi_{\text{MLP}}(\boldsymbol{x}) = \exp(\boldsymbol{W}_1^\top \boldsymbol{x})$
- 结合 $3 \times 3$ 深度卷积门控机制聚合局部时空模式

## 实验关键数据

### 实验规模
- **12 个基准数据集**：覆盖 4 种物种（人、鼠、犬、猕猴）和 4 种采集模态（Scalp-EEG、SEEG、ECoG、LFP）
- **33 个基线方法**：23 个监督学习 + 10 个自监督学习
- 评估协议：受试者无关嵌套交叉验证 (SANCV)

### 核心结果（监督学习 FAPEX-Base）

| 数据集 | SEN | F1 | AUROC |
|--------|-----|-----|-------|
| Beirut (Scalp-EEG, 人) | 84.7 | 84.3 | 85.8 |
| Canine (ECoG, 犬) | 86.0 | 84.7 | 74.5 |
| FMCE (ECoG/SEEG, 人) | 88.8 | 90.7 | 97.2 |
| cTLE-RatLFP (LFP, 鼠) | 81.8 | 83.2 | 91.2 |
| KAIME (EEG+SEEG, 猕猴) | 87.0 | 95.6 | 90.1 |
| PCS (Scalp-EEG, 人) | 91.5 | 91.5 | 96.3 |

### 自监督预训练提升（FAPEX-Base SSL）
- 预训练后 F1 在多数数据集上进一步提升 2-10 个百分点
- 在 IESS 数据集上 F1 从 72.4 → 84.9，提升 12.5 个百分点
- 在 PCS 数据集上 F1 从 91.5 → 95.0

### 跨域迁移
- Source-Only Transfer 场景下，相比 Neuro-BERT 和 CBraMod，F1 相对提升通常 >30%
- 即使在有目标域标签的 CDAC/MME 方案下，FAPEX 仍优于或持平大多数场景

## 亮点
1. **理论性强**：FrNFO 不仅是工程上的创新，还具有散射变换视角下的可证明鲁棒性
2. **实验极其充分**：12 个数据集、4 种物种、33 个基线、多种迁移学习设定，是癫痫预测领域最大规模的对比研究之一
3. **可解释性好**：FrNFO 的频率响应可视化显示随深度增加逐步细化子频带区分，且高低频均被有效放大，克服了传统模型的低频偏置
4. **架构紧凑**：不走大模型路线，而是通过精心设计的数学结构实现高性能

## 局限与展望
1. 通道对齐预处理将所有数据统一到 64 通道，这一硬约束可能丢失原始通道布局信息
2. 部分私有数据集（AGS、ATLE、IESS 等）无法复现
3. 论文未报告推理延迟和模型参数量的详细对比，对可穿戴设备部署场景的可行性讨论不足
4. 消融实验仅在附录中简要提及，核心模块的独立贡献尚需更透明的量化

## 与相关工作的对比
- **vs Medformer / TSLANet**（多尺度时序模型）：FAPEX 在绝大多数数据集上 F1 超出 5-10 个百分点，尤其在跨物种场景优势显著
- **vs EEG 基础模型 (Neuro-BERT / CBraMod / LaBraM)**：使用相同预训练数据时 FAPEX 仍大幅领先，说明增益来源于架构设计而非预训练规模
- **vs SeizureFormer**（癫痫专用模型）：FAPEX 在 SEN 和 F1 上均有 10-20 个百分点的优势
- **vs 频域方法 (FreTS / NFM / ATFNet)**：FrNFO 的分数阶设计在非平稳信号上显著优于固定频率基变换

## 启发与关联
- FrNFO 的可学习分数阶时频分解思想可推广到其他生物信号分析任务（如心电、肌电）
- 幅度-相位交叉编码的 SSM 架构为其他需要同时建模幅度和相位的信号处理任务提供了范式
- 跨物种泛化的成功暗示癫痫发作前的神经动力学可能存在跨物种共享的通用模式

## 评分
- 新颖性: 9/10 — 分数阶帧理论与深度学习的创新结合，理论扎实
- 实验充分度: 10/10 — 12 数据集、4 物种、33 基线、多种迁移设定，极为全面
- 写作质量: 8/10 — 整体清晰，但数学符号密集，部分记号不一致
- 价值: 9/10 — 对癫痫预测领域贡献重大，临床转化潜力高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[AAAI 2026\] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals](../../AAAI2026/medical_imaging/mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)

</div>

<!-- RELATED:END -->
