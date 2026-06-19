---
title: >-
  [论文解读] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception
description: >-
  [AAAI 2026][AI安全][深伪音频检测] 首次建立全类型（语音/声音/歌声/音乐）音频深伪检测基准，提出小波提示调优（WPT）方法通过离散小波变换增强 SSL 特征的全频域感知能力，在不增加训练参数的前提下超越全量微调，co-training 后平均 EER 仅 3.58%。 音频生成技术的快速发展使得合成任何类型…
tags:
  - "AAAI 2026"
  - "AI安全"
  - "深伪音频检测"
  - "小波提示学习"
  - "自监督学习"
  - "跨类型检测"
  - "频域分析"
---

# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception

**会议**: AAAI 2026  
**arXiv**: [2504.06753](https://arxiv.org/abs/2504.06753)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 深伪音频检测, 小波提示学习, 自监督学习, 跨类型检测, 频域分析

## 一句话总结

首次建立全类型（语音/声音/歌声/音乐）音频深伪检测基准，提出小波提示调优（WPT）方法通过离散小波变换增强 SSL 特征的全频域感知能力，在不增加训练参数的前提下超越全量微调，co-training 后平均 EER 仅 3.58%。

## 研究背景与动机

音频生成技术的快速发展使得合成任何类型的音频变得容易——包括伪造语音、声音、歌声和音乐，对媒体、网络安全和政治传播构成威胁。

**现有问题**：

1. 当前检测模型在单一类型检测上表现优异，但**跨类型泛化能力极差**——语音训练的模型在声音/音乐上接近随机
2. 没有人系统研究过**全类型 ADD 任务**，即同时检测语音、声音、歌声、音乐四类深伪音频
3. 基于 SSL 的微调方法虽效果好，但**参数量巨大、超参敏感**
4. 人类感知不同类型音频的主要差异在于**频域分布**，但主流 SSL 模型专为语音识别设计，缺乏全频域信息捕获能力

**关键动机**：不同类型音频的深伪特征应存在某种**类型不变的频域特征**——如果能捕获这种特征，就能实现全类型检测。

## 方法详解

### 整体框架

整体架构为 **SSL-AASIST**：SSL 前端提取特征 + AASIST 后端分类。提出两种高效训练范式：

1. **PT-SSL-AASIST（Prompt Tuning）**：在每个 Transformer 层前插入可学习的 Prompt Token，冻结 SSL 其他参数
2. **WPT-SSL-AASIST（Wavelet Prompt Tuning）**：对部分 Prompt Token 进行离散小波变换（DWT），获得不同频带的 Token，增强全频域感知

### 关键设计

**PT-SSL-AASIST 设计**：

- 对输入音频 X 补零/截断到固定长度 L，通过冻结的 SSL 前端 CNN 特征提取器得到初始嵌入 E_0
- 为每个 Transformer 层初始化可学习 Prompt Token P_k（Xavier 均匀初始化）
- 每层计算：[Z_i, E_i] = L_i([P_i, E_{i-1}])
- 上一层的 Prompt 输出 Z_i 被丢弃并替换为新的 P_i
- 最终输出 I = [Z_24, E_24] 送入 AASIST 后端进行时频图注意力分类

**WPT-SSL-AASIST 设计**：

- 将部分 Prompt Token 替换为小波 Prompt Token，使用 Haar 小波进行 DWT 变换
- Haar 小波低通滤波器 L = [1,1]/sqrt(2)，高通滤波器 H = [1,-1]/sqrt(2)
- 对初始小波 Token T_k 进行 DWT，得到四个子带：
    - **LL**（低频分量）、**LH**（垂直高频）、**HL**（水平高频）、**HH**（对角高频）
    - 每个分量 w/2 x d/2，重塑为 w/4 x d
    - 拼接四个分量形成小波 Prompt W_k
- 每层计算变为：[Z_i, E_i] = L_i([W_i, P_i, E_{i-1}])
- 关键发现：每个 Token 对应一个特定频率分量，WPT=4 时自然对齐四个频带

**参数效率**：PT/WPT 只学习 Prompt Token（约 0.69M 参数），比全量微调 FT（315.89M）减少 **458 倍**。

### 损失函数 / 训练策略

- 使用**加权交叉熵（WCE）损失**训练二分类（真/伪）
- 所有音频降采样到 16kHz，截断/补零到 64600 样本（约 4s）
- FT 学习率 1e-6，batch size 14；FR/PT/WPT 学习率 5e-4，batch size 32
- 单类型训练 50 epochs（每 10 步 LR 减半）；联合训练 20 epochs（每 4 步 LR 减半）
- SSL 特征维度 (201, 1024)

## 实验关键数据

### 主实验

**数据集规模**：

| 类型 | 数据来源 | 训练集 | 开发集 | 评估集 |
|---|---|---|---|---|
| 语音 | 19LA | 25,380 | 24,844 | 71,237 |
| 声音 | Codecfake-A3 | 69,378 | 9,911 | 19,823 |
| 歌声 | CtrSVDD | 84,404 | 43,625 | 92,769 |
| 音乐 | FakeMusicCaps | 20,861 | 6,058 | 6,122 |
| 全部 | 联合 | 199,023 | 84,438 | 189,951 |

**联合训练结果（EER%）**：

| 检测模型 | 语音 | 声音 | 歌声 | 音乐 | 平均 |
|---|---|---|---|---|---|
| Spec-Resnet | 29.37 | 23.37 | 37.17 | 42.75 | 33.17 |
| AASIST | 3.78 | 0.86 | 20.01 | 11.70 | 9.09 |
| FR-XLSR-AASIST | 3.02 | 5.45 | 10.86 | 22.67 | 10.50 |
| FT-XLSR-AASIST | 1.77 | 0.49 | 8.93 | 8.71 | 4.98 |
| PT-XLSR-AASIST | 2.00 | 1.11 | 14.54 | 9.29 | 6.74 |
| **WPT-XLSR-AASIST** | **0.72** | **1.29** | **7.47** | **4.83** | **3.58** |

WPT 以仅 0.69M 参数（FT 的 1/458）实现 3.58% 平均 EER，超越 315.89M 参数的 FT（4.98%）。

**单类型训练跨类型泛化**：

| 训练类型 | 最佳模型 | 域内 EER | 平均 EER | 跨类型发现 |
|---|---|---|---|---|
| 语音 | FR-XLSR-AASIST | 1.28% | 32.58% | 语音到歌声泛化较好 |
| 声音 | AASIST | 0.43% | 22.71% | 声音到音乐有关联 |
| 歌声 | FR-XLSR-AASIST | 9.45% | 23.16% | 歌声到语音泛化好 |
| 音乐 | FR-MERT-AASIST | 7.62% | 28.63% | MERT 对音乐最强 |

### 消融实验

**Prompt Token 数量消融**（语音训练 XLSR-AASIST）：

| Token 数 | 参数量 | 语音 EER | 平均 EER |
|---|---|---|---|
| 2 | 0.50M | 0.75% | 30.94% |
| 10 | 0.69M | **0.22%** | **30.79%** |
| 100 | 2.90M | 3.01% | 31.28% |
| 200 | 5.36M | 4.99% | 33.36% |

过多 Prompt Token 稀释了音频 Token 的信息密度，10 个为最优。

**训练范式对比**（语音训练 XLSR-AASIST）：

| 模型 | 参数量 | 语音 EER | 平均 EER |
|---|---|---|---|
| FR-XLSR-AASIST | 0.45M | 1.28% | 32.58% |
| FT-XLSR-AASIST | 315.89M | 0.38% | 27.68% |
| PT-XLSR-AASIST | 0.69M | 0.22% | 30.79% |
| WPT-XLSR-AASIST | 0.69M | 0.15% | 26.86% |

### 关键发现

- **语音和歌声之间存在共享特征**，跨类型泛化最好；声音和音乐之间也有关联
- WPT 学到了**类型不变的深伪检测 Prompt**：t-SNE 可视化中 WPT 的真/伪样本不按类型聚类
- 注意力图分析：WPT 聚焦于第 4 个 Token（对应 **HH 频带——对角高频**），该频带在所有音频类型上一致
- FR/PT/WPT 收敛速度显著快于 FT，且波动更小

## 亮点与洞察

1. **首创全类型 ADD 基准**：涵盖语音、声音、歌声、音乐四类深伪检测，填补领域空白
2. **小波 Prompt 设计精巧**：将 DWT 嵌入 Prompt Token 初始化中，不增加训练参数就获得全频域感知
3. **458 倍参数效率提升**：WPT 仅需 0.69M 参数，超越 315.89M 的 FT 方法
4. **HH 频带的类型不变性发现**：注意力图清晰显示 WPT 聚焦于 HH Token，为频域深伪检测提供新认识
5. 实验设计系统全面：单类型到跨类型到联合训练的递进分析非常清晰

## 局限与展望

1. 基准数据集均为相对干净的环境，未考虑噪声、部分伪造等更复杂真实场景
2. 所有音频截断到约 4 秒，对长音频的适用性未验证
3. WPT 仅使用 Haar 小波（最简单的小波变换），更复杂的小波族可能效果更好
4. 音乐类型检测 EER 仍显著高于语音，跨类型统一检测仍有提升空间
5. 未探讨对抗攻击场景下的检测鲁棒性

## 相关工作与启发

- **XLSR-AASIST**（Tak et al. 2022）：SSL+AASIST 的经典框架，本文基线
- **Visual Prompt Tuning**（Jia et al. 2022）：视觉领域的 Prompt Tuning 方法，启发了 PT-SSL
- **CtrSVDD**（Zhang et al. 2024）：首个歌声深伪检测挑战赛
- **FakeMusicCaps**（Comanducci et al. 2024）：首个音乐深伪检测数据集
- 启发：频域信息是跨模态/跨类型检测的关键桥梁，DWT 子带分解提供了自然且高效的频域特征注入方式

## 评分

- 新颖性: 5/5 - 全类型 ADD 基准 + WPT 方法都是首创
- 技术深度: 4/5 - 小波变换与 Prompt Tuning 的结合设计精巧
- 实验充分度: 5/5 - 单类型/跨类型/联合训练 + 多种消融 + 可视化分析
- 写作质量: 4/5 - 结构清晰，图表丰富
- 综合: 4.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)
- [\[CVPR 2026\] X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection](../../CVPR2026/ai_safety/x-avdt_audio-visual_cross-attention_for_robust_deepfake_detection.md)
- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](../../ICML2026/ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ICCV 2025\] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos](../../ICCV2025/ai_safety/fakeradar_probing_forgery_outliers_to_detect_unknown_deepfake_videos.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)

</div>

<!-- RELATED:END -->
