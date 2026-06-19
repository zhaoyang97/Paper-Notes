---
title: >-
  [论文解读] MOSPA: Human Motion Generation Driven by Spatial Audio
description: >-
  [NeurIPS 2025 Spotlight][人体理解][空间音频] 首次定义"空间音频驱动人体运动生成"这一新任务，构建包含 9+ 小时、27 种场景、12 名受试者的双耳音频-运动配对 SAM 数据集，提出 MOSPA 扩散模型在融合 MFCC/tempogram/RMS 等音频特征与声源位置及运动风格条件后，以 FID 7.98 大幅领先 EDGE（14.0）、POPDG（21.0）等音乐/舞蹈基线。
tags:
  - "NeurIPS 2025 Spotlight"
  - "人体理解"
  - "空间音频"
  - "运动生成"
  - "扩散模型"
  - "SAM 数据集"
  - "双耳音频"
---

# MOSPA: Human Motion Generation Driven by Spatial Audio

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2507.11949](https://arxiv.org/abs/2507.11949)  
**代码**: 有（项目页公开）  
**领域**: 人体运动生成 / 空间音频  
**关键词**: 空间音频, 运动生成, 扩散模型, SAM 数据集, 双耳音频

## 一句话总结

首次定义"空间音频驱动人体运动生成"这一新任务，构建包含 9+ 小时、27 种场景、12 名受试者的双耳音频-运动配对 SAM 数据集，提出 MOSPA 扩散模型在融合 MFCC/tempogram/RMS 等音频特征与声源位置及运动风格条件后，以 FID 7.98 大幅领先 EDGE（14.0）、POPDG（21.0）等音乐/舞蹈基线。

## 研究背景与动机

**领域现状**：条件运动生成已广泛探索文本→运动（MDM、MoMask）、音乐→舞蹈（EDGE、Bailando）、语音→手势（GestureDiffuCLIP）等任务。这些方法从音频中提取语义和节奏信息来驱动运动，但都忽略了声音的空间属性——方向、距离和强度。

**现有痛点**：人类对声音的反应天然受空间感知驱动——听到左侧尖锐声响会捂耳向右闪，听到前方轻柔音乐可能走近。现有音乐/语音→运动方法仅编码时域语义特征，无法建模这种空间依赖性。更关键的是，缺少专门的空间音频-运动配对数据集来训练和评估这类模型。

**核心矛盾**：空间音频不仅编码语义（什么声音）还编码空间特性（从哪里来、多大声），后者对人体运动有显著影响，但这种空间信息在现有的音频特征提取和运动生成流程中完全缺失。此外，不同个体对同一声音的反应强度差异很大（有人迟钝、有人敏感），这种多样性也需要建模。

**本文目标** (1) 构建首个空间音频-运动配对数据集；(2) 设计能同时利用空间音频的语义特征和空间特征来生成合理人体运动的生成模型。

**切入角度**：采用双耳音频（binaural audio）——最接近人类听觉的空间音频形式，天然编码了声源方向和距离信息（左右耳信号差异）。通过 MFCC 捕获时域语义、RMS 能量编码距离信息、运动风格标签（迟钝/中性/敏感）控制反应强度。

**核心 idea**：用包含声源位置的双耳音频特征代替普通音频特征作为扩散模型的条件，实现空间感知的人体运动生成。

## 方法详解

### 整体框架

MOSPA 基于扩散模型（DDPM），输入为双耳空间音频特征 $\mathbf{a}$、声源位置 $\mathbf{s}$ 和运动风格 $g$，输出为 $T=240$ 帧（30fps，8 秒）的 SMPL-X 人体运动序列。去噪器 $\mathcal{G}$ 直接预测干净运动 $\hat{\mathbf{x}_0} = \mathcal{G}(\mathbf{x}_t, t; \mathbf{a}, \mathbf{s}, g)$（而非预测噪声），使用 encoder-only Transformer 架构实现。

### 关键设计

1. **双耳空间音频特征提取**:

    - 功能：从双耳音频中提取同时编码语义、时域和空间信息的特征向量
    - 核心思路：为左右耳分别提取 1136 维特征，包括：MFCC 及其 delta（捕获频谱包络和动态变化，40 维）、constant-Q 和 STFT chromagram（音调信息，24 维）、onset strength 和 tempogram（节奏和节拍，1069 维）、RMS 能量 $E_{\text{rms}}$（信号强度→间接编码距离，1 维）、活跃帧标记 $F_{\text{active}} = E_{\text{rms}} > 0.01$（1 维）。左右耳拼接得到 $\mathbf{a} \in \mathbb{R}^{T \times 2272}$。RMS 的关键作用是：声源越近，RMS 越大，且左右耳 RMS 差异编码了方向信息
    - 设计动机：传统音乐→舞蹈方法主要用 MFCC 和节拍。本文额外引入 RMS 能量来捕获距离信息，并通过双耳拼接保留空间差异——这是建模"声音从哪里来"的关键

2. **运动表示与残差特征融合**:

    - 功能：用位置+旋转+速度的冗余表示增强运动生成精度
    - 核心思路：每个运动向量 $\mathbf{x}$ 包含 $J=25$ 个 SMPL-X 关节的全局位置 $\mathbf{p} \in \mathbb{R}^{T \times 75}$、6D 格式的局部旋转 $\mathbf{r} \in \mathbb{R}^{T \times 150}$ 和关节速度 $\mathbf{v} \in \mathbb{R}^{T \times 75}$，每帧 300 维。速度作为残差信息帮助模型捕获空间音频对运动的细微影响，如反应速度的变化
    - 设计动机：仅用旋转表示虽然紧凑但损失了全局位移信息（对空间音频响应至关重要——需要知道角色朝哪个方向移动）。冗余的位置+速度特征为模型提供了更直接的监督信号

3. **条件融合与 Transformer 去噪器**:

    - 功能：将多种条件信号融合后引导扩散去噪过程
    - 核心思路：时间步 $t$、运动向量 $\mathbf{x}_t$、音频特征 $\mathbf{a}$、声源位置 $\mathbf{s}$ 和运动风格 $g$ 分别通过独立的 FFN 投影到相同潜在维度（512），对 $\mathbf{a}$ 和 $\mathbf{s}$ 施加随机 mask（类似 classifier-free guidance），然后拼接为完整 token 序列 $\mathbf{z}$，加位置编码后送入 encoder transformer（4 层、8 头、512 维）。最后 $T$ 个 token 通过 FFN 输出预测的干净运动 $\hat{\mathbf{x}_0}$
    - 设计动机：随机 mask 条件使推理时可通过调节 guidance 强度控制生成；encoder-only transformer 的自注意力机制让音频的不同时步可以与运动的不同时步自由交互

### 损失函数 / 训练策略

总损失由五项组成：$\mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{geo}}\mathcal{L}_{\text{geo}} + \lambda_{\text{foot}}\mathcal{L}_{\text{foot}} + \lambda_{\text{traj}}\mathcal{L}_{\text{traj}} + \lambda_{\text{rot}}\mathcal{L}_{\text{rot}}$

- $\mathcal{L}_{\text{data}}$：预测与真实运动的 MSE + 帧间变化率 MSE（时序平滑性）
- $\mathcal{L}_{\text{geo}}$：FK 正运动学后的关节位置 MSE + 速度 MSE（物理一致性）
- $\mathcal{L}_{\text{foot}}$：脚部接触一致性损失（防止脚滑）
- $\mathcal{L}_{\text{traj}}$ 和 $\mathcal{L}_{\text{rot}}$：轨迹和旋转的强调损失（加速收敛和方向准确性）

训练策略：所有 $\lambda$ 初始为 1，在第 5000/6000 epoch 时将 $\lambda_{\text{traj}}$ 和 $\lambda_{\text{rot}}$ 增至 3 以强调末期的轨迹和旋转精度。1000 步余弦噪声调度，AdamW 优化器 lr=$10^{-4}$，batch size 128，单卡 RTX 4090 训练约 18 小时。

## 实验关键数据

### 主实验

SAM 数据集上的定量评估（8:1:1 划分，2400/300/300 序列）：

| 方法 | R-prec Top1↑ | R-prec Top3↑ | FID↓ | Diversity→ | APD→ |
|------|-------------|-------------|------|-----------|------|
| Real Motion | 1.000 | 1.000 | 0.001 | 23.62 | 59.44 |
| Bailando | 0.077 | 0.182 | 168.4 | 17.35 | 23.12 |
| LODGE | 0.444 | 0.679 | 102.3 | 21.10 | 11.80 |
| POPDG | 0.762 | 0.934 | 21.0 | 22.54 | 35.00 |
| EDGE | 0.886 | 0.977 | 14.0 | 23.10 | 43.88 |
| **MOSPA** | **0.937** | **0.996** | **7.98** | **23.58** | **53.92** |

### 消融实验

| 配置变更 | FID | R-prec Top1 | 说明 |
|---------|-----|-------------|------|
| 完整模型（512d, 8h, 1000步, +genre） | **7.98** | **0.937** | 最优配置 |
| 256d | 9.23 | 0.891 | 低维度降低质量 |
| 4 头 | 9.28 | 0.923 | 少头数轻微影响 |
| 100 步扩散 | 8.46 | 0.930 | 减少步数轻微降质 |
| 4 步扩散 | 8.39 | 0.934 | 4步仍有竞争力 |
| 去除 genre 条件 | 10.93 | 0.889 | genre 对反应强度建模很重要 |
| 去除 MFCC | 9.07 | 0.907 | MFCC 对语义建模关键 |
| 去除 tempogram | 10.79 | 0.917 | tempogram 对质量影响最大 |

### 关键发现

- **MOSPA 全面碾压所有基线**：FID 7.98 不到 EDGE（14.0）的一半，R-precision Top1 高出 5.1 个百分点。Diversity 和 APD 最接近真实运动
- **Bailando 和 LODGE 惨败**：FID 分别为 168 和 102，因为它们专为音乐/长序列设计，无法处理突然变化的空间音频
- **genre 条件很重要**：去除后 FID 恶化 37%（10.93 vs 7.98），说明反应强度建模是空间音频运动生成的关键
- **tempogram 贡献最大**：去除后 FID 恶化 35%（10.79 vs 7.98），说明节拍/节奏信息对运动时序至关重要
- **扩散步数可大幅减少**：4 步扩散 FID 8.39 仍有竞争力，为实时应用提供了可能
- 用户研究（25 人）：MOSPA 在意图对齐、运动质量和 GT 相似度三项上均获得最多投票

## 亮点与洞察

- **开创性地定义了一个新任务及其配套数据集**：空间音频→运动这个方向此前完全未被探索。27 种场景、3 种反应强度、16 种声源位置、12 名受试者的 SAM 数据集设计考虑周全，为后续研究提供了 benchmark。数据采集方案（Vicon 动捕 + 双麦克风同步录音 + Deity PR-2 记录器）可复制
- **RMS 能量作为距离代理的巧妙设计**：在所有音频特征中，RMS 能量和活跃帧标记是最简单的两个，但正是它们编码了空间音频区别于普通音频的关键信息——距离和响度。双耳的 RMS 差异则隐式编码了方向
- **运动风格的三级分类自然且有效**：将人对声音的反应分为迟钝/中性/敏感三档是直觉合理的建模选择，genre 消融证实了其重要性，同时为生成多样化运动提供了自然的控制接口
- **从 VR/游戏角度看本文高度实用**：生成的虚拟角色能根据空间声音做出方向感知的反应（如听到枪声向反方向逃跑），这是沉浸式体验的基础能力

## 局限与展望

- **缺乏物理约束**：生成的运动可能包含物理不合理的动作（如穿透地面、不平衡的姿态），未集成物理仿真引擎（如 Isaac Gym）进行约束
- **仅建模身体运动**：排除了 SMPL-X 支持的手指和面部运动，而这些在对声音的精细反应中很重要（如捂耳时的手指细节、惊恐的面部表情）
- **不考虑场景环境**：运动生成不感知周围物体和场景几何，无法生成与环境交互的运动（如听到声音后躲到桌子后面）
- **数据集规模有限**：9 小时的运动数据对深度生成模型来说偏少，可能限制了泛化能力。扩展到更多场景、更多受试者将提升鲁棒性
- **评估指标的局限**：R-precision 和 FID 依赖训练好的双向 GRU 特征提取器，这些指标可能无法捕捉空间正确性（如角色是否朝正确方向移动）

## 相关工作与启发

- **vs EDGE / Bailando（音乐→舞蹈）**：这些方法仅利用节拍和旋律信息，无法感知声源方向和距离。适配到空间音频后效果很差，说明空间特征需要专门建模
- **vs GestureDiffuCLIP（语音→手势）**：语音→手势关注语义对应（说话时的手势），而空间音频→运动关注全身的空间反应（声音从哪来就往哪个方向反应），是不同层面的问题
- **vs 之前的空间音频研究**：Sound-of-Pixels、Self-supervised Moving Vehicle Tracking 等工作从视频/音频学空间信息，但都是感知任务而非生成任务。本文首次将空间音频用于运动生成
- 启发：可结合场景感知和物理仿真做更真实的空间音频响应运动生成，也可扩展到机器人领域——让机器人根据听到的声音做出空间感知的反应

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次定义空间音频驱动运动生成任务并构建专用数据集和方法，开拓了全新方向
- 实验充分度: ⭐⭐⭐⭐ 与 4 个基线对比、多维消融、用户研究、OOD 测试，比较全面；但缺乏空间正确性专门评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰，数据集构建描述详细，方法图示直观，附录信息丰富
- 价值: ⭐⭐⭐⭐⭐ 数据集和任务定义的贡献大于方法本身，为 VR/机器人/游戏等应用开辟了新的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Audio-Driven Talking Face Generation with Stabilized Synchronization Loss](../../ECCV2024/human_understanding/audio-driven_talking_face_generation_with_stabilized_synchronization_loss.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](../../CVPR2025/human_understanding/keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[CVPR 2025\] MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](../../CVPR2025/human_understanding/moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)
- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](../../ICCV2025/human_understanding/genmo_a_generalist_model_for_human_motion.md)

</div>

<!-- RELATED:END -->
