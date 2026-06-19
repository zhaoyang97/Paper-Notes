---
title: >-
  [论文解读] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation
description: >-
  [CVPR 2025][视频生成][说话人动画] 提出首个基于自回归 Transformer 的实时流式音频驱动肖像动画框架 Teller，通过 RVQ 将面部运动离散化为 token，结合高效时序模块精炼身体细节，以 25 FPS 实时速度（生成 1s 视频仅需 0.92s vs Hallo 20.93s）达到与扩散模型可比的动画质量。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "说话人动画"
  - "自回归生成"
  - "实时流式"
  - "动作离散化"
  - "时序精炼"
---

# Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation

**会议**: CVPR 2025  
**arXiv**: [2503.18429](https://arxiv.org/abs/2503.18429)  
**代码**: [https://teller-avatar.github.io/](https://teller-avatar.github.io/)  
**领域**: 视频生成  
**关键词**: 说话人动画, 自回归生成, 实时流式, 动作离散化, 时序精炼

## 一句话总结
提出首个基于自回归 Transformer 的实时流式音频驱动肖像动画框架 Teller，通过 RVQ 将面部运动离散化为 token，结合高效时序模块精炼身体细节，以 25 FPS 实时速度（生成 1s 视频仅需 0.92s vs Hallo 20.93s）达到与扩散模型可比的动画质量。

## 研究背景与动机
1. **领域现状**：音频驱动的肖像动画（talking head）近年取得显著进展，扩散模型方法（Hallo、EMO、LOOPY）能生成高质量动画，但推理速度极慢（~20s/s）完全无法满足实时需求。
2. **现有痛点**：(a) 扩散模型需多步迭代推理，生成单帧就需多次前向传播；(b) 基于 GAN 的方法（SadTalker、LivePortrait）虽快但运动表现力不足，特别是身体配饰（耳环、项链）和颈部肌肉的自然运动被忽视。
3. **核心矛盾**：高质量动画需要捕获丰富的面部和身体运动细节，但计算预算有限（实时需 25+ FPS），扩散模型的质量与自回归模型的速度如何兼得？
4. **本文目标**：设计首个实时流式的高质量音频驱动肖像动画框架。
5. **切入角度**：将面部运动 latent 离散化为 token 序列，利用自回归 Transformer 的高效 next-token prediction 能力实现音频到运动的实时映射。
6. **核心 idea**：两阶段框架——FMLG（RVQ + AR Transformer）生成面部运动 token → ETM（高效时序模块）精炼身体细节。

## 方法详解

### 整体框架
Teller 分两阶段：**Stage 1 (FMLG)**：LivePortrait 提取隐式关键点运动 latent $m \in \mathbb{R}^{25 \times 3}$（21 个关键点 + 头部姿态 + 表情形变），RVQ 将连续 latent 编码为离散 token，AR Transformer 接收 Whisper 编码的音频嵌入，以 next-token prediction 方式生成运动 token 序列。**Stage 2 (ETM)**：3D U-Net + temporal self-attention 单步精炼，增强颈部肌肉、耳环等配饰的物理一致性。

### 关键设计

1. **Facial Motion Latent Generation (FMLG)**

    - 功能：将连续面部运动映射为离散 token，实现高效的音频到运动实时生成
    - 核心思路：每 4 帧的运动 latent（$4 \times 25 \times 3$）被压缩为 32 个离散 token。RVQ 训练目标包括重建损失 $\mathcal{L}_{recon}$ 和 commitment 损失 $\mathcal{L}_{commit}$。AR Transformer 基于 Qwen1.5-4B 架构，以 200ms 音频块为单位处理（对应 Whisper 的 $10 \times 512$ 嵌入和 32 个运动 token）。创新地使用双 token 预测头——每步同时预测一对 token，推理速度翻倍。两个头的损失用正则项 $\|\mathcal{L}_{head0} - \mathcal{L}_{head1}\|_2^2$ 平衡学习。
    - 设计动机：离散 token 使自回归 next-token prediction 变得可行，避免了扩散模型的多步迭代

2. **Efficient Temporal Module (ETM)**

    - 功能：单步精炼身体配饰和肌肉的自然运动
    - 核心思路：VAE 编码器提取视频帧特征 $x \in \mathbb{R}^{b \times t \times h \times w \times c}$，reshape 为 $(b \times h \times w) \times t \times c$ 后在时序维度做 self-attention，通过残差连接融合时序依赖到空间特征。使用 MediaPipe 检测面部关键点定义颈部、耳朵等区域的 bounding box，通过 region-specific mask 的重建损失 $\mathcal{L}_{ETM}$ 聚焦配饰运动的物理一致性。只需**单步**前向传播（不像扩散模型需多步），保持实时性。
    - 设计动机：LivePortrait 基于隐式关键点驱动，天然缺乏对非面部区域（如耳环、项链）运动的建模

3. **流式推理设计**

    - 功能：实现端到端的实时流式动画
    - 核心思路：音频按 200ms 分块，Whisper 编码 7ms，AR Transformer 生成 32 token（每 16 token 约 6ms），运动解码 10ms；Stage 2 的 VAE 编解码 25ms + ETM 21ms。总计单块约 180ms < 200ms 的音频时长，维持实时。生成 4 帧后插值到 5 帧达到 25 FPS。
    - 设计动机：200ms 分块是 Whisper 的自然约束，与人类对音视频同步的感知阈值一致

### 损失函数 / 训练策略
- RVQ 损失：$\mathcal{L}_{vq} = \mathcal{L}_{recon} + \mathcal{L}_{commit}$
- AR 损失：$\mathcal{L}_{ar} = \sum[\mathcal{L}_{head0} + \mathcal{L}_{head1} + \|\mathcal{L}_{head0} - \mathcal{L}_{head1}\|_2^2]$
- ETM 损失：$\mathcal{L}_{ETM}$ 带区域 mask 的重建损失
- 预训练数据：AV Speech (662h) + VFHQ (2h)，SFT 数据 32h

## 实验关键数据

### 主实验（HDTF 数据集）

| 方法 | FID↓ | FVD↓ | Sync-C↑ | Sync-D↓ | 1s生成时间 |
|------|------|------|---------|---------|-----------|
| SadTalker | 22.18 | 233.67 | 7.326 | 7.848 | 18.89s |
| EchoMimic | 23.05 | 290.19 | 6.664 | 8.839 | 31.10s |
| AniPortrait | 28.16 | 235.10 | 4.547 | 10.657 | 29.36s |
| Hallo | 20.64 | 174.19 | 7.497 | 7.741 | 20.93s |
| **Teller** | 21.35 | **173.46** | **7.696** | **7.536** | **0.92s** |
| Real video | - | - | 8.094 | 6.976 | - |

### 消融实验

| 配置 | FVD↓ | Sync-C↑ | 说明 |
|------|------|---------|------|
| Full Teller | 173.46 | 7.696 | 完整模型 |
| w/o ETM | ~190 | ~7.5 | 配饰运动僵硬 |
| w/o 双头预测 | ~185 | ~7.6 | 推理速度降低 ~40% |
| 单 token 预测 | - | - | 速度减半但质量接近 |

### 关键发现
- Teller 推理速度是 Hallo 的 **22.7 倍**（0.92s vs 20.93s），且 FVD、Sync-C/D 指标更优
- ETM 对颈部肌肉和配饰运动的改善在定性评估中非常明显（人类评估显著优势）
- 4 帧压缩到 32 token 是帧数/冗余度的最优 trade-off
- RAVDESS 情感数据集上，Teller 在"愤怒"和"厌恶"表情上表现尤为突出

## 亮点与洞察
- **首个自回归实时 talking head 框架**，打破了"高质量=扩散模型=慢"的固有认知。证明了 AR 在音频驱动动画中的可行性
- **双 token 预测头**设计简洁有效——每步预测两个 token 直接将推理速度翻倍，正则项保证两个头学习平衡
- **ETM 模块**解决了隐式关键点驱动方法长期忽视的配饰运动问题，单步精炼保持了实时性

## 局限与展望
- 基于 LivePortrait 的隐式关键点表示，继承了其对大角度侧脸的限制
- 200ms 分块引入固定延迟，对超低延迟场景可能不够
- ETM 的 region mask 依赖 MediaPipe 关键点检测，面部遮挡时可能失效
- 仅支持上半身，全身动画是未来方向

## 相关工作与启发
- **vs Hallo/EMO**: 扩散模型生成质量高但 20+ 秒/秒，完全无法实时；Teller 用 AR 替代扩散实现 22x 加速
- **vs SadTalker**: 同为非扩散方法但用 FaceVid2Vid 做合成，缺乏配饰运动建模；Teller 的 ETM 补充了这一短板
- **vs VASA-1**: VASA-1 同样用扩散做运动 latent 生成，但非实时；Teller 的 RVQ + AR 方案实现了实时

## 评分
- 新颖性: ⭐⭐⭐⭐ AR + RVQ 在 talking head 中首次应用，ETM 设计实用
- 实验充分度: ⭐⭐⭐⭐ 多数据集对比、human eval、实时性分析全面
- 写作质量: ⭐⭐⭐ 框架清晰但写作有些粗糙（拼写错误较多）
- 价值: ⭐⭐⭐⭐⭐ 实时流式 talking head 的里程碑，工业应用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation](wav2sem_plug-and-play_audio_semantic_decoupling_for_3d_speech-driven_facial_anim.md)
- [\[CVPR 2025\] HunyuanPortrait: Implicit Condition Control for Enhanced Portrait Animation](hunyuanportrait_implicit_condition_control_for_enhanced_portrait_animation.md)
- [\[CVPR 2025\] Parallelized Autoregressive Visual Generation](parallelized_autoregressive_visual_generation.md)
- [\[CVPR 2026\] PersonaLive! Expressive Portrait Image Animation for Live Streaming](../../CVPR2026/video_generation/personalive_expressive_portrait_image_animation_for_live_streaming.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)

</div>

<!-- RELATED:END -->
