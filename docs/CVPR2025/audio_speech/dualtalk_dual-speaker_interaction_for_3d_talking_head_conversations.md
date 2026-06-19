---
title: >-
  [论文解读] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
description: >-
  [音频/语音] 提出 DualTalk——首个统一建模说话者和倾听者行为的多轮双人交互 3D 说话人头生成框架，配套构建了包含 50 小时、1000+ 身份的双人对话数据集。 领域现状 3D 说话人头生成是计算机视觉的活跃研究方向，在客服、远程办公、教育和娱乐中有广泛应用。现有方法要么只建模说话行为（如 FaceFormer…
tags:
  - "音频/语音"
---

# DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations

| 属性 | 值 |
|------|------|
| 会议 | CVPR 2025 |
| arXiv | [2505.18096](https://arxiv.org/abs/2505.18096) |
| 代码 | [项目页面](https://ziqiaopeng.github.io/dualtalk) |
| 领域 | 人体理解 / 3D 说话人头生成 |
| 关键词 | dual-speaker, talking head, listener modeling, role transition, 3D face animation |

## 一句话总结

提出 DualTalk——首个统一建模说话者和倾听者行为的多轮双人交互 3D 说话人头生成框架，配套构建了包含 50 小时、1000+ 身份的双人对话数据集。

## 研究背景与动机

### 领域现状

3D 说话人头生成是计算机视觉的活跃研究方向，在客服、远程办公、教育和娱乐中有广泛应用。现有方法要么只建模说话行为（如 FaceFormer、CodeTalker、SelfTalk），要么只建模倾听行为（如 Learning2Listen），二者独立发展。

### 现有痛点

1. **说话者和倾听者割裂建模**：真实对话中人需要在说话和倾听之间无缝切换，表情随对方反馈动态调整。单角色模型无法捕捉这种交互动态
2. **仅音频驱动的局限**：Audio2Photoreal 等方法仅用音频建模，缺少对方面部表情的视觉反馈，无法实现基于对方表情的自适应调整
3. **短时反应 vs 连续对话**：倾听者模型通常只生成简短、孤立的反应（几秒钟），不支持多轮连续对话中的长时间交互
4. **缺乏双人交互数据集**：现有 3D 人脸数据集（VOCASET、BIWI 等）不包含交互信息，L2L 数据集虽有交互但不支持多轮对话

### 本文目标

定义**多轮双人交互 3D 说话人头生成**新任务：给定双方音频和 Speaker-A 的面部运动，生成 Speaker-B 在整个多轮对话过程中的面部运动（包括说话和倾听两种状态）。

### 切入角度与核心 idea

将对话参与者视为在两种状态（说话/倾听）间切换的统一实体，用单一框架同时建模两种行为；通过跨模态时序增强和双人交互模块捕捉发言者的言语-视觉信号与倾听者反馈之间的动态耦合。

## 方法详解

### 整体框架

DualTalk 包含四个模块：(1) Dual-Speaker Joint Encoder——分别编码双方音频和 Speaker-A 的 blendshape；(2) Cross-Modal Temporal Enhancer——跨模态注意力 + BiLSTM 对齐时序特征；(3) Dual-Speaker Interaction Module——Transformer 编解码器建模双人动态；(4) Expressive Synthesis Module——自适应表情调制 + blendshape 参数预测。

### 关键设计 1：Dual-Speaker Joint Encoder

- **功能**：将双方音频和 Speaker-A 的面部运动编码到统一特征空间
- **核心思路**：两个独立的 Wav2Vec 2.0 编码器分别处理 Speaker-A 和 Speaker-B 的音频 $\mathbf{A}_A$, $\mathbf{A}_B$，线性投影到共享维度 $d$。同时，两层全连接网络 + ReLU 编码 Speaker-A 的 56 维 blendshape 系数 $\mathbf{M}_A$
- **设计动机**：双方音频提供语音内容和韵律信息，Speaker-A 的 blendshape 提供视觉反馈。分别编码后投影到共享空间，便于后续跨模态融合

### 关键设计 2：Cross-Modal Temporal Enhancer

- **功能**：对齐音频和面部运动的时序特征，确保跨帧一致性
- **核心思路**：先用 cross-attention（$Q = \mathbf{Z}_A$, $K = V = \mathbf{M}'_A$）让 blendshape 特征受音频调制；再用 BiLSTM 捕获前后文时序依赖；最后将原始音频特征 $\mathbf{Z}_A$ 与时序增强特征 $\mathbf{T}$ 拼接
- **设计动机**：音频和面部运动的时间尺度不同（音频采样率 16kHz vs 面部 30fps），跨模态注意力实现对齐；BiLSTM 的双向结构确保模型同时利用过去和未来上下文，这对于自然的面部动画至关重要

### 关键设计 3：Dual-Speaker Interaction Module + Expressive Synthesis

- **功能**：建模双人交互动态并生成表情丰富的面部动画
- **核心思路**：
    - Transformer Encoder 捕获长距离依赖和复杂交互模式
    - Modal Alignment Attention（受 FaceFormer 启发的偏置注意力）对齐时序信息
    - Transformer Decoder 迭代精炼生成上下文丰富的表示
    - 自适应表情调制：$\mathbf{D}' = \mathbf{D} + \alpha \cdot \sigma(\mathbf{D}\mathbf{W}_m + \mathbf{b}_m)$
    - 最终线性层映射到 56 维 blendshape 参数
- **设计动机**：Transformer 架构适合建模长序列中的远距离交互关系。自适应表情调制引入了根据上下文动态调整表情强度的能力
- **损失函数**：基于 blendshape 参数的回归损失（文中实验详述）

## 实验关键数据

### 数据集对比

| 数据集 | 时长 | 身份数 | 交互 | 多轮 |
|--------|------|--------|------|------|
| VOCASET | 0.5h | 12 | ✗ | ✗ |
| L2L | 72h | 6 | ✓ | ✗ |
| **DualTalk** | **50h** | **1000+** | **✓** | **✓** |

首个同时具备交互和多轮对话的大规模 3D 面部数据集，平均每段对话 2.5 轮。

### 主实验表（说话表现 Frechet Distance ↓）

| 方法 | FD-EXP | FD-JAW | FD-POSE |
|------|--------|--------|---------|
| FaceFormer | 34.90 | 5.40 | 8.00 |
| CodeTalker | 48.57 | 6.89 | 10.74 |
| SelfTalk | 35.77 | 5.49 | 8.14 |
| L2L | 24.61 | 3.69 | 7.08 |
| **DualTalk** | **11.14** | **1.90** | **3.83** |

DualTalk 在所有表情/下颌/姿态指标上大幅领先，FD-EXP 降低 55%（相比 L2L）。

### 倾听行为表现

| 方法 | SID-EXP ↑ | SID-JAW ↑ | SID-POSE ↑ |
|------|-----------|-----------|------------|
| FaceFormer | 0.54 | 0.36 | 0.50 |
| L2L | 2.86 | 1.89 | 1.19 |
| **DualTalk** | **3.48** | **2.23** | **1.72** |

DualTalk 生成的倾听反应更丰富多样（SID 越高表示多样性越好）。

### 关键发现

- 纯说话模型（FaceFormer, CodeTalker）SID 接近 0，即生成的倾听反应几乎是静止的
- DualTalk 在跨轮次的角色转换上保持连贯，不出现突变
- MSE 指标上 DualTalk 也一致最优
- rPCC（皮尔逊相关系数误差）显示 DualTalk 生成的时间相关性最接近真实对话

## 亮点与洞察

1. **任务定义的开创性**：首次明确提出"多轮双人交互"任务，填补了说话/倾听分离建模的研究空白
2. **统一框架设计**：不分别训练说话和倾听模型，单一模型处理角色切换，更符合人类对话的实际情况
3. **数据集规模与多样性**：50 小时、1000+ 身份、双通道音频、多轮标注，为后续研究提供了坚实基础
4. **性能提升显著**：FD-EXP 从 24.61 降到 11.14（L2L→DualTalk），说明双人交互建模带来的增益巨大

## 局限性

1. 仅用 blendshape 系数表示面部运动，精细度受限于 56 维参数的表达力
2. 数据集来源于特定的对话场景，可能不能完全覆盖所有情感和文化背景
3. 需要 Speaker-A 的真实面部运动作为输入，在没有视觉输入的纯音频场景无法应用
4. 多轮对话的评估指标尚不完善，如何量化"角色切换自然度"有待进一步研究

## 相关工作与启发

- **FaceFormer**（CVPR 2022）：Transformer-based 音频驱动说话人头，DualTalk 的 Modal Alignment Attention 受其启发
- **Learning2Listen**（CVPR 2022）：倾听者建模先驱，但仅支持单轮短反应
- **Audio2Photoreal**（CVPR 2024）：全身对话生成，但仅依赖音频无视觉反馈
- **启发**：双人交互建模思路可扩展到全身动作（手势、身体姿态）和多人（>2 人）对话场景

## 评分

⭐⭐⭐⭐ — 新任务定义有开创性，统一框架设计合理，配套数据集具有长期研究价值。实验结果说服力强，但 blendshape 表达力和评估指标可进一步完善

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[ACL 2025\] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5](../../ACL2025/audio_speech/childmandarin_a_comprehensive_mandarin_speech_dataset_for_young_children_aged_3-.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)
- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](../../NeurIPS2025/audio_speech/megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
