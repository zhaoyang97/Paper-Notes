---
title: >-
  [论文解读] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation
description: >-
  [AAAI 2026 Oral][视频理解][情感视频数据集] 提出 EmoVid，首个面向艺术化/非写实内容的大规模多模态情绪视频数据集（22,758 个视频片段），覆盖动画、电影和表情贴纸三种类型，并通过微调 Wan2.1 模型展示了情绪条件化视频生成的有效性，在情绪准确率指标上显著优于基线。 视频是叙事和表达的强大媒介…
tags:
  - "AAAI 2026 Oral"
  - "视频理解"
  - "情感视频数据集"
  - "视频生成"
  - "情绪标注"
  - "文本到视频"
  - "情感计算"
---

# EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.11002](https://arxiv.org/abs/2511.11002)  
**代码**: [https://zane-zyqiu.github.io/EmoVid](https://zane-zyqiu.github.io/EmoVid) (Project Page)  
**领域**: Video Understanding  
**关键词**: 情感视频数据集, 视频生成, 情绪标注, 文本到视频, 情感计算

## 一句话总结
提出 EmoVid，首个面向艺术化/非写实内容的大规模多模态情绪视频数据集（22,758 个视频片段），覆盖动画、电影和表情贴纸三种类型，并通过微调 Wan2.1 模型展示了情绪条件化视频生成的有效性，在情绪准确率指标上显著优于基线。

## 研究背景与动机

视频是叙事和表达的强大媒介，情绪在观众参与度中起着关键作用。近年来视频生成模型在视觉连贯性和运动方面取得了长足进步，但**对情绪表达的关注非常有限**。特别是在创意应用（如漫画人物动画、表情包制作、电影剪辑）中，情感表达力至关重要但尚未被充分探索。

现有视频情绪数据集存在以下问题：

**规模有限**：如 CAER（12h）、MELD（1.4h）、DEAP（2h）等数据集规模较小

**内容单一**：几乎所有数据集都聚焦于真实场景中的人脸表情，缺乏风格化和非写实内容

**模态不全**：很多数据集缺少音频或文本描述

**不适用于生成任务**：现有数据集主要面向情绪识别，缺乏丰富的视觉多样性

核心矛盾：情绪在视频创作中极为重要，但缺乏适合创意场景的情绪基准数据集和对应的评估协议。EmoVid 填补了这一空白，覆盖了动画、电影和表情贴纸三种艺术化视频类型。

## 方法详解

### 整体框架
EmoVid 的构建和应用包括三个部分：
1. 数据集构建（收集、标注、属性提取）
2. 数据分析（情绪模式、颜色-情绪关联、时序动态）
3. 基准评估与模型微调（T2V 和 I2V 任务）

### 关键设计

1. **多源数据收集**:

    - 功能：从三种来源收集视频——MagicAnime 数据集（2,807 个动画面部片段）、Condensed Movies（13,255 个电影片段，经 PySceneDetect 切分并保留 4-30 秒片段）、Tenor API（6,696 个 GIF 表情贴纸）
    - 核心思路：覆盖多种艺术风格和内容类型，涵盖美式/中式/日式动画、电影场景、网络表情包
    - 设计动机：确保情绪表达的多样性和跨域通用性

2. **人机协作标注方案**:

    - 功能：采用 Mikels 八情绪模型（amusement, awe, contentment, excitement, anger, disgust, fear, sadness），结合人工标注和 VLM 自动标注
    - 核心思路：先用人工标注 20% 数据（每个视频由 3 人标注，至少 2 人一致才保留），再在此基础上微调 NVILA-Lite-2B 模型标注剩余 80%
    - 质量验证：随机抽取 1% 验证集，计算三名人工标注者与 VLM 之间的 Cohen's kappa 分数，差异 < 4%，表明 VLM 标注质量与人工相当
    - 设计动机：平衡标注精度和资源消耗

3. **多维度属性标注与分析**:

    - 功能：为每个片段提取颜色属性（colorfulness, brightness, hue）、用 NVILA-8B 生成文本描述
    - 核心发现：
        - 积极情绪（正效价）的视频更亮、更丰富多彩
        - 高唤醒度情绪更暗但颜色更丰富
        - 电影片段的情绪马尔可夫转移矩阵显示强自持性（fear 0.53, anger 0.46）
        - 同效价内的转换远多于跨效价转换
    - 设计动机：为情绪感知视频生成提供可利用的颜色和时序先验

### 损失函数 / 训练策略
- 使用 DiffSynth Studio 框架微调 Wan2.1 模型
- LoRA 配置：rank=32, lr=1e-4, epochs=3, batch_size=1
- 训练数据平衡处理：2,727 动画 + 8,000 电影 + 6,616 贴纸

## 实验关键数据

### 主实验

**T2V 任务**:

| 方法 | FVD↓ | CLIP↑ | Flicker↓ | EA-2cls↑ | EA-8cls↑ |
|------|------|-------|----------|----------|----------|
| VideoCrafter-V2 | 610.1 | 0.3012 | 0.0184 | 80.42 | 42.50 |
| HunyuanVideo | 552.6 | 0.2776 | 0.0116 | 76.87 | 40.41 |
| CogVideoX | 584.0 | 0.3013 | 0.0213 | 82.91 | 44.58 |
| WanVideo (before) | 594.3 | 0.2982 | 0.0091 | 84.17 | 44.16 |
| **WanVideo (after)** | **573.7** | **0.3021** | 0.0143 | **88.33** | **48.33** |

**I2V 任务**:

| 方法 | FVD↓ | SD↑ | Flicker↓ | EA-2cls↑ | EA-8cls↑ |
|------|------|-----|----------|----------|----------|
| DynamiCrafter512 | 512.3 | 0.7288 | 0.0280 | 90.41 | 71.25 |
| CogVideoX | 528.4 | 0.7214 | 0.0331 | 90.83 | 70.83 |
| WanVideo (before) | 517.9 | 0.7146 | 0.0325 | 91.25 | 71.30 |
| **WanVideo (after)** | **517.8** | 0.7193 | 0.0324 | **94.58** | **76.25** |

### 消融实验

| 配置 | EA-8cls (T2V) | EA-8cls (I2V) | 说明 |
|------|--------------|--------------|------|
| WanVideo 原始 | 44.16 | 71.30 | 未微调基线 |
| WanVideo + EmoVid 微调 | **48.33** | **76.25** | 情绪准确率显著提升 |
| 提升量 | +4.17 | +4.95 | 证明 EmoVid 有效 |
| 超越最强竞品 | +3.75 (vs CogVideoX) | +5.42 (vs DynamiCrafter) | 全面优于所有竞品 |

### 关键发现
- 微调后模型在通用视觉指标（FVD, CLIP）上保持甚至略有改善，**情绪准确率（EA）大幅提升**
- I2V 任务的情绪准确率改善（+4.95）比 T2V（+4.17）更显著
- 电影片段中发现 "hold, intra-valence drift, arousal leap" 情绪轨迹模式
- 负面情绪展现链式升级模式：sadness → fear/anger
- 定性结果显示微调模型能更精准捕捉情绪意图（如面部表情更到位、情绪一致的运动模式）

## 亮点与洞察
- **首个艺术化内容的情绪视频基准**：填补了风格化视频情感计算的数据空白
- **多模态完备性**：同时包含视频、音频、文本描述、颜色属性、情绪标签
- **情绪时序分析**：马尔可夫转移矩阵揭示了电影中情绪演化的规律
- **实用应用**：LoRA 微调生成的表情贴纸可直接用于社交媒体
- **评估协议完善**：EA-2cls 和 EA-8cls 指标为情绪感知生成提供了标准化评估

## 局限与展望
- 假设每个片段只传达单一情绪，现实中情绪可能是复杂和复合的
- 音频模态尚未充分利用，未来可构建统一的视频-音频-文本多模态模型
- 情绪分布不均衡（如 amusement 和 awe 较少），可能影响模型对少数类的表现
- 表情贴纸没有音频（GIF 格式限制），模态完整性存在缺口
- 微调增加了 flicker 指标值（0.0091→0.0143），视觉稳定性略有下降

## 相关工作与启发
- 联结了情感计算和视频生成两个方向，为 affective video computing 开辟新赛道
- 颜色-情绪关联的量化分析可为可控视频生成提供先验指导
- Mikels 八情绪模型在图像领域验证充分，本文验证了其在视频领域的适用性
- 人机协作标注方案（20% 人工 + 80% VLM）是大规模标注的实用范式

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个面向艺术化内容的情绪视频数据集，定位独特
- 实验充分度: ⭐⭐⭐⭐ — T2V/I2V 双任务评估充分，但消融较简单
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，数据分析详尽
- 价值: ⭐⭐⭐⭐ — 数据集+基准+分析洞察对社区有持续贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DeRVOS: Decoupling Consistent Trajectory Generation and Multimodal Understanding for Referring Video Object Segmentation](../../CVPR2026/video_understanding/dervos_decoupling_consistent_trajectory_generation_and_multimodal_understanding_.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](../../CVPR2026/video_understanding/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](../../CVPR2026/video_understanding/openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[CVPR 2026\] PyraTok: Language-Aligned Pyramidal Tokenizer for Video Understanding and Generation](../../CVPR2026/video_understanding/pyratok_language-aligned_pyramidal_tokenizer_for_video_understanding_and_generat.md)

</div>

<!-- RELATED:END -->
