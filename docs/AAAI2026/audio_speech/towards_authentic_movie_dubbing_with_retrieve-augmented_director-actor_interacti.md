---
title: >-
  [论文解读] Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning
description: >-
  [AAAI 2026][音频/语音][电影配音] Authentic-Dubber 模拟真实配音工作流程中导演与演员的交互过程，通过构建多模态参考素材库、基于情感相似度的检索增强策略和渐进式图语音生成方法，显著提升了自动电影配音的情感表现力，在V2C-Animation数据集上的情感准确率和MOS评分均达到SOTA。
tags:
  - "AAAI 2026"
  - "音频/语音"
  - "电影配音"
  - "情感表达"
  - "检索增强生成"
  - "图神经网络"
  - "多模态情感建模"
---

# Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning

**会议**: AAAI 2026  
**arXiv**: [2511.14249](https://arxiv.org/abs/2511.14249)  
**代码**: [https://github.com/AI-S2-Lab/Authentic-Dubber](https://github.com/AI-S2-Lab/Authentic-Dubber)  
**领域**: 音频语音  
**关键词**: 电影配音, 情感表达, 检索增强生成, 图神经网络, 多模态情感建模

## 一句话总结
Authentic-Dubber 模拟真实配音工作流程中导演与演员的交互过程，通过构建多模态参考素材库、基于情感相似度的检索增强策略和渐进式图语音生成方法，显著提升了自动电影配音的情感表现力，在V2C-Animation数据集上的情感准确率和MOS评分均达到SOTA。

## 研究背景与动机

### 领域现状
自动电影配音（Visual Voice Cloning, V2C）旨在根据给定脚本生成生动语音，同时模仿说话人音色并保证唇音同步。现有工作已在发音质量（Speaker2Dubber）、音视频同步（FlowDubber）和表现力（ProDubber）方面取得进展。

### 核心痛点

**现有方法模拟的是一个过度简化的配音流程**：演员直接根据目标片段进行配音，没有任何准备和参考。这忽视了真实配音工作中**导演与演员之间的关键交互过程**。

在真实的电影配音工作流中：

**导演**会提供丰富的参考素材（情感参考片段）给配音演员

**演员**需要充分学习和内化这些素材中的情感线索，特别是情感表达
3. 只有在充分理解情感上下文后，演员才能进行富有情感表现力的配音

现有模型仅依赖目标片段本身的跨模态建模来生成语音，导致**情感表达能力受限**——因为单个片段所包含的情感信息有限，模型难以捕捉丰富的情感细节。

### 本文切入角度
借鉴真实配音工作流程，设计"导演提供素材→演员学习素材→演员配音"的三阶段架构，用检索增强（RAG）的思路引入外部情感知识，用渐进图结构累积情感信息。

## 方法详解

### 整体框架
Authentic-Dubber包含三个核心模块：（1）多模态参考素材构建——模拟导演提供参考；（2）基于情感相似度的检索增强——模拟演员高效学习素材；（3）渐进式图语音生成——模拟演员最终配音。输入为脚本文本、无声视频和音色提示音频，输出为情感表达丰富的配音语音。

### 关键设计

#### 1. **多模态参考素材库构建 (Multimodal Reference Footage Library, MRFL)**
- **功能**：基于V2C数据集，为每个样本提取四种模态的情感向量，构建情感参考素材库
- **核心思路**：设计四个专用的情感提取器：
    - **场景情感提取器**：使用VideoLLaMA 2生成场景情感描述（融入色调、亮度、饱和度等低层视觉特征），再通过RoBERTa情感模型提取场景情感向量 $S_i$
    - **面部情感提取器**：使用VideoLLaMA 2生成面部表情变化描述，再通过RoBERTa提取面部情感向量 $F_i$
    - **文本情感提取器**：双路径设计——直接文本情感 $T_i^{self}$ + 基于COMET的常识反应情感 $T_i^{react}$，拼接得到完整文本情感向量 $T_i$
    - **音频情感提取器**：使用Emotion2Vec提取音频情感向量 $A_i$
- **设计动机**：间接情感（场景/面部/文本）和直接情感（音频）分别对应不同维度的情感线索。LLM的深度理解能力可以将多模态信号统一到语义空间，比直接用I3D或EmoFan提取嵌入更有效（消融实验验证）

#### 2. **基于情感相似度的检索增强 (Emotion-Similarity-based Retrieval-Augmentation, ESRG)**
- **功能**：以目标片段的基础情感作为查询，从MRFL中检索最相关的多模态情感信息
- **核心思路**：
    - **说话人无关策略**：在动画配音场景中，角色是虚拟创建的，特定说话人的参考素材有限，因此采用跨说话人检索以获得更丰富的情感多样性
    - **三路并行检索**：
    - 场景查询 $S$ → 检索Top-K场景信息 $S_{r1 \to rk}$ + 匹配音频 $A_{r1 \to rk}^s$
    - 面部查询 $F$ → 检索Top-K面部信息 $F_{r1 \to rk}$ + 匹配音频 $A_{r1 \to rk}^f$
    - 文本查询 $T$ → 检索Top-K文本信息 $T_{r1 \to rk}$ + 匹配音频 $A_{r1 \to rk}^t$
    - 文本检索的特殊设计：分别计算 $T^{self}$ 和 $T^{react}$ 的相似度，取平均值作为检索标准
    - 相似度度量：使用余弦相似度（实验证明优于点积和欧氏距离）
- **设计动机**：真实配音中演员不可能看到目标语音（因为还没有配），所以用间接情感信息检索，再通过索引查找获得匹配的直接情感音频

#### 3. **渐进式图语音生成 (Progressive Graph-based Speech Generation, PGSG)**
- **功能**：以渐进的"构建-编码"范式，通过三层图结构逐步积累情感知识
- **核心思路**：三阶段渐进图结构：

  **阶段一 — 基础情感图 $\mathcal{G}_{beg}$**：
  - 节点：目标片段的场景情感 $S$、面部情感 $F$、文本情感 $T$
  - 边：三个节点两两相连
  - 使用图注意力编码器（GAE）编码，学习基础情感知识

  **阶段二 — 间接情感扩展图 $\mathcal{G}_{ieg}$**：
  - 基于编码后的 $\tilde{\mathcal{G}}_{beg}$，将检索到的间接情感节点添加到图中
  - 检索节点连接到同模态的基础情感节点
  - 编码后累积学习间接情感信息

  **阶段三 — 直接情感扩展图 $\mathcal{G}_{deg}$**：
  - 基于编码后的 $\tilde{\mathcal{G}}_{ieg}$，将匹配的直接情感音频添加为新节点
  - 通过GAE编码学习直接情感知识

  **情感知识语音合成器**：
  - 三层图的节点表示 $H_{beg}$、$H_{ieg}$、$H_{deg}$ 通过层级交叉注意力聚合：
  $E_{t,v,r}^{beg} = \text{Conv1D}([H_{t,v,r}; \text{CA}(H_{t,v,r}, H_{beg}, H_{beg})])$
  - 逐层叠加：基础→间接→直接，模拟演员从浅到深内化情感的过程
  - 最终表示送入Mel解码器生成Mel频谱，通过BigVGAN vocoder转换为语音

- **设计动机**：真实配音流程是渐进的：先理解基本情感，再参考类似素材深化理解，最后结合真实音频进行表演。渐进图结构完美对应这一流程

### 跨模态对齐
继承StyleDubber的跨模态对齐器（Cross-Modal Aligner），基于输入脚本和视觉帧实现音视频同步，并从音色提示中学习声音特征。

## 实验关键数据

### 主实验（V2C-Animation数据集）

| 方法 | EMO-ACC(↑) | WER(↓) | SECS(↑) | MCD-DTW-SL(↓) | MOS-DE(↑) | MOS-SE(↑) |
|------|-----------|--------|---------|---------------|-----------|-----------|
| Ground-Truth | 99.96 | 22.03 | 100.00 | 0.00 | 4.416 | 4.497 |
| FastSpeech2 | 42.39 | 33.30 | 25.47 | 14.72 | 3.058 | 3.063 |
| V2C-Net | 43.07 | 67.98 | 40.65 | 19.16 | 3.146 | 3.149 |
| HPMDubbing | 43.94 | 135.72 | 34.11 | 12.64 | 3.362 | 3.320 |
| StyleDubber | 45.73 | 24.70 | 83.46 | 9.40 | 3.676 | 3.738 |
| Speaker2Dubber | 44.55 | **18.27** | 81.26 | 9.82 | 3.432 | 3.461 |
| **Authentic-Dubber** | **47.21** | 25.95 | **84.40** | **9.68** | **3.792** | **3.889** |

### 消融实验

| # | 配置 | EMO-ACC(↑) | MOS-DE(↑) | MOS-SE(↑) |
|---|------|-----------|-----------|-----------|
| - | 完整模型 | **47.21** | **3.792** | **3.889** |
| 1 | w/o Scene Caption (用I3D替代) | 46.34 | 3.582 | 3.612 |
| 2 | w/o Face Caption (用EmoFan替代) | 46.52 | 3.653 | 3.684 |
| 3 | w/o 两种Caption | 46.02 | 3.520 | 3.608 |
| 4 | w/o 场景检索 | 46.27 | 3.591 | 3.666 |
| 5 | w/o 面部检索 | 46.64 | 3.657 | 3.690 |
| 6 | w/o 文本检索 | 45.99 | 3.540 | 3.614 |
| 7 | w/o 所有检索 | 45.23 | 3.511 | 3.527 |
| 8 | w/o 间接信息 | 45.95 | 3.542 | 3.581 |
| 9 | w/o 直接音频 | 45.30 | 3.492 | 3.571 |
| 10 | w/o 图建模 | 45.92 | 3.518 | 3.549 |
| 11 | w/o 构建编码范式 | 46.85 | 3.705 | 3.749 |
| 12 | w/o 层级聚合 | 46.71 | 3.661 | 3.710 |

### 关键发现
1. **情感准确率（EMO-ACC）提升显著**：47.21% vs 之前SOTA的45.73%（StyleDubber），相对提升3.2%
2. **LLM生成的情感描述比直接视觉特征更有效**：去除LLM Caption后EMO-ACC下降0.7-1.2%，证明LLM的深度语义理解贡献显著
3. **检索增强策略每个模态都有贡献**：去除所有检索时EMO-ACC下降2.0%，文本检索最重要（去除后下降1.2%）
4. **渐进图结构的每个组件都不可或缺**：去除直接音频或图建模后下降最大
5. **说话人无关检索优于说话人特定检索**：K=3时达到最优47.21%，过多检索引入噪声
6. **余弦相似度是最优的相似度度量**：比点积和欧氏距离更稳定

## 亮点与洞察
1. **工作流程建模思路独特**：不是简单地增加模型容量或数据，而是从实际工作流程中提炼出"导演-演员交互"的核心机制，将领域知识转化为模型设计
2. **RAG与配音的结合自然合理**：将参考素材类比为检索知识库，将情感理解类比为知识密集型任务，这一类比非常贴切
3. **渐进图结构设计精巧**：基础情感→间接情感→直接情感的三层递进，对应从浅层到深层的情感理解过程
4. **消融实验极其充分**：12组消融覆盖了所有设计选择，包括LLM语义理解、检索策略、图结构等
5. **说话人无关检索的发现有实践价值**：在动画配音等虚拟角色场景中，跨说话人检索获得更好效果

## 局限与展望
1. 仅在V2C-Animation一个数据集上评估，且该数据集为动画电影，真人电影配音的效果未知
2. 情感准确率（EMO-ACC）绝对值仍然较低（47.21% vs GT的99.96%），与人类水平有较大差距
3. WER（25.95）不是最优的（Speaker2Dubber达到18.27），说明情感增强可能轻微影响发音准确性
4. 检索库的构建和检索过程增加了推理时的计算开销，实时性可能受影响
5. 当前固定Top-K=3，缺乏动态调整K值的机制
6. 未探索可控属性（如语速、音高）的显式建模

## 相关工作与启发
- **RAG (Retrieval-Augmented Generation)**：Authentic-Dubber与标准RAG的区别在于：(1)计算多种情感模态的相似度 (2)使用渐进图结构而非直接拼接检索结果
- **StyleDubber**：作为跨模态对齐的基础架构
- **Emotion2Vec**：通用情感表示模型，用于提取直接情感音频特征
- 启发：**将领域工作流程（domain workflow）转化为模型架构是一种被低估的设计方法论**，特别适合有明确人类流程的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ （工作流程建模思路新颖，但各组件（RAG、GNN、LLM情感提取）均为已有技术的组合）
- 实验充分度: ⭐⭐⭐⭐⭐ （主实验+12组消融+检索分析+相似度度量分析+频谱可视化，非常全面）
- 写作质量: ⭐⭐⭐⭐ （"导演-演员"的隐喻贯穿全文，叙事流畅）
- 价值: ⭐⭐⭐⭐ （对电影配音和情感语音合成领域有推动作用，RAG与多模态情感的结合有启发性）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[CVPR 2026\] Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](../../CVPR2026/audio_speech/omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](../../ACL2025/audio_speech/wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[CVPR 2025\] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](../../CVPR2025/audio_speech/dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)

</div>

<!-- RELATED:END -->
