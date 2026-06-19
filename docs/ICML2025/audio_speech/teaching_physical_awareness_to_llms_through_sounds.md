---
title: >-
  [论文解读] Teaching Physical Awareness to LLMs through Sounds
description: >-
  [ICML 2025][音频/语音][物理感知] 提出 ACORN 框架，通过基于物理的声学通道仿真器生成大规模训练数据，配合同时捕获幅度和相位信息的音频编码器，教会 LLM 从声音中理解物理世界现象。 大语言模型已经在文本和多模态理解上取得了显著进展，但它们从根本上缺乏物理感知能力——即理解真实物理世界现象的能力…
tags:
  - "ICML 2025"
  - "音频/语音"
  - "物理感知"
  - "声学通道"
  - "LLM音频理解"
  - "多普勒效应"
  - "声学仿真"
---

# Teaching Physical Awareness to LLMs through Sounds

**会议**: ICML 2025  
**arXiv**: [2506.08524](https://arxiv.org/abs/2506.08524)  
**代码**: 无  
**领域**: 音频语音  
**关键词**: 物理感知, 声学通道, LLM音频理解, 多普勒效应, 声学仿真

## 一句话总结

提出 ACORN 框架，通过基于物理的声学通道仿真器生成大规模训练数据，配合同时捕获幅度和相位信息的音频编码器，教会 LLM 从声音中理解物理世界现象。

## 研究背景与动机

大语言模型已经在文本和多模态理解上取得了显著进展，但它们从根本上缺乏物理感知能力——即理解真实物理世界现象的能力。人类能通过声音直觉地感知环境：多普勒效应告诉我们车辆在靠近还是远离，多径效应揭示我们是在室内还是室外，双耳听觉让我们定位声源方向。然而，现有 Audio LLM 主要聚焦于语音识别和音频内容理解，无法从声音中提取物理属性（如运动状态、空间关系）。

这带来实际安全隐患：例如，语音控制的汽车可能接受车外人员的"开窗"指令，因为它无法判断声音来源的物理位置。核心挑战在于数据：收集和标注大规模物理声学数据代价高昂且几乎不可行，因为多普勒效应、多径反射等物理现象无法由人工直接标注。

本文的关键洞察是：接收到的声音可以分解为声源和物理通道两个独立分量（$y = h \circledast s$），因此可以用真实声源与仿真物理通道的卷积来合成训练数据，绕开了数据收集的瓶颈。

## 方法详解

### 整体框架

ACORN 框架包含三个核心组件：（1）基于物理的声学通道仿真器，生成多样化的通道脉冲响应（CIR）；（2）联合捕获幅度和相位信息的音频编码器；（3）与 LLM 连接的端到端架构。系统通过仿真器生成 100 万个 ⟨Audio, Question, Answer⟩ 元组（AQA-PHY 数据集），对 LLM 进行有监督微调。

### 关键设计

1. **声学通道仿真器**: 基于信号处理理论建模五个独立组件：LOS 直达路径、早期反射、混响、多普勒效应和麦克风阵列接收。每个组件可独立控制和随机化。CIR 建模为 $h(\tau) = \sum_{i=0}^{N} \alpha_i \delta(\tau - \tau_i) + R(\tau)$，其中 $R(\tau)$ 为混响尾部。多普勒效应通过时变延迟建模 $h(t, \tau) = \delta(\tau - \frac{d_0 + v \cdot t}{c})$。不同任务采用不同配置：目标参数精确控制，非关键参数随机化，最大化通道多样性。设计动机是组件级建模比环境级重建更灵活、可扩展。

2. **幅度-相位音频编码器**: 区别于传统仅关注幅度（如 Whisper 的 Mel 频谱图）的编码器，ACORN 编码器同时提取 STFT 的三个分量：幅度 $M(f,t) = |X(f,t)|$、相位正弦 $\sin(\angle X(f,t))$ 和相位余弦 $\cos(\angle X(f,t))$。使用 sin/cos 而非直接使用相位角是为了避免 $\pi$ 到 $-\pi$ 的相位缠绕问题。三个分量各经 3×3 1D 卷积（128→1280 通道）+ GELU 激活，拼接后（3840 通道）经两层 3×3 卷积融合降维至 1280 通道，加正弦位置编码保留时间上下文，最后通过 32 层 Transformer 输出音频 token。总参数量约 0.65B，幅度部分从 Whisper-large-v2 初始化以利用预训练的幅度表示，相位子网络从头训练。

    - **音频预处理**：16kHz 采样，STFT 窗长 254（对应 128 个频率 bin）、hop 10ms，直接保留完整频谱分辨率而不转换为 mel 谱图，以保留物理信号的精细特征
    - 设计动机是物理效应（如多普勒频移、多径时延）主要体现在微妙的相位关系中，仅依赖幅度无法捕获。实验证实，引入相位使距离估计误差降低了 7 倍

3. **模块化任务配置**: 五个声学感知任务：LOS 检测（判断是否存在视距路径）、多普勒估计（估计频率偏移）、到达方向估计（利用 TDoA $\tau_\theta = d\cos(\theta)/c$）、多径分析（判断混响程度）和距离估计（基于回波分析）。每个任务通过选择性启用/禁用仿真组件来控制物理参数。

### 损失函数 / 训练策略

- 使用标准的 next-token prediction loss，以 answer 文本为标签
- 音频编码器从 Whisper-large-v2 初始化（利用预训练的幅度表示），相位子网络从头训练
- LLM 使用 LoRA 微调，减少训练开销并利用其预训练的语言能力
- 音频 token 通过线性投影层映射到 LLM 词嵌入维度，用 `<soa>` 和 `<eoa>` 标记包裹
- 4 × A100 GPU，batch size 32，7 个 epoch，总训练时间约 61 小时
- 每个任务生成 20 万闭式 QA + 1 万开放式 QA

## 实验关键数据

### 主实验

| 任务 | 指标 | ACORN+Qwen2 | Whisper+Qwen2 | 提升 |
|--------|------|------|----------|------|
| LOS 检测 | BCA↑ | 0.924 | 0.881 | +4.3pp |
| 多普勒估计 | MAE_f↓ | 0.181 | 1.042 | 82.6%降低 |
| DoA 估计 | MAE_t↓ | 0.907 | 2.716 | 66.6%降低 |
| 多径分析 | TCA↑ | 0.903 | 0.848 | +5.5pp |
| 距离估计 | REP↓ | 1.599 | 10.609 | 84.9%降低 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 有LOS vs 无LOS（多径分析TCA） | 0.895 vs 0.912 | LOS 直达信号掩盖多径特征 |
| 有多普勒 vs 无多普勒（LOS检测BCA） | 0.912 vs 0.936 | 多普勒引入信号畸变 |
| SNR <10dB vs >40dB（距离MAE） | 5.33 vs 0.80 | 高 SNR 显著提升精度 |
| Merged vs Sole 训练 | 各任务详见表 | 合并训练接近独立训练 |

### 关键发现

- 相位信息对物理感知至关重要：ACORN 编码器在所有任务上全面超越仅用幅度的 Whisper，尤其距离估计提升 7 倍
- 方法具有模型无关性：Llama3.1-8B 和 Qwen2-7B 配合 ACORN 编码器都获得一致提升
- 开放式 QA 表现合理，模型能用自然语言解释物理现象并进行多步计算
- 零样本迁移到真实世界环境可行：在车辆环境中 LOS 检测达 0.870，DoA 估计达 0.925
- 模型对多种声学干扰具有较好的鲁棒性，性能下降有限

## 亮点与洞察

- 声音 = 声源 × 物理通道的分解思想非常优雅，将不可能的数据收集问题转化为可控的仿真问题
- 相位信息的引入是关键创新，揭示了传统音频编码器（基于 Mel 频谱）丢失的重要物理信息
- 开辟了 "LLM 物理感知" 这一全新研究方向，不同于视觉或文本的物理推理
- 组件化的仿真器设计使得每个物理现象可独立控制和研究
- 真实车辆场景的验证增强了实际应用价值

## 局限与展望

- 仅支持单轮对话，无法进行多轮交互推理
- 真实世界实验规模有限，仅在一辆车上的少量场景
- 仿真器虽然多样化，但与真实声学环境仍有差距（域迁移问题）
- 各任务独立建模，未探索物理现象之间的关联和联合推理
- 未引入思维链（Chain-of-Thought）等推理增强技术
- 仅测试了 7-8B 规模的 LLM，更大模型是否能更好地学习物理推理未知

## 相关工作与启发

- 与 BAT（空间音频理解）的区别：ACORN 直接从单通道提取相位，避免了成对计算的二次增长
- 与 NEWTON 等物理推理工作的互补：它们通过文本/视觉推理物理，ACORN 通过声学
- 声学传感领域的传统方法（ToA、TDoA、FMCW）依赖手工特征，ACORN 用端到端学习替代
- AudioGPT/Pengi/Qwen-Audio 等 Audio LLM 聚焦语义理解，ACORN 首次关注物理属性理解
- 启发：类似的"物理通道分离 + 仿真"思路可推广到无线通信信号、雷达信号等领域
- 车辆安全场景（判断语音来自车内/车外）是非常有说服力的应用案例

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 开辟全新方向，声音→物理感知独树一帜
- 实验充分度: ⭐⭐⭐⭐ 五个任务全面评估，含真实车辆实验，但规模有限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，物理原理解释到位，图表设计合理
- 价值: ⭐⭐⭐⭐ 展示了LLM物理感知的可行性，为具身AI提供新路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](../../AAAI2026/audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[ACL 2025\] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](../../ACL2025/audio_speech/mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](../../ECCV2024/audio_speech/action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)

</div>

<!-- RELATED:END -->
