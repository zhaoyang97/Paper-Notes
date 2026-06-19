---
title: >-
  [论文解读] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching
description: >-
  [ACL2025][图像生成][Flow Matching] 提出OZSpeech，首个将最优传输条件流匹配(OT-CFM)与学习先验分布相结合实现单步采样的零样本TTS系统，在内容准确性(WER)、推理速度和模型大小上均大幅领先现有方法。 - 零样本TTS的挑战：Zero-Shot TTS需要从未见过的说话者的少量语音片段…
tags:
  - "ACL2025"
  - "图像生成"
  - "Flow Matching"
  - "Zero-Shot TTS"
  - "One-Step Sampling"
  - "Learned Prior"
  - "Neural Codec"
---

# OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching

**会议**: ACL2025  
**arXiv**: [2505.12800](https://arxiv.org/abs/2505.12800)  
**代码**: [OZSpeech Demo](https://ozspeech.github.io/OZSpeech_Web/)  
**领域**: 图像生成  
**关键词**: Flow Matching, Zero-Shot TTS, One-Step Sampling, Learned Prior, Neural Codec  

## 一句话总结

提出OZSpeech，首个将最优传输条件流匹配(OT-CFM)与学习先验分布相结合实现单步采样的零样本TTS系统，在内容准确性(WER)、推理速度和模型大小上均大幅领先现有方法。

## 研究背景与动机

- **零样本TTS的挑战**：Zero-Shot TTS需要从未见过的说话者的少量语音片段中克隆其声学特征，是语音合成领域的核心难题
- **自回归方法的缺陷**：VALL-E等自回归模型存在非确定性采样导致的无限重复问题，在高精度场景下不够可靠
- **扩散模型的瓶颈**：E2 TTS等扩散模型虽能生成高质量音频，但多步采样带来高计算成本，难以满足实时应用需求
- **现有加速方案不足**：Consistency Models需要在t∈[0,1]全范围训练代价高；Shortcut Models引入额外约束更加消耗资源
- **传统OT-CFM的局限**：传统方法从高斯噪声开始构建输出分布，噪声与目标差距大，需要多步采样才能收敛
- **核心动机**：能否从一个更接近目标分布的learned prior出发，让流匹配仅需单步即可完成高质量语音合成？

## 方法详解

### 整体框架

OZSpeech的整体流程分为三个核心模块：

1. **Prior Codes Generator (fψ)**：将文本（音素）转换为先验码序列，作为flow matching的起始点
2. **OT-CFM Vector Field Estimator (vθ)**：从先验码出发，结合声学prompt中的韵律和声学细节，估计向目标分布的速度场
3. **FACodec**：将波形分解为说话人身份、韵律、内容和声学细节的解耦表示，并最终解码回语音波形

关键创新：不从高斯噪声出发，而是从learned prior出发，prior已接近目标分布，因此flow matching可以单步完成。

### Prior Codes Generator

采用层级结构的级联神经网络，每个码序列的生成依赖于前序码序列：

$$p(\mathbf{q}_{1:6}|\mathbf{p};\psi) = p(\mathbf{q}_1|\mathbf{p};f_\psi^1)\prod_{j=2}^{6}p(\mathbf{q}_j|\mathbf{q}_{j-1};f_\psi^j)$$

- 第一层以音素嵌入为条件生成内容码
- 后续层依次生成韵律和声学细节码
- 使用Duration Predictor对齐音素与输出码序列，MSE损失最小化log尺度时长预测误差

### One-Step OT-CFM重构

核心数学重构：将标准OT-CFM中的随机时间步t替换为可学习的先验依赖时间变量τ：

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{\mathbf{x}_{pr},\mathbf{x}_1}\left\|\mathbf{v}_\theta(\mathbf{x}_{pr},\tau) - \frac{\mathbf{x}_1 - \mathbf{x}_{pr}}{1-\tau}\right\|^2$$

**关键区别**：
- 不访问x₀（噪声分布），不强制x₀服从正态分布
- 当先验分布xpr接近目标分布x₁时，采样步数和步长均大幅减少
- 最终实现单步采样（NFE=1）

### Folding机制与Quantizer Encoding

- **Folding**：将6个quantizer序列沿hidden维度折叠（ℝ^{6×L×D} → ℝ^{L×D'}），同时建模所有quantizer，避免顺序处理的高计算量
- **Quantizer Encoding**：为每个quantizer添加可学习标识嵌入ω，防止模型在同一序列中混淆不同quantizer
- 输入中对prior codes加高斯噪声保证鲁棒性和多样性

### 损失函数

总损失由四部分组成：

$$\mathcal{L}_{total} = \mathcal{L}_{prior} + \mathcal{L}_{dur} + \mathcal{L}_{CFM} + \mathcal{L}_{anchor}$$

- **L_prior**：最小化先验码生成的负对数似然
- **L_dur**：时长预测器的MSE损失
- **L_CFM**：流匹配速度场回归损失
- **L_anchor**：正则化项，防止embedding坍缩，通过最小化估计目标与真实目标之间的负对数似然实现

## 实验

### 实验设置

- **训练数据**：LibriTTS（500小时多说话人英语音频）
- **评测数据**：LibriSpeech test-clean
- **评估指标**：UTMOS（语音质量）、WER（内容准确性）、SIM-O/SIM-R（说话人相似度）、F0/Energy（韵律）、NFE/RTF（延迟）
- **对比方法**：F5-TTS、VoiceCraft、NaturalSpeech 2、VALL-E

### 主实验结果

| 模型 | 训练数据 | WER↓ | SIM-O↑ | UTMOS↑ | NFE↓ | RTF↓ |
|------|---------|------|--------|--------|------|------|
| F5-TTS | 95,000h | 0.24 | 0.53 | 3.76 | 32 | 0.70 |
| VoiceCraft | 9,000h | 0.18 | 0.51 | 3.55 | - | 1.70 |
| NaturalSpeech 2 | 585h | 0.09 | 0.31 | 2.38 | 200 | 1.66 |
| VALL-E | 500h | 0.19 | 0.40 | 3.68 | - | 0.86 |
| **OZSpeech** | **500h** | **0.05** | **0.40** | 3.15 | **1** | **0.26** |

（以上为3s prompt设置）

**关键发现**：
- OZSpeech在WER上全面SOTA，5s prompt下相比次优方法降低44%
- 推理速度比次快的F5-TTS快约3倍
- 模型大小仅为其他方法的29%-71%（可训练参数仅17%-43%）
- 仅需500小时训练数据即超越使用95,000小时的F5-TTS在WER上的表现

### 消融实验

**Prompt策略对比**：
- Arbitrary Segment（随机段选取）在所有指标上优于First Segment（固定起始段）
- First Segment容易过拟合到将prompt转移到目标开头
- Arbitrary Segment隐藏prompt位置，提升泛化能力

### 噪声容忍度分析

- 在有噪声的audio prompt场景下，其他模型的WER随SNR降低急剧上升
- OZSpeech的WER保持稳定，展现出优异的噪声容忍能力
- 这证明了learned prior机制对噪声的天然鲁棒性

## 亮点与洞察

1. **理论优雅**：通过将learned prior引入OT-CFM框架，用数学推导自然地实现单步采样，不需要额外的蒸馏阶段
2. **效率极致**：NFE=1，RTF=0.26，比NaturalSpeech 2的200步采样减少了两个数量级
3. **小模型大能力**：145M可训练参数在WER上击败所有大模型基线
4. **解耦表示的价值**：使用FACodec将语音解耦为内容/韵律/声学细节/说话人，实现对各属性的精确控制
5. **低资源友好**：传统OT-CFM（如F5-TTS）在500小时数据上WER>0.95，而OZSpeech仅需500小时即达SOTA
6. **Folding机制巧妙**：同时建模6个quantizer避免顺序处理，大幅减少计算量

## 局限性

1. **UTMOS略低**：由于FACodec在声学和语义表示间的权衡，整体语音质量得分不如某些基线
2. **SIM-O/SIM-R非最优**：说话人相似度指标与最好方法仍有差距
3. **仅英语评估**：所有实验基于英语数据集，跨语言泛化未验证
4. **FACodec依赖**：系统性能高度依赖FACodec的预训练质量
5. **训练数据规模有限**：仅在500小时数据上验证，大规模数据能否进一步提升待探索

## 相关工作

- **自回归TTS**：VALL-E系列将TTS重定义为条件编码语言建模任务
- **扩散/流匹配TTS**：E2 TTS、NaturalSpeech 2/3、F5-TTS等基于扩散/流匹配的方法
- **蒸馏加速**：Consistency Models、Shortcut Models等减少采样步数的方法
- **神经编解码器**：SoundStream、EnCodec、FACodec等离散语音表示方法

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐⭐ 首次将learned prior与OT-CFM结合实现单步零样本TTS
- **实验充分性**：⭐⭐⭐⭐ 多维度评估，含消融和噪声分析
- **实用价值**：⭐⭐⭐⭐⭐ 低延迟、小模型、少数据的优势组合极具部署潜力
- **写作质量**：⭐⭐⭐⭐ 数学推导清晰，动机充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](rvc_rhythm_voice_conversion.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](../../CVPR2026/image_generation/few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)
- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[CVPR 2026\] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching](../../CVPR2026/image_generation/stable_mean_flow_lyapunov-inspired_one-step_flow_matching.md)
- [\[ECCV 2024\] FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior](../../ECCV2024/image_generation/freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)

</div>

<!-- RELATED:END -->
