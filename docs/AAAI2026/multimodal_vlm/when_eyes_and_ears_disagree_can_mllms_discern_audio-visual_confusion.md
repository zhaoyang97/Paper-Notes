---
title: >-
  [论文解读] When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?
description: >-
  [AAAI 2026][多模态VLM][多模态大语言模型] 发现多模态大语言模型（MLLMs）在音视觉信息不对称时严重受视觉主导而无法识别缺失音频的"音视觉混淆"现象，提出 AV-ConfuseBench 基准和 RL-CoMM 方法（引入外部音频模型做参考的阶梯式推理奖励 + 答案置信度优化），在仅用约 20% 训练数据的情况下提升基线模型准确率 10~30%。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "多模态大语言模型"
  - "音视觉混淆"
  - "幻觉"
  - "强化学习"
  - "协作式多模型"
---

# When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?

**会议**: AAAI 2026  
**arXiv**: [2511.10059](https://arxiv.org/abs/2511.10059)  
**代码**: [https://github.com/rikeilong/AVConfusion](https://github.com/rikeilong/AVConfusion)  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 音视觉混淆, 幻觉, 强化学习, 协作式多模型

## 一句话总结

发现多模态大语言模型（MLLMs）在音视觉信息不对称时严重受视觉主导而无法识别缺失音频的"音视觉混淆"现象，提出 AV-ConfuseBench 基准和 RL-CoMM 方法（引入外部音频模型做参考的阶梯式推理奖励 + 答案置信度优化），在仅用约 20% 训练数据的情况下提升基线模型准确率 10~30%。

## 研究背景与动机

**音视觉混淆——一个被忽视的 MLLM 缺陷**：

MLLMs（如 Qwen2.5-Omni、Gemini 2.5）在音视觉理解任务上取得了显著进展，但论文发现一个关键问题：

> 当给定的音视觉信息不对称时，MLLM 能否识别出视觉存在但音频缺失的混淆对象？

论文通过实验揭示了触目惊心的结果：

**音频静音场景**：将视频中某乐器声音静音后询问"是否有该乐器声音"，Qwen2.5-Omni-7B 仍有 90.41% 的概率回答"是"——**几乎完全被视觉信息误导**

**音频篡改场景**：将音乐替换为鸟叫声，MLLMs 生成的内容仍主要描述视觉信息，对实际音频内容极不敏感

**即使 Gemini 2.5 Pro 开启思考模式**：仍有 38.36% 的响应被视觉引导输出错误

**开源模型更糟**：Video-LLaMA2 准确率仅 2.73%，几乎100%回答"是"

**根本原因分析**：MLLMs 在同步音视频数据上训练后，形成了视觉-音频的绑定认知——"看到某物就认为听到了它的声音"。模型推理过程被视觉信息主导，即使音频层面的不确定性很高（如通过 entropy 指标观察到模型后续层的不确定性分数持续偏高），仍倾向于依赖视觉做出判断。

## 方法详解

### 整体框架

RL-CoMM（Reinforcement Learning-based Collaborative Multi-MLLM）基于 Qwen2.5-Omni-3B 构建，包含三个训练阶段：

1. **Warm-up**：少量高质量 Q&A 对的监督微调，建立结构化推理格式
2. **Step-RR**（Step-wise Reasoning Reward）：阶梯式推理奖励的 GRPO 优化
3. **Ans-CO**（Answer-centered Confidence Optimization）：答案置信度优化

核心创新：引入**外部大型音频语言模型（LALM）**作为参考模型 $\pi_{ref}$，**Omni-LLM** 作为策略模型 $\pi_\theta$，通过异构模型协作弥补 Omni-LLM 的音频感知弱点。

### 关键设计

#### 1. AV-ConfuseBench 基准构建

论文提出的迷你基准包含两个设置：

**音频静音设置（Audio-muted）**：
- 多乐器演奏场景中静音某特定音源
- 问题形式："This is a video of audio corruption where some instrument sound is muted. Is there a/an {muted-object} sound?"
- 真实答案统一为"No"
- 39 个视频产生 73 个 Q&A 对
- 评估指标：准确率 + "Yes"回答比例

**音频篡改设置（Audio-modified）**：
- 将背景声替换为完全不同步的声音（风声、鸟叫、雨声、电钻声、雷声）
- 20 个视频 × 5 种声音 = 100 个 Q&A 对
- 问题："Describe what you see and what you hear"
- 评估指标：AI 辅助评分（音频准确度 A-Acc，视觉准确度 V-Acc，0-5 分）

#### 2. Step-wise Reasoning Reward (Step-RR)

Step-RR 的关键在于用外部 LALM 提供"纯音频视角"的参考推理，设计了结构化的输出格式：

- 策略模型输出三个标签：`<a-think>`（音频推理）、`<v-think>`（视觉推理）、`<answer>`（答案）
- 参考模型基于标准答案生成纯音频推理 `<a-think>`

**三种奖励信号**：

**格式奖励** $r_{format}$：输出是否符合指定的三段式格式（0/1）

**音频推理合理性奖励 (ARR)** $r_{arr}$：使用 Qwen3 Embedding-0.6B 计算策略模型的音频推理 $o_1^i$ 与参考模型音频推理 $o_{ref}$ 的语义相似度：

$$r_{arr}^i = \begin{cases} 1, & \text{if } \mathcal{S}(o_1^i | o_{ref}) > \omega \text{ 且 } o_3^i = y \\ 0, & \text{otherwise} \end{cases}$$

其中 $\omega = 0.8$ 为相似度阈值。设计动机：确保策略模型的音频推理不被视觉信息污染，与纯音频参考保持语义一致。

**音视觉关联奖励 (AVC)** $r_{avc}$：评估音频推理和视觉推理的关联性，包含软匹配机制：

$$r_{avc}^i = \begin{cases} 1 + \mathcal{I}(o_1^i | o_2^i), & \text{if } o_3^i = y \\ \mathcal{I}(o_1^i | o_2^i), & \text{if } o_3^i \neq \text{null}, \neq y \\ 0, & \text{otherwise} \end{cases}$$

设计动机：奖励合理的音视觉跨模态推理——答案正确时给予更高基础分加关联分，答案错误但有推理过程时仍给予关联分激励。

**组优势计算**：$r^i = r_{format} + r_{arr} + r_{avc}$，按标准 GRPO 归一化：

$$A^i = \frac{r^i - \text{mean}(\{r^1, ..., r^G\})}{\text{std}(\{r^1, ..., r^G\})}$$

注意：训练中**移除了 KL 惩罚**，因为参考模型和策略模型的结构异构，KL 散度没有意义。

#### 3. Answer-centered Confidence Optimization (Ans-CO)

解决异构推理差异导致的答案不确定性问题：

$$\mathcal{L}_{OP} = \underbrace{-\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(o_t | o_{<t}, x)}_{\text{NLL Loss}} + \lambda \cdot \underbrace{\frac{1}{|\mathcal{N}|}\sum_{t \in \mathcal{N}} H_t}_{\text{Entropy Minimization}}$$

其中 $\mathcal{N} = \{t | t > T_{prompt+think}\}$ 仅裁剪答案部分的 token，$H_t$ 为 token 级别的 entropy。$\lambda = 0.5$，当答案不确定性 $u > 0.75$ 时 $\lambda = 0$（避免过度自信降低泛化性）。

设计动机：Step-RR 优化了推理过程，但音频和视觉推理可能产生冲突的信号，Ans-CO 通过降低答案预测的 entropy 来确保最终答案的确定性。

### 损失函数 / 训练策略

- **基础模型**：Qwen2.5-Omni-3B
- **参考模型**：大型音频语言模型（LALM，如 Qwen-Audio）
- **Warm-up**：100 个高质量 Q&A 对的 SFT，使用 LLaMA-Factory
- **语义评价模型**：Qwen3 Embedding-0.6B
- **硬件**：8 × NVIDIA A800 GPU
- **少样本学习**：Step-RR 和 Ans-CO 仅用 Music-AVQA 和 AVQA 中的少量训练样本

## 实验关键数据

### 主实验

**Music-AVQA 和 AVQA 结果**：

| 方法 | Exist | Localis | Count | Comp | Temp | 平均 | AVQA 平均 |
|------|-------|---------|-------|------|------|------|-----------|
| PSTP-Net (专用) | 76.18 | 73.23 | 71.80 | 71.79 | 69.00 | 72.57 | 90.20 |
| CAD (专用) | 83.42 | 73.97 | 76.37 | 74.88 | 76.16 | 76.96 | 92.20 |
| Qwen2.5-Omni-3B | 60.02 | 53.84 | 61.29 | 58.16 | 46.57 | 54.95 | 83.78 |
| + SFT | 73.67 | 74.09 | 75.43 | 68.47 | 60.44 | 70.41 | 90.41 |
| + GRPO | 77.69 | 71.10 | 67.33 | 64.23 | 70.14 | 70.05 | 85.31 |
| + **RL-CoMM** | **85.61** | **76.68** | **84.08** | **70.74** | **76.30** | **79.46** | **95.87** |
| Δ (vs 基线) | +25.59 | +22.84 | +22.79 | +12.58 | +29.73 | **+24.51** | +12.09 |

**AVHBench 幻觉评估结果**：

| 方法 | 音频驱动视觉幻觉 Acc | 视频驱动音频幻觉 Acc | 音视觉匹配 Acc |
|------|-------------------|-------------------|--------------|
| OneLLM | 53.7 | 44.3 | 60.1 |
| Qwen2.5-Omni-3B | 65.85 | 59.65 | 48.77 |
| + GRPO | 72.98 | 62.84 | 49.73 |
| + **RL-CoMM** | **78.96** | **65.63** | **51.85** |

### 消融实验

**AV-ConfuseBench 上的训练策略对比**：

| 方法 | 音频静音 Acc ↑ | Yes 比例 ↓ | 音频篡改 A-Acc ↑ | V-Acc ↑ |
|------|-------------|-----------|----------------|---------|
| Qwen2.5-Omni-3B 基线 | 8.22 | 91.78% | 1.14 | 4.10 |
| + SFT | 5.48 (更差!) | 94.52% | - | - |
| + GRPO | 15.07 | 84.93% | 1.84 | 4.47 |
| + **RL-CoMM** | **27.40** | **72.60%** | **2.36** | **4.54** |

**Step-RR 和 Ans-CO 组件消融**（Music-AVQA 平均准确率）：

| 配置 | 平均准确率 |
|------|-----------|
| Qwen2.5-Omni-3B 基线 | 54.95 |
| + Format + Accuracy 奖励 | 70.05 |
| + Format + Step-wise Reasoning 奖励 | 74.49 |
| + Format + Step-wise Reasoning + **Ans-CO** | **79.46** |

### 关键发现

1. **SFT 反而有害**：在 AV-ConfuseBench 上 SFT 使准确率从 8.22% 降至 5.48%——SFT 强化了视觉-音频的绑定，加剧了混淆
2. **RL 思考式训练显著优于 SFT**：强迫模型在训练中"反思和试错"可以模拟人类思维来解决困难的音视觉任务
3. **RL-CoMM 提升巨大**：基线模型上提升 24.51%（Music-AVQA），19.18%（AV-ConfuseBench）
4. **Step-RR 优于简单准确率奖励**：74.49% vs 70.05%，步进式推理奖励有效纠正了视觉偏差
5. **异构参考模型的挑战**：ARR 奖励在训练中波动较大、峰值未达预期，说明 Omni-LLM 仍难以做到不受视觉影响的纯音频推理
6. **Ans-CO 的互补作用**：在推理奖励基础上再加 5 个百分点，证明推理优化和答案优化可以解耦但互补

## 亮点与洞察

1. **"音视觉混淆"现象的发现极具价值**：揭示了 MLLMs 的一个根本性认知缺陷——视觉主导导致音频感知的"失明"
2. **异构模型协作的RL设计巧妙**：用纯音频模型的推理作为参考来纠正 Omni 模型的视觉偏差
3. **SFT 有害的反直觉发现**：强化了"同步训练 ≠ 独立感知"的认知
4. **移除 KL 惩罚的合理性**：异构模型间 KL 散度无意义，需要审慎考虑 RL 框架中的默认组件
5. **实验设计（静音/篡改两种设置）完整**：分别测试了"是否能识别缺失"和"是否能平衡不一致信息"

## 局限与展望

1. **基础模型规模较小**：仅用 Qwen2.5-Omni-3B，更大模型可能缓解部分问题
2. **AV-ConfuseBench 规模有限**：仅 73+100 个样本，可能不足以全面评估
3. **音视觉匹配任务未取得最优**：RL-CoMM 在此任务上提升有限，需改进音视觉组合的奖励模型
4. **仅限于音乐/乐器场景**：更广泛的音视觉场景（语音、环境声等）尚未测试
5. **Warm-up 数据的构建**：100 个高质量 Q&A 对的构建标准和复现性需要更详细说明

## 相关工作与启发

- **Qwen2.5-Omni / Gemini 2.5**：当前最先进的全模态大模型
- **AVHBench**：音视觉幻觉评估基准
- **GRPO** (DeepSeek R1)：去 critic 的 RL 优化框架
- **Music-AVQA / AVQA**：音视觉问答标准基准
- **Entropy-based metric**：量化模型预测不确定性的方法
- 启发：多模态模型中的"模态主导性"是一个普遍问题（视觉通常主导音频/文本），需要在训练中显式引入模态解耦机制

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 音视觉混淆现象的发现和异构模型协作 RL 方案都具有很强原创性
- 实验充分度: ⭐⭐⭐⭐ — Music-AVQA/AVQA/AVHBench/AV-ConfuseBench 四个基准，消融充分，但 AV-ConfuseBench 规模偏小
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，方法描述较完整，公式标注准确
- 价值: ⭐⭐⭐⭐⭐ — 揭示了多模态 LLM 的根本性缺陷，为多模态推理的可靠性研究开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EgoAVU: Egocentric Audio-Visual Understanding](../../CVPR2026/multimodal_vlm/egoavu_egocentric_audio-visual_understanding.md)
- [\[ICML 2026\] AVI-Bench: Toward Human-like Audio-Visual Intelligence of Omni-MLLMs](../../ICML2026/multimodal_vlm/avi-bench_toward_human-like_audio-visual_intelligence_of_omni-mllms.md)
- [\[CVPR 2026\] HAVE-Bench: Hierarchical Audio-Visual Evaluation from Perception to Interaction](../../CVPR2026/multimodal_vlm/have-bench_hierarchical_audio-visual_evaluation_from_perception_to_interaction.md)
- [\[ICML 2026\] Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?](../../ICML2026/multimodal_vlm/robust-u1_can_mllms_self-recover_corrupted_visual_content_for_robust_understandi.md)
- [\[CVPR 2026\] FAVE: A Structured Benchmark for Fine-Grained Audio-Visual Temporal Evaluation in Multimodal LLMs](../../CVPR2026/multimodal_vlm/fave_a_structured_benchmark_for_fine-grained_audio-visual_temporal_evaluation_in.md)

</div>

<!-- RELATED:END -->
