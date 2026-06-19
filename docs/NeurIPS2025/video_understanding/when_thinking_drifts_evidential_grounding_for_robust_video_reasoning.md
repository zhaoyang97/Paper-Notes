---
title: >-
  [论文解读] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning
description: >-
  [NeurIPS 2025][视频理解][视频推理] 系统揭示了CoT推理在视频理解中经常导致性能下降的"视觉思维漂移"现象，并提出Visual Evidence Reward（VER）强化学习框架，通过显式奖励与视觉证据对齐的推理链来纠正这一问题。 CoT在视频推理中反而有害 Chain-of-Thought（CoT）推理…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "视频推理"
  - "Chain-of-Thought"
  - "视觉思维漂移"
  - "强化学习"
  - "视觉证据奖励"
---

# When Thinking Drifts: Evidential Grounding for Robust Video Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.06077](https://arxiv.org/abs/2510.06077)  
**代码**: 暂无  
**领域**: 视频理解  
**关键词**: 视频推理, Chain-of-Thought, 视觉思维漂移, 强化学习, 视觉证据奖励

## 一句话总结

系统揭示了CoT推理在视频理解中经常导致性能下降的"视觉思维漂移"现象，并提出Visual Evidence Reward（VER）强化学习框架，通过显式奖励与视觉证据对齐的推理链来纠正这一问题。

## 研究背景与动机

### CoT在视频推理中反而有害

Chain-of-Thought（CoT）推理在文本任务中表现优异，但将其直接应用于视频理解时却常常适得其反。作者在10个视频基准、多个MLLM上进行了系统研究，发现一个反直觉的现象：**让模型"先思考再回答"反而降低了准确率**，尤其在开源模型（如Qwen2.5-VL、Video-R1）上表现明显。

进一步在MVBench的20个子任务上分析显示，CoT在需要快速视觉判断的任务（如场景转换检测）上损害最大，因为额外的推理token引入了过度合理化和幻觉细节。而在需要多步逻辑的任务（如计数物体运动）上CoT仍有帮助。

### 视觉思维漂移（Visual Thinking Drift）

作者将上述失败模式归纳为**视觉思维漂移**：模型的推理链逻辑上看似完美，但实际上已脱离视频内容——基于幻觉的视觉细节或片面的时间信息构建论证。推理越长，错误概率越高。

### 贝叶斯视角的解释

从贝叶斯框架出发，推理链的生成可以分解为：

$$p(c_{1:T}, a \mid q, \mathbf{v}) = p(a \mid c_{1:T}, q, \mathbf{v}) \prod_{t=1}^{T} p(c_t \mid c_{<t}, q, \mathbf{v})$$

每步生成中，视觉信号和语言先验共同影响softmax分布：

$$p(c_t \mid c_{<t}, q, \mathbf{v}) \propto \exp(\underbrace{\mathbf{h}_{c_{<t}}^\top W_{\text{lang}}}_{\text{语言先验}} + \underbrace{\mathbf{h}_{\mathbf{v}}^\top W_{\text{vis}}}_{\text{视觉似然}})$$

问题在于 $\|W_{\text{lang}}\| \gg \|W_{\text{vis}}\|$，随着推理链增长，自注意力越来越集中在已生成的文本token上，视觉信号被稀释。若每步正确概率为 $1-\varepsilon$，整条链正确概率为 $(1-\varepsilon)^T \approx 1-T\varepsilon$，失败率与链长线性增长。一旦早期token引入幻觉事实，后续所有推理都建立在错误基础上，且自回归解码无法回溯验证。

## 方法详解

### 整体框架

Video-VER采用两阶段训练流水线：先SFT冷启动推理能力，再通过带Visual Evidence Reward的GRPO强化学习优化。

### 关键设计

#### 1. Visual Evidence Reward（VER）

**核心思路**：不仅奖励正确答案，还显式奖励推理过程中引用了真实视觉证据的响应。

对每个问题 $q$，策略模型 $\pi_\theta$ 生成一组 $G$ 个响应 $\{o_i\}_{i=1}^G$。一个辅助LLM判官（Llama-3.1-70B-Instruct）评估每个响应是否引用了视觉证据，给出二元分数 $e_i \in \{0, 1\}$。证据增强奖励为：

$$r_i^{\text{evid}} = \begin{cases} r_i + \alpha & \text{若 } o_i \text{ 正确且 } e_i = 1 \\ r_i & \text{否则} \end{cases}$$

其中 $\alpha = 0.3$ 为证据奖励权重。归一化后的优势函数为：

$$A_i = \frac{r_i^{\text{evid}} - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

#### 2. 基于GRPO的策略优化

使用带裁剪的GRPO目标函数：

$$\mathcal{J}_{\text{evid-GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i\right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

奖励系统包含四个成分：准确性奖励、视觉证据奖励（$\alpha=0.3$）、格式奖励和长度奖励（目标320-512 token）。

#### 3. 视觉证据生成（倒置提示）

**设计动机**：通用视频描述粒度太粗，缺乏回答特定问题所需的视觉细节。

采用**倒置提示策略**：给Qwen2.5-VL-72B提供（视频、问题、正确答案）三元组，让它生成支持该答案的视觉观察列表。相比标准CoT中同时探索推理路径和最终答案，倒置提示采样自更低熵的分布 $p(e_{1:K} \mid q, a, \mathbf{v})$，天然优先视觉接地而非语言先验。外部VLM仅用于离线生成训练数据，推理时不依赖。

### 训练策略

- **阶段一（SFT）**：在Video-R1-COT-165k数据集上微调，赋予基础推理能力
- **阶段二（GRPO+VER）**：使用Reversed-in-Time和Video-R1-260k的混合数据集进行强化学习，GRPO组大小 $G=8$，训练2000次迭代，8×H200 GPU

## 实验关键数据

### 主实验

| 基准 | Qwen2.5-VL-7B (DA) | Qwen2.5-VL-7B (CoT) | Video-VER (CoT) | 提升(vs CoT) |
|------|-------|-------|-------|-------|
| MVBench | 63.6 | 59.8 | **64.1** | +4.3 |
| Video-MME | 59.2 | 54.7 | **59.3** | +4.6 |
| VideoMMMU | 47.3 | 47.8 | **52.7** | +4.9 |
| MMVU | 64.2 | 60.5 | **65.1** | +4.6 |
| VideoHallucer | 51.8 | 44.1 | **53.1** | +9.0 |
| EventHallusion | 64.5 | 67.3 | **70.0** | +2.7 |
| TempCompass | 73.7/52.2 | 71.3/49.9 | **74.0/52.8** | +2.7/+2.9 |

Video-VER在10个基准中9个排名第一，平均比基座模型CoT提升+4.0%。

### 消融实验

| 视觉证据类型 | MVBench | Video-MME | VideoMMMU | MMVU | 平均 |
|-------------|---------|-----------|-----------|------|------|
| 问题相关证据(QD-VE) | **64.1** | **59.3** | **52.7** | **65.1** | - |
| 通用视频描述(VC) | 63.9 | 58.7 | 52.2 | 64.8 | - |

| 帧数 | MVBench | Video-MME | VideoMMMU | MMVU |
|------|---------|-----------|-----------|------|
| 32帧 | **64.1** | **59.3** | **52.7** | **65.1** |
| 16帧 | 63.2 | 56.0 | 50.0 | 64.8 |
| 8帧 | 60.5 | 53.3 | 45.4 | 63.2 |

### 关键发现

1. 问题相关视觉证据在10个基准中9个上优于通用描述，证实了问题对齐的重要性
2. 更多帧（32帧）在8/10基准上最优，方法能有效利用更长的时间上下文
3. 在GPT-4o上也有相当比例问题直接回答优于CoT，说明视觉思维漂移是普遍问题
4. 多数投票（20次采样）可显著提升CoT表现，但计算开销巨大

## 亮点与洞察

1. **现象定义有价值**：首次系统定义和分析"视觉思维漂移"，为理解MLLM视频推理失败提供了贝叶斯理论框架
2. **倒置提示策略巧妙**：通过固定答案再生成证据，将高熵的CoT生成问题转化为低熵的证据检索问题
3. **方法高效轻量**：仅需额外的LLM判官和证据生成，无需修改模型架构，可直接应用于已有MLLM
4. **最大提升在幻觉检测**：VideoHallucer上+9.0%的绝对提升，直接验证了VER对抑制幻觉的有效性

## 局限与展望

- 帧采样不完整时（遗漏关键帧），即使VER也无法纠正，视觉表征质量是前提
- LLM判官（Llama-3.1-70B）本身的能力限制了奖励信号质量
- 视觉证据生成依赖Qwen2.5-VL-72B，虽然仅用于离线训练但成本不低
- 主要验证在中等长度视频，长视频中稀疏关键信息场景尚未充分探索
- 仅在闭合式任务（MCQ）上验证，开放式QA的扩展是重要未来方向

## 相关工作与启发

- **视频推理**: Video-R1 (GRPO+规则奖励), VideoChat-R1, TinyLLaVA-Video-R1
- **幻觉缓解**: V-DPO (偏好优化减少幻觉), Self-Introspective Decoding
- **文本推理**: DeepSeek-R1, Open Reasoner Zero, Kimi k1.5
- **启发**: 可将VER思路扩展到图像推理领域，或结合动态帧选择进一步优化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统分析视频CoT推理失败并提出理论解释
- 实验充分度: ⭐⭐⭐⭐⭐ — 10个基准、多个消融、定性分析全面
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，贝叶斯分析优雅
- 价值: ⭐⭐⭐⭐☆ — VER思路通用性强，但局限于闭合式任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ACL 2025\] RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning](../../ACL2025/video_understanding/raven_robust_advertisement_video_violation_temporal_grounding_via_reinforcement_.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)
- [\[NeurIPS 2025\] Video Finetuning Improves Reasoning Between Frames](video_finetuning_improves_reasoning_between_frames.md)
- [\[NeurIPS 2025\] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)

</div>

<!-- RELATED:END -->
