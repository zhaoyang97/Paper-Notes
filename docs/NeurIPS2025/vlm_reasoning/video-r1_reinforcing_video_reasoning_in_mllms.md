---
title: >-
  [论文解读] Video-R1: Reinforcing Video Reasoning in MLLMs
description: >-
  [NeurIPS 2025][多模态VLM][视频推理] 受DeepSeek-R1启发，首次系统探索将R1范式（规则RL）应用于视频推理，提出T-GRPO算法显式鼓励模型利用时序信息，并构建图文混合训练数据集，在VSI-Bench上以37.1%准确率超越GPT-4o。 DeepSeek-R1展示了规则RL可以在文本领域激发出…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视频推理"
  - "强化学习"
  - "GRPO"
  - "时序建模"
  - "思维链"
---

# Video-R1: Reinforcing Video Reasoning in MLLMs

**会议**: NeurIPS 2025  
**arXiv**: [2503.21776](https://arxiv.org/abs/2503.21776)  
**代码**: [GitHub](https://github.com/tulerfeng/Video-R1)  
**领域**: 多模态VLM  
**关键词**: 视频推理, 强化学习, GRPO, 时序建模, 思维链

## 一句话总结

受DeepSeek-R1启发，首次系统探索将R1范式（规则RL）应用于视频推理，提出T-GRPO算法显式鼓励模型利用时序信息，并构建图文混合训练数据集，在VSI-Bench上以37.1%准确率超越GPT-4o。

## 研究背景与动机

DeepSeek-R1展示了规则RL可以在文本领域激发出强大的推理能力和长思维链，Kimi k1.5和Skywork R1V等工作开始将RL扩展到图像推理。但**视频推理领域的RL探索几乎空白**。

直接将GRPO算法应用于视频推理面临两大根本挑战：

**缺乏时序建模的显式奖励信号**：标准GRPO没有机制鼓励模型进行时序推理。模型可能"走捷径"——仅关注单帧的表面视觉模式而非跨帧的时序变化。例如，对于需要观察物体运动方向的问题，模型可能只看一帧就猜答案，而非推理多帧间的变化。作者和Video-UTR都观察到了这种shortcut现象。

**高质量视频推理数据的稀缺**：现有视频数据集主要聚焦简单识别任务而非推理。需要强推理能力或长推理路径的样本极为稀少，限制了RL的有效性。

## 方法详解

### 整体框架

Video-R1基于Qwen2.5-VL-7B-Instruct，采用两阶段训练：(1) SFT冷启动（Video-R1-CoT-165k）→ (2) T-GRPO强化学习（Video-R1-260k）。训练数据包含图像和视频的混合数据。

### 关键设计

1. **T-GRPO（Temporal Group Relative Policy Optimization）**：在GRPO基础上引入对比奖励机制，显式鼓励时序推理。核心思想是：对同一视频问题，分别用**时序正确的帧序列**和**随机打乱的帧序列**生成两组回答，比较两组的正确率来判断模型是否真正利用了时序信息。

时序奖励定义为：

$$r_t = \begin{cases} \alpha, & \text{if } p \geq \tilde{p} \\ 0, & \text{otherwise} \end{cases}$$

其中 $p$ 和 $\tilde{p}$ 分别是正序组和乱序组的正确答案比例，$\alpha=0.3$。仅当正序组表现优于或等于乱序组时才给予正向时序奖励。**关键**：$r_t$ 仅施加于正确回答，避免稀释奖励信号。

最终奖励为：

$$R_i = \begin{cases} r_i + r_t, & \text{if } o_i \text{ is correct} \\ r_i, & \text{otherwise} \end{cases}$$

2. **图文混合训练数据策略**：为弥补视频推理数据的不足，将高质量图像推理数据纳入训练。Video-R1-260k的构成：

    - 视频通用数据 116K（时序理解和推理）
    - 图像通用 15K + 图表 21K + OCR 16K + 数学 37K + 知识 37K + 空间 20K
   
   图像数据提供广泛的推理技能基础（数学、空间逻辑、领域知识），视频数据提供时序推理复杂性。模型能将从图像中学到的推理能力迁移到动态视频场景。

3. **多类型规则奖励设计**：

    - 多选题：答案是否匹配
    - 数值QA：预测数字是否精确匹配
    - OCR：词错误率（WER）
    - 自由问答：ROUGE-1/2/L的平均值
    - 回归：1 - 相对误差

4. **长度奖励控制**：引入长度奖励 $r_l = \omega$（$\omega = 0.2$），当正确回答的长度在 $[l_{\min}=320, l_{\max}=512]$ 范围内时给予额外奖励，平衡"深度思考"与"过度思考"。

### 损失函数 / 训练策略

- SFT阶段：在Video-R1-CoT-165k上训练1 epoch，用Qwen2.5-VL-72B生成CoT标注
- RL阶段：T-GRPO训练仅1K步（=约15小时），Adam优化器，学习率1e-6
- 正序组大小 $G=8$，乱序组大小 $\tilde{G}=4$（效率考虑）
- 训练时最大16帧，推理时可增至64帧
- KL散度系数 $\beta=0.04$，最大响应长度768 tokens

## 实验关键数据

### 主实验（64帧推理）

| 模型 | VSI-Bench | VideoMMMU | MMVU(mc) | MVBench | TempCompass | VideoMME |
|------|----------|-----------|---------|---------|-------------|---------|
| GPT-4o | 34.0 | 61.2 | 75.4 | - | - | 71.9 |
| Qwen2.5-VL-7B (CoT) | 31.4 | 50.4 | 60.0 | 59.2 | 72.9 | 59.6 |
| Qwen2.5-VL-7B-SFT | 34.8 | 49.4 | 61.6 | 60.6 | 70.0 | 58.8 |
| **Video-R1-7B** | **37.1** | **52.4** | **63.8** | **64.8** | **73.2** | **61.4** |

Video-R1-7B在VSI-Bench上37.1%**超越GPT-4o**（34.0%）。

### 消融实验

| 变体 | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|------|----------|-----------|------|---------|-------------|---------|
| wo-image（仅视频数据） | 32.3 | 45.8 | 60.6 | 60.9 | 69.8 | 53.8 |
| wo-temporal（GRPO替代T-GRPO） | 32.7 | 48.3 | 62.1 | 61.1 | 71.3 | 54.5 |
| zero（跳过SFT冷启动） | 31.8 | 49.5 | 63.8 | 60.4 | 70.9 | 53.8 |
| **Video-R1-7B（完整）** | **34.6** | **49.8** | **64.2** | **62.7** | **72.6** | **57.4** |

移除图像数据、T-GRPO或SFT冷启动均导致全面下降。

### 时序推理占比分析

| 模型 | 时序推理回答占比 |
|------|----------------|
| Video-R1 (T-GRPO) | **75.0%** |
| Video-R1-wo-temporal (GRPO) | 60.2% |

T-GRPO将时序推理行为占比提升了14.8个百分点。

### 关键发现

- **RL优于SFT**：SFT在VideoMME等benchmark上甚至轻微下降（可能过拟合），但仅1K步RL就带来显著提升，验证了"SFT memorizes, RL generalizes"观点
- **更多帧 = 更好推理**：从16→32→64帧，几乎所有benchmark持续提升
- **涌现行为**（Aha Moment）：模型在遇到模糊时序线索时会自我反思、重新审视视频证据，说明模型在主动学习而非简单记忆模式
- 训练曲线显示RL初期回答长度先下降（抛弃SFT的次优推理风格），后上升并稳定（形成新推理策略）
- $\alpha$ 在0.2-0.3范围内不敏感，0.1和0.4略差
- 扩展到10K步训练进一步提升（TempCompass 73.2→74.2, MVBench 64.8→65.5）

## 亮点与洞察

1. **首次系统探索R1范式在视频推理中的应用**，建立了该方向的基础框架
2. **T-GRPO设计精巧**：通过正序/乱序帧的对比学习，以极小的额外成本获得时序推理能力，思路具有普适性
3. **图文混合训练的实用价值**：以图像推理数据补充视频推理数据不足，是资源受限下的务实解决方案
4. **仅1K步RL即显著提升**，展示了数据设计和算法设计的高效性
5. **自我反思行为的涌现**：模型在处理模糊时序信息时展现出类似"aha moment"的推理回路

## 局限与展望

- 训练仅用16帧，限制了长程时序依赖的捕获
- T-GRPO引入额外计算开销（需等量乱序推理），可通过vLLM等推理加速框架缓解
- 长度奖励使用固定区间而非按问题复杂度自适应调整
- 图像→视频的知识迁移方式较粗放（简单混合），缺乏更有原则性的设计
- 规则奖励针对特定任务类型手工设计，缺少通用的视频奖励模型

## 相关工作与启发

- DeepSeek-R1证明了规则RL可激发推理能力，本文将其扩展到视频模态
- Video-UTR也发现了GRPO在视频上的shortcut问题，但用不同方法解决
- SFT memorizes, RL generalizes (Chu et al. 2025) 的观点在视频领域得到验证
- 开源所有代码、模型和数据，为社区后续研究提供了基础

## 评分

- 新颖性: ⭐⭐⭐⭐ T-GRPO的对比时序奖励设计巧妙，首次系统探索R1+视频推理
- 实验充分度: ⭐⭐⭐⭐⭐ 6个benchmark、详尽消融、训练曲线分析、时序推理占比分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题-方案对应明确
- 实用价值: ⭐⭐⭐⭐⭐ 全部开源，VSI-Bench超越GPT-4o，社区影响大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](../../ICLR2026/multimodal_vlm/vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[NeurIPS 2025\] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)
- [\[ICLR 2026\] SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](../../ICLR2026/multimodal_vlm/sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)

</div>

<!-- RELATED:END -->
