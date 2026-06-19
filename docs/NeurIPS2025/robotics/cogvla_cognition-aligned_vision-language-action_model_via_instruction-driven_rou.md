---
title: >-
  [论文解读] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification
description: >-
  [NeurIPS 2025][机器人][VLA] CogVLA 提出模仿人类多模态认知的三阶段VLA架构（EFA-Routing视觉聚合压缩至25% + LFP-Routing LLM内指令感知剪枝50% + V-L-A耦合注意力），在LIBERO上以97.4%成功率和2.5×训练/2.8×推理加速超越OpenVLA-OFT等SOTA方法，真实机器人任务达70.0%成功率。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "VLA"
  - "token routing"
  - "sparsification"
  - "instruction-driven"
  - "robotic manipulation"
---

# CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification

**会议**: NeurIPS 2025  
**arXiv**: [2508.21046](https://arxiv.org/abs/2508.21046)  
**代码**: [https://jiutian-vl.github.io/CogVLA-page](https://jiutian-vl.github.io/CogVLA-page)  
**领域**: 机器人 / 多模态VLM  
**关键词**: VLA, token routing, sparsification, instruction-driven, robotic manipulation

## 一句话总结
CogVLA 提出模仿人类多模态认知的三阶段VLA架构（EFA-Routing视觉聚合压缩至25% + LFP-Routing LLM内指令感知剪枝50% + V-L-A耦合注意力），在LIBERO上以97.4%成功率和2.5×训练/2.8×推理加速超越OpenVLA-OFT等SOTA方法，真实机器人任务达70.0%成功率。

## 研究背景与动机

**领域现状**：VLA模型（如OpenVLA、π₀、RT-2）通过将视觉-语言-动作统一在预训练VLM上进行机器人控制，实现了端到端的操作能力。然而，将VLM适配到动作空间的后训练过程计算代价巨大——例如在LIBERO单任务上fine-tune一个7B VLA模型需要超过600 A100 GPU小时。

**现有痛点**：已有的稀疏化/加速方法（Mixture-of-Depths、层跳过、早期退出）有两个核心问题：(a) 它们仅关注LLM内部计算优化，忽略了从感知到控制的端到端跨模态语义耦合——视觉压缩可能丢弃任务关键特征，token跳过可能破坏上下文连贯性；(b) 视觉-语言-动作三模态的注意力模式本质上不同（视觉需要选择性注意、语言需要因果推理、动作需要时间连贯性），但被统一为相同的注意力策略。

**切入角度**：从人类认知科学中汲取灵感——人类操作物体时有高度优化的多模态协调机制：视觉注意系统（VAS）选择性聚焦任务相关目标→辅助运动区（SMA）注入动作意图过滤无关信息→前运动皮层（PMC）动态整合产生连贯动作轨迹。这三阶段对应CogVLA的EFA-Routing→LFP-Routing→CAtten。**核心idea**：指令驱动的跨模态渐进式稀疏化——不是盲目压缩，而是根据任务指令在每个阶段选择性保留最相关的信息。

## 方法详解

### 整体框架
CogVLA在标准VLA pipeline（视觉编码器→LLM→动作输出）中嵌入三阶段渐进式稀疏化：Stage 1在视觉编码器中做指令引导的跨分支聚合（25%压缩）；Stage 2在LLM中做指令引导的token剪枝（50%稀疏化）；Stage 3用hybrid注意力掩码确保V-L用因果注意力、Action用双向注意力。动作通过parallel decoding一次性生成整个action chunk。

### 关键设计

1. **EFA-Routing（Encoder-FiLM based Aggregation Routing）**:

    - 功能：在视觉编码器内部基于任务指令聚合和压缩视觉token至原始规模的25%
    - 核心思路：两步聚合——(a) Intra-encoder Aggregation：通过Encoder-FiLM模块将指令embedding转化为scale/shift向量(γ,β)调制每个编码器分支（SigLIP和DINOv2）内的Self-Attention输出，引入可学习的aggregation token逐层聚合指令相关信息，最终仅保留aggregation token丢弃原始image token（压缩至25%）；(b) Cross-encoder Aggregation：通过指令条件的routing gate（MLP→Sigmoid）动态计算SigLIP和DINOv2两个分支的融合权重α——不同指令对语义(SigLIP)vs空间(DINOv2)特征的需求不同
    - 设计动机：双编码器（语义+空间）是必要的但产生冗余token。FiLM调制是一种轻量级的条件化方式，比cross-attention更高效。指令条件的动态融合避免了固定50/50比例带来的信息损失

2. **LFP-Routing（LLM-FiLM based Pruning Routing）**:

    - 功能：在LLM的每一层中基于指令感知剪枝50%的视觉token，减少注意力计算量
    - 核心思路：在每个Transformer层l，先通过LLM-FiLM对视觉token做指令条件的调制（γ_LLM, β_LLM），然后通过Task-Guided Pruning Router（MLP）为每个token计算routing weight R_l^j。设定保留率β，计算当前层routing weight的β分位数作为阈值——超过阈值的token正常计算Self-Attention+FFN，低于阈值的token直接skip（原值传递）。被保留的token通过其routing weight进行加权
    - 设计动机：EFA-Routing虽然压缩了token数量，但聚合过程可能仍保留了与当前LLM计算无关的语义信息。LFP-Routing在更深层进一步过滤——模拟人类SMA将动作意图注入视觉处理流的功能

3. **V-L-A Coupled Attention (CAtten)**:

    - 功能：在压缩后的多模态输入上保持跨模态逻辑一致性和动作时间连贯性
    - 核心思路：设计hybrid注意力掩码M_hybrid：(a) 视觉-语言区域用因果注意力M_causal^VL（保持序列推理能力，视觉token已包含指令意图所以语言看不到视觉=合理）；(b) 动作token内部用双向注意力M_bi^act（action chunk内的所有token互相可见，实现parallel decoding——一次前向生成K步动作而非K×D次自回归）；(c) 动作token可以看到V-L的所有token（获取完整上下文），但V-L看不到动作（因果方向）
    - 设计动机：标准因果注意力在稀疏化后的VLA中会导致动作生成不连贯（action token 2看不到token 1的信息），双向注意力让所有action token共享信息确保时间一致性。同时parallel decoding将推理从K×D次前向减少为1次

### 损失函数 / 训练策略
使用standard action prediction loss（MSE或token classification loss）训练。4×A800 GPU训练，因为稀疏化后训练成本仅4.7h/10k steps（OpenVLA需12.5h/10k steps）。

## 实验关键数据

### 主实验（LIBERO Benchmark）

| 方法 | Spatial SR | Object SR | Goal SR | Long SR | Avg SR | 排名 |
|------|-----------|-----------|---------|---------|--------|------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 9 |
| π₀ fine-tuned | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | 5 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | 2 |
| PD-VLA | 95.5 | 96.7 | 94.9 | 91.7 | 94.7 | 3 |
| **CogVLA** | **98.6** | **98.8** | **96.6** | **95.4** | **97.4** | **1** |

| 方法 | 推理时间↓ | 吞吐量↑ | FLOPs↓ | 训练成本/10k步↓ | SR |
|------|---------|--------|--------|---------------|------|
| OpenVLA | 0.254s | 3.9Hz | 8.48T | 11.7h | 76.5% |
| OpenVLA-OFT | 0.132s | 60.6Hz | 8.45T | 12.5h | 97.1% |
| **CogVLA** | **0.091s** | **87.9Hz** | **2.72T** | **4.7h** | **97.4%** |

### 消融实验

| 配置 | 推理时间 | FLOPs | 说明 |
|------|---------|-------|------|
| Full CogVLA | 0.091s | 2.72T | 完整方法 |
| w/o Stage 1 (EFA-Routing) | 0.162s | 5.38T | 视觉token未压缩→FLOPs翻倍 |
| w/o Stage 2 (LFP-Routing) | 0.117s | 3.52T | LLM内无剪枝→计算增加 |

### 真实机器人实验（Cobot Agilex ALOHA）

| 方法 | 物体放置 | 抽屉操作 | T恤折叠 | 平均SR |
|------|---------|---------|---------|--------|
| OpenVLA-OFT | 7/10→5/10 | 8/10→5/10 | 7/10→5/10 | 56.7% |
| PD-VLA | 8/10→4/10 | 6/10→4/10 | 7/10→4/10 | 50.0% |
| **CogVLA** | **9/10→7/10** | **8/10→7/10** | **9/10→6/10** | **70.0%** |

### 关键发现
- CogVLA在性能和效率上同时达到SOTA——97.4% SR排名第一，同时FLOPs仅为OpenVLA的32%
- 真实机器人实验（70.0% vs OFT 56.7%）验证了sim-to-real迁移能力，尤其在长程任务（T恤折叠3步）上优势明显
- 75%视觉token+50%LLM token可以被安全移除而不损失甚至提升性能——大量token确实与任务无关
- Stage 1和Stage 2的贡献互补：Stage 1主要降FLOPs（5.38T→2.72T），Stage 2主要降推理时间

## 亮点与洞察
- **认知科学启发的三阶段设计**（VAS→SMA→PMC）不只是metaphor——实际对应了信息处理中"选择→过滤→协调"的合理计算流
- **指令驱动是关键**：FiLM调制和routing gate都以任务指令为条件，实现了"根据你要做什么来决定看什么、想什么"——这比无条件压缩（如ViT的token merging）更有效
- **parallel decoding + 双向action注意力**是VLA效率提升的重要方向——autoregressive生成K步动作的延迟被消除
- 87.9Hz吞吐量意味着在实际机器人控制中（通常需要10-50Hz）绰绰有余

## 局限与展望
- 仅在LIBERO（10任务×4套=40任务）和3个真实任务上验证，任务多样性有限
- 压缩率（25%视觉+50%LLM剪枝）是固定的，不同任务复杂度可能需要不同压缩率——自适应压缩率是明显的改进方向
- FiLM调制使用MLP生成，引入的参数量和计算虽然轻量但非零——在更大规模VLA上的扩展性待验证
- 双向action注意力假设action chunk内的动作可以独立并行生成，但高度依赖前序动作的精细操作可能需要时间因果性

## 相关工作与启发
- **VLA效率化的趋势**：从OpenVLA(3.9Hz)到OFT(60.6Hz)再到CogVLA(87.9Hz)——效率提升是VLA实用化的关键路径
- **FiLM调制在跨模态条件化中的复兴**：FiLM（Feature-wise Linear Modulation）最初用于视觉问答，在VLA领域重新获得价值——轻量级、可端到端训练、不改变主干结构
- **token稀疏化的任务感知方向**：CogVLA证明"根据任务删token"远优于"根据attention score删token"——任务语义应该是稀疏化的核心依据

## 评分
- 新颖性: ⭐⭐⭐⭐ 认知启发的三阶段指令驱动稀疏化有创意，但各组件（FiLM/token pruning/parallel decoding）非全新
- 实验充分度: ⭐⭐⭐⭐ LIBERO+真实机器人+效率对比+消融，但任务种类仍有限
- 写作质量: ⭐⭐⭐⭐ 架构描述清晰，认知科学类比有启发性
- 价值: ⭐⭐⭐⭐ 对VLA的高效部署有直接实用价值，87.9Hz吞吐量使实时控制成为可能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](../../AAAI2026/robotics/semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)
- [\[ICML 2025\] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](../../ICML2025/robotics/hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](../../ICML2026/robotics/dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)

</div>

<!-- RELATED:END -->
