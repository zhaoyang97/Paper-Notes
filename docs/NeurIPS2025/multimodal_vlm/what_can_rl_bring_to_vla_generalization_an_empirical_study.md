---
title: >-
  [论文解读] What Can RL Bring to VLA Generalization? An Empirical Study
description: >-
  [NeurIPS 2025][多模态VLM][VLA模型] 本文系统研究了RL微调对VLA（视觉-语言-动作）模型泛化能力的影响，发现PPO是最有效的RL算法且显著优于DPO和GRPO，RL在语义理解和执行鲁棒性方面的OOD泛化远超SFT，同时在视觉鲁棒性上与SFT持平。 VLA模型将感知、语言理解和具身控制统一到一个端到端…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLA模型"
  - "强化学习"
  - "PPO"
  - "泛化性"
  - "机器人操作"
---

# What Can RL Bring to VLA Generalization? An Empirical Study

**会议**: NeurIPS 2025  
**arXiv**: [2505.19789](https://arxiv.org/abs/2505.19789)  
**代码**: [项目页](https://rlvla.github.io)  
**领域**: 多模态VLM  
**关键词**: VLA模型, 强化学习, PPO, 泛化性, 机器人操作

## 一句话总结
本文系统研究了RL微调对VLA（视觉-语言-动作）模型泛化能力的影响，发现PPO是最有效的RL算法且显著优于DPO和GRPO，RL在语义理解和执行鲁棒性方面的OOD泛化远超SFT，同时在视觉鲁棒性上与SFT持平。

## 研究背景与动机
VLA模型将感知、语言理解和具身控制统一到一个端到端框架中，通过在大规模机器人数据上训练实现跨任务泛化。然而，VLA的训练几乎完全依赖**监督微调（SFT）/行为克隆**——直接复制专家示演数据。

SFT的根本缺陷在于**分布偏移下的复合误差**：策略一旦偏离专家轨迹，就会进入训练分布外状态，误差不断累积（理论上遗憾值关于时间步二次增长），导致应用鲁棒性差。人们自然会想到用强化学习（RL）来解决——RL通过试错直接优化任务奖励，可以探索专家数据之外的状态并学习纠正行为。

在LLM和VLM领域，RL微调（如RLHF）已被证明能显著提升OOD泛化和推理能力。有研究甚至明确指出"SFT记忆，RL泛化"。RL在机器人领域也有成功先例。然而，**RL微调对VLA具体带来哪些泛化优势以及与SFT相比各自的强项在哪里**，仍缺乏系统性理解。

本文的核心研究问题非常直接：**RL能为VLA泛化带来什么独特好处？** 围绕这个问题，作者建立了一套从视觉、语义、执行三维度全面评估泛化的基准，并给出了清晰答案。

## 方法详解

### 整体框架
以OpenVLA（7B参数，SigLIP+DINOv2视觉编码器+Llama-2语言骨干）为基础模型，在ManiSkill仿真中的pick-and-place任务上对比SFT和RL（PPO/DPO/GRPO）微调，评估在视觉/语义/执行三个维度的OOD泛化性能。

### 关键设计

1. **PPO作为VLA的最优RL算法**:

    - 对比三种主流RL算法：PPO、GRPO、DPO
    - **PPO一致性优于GRPO**：原因在于机器人POMDP中每个动作都改变环境状态（非平稳），GRPO的组内优势估计在这种非平稳动态下不稳定。GRPO在LLM中有效是因为语言生成的环境（prompt→response）相对"静态"
    - **PPO优于DPO**：DPO依赖离线数据集中的成对偏好，稀疏奖励下难以区分轨迹质量，且离线数据与在线执行的分布偏移严重
    - 设计动机：直接验证LLM领域的RL算法是否能迁移到VLA，揭示了POMDP机器人任务与语言生成任务的本质差异

2. **高效PPO训练方案（3个关键设计）**:

    - **共享Actor-Critic骨干**：Actor和Critic共用整个Transformer，仅在第一个动作token位置的隐向量 $h^0$ 上接三层MLP作为值函数头，比独立Critic减少83%显存（44.4GB vs 81.3GB）且速度快35%
    - **VLA预热（Warmup）**：用140条示演轨迹对原始OpenVLA做SFT预热，使RL收敛步数减少约50%（最终性能相当）
    - **最小PPO epoch = 1**：实验发现增加PPO epoch数对收益无提升但线性增加时间，固定epoch=1实现最快训练
    - 整体方案在单张A100 GPU上约42小时收敛
    - 设计动机：VLA模型7B规模大，需要极其精简的PPO设计才能实际可行

3. **三维度泛化评估基准**:

    - **视觉**：未见桌面背景、动态前景/全图纹理覆盖（弱/强两档）
    - **语义**：未见物体、未见容器、改写指令、多物体选择、干扰容器
    - **执行**：物体/容器位置偏移、机器人初始姿态变化、执行中物体重定位
    - 训练时在16种桌面、16种物体、位置扰动上随机化；测试时至少一个因素OOD
    - 设计动机：从VLA的三个核心组件（Vision-Language-Action）出发，全面覆盖可能的分布偏移来源

### 损失函数 / 训练策略
- SFT：使用next-token交叉熵损失，动作离散化为256 bin（RT-2方案）
- PPO：裁剪代理目标 + GAE优势估计，LoRA rank=32
- 奖励设计：稀疏奖励——抓取并持续持有正确物体得0.1，成功放置得1.0
- SFT数据：由Octo-Small + 运动规划器收集（过滤了闲置动作）

## 实验关键数据

### 主实验（OOD泛化对比，成功率）

| 维度 | 任务 | SFT-16k | RL (PPO) | 提升 |
|------|------|---------|----------|------|
| 视觉 | 未见桌面 | 0.719 | 0.844 | +17% |
| 视觉 | 动态纹理(强) | 0.557 | 0.630 | +13% |
| 语义 | 未见物体 | 0.453 | **0.714** | **+58%** |
| 语义 | 未见容器 | 0.615 | 0.750 | +22% |
| 语义 | 多物体(OOD) | 0.297 | **0.578** | **+95%** |
| 执行 | 未见位置 | 0.568 | **0.807** | **+42%** |
| 执行 | 未见机器人姿态 | 0.339 | **0.797** | **+135%** |
| 执行 | 中途物体移位 | 0.286 | **0.745** | **+160%** |

### 消融实验（RL算法对比 + PPO设计因素）

| 配置 | 主要结果 | 说明 |
|------|---------|------|
| PPO vs GRPO | PPO一致优于GRPO | POMDP非平稳动态干扰GRPO优势估计 |
| PPO vs DPO | PPO显著优于DPO | 稀疏奖励下轨迹偏好难以区分 |
| 共享vs独立Critic | 共享略优 | 减少83%显存，速度快35% |
| h0 vs hn vs concat | h0最优 | 第一个动作token嵌入最信息化 |
| 有vs无Warmup | 有Warmup收敛快50% | 最终性能相当 |
| PPO epoch 1 vs 3 vs 5 | epoch=1最优 | 多epoch无收益但线性增加时间 |

### 关键发现
- **RL泛化优势的维度分布**：执行>语义>视觉。RL在执行维度上的提升最为惊人（机器人姿态+135%，中途移位+160%），语义维度有显著提升，视觉维度与SFT持平
- **RL学习了纠正行为**：可视化显示RL策略覆盖更宽的工作空间和末端执行器姿态范围，而SFT轨迹聚集在运动规划器路径附近——这是RL在执行维度泛化的关键
- **数据规模效应**：SFT性能在约16k轨迹后饱和，但RL仅需约0.4M步的在线交互就在OOD上超越SFT-16k（42.6%提升）
- **视觉鲁棒性的平等**：RL和SFT都无法超越训练时施加的视觉随机化范围，说明视觉泛化更多来自数据增强而非学习算法
- **SFT的根本局限**：面对OOD执行（如中途物体移动），SFT策略完全无法恢复——因为示演数据中从未出现过这种情况

## 亮点与洞察
- **明确回答了核心问题**：RL的泛化优势集中在语义理解和执行鲁棒性上，在视觉方面无额外增益——这对实际部署策略有重要指导意义
- **PPO方案极为精简**：共享骨干+warmup+单epoch=1的设计使7B VLA的RL训练在单A100上约42小时可行，大幅降低准入门槛
- **实验设计教科书级**：三维度×多任务的泛化评估框架可作为VLA泛化研究的标准基准

## 局限与展望
- 仅在pick-and-place任务上验证，复杂多步操作和多任务场景有待扩展
- 示演数据来自运动规划器而非人类，可能缺少人类收集数据的自然变异性
- 所有实验在仿真中进行，sim-to-real迁移（初步Franka实验RL成功率27% vs SFT 0%）仍需大规模验证
- 未探索奖励设计对RL泛化的具体影响

## 相关工作与启发
- **vs FLaRe (Hu et al.)**: FLaRe验证了PPO对VLA的可行性但未系统分析泛化，本文填补了这一空白
- **vs GRAPE (Zhang et al.)**: GRAPE使用DPO+密集奖励，本文表明在稀疏奖励下PPO更优且DPO在POMDP中困难
- **vs "SFT记忆, RL泛化"**: 本文将LLM领域的这一发现在VLA具身场景中具体化——RL的泛化优势在执行维度尤为突出

## 评分
- 新颖性: ⭐⭐⭐⭐ 研究问题重要且答案清晰，但方法上主要是系统性实验而非新算法
- 实验充分度: ⭐⭐⭐⭐⭐ 三维度×16+任务的全面评估、多RL算法对比、PPO设计消融、数据规模实验、action chunk扩展、初步真实世界验证
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、图表丰富、结论明确，研究问题和回答的对应关系一目了然
- 价值: ⭐⭐⭐⭐⭐ 为VLA社区提供了明确的训练策略指导和标准化评估框架，PPO方案的工程细节对从业者直接有用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Multimodal Foundation Models Understand Schematic Diagrams? An Empirical Study on Information-Seeking QA over Scientific Papers](../../ACL2025/multimodal_vlm/can_multimodal_foundation_models_understand_schematic_diagrams_an_empirical_stud.md)
- [\[NeurIPS 2025\] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)
- [\[ACL 2025\] Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings](../../ACL2025/multimodal_vlm/towards_storage-efficient_visual_document_retrieval_an_empirical_study_on_reduci.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent](../../ICCV2025/multimodal_vlm/gtr_guided_thought_reinforcement_prevents_thought_collapse_in_rl-based_vlm_agent.md)

</div>

<!-- RELATED:END -->
