---
title: >-
  [论文解读] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons
description: >-
  [CVPR 2025][多模态VLM][通用具身智能体] GEA 将预训练的多模态 LLM（LLaVA-OneVision）通过学习式多具身动作分词器适配到操控/导航/游戏/UI控制/规划五大领域，先用 220 万条跨域专家轨迹 SFT，再用在线 PPO 强化学习微调，单模型在多个基准上超越或接近领域专用模型。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "通用具身智能体"
  - "多模态LLM微调"
  - "强化学习"
  - "跨域迁移"
  - "动作分词器"
---

# From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons

**会议**: CVPR 2025  
**arXiv**: [2412.08442](https://arxiv.org/abs/2412.08442)  
**代码**: 待开源 (Apple)  
**领域**: 多模态VLM / Agent / Embodied AI  
**关键词**: 通用具身智能体、多模态LLM微调、强化学习、跨域迁移、动作分词器

## 一句话总结
GEA 将预训练的多模态 LLM（LLaVA-OneVision）通过学习式多具身动作分词器适配到操控/导航/游戏/UI控制/规划五大领域，先用 220 万条跨域专家轨迹 SFT，再用在线 PPO 强化学习微调，单模型在多个基准上超越或接近领域专用模型。

## 研究背景与动机

**领域现状**：当前基于 MLLM 的具身智能体通常针对单一领域训练——OpenVLA 做机器人操控，SeeClick/CogAgent 做 UI 导航，各模型无法跨域。虽然这些领域共享很多相似性（视觉推理、长序列决策、部分可观测性），但还没有一个模型能同时胜任所有场景。

**现有痛点**：(1) 不同领域的动作空间差异巨大（末端执行器 7-DoF vs 离散导航指令 vs 屏幕点击坐标），直接统一困难；(2) SFT 只能学到专家演示的行为，缺乏错误恢复能力，在交互式任务中会因 covariate shift 导致错误累积；(3) Gato 等先驱模型没有利用预训练 MLLM 的知识，也没有结合在线 RL，泛化能力有限。

**核心矛盾**：跨域数据的动作空间和环境差异如此之大，简单混合会互相干扰；但分开训练又丧失了跨域迁移的收益。

**本文目标** 如何设计统一的动作表示 + 训练策略，让一个 MLLM 成为跨越数字和物理世界的通用具身智能体？

**切入角度**：用 Residual VQ-VAE 学习跨具身的连续动作分词器，把不同机器人的动作都编码成 LLM 词表中的 token；离散动作直接用自然语言表示。这样所有域的动作预测都归结为 next-token prediction。再加上在线 RL 解决 SFT 的 covariate shift 问题。

**核心 idea**：用学习式动作分词器统一异构动作空间 + SFT 跨域微调 + 在线 PPO 强化学习，把 MLLM 变成通用具身智能体。

## 方法详解

### 整体框架
GEA 基于 LLaVA-OneVision 7B，输入为环境提示（描述具身类型）+ 任务指令 + 最近 c=3 帧视觉观测和历史动作，输出为动作 token 序列。训练分两阶段：Stage 1 在 220 万条跨域轨迹上做标准自回归 SFT 得到 GEA-Base；Stage 2 在可交互模拟器中用 PPO 在线 RL + 持续 SFT 得到最终 GEA。

### 关键设计

1. **多具身动作分词器（Multi-Embodiment Action Tokenizer）**:

    - 功能：将不同具身的连续/离散动作统一编码为 LLM 的 token 序列
    - 核心思路：对离散动作（如导航指令 "right"、UI 操作 "tap"）直接用自然语言分词；对连续动作（如末端执行器位移、关节速度），训练一个 Residual VQ-VAE (RVQ)，用 2 个 codebook 各 512 token 把任意维度的连续动作向量编码为 2 个离散 token。不同具身的动作向量 pad 到最大维度统一输入 RVQ，推理时截断到目标具身的维度。用 LLM 词表中最不常用的 token 替换为 RVQ 的 token
    - 设计动机：之前的均匀离散化方法（如 OpenVLA 用 256 bins）精度有限且无法跨具身；RVQ 通过层级残差编码，用极少 token 就能精确表示连续动作，且单一分词器适配所有具身消除了跨域冲突

2. **两阶段训练：SFT + 在线 PPO**:

    - 功能：先学基本策略，再通过在线交互修复 covariate shift 并学习错误恢复行为
    - 核心思路：Stage 1 用标准交叉熵 loss 在 220 万条跨域轨迹上全参数微调 MLLM（75k 更新，batch=256）；Stage 2 冻结非 LLM 部分，用 LoRA 微调 LLM，损失函数为 $\mathcal{L}_{GEA} = \sum_{i \in \mathcal{E}_{PPO}} \mathcal{L}_{PPO}(\mathcal{M}_i) + \lambda \sum_{i \in \mathcal{E}} \mathcal{L}_{SFT}(\mathcal{D}_i)$，其中 $\lambda=0.1$。RL 环境包括 Habitat Pick、LangR、Procgen 三个域
    - 设计动机：纯 SFT 只能模仿专家，遇到自身犯错产生的 OOD 状态就会崩溃（covariate shift）。实验表明 Success SFT 和 Offline RL 都不如在线 PPO 有效。同时保留 SFT loss 防止非 RL 域的性能退化

3. **跨域数据混合与 PopArt 归一化**:

    - 功能：让模型同时从操控、导航、游戏、UI 等不同域的数据中受益
    - 核心思路：训练数据包含 12 个数据集（OpenX 120 万轨迹、Meta-World 4.5 万、CALVIN 1.8 万、Procgen 32 万等），总计 220 万+ 轨迹。RL 阶段用 PopArt 对不同环境的 reward 分布做归一化，用 constrained decoding 限制每个环境只输出合法动作 token，用归一化 entropy 系数统一 PPO 超参
    - 设计动机：消融实验证明跨域训练在所有域上都优于单域训练（平均 +3-6%），说明不同具身任务之间确实存在正向迁移。PopArt 和 constrained decoding 是 RL 训练稳定的关键

### 损失函数 / 训练策略
Stage 1：标准自回归 cross-entropy loss on actions，学习率 1e-5，AdamW + cosine decay，8 节点 64 卡 H100 训 2 天。Stage 2：PPO loss（clipped surrogate objective）+ 0.1 权重的 SFT loss，学习率 3e-4（LoRA），entropy 系数 1e-4，8 节点 64 卡训 1 天（100M 累计步）。

## 实验关键数据

### 主实验

| 任务 | 指标 | GEA | 之前最佳 | 说明 |
|--------|------|------|----------|------|
| Meta-World (45 tasks) | 成功率 | **94.7%** | 87.0% (Gato) | 超 Gato +7.7% |
| CALVIN (ABC→D) | 成功率 | **90.0%** | 82.4% (MLLM+IL) | 接近专用方法 92.2% |
| Habitat Pick (20 scenes) | 成功率 | **82.5%** | 81.0% (RL+state) | 不用 state 即超 RL 专家 |
| Procgen (16 games) | 专家分比例 | **44.0%** | 25% (专用方法) | 超专用方法 +19% |
| AndroidControl (35 tasks) | 成功率 | **57.3%** | 45% (GPT-4o+SoM) | 超 GPT-4o +12.3% |
| BabyAI (17 tasks) | 成功率 | 91.1% | 93.2% (Gato) | 接近且用更少数据 |

### 消融实验

| 配置 | Habitat Pick | CALVIN | Procgen | AndroidControl | BabyAI |
|------|---------|---------|---------|---------|---------|
| GEA-Base (all domains) | 57.0 | 48.0 | 24.5 | 50.5 | 84.7 |
| Domain Specific only | 54.5 | 35.5 | 23.7 | 48.9 | 82.1 |
| Only LLM init | 9.5 | 0.0 | 7.6 | 26.4 | 49.4 |
| Only ViEncoder init | 34.5 | 13.0 | 24.5 | 28.3 | 70.6 |
| No pretrained weights | 9.0 | 0.0 | 7.4 | 14.1 | 44.4 |

### 关键发现
- **跨域数据带来一致性提升**：所有域上多域训练都优于单域训练，CALVIN 的提升最显著（+12.5%），说明操控数据的互相迁移效果最强
- **在线 RL 远胜其他方法**：Habitat Pick 上 SFT 60.5% → 在线 PPO 82.5%，而 Success SFT 和 Offline IQL 反而降分。这验证了在线交互对克服 covariate shift 的重要性
- **视觉编码器初始化比 LLM 更重要**：只有视觉编码器预训练时性能远高于只有 LLM 预训练，说明视觉泛化是瓶颈
- **模型规模效应一致**：从 0.5B 到 7B，所有域的性能持续提升，且不同 MLLM backbone（LLaVA-OV vs MM1.5）差异很小，说明预训练数据质量比架构选择更重要

## 亮点与洞察
- **RVQ 动作分词器是比均匀离散化更优雅的方案**：只需 2 个 token 就能精确表示任意维度的连续动作，且自然支持跨具身，解决了困扰 VLA 模型的核心工程问题
- **在线 RL 对具身智能体不可或缺**：这篇论文用严格实验证明了 SFT-only 的天花板，以及 Success SFT、Offline RL 的无效性。这个结论对所有想用 MLLM 做 agent 的工作都有指导意义
- **跨域正迁移的实证证据**：不是手动设计的辅助任务，而是简单混合不同域数据就能带来提升，这暗示了不同具身任务之间共享底层能力（空间推理、物体关系理解等）

## 局限与展望
- 只用 3 帧历史观测作为上下文，在部分可观测任务（导航）上表现受限，需要更长的上下文或显式记忆
- RL 只在 3 个域上做了（Habitat Pick、LangR、Procgen），其他域如 Maniskill 和 AndroidControl 表现还不够好，扩展 RL 到更多环境可能进一步提升
- 无法零样本控制全新具身——仍需该具身类型的数据训练动作分词器
- Apple 未开源具体训练数据和模型权重的最终时间线

## 相关工作与启发
- **vs Gato**: Gato 不用预训练 MLLM、不用 RL、用均匀离散化；GEA 在 Meta-World 上超 Gato 7.7%，在 Atari 上超 1.7%，证明 MLLM 预训练 + 学习式分词 + RL 的价值
- **vs OpenVLA**: OpenVLA 只做机器人操控、用均匀离散化、不用 RL；GEA 的学习式分词器更精确，且跨域训练带来额外增益
- **vs Magma**: Magma 用 SoM/ToM 桥接不同域（不需要学习分词器），但不用 RL；GEA 的 RL 阶段在交互式任务上优势明显。两者的设计思路互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 单个模块不算新，但整体系统设计和跨域 RL 的规模是首次
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个数据集、10 个评测基准、大量消融分析（数据、模型、RL策略），极其充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法和实验的 lessons 提炼到位
- 价值: ⭐⭐⭐⭐ 为构建通用具身智能体提供了清晰的 recipe 和重要的实证结论

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices](multi-layer_visual_feature_fusion_in_multimodal_llms_methods_analysis_and_best_p.md)
- [\[CVPR 2025\] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[CVPR 2025\] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)
- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)

</div>

<!-- RELATED:END -->
