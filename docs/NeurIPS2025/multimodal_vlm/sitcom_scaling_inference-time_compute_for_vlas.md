---
title: >-
  [论文解读] SITCOM: Scaling Inference-Time COMpute for VLAs
description: >-
  [NeurIPS 2025][多模态VLM][推理时计算缩放] SITCOM 提出了一种受模型预测控制（MPC）启发的推理时计算框架，通过学习的动力学模型对预训练 VLA 进行多步rollout仿真并利用奖励模型选择最优轨迹，将单步 VLA 转化为鲁棒的长程规划器，在 SIMPLER 环境中将任务完成率从 48% 提升至 72%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "推理时计算缩放"
  - "VLA"
  - "世界模型"
  - "模型预测控制"
  - "机器人操纵"
---

# SITCOM: Scaling Inference-Time COMpute for VLAs

**会议**: NeurIPS 2025  
**arXiv**: [2510.04041](https://arxiv.org/abs/2510.04041)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 推理时计算缩放, VLA, 世界模型, 模型预测控制, 机器人操纵

## 一句话总结

SITCOM 提出了一种受模型预测控制（MPC）启发的推理时计算框架，通过学习的动力学模型对预训练 VLA 进行多步rollout仿真并利用奖励模型选择最优轨迹，将单步 VLA 转化为鲁棒的长程规划器，在 SIMPLER 环境中将任务完成率从 48% 提升至 72%。

## 研究背景与动机

机器人学习长期受制于标注数据获取成本高、泛化能力有限和长程规划困难三大挑战。视觉-语言-动作（VLA）模型通过将自然语言指令落地为控制命令取得了显著进展，但它们在实际部署中仍面临关键局限：

**缺乏前瞻能力**：VLA 本质上是单步预测模型，无法评估动作的长期后果

**累积误差**：开环执行中小误差逐步累积，导致多步任务失败

**动态环境适应性差**：无法在执行过程中根据环境变化调整计划

现有解决方案要么通过链式思维（CoT）数据进行显式推理训练（依赖昂贵的标注），要么使用世界模型但通常计算开销大且任务特定性强。

SITCOM 的切入角度：将推理时计算缩放的理念从语言模型迁移到机器人控制——不改变训练范式，而是在推理时通过并行rollout和奖励排序来"思考更多"后再行动。这类似于 MPC 的思路：在每个决策步进行前瞻仿真、评估、选择。

## 方法详解

### 整体框架

SITCOM 的推理流程：
1. **候选生成**：在每个决策步，使用高采样温度从 VLA 策略采样 n 个候选动作
2. **Rollout 仿真**：对每个候选动作，使用动力学模型预测下一状态图像，然后在该预测图像上继续用 VLA 采样动作，迭代 l 步生成完整轨迹
3. **奖励评估**：对每条轨迹的最终状态计算奖励（包含夹爪-物体距离、物体-目标距离、抓取成功指示）
4. **轨迹选择与执行**：选择最高奖励轨迹的动作序列在真实环境中执行
5. **重复**：按环境的重规划频率循环直至任务完成

提供两种 rollout 模式：
- **SITCOM (EnvSim)**：使用环境实例进行 oracle rollout（作为上界）
- **SITCOM (Dynamics)**：使用学习的动力学模型进行 rollout（实际可部署方案）

### 关键设计

1. **Transformer 动力学模型**：采用编码器-解码器架构，编码器处理图像 patch 并与动作信息拼接，解码器预测下一帧图像的 patch。使用 L1 像素损失 + LPIPS 感知损失联合训练，平衡低层精度和高层视觉连贯性。两阶段训练策略：(1) 在约 25,000 条 BridgeV2 轨迹上预训练以学习通用动力学；(2) 在 SIMPLER 环境轨迹上微调以适配目标环境视觉和物理特性，弥合 Real2Sim 差距。

2. **DAgger 风格适应策略**：初始训练模型仅预测单步前一步时表现良好，但在多步 rollout 时因累积误差导致严重目标物重建问题。借鉴 DAgger 思想，在训练阶段让模型从自身预测结果出发进行多步预测（而非始终从真值出发），使训练分布与推理时的自回归分布更一致，显著减少了长程 rollout 中的预测漂移。

3. **VLA 微调**：由于 SIMPLER 环境无公开专家数据，自行策划了约 100 条专家轨迹：先用预训练模型生成轨迹 → 启发式规则筛选成功执行 → 人工过滤确保高质量。使用标准交叉熵损失对离散化动作 token 进行微调。

### 损失函数 / 训练策略

动力学模型使用 L1 + LPIPS 复合损失：L1 确保像素级精度，LPIPS 保证感知层面的视觉真实感。奖励设计包含三个信号：夹爪-物体间隙（引导接近）、物体-目标距离（引导放置）、抓取成功指标。

默认配置：rollout 长度 10 步，5 条候选轨迹。单次动作规划时间随候选数增加线性增长（5 候选约 35 秒，25 候选约 160 秒）。

## 实验关键数据

### 主实验 - SIMPLER 环境任务成功率

| 任务 | OpenVLA | OpenVLA-SFT | SITCOM (EnvSim) | SITCOM (World Model) |
|------|---------|-------------|-----------------|---------------------|
| 放胡萝卜到盘子 | 0.0 | 0.50 | 0.71 | 0.66 |
| 放勺子到桌布 | 0.0 | 0.63 | 0.83 | 0.83 |
| 绿块叠黄块 | 0.042 | 0.17 | 0.58 | 0.62 |
| 放茄子到篮子 | 0.0 | 0.63 | 0.92 | 0.79 |
| **平均** | **0.01** | **0.48** | **0.76** | **0.72** |

### 动力学模型质量评估

| 模型 | FID↓ | OFL↓ |
|------|------|------|
| 基础模型 (仅BridgeV2) | 17.0 | 1.665 |
| **微调模型** | **11.2** | **0.992** |

### 消融实验 - 候选数量影响

| 候选数 | 1 | 5 | 10 | 15 | 20 | 25 |
|--------|---|---|----|----|----|----|
| 规划时间(s) | 21 | 35 | 75 | 100 | 130 | 160 |

### 关键发现

- 推理时计算缩放在机器人控制中有效：从 OpenVLA-SFT 的 48% 提升到 SITCOM 的 72-76%
- 学习的动力学模型（72%）与 oracle 仿真器（76%）差距仅 4%，验证了方法的可行性
- VLA 微调本身就能从 1% 提升到 48%，解决了约 40% 的 Real2Sim 差距
- 增加候选数量持续带来收益直到 25 个（某些任务更早饱和）
- 复杂任务（如放茄子到篮子）从更长 rollout 中获益更多
- DAgger 式适应显著改善了长程 rollout 中的目标物重建质量，但预测漂移问题仍未完全解决
- 主要失败模式：VLA 的 Real2Sim 差距和精细操控能力不足（如夹爪闭合时机的微小偏差导致物体滑落）

## 亮点与洞察

- 核心思想简洁有力：将 LLM 领域的"推理时计算缩放"理念迁移到机器人控制，用更多计算换取更好决策
- MPC 风格的 rollout + 排序框架通用性强，可搭配任意 VLA 策略使用
- DAgger 式训练策略巧妙地解决了自回归预测的分布偏移问题
- 对失败模式的定性分析深入且诚实——清楚指出了精细操控和 Real2Sim 差距是主要瓶颈
- 两阶段动力学模型训练（大规模预训练 + 域内微调）是实用且可复制的方案

## 局限与展望

- 奖励信号依赖 oracle 仿真器状态（假设完美的环境知识），真实世界部署时需替换为学习的奖励模型
- 确定性动力学模型难以处理随机环境，未来可探索概率性动作条件视频扩散模型
- 推理时间瓶颈严重（5 候选需 35 秒），限制了实时控制频率
- 仅在模拟环境（SIMPLER）中验证，缺乏真实机器人实验
- 动力学模型仅用成功轨迹训练，对失败状态的预测不可靠
- rollout 长度需手动设定且不同任务最优值不同，缺乏自适应机制

## 相关工作与启发

- 与 CoT 推理（ECoT、零样本标注）的显式分解不同，SITCOM 通过隐式的仿真评估来进行长程推理
- 与 Dreamer、TD-MPC2 等世界模型方法相比，SITCOM 在像素空间预测但使用更轻量的 Transformer，平衡了视觉保真度和计算效率
- 与 GAIA-1、UniSim 等生成式视频模型相比，避免了昂贵的扩散架构
- 启发方向：结合动作分块（action chunking）减少推理调用次数；开发通用的基于视觉的奖励模型

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](../../ICCV2025/multimodal_vlm/scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[ACL 2025\] Inference Compute-Optimal Video Vision Language Models](../../ACL2025/multimodal_vlm/inference_compute_optimal_video_vlm.md)
- [\[NeurIPS 2025\] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)

</div>

<!-- RELATED:END -->
