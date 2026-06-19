---
title: >-
  [论文解读] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning
description: >-
  [NeurIPS 2025][机器人][视觉语言动作模型] AutoVLA 将物理动作 token 直接集成到预训练 VLM（Qwen2.5-VL-3B）中，通过 SFT 赋予模型快/慢双思维模式能力，再用 GRPO 强化微调实现自适应推理切换并优化规划性能，在 nuPlan、Waymo、nuScenes 和 CARLA 四大自动驾驶基准上取得有竞争力的端到端驾驶性能。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "视觉语言动作模型"
  - "端到端自动驾驶"
  - "动作token化"
  - "强化微调"
  - "自适应推理"
---

# AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2506.13757](https://arxiv.org/abs/2506.13757)  
**代码**: [https://autovla.github.io/](https://autovla.github.io/)  
**领域**: 自动驾驶 / 多模态VLM  
**关键词**: 视觉语言动作模型, 端到端自动驾驶, 动作token化, 强化微调, 自适应推理

## 一句话总结

AutoVLA 将物理动作 token 直接集成到预训练 VLM（Qwen2.5-VL-3B）中，通过 SFT 赋予模型快/慢双思维模式能力，再用 GRPO 强化微调实现自适应推理切换并优化规划性能，在 nuPlan、Waymo、nuScenes 和 CARLA 四大自动驾驶基准上取得有竞争力的端到端驾驶性能。

## 研究背景与动机

**领域现状**：端到端自动驾驶用统一模型直接从传感器到动作，避免模块化的误差累积。近期 VLM 的世界知识和推理能力使 VLA 模型成为研究热点，但现有方案在动作生成的物理可行性和推理效率上存在根本问题。

**现有痛点**：(1) **动作生成不可行**：一些方法（EMMA、OpenEMMA）直接用 VLM 生成文本格式的 waypoint 坐标，但语言模型在精确数值推理上有固有局限，导致轨迹物理不可行或模态坍缩。为此引入 meta-action（AlphaDrive、SENNA）或 latent token（CarLLaVa、Orion）作为中间表示，但破坏了端到端优化或增加了复杂度。(2) **推理策略僵化**：大多数方法采用固定推理深度——要么总是做冗长 CoT（直行场景浪费10秒+），要么跳过推理（复杂场景性能差）。DriveVLM 有双模式但用两个独立模型。

**核心矛盾**：如何在单一自回归模型中同时实现高质量的语义推理和物理可行的轨迹规划，且根据场景复杂度自适应调节推理深度。

**本文目标** (1) 将物理约束的轨迹规划直接嵌入 LM 的 token 空间；(2) 让模型在快思维（~1s）和慢思维（~10s）之间自动切换；(3) 在训练后阶段用 RL 进一步对齐规划性能。

**切入角度**：从 Waymo 真实驾驶数据中用 K-disk 聚类构建 2048 个物理动作 token 的码本，每个 token 对应 0.5 秒的合理车辆运动 $(\Delta x, \Delta y, \Delta\theta)$，将规划转化为 next-token prediction。

**核心 idea**：在 VLM 中加入基于 K-disk 聚类的物理动作码本实现可行轨迹的 token 级生成，并用 GRPO + CoT 长度惩罚实现自适应快/慢推理切换。

## 方法详解

### 整体框架

AutoVLA 以 Qwen2.5-VL-3B 为骨干，接收三个前方摄像头的 4 帧图像序列（2Hz）、高层导航指令和自车状态作为输入。输出是文本 token（推理内容）+ 物理动作 token（轨迹）的混合序列。训练两阶段：**SFT** 用带/不带 CoT 的混合数据训练双模式能力；**RFT** 用 GRPO + 驾驶奖励 + CoT 长度惩罚优化性能和效率。

### 关键设计

1. **物理动作 Token 化**:

    - 功能：将连续车辆轨迹离散化为语言模型可原生处理的 token 序列
    - 核心思路：从 Waymo 真实轨迹提取 0.5s 运动片段，用 K-disk 聚类（$\delta=0.05$m 距离阈值）选出 K=2048 个代表性 token，每个编码 $(\Delta x, \Delta y, \Delta\theta)$。作为新词汇加入 VLM（`<action_0>` 到 `<action_2047>`）。推理时生成 10 个 token 解码为 5 秒轨迹。重建精度 ADE=0.018m，运动覆盖率 99.42%，远优于 RT-1（ADE=0.101m）和 FAST DCT（ADE=0.028m）
    - 设计动机：文本数值生成存在精度差+物理不可行+计算慢的问题；动作 token 天然满足运动学约束，把规划变成 next-token prediction 完美契合 LM 范式

2. **双模式自适应推理（快/慢思维）**:

    - 功能：根据场景复杂度自适应切换推理深度
    - 核心思路：SFT 用两类数据混合训练——**快思维**数据仅含动作 token（简短模板说明不需要推理），**慢思维**数据含结构化 CoT（场景描述→关键物体→意图预测→决策）+ 动作 token。CoT 数据通过 Qwen2.5-VL-72B 自动标注，人工抽检准确率 88.8%。SFT 损失对 CoT 样本赋予高权重（$\lambda_{cot}=40$）
    - 设计动机：快思维 ~1.07s，慢思维 ~10.52s，10倍差距使固定策略在实际部署中不可接受。数据量超过 50k 时 CoT 才开始优于纯动作训练（训练集小时 CoT 反而有害）

3. **GRPO 强化微调**:

    - 功能：进一步优化规划性能并实现自适应快/慢切换
    - 核心思路：对每个场景采样 G 组输出，计算组内相对优势 $A_i = (r_i - \text{mean}) / \text{std}$。奖励 $r = r_{Driving} - \lambda_r r_{CoT}$，驾驶奖励用 PDMS（nuPlan）或 ADE（Waymo），CoT 长度惩罚抑制简单场景的冗余推理。使用 LoRA 适配器，学习率 $3\times10^{-5}$，$\beta=0.04$，训练 6000 步
    - 设计动机：SFT 模型可能因生成误差累积产生次优轨迹；GRPO 的组采样天然适配规划的多模态性（同场景多条合理轨迹），CoT 惩罚使模型学会"该想时想、不该想时直接行动"

### 损失函数 / 训练策略

SFT 阶段：$\mathcal{L}^{SFT} = w_i \cdot (\mathcal{L}_{LM} + \lambda_a \mathcal{L}_{action})$，$\lambda_a=1$，$\lambda_{cot}=40$。RFT 阶段使用 GRPO 目标函数（PPO clip + KL散度正则）。SFT 用 FSDP 8×L40S 训练 5 epoch，batch=32。CARLA 实验用独立码本（因模拟器动力学不同）。

## 实验关键数据

### 主实验

| 基准 | 指标 | AutoVLA (Post-RFT) | AutoVLA (Best-of-N) | TrajHF | Centaur |
|------|------|------|------|------|------|
| NAVSIM (nuPlan) | PDMS↑ | 89.11 | **92.12** | 93.95 | 92.10 |
| NAVSIM | Collision↑ | 98.41 | 99.14 | 99.30 | 99.23 |
| NAVSIM | TTC↑ | **98.04** | 97.12 | 98.02 | 97.17 |
| Bench2Drive (CARLA) | Driving Score↑ | **78.84** | - | - | - |
| Bench2Drive | Success Rate↑ | **57.73%** | - | - | - |

### 消融实验

| 配置 | PDMS↑ | 运行时间 | 说明 |
|------|-------|---------|------|
| SFT (动作only) | 79.28 | 1.07s | 无推理，快但性能受限 |
| SFT (CoT) | 80.54 | 10.52s | 总做推理，慢但性能好 |
| Post-RFT | **89.11** | **3.49s** | 自适应切换 |
| 提升 vs SFT CoT | **+10.6%** | **-66.8%** | 性能和效率同时大幅提升 |

| 动作生成方式 | PDMS↑ | 平均L2↓ | 碰撞率↓ | 运行时间 |
|-------------|-------|---------|---------|---------|
| 文本Waypoint | 71.31 | 0.89m | 0.36% | 7.65s |
| FAST (DCT) | 67.63 | - | - | - |
| K-disk Token (本文) | **80.54** | **0.70m** | **0.31%** | **3.95s** |

### 关键发现

- **RFT 是最大的性能飞跃**：PDMS 提升 10.6%（80.54→89.11）同时运行时间减少 66.8%（10.52s→3.49s）。核心机制是 CoT 长度惩罚使模型在简单场景自动切换到快思维
- 物理动作 token 化 vs 文本 waypoint：PDMS 80.54 vs 71.31，运行时间 3.95s vs 7.65s，全面碾压。语言模型在精确数值推理上的固有局限被量化了
- 数据量越大 CoT 优势越明显：<50k 训练样本时 CoT 不如纯动作训练（学习结构化推理需要足够数据）
- 预训练 + 跨数据集训练对 Waymo 长尾场景帮助显著（RFS 分数从 56.5 提升到 64.8）
- 闭环测试（CARLA）中 AutoVLA 以 57.73% 成功率超过 Orion（54.62%），验证了方案在交互环境中的有效性

## 亮点与洞察

- **动作码本是 VLA for Driving 的关键缺失环节**：K-disk 聚类在 K=2048 时运动覆盖率 99.42%，说明 2048 个 token 即可描述绝大部分驾驶行为。这种"连续空间→离散 token"的方案保留了物理可行性，同时完全融入 LM 的自回归范式
- **GRPO + CoT 惩罚 = "该想则想"**：这种自适应推理思路非常实用——直行不需 CoT，遇施工区确实需要。10x 的运行时间差距让固定推理策略不可接受
- **72B→3B 的知识蒸馏流水线**：用 Qwen2.5-VL-72B 自动生成推理标注、3B 学习的方案可复用，人工验证准确率 88.8% 证明了可靠性

## 局限与展望

- 推理速度仍远低于实时（快模式 ~1Hz，慢模式 ~0.1Hz），作者提到量化和模型压缩是优先方向
- 仅用前方三个摄像头，未覆盖后方和侧后方，限制了全面感知（如倒车、变道检查盲区）
- 动作码本从特定数据集聚类，跨车型/跨域迁移需要重新构建
- Best-of-N 需要 oracle scorer，实际部署中缺乏 ground truth——需要替代的自评估机制
- CoT 推理质量完全依赖蒸馏源模型（72B），无法生成超出蒸馏源能力的推理

## 相关工作与启发

- **vs EMMA/OpenEMMA**: 用 VLM 直接生成文本 waypoint，物理不可行+模态坍缩。AutoVLA 的动作 token 化从根本上解决了这两个问题
- **vs DriveVLM**: 也有快/慢双过程，但用两个独立模型且无法端到端优化。AutoVLA 在单一模型内实现自适应切换
- **vs AlphaDrive**: 同用 GRPO，但仅处理高层 meta-action。AutoVLA 将 RFT 扩展到底层轨迹规划
- **vs Orion**: 集成生成式规划器到 VLM，增加了复杂度。AutoVLA 通过动作码本达类似效果但结构更简单

## 评分

- 新颖性: ⭐⭐⭐⭐ 物理动作 token + 自适应推理 + GRPO 微调的组合在自动驾驶 VLA 中属首创
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 nuPlan/Waymo/nuScenes/CARLA 四大基准，开环+闭环，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，对比范式图（Fig.2）一目了然
- 价值: ⭐⭐⭐⭐ 为 VLA 在自动驾驶中的应用提供了实用统一框架，动作 token 化和自适应推理思路有广泛影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[CVPR 2025\] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](../../CVPR2025/robotics/tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)
- [\[CVPR 2026\] End-to-End Language-Action Model for Humanoid Whole Body Control](../../CVPR2026/robotics/end-to-end_language-action_model_for_humanoid_whole_body_control.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](profit_a_specialized_optimizer_for_deep_fine_tuning.md)
- [\[CVPR 2025\] Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](../../CVPR2025/robotics/reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)

</div>

<!-- RELATED:END -->
