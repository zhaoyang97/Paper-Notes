---
title: >-
  [论文解读] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models
description: >-
  [ICML2025][机器人][世界模型] 提出 SENSEI 框架：利用 VLM 成对比较观测图像的"有趣程度"，蒸馏出语义内在奖励，再与集成不确定性驱动的新颖性奖励结合，通过世界模型实现语义有意义的无任务探索，并显著加速下游任务学习。 内在动机是强化学习（RL）中实现自主探索的核心研究方向。传统的内在奖励设计（如信息增益…
tags:
  - "ICML2025"
  - "机器人"
  - "世界模型"
  - "内在动机"
  - "VLM引导探索"
  - "语义奖励蒸馏"
  - "Model-based RL"
  - "DreamerV3"
---

# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models

**会议**: ICML2025  
**arXiv**: [2503.01584](https://arxiv.org/abs/2503.01584)  
**代码**: [项目主页](https://sites.google.com/view/sensei-paper)  
**领域**: 世界模型 / 内在探索  
**关键词**: 世界模型, 内在动机, VLM引导探索, 语义奖励蒸馏, Model-based RL, DreamerV3

## 一句话总结

提出 SENSEI 框架：利用 VLM 成对比较观测图像的"有趣程度"，蒸馏出语义内在奖励，再与集成不确定性驱动的新颖性奖励结合，通过世界模型实现语义有意义的无任务探索，并显著加速下游任务学习。

## 研究背景与动机

内在动机是强化学习（RL）中实现自主探索的核心研究方向。传统的内在奖励设计（如信息增益、预测误差、状态覆盖）虽然具有通用性，但往往只能发现低层次交互，无法高效解锁语义上有意义的行为。例如，一个机械臂面对桌面物体时，传统方法可能只会做无意义的手臂伸展，而不会主动去开抽屉或抓取物体。

近年来 LLM/VLM 给探索注入"人类先验"提供了新可能，但现有方法存在明显局限：

- **Motif** 依赖语言嵌入环境，需要事件字幕
- **ELLM** 需要文本形式的状态表示
- **OMNI** 需要高层动作空间
- 这些方法都**没有学习内部的"有趣性模型"**，持续依赖外部 LLM 指导

SENSEI 的动机：能否让基于 **图像观测 + 低层动作** 的 Model-based RL 智能体，通过 VLM 反馈学会一个内化的"什么是有趣的"模型，从而在无任务阶段自主探索语义有意义的行为？

## 方法详解

SENSEI 分为三个核心模块：奖励蒸馏、世界模型学习、探索策略优化。

### 1. 语义奖励蒸馏（VLM-Motif）

**预训练阶段**：利用 Plan2Explore 收集初始数据集 $\mathcal{D}^{\text{init}}$，然后用 VLM（GPT-4）对图像对进行成对比较，判断哪张更"有趣"。

标注函数定义：

$$\text{VLM}: \mathcal{O} \times \mathcal{O} \to \mathcal{Y}, \quad \mathcal{Y} = \{1, 2, \emptyset\}$$

基于偏好对，用交叉熵损失训练奖励模型 $R_\psi: \mathcal{O} \to \mathbb{R}$，将 VLM 的语义偏好蒸馏为可微奖励函数。探索时智能体获得语义奖励：

$$r_t^{\text{sem}} \leftarrow R_\psi(\boldsymbol{o}_t)$$

### 2. 世界模型（RSSM + 语义奖励预测头）

基于 DreamerV3 的 RSSM 架构，包含随机状态 $\boldsymbol{z}_t$ 和确定性记忆 $\boldsymbol{h}_t$：

- **后验**: $\boldsymbol{z}_t \sim q_\phi(\boldsymbol{z}_t \mid \boldsymbol{h}_t, \boldsymbol{o}_t)$
- **动力学**: $\boldsymbol{h}_{t+1} = f_\phi(\boldsymbol{a}_t, \boldsymbol{h}_t, \boldsymbol{z}_t)$
- **先验**: $\hat{\boldsymbol{z}}_{t+1} \sim p_\phi(\hat{\boldsymbol{z}}_{t+1} \mid \boldsymbol{h}_{t+1})$

**关键扩展**：新增语义奖励预测头 $\hat{r}_t^{\text{sem}}$，使世界模型能在想象（imagination）中预测语义有趣性，无需实际查询 VLM。

同时训练集成预测器（N 个模型）预测下一隐状态，利用**集成分歧**度量认知不确定性：

$$r_t^{\text{dis}} = \frac{1}{J} \sum_{j=1}^{J} \text{Var}(\hat{z}_{j,t}^n)$$

### 3. Go-and-Explore 探索策略

核心思想：在无趣状态下主要追求"有趣性"（go），到达有趣状态后切换为追求"新颖性"（explore）。通过自适应阈值实现切换：

$$r_t^{\text{expl}} = \hat{r}_t^{\text{sem}} + \begin{cases} \beta^{\text{explore}} \cdot r_t^{\text{dis}}, & \text{if } \hat{r}_t^{\text{sem}} \geq Q_k(\hat{r}^{\text{sem}}) \\ \beta^{\text{go}} \cdot r_t^{\text{dis}}, & \text{otherwise} \end{cases}$$

其中 $Q_k$ 为 $\hat{r}^{\text{sem}}$ 的第 $k$ 分位数（移动平均估计），$\beta^{\text{explore}} > \beta^{\text{go}}$。这种设计避免了纯语义奖励导致的局部最优困境。

### 损失函数

- 奖励蒸馏：基于偏好对的交叉熵损失
- 世界模型：ELBO（证据下界）联合优化，包含重构损失（观测、奖励、continuation、语义奖励）
- 策略：Actor-Critic，基于世界模型 imagination 中的 $r_t^{\text{expl}}$ 优化

## 实验关键数据

### 实验环境

| 环境 | 类型 | 观测 | 动作 | 特点 |
|------|------|------|------|------|
| MiniHack (KeyRoom/KeyChest) | 地牢游戏 | 像素（自我中心视角） | 离散 | 需钥匙开门/宝箱 |
| Robodesk | 机器人操作 | 像素 | 连续 | 多物体交互 |
| Pokémon Red | RPG游戏 | 游戏画面 | 离散（Game Boy按键） | 开放世界+战斗系统 |

### 无任务探索结果

**MiniHack**：SENSEI 在 KeyRoom-S15 和 KeyChest 中，拾取钥匙、开门/宝箱的交互次数显著优于 Plan2Explore 和纯 VLM-Motif（$\beta=0$）。

**Robodesk**：1M 步探索中，SENSEI 与大多数物体（抽屉、滑门、方块、球等）的交互次数均超过 Plan2Explore 和 RND。

**消融实验**：纯 VLM-Motif（$\beta=0$）容易陷入局部最优——例如在 KeyRoom 中拾取钥匙后无法继续探索开门。

### 下游任务学习

| 方法 | KeyRoom-S15 | KeyChest |
|------|-------------|----------|
| SENSEI预探索→DreamerV3 | **最快收敛** | **最快收敛** |
| P2X预探索→DreamerV3 | 不稳定 | 较慢 |
| DreamerV3从头训练 | 慢 | 较慢 |
| PPO从头训练 | >20M步才稳定 | 早期有成功但不稳定 |

SENSEI 在 KeyRoom-S15 上比 PPO 快约 **两个数量级**。

### Pokémon Red 任务探索

- SENSEI 是唯一到达第一个道馆的方法（经过9个地图段）
- 750k 步后 SENSEI 的宝可梦等级持续高于 Plan2Explore（从 episode 390 起平均升级次数为 P2X 的 2 倍）
- 第二轮标注迭代后成功击败第一个道馆获得 Boulder Badge

## 亮点与洞察

1. **内化的有趣性模型**：不同于持续依赖外部 LLM，SENSEI 通过世界模型的奖励预测头内化了语义有趣性，可在 imagination 中直接预测，大幅提高样本效率
2. **Go-and-Explore 切换机制**：优雅地解耦了"前往有趣状态"和"从有趣状态出发探索新颖行为"两个阶段，避免纯语义奖励的局部最优
3. **零先验知识的 Sensei general**：环境描述可由 VLM 自动生成而非人类提供，效果几乎无损，增强了泛化性
4. **跨域验证**：在离散动作地牢游戏、连续动作机器人操作、复杂 RPG 三类截然不同的环境中一致有效
5. **迭代标注**：Pokémon 实验展示了迭代精化语义奖励的潜力（多轮标注解决分布外问题）

## 局限与展望

- **依赖全观测**：VLM 标注在存在遮挡时效果下降（Robodesk 需要多视角缓解）
- **初始数据集质量**：语义奖励蒸馏依赖 $\mathcal{D}^{\text{init}}$ 的行为丰富度，若初始探索太差则 VLM 标注缺少信号
- **VLM 标注噪声**：当前 VLM 的判断并不总是准确，消融实验表明更强的 VLM 能带来更大收益
- **可扩展性**：标注成本（GPT-4 API 调用）在大规模应用中可能成为瓶颈
- **单一时间步标注**：仅基于单帧图像对比而非视频序列，丢失了时序信息
- 未在真实机器人或照片级真实环境中验证

## 相关工作与启发

- **Plan2Explore** (Sekar et al., 2020)：集成分歧作为内在奖励的 MBRL 探索基线，SENSEI 在此基础上加入语义偏置
- **Motif** (Klissarov et al., 2023)：用 LLM 对事件字幕偏好标注蒸馏奖励，SENSEI 将其扩展为视觉版（VLM-Motif）
- **DreamerV3** (Hafner et al., 2023)：底层世界模型和策略优化框架
- **RL-VLM-F** (Wang et al., 2024)：类似的 VLM 偏好蒸馏但非 model-based、且需要明确任务描述
- **Go-Explore** (Ecoffet et al., 2021)：启发了 SENSEI 的 go-then-explore 切换策略

## 评分

- 新颖性: ⭐⭐⭐⭐ — VLM偏好蒸馏+世界模型内化预测+自适应切换的组合新颖且合理
- 实验充分度: ⭐⭐⭐⭐⭐ — 三类环境全面验证，消融实验详尽（Go-Explore、零先验、标注噪声、数据质量）
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机与方法衔接流畅
- 价值: ⭐⭐⭐⭐ — 为VLM引导的自主探索提供了实用且可扩展的范式，随VLM进步价值更大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[CVPR 2026\] Dexterous World Models](../../CVPR2026/robotics/dexterous_world_models.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/robotics/self-improving_embodied_foundation_models.md)
- [\[ICLR 2026\] Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](../../ICLR2026/robotics/theory_of_space_can_foundation_models_construct_spatial_beliefs_through_active_e.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](../../CVPR2025/robotics/universal_actions_for_enhanced_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
