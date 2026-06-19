---
title: >-
  [论文解读] Test-Time Adaptation with Binary Feedback
description: >-
  [ICML 2025][强化学习][测试时自适应] 本文提出 BiTTA，一个利用二元反馈（正确/错误）的测试时自适应框架，通过强化学习驱动的双路径优化策略，在严重域偏移下以最小标注成本实现 13.3% 的准确率提升。 深度学习模型在训练和测试数据存在领域偏移时性能显著下降。测试时自适应（TTA）通过在测试时利用未标注样本适…
tags:
  - "ICML 2025"
  - "强化学习"
  - "测试时自适应"
  - "二元反馈"
  - "领域偏移"
  - "在线学习"
---

# Test-Time Adaptation with Binary Feedback

**会议**: ICML 2025  
**arXiv**: [2505.18514](https://arxiv.org/abs/2505.18514)  
**代码**: [GitHub](https://github.com/taeckyung/BiTTA)  
**领域**: 机器学习 / 测试时自适应  
**关键词**: 测试时自适应, 二元反馈, 强化学习, 领域偏移, 在线学习

## 一句话总结

本文提出 BiTTA，一个利用二元反馈（正确/错误）的测试时自适应框架，通过强化学习驱动的双路径优化策略，在严重域偏移下以最小标注成本实现 13.3% 的准确率提升。

## 研究背景与动机

深度学习模型在训练和测试数据存在领域偏移时性能显著下降。测试时自适应（TTA）通过在测试时利用未标注样本适应预训练模型来解决这一问题。然而：

**现有 TTA 方法的脆弱性**: 在严重域偏移下，基于熵/置信度的自监督指标（如 TENT）不可靠，导致适应失败

**主动 TTA 的高成本**: 近期的 active TTA 方法需要完整类别标签，标注代价过高（50类分类任务中全标注平均 11.7 秒/样本，错误率 12.7%）

**二元反馈的效率**: 相比之下，二元比较只需 1.6 秒/样本，错误率仅 0.8%。从信息论角度，全标注需要 $\log(\text{num\_class})$ 倍于二元反馈的比特数

核心洞察：二元反馈虽然只提供 1 bit 信息，但因为是基于适应模型的预测（通常优于随机），所以信息含量更高，可以直接指导模型行为。

## 方法详解

### 整体框架

BiTTA 将 TTA 建模为强化学习问题：
- **状态**: 测试样本 $x$
- **动作**: 模型预测 $y^* = \arg\max_y f_\theta(y|x)$
- **策略**: 预测概率 $\pi_\theta(y|x)$
- **目标**: 最大化期望奖励 $J(\theta) = \mathbb{E}_{x, y \sim \pi_\theta}[R(x, y)]$

使用 REINFORCE 算法处理不可微的二元反馈，通过 MC-dropout 近似策略：

$$\pi_\theta(y|x) = \frac{1}{N}\sum_{n=1}^N f_\theta^d(y|x)$$

### 双路径优化

**路径1：Binary Feedback-guided Adaptation (BFA)**

选择 top-$k$ 不确定样本（MC-dropout 置信度最低的），查询二元反馈：

$$R_{\text{BFA}}(x, y) = B(x, y) = \begin{cases} 1 & \text{正确} \\ -1 & \text{错误} \end{cases}$$

正确预测的样本存入 $\mathcal{M}_C$，错误的存入 $\mathcal{M}_I$（FIFO 缓存）。

**路径2：Agreement-Based self-Adaptation (ABA)**

在剩余未标注样本中，选取标准预测与 MC-dropout 预测**一致**的样本作为"置信样本"：

$$\mathcal{S}_{\text{ABA}} = \{x \in \mathcal{B} \setminus \mathcal{S}_{\text{BFA}} \mid y^* = \arg\max_y \pi_\theta(y|x)\}$$

关键优势：不依赖固定阈值（传统方法的痛点），而是动态地基于预测一致性选择。

### 联合损失函数

$$\mathcal{L}_{\text{BiTTA}} = \alpha \cdot \underbrace{\frac{1}{|\mathcal{M}_C|}\sum_{x \in \mathcal{M}_C}(-\log \pi_\theta) + \frac{1}{|\mathcal{M}_I|}\sum_{x \in \mathcal{M}_I}(+\log \pi_\theta)}_{\text{BFA: 最小化正确CE + 最大化错误CE}} + \beta \cdot \underbrace{\frac{1}{|\mathcal{S}_\text{ABA}|}\sum_{x \in \mathcal{S}_\text{ABA}}(-\log \pi_\theta)}_{\text{ABA: 最小化一致样本CE}}$$

其中 $\alpha = \beta = 1$。BFA 对正确预测增强、错误预测弱化，ABA 对一致（大概率正确）预测进行巩固。对不确定且未获得反馈的样本，不施加任何梯度（奖励为0），避免噪声信号的有害适应。

## 实验关键数据

### CIFAR10-C（严重腐蚀级别5）平均准确率

| 标签类型 | 方法 | 平均准确率 (%) |
|---|---|---|
| - | SrcValid（不适应） | 57.23 |
| - | BN-Stats | 78.42 |
| 二元 | TENT* | 80.49 |
| 二元 | SAR* | 83.78 |
| 二元 | CoTTA* | 78.42 |
| 二元 | RoTTA* | 80.98 |
| 二元 | SimATTA*（全标签改为二元） | 81.09 |
| **二元** | **BiTTA** | **87.20** |

BiTTA 比次优基线 SAR* 高出 3.42%p，比 SrcValid 高出约 30%p。

### 跨数据集平均性能提升

BiTTA 平均超越 SOTA 基线 **13.3%p**。

### 关键对比

- BiTTA（仅二元反馈）**超越** SimATTA（完整类别标签的 active TTA）
- BiTTA **超越** GPT-4o 作为标注器的 active TTA（Figure 7）
- 在 CIFAR100-C 和 Tiny-ImageNet-C 上同样保持优势

### ABA 有效性验证

- 预测一致的样本准确率稳定且高（~90%+）
- 预测不一致的样本准确率低且不稳定
- 基于一致性的动态选择优于固定阈值策略

### 消融实验关键结论

| 组件 | 移除后影响 |
|---|---|
| BFA (无二元反馈) | 大幅下降，退化为纯自适应 |
| ABA (无一致性自适应) | 显著下降，仅靠少量反馈不足 |
| MC-dropout (换标准 softmax) | 校准变差，BFA 选样质量下降 |
| 记忆缓存 | 早期适应不稳定 |

## 亮点与洞察

1. **设定的实用性**: 二元反馈（对/错）相比全标签标注成本降低一个数量级，错误率也大幅降低——非常适合实际部署场景
2. **超越全标签 active TTA**: 看似信息量更少的二元反馈，通过 BiTTA 的精心设计反而比完整标签表现更好——因为更关注修正错误而非记忆标签
3. **双路径协同**: BFA 探索不确定区域（学习新知识），ABA 巩固确定区域（保持已有知识），形成互补
4. **MC-dropout 的多重作用**: 同时服务于策略估计、不确定性量化和一致性检测，设计高效

## 局限性

1. MC-dropout 需要多次前向传播（$N$ 次），增加推理延迟，在实时TTA场景中可能成为瓶颈
2. 每批次仅查询 $k$ 个二元反馈，在极端域偏移下反馈数量可能不足
3. 假设oracle的二元反馈完全准确，现实中人类注释者仍有误差
4. 仅在图像分类任务上评估，未验证在其他模态（NLP、语音）上的有效性
5. FIFO 缓存大小固定为批次大小，未探索自适应缓存策略

## 相关工作

- **测试时自适应**: TENT (Wang et al., 2021)、CoTTA (Wang et al., 2022)、SAR (Niu et al., 2023)、RoTTA (Yuan et al., 2023)、SoTTA (Gong et al., 2023)
- **主动测试时自适应**: SimATTA (Gui et al., 2024)——使用完整类别标签
- **RLHF**: Ouyang et al. (2022) 在 LLM 中利用人类反馈，启发了本文在 TTA 中的应用
- **不确定性估计**: MC-dropout (Gal & Ghahramani, 2016)
- **主动学习**: Settles (2009)——不确定性驱动的样本选择策略

## 评分

⭐⭐⭐⭐

问题设定新颖实用（二元反馈 TTA），方法设计合理（RL + 双路径），实验结果令人印象深刻（超越全标签 active TTA）。MC-dropout 的多角色复用是一个巧妙的工程决策。不过核心技术（REINFORCE + 交叉熵）相对直接，缺乏更深入的理论分析。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](revise_learning_to_refine_at_test-time_via_intrinsic_self-verification.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](../../AAAI2026/reinforcement_learning/aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)

</div>

<!-- RELATED:END -->
