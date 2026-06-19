---
title: >-
  [论文解读] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization
description: >-
  [NeurIPS 2025][自监督学习][自监督强化学习] 针对自监督强化学习（SS-RLVR）在长期训练中普遍出现的"策略崩溃"问题，提出 M-GRPO：通过动量模型提供稳定的伪标签目标 + 基于四分位距（IQR）的低熵轨迹过滤防止熵崩溃，在无标注 MATH 数据集上训练 Qwen3-4B-Base，最终 checkpoint 即超越 SRT 手动选取的最佳 checkpoint，AIME24 +2.92%、GPQA +5.05%。
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "自监督强化学习"
  - "策略崩溃"
  - "动量锚定"
  - "GRPO"
  - "熵过滤"
  - "伪标签"
---

# M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2512.13070](https://arxiv.org/abs/2512.13070)  
**代码**: [https://github.com/M_GRPO](https://github.com/M_GRPO)  
**领域**: 自监督  
**关键词**: 自监督强化学习, 策略崩溃, 动量锚定, GRPO, 熵过滤, 伪标签

## 一句话总结

针对自监督强化学习（SS-RLVR）在长期训练中普遍出现的"策略崩溃"问题，提出 M-GRPO：通过动量模型提供稳定的伪标签目标 + 基于四分位距（IQR）的低熵轨迹过滤防止熵崩溃，在无标注 MATH 数据集上训练 Qwen3-4B-Base，最终 checkpoint 即超越 SRT 手动选取的最佳 checkpoint，AIME24 +2.92%、GPQA +5.05%。

## 研究背景与动机

1. **领域现状**：基于可验证奖励的强化学习（RLVR）是 LLM 后训练的核心手段，但依赖昂贵的人工标注数据和奖励模型基础设施。近期自监督 RLVR（SS-RLVR）方法（如 SRT、Intuitor、TTRL、CoReward）尝试通过模型自身的一致性信号（如 majority voting）构造伪奖励，免去真实标签。
2. **现有痛点**：
    - **策略崩溃**：作者复现 SRT 和 Intuitor 在 MATH 数据集上的自监督训练，发现训练奖励先升后骤降或渐降，验证集准确率同步退化——这是所有 SS-RLVR 方法的共性失败模式
    - **扩大 rollout 数量只能延缓崩溃**：将 rollout 数从 16 增大到 128 虽能提高峰值性能，但崩溃仍不可避免，只是延后发生
    - **熵崩溃**：训练初期策略熵急剧下降，导致模型过早自信、锁死在次优策略上
3. **核心矛盾**：自监督 RL 中伪标签来源于当前策略模型自身——策略快速变化导致伪标签不稳定，不稳定的伪标签又反过来加剧策略漂移，形成恶性循环。
4. **本文要解决什么？** 打破"快速变化策略 → 不稳定伪标签 → 策略崩溃"的恶性循环，同时防止伴随的熵崩溃。
5. **切入角度**：从自监督视觉表示学习的动量对比（MoCo）获得灵感——用一个缓慢演化的动量模型作为稳定锚点。
6. **核心 idea**：双模型框架：当前策略模型 $\pi_{\theta_q}$ 用于训练更新，动量模型 $\pi_{\theta_k}$（EMA 参数）提供稳定的 rollout，两者的输出共同投票生成伪标签；同时用 IQR 方法自适应过滤低熵轨迹保持探索多样性。

## 方法详解

### 整体框架

基于 GRPO（Group Relative Policy Optimization）改造。引入动量模型 $\pi_{\theta_k}$，其参数是当前策略 $\pi_{\theta_q}$ 的指数移动平均。每个 prompt 同时从两个模型采样 rollout，汇总后 majority voting 得到伪标签，再用 GRPO 目标更新当前策略。

### 关键设计

1. **动量锚定的自监督 RL（M-GRPO）**
    - 做什么：引入动量模型 $\pi_{\theta_k}$ 参与伪标签生成，稳定训练目标
    - 核心思路：
     - 每个 prompt $x$，当前策略采样 $M$ 个 rollout $\{y_i^q\}$，动量模型采样 $N$ 个 rollout $\{y_j^k\}$，汇总为 $G = M + N$ 个候选
     - Majority voting 选出最高共识答案 $y_v$ 作为伪真值
     - 基于 $y_v$ 为当前策略的 $M$ 个 rollout 打二值奖励（一致=1，不一致=0）
     - 按 GRPO 方式计算归一化优势函数 $\hat{A}_i$ 并优化
    - 动量更新规则：$\pi_{\theta_k} \leftarrow m \cdot \pi_{\theta_k} + (1-m) \cdot \pi_{\theta_q}$，$m=0.99$
    - 设计动机：
     - 动量模型演化缓慢，提供的 rollout 具有时间一致性，减少 majority voting 结果的波动
     - 类比 MoCo 中动量编码器对对比学习的稳定作用
     - 扩大了投票池的多样性（两个略微不同的策略视角），提高伪标签质量

2. **基于 IQR 的轨迹熵过滤**
    - 做什么：自适应剔除低熵轨迹，防止策略过早收敛
    - 核心思路：
     - 对每个 prompt 的 $G$ 条轨迹，计算各自的轨迹级熵
     - 计算熵分布的 $Q_1$、$Q_3$ 和 $\text{IQR} = Q_3 - Q_1$
     - 将熵低于 $Q_1 - k \cdot \text{IQR}$（$k=0.75$）的轨迹标记为低熵异常值并剔除
     - 仅用过滤后的轨迹参与 voting 和策略优化
    - 设计动机：
     - 低熵轨迹对应过度自信的策略输出，其伪标签质量差且会压制探索
     - 比静态阈值（如固定去掉最低 10%）更灵活——训练初期大多数轨迹熵高，IQR 自动放宽；训练后期熵自然降低，IQR 自动收紧
     - 保留高熵轨迹以维持策略多样性

3. **训练流程整合**
    - 每轮迭代：采样 batch → 双模型 rollout → IQR 过滤 → majority voting → 计算优势 → 更新当前策略 → EMA 更新动量模型
    - 动量模型的 rollout 数量 $N = G/4$（即当前模型贡献 3/4 的 rollout）

### 损失函数 / 训练策略

- 策略目标：$\mathcal{J}(\theta_q) = \mathbb{E}\left[\sum_{i=1}^{M} \hat{A}_i \log \pi_{\theta_q}(y_i^q | x)\right]$
- KL 正则化系数：0.005
- 优化器：AdamW，学习率 $10^{-6}$，cosine warmup（0.1 ratio）
- clip ratio：0.2
- 训练温度 1.1，评估温度 0.8
- 最大响应长度 3072

## 实验关键数据

### 主实验：Qwen3-4B-Base 在无标注 MATH 上训练

| 方法 | MATH500 | AIME24 | AIME25 | GPQA Dia | GPQA | LiveCode |
|------|---------|--------|--------|----------|------|----------|
| 原始模型 | 61.50% | 0.83% | 5.00% | 34.41% | 29.91% | 9.61% |
| SRT_Best（手动选最优ckpt） | 79.20% | 12.50% | 11.67% | 38.26% | 35.04% | 19.69% |
| SRT_Final（最终ckpt） | 47.50% | 7.50% | 8.75% | 28.54% | 25.89% | 16.12% |
| **M-GRPO+IQR_Final** | **79.75%** | **14.58%** | **14.17%** | **39.65%** | 35.49% | — |

### Rollout 缩放分析（M-GRPO+IQR）

| G（rollout数） | MATH500 | AIME24 | AIME25 | GPQA Dia | MMLU-pro | mbpp |
|----------------|---------|--------|--------|----------|----------|------|
| 8 | 77.60% | 11.25% | 10.42% | 39.02% | 56.05% | 68.60% |
| 16 | 79.75% | 14.43% | 10.00% | 39.65% | 57.05% | 70.40% |
| 32 | 79.75% | 14.58% | 14.17% | 39.65% | 55.47% | 70.60% |
| 256 | 79.50% | 16.67% | 14.17% | 40.66% | 55.08% | 70.40% |

### 关键发现

- **SRT 的策略崩溃是灾难性的**：SRT_Final 在 MATH500 上从峰值 79.20% 跌至 47.50%，比原始模型（61.50%）还差——说明不受控的自监督训练会遗忘预训练能力
- **M-GRPO 彻底消除策略崩溃**：最终 checkpoint 直接优于 SRT 手动挑选的最佳 checkpoint，无需人工干预
- **Rollout 缩放在 M-GRPO 中是稳定的**：从 G=8 到 G=32 性能持续提升，但 G=256 后收益饱和，说明 M-GRPO 已充分利用 rollout 信息
- **IQR 过滤成功维持策略熵**：对比 SRT 训练早期熵急剧下降，M-GRPO 的熵缓慢平稳下降，避免了过早收敛
- **跨任务泛化**：仅在 MATH 上训练，但在 GPQA（科学推理）、AIME（竞赛数学）、LiveCode（代码）上均有提升

## 亮点与洞察

- **"策略崩溃"诊断深入且可复现**：不仅指出问题存在，还通过系统性的 rollout 缩放实验揭示了"更多 rollout 延缓但不能阻止崩溃"的规律，为后续研究建立了清晰的 baseline
- **MoCo 到 RL 的类比精妙**：自监督对比学习中动量编码器解决负样本一致性问题 → 自监督 RL 中动量策略解决伪标签一致性问题，跨领域迁移很自然
- **IQR 过滤的自适应性**：相比静态阈值（如 EdgeGRPO），IQR 方法无需调超参即可适应训练过程中熵分布的动态变化，是一个简洁但有效的工程贡献
- **最终 checkpoint 即最优 checkpoint**：消除了实际部署中"何时停止训练"的难题，显著降低了自监督 RL 的使用门槛

## 局限性 / 可改进方向

- 仅在 Qwen3-4B-Base 上验证，缺少更大规模模型（如 7B、14B 以上）的实验证据
- 动量系数 $m=0.99$ 固定不变，自适应调整 $m$（如根据训练阶段或策略变化率动态调节）可能带来进一步提升
- 动量模型的 rollout 比例固定为 $N = G/4$，其最优比例可能随任务和模型规模变化
- 实验数据集仅限 MATH，未在其他自监督 RL 场景（如代码生成、对话优化）中验证方法的通用性
- IQR 系数 $k=0.75$ 的敏感性分析不足，缺少消融研究说明不同 $k$ 值的影响
- 未与其他稳定训练方法（如 replay buffer、conservative policy update）进行对比
- 双模型架构引入额外的推理开销（动量模型需要额外采样 $N$ 条轨迹），在资源受限场景下需要权衡

## 相关工作与启发

- **vs SRT / Intuitor / TTRL**：这些方法均使用单模型的自身一致性作为伪标签，策略快速漂移导致伪标签质量退化；M-GRPO 通过引入缓慢演化的动量模型切断了这一恶性循环
- **vs DAPO / GRPO**：DAPO 和 GRPO 在有监督 RLVR 场景下设计，依赖真实奖励信号；M-GRPO 将其扩展到自监督场景，核心创新在于伪标签的稳定生成机制
- **vs MoCo（自监督视觉表示学习）**：MoCo 用动量编码器稳定负样本队列→M-GRPO 用动量策略稳定 majority voting 结果，是同一范式在不同领域的成功应用
- **vs EdgeGRPO（静态熵过滤）**：EdgeGRPO 使用固定比例过滤低熵轨迹；M-GRPO 的 IQR 方法根据熵的实际分布自适应调整阈值，更加鲁棒
- **启发**：动量锚定范式可推广到任何依赖自生成信号的迭代优化过程——如 self-play、self-distillation、iterative refinement 等

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 MoCo 动量机制引入自监督 RL 是自然但有效的创新；IQR 过滤是较增量式的贡献
- 实验充分度: ⭐⭐⭐ 诊断性实验（崩溃复现、rollout 缩放）出色，但模型规模和数据集多样性不足
- 写作质量: ⭐⭐⭐⭐ 问题诊断清晰、方法动机充分，图表直观
- 价值: ⭐⭐⭐⭐ 为自监督 RL 训练的稳定性提供了原则性解决方案，有较好的实用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](../../ICCV2025/self_supervised/improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ACL 2026\] LLMSurgeon: Diagnosing Data Mixture of Large Language Models](../../ACL2026/self_supervised/llmsurgeon_diagnosing_data_mixture_of_large_language_models.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
