---
title: >-
  [论文解读] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks
description: >-
  [ICML 2025][强化学习][并行数据采集] 系统研究 on-policy RL 中并行数据采集的两个维度（并行环境数 $N_{\text{envs}}$ vs 轨迹长度 $N_{\text{RO}}$）对 PPO 性能的影响，发现在固定数据预算下增加并行环境数比增加轨迹长度更有效，且更大的数据集可改善网络可塑性和优化稳定性。
tags:
  - "ICML 2025"
  - "强化学习"
  - "并行数据采集"
  - "PPO"
  - "网络可塑性"
  - "样本效率"
  - "偏差-方差权衡"
  - "Atari"
---

# The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks

**会议**: ICML 2025  
**arXiv**: [2506.03404](https://arxiv.org/abs/2506.03404)  
**作者**: Walter Mayor, Johan Obando-Ceron, Aaron Courville, Pablo Samuel Castro
**领域**: 强化学习  
**关键词**: 并行数据采集, PPO, 网络可塑性, 样本效率, 偏差-方差权衡, Atari

## 一句话总结

系统研究 on-policy RL 中并行数据采集的两个维度（并行环境数 $N_{\text{envs}}$ vs 轨迹长度 $N_{\text{RO}}$）对 PPO 性能的影响，发现在固定数据预算下增加并行环境数比增加轨迹长度更有效，且更大的数据集可改善网络可塑性和优化稳定性。

## 研究背景与动机

**领域现状**：并行数据采集是现代 RL 算法的标配技术，GPU 加速模拟器（Isaac Gym、EnvPool、PGX）使得单设备运行数千并行环境成为可能。

**现有痛点**：尽管并行化被广泛使用，但如何选择并行采集的两个关键参数（并行环境数 $N_{\text{envs}}$ 与轨迹长度 $N_{\text{RO}}$）缺乏系统研究。先前工作（Singla et al., 2024）发现单纯增加数据量会出现收益递减，但未分析数据结构的影响。

**核心矛盾**：在 PPO 中，数据批大小 $|B| = N_{\text{envs}} \times N_{\text{RO}}$，两个因子的不同组合引发不同的权衡：
   - $N_{\text{envs}}$ 影响数据**多样性**（状态-动作空间覆盖度）
   - $N_{\text{RO}}$ 引发**偏差-方差权衡**（长轨迹偏差小但方差大）
   - 训练 epoch 数需平衡样本效率与过拟合

**本文目标**：系统分析并行数据采集方式、网络可塑性、学习表示和样本效率之间的交互作用，给出实践指导。

**切入角度**：固定总环境交互步数（100M），在 PPO 和 PQN 上系统变化 $N_{\text{envs}}$ 和 $N_{\text{RO}}$，通过 weight norm、gradient kurtosis 等指标分析优化稳定性。

## 方法详解

### 整体框架

本文不是提出新算法，而是对 PPO 的并行数据采集策略进行系统的实证分析。核心变量：

- **$N_{\text{envs}}$（并行环境数）**：同时运行的环境实例数，影响一次采集中状态-动作的多样性
- **$N_{\text{RO}}$（轨迹长度）**：每个环境每次采集的步数，影响回报估计的偏差-方差
- **$|B| = N_{\text{envs}} \times N_{\text{RO}}$**：数据批大小
- **Epoch 数**：在同一个 batch 上训练的遍数，更多 epoch 提高样本效率但可能导致过拟合

研究设计分三个维度：

1. **固定数据预算**：保持 $|B|$ 不变（如 1024），变换 $(N_{\text{envs}}, N_{\text{RO}})$ 的组合（如 8×128 vs 128×8），分析哪种分配更优
2. **扩大数据量**：增加 $N_{\text{envs}}$ 或 $N_{\text{RO}}$（不保持另一个不变），分析两种扩展方式的效果差异
3. **多 epoch 交互**：在不同 data 规模下变化 epoch 数，分析数据量对过拟合/可塑性损失的缓解作用

### 关键设计

#### 1. 固定预算下的分配策略

- 默认 PPO 配置：$N_{\text{envs}}=8, N_{\text{RO}}=128$（batch = 1024）
- 对比配置：$N_{\text{envs}}=128, N_{\text{RO}}=8$（batch = 1024）
- **核心发现**：在相同数据量下，优先增加 $N_{\text{envs}}$ 优于增加 $N_{\text{RO}}$
- **机制解释**：更多并行环境意味着更多独立的初始状态分布采样，提供更高的状态-动作覆盖度；而长轨迹来自同一条 Markov 链，轨迹内相关性更强

#### 2. 优化稳定性分析

通过监控以下指标量化并行策略对网络训练的影响：

- **Weight Norm（权重范数）**：$N_{\text{envs}}$ 越大，weight norm 越低，表明更稳定的参数更新
- **Gradient Kurtosis（梯度峭度）**：高 kurtosis 表示梯度分布有重尾/尖峰（不稳定信号），$N_{\text{envs}}$ 的增加显著降低 kurtosis
- **可塑性指标**：大 $N_{\text{envs}}$ 可缓解 dormant neurons 和特征 rank 退化问题

#### 3. 多 Epoch 与数据量的交互

- 在 PPO 中增加 epoch 数可提高样本效率，但通常导致性能崩溃（过拟合到旧数据）
- **关键发现**：更大的数据集尺寸可延缓甚至避免多 epoch 带来的性能退化
- 增加 $N_{\text{envs}}$ 比增加 $N_{\text{RO}}$ 更能有效缓解 epoch 导致的性能崩溃

## 实验关键数据

### 实验设置

| 配置项 | 设定 |
|-------|------|
| 算法 | PPO（CleanRL 实现）+ PQN |
| 评估环境 | Atari-10（Arcade Learning Environment） |
| 总交互步数 | 100M 环境步 |
| 随机种子数 | 5 |
| 评估指标 | Human-normalized IQM + 95% bootstrap CI |
| 默认 $N_{\text{envs}}$ | 8 |
| 默认 $N_{\text{RO}}$ | 128 |
| 硬件 | NVIDIA Tesla A100 GPU |
| 单实验耗时 | 约 2-3 天 |

### 固定预算下 $N_{\text{envs}}$ vs $N_{\text{RO}}$ 对比

| 配置 | $N_{\text{envs}}$ | $N_{\text{RO}}$ | Batch Size | IQM 表现 |
|------|------|--------|------------|----------|
| 默认 | 8 | 128 | 1024 | 基线 |
| 高并行 | 128 | 8 | 1024 | 显著优于基线 |
| 高轨迹 | 8 | 高 | 大 | 提升有限 |
| 扩展并行 | 64/128/256 | 128 | 大 | 持续提升 |

### 网络可塑性与稳定性指标

| 指标 | 低 $N_{\text{envs}}$ | 高 $N_{\text{envs}}$ | 趋势 |
|------|---------------------|---------------------|------|
| Weight Norm | 高 | 低 | 稳定性改善 |
| Gradient Kurtosis | 高（重尾分布） | 低（近正态） | 优化更平稳 |
| Dormant Neurons | 较多 | 较少 | 可塑性保持 |
| 多 Epoch 性能退化 | 严重 | 轻微 | 抗过拟合能力增强 |

## 亮点与洞察

- **"不是所有数据相等"**：相同数据量下，来自更多独立环境的数据比来自更长轨迹的数据更有价值，因为独立环境提供了更好的状态空间覆盖和更低的样本相关性
- **并行环境数 vs 轨迹长度的不对称性**揭示了深层机制：$N_{\text{envs}}$ 提升的是数据多样性（不同初始状态），而 $N_{\text{RO}}$ 提升的仅是轨迹内的时序延伸，后者受 Markov 链内相关性限制
- **可塑性的联系很有启发性**：并行数据采集不仅是计算加速工具，更是改善网络优化健康度的手段——更低的 weight norm 和 gradient kurtosis 直接关联更好的长期学习能力
- **Epoch 数的"安全扩展"**：在足够大的数据集下，可以放心增加训练遍数而不用担心性能崩溃，这为 PPO 的超参数调优提供了实用指导
- 实验在 PPO 和 PQN 两种算法上验证，并扩展到不同网络架构，增强了结论的普适性

## 局限与展望

- **仅限 on-policy 算法**：结论基于 PPO（和 PQN），是否适用于 off-policy 算法（SAC、TD3）未验证
- **环境有限**：仅在 Atari-10 上测试，连续控制（MuJoCo）、3D 环境的验证不充分
- **缓存截断导致信息不全**：论文后半部分（Section 5-7）涉及的架构分析、超参数敏感度、连续控制实验等细节未能完全覆盖
- **计算成本分析缺失**：增加 $N_{\text{envs}}$ 的 wall-clock 时间和 GPU 显存开销未详细报告
- **理论解释有限**：观察到 $N_{\text{envs}}$ 优于 $N_{\text{RO}}$ 但缺乏严格理论证明
- **未研究极端并行场景**：当 $N_{\text{envs}}$ 极大时（如 4096+），是否存在收益递减或负面效应

## 相关工作与启发

- **vs A3C/IMPALA**：早期并行 RL 关注分布式架构设计，本文关注并行策略对学习动态的影响
- **vs Singla et al. (2024)**：前者观察到数据扩展的收益递减，本文进一步拆解 $N_{\text{envs}}$ vs $N_{\text{RO}}$ 的差异
- **vs 可塑性研究（Lyle 2023, Moalla 2024）**：可塑性损失是深度 RL 的核心挑战，本文提出并行数据采集是一种"自然缓解"手段
- **对实践的启示**：在调配 PPO 时，应优先增加并行环境数而非轨迹长度；在增加 epoch 之前先确保数据集够大

## 评分

- 新颖性: ⭐⭐⭐ 实证分析为主，无新算法，但系统性分析有价值
- 实验充分度: ⭐⭐⭐⭐ Atari-10 + PPO/PQN + 多种指标 + 架构分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 对 RL 实践者的并行策略调优有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC](beyond_the_rainbow_high_performance_deep_reinforcement_learning_on_a_desktop_pc.md)
- [\[ICML 2025\] Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning](network_sparsity_unlocks_the_scaling_potential_of_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](../../NeurIPS2025/reinforcement_learning/confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[ICML 2025\] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources](heterogeneous_data_game_characterizing_the_model_competition_across_multiple_dat.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](../../ICLR2026/reinforcement_learning/deep_spi_safe_policy_improvement_via_world_models.md)

</div>

<!-- RELATED:END -->
