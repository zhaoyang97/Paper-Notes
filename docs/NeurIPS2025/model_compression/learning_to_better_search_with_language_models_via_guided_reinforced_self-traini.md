---
title: >-
  [论文解读] Learning to Better Search with Language Models via Guided Reinforced Self-Training
description: >-
  [NeurIPS 2025][模型压缩][搜索策略学习] 提出 Guided-ReST，通过将最优解作为子目标逐步融入模型自生成的搜索轨迹中，生成高质量训练数据并蒸馏更高效的搜索策略，在Countdown和代码自修复任务上显著提升搜索效率和准确率。 领域现状： 语言模型在复杂推理任务中仍面临挑战。近期研究（如Stream o…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "搜索策略学习"
  - "自训练"
  - "强化学习"
  - "测试时计算效率"
  - "语言模型推理"
---

# Learning to Better Search with Language Models via Guided Reinforced Self-Training

**会议**: NeurIPS 2025  
**arXiv**: [2410.02992](https://arxiv.org/abs/2410.02992)  
**代码**: [GitHub](https://github.com/snu-mllab/guided-rest)  
**领域**: Model Compression (LLM推理优化)  
**关键词**: 搜索策略学习, 自训练, 强化学习, 测试时计算效率, 语言模型推理

## 一句话总结

提出 Guided-ReST，通过将最优解作为子目标逐步融入模型自生成的搜索轨迹中，生成高质量训练数据并蒸馏更高效的搜索策略，在Countdown和代码自修复任务上显著提升搜索效率和准确率。

## 研究背景与动机

**领域现状**: 语言模型在复杂推理任务中仍面临挑战。近期研究（如Stream of Search, SoS）表明，训练模型模仿搜索轨迹（包含探索和回溯的完整决策过程）比仅训练最终解答能获得更好的泛化性能。

**现有痛点**: SoS训练使用的搜索轨迹通常是噪声的、次优的，导致测试时计算资源被大量浪费在冗余探索和重复回溯上。标准的ReST只是过滤成功轨迹进行微调，没有利用最优解的结构信息。

**核心矛盾**: 最优解虽然能提供有价值的引导，但直接用它们做行为克隆泛化能力差；而搜索轨迹虽然泛化性好但效率低。如何兼顾两者优势？

**本文目标**: 如何生成高质量的搜索轨迹数据，使模型内化高效的搜索策略，在有限token预算下最大化问题求解准确率。

**切入角度**: 借鉴jump-start RL的双策略框架思想，用最优解作为路标指导搜索过程。

**核心 idea**: 最优解不适合直接模仿，但可以作为step-by-step的路标，渐进式地融入模型的搜索过程以生成高质量轨迹。

## 方法详解

### 整体框架

Guided-ReST 分为两个阶段：(1) 利用子目标增强算法生成高质量搜索轨迹并进行监督微调；(2) 基于操作级MDP的PPO强化学习进一步优化。

### 关键设计

1. **子目标增强算法 (Subgoal Augmentation)**:

    - **功能**: 将最优解分解为一系列子目标节点，逐步融入模型自生成的搜索轨迹中
    - **为什么**: 直接用最优解的部分解作为提示虽能提高成功率（从55.96%到100%），但会导致轨迹分布偏离模型行为分布（交叉熵从0.0481升至0.2539），直接微调可能损害搜索能力
    - **怎么做**: 对于一条未能到达第 $t$ 个子目标的搜索轨迹，随机选择第 $(t-1)$ 个子目标节点的一个失败子节点，用正确的第 $t$ 个子目标替换，截断后续不一致的搜索历史，然后让模型从增强的子目标继续搜索。迭代执行直到所有子目标都被纳入。
    - **区别**: 与ReST直接丢弃失败轨迹不同，Guided-ReST在失败点注入修正信号，教模型在哪里回溯以及如何从失败中恢复。

2. **操作级强化学习 (Operation-level RL)**:

    - **功能**: 在Guided-ReST微调基础上使用PPO进一步优化
    - **为什么**: 标准PPO基于token级MDP，但优化目标是搜索策略而非token生成
    - **怎么做**: 将每个动作 $a_h$ 定义为单个树操作（节点生成/探索/验证/回溯）的token序列 $(a_{h,1}, \ldots, a_{h,L})$，重新定义重要性比率：$\log r_h(\theta) = \sum_{\ell=1}^{L}(\log\pi_\theta(a_{h,\ell}|a_{h,1:\ell-1}, s_h) - \log\pi_{\theta_{old}}(a_{h,\ell}|a_{h,1:\ell-1}, s_h))$
    - **区别**: 操作级MDP比token级MDP提升约2×的token效率

3. **Episode级扩展 (代码自修复)**:

    - **功能**: 将方法扩展到代码自修复的多轮交互场景
    - **为什么**: 代码任务响应更长，细粒度树搜索不可行
    - **怎么做**: 简化版本——在每轮截断失败episode，在用户反馈中增强最优解提示，让模型继续生成。随着轮次增加，带有最优解提示的轮数逐渐增多。

### 损失函数 / 训练策略

- **阶段一**: 监督学习微调：$\max_\theta \mathbb{E}_{q,Z}[\log\pi_\theta(Z|q)]$，其中 $Z$ 为Guided-ReST生成的搜索轨迹
- **阶段二**: PPO目标：$\max_\theta \mathbb{E}[\min(r_h(\theta)A_h, \text{clip}(r_h(\theta), 1-\epsilon, 1+\epsilon)A_h)]$
- 使用Monte Carlo方法计算优势函数，去掉KL惩罚项
- Countdown使用Llama-3.2-1B-Instruct (4K tokens)，代码任务使用Qwen2.5-7B-Instruct (16K tokens)

## 实验关键数据

### 主实验

**Countdown 最大token预算下的准确率:**

| 方法 | Seen Targets | Unseen Targets |
|------|-------------|----------------|
| BC | <40% | <40% |
| SoS | ~65% | ~63% |
| ReST | ~72% | ~70% |
| PPO (from SoS) | ~77% | ~75% |
| **Guided-ReST + PPO** | **87%** | **87%** |

**代码自修复 Pass@k 准确率 (CodeContests):**

| 方法 | Pass@1 | Pass@4 | Pass@16 | Pass@32 |
|------|--------|--------|---------|---------|
| Base | 4.5 | 11.3 | 20.6 | 25.8 |
| ReST | 9.4 | 17.0 | 25.9 | 30.4 |
| **Guided-ReST** | **10.5** | **19.4** | **28.9** | **33.9** |

### 消融实验

| 消融维度 | 结果 |
|---------|------|
| 部分解长度 vs 准确率 | 0步: 55.96%, 1步: 85.28%, 2步: 95.27%, 3步: 100% |
| 部分解长度 vs 交叉熵 | 0步: 0.0481, 1步: 0.1245, 2步: 0.2412, 3步: 0.2539 |
| 操作级 vs token级 MDP | 操作级: 87%, token级: 83%，且token效率提升1.5-2× |
| 密集子目标奖励的PPO | 几乎无效，甚至略微退化，说明Guided-ReST预训练的必要性 |
| Pass@k (Guided-ReST vs ReST) | k=32时 Guided-ReST: 96.8% vs ReST: 76.6%，差距随k增大 |

### 关键发现

- Guided-ReST在unseen targets上也达到87%，证明并非简单记忆最优解
- 达到PPO相当准确率只需约50%的token，token效率大幅提升
- Guided-ReST与PPO有强协同效应，而ReST+PPO无此效果，核心原因在于pass@k覆盖率差异
- 密集奖励的直接RL无法替代Guided-ReST的数据生成作用

## 亮点与洞察

- **核心洞察深刻**: 最优解不适合直接模仿但可作为路标——这一insight非常优雅且经过充分验证
- **子目标增强算法**: 设计精巧，在失败点注入修正而非简单重启，保持轨迹的高似然性
- **分析透彻**: 通过pass@k分析揭示了Guided-ReST与PPO协同的根本原因（更广的正确候选覆盖）
- **实用性强**: 方法简洁，应用范围从正式任务（Countdown）扩展到代码自修复

## 局限与展望

- 假设可获得最优解，实际中并非总是可行（论文提出用强模型生成作为relaxation）
- Countdown任务假设有oracle能精确识别首个错误步并回溯，这对语言模型不现实
- 代码自修复的提升幅度不如Countdown显著，可能因训练数据少、子目标增强不完整
- 仅在1B和7B模型上实验，更大规模模型的效果未知
- 未探索其他搜索算法（如MCTS）的Guided-ReST变体

## 相关工作与启发

- 与Stream of Search (SoS) 的关系最密切，在其基础上解决搜索轨迹质量和效率问题
- 受Jump-Start RL启发的双策略框架思想值得在更多场景推广
- 对DeepSeek-R1等推理模型的训练有参考价值：如何更高效地利用测试时计算
- 教模型"在哪里回溯"的思想可推广到更通用的推理纠错场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 子目标增强和操作级MDP设计新颖，但整体框架是已有技术的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ 消融充分，分析深入（pass@k分析、MDP对比、密集奖励对比等）
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，动机充分，图示直观
- 价值: ⭐⭐⭐⭐ 对搜索型推理模型的训练效率提升有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[NeurIPS 2025\] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)
- [\[NeurIPS 2025\] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [\[ICML 2026\] A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search](../../ICML2026/model_compression/a_language-guided_bayesian_optimization_for_efficient_lora_hyperparameter_search.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
