---
title: >-
  [论文解读] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards
description: >-
  [NeurIPS 2025 Spotlight][强化学习][RLVR] 发布包含100+过程生成推理任务的Reasoning Gym库，覆盖代数、算术、算法、逻辑、几何、图论、游戏等领域，每个任务支持无限数据生成和参数化难度控制，实验证明RLVR训练在域内/跨域均实现显著技能迁移且能提升MATH、GSM8K等外部基准表现。
tags:
  - "NeurIPS 2025 Spotlight"
  - "强化学习"
  - "RLVR"
  - "可验证奖励"
  - "过程生成"
  - "课程学习"
  - "推理迁移"
  - "难度悬崖"
---

# Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.24760](https://arxiv.org/abs/2505.24760)  
**代码**: [GitHub](https://github.com/open-thought/reasoning-gym/)  
**领域**: LLM推理 / 强化学习  
**关键词**: RLVR, 可验证奖励, 过程生成, 课程学习, 推理迁移, 难度悬崖

## 一句话总结
发布包含100+过程生成推理任务的Reasoning Gym库，覆盖代数、算术、算法、逻辑、几何、图论、游戏等领域，每个任务支持无限数据生成和参数化难度控制，实验证明RLVR训练在域内/跨域均实现显著技能迁移且能提升MATH、GSM8K等外部基准表现。

## 研究背景与动机

**领域现状**：LLM推理能力近期大幅跃升（o1、DeepSeek-R1、QwQ-32B），核心驱动力是基于可验证奖励的强化学习（RLVR）。RLVR利用outcome-based反馈引导模型发展开放式推理过程。

**现有痛点**：(1) **数据瓶颈**——当前RLVR依赖人工整理的问答对或互联网爬取的内容，既昂贵又不可持续，且随着推理模型不断进步数据稀缺将成为越来越严重的约束。(2) **记忆化问题**——固定数据集导致模型可能记住答案而非学会推理。(3) **难度不可控**——现有benchmark无法按需调整难度，不支持课程学习。(4) **评估依赖人工判断**——部分推理任务的正确性需要主观评估。

**核心矛盾**：RLVR的成功关键是大量高质量、可自动验证的训练数据，但这恰好是当前最稀缺的资源。固定数据集同时面临规模上限、记忆化和难度固化三重限制。

**本文目标** 构建一个可以过程化生成无限训练数据、自动验证、参数化控制难度的推理任务库，从根本上解决RLVR的数据瓶颈。

**切入角度**：将推理任务设计为"环境"而非"数据集"——每个任务是一个生成算法，参数控制问题特性，验证器自动判断答案正确性。这类似于RL中的环境（environment）概念。

**核心 idea**：用过程化生成的方式构建100+可验证推理环境，支持无限数据、动态难度和自动验证，为RLVR训练提供可扩展的基础设施。

## 方法详解

### 整体框架
Reasoning Gym（RG）包含三层设计：(1) **任务生成器**——每个推理任务是一个参数化的生成算法，可以产生无限多不重复的问题实例；(2) **验证器**——每个任务配备自动验证器，判断模型输出是否正确，提供二元奖励；(3) **参数控制**——每个任务有三类参数：难度参数（控制复杂度）、结构参数（控制问题属性）、风格参数（控制呈现方式但不影响难度）。

任务覆盖的领域分类：数学（代数、算术、几何）、算法思维（搜索、优化、过程）、逻辑推理（形式证明、推理规则）、模式识别（序列、视觉类比）、约束满足（游戏、谜题、规划）。

### 关键设计

1. **过程化生成与算法可验证性**:

    - 功能：消除数据集大小限制和记忆化风险
    - 核心思路：每个任务不是固定的问答对集合，而是一个生成函数——给定种子和参数即可产生新的问题。所有任务都有确定性的正确答案验证算法，无需人工判断。解空间足够大，使得奖励黑客（reward hacking）困难
    - 设计动机：过程化生成从根本上解决了三个问题：(1) 无限数据量——不存在数据集上限；(2) 无记忆化——每个实例都是新的；(3) 可持续——不依赖人工标注或互联网爬取

2. **参数化难度控制**:

    - 功能：支持动态课程学习和精细的能力诊断
    - 核心思路：难度参数直接控制问题复杂度（图节点数、多项式次数、单词长度等），结构参数控制问题属性（维度、约束类型、证明深度），风格参数变换呈现方式（变量名、数字格式、问题措辞）
    - 设计动机：参数化使研究者可以精确控制实验变量——例如只改变图的节点数来研究模型在不同规模上的表现，或只改变表示方式来测试模型是否真正理解问题

3. **多域覆盖与评测标准化**:

    - 功能：全面评估不同类型的推理能力
    - 核心思路：涵盖代数、算术、几何、算法、逻辑、模式识别、约束满足等大类，每类下有多个具体任务。每个任务都有easy和hard两套参数配置用于标准化评测
    - 设计动机：推理能力不是单一维度的——一个模型可能擅长数学但不擅长空间推理。宽覆盖使得研究者可以识别模型的具体优缺点

### 损失函数 / 训练策略
使用GRPO算法进行RLVR训练，奖励由准确性奖励（1.0分）和格式奖励（0.2分）组成。课程学习策略：当模型在20个连续训练步中准确率超过70%时自动提升难度级别。

## 实验关键数据

### 主实验（零样本评估 on hard config）

| 模型 | 总体准确率 | 类型 |
|------|-----------|------|
| o3-mini | 63.5% | 推理优化 |
| DeepSeek-R1 | 59.5% | 推理优化 |
| Grok 3 Mini | 55.1% | 推理优化 |
| Llama 4 Maverick | 41.5% | 通用型 |
| Claude 3.5 Sonnet | 40.3% | 通用型 |
| Gemma 3 27B | 20.3% | 通用型 |

### 跨域迁移实验（Qwen2.5-3B-Instruct训练后）

| 训练域到测试域 | 提升 | 说明 |
|----------------|------|------|
| Algorithmic到Algebra | +29.1% | 过程推理迁移到代数 |
| Algorithmic到Geometry | +22.3% | 过程推理迁移到几何 |
| Games到Algebra | +21.8% | 约束满足迁移到代数 |
| Logic到Cognition | +13.3% | 逻辑推理迁移到认知 |
| RG-Math到MATH基准 | +9.7% | 外部基准迁移 |
| RG-Math到Big-Bench Hard | +7.66% | 外部基准显著提升 |

### 消融：课程学习 vs 固定难度

| 任务 | 固定难度 | 课程学习 | 说明 |
|------|---------|---------|------|
| Spell Backwards | 基线水平 | 更高 | 渐进难度有效 |
| Count Primes | 基线水平 | 相当 | 模型未能突破初始难度 |
| 多数任务 | 基线 | 优于或持平 | 课程学习总体有益 |

### 关键发现
- **推理模型 vs 通用模型有22%的能力鸿沟**：最好的推理模型（63.5%）与最好的通用模型（41.5%）差距巨大，说明RLVR不是微小改进而是能力跃迁
- **难度悬崖现象**：从easy到hard配置，代码任务性能暴降62%、图任务降30%、几何降33%，说明当前模型的推理能力表面强但脆弱
- **跨域迁移显著**：算法训练模型在代数上提升29%、几何上提升22%，说明RLVR确实发展出可迁移的推理原语而非域特定的模式匹配
- **从0到1的能力涌现**：Games类任务基线准确率为0，RLVR后达3.3%——虽然绝对值不高，但证明RLVR可以bootstrapping全新的推理能力
- **外部基准迁移有效**：RG-Math在MATH上提升9.7%，Big-Bench Hard上提升7.66%，MMLU-Pro多个学科均有提升，证明技能可迁移到真实场景

## 亮点与洞察
- **"环境"而非"数据集"的设计哲学**改变了推理训练的范式：数据集有上限，环境没有。这个概念本身就是重要贡献，可以激发更多人设计推理环境而非收集推理数据
- **跨域迁移结果**是最令人惊喜的发现：在算法任务上训练的模型在代数和几何上提升了20%+，暗示不同推理领域共享底层的推理原语（如分解、回溯、验证），且RLVR可以发展这些原语
- **难度悬崖现象**的系统性记录为理解当前推理模型的能力边界提供了重要证据——模型更可能学到了特定复杂度范围内的解题模板而非真正的推理策略

## 局限与展望
- 目前的任务偏重"有明确答案的推理"（如数学、图算法），对开放式推理（如策略规划、创意思维）覆盖不足
- 过程化生成的任务不一定能覆盖自然产生的推理问题的分布——真实世界的推理问题可能有生成器无法模拟的结构
- 验证器限于精确匹配，对部分正确的推理路径无法给出细粒度反馈（如过程奖励）
- 实验仅在3B规模的Qwen模型上进行，更大规模模型上的RLVR效果和迁移模式可能不同

## 相关工作与启发
- **vs DeepSeek-R1的RLVR数据**: DeepSeek-R1使用固定的数学和编程数据做RLVR，RG提供了更广泛领域、更可控的替代方案
- **vs MATH/GSM8K**: 这些是固定评测集，RG既可用于训练也可用于评测，且任务多样性远超单一数学benchmark
- **vs ARC Benchmark**: ARC是固定的视觉推理benchmark，RG包含ARC风格的任务但以文本格式呈现，且可过程化生成

## 评分
- 新颖性: ⭐⭐⭐⭐ "推理环境"的概念和大规模实现有明确贡献，但过程化测试数据生成并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 零样本评估、域内/跨域迁移、外部基准迁移、课程学习消融全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 实验发现的组织极为清晰，每个实验都有明确的research question和takeaway
- 价值: ⭐⭐⭐⭐⭐ 作为开源基础设施对RLVR研究有直接推动作用，跨域迁移发现对理解推理本质有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](../../ICLR2026/reinforcement_learning/longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](../../ICLR2026/reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](generalizing_verifiable_instruction_following.md)
- [\[ICML 2026\] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](../../ICML2026/reinforcement_learning/unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)

</div>

<!-- RELATED:END -->
