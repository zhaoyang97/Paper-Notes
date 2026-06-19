---
title: >-
  [论文解读] AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs
description: >-
  [ICML 2025][优化/理论][对抗提示] 提出 AdvPrompter——用一个 LLM（AdvPrompter）在秒级速度内为目标 LLM 生成人类可读的对抗提示后缀，通过交替优化算法训练，在 AdvBench 和 HarmBench 上实现高攻击成功率，且可迁移到闭源黑盒 LLM，同时展示了用生成的对抗后缀进行对抗训练以增强目标 LLM 鲁棒性的策略。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "对抗提示"
  - "LLM越狱"
  - "红队测试"
  - "自适应攻击"
  - "对抗训练"
---

# AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs

**会议**: ICML 2025  
**arXiv**: [2404.16873](https://arxiv.org/abs/2404.16873)  
**代码**: [https://github.com/facebookresearch/advprompter](https://github.com/facebookresearch/advprompter)  
**领域**: 优化/AI安全  
**关键词**: 对抗提示, LLM越狱, 红队测试, 自适应攻击, 对抗训练

## 一句话总结
提出 AdvPrompter——用一个 LLM（AdvPrompter）在秒级速度内为目标 LLM 生成人类可读的对抗提示后缀，通过交替优化算法训练，在 AdvBench 和 HarmBench 上实现高攻击成功率，且可迁移到闭源黑盒 LLM，同时展示了用生成的对抗后缀进行对抗训练以增强目标 LLM 鲁棒性的策略。

## 研究背景与动机

**领域现状**：LLM 经过安全对齐（RLHF 等）后仍然容易被越狱攻击（jailbreaking）——精心设计的提示可以绕过安全机制，诱导生成有害内容。

**现有痛点**：
   - 手动红队测试（manual red-teaming）耗时费力，无法规模化
   - 自动方法如 GCG（梯度优化 token）生成的后缀不可读、含义模糊，且需要数分钟的梯度搜索
   - AutoDAN 生成可读但不自适应于输入指令
   - PAIR 等基于 LLM 对话的方法速度慢（需多轮对话）

**核心矛盾**：高攻击成功率 vs 生成速度 vs 人类可读性 vs 输入自适应性——现有方法最多满足其中两个。

**本文目标**：同时实现四个目标——高成功率、秒级速度、人类可读、自适应输入。

**切入角度**：训练一个专门的"对抗提示生成 LLM"（AdvPrompter），而非在推理时搜索。训练完成后生成仅需一次前向传播（~1-2秒）。

**核心 idea**：将对抗提示搜索问题转化为 LLM 微调问题——AdvPrompter 学会自动为任意有害指令生成伪装后缀。

## 方法详解

### 整体框架
AdvPrompterTrain 交替执行两个步骤：
1. **AdvPrompterOpt**：基于梯度优化为当前有害指令生成目标后缀（离线，慢但高质量）
2. **AdvPrompter 微调**：用生成的(指令, 后缀)对微调 AdvPrompter LLM（使其学会快速生成类似后缀）
训练完成后，AdvPrompter 可在1-2秒内为任意新指令生成对抗后缀。

### 关键设计

1. **AdvPrompterOpt——梯度引导的目标后缀生成**:

    - 功能：为给定有害指令找到最优对抗后缀
    - 核心思路：
        - 目标：找到后缀 $s$ 使得 TargetLLM 对"指令+后缀"输出肯定回复
        - 损失函数：$\mathcal{L} = -\log P_{\text{target}}(\text{"Sure, here is"}| \text{instruction} + s)$
        - 在 token 嵌入空间中做连续松弛和投影梯度下降
        - 额外约束：后缀应人类可读（通过困惑度惩罚）——$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attack}} + \lambda \cdot \text{PPL}(s)$
    - 设计动机：GCG 的离散搜索效率低且生成不可读后缀；连续松弛+困惑度约束同时解决效率和可读性
    - 注意：此步骤仅在训练时使用，推理时完全不需要

2. **交替训练——从搜索到生成的蒸馏**:

    - 功能：将 AdvPrompterOpt 的"搜索能力"蒸馏到 AdvPrompter 的"生成能力"中
    - 核心思路：
        - 每轮：(a) 对一批有害指令运行 AdvPrompterOpt 得到目标后缀; (b) 用(指令, 后缀)对微调 AdvPrompter
        - 交替执行直到 AdvPrompter 学会自主生成高质量后缀
    - 设计动机：搜索是慢的但可以离线做，生成是快的但需要学习——交替训练将前者的质量转化为后者的速度

3. **推理时的自适应生成**:

    - 功能：对任意新有害指令在1-2秒内生成对抗后缀
    - 核心思路：将有害指令作为输入，AdvPrompter 自回归生成后缀——纯前向传播
    - 关键特性：不同指令生成不同后缀（自适应），而非使用通用攻击模板
    - 设计动机：GCG 需要对每个指令重新搜索（分钟级），AdvPrompter 一次前向传播搞定

4. **对抗训练增强鲁棒性**:

    - 功能：用 AdvPrompter 生成的对抗后缀来增强目标 LLM 的安全性
    - 核心思路：在安全微调数据中混入 AdvPrompter 生成的对抗样本→TargetLLM 学会拒绝这些攻击
    - 设计动机：攻击和防御是一体两面——快速生成攻击的能力天然可用于防御训练

### 损失函数 / 训练策略
- AdvPrompterOpt：攻击损失 + 困惑度正则化
- AdvPrompter 微调：标准语言建模损失（交叉熵）
- 交替优化：每轮先搜索后微调
- AdvPrompter 使用 Llama-2-7B 作为基架构

## 实验关键数据

### 主实验
AdvBench 攻击成功率（开源目标 LLM）：

| 方法 | Llama-2 ASR↑ | Vicuna ASR↑ | 速度 | 可读 | 自适应 |
|------|-------------|------------|------|------|--------|
| GCG | 56% | 98% | ~10min/条 | ✗ | ✗ |
| AutoDAN | 63% | 95% | ~5min/条 | ✓ | ✗ |
| PAIR | 12% | 54% | ~20轮对话 | ✓ | ✓ |
| **AdvPrompter** | **52%** | **96%** | **1-2sec** | **✓** | **✓** |

### 黑盒迁移攻击

| 目标 LLM | GCG 迁移 | AdvPrompter 迁移 |
|---------|---------|----------------|
| GPT-3.5 | 8% | **21%** |
| GPT-4 | 3% | **11%** |
| Claude-2 | 5% | **15%** |

### HarmBench 扩展评估

| 方法 | 标准ASR | 功能性ASR（实际有害性） |
|------|---------|---------------------|
| GCG | 47.6% | 32.1% |
| **AdvPrompter** | **49.8%** | **38.2%** |

### 对抗训练防御效果

| 训练策略 | 原始ASR | 对抗训练后ASR | 正常功能保留 |
|---------|---------|-------------|-----------|
| 无对抗训练 | 96% | - | ✓ |
| GCG 后缀训练 | 96%→82% | 14%↓ | ✓ |
| **AdvPrompter 后缀训练** | **96%→68%** | **28%↓** | **✓** |

### 消融实验

| 配置 | ASR (Vicuna) | 速度 | 说明 |
|------|-------------|------|------|
| 无困惑度约束 | 98% | 1s | 不可读后缀 |
| 困惑度约束弱 | 96% | 1s | 半可读 |
| **困惑度约束强** | **92%** | **1-2s** | 完全可读 |
| 交替训练 1轮 | 71% | 1s | 训练不足 |
| 交替训练 5轮 | 92% | 1s | 收敛 |
| **交替训练 10轮** | **96%** | **1-2s** | 最优 |

### 关键发现
- AdvPrompter 在速度上比 GCG 快 300-600×，同时保持竞争力的攻击成功率
- 人类可读后缀的迁移攻击成功率更高——因为黑盒 LLM 对"正常看起来的文本"更不设防
- 交替训练的蒸馏效果好——10轮后 AdvPrompter 的生成质量接近 AdvPrompterOpt 的搜索质量
- 对抗训练使用 AdvPrompter 后缀比 GCG 后缀效果更好——因为更多样化、更"像真正的攻击"
- 困惑度约束的强度是可读性-攻击力的trade-off旋钮

## 亮点与洞察
- **从搜索到生成的范式转换**——将对抗提示从"每次搜索"变为"一次训练，永久生成"，效率提升数百倍
- 交替优化的训练框架优雅地解决了"搜索质量高但慢"和"生成速度快但需要数据"的矛盾
- 人类可读性不仅是美学需求——更可读的对抗提示在黑盒迁移场景中更有效！这是一个反直觉但重要的发现
- 攻击即防御——用 AdvPrompter 同时做红队和蓝队，形成迭代安全改进循环
- 对 LLM 安全领域有基础性工具价值——持续、快速、多样化的红队测试

## 局限与展望
- 训练 AdvPrompter 仍需要目标 LLM 的梯度（AdvPrompterOpt 步骤）——对完全黑盒目标 LLM 不适用
- 随着目标 LLM 安全对齐的改进，AdvPrompter 可能需要重新训练
- 可读性和攻击成功率之间存在 trade-off——完全可读后缀的 ASR 略低
- 仅评估了英文——多语言越狱待探索
- 伦理风险：该工具可能被恶意使用——文章讨论了负责任使用策略

## 相关工作与启发
- **vs GCG**: 离散 token 搜索→不可读、慢；AdvPrompter 连续优化+蒸馏→可读、快
- **vs AutoDAN**: 使用固定模板变体→不自适应输入；AdvPrompter 对每个输入生成定制后缀
- **vs PAIR**: 多轮 LLM 对话→慢；AdvPrompter 一次前向传播→快
- **启发**：对抗攻击的"生成式"方法可能比"搜索式"方法更实用——类似从 MCMC 到 normalizing flow 的范式转换

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从搜索到生成的范式转换极具原创性
- 实验充分度: ⭐⭐⭐⭐⭐ 开源/闭源目标、AdvBench/HarmBench、对抗训练、充分消融
- 写作质量: ⭐⭐⭐⭐ 清晰直观
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 安全研究有基础工具价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Kernel Learning with Adversarial Features: Numerical Efficiency and Adaptive Regularization](../../NeurIPS2025/optimization/kernel_learning_with_adversarial_features_numerical_efficiency_and_adaptive_regu.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](../../NeurIPS2025/optimization/beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] MergeBench: A Benchmark for Merging Domain-Specialized LLMs](../../NeurIPS2025/optimization/mergebench_a_benchmark_for_merging_domain-specialized_llms.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](../../NeurIPS2025/optimization/efficient_adaptive_federated_optimization.md)

</div>

<!-- RELATED:END -->
