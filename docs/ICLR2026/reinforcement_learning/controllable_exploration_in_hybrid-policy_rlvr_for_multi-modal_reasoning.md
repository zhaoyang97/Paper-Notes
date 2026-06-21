---
title: >-
  [论文解读] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning
description: >-
  [ICLR 2026][强化学习][RLVR] CalibRL 将专家数据重新定义为分布校准基线（而非严格模仿目标），通过 LeakyReLU 不对称激活 + 优势加权实现对 MLLM 推理训练中探索-利用平衡的精细控制，解决 RLVR 中的熵崩溃问题，在几何推理等任务上大幅超越 GRPO/DAPO。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "RLVR"
  - "混合策略优化"
  - "多模态推理"
  - "熵崩溃"
  - "可控探索"
---

# Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning

**会议**: ICLR 2026  
**arXiv**: [2602.20197](https://arxiv.org/abs/2602.20197)  
**代码**: [https://github.com/zhh6425/CalibRL](https://github.com/zhh6425/CalibRL)  
**领域**: 多模态VLM / 强化学习  
**关键词**: RLVR, 混合策略优化, 多模态推理, 熵崩溃, 可控探索

## 一句话总结
CalibRL 将专家数据重新定义为分布校准基线（而非严格模仿目标），通过 LeakyReLU 不对称激活 + 优势加权实现对 MLLM 推理训练中探索-利用平衡的精细控制，解决 RLVR 中的熵崩溃问题，在几何推理等任务上大幅超越 GRPO/DAPO。

## 研究背景与动机

**领域现状**：RLVR 已成为提升 MLLM 推理能力的主流范式（如 DeepSeek-R1），但性能提升往往伴随策略熵的显著下降——熵枯竭成为进一步改进的瓶颈。

**现有痛点**：
   - 传统熵正则化鼓励随机性但无方向性→探索效率低
   - SFT-then-RL 范式中 SFT 将策略固定在静态演示分布→削弱后续 RL 探索
   - 混合策略框架中直接注入 SFT 监督→当前策略与专家轨迹的分布不匹配→高偏差方差→加速熵崩溃

**核心矛盾**：专家数据是双刃剑——提供有用指导但也压缩策略分布。将专家轨迹的概率 $\pi_\theta(\tau^{expert})$ 最大化必然降低其他轨迹概率→总熵下降

**切入角度**：不应"模仿"专家（绝对目标），而应以专家为"参考基线"做相对校准——欠表示的正确推理路径被强化，过度自信的错误预测被抑制。

**核心 idea**：将专家监督从刚性模仿信号转为精细化校准机制，通过 log-probability gap + LeakyReLU 不对称门控实现有方向、受管制的探索。

## 方法详解

### 整体框架
CalibRL 要解决的是 RLVR 训练 MLLM 时的熵崩溃：以往无论是 SFT-then-RL 还是把专家监督直接塞进 RL 的混合策略框架，本质都在最大化专家轨迹的对数似然，逼着策略向专家分布单向收拢——专家轨迹概率一抬高，其他轨迹概率必然被压低，总熵随之下降、探索枯竭。CalibRL 的破局点是换一个视角：不把专家当"必须达到的模仿目标"，而当成"衡量当前策略偏到哪一侧的参考坐标"，在 GRPO 之上挂一个可控探索项 $\mathcal{L}_{exploration}$ 做**相对校准**。

对一个 prompt 采样出的 group 里的每条响应，这套机制分三步走：先算模型对自己答案相对专家答案的偏好（log-probability gap $\Delta\ell_i$），判断模型相对专家是过自信还是欠自信；再结合答案对错，用不对称的 LeakyReLU 门控决定这条响应该往上推（强化）还是往下压（抑制）、并在它跨过专家基线后自动收力；最后按组内优势的绝对值加权，让罕见却有信息量的响应主导更新。三步分别管"信号""方向与力度""权重"，合成一个探索项后并入 GRPO 目标。

### 关键设计

**1. Log-Probability Gap：用模型自身 vs 专家的相对信心当探索信号**

熵正则化只鼓励随机性、没有方向，在 MLLM 巨大的状态空间里探索效率很低。CalibRL 想要"有方向的探索"，于是先定义一个相对量：

$$\Delta\ell_i = \log\frac{\pi_\theta(\tau_i^{policy}\mid q_i)}{\pi_\theta(\tau_i^{expert}\mid q_i)} = \log\pi_\theta(\tau_i^{policy}\mid q_i) - \log\pi_\theta(\tau_i^{expert}\mid q_i)$$

$\Delta\ell_i$ 为正说明模型已经更偏好自己生成的轨迹、相对专家更自信；为负说明模型对自身响应的信心还低于专家基线。后续要强化还是抑制，全部由这个相对信号驱动——这就是把专家从"模仿目标"降级为"参考基线"的关键：模型不需要逼近专家的概率，只需要知道自己相对专家偏到了哪一侧。

**2. LeakyReLU 不对称激活：跨过专家基线后就松手，避免过度自信**

有了信号还要决定方向与力度。CalibRL 先用一个独立的对错符号 $s_i$ 标记响应（正确 $s_i=+1$、错误 $s_i=-1$，单独定义而非直接用归一化奖励，是因为含格式奖励的实际 reward 可能超出 $[0,1]$）。探索项写成：

$$\mathcal{L}_{exploration} = |\hat{A}_i| \cdot \text{LeakyReLU}(-s_i \cdot \Delta\ell_i, \alpha)$$

把 $s_i$ 乘进去保证优化方向始终和正确性对齐：对正确但模型还不够自信的响应（$\Delta\ell_i$ 偏负）要强化，对错误却被模型过度自信的响应（$\Delta\ell_i$ 偏正）要抑制。LeakyReLU 的不对称正好刻画"力度该多大"——输入为正时梯度为 1（响应还在专家基线的"该管"那一侧，全力推/压），输入为负时梯度缩放成 $\alpha\in(0,1)$（响应概率一旦跨过专家基线，就减弱进一步的强化/抑制，防止把策略再次推向过度自信）。这里特意没用纯 ReLU（输入为负直接切断梯度会丢掉有用信号），也没用线性激活（无法区分"还需要管"和"已经够了"两个区域），LeakyReLU 在两者之间取了平衡。

**3. 优势加权：让罕见但有信息量的响应主导更新**

最后用 group-wise 的优势绝对值 $|\hat{A}_i|$ 给每条响应的探索项加权。GRPO 的优势本身就是组内归一化的，当一个 group 里多数响应都错、只有少数正确时，那条罕见正确响应的 $|\hat{A}_i|$ 会很大，于是它获得更大权重、被当成宝贵的探索信号重点强化；反过来在多数都对的组里出现一条罕见错误响应时，加权也会放大对它的抑制；大家都对/都错时权重小、更新温和。这样更新幅度就和"这条响应有多稀有、多有信息量"挂钩，把探索集中在真正能学到东西的偏差上。

### 损失函数 / 训练策略
- 最终目标把探索项并入 GRPO（论文式 10）：$\mathcal{J}(\theta) = \mathcal{J}_{GRPO}(\theta) - \lambda\,\mathcal{L}_{exploration}$，$\lambda$ 平衡标准 PPO 式优化与专家引导的探索（长 CoT 训练下 GRPO 的 KL 项通常省略）
- 关键超参：$\alpha=0.5$（LeakyReLU 斜率，消融里是 sweet spot），$\lambda$ 控制探索权重
- 专家数据来自 ViRL39K 几何题用 GPT-4o 生成的 CoT（经正确性/格式/逻辑三重校验，约 9.7K 条）；消融证实专家基线优于参考策略基线

## 实验关键数据

### 主实验（几何推理，任务内基准）

| 方法 | GeoEval↑ | Geo3K↑ | GeoQA↑ | 平均↑ |
|------|---------|--------|--------|------|
| GRPO | 26.15 | 39.77 | 52.52 | 39.48 |
| SFT+GRPO | 6.00 | 18.64 | 40.98 | 21.87 |
| DAPO | 25.19 | 40.93 | 52.52 | 39.55 |
| **CalibRL** | **33.44** | **40.60** | **60.74** | **44.93** |

### 消融实验

| LeakyReLU $\alpha$ | 效果 |
|-------------------|------|
| 0.3 | 早期激进探索但不稳定，熵波动 |
| **0.5** | **平衡的熵增长，无震荡** |
| 0.8 | 过度约束，快速熵衰减 |

### 关键发现
- **SFT+GRPO 反而最差**：直接混合 SFT 和 RL 导致严重熵崩溃——支持"模仿→校准"范式转换的必要性
- **CalibRL 在任务内和任务外均最优**：平均超 GRPO +5.45%（任务内），泛化也更好
- **熵曲线稳定**：CalibRL 训练过程中策略熵平稳增长，其他方法则持续下降
- **$\alpha=0.5$ 是 sweet spot**：过小→不稳定，过大→探索受限

## 亮点与洞察
- **"校准而非模仿"是对混合策略 RL 的深刻重新理解**——专家数据不应被视为必须达到的目标，而应是衡量当前策略偏差的参考坐标。
- **LeakyReLU 提供了优雅的梯度门控机制**——一个简单的激活函数选择就实现了"需要时强化、足够时减弱"的自适应控制。
- **"SFT+GRPO 反而最差"的发现对实践有重要警示**——简单叠加 SFT 和 RL 可能互相抵消。

## 局限与展望
- 仅在几何推理上验证——数学/代码推理等场景待探索
- LeakyReLU 的 $\alpha$ 需要调优——更自适应的激活函数设计可能更好
- 专家数据质量影响校准基线的可靠性
- 未探索与其他 RL 变体（Dr.GRPO, CPPO）的结合

## 相关工作与启发
- **vs GRPO/DAPO**: 标准策略优化不处理熵崩溃；CalibRL 通过校准机制维持探索
- **vs LUFFY**: 也是混合策略框架但仍用模仿式监督→仍有分布不匹配
- **vs SFT+GRPO**: 直接串联导致灾难性干扰；CalibRL 的校准范式避免了这个问题

## 评分
- 新颖性: ⭐⭐⭐⭐ "校准而非模仿"的范式有洞察力，LeakyReLU 应用巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融充分但任务范围偏窄（主要是几何推理）
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，理论动机清晰
- 价值: ⭐⭐⭐⭐ 解决 RLVR 熵崩溃的实用方案，对混合策略训练有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](../../ACL2026/reinforcement_learning/healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] Generalization of RLVR Using Causal Reasoning as a Testbed](generalization_of_rlvr_using_causal_reasoning_as_a_testbed.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](../../ACL2026/reinforcement_learning/semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)

</div>

<!-- RELATED:END -->
