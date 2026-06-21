---
title: >-
  [论文解读] Why is Your Language Model a Poor Implicit Reward Model?
description: >-
  [ICLR 2026][Reasoning][隐式奖励模型] 本文通过理论和实验揭示了隐式奖励模型（IM-RM，如DPO）比显式奖励模型（EX-RM）泛化更差的根本原因——IM-RM过度依赖表面token级线索而非语义表示，导致在token分布偏移下准确率大幅下降，同时反驳了"生成-验证差距"假说。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "隐式奖励模型"
  - "显式奖励模型"
  - "泛化差距"
  - "token级线索"
  - "DPO vs RLHF"
---

# Why is Your Language Model a Poor Implicit Reward Model?

**会议**: ICLR 2026  
**arXiv**: [2507.07981](https://arxiv.org/abs/2507.07981)  
**代码**: 无  
**领域**: LLM推理 / 对齐RLHF  
**关键词**: 隐式奖励模型, 显式奖励模型, 泛化差距, token级线索, DPO vs RLHF

## 一句话总结
本文通过理论和实验揭示了隐式奖励模型（IM-RM，如DPO）比显式奖励模型（EX-RM）泛化更差的根本原因——IM-RM过度依赖表面token级线索而非语义表示，导致在token分布偏移下准确率大幅下降，同时反驳了"生成-验证差距"假说。

## 研究背景与动机

**领域现状**：奖励模型是LLM后训练和推理管线的核心组件。目前主流有两种近乎相同的奖励模型：显式奖励模型（EX-RM，在隐藏表示上加线性头）和隐式奖励模型（IM-RM，通过 $\ln \pi_\theta(\mathbf{y}|\mathbf{x})$ 隐式定义奖励，即DPO的核心思想）。两者可以使用相同的数据、损失函数和基础语言模型训练，唯一区别在于奖励的计算方式。

**现有痛点**：尽管EX-RM和IM-RM几乎完全相同，先前工作反复观察到IM-RM的泛化能力明显更差，尤其是在分布外评估中排名响应的准确率更低。这个泛化差距非常令人困惑——为什么计算奖励方式的微小差异会导致如此大的性能差距？

**核心矛盾**：直觉上一种解释是"生成-验证差距"——IM-RM既要给正确回答打高分，又要通过底层语言模型生成正确回答，如果生成比验证更难，IM-RM的准确率就应该落后。但这个直觉论证是否成立？真正的原因到底是什么？

**本文目标**
   - 反驳"生成-验证差距"假说：证明IM-RM的验证并不需要学会生成
   - 找到真正原因：从学习动力学角度刻画EX-RM和IM-RM的行为差异
   - 实验验证：在受控和真实场景下验证理论预测

**切入角度**：从梯度更新对未见样本奖励的影响出发，分析学习动力学。发现EX-RM的奖励变化只依赖于隐藏表示的内积，而IM-RM的变化还额外依赖于具体token。

**核心 idea**：IM-RM之所以泛化差，是因为其学习动力学天然倾向于过拟合表面token级线索，而非利用语义层面的隐藏表示结构。

## 方法详解

### 整体框架
本文不提新方法，而是回答一个长期困惑的问题：EX-RM 和 IM-RM 几乎是同一个东西——同样的数据、同样的损失、同样的基座语言模型，唯一区别是奖励怎么算（EX-RM 在隐藏表示上接线性头，IM-RM 直接用 $\ln\pi_\theta$ 隐式定义），可为什么 IM-RM 的泛化总是更差？作者分三步推进：先用一个反例干掉社区里流行的"生成-验证差距"解释；再从单步梯度更新的学习动力学切入，证明 EX-RM 的奖励变化只看隐藏表示、而 IM-RM 还被具体 token 牵着走；最后在受控数据集和真实的 1B–8B 模型上，验证这个 token 依赖性正是泛化差距的根源。

### 关键设计

**1. 反驳"生成-验证差距"假说：验证一个答案，并不需要会生成它**

流行的直觉是：IM-RM 既要给好答案打高分，又要靠底层语言模型把好答案生成出来，而生成比验证难，所以它的验证准确率被拖累。Theorem 1 直接构造反例打掉这个论证——存在一个分布 $\pi$，它诱导的 IM-RM 能以 margin $\delta$ 把正确性验证出来，但 $\pi$ 生成正确回答的概率相比参考分布 $\pi_{\text{ref}}$ 最多只涨一个常数因子 $\exp(\delta/\beta)$。换句话说，如果 $\pi_{\text{ref}}$ 本身就无法高效生成，$\pi$ 也照样生成不好，却不妨碍它成为一个好验证器，验证能力和生成能力被解耦了。实验把反例坐实：在 NP-hard 的哈密顿回路验证任务上，基于 Pythia-1B 的 IM-RM 测试准确率达到 0.993，但它生成出的正确哈密顿回路数为 0。

**2. EX-RM 的学习动力学：奖励变化只由隐藏表示的相似度决定**

要找真正的原因，作者去看一次梯度更新对某个未见样本 $(\bar{\mathbf{x}}, \bar{\mathbf{y}})$ 奖励的影响。在固定隐藏表示的假设下（Assumption 1），EX-RM 的奖励变化为

$$\Delta r_{\theta_{\text{EX}}}(\bar{\mathbf{x}}, \bar{\mathbf{y}}) = \langle \mathbf{h}_{\bar{\mathbf{x}},\bar{\mathbf{y}}}, \mathbf{h}_{\mathbf{x},\mathbf{y}^+} - \mathbf{h}_{\mathbf{x},\mathbf{y}^-} \rangle \cdot \eta g(\theta_{\text{EX}})$$

右边的 $\eta$ 是学习率、$g(\theta_{\text{EX}})>0$ 是一个正标量（等于 $\sigma$ 作用在当前奖励间隔上，刻画模型在这对训练样本上还有多"错"），两者都恒为正、不改变方向。真正决定奖励涨还是跌的，是隐藏表示之间的内积：只要未见样本 $\bar{\mathbf{y}}$ 与正样本 $\mathbf{y}^+$ 语义相近（表示对齐），奖励就增加，和用的是哪些具体 token 无关。由于预训练表示本身已经编码了语义，EX-RM 天然能把"换了 token 但意思一样"的回答泛化对。

**3. IM-RM 的学习动力学：token 不匹配时，奖励可能被反向推**

同样看奖励变化，IM-RM 的表达式里给每对位置都乘上一个系数 $\rho_{k,l}(\mathbf{v})\in[-2,2]$，它由两段回答在第 $k$、$l$ 位的 token 及其 next-token 分布共同决定。当 $\bar{\mathbf{y}}_k = \mathbf{v}_l$（token 对得上）时系数为正，起的作用和 EX-RM 类似；可一旦 $\bar{\mathbf{y}}_k \neq \mathbf{v}_l$（token 对不上），系数就可能变负——这时哪怕两个回答的隐藏表示语义对齐，奖励也可能被往**反方向**推。于是出现一个反直觉的局面：语义相同但 token 不同的 response，会被 IM-RM 判成相反的奖励方向。这正解释了为什么把回答做一次 paraphrase，IM-RM 的准确率能从 1.0 直接掉到 0.02。

**4. 理论泛化差距（Theorem 2）：IM-RM 对没见过的 token 只能瞎猜**

在单 token 回答的简化设定下，作者把上面这件事推到极致并严格证明：训练收敛后，IM-RM 对任何"没在训练集出现过的 token 对"的奖励差恒等于初始常数，于是排序准确率被钉死在 0.5（随机水平）。与之对照，EX-RM 的线性头方向会收敛到最大间隔分离超平面 $\mathbf{u}^*$，凡是 $\mathbf{u}^*$ 能分对的样本它都排得对。这里的假设确实偏强（单 token、固定表示），但后续实验表明，换到全参数训练、任意长度回答时，结论照样成立。

### 损失函数 / 训练策略
两类模型均使用 Bradley-Terry 对数似然损失训练：$\mathcal{L}(r) = \frac{1}{|\mathcal{D}_T|} \sum -\ln \sigma(r(\mathbf{x}, \mathbf{y}^+) - r(\mathbf{x}, \mathbf{y}^-))$

## 实验关键数据

### 主实验（受控环境：Persona数据集）

| 评估条件 | EX-RM准确率 | IM-RM准确率 |
|----------|------------|------------|
| 原始回答 (训练集) | 1.00 | 1.00 |
| 原始回答 (测试集) | 1.00 | 1.00 |
| Paraphrase回答 (训练集) | 1.00 | **0.022** |
| Paraphrase回答 (测试集) | 1.00 | **0.019** |

IM-RM在token分布发生变化（同义改写）时准确率几乎为0，而EX-RM完美泛化。

### 真实场景实验（UltraFeedback训练，6个1B-8B模型）

| 评估类型 | EX-RM准确率 | IM-RM准确率 | EX-RM奖励间隔 | IM-RM奖励间隔 |
|---------|------------|------------|-------------|-------------|
| 分布内 | **0.752** | 0.646 | **1.014** | 0.813 |
| Token级偏移 | **0.665** | 0.602 | **0.976** | 0.763 |
| 领域偏移 | 0.621 | **0.720** | **0.807** | 0.726 |

### 消融实验

| 配置 | 关键发现 |
|------|---------|
| 哈密顿回路验证 | IM-RM测试准确率0.993，但生成正确回路数=0，证明验证≠生成 |
| 中间token表示的EX-RM | 排除了"EX-RM用全序列表示而IM-RM用中间表示"的解释 |
| 无参考分布的IM-RM | 排除了"参考分布偏移"的解释 |
| Token偏移（翻译/改写） | EX-RM在法语/西班牙语翻译后仍大幅优于IM-RM |

### 关键发现
- IM-RM在**token级偏移**下一致弱于EX-RM（改写、翻译场景），但在**领域偏移**下可能持平甚至更好
- EX-RM始终产生更大的绝对奖励间隔（absolute reward margin），这对后续RL优化有利
- 分布内评估中IM-RM也弱于EX-RM，因为分布内测试样本与训练样本语义相似但token不同，更接近token偏移

## 亮点与洞察
- **学习动力学视角的创新性**：通过分析单步梯度更新的影响，精确刻画了EX-RM（只看表示）和IM-RM（还看token）的本质差异。这个角度既优雅又有强解释力，远超"生成-验证差距"这一直觉论证
- **反直觉的"反驳"**：Theorem 1 + 哈密顿回路实验干净利落地反驳了流行的"生成-验证差距"假说，令人印象深刻
- **"越改写越差"现象的理论解释**：$\rho_{k,l}$ 系数的正负取决于token是否匹配，这个发现可以指导DPO实践——例如，在DPO训练集中加入同义改写的正负样本对，可能有效缓解IM-RM的脆弱性
- **对RLHF vs DPO之争的新视角**：提供了DPO弱于RLHF的一个新理论解释（token级过拟合），且与已有的"生成-验证差距"解释互补

## 局限与展望
- 理论分析假设固定隐藏表示（Assumption 1）和单token回答（Assumption 2），虽然实验验证了结论的普适性，但更一般的理论保证仍然缺失
- 只考虑了准确率作为评估指标，未探讨奖励模型在实际RL训练中的下游影响
- 发现了IM-RM在**领域偏移**时可能优于EX-RM，但未深入分析原因——何时应选IM-RM？
- **可改进方向**：能否设计一种"token-invariant"的IM-RM训练方式（如在DPO训练中加入paraphrase数据增强）来弥补泛化差距？

## 相关工作与启发
- **vs DPO (Rafailov et al., 2023)**：DPO本质上就是IM-RM，本文揭示了它泛化弱于RLHF（先训EX-RM再RL优化）的一个关键原因
- **vs Swamy et al. (2025)**：他们主张"生成-验证差距"是DPO弱于RLHF的原因，本文直接反驳这一假说（至少在奖励模型准确率层面）
- **vs Im & Li (2025)**：他们在类似条件下证明IM-RM在**相同回答不同prompt**时能泛化，而本文证明在**不同回答**时IM-RM无法泛化，更贴近现实场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 学习动力学分析角度新颖，token依赖性的发现极具洞察力
- 实验充分度: ⭐⭐⭐⭐ 受控+真实场景+多种消融，排除了多个替代假说，但缺少下游RL任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链条极其清晰：反驳旧假说→提出新解释→理论证明→实验验证
- 价值: ⭐⭐⭐⭐⭐ 对RLHF/DPO社区有重大指导意义，揭示了reward model设计的隐含偏差

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reasoning with Sampling: Your Base Model is Smarter Than You Think](reasoning_with_sampling_your_base_model_is_smarter_than_you_think.md)
- [\[ICLR 2026\] OR-PRM: A Process Reward Model for Algorithmic Problem in Operations Research](or-prm_a_process_reward_model_for_algorithmic_problem_in_operations_research.md)
- [\[ICLR 2026\] R-HORIZON: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](r-horizon_how_far_can_your_large_reasoning_model_really_go_in_breadth_and_depth.md)
- [\[ICML 2026\] The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning](../../ICML2026/llm_reasoning/the_quality-utility_paradox_why_high-reward_data_impairs_small_model_mathematica.md)
- [\[ICLR 2026\] Enhancing Language Model Reasoning with Structured Multi-Level Modeling](enhancing_language_model_reasoning_with_structured_multi-level_modeling.md)

</div>

<!-- RELATED:END -->
