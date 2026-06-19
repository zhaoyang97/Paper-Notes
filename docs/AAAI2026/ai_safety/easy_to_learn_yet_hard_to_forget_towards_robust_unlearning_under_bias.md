---
title: >-
  [论文解读] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias
description: >-
  [AAAI2026][AI安全][machine unlearning] 提出 CUPID 框架，通过损失景观的锐度分析将遗忘集划分为因果/偏差子集，并识别和分离模型中的因果/偏差通路，实现对有偏模型的精准类别遗忘，有效解决"捷径遗忘"问题。 机器遗忘（machine unlearning）的目标是高效移除预训练模型中特定数…
tags:
  - "AAAI2026"
  - "AI安全"
  - "machine unlearning"
  - "shortcut learning"
  - "data bias"
  - "loss landscape"
  - "causal pathway"
---

# Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias

**会议**: AAAI2026  
**arXiv**: [2602.21773](https://arxiv.org/abs/2602.21773)  
**代码**: 待确认  
**领域**: AI安全  
**关键词**: machine unlearning, shortcut learning, data bias, loss landscape, causal pathway

## 一句话总结

提出 CUPID 框架，通过损失景观的锐度分析将遗忘集划分为因果/偏差子集，并识别和分离模型中的因果/偏差通路，实现对有偏模型的精准类别遗忘，有效解决"捷径遗忘"问题。

## 背景与动机

机器遗忘（machine unlearning）的目标是高效移除预训练模型中特定数据的影响，以满足"被遗忘权"等隐私法规要求。现有方法普遍假设目标信息在模型参数中是可分离的，但现实数据常包含虚假关联（spurious correlation）——例如"waterbird"类别与"水面"背景高度相关。模型会学到这种捷径（shortcut），导致类别特征与偏差特征深度纠缠。

作者首次系统研究了在有偏模型上执行遗忘算法的行为，发现两个关键现象：

1. **"易学难忘"不对称性**：偏差对齐样本（bias-aligned，即虚假特征与类别标签一致的样本）学习最快却最难遗忘；偏差冲突样本（bias-conflicting）反而容易被遗忘
2. **反直觉的去偏效应**：遗忘过程反常地提升了目标类别偏差冲突样本的准确率

这两个现象共同构成了作者定义的 **shortcut unlearning**——模型在被要求遗忘目标类别时，主要擦除的是虚假捷径特征而非真正的因果类别特征。

## 核心问题

如何在模型内部表征高度纠缠的情况下，让遗忘算法精准擦除因果性的类别信息，而非走捷径只擦除偏差特征？核心挑战包括：

- 需要区分模型中依赖因果特征 vs 捷径特征的参数
- 需要对不同参数子集施加不同的更新策略
- 整个过程不应依赖保留集（retain set），以适应隐私约束场景

## 方法详解

CUPID（Causal Unlearning via Pathway Identification and Disentanglement）包含三个阶段：

### 阶段一：Sharpness-Aware Partitioning（基于锐度的样本划分）

核心直觉来自泛化理论：模型对"容易学"的偏差对齐样本收敛到平坦最小值（低曲率），对"难学"的偏差冲突样本处于尖锐区域（高曲率）。

对遗忘集中每个样本计算局部锐度：

- 先在梯度方向上做 $\eta$ 步长的对抗扰动：$\theta_{adv} = \theta_o + \eta \frac{\nabla L(\theta_o, x_i)}{\|\nabla L(\theta_o, x_i)\|}$
- 锐度定义为扰动前后的损失差：$\omega_{sharpness}(x_i) = L(\theta_{adv}, x_i) - L(\theta_o, x_i)$

按锐度值的 top-$k$% 阈值，将遗忘集划分为：

- $\mathcal{D}_f^{bias}$（锐度低，近似偏差对齐样本）
- $\mathcal{D}_f^{causal}$（锐度高，近似偏差冲突/因果样本）

实验表明 $k=5\%$ 效果最优——并非越"纯"越好，适度包含部分偏差对齐样本可以正则化因果梯度方向。

### 阶段二：Causal Pathway Identification（因果通路识别）

目标是将模型参数 $\theta_o$ 分离为因果通路和偏差通路。对每个参数 $\theta_{o,i}$，结合其幅值和 Hessian 对角元素定义因果掩码：

$$m_c(\theta_{o,i}) = \mathbb{1}\left(\frac{1}{2}\theta_{o,i}^2 \cdot \mathbb{E}_{x \sim \mathcal{D}_f^{causal}}[H(\theta_o, x)_{ii}] \geq \tau_p\right)$$

其中 $\tau_p$ 设为选择前 50% 最具影响力的参数。该设计借鉴了经典的网络剪枝思想（LeCun et al. 1989），用参数幅值 × 二阶导数衡量参数的显著性。$m_c=1$ 的参数构成因果通路，其余为偏差通路。

### 阶段三：Targeted Pathway Update（定向通路更新）

对两条通路施加不同的梯度更新：

1. 计算因果梯度方向 $g_{causal}$（在 $\mathcal{D}_f^{causal}$ 上的平均梯度）
2. 将全遗忘集梯度 $g_f$ 投影到因果方向：$g_{proj} = \frac{g_f \cdot g_{causal}}{\|g_{causal}\|^2} g_{causal}$
3. 正交分量作为偏差梯度：$g_{bias} = g_f - g_{proj}$

最终更新规则：

$$\theta_{t+1} \leftarrow \theta_t + \alpha \cdot [(\omega_{sharpness} \cdot g_{proj} \odot m_c) + (g_{bias} \odot (1 - m_c))]$$

- 因果通路（$m_c=1$）：用投影因果梯度更新，并以样本锐度加权，对"难样本"施加更强遗忘力度
- 偏差通路（$m_c=0$）：仅用偏差梯度更新，避免误删因果信息

## 实验关键数据

在三个有偏数据集上评估（训练集偏差比例 99.5:0.5，测试集 50:50）：

**有偏训练集上的遗忘效果（Table 1）**：

| 方法 | Waterbirds FA↓ | BAR FA↓ | NICO++ FA↓ |
|------|----------------|---------|------------|
| Retrain（上界） | 0.00 | 0.00 | 0.00 |
| NegGrad | 34.96 | 58.59 | 22.33 |
| DELETE | 18.42 | 34.86 | 27.84 |
| **CUPID** | **6.91** | **7.70** | **7.71** |

**无偏测试集上的泛化遗忘（Table 2）**：

| 方法 | Waterbirds FA↓ | BAR FA↓ | NICO++ FA↓ |
|------|----------------|---------|------------|
| Retrain | 0.00 | 0.00 | 0.00 |
| DELETE | 8.73 | 34.38 | 22.95 |
| **CUPID** | **6.02** | **3.75** | **8.34** |

CUPID 在所有数据集上均取得最低 FA，且 $\triangle_{gap}$ 和 WGA 也最低，表明遗忘效果在偏差对齐和偏差冲突两组样本间最均衡。

**消融实验（Table 3，Waterbirds）**：

| 锐度划分 | 通路识别 | 定向更新 | FA↓ |
|----------|---------|---------|-----|
| ✗ | ✗ | ✗ | 34.96 |
| ✓ | ✗ | ✗ | 20.38 |
| ✓ | ✓ | ✗ | 14.56 |
| ✓ | ✓ | ✓ | **6.91** |

三个组件逐步贡献，缺一不可。

## 亮点

- **问题定义新颖**：首次形式化 shortcut unlearning 问题，揭示了"易学难忘"的不对称现象，指出现有遗忘方法在有偏数据上的根本性失败模式
- **方法设计精巧**：利用损失景观几何性质（平坦 vs 尖锐区域）作为无监督信号区分样本类型，无需偏差标签
- **不需要保留集**：CUPID 仅使用遗忘集即可工作，在隐私受限场景下更实用
- **效果显著**：在 BAR 数据集上 FA 仅 3.75%，而次优方法高达 30.26%，性能差距极大
- **Grad-CAM 可视化**清晰展示 CUPID 注意力从虚假特征转移，证实了方法的有效性

## 局限与展望

- 仅在图像分类任务上验证，未涉及 NLP 或生成模型
- Hessian 对角线的计算成本可能在大规模模型上成为瓶颈
- 锐度阈值 $k$ 和通路比例 $\tau_p$ 需要调参，对不同偏差强度可能需要不同设置
- 只考虑了单一偏差属性的场景，多偏差共存时的行为未探索
- 未讨论在 LLM 概念遗忘（如 RLHF unlearning）上的适用性

## 与相关工作的对比

- **NegGrad**（直接取反梯度）：在有偏数据上表现糟糕，FA 高达 34.96%，因为梯度取反优先擦除最显著的捷径特征
- **SALUN**（基于显著性的遗忘）：尝试选择关键参数更新，但未区分因果/偏差通路，在有偏数据上仍受限
- **Bad Teaching**（不需要保留集）：使用不胜任教师进行蒸馏，但在偏差场景下 FA 极高（Waterbirds 88.35%）
- **DELETE**（最新蒸馏方法）：是最强基线，但仍存在明显的 $\triangle_{gap}$，说明遗忘不均衡

CUPID 的核心优势在于将"参数选择"从单纯的显著性分析提升为"因果 vs 偏差通路"的分离，这是其他方法所缺乏的。

## 启发与关联

- 损失景观锐度作为无监督信号区分样本类型的思路具有普适性，可能拓展到数据清洗、异常检测等领域
- "因果通路 vs 偏差通路"的分离框架可能启发模型去偏（debiasing）方向的后续工作
- 与模型编辑（model editing）领域的参数定位思想有共通之处，可以交叉借鉴
- Shortcut unlearning 问题提醒我们：在讨论 LLM 安全对齐的"遗忘"能力时，同样需要警惕模型是否只是遗忘了表面模式

## 评分
- 新颖性: 9/10（首次形式化 shortcut unlearning，问题洞察深刻）
- 实验充分度: 8/10（三个数据集 + 消融 + 可视化，但缺乏 NLP 实验）
- 写作质量: 9/10（分析驱动的叙事结构清晰，图表说服力强）
- 价值: 8/10（为有偏场景下的遗忘问题提供了坚实基础）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](../../ICML2026/ai_safety/how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](../../ICML2026/ai_safety/forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[ICML 2025\] Rethinking the Bias of Foundation Model under Long-tailed Distribution](../../ICML2025/ai_safety/rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)
- [\[CVPR 2026\] Bias at the End of the Score](../../CVPR2026/ai_safety/bias_at_the_end_of_the_score.md)

</div>

<!-- RELATED:END -->
