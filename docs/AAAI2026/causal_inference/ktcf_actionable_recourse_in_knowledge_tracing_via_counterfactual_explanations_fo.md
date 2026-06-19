---
title: >-
  [论文解读] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education
description: >-
  [AAAI2026][因果推理][Knowledge Tracing] 提出 KTCF，一种面向知识追踪（KT）的反事实解释生成方法，通过考虑知识概念间关系生成稀疏且可操作的反事实解释，并将其后处理为顺序化的教学指令，在有效性、稀疏性和可操作性指标上全面超越基线方法。 - 深度学习驱动的知识追踪（Knowledge Trac…
tags:
  - "AAAI2026"
  - "因果推理"
  - "Knowledge Tracing"
  - "Counterfactual Explanation"
  - "XAI"
  - "Actionable Recourse"
  - "Education"
---

# KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education

**会议**: AAAI2026  
**arXiv**: [2601.09156](https://arxiv.org/abs/2601.09156)  
**代码**: 待确认  
**领域**: 因果推理  
**关键词**: Knowledge Tracing, Counterfactual Explanation, XAI, Actionable Recourse, Education  

## 一句话总结
提出 KTCF，一种面向知识追踪（KT）的反事实解释生成方法，通过考虑知识概念间关系生成稀疏且可操作的反事实解释，并将其后处理为顺序化的教学指令，在有效性、稀疏性和可操作性指标上全面超越基线方法。

## 背景与动机
- 深度学习驱动的知识追踪（Knowledge Tracing, KT）在教育领域广泛应用，用于建模学生的知识掌握程度并预测未来表现
- 欧盟 AI 法案将教育类 AI 归为高风险系统，要求模型具备可解释性；美国教育部也强调 AI 应保护人类决策权
- 现有 KT 可解释性研究主要关注"模型关注了什么"（attention heatmap）或"模型是否准确捕捉了学习过程"（心理测量参数），属于 what 类问题
- 缺乏回答"why"和"how"类问题的方法——即"为什么模型给出这个预测""学生应该怎么做才能改变预测结果"
- 反事实解释天然具有因果性和局部性，且对非专业的教育利益相关者（学生、教师、家长）易于理解，但在 KT 领域尚未被充分探索

## 核心问题
给定学生的学习历史序列 $X^{\text{orig}} = [(kc_1, r_1), \ldots, (kc_t, r_t)]$ 和训练好的 KT 模型 $f$，当模型对目标知识概念 $kc_{\text{target}}$ 预测为"答错"时，如何找到一组最小改动的反事实响应 $R^{\text{cf}}$，使得模型预测翻转为"答对"，同时保证：

1. **干预稀疏性**：改动尽可能少
2. **可操作性**：只允许将"答错"改为"答对"（不能反向操作）
3. **知识概念关系一致性**：建议修改的知识概念与目标概念在知识图谱上相关
4. **顺序化输出**：解释以有序的教学指令呈现，最小化学生的总学习负担

## 方法详解

### 反事实解释生成（Algorithm 1）
KTCF 将反事实生成建模为一个优化问题，损失函数为三项之和：

$$\mathcal{L}_{\text{KTCF}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{spar}} \cdot \mathcal{L}_{\text{spar}} + \lambda_{\text{kc}} \cdot \mathcal{L}_{\text{kc}}$$

- **预测损失** $\mathcal{L}_{\text{pred}}$：反事实输入经 KT 模型的预测值与目标概率 1.0 之间的二元交叉熵，驱动预测翻转
- **稀疏损失** $\mathcal{L}_{\text{spar}}$：原始响应与反事实响应之间的 Hamming 距离，鼓励最少改动
- **KC 损失** $\mathcal{L}_{\text{kc}}$：利用预定义的知识概念关系无向图 $G_{\text{kc}}$，对与目标 KC 图距离较远的改动进行惩罚，确保建议修改的 KC 与目标 KC 在知识结构上相关

使用 Adam 优化器进行随机优化，每步更新后通过**可操作性掩码** $\mathbf{m}$ 投影，确保只有原本答错的位置才能被修改，从而实现 100% 可操作性。

### 初始化策略
论文对比了 5 种初始化方法：高斯噪声（-rn）、随机二值（-rand）、soft relaxation（-sr）、凸组合（-cc）、Gumbel-Sigmoid relaxation（-gs）。实验表明软松弛类初始化（-rn, -sr, -gs）效果最优。

### 后处理：教学指令序列化（Algorithm 2）
将反事实解释转换为学生可执行的顺序学习路径：

1. 提取反事实中所有被修改的 KC 集合 $\overline{V}^{\text{CF}}$
2. 在 KC 关系图上用 Dijkstra 算法计算所有 KC 对之间的最短路径距离
3. 构造以这些 KC 为节点的完全子图，边权为最短路径距离
4. 用贪心策略从目标 KC 出发求解旅行商路径问题（TSPP），再反转路径
5. 输出的序列即为建议学生依次复习的 KC 顺序，最小化总学习负担（总路径距离）

## 实验关键数据

### 数据集与设置
- **数据集**：XES3G5M，包含 18,066 名学生的 5,549,635 条交互序列（数学领域）
- **KC 关系图**：1,175 个节点，1,304 条边
- **KT 模型**：DKT（LSTM），测试 AUC 0.8226，准确率 0.8253
- **评估**：从错误率 >45% 的学生中随机抽取 200 个实例

### 主要结果（Table 1）

| 方法 | Validity↑ | Sparsity↓ | Actionability↓ |
|------|----------|----------|----------------|
| Wachter-rand | 0.725 | 67.36 | 40.53 |
| DiCE-rand | 0.880 | 75.59 | 35.32 |
| **KTCF-rn** | **0.930** | **49.85** | **0.000** |
| **KTCF-gs** | 0.920 | 49.92 | **0.000** |

- KTCF-rn 相比 Wachter 有效性提升 28.3%，稀疏性降低 26.0%；相比 DiCE 有效性提升 5.7%，稀疏性降低 34.0%
- 所有 KTCF 变体的 Actionability 均为 0（完全消除不可操作修改），而基线方法存在大量不可操作建议
- KTCF-gs 计算速度最快（2.2s），比 Wachter 快 41.9%

### 定性分析（Table 2）
以"区分闰年与非闰年"为目标 KC 的实例：

- KTCF 仅建议 5 个 KC（取模运算、日历周期等），总路径距离 26
- Wachter 建议 14 个 KC（含无关的树状图、小数单位等），总路径距离 66
- DiCE 建议 20 个 KC，总路径距离 79

## 亮点
- **首次系统化**地将反事实 XAI 引入知识追踪领域，并提供了完整的概念化框架（explanandum/explanans）
- **可操作性保证**：通过掩码机制实现 100% 可操作性，确保只建议"把错的练对"，永远不会建议"故意答错"
- **教育理论根基**：与 Bloom 的掌握学习理论和 2-Sigma 问题对接，赋予方法教育学理论意义
- **KC 关系约束**：利用知识概念图谱嵌入损失函数，使建议的修改在知识结构上连贯
- **后处理方案**巧妙地将 TSPP 用于生成最优学习路径，减少学生学习负担

## 局限与展望
- 仅在 DKT（LSTM）上验证，未测试 Transformer 架构的 KT 模型（如 AKT、SAINT 等），泛化性有待验证
- KC 关系图依赖预定义结构，对无此类信息的数据集不适用
- 二值响应的反事实生成对初始化策略敏感，虽然 KC 损失提升了鲁棒性，但根本问题未解决
- 后处理采用贪心 TSPP 非最优解，路径质量有提升空间
- 缺乏真实用户研究验证实际教学效果（仅有定性案例分析）
- 评估仅基于 200 个实例，规模较小

## 与相关工作的对比
- **vs. Wachter/DiCE**：这两种是通用反事实方法，不考虑知识概念关系和教育领域的可操作性约束，导致建议冗余且含不可操作修改。KTCF 通过 KC 损失和可操作性掩码全面解决这些问题
- **vs. 注意力热力图解释**（AKT、EKT 等）：仅回答"模型关注了哪里"，不能提供可操作的教学建议
- **vs. 心理测量参数解释**（IRT-based）：预测难度/区分度参数用于诊断，但同样不提供"怎么改进"的行动指南
- **vs. SHAP/LRP 后验解释**（Lu et al. 2024）：解释特征贡献方向但不产生反事实指导；Lu 的用户研究表明解释能提升信任，KTCF 在此基础上更进一步提供可操作步骤

## 启发与关联
- 反事实解释的后处理思路（TSP 路径优化）可推广到其他需要排序建议的场景，如推荐系统中的学习路径规划
- KC 关系图约束的思想可迁移到医疗诊断等领域——用疾病关系图约束反事实建议的合理性
- 可操作性掩码是一个通用技巧，适用于任何具有单向可操作约束的反事实生成问题
- 未来结合 LLM 将反事实解释转化为自然语言教学指令是一个有前景的方向

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次将反事实 XAI 系统化应用于 KT，概念框架完整)
- 实验充分度: ⭐⭐⭐ (消融实验充分，但评估规模较小且缺少用户研究)
- 写作质量: ⭐⭐⭐⭐ (教育理论与技术方法结合自然，定性分析直观)
- 价值: ⭐⭐⭐⭐ (对教育 AI 可解释性有实际推动作用)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](../../CVPR2026/causal_inference/maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
