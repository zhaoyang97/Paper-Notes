# Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees

**会议**: NeurIPS 2025  
**arXiv**: [2505.18659](https://arxiv.org/abs/2505.18659)  
**代码**: 无  
**领域**: LLM评估 / 统计推断  
**关键词**: LLM evaluation, prediction-powered inference, autoevaluator, e-value, sample efficiency  

## 一句话总结
提出R-AutoEval+，通过e-value赌注算法自适应调整对合成数据（LLM评判器）的依赖权重，首次同时提供有限样本可靠性保证和可证明的采样效率改善，在GSM8K上比纯真实数据方法节省87个token。

## 背景与动机
用LLM作为评判器（autoevaluator/LLM-as-judge）可以低成本生成大量合成评估数据，减少方差，但引入了偏差——LLM评判不一定准确。Prediction-Powered Inference (PPI)可以结合真实数据和合成数据，但现有PPI方法缺乏采样效率保证，有时甚至不如仅用真实数据。需要一种方法在LLM评判器质量足够时受益，质量不足时自动退化到纯真实数据方法。

## 核心问题
如何在利用LLM评判器降低评估成本的同时，保证统计可靠性，并证明采样效率不会比纯真实数据方法差？

## 方法详解

### 整体框架
R-AutoEval+ = R-Eval（纯真实数据方法）+ 自适应PPI权重
- 当LLM评判器质量好时，增加对合成数据的依赖→更高效
- 当LLM评判器质量差时，自动减少依赖→退化为R-Eval→至少不比基线差

### 关键设计
1. **e-value赌注算法**：通过在线赌注协议（UP或WSR策略）自适应调整合成数据的权重，数学保证不可靠性风险有界
2. **有限样本可靠性保证**：不依赖渐近理论，在任意样本量下保证置信区间的覆盖率
3. **可证明采样效率**：当autoevaluator足够准确时，严格证明比R-Eval需要更少的真实数据
4. **自动回退**：当autoevaluator质量不足时，自动恢复为R-Eval

## 实验关键数据
| 任务 | R-AutoEval+ | R-Eval | R-AutoEval |
|------|------------|--------|-----------|
| TriviaQA量化 | 最优 | 次优 | 不可靠 |
| GSM8K推理预算 | **节省87 tokens** | 基线 | - |
| Instruct-Induction | 可靠高效 | 可靠但低效 | - |

- 可靠性参数：α=0.1, δ=0.1
- 在所有场景中保持可靠性

### 消融实验要点
- UP vs WSR赌注策略
- 候选因子S的影响：S≥2时最优
- in-context样本数对autoevaluator质量的影响

## 亮点
- **首个双保证方法**：同时有有限样本可靠性和采样效率保证
- **自适应退化**：最坏情况退化为R-Eval而非更差
- **理论-实践统一**：e-value赌注的理论框架在LLM评估中的巧妙应用
- **实用场景覆盖**：LLM量化评估、prompt选择、推理预算分配

## 局限性 / 可改进方向
- 需要真实世界未标注数据
- 效率保证仅在高可靠性水平(1-δ)下成立
- 候选因子集固定

## 评分
- 新颖性: ⭐⭐⭐⭐ e-value应用到LLM评估是新颖的交叉
- 实验充分度: ⭐⭐⭐⭐ 三个不同场景验证
- 写作质量: ⭐⭐⭐⭐ 理论保证清晰
- 价值: ⭐⭐⭐⭐ 对LLM自动评估的可靠性有重要意义
