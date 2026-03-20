# Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering

**会议**: ACL 2025  
**arXiv**: [2502.13962](https://arxiv.org/abs/2502.13962)  
**代码**: [https://github.com/wjurayj/final_answer](https://github.com/wjurayj/final_answer)  
**领域**: LLM效率  
**关键词**: test-time scaling, selective question answering, confidence calibration, compute budget, abstention  

## 一句话总结
首次评估 test-time scaling 模型（DeepSeek R1、S1）在选择性问答（允许拒绝回答）场景下的表现，发现增加推理计算不仅提高正确率还提升对正确答案的置信度，提出基于置信度阈值的选择函数和 "Jeopardy Odds" 效用函数来评估非零错误惩罚下的 test-time scaling 性能。

## 研究背景与动机
1. **领域现状**：Test-time scaling（如 DeepSeek R1、S1）通过增长推理链长度在数学推理 benchmark 上取得突破性进展，但所有评估都在"零风险"设置下（答错不扣分，模型总是回答）
2. **现有痛点**：
   - 现实场景中错误回答通常有可衡量的代价（医疗诊断、法律咨询、自动驾驶）
   - 模型"猜测"在不确定时是否有价值？"拒绝回答"比"答错"更好的场景被完全忽略
   - 不同 test-time scaling 模型的置信度校准能力差异未被揭示
3. **核心矛盾**：test-time scaling 研究只报告准确率，忽视了模型是否"知道自己对不对"的能力
4. **本文要解决什么？** 在允许拒绝回答的场景下评估 test-time scaling，并提出标准化的评估方法
5. **切入角度**：用推理链末尾的 token log-probability 作为置信度指标，结合不同错误惩罚的效用函数
6. **核心idea一句话**：增加 test-time compute 不仅找到更多正确答案，还增强了模型区分正确/错误答案的置信度，但不同模型的校准能力差异巨大

## 方法详解

### 整体框架
评估框架定义两个轴：(1) **Compute Budget**（推理 token 数，500-8000），通过 budget forcing 严格控制；(2) **Confidence Threshold**（置信度阈值，0/0.5/0.95），低于阈值的回答被拒绝。在三种 utility 设置下评估。

### 关键设计

1. **置信度提取**:
   - 做什么：量化模型对回答的确信程度
   - 核心思路：使用答案 token 的 log-probability 之和作为置信度。所有答案格式统一（3 位数字），确保 token 数一致
   - 设计动机：简单直接，且直接来自模型内部信号——不需要额外的置信度估计模块

2. **选择函数（Selection Function）**:
   - 做什么：根据置信度决定是否输出回答
   - 核心思路：只接受置信度高于阈值的回答，否则弃权。评估 threshold ∈ {0.0, 0.5, 0.95}
   - 与标准评估的区别：threshold=0 等价于标准评估（永远回答），threshold>0 允许弃权

3. **效用函数（Utility Function）**:
   - 做什么：在不同错误惩罚下衡量模型价值
   - 核心思路：$f(\mathcal{M}, x) = \{1 \text{ if correct}, 0 \text{ if abstain}, r_t \text{ if wrong}\}$
   - 三种设置：Exam Odds ($r_t=0$，答错不扣分)、**Jeopardy Odds** ($r_t=-1$，答错扣等额分)、High-Stakes ($r_t=-20$，答错扣 20 倍)
   - 设计动机：Exam Odds 是当前标准但不现实；Jeopardy Odds 更贴近实际——答错比不答更差

## 实验关键数据

### 主实验

**AIME 2024+2025, DeepSeek R1-32B**:

| Threshold | Budget 500 Acc | Budget 4000 Acc | Budget 8000 Acc | Response Rate |
|:---------:|:-------------:|:---------------:|:---------------:|:------------:|
| 0.0 | ~30% | ~55% | ~60% | 100% |
| 0.5 | ~40% | ~65% | ~70% | ~70% |
| 0.95 | ~80% | ~85% | ~80% | ~20-40% |

高 threshold 下少量回答但准确率极高；增加 compute 主要增加了回答数量（覆盖率）。

**Jeopardy Odds 效用 (R1-32B vs S1-32B)**:

| 模型 | Threshold=0 Budget=8K | Threshold=0.95 Budget=8K |
|------|:--------------------:|:------------------------:|
| R1-32B | ~0.20 | **~0.45** |
| S1-32B | ~0.18 | ~0.25 |

在 Jeopardy Odds 下，R1-32B 的置信度校准远优于 S1-32B——标准评估看不出的差异。

### 关键发现
- **Test-time compute 增加的是置信度而不仅是准确率**：随计算预算增加，正确答案的置信度上升，错误答案的置信度不变甚至下降（图3 的可视化非常清晰）
- **不同模型的校准能力差异巨大**：R1-32B 能很好地区分正确/错误答案的置信度，S1-32B 则不能——在 Exam Odds 下两者相当，但 Jeopardy Odds 下 R1 大幅领先
- **高 threshold + 高 budget 时准确率可能下降**：因为额外发现的答案正确率较低，拉低了整体已回答问题的准确率（但效用仍在上升）
- **budget forcing（强制延长推理）可能有害**：突然截断或强制延长推理链可能让模型行为偏离训练分布

## 亮点与洞察
- **揭示了 test-time scaling 的"隐藏维度"**：当前评估只看准确率（accuracy surface），本文展示了完整的 accuracy-coverage-confidence 三维曲面——对理解 test-time scaling 的真正价值至关重要
- **Jeopardy Odds 作为标准评估协议**：非常好的建议——$r_t=-1$ 是最自然的"答错 vs 不答"权衡点，简单且直觉一致
- **R1 vs S1 的深层差异**：标准 benchmark 上相当的两个模型，在置信度校准上差异巨大——这个发现对模型选择有重要指导意义

## 局限性 / 可改进方向
- **置信度估计方法简单**：仅用 token log-prob 作为置信度，更好的方法（如基于推理链一致性的置信度）可能效果更好
- **仅在 AIME（数学竞赛）上评估**：是否推广到其他任务（代码、自然语言理解）未验证
- **Budget forcing 的副作用**：强制延长推理链可能不是最优的 compute 控制方式
- **未考虑 compute 成本本身**：效用函数中没有加入计算成本项

## 相关工作与启发
- **vs 标准 test-time scaling 评估**: 标准评估是 Exam Odds（threshold=0），本文补充了 Jeopardy/High-Stakes 维度
- **vs Selective classification (Geifman & El-Yaniv)**: 经典选择性分类框架的 LLM 推理版本，首次应用到 test-time scaling 场景
- **vs SQuAD 2.0 的弃权机制**: SQuAD 2.0 允许拒答但不惩罚答错，本文引入错误惩罚使场景更现实
- 未来可以将置信度作为 test-time compute allocation 的信号——置信度高就停止推理，低就继续

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在 test-time scaling 中引入选择性问答评估，视角新颖
- 实验充分度: ⭐⭐⭐ 只在 AIME 上评估两个模型，但分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 可视化精美（3D surface），论述清晰
- 价值: ⭐⭐⭐⭐ 对 test-time scaling 评估方法论有重要贡献，Jeopardy Odds 值得推广
