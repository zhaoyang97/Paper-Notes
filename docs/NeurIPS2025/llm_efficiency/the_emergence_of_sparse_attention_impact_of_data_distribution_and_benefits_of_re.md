<!-- 由 src/gen_stubs.py 自动生成 -->
# The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition

**会议**: NeurIPS 2025  
**arXiv**: [2505.17863](https://arxiv.org/abs/2505.17863)  
**代码**: 无  
**领域**: LLM效率 / 训练动态  
**关键词**: sparse attention, emergence, training dynamics, data repetition, power law

## 一句话总结
从理论和实验两方面解释稀疏attention的涌现时机：推导出plateau时间遵循幂律缩放（与序列长度和维度相关），并证明数据重复（in-context或cross-sample）可将涌现加速2-4倍。

## 研究背景与动机
1. **领域现状**：LLM训练中存在"突然涌现"现象——模型在训练过程中某个时刻突然获得新能力（如induction head出现）。
2. **现有痛点**：涌现的时机难以预测，数据分布如何影响涌现时间缺乏定量理解。
3. **核心贡献**：提供涌现时间的精确幂律公式，并揭示重复数据可加速涌现。

## 方法详解

### 关键设计
1. **理论分析**：在single-location线性回归任务上推导attention学习动态。发现两阶段行为——慢速plateau（FFN先学习）→ 突然转变（attention涌现）。
2. **Plateau时间公式**：无重复时 $T_{plateau} \propto \sqrt{dT} \ln(\epsilon\sqrt{dT})$；in-context重复B次后缩短为 $\sqrt{dT/B}$。
3. **Cross-sample重复**：多个数据样本共享相同pattern也类似加速涌现。

## 实验关键数据
| 设置 | Plateau缩放 | 加速比 |
|------|-----------|--------|
| 无重复 | $\propto T^{0.99}$ | 基线 |
| In-context重复B次 | $\propto T^{0.99} \cdot B^{-0.99}$ | ~B倍 |
| Cross-sample重复 | 类似减少 | 2-4倍 |

实验在associative recall、ICL和factual recall任务上验证理论预测。

## 亮点与洞察
- **涌现可预测**：plateau时间遵循简单的幂律——为预测训练里程碑提供工具
- **重复的双刃剑**：重复数据加速涌现但增加过拟合风险——解释了为什么数据多样性和重复之间需要平衡

## 局限性 / 可改进方向
- 理论分析限于线性回归toy model，向真实Transformer的推广是定性的
- 仅考虑单层attention的涌现，多层交互未涉及

## 评分
- 新颖性: ⭐⭐⭐⭐ 涌现时间的定量预测是新贡献
- 实验充分度: ⭐⭐⭐⭐ toy model理论+标准Transformer验证+多任务测试
- 写作质量: ⭐⭐⭐⭐ 理论-实验结合清晰
- 价值: ⭐⭐⭐⭐ 对理解LLM训练动态有重要启示

## 相关工作与启发
- **vs Olsson et al. (2022)**：他们发现induction head的涌现与ICL能力突变相关，本文解释了为什么——sparse attention本质上导致突变式学习
- **vs Chan et al. (2022)**：他们发现burstiness数据促进ICL涌现，本文给出了定量解释（T/B的power law）
- **与训练数据curation的关系**：cross-sample repetition加速涌现但增加过拟合风险，这对LLM预训练数据配比有重要指导意义
