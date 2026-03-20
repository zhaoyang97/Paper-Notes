# Segment-Level Attribution for Selective Learning of Long Reasoning Traces

**会议**: ICLR2026  
**arXiv**: [2602.00425](https://arxiv.org/abs/2602.00425)  
**代码**: [GitHub](https://github.com/SiyuanWangw/SegmentSelectiveSFT)  
**领域**: llm_reasoning  
**关键词**: reasoning trace, integrated gradients, selective SFT, segment attribution, CoT compression  

## 一句话总结
用Integrated Gradients计算长推理链中每个segment对最终答案的归因强度和方向一致性，识别重要segment进行选择性SFT，相比全CoT训练提升准确率达4.7%同时缩短输出18%。

## 背景与动机
1. 大推理模型(LRM)生成数千token的CoT，但仅少部分真正对答案预测有贡献，大量冗余重复/截断内容
2. 对冗余CoT做全量SFT会使模型学习冗长无信息模式，浪费学习能力甚至降低性能
3. 现有压缩方法token-level分析忽略语义完整性，segment-level的困惑度/熵指标与重要性不完全一致
4. 困惑度方法存在假阳性（高估过渡文本）和假阴性（低估验证/中间结论）问题
5. 需要直接度量segment对正确答案预测的因果贡献

## 方法
**Segment分割**: 用transition关键词（"\n\nWait", "\n\nAlternatively"等）将长CoT分割为语义片段。

**Integrated Gradients归因**: 对每个token计算IG值（从padding baseline到实际embedding的路径积分），衡量其对正确答案概率的贡献。

**两个Segment-level指标**:
- **Attribution Strength**: $\text{Strength}(S) = \sum|IG(o_n)| / \sqrt{N}$，衡量影响量级(√N归一化避免长度偏差)
- **Direction Consistency**: $\text{Consistency}(S) = |\sum IG(o_n)| / \sum|IG(o_n)|$，衡量正负贡献一致性。极高一致性=浅层推理，中等一致性=反思性推理

**重要Segment选择**: 取累计归因强度达τ=70%的top segment，再过滤掉一致性>β=0.8的浅层segment。约33%的segment被标为重要(占45% token)。

**Selective SFT**: 仅在重要segment上计算loss，不重要segment的loss被mask为0，保留完整CoT上下文。

## 实验
| 模型 | 方法 | Overall Acc | 输出长度 |
|------|------|:-----------:|:--------:|
| R1-Distill-Qwen-1.5B | Full SFT | 44.8 | 16520 |
| R1-Distill-Qwen-1.5B | **Segment Selective** | **46.9**(+4.7%) | 13506(-18%) |
| R1-Distill-Qwen-7B | Full SFT | 62.1 | 9693 |
| R1-Distill-Qwen-7B | **Segment Selective** | **64.5**(+3.9%) | 8499(-12%) |
| Qwen2.5-7B-Instruct | Full SFT | 44.2 | 10317 |
| Qwen2.5-7B-Instruct | **Segment Selective** | **45.6**(+3.2%) | 9852(-5%) |

**关键发现**: (1) 30-40%的segment贡献80%+的总归因，验证大量冗余; (2) 重要segment具有更低困惑度/熵，不重要segment更多重复(高BLEU)和截断(49%); (3) Selective SFT一致优于全量SFT和剪枝方法; (4) 在AIME24等OOD难题上提升最显著(+13.3%); (5) 该方法可泛化到RL场景。

## 亮点
- 用IG归因直接度量segment对答案的因果贡献，比PPL/熵等间接指标更可靠
- 方向一致性(consistency)指标设计巧妙：区分浅层确认vs反思性推理
- Selective SFT同时提升准确率和效率（缩短输出），双赢
- 分析透彻：验证了不重要segment确实对应重复/截断/废话

## 局限
- IG计算需多步插值前向传播，计算开销较大（虽是一次性成本）
- 关键词分割方式较简单，可能不适应所有推理风格
- 仅在数学推理数据集上验证，对代码生成/自然语言推理的效果未知
- τ和β阈值需在验证集上搜索，增加调参成本

## 相关工作
- CoT压缩: Xia et al. 2025b token-level分析; Cui et al. 2025b segment-level PPL; Li et al. 2025b 基于熵
- Selective SFT: Lin et al. 2024 selective learning framework
- 归因方法: Sundararajan et al. 2017 Integrated Gradients; 本文首次应用于推理链segment
- 长推理冗余: Wang et al. 2025d 分析截断思维; Wu et al. 2025 冗长降低推理性能

## 评分
- 新颖性: ⭐⭐⭐⭐ (IG+segment归因+selective SFT组合新颖)
- 实验充分度: ⭐⭐⭐⭐ (多模型+ID/OOD+消融充分)
- 写作质量: ⭐⭐⭐⭐ (分析细致，可视化好)
- 价值: ⭐⭐⭐⭐ (对长推理链训练有直接工程价值)
