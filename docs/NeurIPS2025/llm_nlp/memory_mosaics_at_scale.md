# Memory Mosaics at Scale
**会议**: NeurIPS 2025
**arXiv**: [2507.03285](https://arxiv.org/abs/2507.03285)  
**代码**: [https://github.com/facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics)
**领域**: LLM 架构设计, 关联存储
**关键词**: 内存马赛克, 高斯核回归, 上下文学习, 组合性

## 一句话总结
Memory Mosaics v2将关联存储扩展至10B参数和1T token，在新任务学习上超越8T token训练的Transformer。

## 研究背景与动机
Transformers在新任务学习上表现有限，关联存储提供透明且可解释的记忆机制。本文研究其大规模扩展可行性。

## 方法详解

### 整体框架
用关联存储替代Transformer注意，实现key-value记忆的显式检索。v2引入三项改进：自适应带宽、门控时变特征提取、三层记忆设计。

### 关键设计
- **自适应高斯核**：$β=β_1·n^α+β_0$，随记忆大小动态调整带宽
- **门控时变关键**: $g_t=e^{W_g x_t}$, $λ_t=e^{-|W_λ x_t|}$，动态平衡历史
- **三层记忆**: 短期(近256步)、长期(64-256步外)、持久(两层FFN)

## 实验关键数据

### 新知识存储学习
| 任务 | Transformer(4k) | MM v2(4k) | Transformer(32k) | MM v2(32k) |
|------|----------|---------|----------|-----------|
| Ruler-QA任务1 | 57.7% | 59.3% | × | 26.5% |
| Ruler-QA任务2 | 任务失败 | 48.8% (8k外推) | × | 未失败 |

### 持久知识vs新知识评估
| 数据集 | Transformer | MM v2 w/o LTM | MM v2 完整 | 新知识依赖 |
|--------|----------|-------------|----------|--------|
| 13任务平均 | 57.1% | 56.6% | 56.8% | 否 |
| Ruler-QA(6任务) | 高失败率 | 失败 | 成功 | 是 |

## 亮点与洞察
1. **可解释性优势**: 显式key-value存储使计算透明，vs Transformer黑箱注意
2. **新任务泛化**: 8T token的Transformer不如1T token的MM v2在未见任务上表现
3. **自适应设计**: 动态带宽随序列长度自动调整，无需手调
4. **三层记忆分工**: 短期精确检索+长期抽象存储+持久知识保留，分工明确

## 局限性
- FLOPs/token略高于Transformer(18.9B vs 16.7B)
- 参数数量增加(9.9B vs 8.8B)
- 长期记忆延迟参数m的选择不够自动化

## 相关工作
Bietti(2023)分析感应头，本工作用关联存储显式实现。补充Transformer电路工作的可解释视角。

## 评分
⭐⭐⭐⭐ (4/5)
