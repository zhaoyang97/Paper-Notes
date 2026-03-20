# Blending Complementary Memory Systems in Hybrid Quadratic-Linear Transformers
**会议**: NeurIPS 2025
**arXiv**: [2506.00744](https://arxiv.org/abs/2506.00744)  
**代码**: [https://github.com/kazuki-irie/hybrid-memory](https://github.com/kazuki-irie/hybrid-memory)
**领域**: Transformer 架构, 混合模型
**关键词**: 混合记忆, 软注意, 快权重, 线性Transformer

## 一句话总结
混合二次(softmax attention)和线性(DeltaNet)Transformer，各司其职：前者精确检索，后者长上下文和复杂计算，在1.3B模型上实现性能和效率均衡。

## 研究背景与动机
二次Transformer(QT)精确但二次复杂度，线性Transformer(LT)线性但检索精度低。本文探索如何同时获得两者优势。

## 方法详解

### 整体框架
提出三种混合策略差异在于何时、如何向两个记忆系统供应信息。通过关联存储(KV-memory+FW-memory)的补充性设计混合。

### 关键设计
- **延迟流式HQLT**: 新k/v入KV-memory，旧的(t-S步前)入FW-memory
- **延迟分块HQLT**: 基于chunk式训练算法，兼容高效并行
- **同步HQLT**: 新k/v同时入两个系统，充分利用DeltaNet表达性

## 实验关键数据

### 语言建模评估
| 模型 | Wiki PPL | LAMBADA PPL | PiQA | HellaSwag | ARC-e | Avg |
|------|---------|-----------|------|----------|-------|-----|
| Transformer++ (340M) | 26.5 | 34.9 | 67.6 | 41.0 | 60.2 | 47.6% |
| DeltaNet (340M) | 27.6 | 35.0 | 67.1 | 40.8 | 58.5 | 46.8% |
| HQLT同步 (340M) | 26.3 | 29.4 | 66.2 | 42.7 | 61.5 | 47.8% |
| Transformer++ (1.3B) | 19.8 | 17.9 | 71.0 | 50.3 | 65.2 | 53.0% |
| HQLT同步 (1.3B) | 19.8 | 15.9 | 72.0 | 51.5 | 68.1 | 53.9% |

### 合成任务(奇偶性/模运算)
| 任务 | Transformer | DeltaNet | 延迟流 | 延迟分块 | 同步 |
|------|-----------|---------|-------|--------|------|
| 奇偶性(模2) | 低精度 | 高精度 | 中等 | 中等 | 高精度 |
| 模运算(模7) | 失败 | 成功 | 失败 | 失败 | 成功 |

## 亮点与洞察
1. **补充性分析**: Table 1清晰对比两类记忆的权衡（精度vs复杂度，有界vs无界）
2. **同步优于延迟**: 充分利用DeltaNet表达优势需要同步处理，延迟策略削弱
3. **三层混合论证**: FW处理旧信息(抽象)，KV处理新(精确)，两者互补
4. **生物学启示**: 补充学习系统(海马体vs皮层)的计算类比

## 局限性
- 仅15B token训练，规模有限
- 下游任务改进边际(0.1-0.9%)
- 没有分析DeltaNet的不同更新规则变体影响

## 相关工作
Munkhdalai et al.首次混合探索，本工作深化分析并采用最新DeltaNet变体。补充Jamba/Zamba的大规模混合研究。

## 评分
⭐⭐⭐⭐ (4/5)
