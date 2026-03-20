# HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling

**会议**: AAAI 2026  
**arXiv**: [2504.09261](https://arxiv.org/abs/2504.09261)  
**代码**: [https://github.com/Zr2223/HACK](https://github.com/Zr2223/HACK)  
**领域**: 图像生成 / 模型压缩  
**关键词**: KV Cache压缩, 视觉自回归模型, VAR, attention head分类, next-scale prediction  

## 一句话总结
发现VAR模型中attention head天然分为Contextual Heads（语义一致性，垂直注意力模式）和Structural Heads（空间连贯性，多对角线模式），提出HACK框架通过非对称预算分配和模式特定压缩策略，在70%压缩率下实现无损生成质量，Infinity-8B上1.75×显存减少和1.57×加速。

## 背景与动机
VAR模型采用next-scale prediction范式，相比传统next-token AR只需少量步骤就能生成高质量图像。但VAR的KV cache跨scale累积增长，注意力复杂度为$O(n^4)$，生成1024×1024图像需处理10k+ tokens。现有LLM的KV cache压缩方法（H2O、SnapKV、CAKE等）直接用于VAR效果差，因为它们使用"one-size-fits-all"策略，未考虑VAR中不同head的功能差异。

## 核心问题
如何在VAR模型的next-scale生成范式中高效压缩KV cache，在不降低生成质量的前提下显著减少显存和计算开销？关键挑战：VAR的attention head具有两种截然不同的功能角色和注意力模式，简单统一压缩会破坏其中一类。

## 方法详解

### 整体框架
三步走：(1) 离线head分类（按注意力方差区分Contextual/Structural heads）→ (2) 非对称预算分配（给压缩敏感的Structural heads更多budget）→ (3) 模式特定压缩策略（不同head类型用不同eviction/merge策略）。Training-free，仅需50个样本做离线分类。

### 关键设计
1. **Head分类（Contextual vs Structural）**: 通过计算注意力矩阵的列方向方差来区分——Contextual heads关注少数语义关键token→低方差；Structural heads按位置动态注意→高方差。方差分布呈长尾特性，存在自然分界点。分类结果跨样本和scale高度稳定（甚至1个样本就足够分类），说明是模型固有属性。功能验证：遮蔽Contextual heads导致语义漂移但结构完整；遮蔽Structural heads保持语义但空间严重扭曲。

2. **非对称预算分配**: $B = \alpha B_C + (1-\alpha) B_S$，给Contextual heads更小预算（$B_C \ll B_S$），因为它们只关注少数关键token，对压缩不敏感（90%压缩仍保持质量）。Structural heads对压缩敏感（50%以上开始退化），需保留更多cache。由于每层的head比例不同，自然形成layer-adaptive效果。

3. **模式特定压缩策略**: Contextual heads用cumulative attention top-K选择+最后一步merge丢弃token（保留语义信息）；Structural heads用scale-aware策略——固定保留前2个scale（初始全局）和最近scale（当前细节），中间scale用attention选择。灵感来自LLM中"initial+recent tokens更重要"的sink token现象。

4. **Efficient Subset Attention**: 不用全部query估计token重要性，而是均匀采样$N_{obs}=32$个query的attention分数作为近似，开销可忽略。

### 损失函数 / 训练策略
完全Training-free。离线分类仅需50个样本+几分钟。部署时静态重排head顺序，按类型分组以支持高效推理。

## 实验关键数据

| 模型/任务 | 方法 | 压缩率 | GenEval↑ | HPSv2.1↑ | ImageReward↑ | FID↓ |
|--------|------|------|---------|---------|--------------|------|
| Infinity-2B T2I | Vanilla | 0% | 0.946 | 30.49 | 0.68 | 10.34 |
| Infinity-2B T2I | H2O | 70% | 0.910 | 29.60 | 0.68 | 10.68 |
| Infinity-2B T2I | SnapKV | 70% | 0.904 | 29.60 | 0.68 | 10.60 |
| Infinity-2B T2I | **HACK** | **70%** | **0.933** | **30.18** | **0.68** | **10.56** |
| Infinity-8B T2I | Vanilla | 0% | 1.049 | 30.99 | 0.81 | 8.75 |
| Infinity-8B T2I | **HACK** | **70%** | **1.043** | **30.69** | **0.82** | **8.62** |
| VAR-d30 Class | Vanilla | 0% | - | - | - | 1.92 (FID) |
| VAR-d30 Class | H2O | 50% | - | - | - | 3.04 |
| VAR-d30 Class | **HACK** | **50%** | - | - | - | **2.06** |
| VAR-d30 Class | **HACK** | **70%** | - | - | - | **2.78** |

效率：Infinity-8B 1.75×显存减少(60.42→34.44GB), 1.57×加速(8.14→5.17s)。1024分辨率下HACK线性增长vs Vanilla指数增长，极端情况5.8×加速。

### 消融实验要点
- 非对称分配 + 模式特定压缩都贡献显著（Table 4, 缺一不可）
- 策略互换（Contextual策略给Structural head）→性能大幅下降（ImageReward 0.859 vs 0.933），证明模式特定设计的必要性
- Head分类方法对比：方差分类 >> Order/Uniform/Random（FID 2.06 vs 2.57/2.63/2.70）
- 分类对样本量不敏感（1~100个样本结果一致）
- Query子集采样$N_{obs}=32$即接近full attention精度

## 亮点
- **"Contextual vs Structural" head的发现是genuinely novel的贡献** — 揭示了VAR模型attention的内在功能分工，不同于LLM中的head分析
- **功能验证实验极其直观** — 选择性遮蔽清晰展示两类head的互补功能
- 70%压缩率几乎无损甚至某些指标超越原始模型 — 说明VAR中确实存在大量冗余
- 复杂度从$O(n^4)$降到$O(Bn^2)$是理论上的重大改进
- 与CAMERA论文的"微专家"概念异曲同工——都是在transformer内部发现功能异质性并据此设计差异化策略

## 局限性 / 可改进方向
- 仅优化attention模块，FFN的开销未处理
- head比例$\alpha$需要手动调整（虽然不太敏感）
- 未与量化方法结合（KV cache量化+HACK可能叠加收益）
- 仅验证了VAR模型，未扩展到传统next-token AR生成模型

## 与相关工作的对比
- **vs H2O/SnapKV**: 这些通用KV压缩方法不区分head类型，对VAR效果差（FID 3.04/3.09 vs HACK 2.06 @50%）因为破坏了Structural heads
- **vs LOOK-M/MEDA**: Merging方法在VAR上退化最严重（FID 6.89/18.88），因为merge操作破坏了空间结构信息
- **vs StreamingLLM**: 位置based方法不能捕捉VAR中token的语义重要性差异

## 启发与关联
- **与CAMERA的"微专家"概念高度共鸣** — CAMERA分析MoE内部的微专家异质性，HACK分析attention head的功能异质性，都利用这种异质性设计差异化压缩
- **与Distillation Dynamics的"U型模式"互补** — HACK发现的Contextual/Structural本质上也是信息压缩(语义汇总)和信息保持(空间结构)的分工
- 可以扩展到MLLM的KV cache压缩——VLM中也可能存在类似的"Contextual vs Structural"分化
- 与cross-layer token budget allocation idea相关——HACK提供了head-level的budget allocation思路

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Contextual/Structural head发现+VAR-specific KV压缩是全新贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 6种VAR模型+T2I/Class-Cond+多种压缩率+详尽消融+效率分析
- 写作质量: ⭐⭐⭐⭐⭐ motivation可视化出色，分析→设计→验证逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 首个VAR KV cache压缩工作，实际加速效果显著，实用价值高
