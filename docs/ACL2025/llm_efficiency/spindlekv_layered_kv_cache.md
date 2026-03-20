# SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

**会议**: ACL 2025  
**arXiv**: [2507.06517](https://arxiv.org/abs/2507.06517)  
**代码**: [https://github.com/tyxqc/SpindleKV](https://github.com/tyxqc/SpindleKV)  
**领域**: 模型压缩 / LLM效率  
**关键词**: KV cache compression, token eviction, codebook, GQA, layer-aware  

## 一句话总结
SpindleKV 提出分层处理 KV cache 压缩的策略——深层使用注意力驱动的 token eviction（利用稀疏注意力），浅层使用基于相似性学习的 codebook 替换（利用 token 间高相似度），并解决了 GQA 兼容性问题，实现 50% KV cache 缩减而不损失性能。

## 研究背景与动机

1. **领域现状**：KV cache 是长上下文 LLM 推理的主要内存瓶颈。现有压缩方法包括：token eviction（H2O/SnapKV/PyramidKV——基于注意力分数移除不重要 token）、token merging（CaM/D2O——合并相似 token）、量化（KIVI/Atom——低精度表示）。
2. **现有痛点**：(1) 所有现有方法在深层效果好但浅层效果差——深层注意力稀疏易压缩，浅层注意力分散难以判断哪些 token 不重要；(2) 基于注意力分数的 eviction 方法与 GQA 不兼容——GQA 中一个 KV head 服务多个 Q head，不同 Q head 的重要性判断可能矛盾；(3) 浅层 KV cache 冗余性未被利用。
3. **核心矛盾**：深层和浅层的冗余模式不同——深层是 inter-token 冗余（注意力集中在少数 token），浅层是 inner-token 组成冗余（token 向量间高度相似，可分解为基向量）。统一方法无法同时优化两种冗余。
4. **本文要解决什么？** 平衡深层和浅层的 KV cache 压缩，提高整体压缩率。
5. **切入角度**：观察到浅层 KV cache token 间余弦相似度很高（因为浅层 Transformer 编码次数少，上下文影响有限），可以用 codebook 替换。
6. **核心idea一句话**：深层用 attention eviction 去除冗余 token，浅层用学习的 codebook 替换高相似度 token，形成"纺锤形"压缩模式。

## 方法详解

### 整体框架
KV cache 按层分为两部分：深层（注意力稀疏）→ 基于注意力分数的 token eviction（保留高注意力 token）；浅层（token 相似度高）→ 基于相似性学习的 codebook 替换（用少量基向量近似原始 token）。整体压缩模式呈"纺锤形（spindle）"——中间层保留最多，深浅层都压缩。

### 关键设计

1. **深层 Token Eviction**:
   - 做什么：在深层使用累积注意力分数移除不重要 token
   - 核心思路：计算窗口内累积注意力分数 $ac_{i,a}$，保留分数最高的 token
   - GQA 兼容改进：不是对 Q head 取平均（如 PyramidInfer），而是设计了 GQA-aware 的评分策略，避免多 Q head 之间的冲突

2. **浅层 Codebook 替换**:
   - 做什么：用 JIT（Just-in-Time）学习的 codebook 基向量替换浅层高相似度 token
   - 核心观察：浅层 KV cache token 间余弦相似度显著高于深层（因为编码次数少，上下文分化弱）
   - 方法：对浅层 KV cache token 做聚类/合并，学习一组 codebook 基向量，用这些基向量近似替换原始 token
   - 设计动机：浅层注意力不稀疏所以 eviction 不适用，但 token 间高相似度意味着可以用更少的代表性向量近似——这是一种不同于 eviction 的冗余利用方式

3. **GQA 兼容性**:
   - 做什么：解决基于注意力的 eviction 在 GQA 模型上的困境
   - 问题：GQA 中一个 KV head 服务多个 Q head，不同 Q head 认为重要的 token 可能不同
   - SpindleKV 的方案保证在 GQA 设置下仍能有效压缩

## 实验关键数据

### 主实验：LongBench + Needle-in-a-Haystack

| 方法 | KV 压缩率 | LongBench Avg | Needle 准确率 |
|------|----------|-------------|-------------|
| 无压缩 | 100% | 基线 | 基线 |
| PyramidKV | 50% | 下降 | 下降 |
| PyramidInfer | 50% | 下降 | 下降 |
| **SpindleKV** | **50%** | **≈基线** | **优于 Pyramid** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| SpindleKV 完整 | 最佳 | 深层 eviction + 浅层 codebook |
| 仅深层 eviction | 有限压缩 | 浅层未压缩，总压缩率受限 |
| 全层 eviction | 浅层掉点 | Eviction 在浅层不适用 |
| 仅浅层 codebook | 有限压缩 | 深层可压缩更多但未利用 |

### 关键发现
- **50% KV 压缩率基本无损**：在多个 LLM 上（LLaMA-2/3、Qwen2）验证
- **SpindleKV 在 GQA 模型上优势明显**：解决了其他 eviction 方法在 GQA 上的兼容性问题
- **浅层 token 间余弦相似度确实显著高于深层**：验证了浅层 codebook 替换的理论基础
- **Needle-in-a-Haystack 任务上保持长序列检索能力**：对信息检索敏感的任务也不掉点

## 亮点与洞察
- **"纺锤形"的直觉**：深层稀疏（可 evict）、浅层相似（可替换）、中间保留最多——这种层级感知的压缩策略比统一策略更合理
- **浅层冗余的新视角**：之前研究都认为浅层难以压缩，SpindleKV 发现浅层的冗余模式（token 相似性）与深层不同但同样可以利用
- **GQA 兼容性**：随着 GQA 成为主流（LLaMA-3、Qwen2 等），这个兼容性优势具有实际部署价值

## 局限性 / 可改进方向
- Codebook 学习引入额外计算开销（JIT learning）
- Codebook 大小是超参数，需要针对不同模型/任务调整
- 浅层/深层的界限划分是预定义的，可能需要自适应策略
- 未与量化方法（KIVI/Atom）结合探索——量化+eviction+codebook 三者联合可能更好

## 相关工作与启发
- **vs PyramidKV (Cai et al., 2024)**: PyramidKV 发现深层可以更多压缩呈金字塔形，但忽略了浅层。SpindleKV 通过 codebook 补充了浅层压缩
- **vs H2O (Zhang et al., 2023)**: H2O 基于累积注意力分数做 eviction，SpindleKV 在深层类似但浅层用不同策略
- **vs SnapKV (Li et al., 2024)**: SnapKV 观察注意力窗口特性做观察驱动 eviction，SpindleKV 进一步区分了层间行为差异

## 评分
- 新颖性: ⭐⭐⭐⭐ 分层处理+浅层 codebook 的思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准，含 Needle-in-a-Haystack 长序列评估
- 写作质量: ⭐⭐⭐⭐ 观察→方法的逻辑通顺
- 价值: ⭐⭐⭐⭐ 对 KV cache 压缩领域有实际贡献
