# Scaling Embedding Layers in Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.01637](https://arxiv.org/abs/2502.01637)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: embedding scaling, n-gram embeddings, inference efficiency, offloading, Scone

## 一句话总结
提出Scone方法，通过为高频n-gram学习上下文化的嵌入（用独立Transformer模型训练），在推理时将这些嵌入卸载到主存/SSD，实现"训练时用更多计算但推理时不增加加速器资源"的新缩放范式，1B参数模型超越1.9B基线。

## 研究背景与动机
1. **领域现状**：传统缩放方式是增大模型参数——但这同时增加推理时的FLOPS和加速器内存。
2. **现有痛点**：增大词汇量来扩展embedding有两个问题：(1) 同时增大output layer导致解码成本暴涨；(2) 尾部token训练不充分。
3. **核心矛盾**：推理成本往往远超训练成本（模型被查询数十亿次），传统缩放方式将推理成本和训练计算绑定。
4. **本文要解决什么**：找到一种"训练时可以用更多计算，但推理时加速器资源不变"的新缩放方式。
5. **切入角度**：embedding lookup本质上是内存取操作（无计算），可以卸载到主存/SSD而几乎不影响延迟。高频n-gram的上下文化embedding可以预计算并缓存。
6. **核心idea一句话**：用独立Transformer为高频n-gram学习上下文化embedding，推理时预计算并offload，解耦训练缩放和推理成本。

## 方法详解

### 整体框架
构建高频n-gram集合(f-grams) → 训练独立f-gram Transformer学习上下文化embedding → 推理前预计算所有f-gram embedding并存入主存/SSD → 推理时对输入token匹配最长f-gram，用缓存embedding替换原始token embedding → 送入主模型。

### 关键设计

1. **F-gram选择**:
   - 做什么：从训练语料中选择最频繁的n-gram（n=2到K）
   - 核心思路：类似BPE的贪心合并策略，K-1次线性扫描语料，选频率最高的
   - 设计动机：高频n-gram覆盖大部分token出现，低频的训练不充分不值得

2. **F-gram Transformer模型**:
   - 做什么：独立的小Transformer，输入n-gram的token embedding序列，输出一个上下文化的embedding向量
   - 核心思路：$e_i = \mathcal{A}_{f\text{-}gram}(\mathcal{T}(\sigma_j), ..., \mathcal{T}(\sigma_i))$，训练时与主模型端到端联合训练
   - 设计动机：比查表更灵活——可以组合性地捕捉n-gram语义，且f-gram模型可以独立缩放

3. **推理时卸载**:
   - 做什么：训练完成后预计算所有f-gram embedding，存入主存或NVMe SSD
   - 核心思路：推理时embedding lookup从主存/SSD取，不占用加速器资源。主存延迟可忽略，NVMe有微小开销但不成瓶颈
   - 设计动机：embedding lookup是O(1)内存读操作，天然适合卸载

### 两种新缩放方式
1. **增加f-gram数量**：更多n-gram → 更多上下文化embedding → 更好的输入表示（只需更多主存）
2. **增大f-gram模型**：更大的Transformer学习embedding → 更高质量的embedding（只需更多训练计算）

## 实验关键数据

### 主实验

| 模型 | 加速器参数 | 推理FLOPS | 困惑度 |
|------|-----------|-----------|-------|
| Baseline 1.9B | 1.9B | ~2x | 基线 |
| Scone 1B + 10M f-grams | 1B | ~1x | 匹配1.9B |
| Scone 1B + 1B f-grams | 1B | ~1x | **超越1.9B** |

### 关键发现
- 10M f-grams即可让1.3B模型匹配1.9B基线
- 1B f-grams让1B模型超越1.9B基线，推理FLOPS和内存仅约一半
- 主存存储f-gram embedding几乎无延迟增加
- NVMe存储有微小延迟但不构成瓶颈
- f-gram模型缩放到更大时持续带来收益

## 亮点与洞察
- **新缩放范式**：打破了"更好的模型必然需要更多推理计算"的假设。通过offload embedding实现"训练时缩放，推理时免费"
- **实用性极强**：主存比GPU显存便宜10-100x，存储1B个embedding只需几十GB主存
- **与BPE的巧妙连接**：f-gram选择策略受BPE启发，但不改变tokenizer——避免了改变词汇量带来的output layer问题

## 局限性 / 可改进方向
- f-gram的最长匹配策略可能不是最优——有时短n-gram的embedding可能更好
- 预计算所有f-gram embedding的存储成本随数量线性增长
- 仅在decoder-only架构上验证
- f-gram模型和主模型的联合训练可能增加训练复杂度

## 相关工作与启发
- **vs vocabulary扩展**：直接扩展vocab会增加logits计算成本，Scone解耦了输入输出
- **vs MoE**：MoE在推理时也不用所有参数，但仍需在加速器上。Scone完全将额外参数移出加速器
- **核心启发**：embedding层是LLM中最"廉价"的操作，其缩放潜力被严重低估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新缩放范式，优雅且实用
- 实验充分度: ⭐⭐⭐⭐ 多规模模型对比，offloading延迟测试
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法描述简洁
- 价值: ⭐⭐⭐⭐⭐ 对推理效率优化有重要实际意义
