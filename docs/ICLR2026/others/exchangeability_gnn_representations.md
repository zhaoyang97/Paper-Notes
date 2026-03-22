# Exchangeability of GNN Representations with Applications to Graph Retrieval

**会议**: ICLR 2026 Oral  
**OpenReview**: [HQcCd0laFq](https://openreview.net/forum?id=HQcCd0laFq)  
**代码**: 有  
**领域**: 图神经网络  
**关键词**: GNN, exchangeability, graph retrieval, LSH, GraphHash  

## 一句话总结
发现训练好的 GNN 节点嵌入沿特征维度是**可交换随机变量**（即 $p(X) = p(X\pi)$ 对任意维度排列 $\pi$），利用此性质通过维度排序将基于传输距离的图相似度近似为欧氏距离，构建高效的局部敏感哈希（LSH）框架 GraphHash，在子图匹配和图编辑距离检索任务上超越基线，可扩展到 100 万图语料库。

## 研究背景与动机
1. **领域现状**：图检索需要计算图间相似度（如图编辑距离、子图同构），计算代价极高。
2. **现有痛点**：基于传输距离的图相似度无法直接用 LSH 加速，因为传输距离不满足标准 LSH 所需的度量性质。
3. **核心idea一句话**：GNN 嵌入沿特征维度的可交换性使得维度排序后的嵌入可用欧氏距离近似传输距离，从而启用 LSH 高效检索。

## 方法详解

### 关键设计
1. **可交换性发现**：证明 GNN 对称初始化+训练后节点嵌入的特征维度可交换
2. **维度排序近似**：对每个维度独立排序后计算欧氏距离，近似 Wasserstein 距离
3. **GraphHash**：基于排序嵌入的 LSH 框架，支持子图匹配和编辑距离检索

## 实验关键数据
- 在 PTC-FR/FM/MR 和 COX2 数据集上超越 FourierHashNet, DiskANN, IVF, CORGII, SWWL
- 扩展到 100 万图语料库，近似误差约 12%
- 跨种子稳定性极高（std < 0.01 AUC）
- 嵌入维度 $D$ 增大性能持续提升

## 亮点与洞察
- **GNN 表征的新理解**：可交换性是 GNN 嵌入此前未被注意的对称性，理论意义超越具体应用
- **无缝接入 LSH 体系**：将图检索降维为标准向量检索问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 可交换性发现和 GraphHash 构造均为原创
- 实验充分度: ⭐⭐⭐⭐ 多数据集+大规模验证
- 写作质量: ⭐⭐⭐⭐ 理论清晰
- 价值: ⭐⭐⭐⭐ 为图检索提供了实用的理论工具
