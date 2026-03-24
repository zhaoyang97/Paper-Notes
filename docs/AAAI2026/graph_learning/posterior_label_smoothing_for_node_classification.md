# Posterior Label Smoothing for Node Classification

**会议**: AAAI 2026  
**arXiv**: [2406.00410](https://arxiv.org/abs/2406.00410)  
**代码**: https://github.com/ml-postech/PosteL  
**领域**: 图学习  
**关键词**: Label Smoothing, 节点分类, 后验分布, 同质/异质图, 伪标签迭代

## 一句话总结
提出PosteL（Posterior Label Smoothing），通过贝叶斯后验分布从邻域标签中推导soft label用于节点分类，自然适应同质图和异质图，在8种backbone×10个数据集的80个组合中76个取得精度提升。

## 研究背景与动机
1. **领域现状**：Label smoothing（向one-hot标签添加uniform噪声）在CV和NLP中广泛使用，但在图节点分类中很少研究。Knowledge distillation的soft label能编码"dark knowledge"提升学生模型性能。
2. **现有痛点**：现有的图label smoothing方法（SALS、ALS）假设节点倾向于与邻居有相同标签，直接聚合邻域标签作为soft label。这在同质图上有效，但在异质图上反而有害——因为异质图中邻居标签恰好与目标节点不同。
3. **核心矛盾**：需要一个label smoothing方法能同时适应同质图（邻居=同标签）和异质图（邻居≠同标签），现有方法只处理前者。
4. **切入角度**："You can tell a person by the company they keep"——从邻域标签的全局统计推导后验分布，同质图下后验偏向多数邻居标签，异质图下后验偏向少数邻居标签。
5. **核心idea一句话**：用贝叶斯后验分布（基于全局label共现统计）推导的soft label，自然适应同质和异质图

## 方法详解

### 整体框架
给定图 $\mathcal{G}=(\mathcal{V},\mathcal{E},X)$ 和训练节点标签，PosteL分两步：
1. 用贝叶斯后验从邻域标签+全局统计推导每个训练节点的soft label
2. 迭代伪标签：用模型预测更新未标注节点标签，重新估计全局统计，再推导更好的soft label

### 关键设计

1. **后验标签平滑**:
   - 做什么：为每个节点推导基于邻域的soft label
   - 核心思路：$P(\hat{Y}_i=k|\{Y_j\}_{j\in\mathcal{N}(i)}) \propto P(\{Y_j\}|\hat{Y}_i=k) \cdot P(\hat{Y}_i=k)$。假设邻居标签条件独立，似然分解为各邻居的条件概率乘积。条件概率和先验都从图的全局标签共现统计估计。最终soft label = $\alpha \cdot$ 后验 $+ (1-\alpha) \cdot$ one-hot + $\beta \cdot$ uniform
   - 设计动机：Lemma 1证明在同质图下，多数邻居标签推高后验概率；Lemma 2证明在异质图下，少数邻居标签反而推高后验概率。这完美适配两种图类型

2. **迭代伪标签**:
   - 做什么：用模型预测扩充label信息，改善全局统计估计
   - 核心思路：训练GNN → 预测未标注节点 → 用伪标签更新似然和先验 → 重新推导soft label → 再训练
   - 设计动机：稀疏图中很多节点没有已标注邻居，伪标签填补信息空缺

### 损失函数 / 训练策略
- 用推导的soft label替代one-hot label训练任意GNN backbone
- 交叉熵损失，1000 epochs，200 epochs early stopping
- 68/20/20 train/val/test split

## 实验关键数据

### 主实验
8种backbone × 10数据集，80个组合中76个提升（95%）。代表性结果：

| 模型+PosteL | Cora | CiteSeer | Chameleon | Squirrel | Texas |
|------------|------|----------|-----------|----------|-------|
| GCN | 提升 | 提升 | 提升 | 提升 | 提升 |
| GPR-GNN | 提升 | 提升 | 提升 | 提升 | 提升 |
| BernNet | 提升 | 提升 | 提升 | 提升 | 提升 |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| w/o 迭代伪标签 | 有效但低于完整版 | 全局统计不够准确 |
| SALS (naive聚合) | 同质图有效，异质图退化 | 不适应异质图 |
| Uniform label smoothing | 轻微提升 | 不利用图结构 |
| PosteL (full) | 76/80提升 | 最佳 |

### 关键发现
- PosteL在**异质图上的提升比同质图更显著**——因为naive方法在异质图上完全失效，而PosteL通过后验自然适应
- 伪标签迭代在**稀疏图**（如Cornell，26%节点无标注邻居）上贡献最大
- PosteL对**所有8种backbone**都有效，说明是真正通用的正则化技术（model-agnostic）

## 亮点与洞察
- **后验分布的巧妙应用**：将label smoothing从"加noise"提升为"推导后验"，理论上证明了对同质/异质图的双重适应性
- **极其简单的实现**：只需统计全局label共现频率→贝叶斯公式→soft label，无需额外参数或训练
- **95%的通过率**（76/80）说明方法极其鲁棒，几乎可以作为GNN训练的标配技巧

## 局限性 / 可改进方向
- 条件独立性假设在密集图中可能不成立
- 只考虑了一阶邻居，多跳邻域的信息未利用
- 伪标签质量取决于初始模型性能，如果初始模型很差可能引入噪声

## 相关工作与启发
- **vs SALS**: 直接聚合邻域标签，只适用同质图。PosteL通过后验分布自然处理异质图
- **vs ALS**: 类似SALS但有自适应refinement。PosteL从概率模型出发，理论更完备
- **vs Knowledge Distillation**: KD需要训练teacher模型，PosteL直接从图结构推导soft label，零额外成本

## 评分
- 新颖性: ⭐⭐⭐⭐ 后验label smoothing概念简单优雅，理论证明solid
- 实验充分度: ⭐⭐⭐⭐⭐ 8模型×10数据集，80个组合，覆盖非常全面
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合好，toy example直观
- 价值: ⭐⭐⭐⭐ 简单通用的GNN正则化技巧，实际价值高
