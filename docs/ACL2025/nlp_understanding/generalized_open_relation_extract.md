# Towards a More Generalized Approach in Open Relation Extraction

**会议**: ACL 2025  
**arXiv**: [2505.22801](https://arxiv.org/abs/2505.22801)  
**代码**: [https://github.com/qingwang-isu/MixORE](https://github.com/qingwang-isu/MixORE)  
**领域**: NLP 理解  
**关键词**: Open Relation Extraction, Generalized OpenRE, Semi-supervised Learning, Contrastive Learning, Novel Relation Detection  

## 一句话总结

提出 MixORE 框架，在更通用的 Open Relation Extraction 设定下（无标注数据同时包含已知和新颖关系，且不做长尾或预分割假设），通过 Semantic Autoencoder 检测新关系 + 开放世界半监督联合学习，在 FewRel/TACRED/Re-TACRED 上全面超越 SOTA。

## 研究背景与动机

### Open Relation Extraction 的发展

传统关系抽取（RE）依赖预定义关系集合和大量标注数据，无法处理新出现的关系类型。Open Relation Extraction (OpenRE) 旨在从无标注数据中主动发现新关系。

### 现有 OpenRE 方法的不合理假设

1. **设定一（无监督 RE）**：假设无标注数据中全为新关系 → 忽略了已知关系的利用
2. **设定二（半监督 OpenRE）**：假设无标注数据被预分割为已知/新颖集合 → 实际场景中不可能提前知道
3. **KNoRD 的 "长尾" 假设**：假设新关系罕见、属于长尾分布、倾向于显式表达 → 新关系不一定是长尾分布（例如新领域概念刚出现时，实例数可能很多）

### 本文的广义设定

放松了"长尾"假设，仅假设无标注数据同时包含已知和新颖实例，对关系的分布不做任何限制。这更贴近真实应用场景。

## 方法详解

### 整体框架

MixORE 是一个两阶段框架：

**Phase 1：Novel Relation Detection（新关系检测）**
- 目标：从无标注数据中识别出潜在的新关系实例
- 输出：新关系的弱标签集合 $\mathcal{D}_w$

**Phase 2：Open-World Semi-Supervised Joint Learning（OW-SS 联合学习）**
- 目标：联合优化已知关系分类和新关系发现
- 输入：有标签数据 $\mathcal{D}_l$ + 弱标签数据 $\mathcal{D}_w$

### 关键设计

#### 1. Relation Encoder

使用 BERTbase 作为编码器，对输入句子插入带类型的实体标记（`<e1:type>`, `</e1:type>` 等），拼接两个实体标记位置的隐藏向量作为关系表示：

$$h_r = [h_{<e1:type>} | h_{<e2:type>}]$$

#### 2. Semantic Autoencoder (SAE) 做新关系检测

核心思想是：已知关系的实例在潜在空间中会聚集在其对应的 one-hot 向量附近，而新关系实例因为不匹配任何已知关系而成为离群点。

- 将每个已知关系用 one-hot 向量表示，训练 SAE 将特征空间映射到 $|\mathcal{C}_{known}|$ 维潜在空间
- SAE 使用 tied weights（转置权重矩阵做解码器），目标函数：
  $$\min_W \|X_l - W^\top S_l\|_F^2 + \lambda \|WX_l - S_l\|_F^2$$
- 使用 Bartels-Stewart 算法获得封闭解，无需迭代更新
- 推理时将无标注数据映射到潜在空间，计算与各已知关系 one-hot 向量的余弦相似度
- **最低 5% 映射分数的实例被标记为离群点**（新关系候选）

#### 3. GMM 聚类获取弱标签

对检测到的离群点使用 Gaussian Mixture Model 聚类为 $|\mathcal{C}_{novel}|$ 个新关系组。保留 GMM 后验概率 > 0.95 的实例作为高质量弱标签 $\mathcal{D}_w$。

#### 4. OW-SS 联合学习

采用持续学习策略，先在 $\mathcal{D}_l$ 上 warm up，再在 $\mathcal{D}_l \cup \mathcal{D}_w$ 上持续训练。

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L} = \mathcal{L}_c + \mathcal{L}_{lm} + \mathcal{L}_e$

**1. 分类损失 $\mathcal{L}_c$（交叉熵）**：
$$\mathcal{L}_c = -\frac{1}{D_c}\sum_{i=1}^{D_c}\sum_{r=1}^{|\mathcal{C}_u|} y_r^i \log(\hat{y_r^i})$$

**2. 标签数据三元组边距损失 $\mathcal{L}_{lm}$**：
- 仅从 $\mathcal{D}_l$ 构建正样本对（避免弱标签的噪声影响）
- 固定正样本对数量 $D_m = 5D_c$，确保每个关系均匀采样
- 使用 triplet margin loss，以余弦距离度量

**3. 聚类 exemplar 损失 $\mathcal{L}_e$**：
- 在多个粒度层使用 K-Means 计算关系 exemplar
- 鼓励实例表示与其聚类中心对齐
- exemplar 随训练每个 epoch 动态更新

**推理阶段**：已知关系用分类结果，新关系用 Faiss K-Means 聚类结果。

## 实验关键数据

### 主实验

**数据集设置**：FewRel（41 关系, 6 novel）、TACRED（41 关系, 6 novel）、Re-TACRED（39 关系, 6 novel）

**FewRel 结果**：

| 方法 | Known F1 | B³ F1 | V-measure F1 | ARI |
|------|----------|-------|-------------|-----|
| ORCA | 0.6210 | 0.5481 | 0.5492 | 0.4318 |
| KNoRD | 0.7738 | 0.7318 | 0.7297 | 0.6945 |
| **MixORE** | **0.8328** | **0.8968** | **0.8802** | **0.8817** |

**TACRED 结果**：

| 方法 | Known F1 | B³ F1 | V-measure F1 | ARI |
|------|----------|-------|-------------|-----|
| KNoRD | 0.8519 | 0.7680 | 0.7883 | 0.7193 |
| **MixORE** | **0.8833** | **0.8682** | **0.8599** | **0.8473** |

**Re-TACRED 结果**：

| 方法 | Known F1 | B³ F1 | V-measure F1 | ARI |
|------|----------|-------|-------------|-----|
| KNoRD | 0.8669 | 0.6389 | 0.7306 | 0.5081 |
| **MixORE** | **0.9156** | **0.8750** | **0.8613** | **0.8925** |

### 关键发现

1. **MixORE 在所有数据集上全面超越所有基线**，在 known 和 novel 关系上都优势明显
2. **与 KNoRD 对比**：在 Re-TACRED 上 ARI 提升 0.3844（0.5081→0.8925），改善巨大
3. **无监督方法 HiURE/AugURE 在 novel 聚类上表现不错但 known 分类很差**（F1 约 0.43-0.49），说明"全部当新关系"的策略不适合广义设定
4. **消融实验显示**：去掉 NRD 阶段（预测全部为已知），known F1 从 0.8606→0.7374（下降 12.3%）；去掉持续学习，novel B³ F1 从 0.8968 降至 0.8134

## 亮点与洞察

1. **问题定义精准**：对现有 OpenRE 假设进行了系统梳理，提出了更合理的广义设定
2. **SAE 做离群点检测非常巧妙**：利用已知关系的 one-hot 向量约束潜在空间结构，新关系自然成为离群点
3. **第一阶段保持轻量**：冻结 BERT 参数，用 Bartels-Stewart 闭式解求解 SAE，高效
4. **5% 显著性水平阈值有理论依据**：与统计假设检验的惯例对齐
5. **三元组损失只用 labeled data 构建正样本对**：避免弱标签噪声引入假阳性对

## 局限性 / 可改进方向

1. **需要预知 novel 关系数量 $|\mathcal{C}_{novel}|$**：这在实际场景中通常未知，可考虑自动确定聚类数的方法（如 BIC/AIC）
2. **5% 阈值可能不够鲁棒**：不同数据集中 novel 关系的占比差异很大
3. **BERTbase 可能不够强**：可以尝试更大的预训练模型如 DeBERTa-v3
4. **数据增强策略较简单**：使用现成的句内/句间增强，可考虑 LLM 生成等更先进方法
5. **只在英语数据集验证**：跨语言场景未探索

## 相关工作与启发

- **开放世界学习与持续学习的结合**：MixORE 将 Open-world SSL（Cao et al., 2022 ORCA）和 Continual RE（Cui et al., 2021）有机结合
- **SAE 的新应用场景**：将零样本学习中的 Semantic Autoencoder 迁移到关系抽取的离群点检测
- **对比学习在 RE 中的发展**：从实例级（Liu et al., 2022）到 exemplar 级的多粒度对比学习
- **启发**：广义设定的提出推动了 OpenRE 方向向更实际场景发展，类似思路可推广到 Open NER、Open Event Extraction

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-------------|------|
| 创新性 | 8 | 问题定义精准，SAE 检测新关系思路新颖 |
| 实验充分性 | 8 | 三个数据集、丰富的消融实验和基线对比 |
| 写作质量 | 8 | 逻辑清晰，方法阐述详尽 |
| 实用价值 | 7 | 广义设定更贴近实际，但预知 novel 数量的假设仍有限制 |
| 总分 | 8 | 高质量工作，对 OpenRE 领域有实质推动 |
