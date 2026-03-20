# SpHOR: A Representation Learning Perspective on Open-set Recognition

**会议**: CVPR 2026 (Findings)  
**arXiv**: [2503.08049](https://arxiv.org/abs/2503.08049)  
**代码**: [https://github.com/nadarasarbahavan/SpHOR](https://github.com/nadarasarbahavan/SpHOR)  
**领域**: 表征学习 / 开放集识别  
**关键词**: open-set recognition, von Mises-Fisher, orthogonal embeddings, spherical representation, familiarity trap  

## 一句话总结
提出SpHOR两阶段解耦训练框架：Stage 1通过正交标签嵌入+球面约束（vMF分布）+Mixup/Label Smoothing做专为OSR设计的表征学习，Stage 2冻结特征训练分类器——在Semantic Shift Benchmark上OSCR/AUROC最高提升5.1%/5.2%，同时引入Angular Separability和Norm Separability两个新度量。

## 背景与动机
开放集识别（OSR）要求模型不仅准确分类已知类，还能将训练时未见的未知类标记为"未知"。现有OSR方法的核心问题是特征表征没有为未知类显式设计：(1) 大多数方法端到端联合训练backbone和分类器，特征空间只是隐式适应未知数据；(2) 欧氏空间中特征幅度无界，导致开放空间风险不可控；(3) SupCon等通用表征学习目标不是专为OSR设计的。Vaze等人发现简单的闭集分类基线就能匹配许多OSR方法，关键在于特征表征质量。但是否能通过显式设计表征来进一步提升OSR性能？这是SpHOR的出发点。

## 核心问题
如何专门为OSR定制表征学习目标，使特征空间显式地为未知类预留开放空间，同时防止"Familiarity Trap"（与已知类语义相似的未知类被高置信度误分类）？

## 方法详解

### 整体框架
两阶段解耦训练：Stage 1学习球面表征（Encoder + Projection网络），Stage 2丢弃Projection网络，冻结Encoder的非归一化特征训练线性分类器。推理时用评分规则（MaxLogit/KNN/PostMax/NNGuide）做已知/未知二分类。

### 关键设计
1. **球面约束 + vMF对齐损失**：L2归一化特征投影到超球面，每个类建模为vMF分布。vMFAL损失（Eq.7）将样本投影 $z_i$ 与对应类标签嵌入 $\mu_c$ 对齐，同时兼容Mixup和Label Smoothing的软标签。理论证明（Theorem 2）该损失分解为Alignment项（拉向正确类嵌入）和Uniformity项（在嵌入周围均匀扩散），对模糊样本（$\max(S_{ik}) \to 1/|C|$）Uniformity主导，把模糊样本推离类中心，解决Familiarity Trap。

2. **正交正则化 $\mathcal{R}_{Ortho}$**：防止标签嵌入坍缩——即使vMFAL优化了特征和嵌入的对齐，所有 $\mu_k$ 可能趋向共线。正则化强制标签嵌入对正交：$\mathcal{R}_{Ortho} = \log \frac{1}{|C|^2 - |C|} \sum_{j \neq i} \exp(\frac{1}{\tau}(\mu_j \cdot \mu_i)^2)$。相比ETF方法，正交约束避免了负相关和特征冗余。

3. **Mixup + Label Smoothing融入表征学习**：关键创新是将这两个技术从分类器层面移到表征学习阶段。Mixup生成语义模糊的样本（模拟未知类），Label Smoothing平滑类标签。消融发现二者有互补效应：Mixup提升Angular Separability（AS），LS提升Norm Separability（NS），联合使用两个指标同时提升。

### 训练策略
- Stage 1: Encoder + 1024维线性Projection网络，vMFAL + $\mathcal{R}_{Ortho}$ 联合训练
- Stage 2: 冻结Encoder提取非归一化特征 $f_i$，训练线性分类器（标准交叉熵），计算量极小
- 训练复杂度 $O(B \cdot C)$，远优于SupCon的 $O(B^2)$

## 实验关键数据

| SSB (ImageNet预训练) | 方法 | Avg Acc↑ | Avg AUROC (Easy/Hard)↑ | Avg OSCR (Easy/Hard)↑ |
|---|---|---|---|---|
| | MLS+MaxLogit | 84.9 | 84.12/74.78 | 75.24/70.83 |
| | MLS+Mixup+MaxLogit | 87.0 | 86.93/78.56 | 78.53/74.84 |
| | SupCon+MaxLogit | 82.9 | 87.48/78.21 | 78.67/71.44 |
| | **SpHOR+MaxLogit** | **92.6** | **93.00/83.20** | **88.40/80.00** |

- SSB上OSCR提升最高5.1%（vs SupCon），AUROC最高5.2%
- Legacy Benchmark A: 平均AUROC 94.6（+0.8 over ConOSR 93.9）
- Legacy Benchmark B: 平均AUROC 94.0（+1.0 over RCSSR 93.0）
- 无预训练时SpHOR仍强劲：MLS+Mixup的AUROC降20-30%，SpHOR仅微降
- 小batch鲁棒：B=16时SpHOR OSCR 81.8 vs SupCon 62.9

### 消融实验要点
- Mixup + LS联合使用：Avg Acc 89.56→92.60，AUROC (Easy) 86.94→93.00，OSCR (Easy) 81.72→88.40
- $\mathcal{R}_{Ortho}$：增加标签嵌入的Dispersion（类间角距），在3/4数据集上提升AUROC
- AS和NS度量揭示：Mixup改善AS（角度分离），LS改善NS（范数分离），二者互补
- MaxLogit是最稳定的评分规则，SpHOR对评分规则选择最不敏感（std 0.99/0.51 vs SupCon 5.70/3.40）

## 亮点
- 解耦训练+专为OSR定制表征：区别于SupCon等通用方法，理论上分析了vMFAL如何促进Alignment和Uniformity
- Mixup在表征学习阶段生成"模拟未知类"样本的洞察很精妙——通过mixup样本的模糊语义自然地建模开放空间
- 引入AS和NS两个新度量解释Mixup和LS的互补机制，给后续工作提供了分析工具
- 训练效率高：$O(B \cdot C)$ vs SupCon的 $O(B^2)$，小batch也稳定

## 局限性 / 可改进方向
- 正交约束要求嵌入维度 $p \geq |C|$，大规模细粒度分类（如1000+类）可能面临维度限制
- 主要用ResNet50验证，Transformer backbone（ViT等）的验证不足
- $\mathcal{R}_{Ortho}$ 在某些数据集上改善有限（Aircraft上AUROC略降），需要进一步理解dataset-dependent行为
- 仅在图像分类场景验证，其他模态（文本、多模态）的推广待探索

## 与相关工作的对比
- **vs MLS (Vaze et al.)**：MLS是闭集训练基线；SpHOR显式设计球面表征，Avg OSCR从~75→88
- **vs SupCon (ConOSR)**：SupCon用通用对比学习，不专为OSR；SpHOR用vMF+正交嵌入，对评分规则更鲁棒，小batch更稳定
- **vs ARPL**：ARPL在欧氏空间用reciprocal points，开放空间无界；SpHOR球面约束天然限制开放空间
- **vs HAFrame/Hier-COS**：这些关注层次化分类；SpHOR关注已知/未知二分类，正交嵌入的目标不同

## 启发与关联
- vMF分布建模+正交嵌入的框架可能推广到其他需要"预留开放空间"的场景（如异常检测、新类发现）
- AS/NS度量可用于分析任何特征空间的开放集分离能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 三个创新（球面+正交+Mixup/LS在表征阶段）的组合有效，理论分析充分
- 实验充分度: ⭐⭐⭐⭐⭐ SSB + 两个Legacy基准、多评分规则、详细消融、新度量分析
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，消融分析深入，但符号较多
- 价值: ⭐⭐⭐⭐ 为OSR提供了表征设计的系统方法论，AS/NS度量有独立价值
