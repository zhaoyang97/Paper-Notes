# Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning

**会议**: AAAI 2026  
**arXiv**: [2410.12657](https://arxiv.org/abs/2410.12657)  
**代码**: https://github.com/realMoana/EPA-GRL  
**领域**: 自监督学习  
**关键词**: 图对比学习, 语义保持增强, 可解释AI, 半监督学习, GNN Explainer

## 一句话总结
提出EPA-GRL（Explanation-Preserving Augmentation），利用少量标签训练的GNN explainer识别图的语义子图（explanation subgraph），增强时只扰动非语义部分（marginal subgraph），实现语义保持的图增强，在6个benchmark上显著优于语义无关的随机增强方法。

## 研究背景与动机
1. **领域现状**：图对比学习（Graph Contrastive Learning）通过生成同一图的两个增强视图来学习不变表示。GraphCL、JOAO等方法使用random node/edge dropping等增强策略。
2. **现有痛点**：现有增强策略都是**语义无关的**——随机扰动可能破坏图中与分类相关的关键子结构（如分子中的benzene ring），导致增强后的图丢失核心语义，下游分类性能受损。
3. **核心矛盾**：好的增强需要同时满足(1)保持语义和(2)引入方差这两个要求，但现有方法只关注后者。
4. **本文要解决什么？** 如何在图增强中保持语义的同时引入足够的方差？
5. **切入角度**：利用GNN可解释性技术（explainer）识别图中的语义子图，增强时保护语义部分只扰动非语义部分。
6. **核心idea一句话**：用少量标签训练explainer → 识别语义子图 → 只扰动非语义部分 = 语义保持的图增强

## 方法详解

### 整体框架
EPA-GRL是两阶段方法：
1. **Pre-training阶段**：用少量带标签图训练GNN分类器 $f_{pt}$ + explainer $\Psi$
2. **Representation Learning阶段**：用frozen explainer生成语义子图 → 只扰动marginal part → 对比学习训练encoder

### 关键设计

1. **GNN Explainer预训练**:
   - 做什么：学习一个parametric explainer，输入图G输出explanation subgraph $G^{(exp)}$
   - 核心思路：基于Graph Information Bottleneck (GIB)原则训练：$\arg\min_{\Psi} \sum CE(Y; f_{pt}(\Psi(G))) + \lambda|\Psi(G)|$。第一项确保子图保持分类信息，第二项确保子图compactness（避免包含无关部分）
   - 设计动机：explainer可以用少量标签学会识别class-discriminative的子结构，然后泛化到未标注图

2. **Explanation-Preserving Augmentation**:
   - 做什么：生成保持语义的增强图
   - 核心思路：$G^{(exp)} = \Psi(G)$（保持不变），$\Delta G = G \setminus G^{(exp)}$（随机扰动）。扰动方式包括：node dropping on $\Delta G$、edge dropping on $\Delta G$、attribute masking on $\Delta G$、subgraph sampling from $\Delta G$、mixup（用另一图的marginal替换当前的marginal）
   - 设计动机：保护语义子图不被破坏，同时marginal部分提供足够方差

3. **理论分析**:
   - 做什么：证明语义保持增强的优越性
   - 核心思路：在modified BA-2motifs图上，证明语义保持的encoder $f_{enc}^{sp}$ 的分类误差接近0，而语义无关的encoder $f_{enc}^{sa}$ 的误差接近1/2（随机猜测）——差异可以任意大

### 损失函数 / 训练策略
- Explainer用GIB loss + CE loss
- Representation learning用GraphCL (contrastive) 或 SimSiam loss
- 半监督：少量标签训练explainer，所有图（含无标签）做对比学习

## 实验关键数据

### 主实验
6个benchmark（BA-2motifs, MUTAG, PROTEINS, DD, NCI1, COLLAB）上的图分类准确率：

| 方法 | BA-2motifs | MUTAG | PROTEINS | NCI1 |
|------|-----------|-------|----------|------|
| GraphCL (random aug) | 基线 | 基线 | 基线 | 基线 |
| JOAO | 略优 | 略优 | 略优 | 略优 |
| AD-GCL | 中 | 中 | 中 | 中 |
| **EPA-GraphCL** | **显著优** | **显著优** | **显著优** | **显著优** |
| **EPA-SimSiam** | **显著优** | **显著优** | **显著优** | **显著优** |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| Random-Aug (语义无关) | GNN分类精度大幅下降 | 语义被破坏 |
| EPA-Aug (语义保持) | GNN分类精度接近原始 | 语义被保留 |
| 无explainer (全图random) | 性能差 | 缺乏语义保护 |
| 不同label比例 (1-10%) | label越多explainer越好 | 但即使1%也有效 |

### 关键发现
- 在BA-2motifs上Random-Aug导致GNN精度从接近100%降到<50%，而EPA-Aug维持>90%
- EPA的embedding分布与原始图保持一致（t-SNE可视化验证），Random-Aug产生大的分布偏移
- EPA可以plugged into不同对比学习框架（GraphCL/SimSiam），都有提升
- 即使只用1-5%的标签训练explainer，EPA也有效

## 亮点与洞察
- **XAI→Data Augmentation的桥接**非常优雅：将可解释AI技术用于数据增强，开创了"用解释来保护语义"的新范式
- **理论保证有说服力**：在BA-2motifs的分析模型上，语义保持 vs 语义无关的分类误差差异可以任意大
- **实用性强**：EPA是plug-and-play的，可以和任意图对比学习框架组合

## 局限性 / 可改进方向
- **Explainer质量依赖标签数量和GNN质量**：如果预训练GNN本身不好，explainer可能识别出错误的语义子图
- **增加了预训练成本**：需要额外训练GNN+explainer，比纯无监督方法多一步
- **假设语义=子图**：有些图的语义可能不是简单的子图结构（如全局属性模式）
- **只在graph-level任务验证**：node-level对比学习是否也能受益未知

## 相关工作与启发
- **vs GraphCL/JOAO**: 随机增强，不保持语义。EPA在所有benchmark上一致优于它们
- **vs AD-GCL**: 用对抗增强学习what NOT to keep，EPA用explainer学习what TO keep，方向互补
- **vs ENGAGE**: ENGAGE用无监督方式识别重要节点，但无标签指导可能捕获的不是class-discriminative结构。EPA用少量标签确保是真正有意义的语义

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ XAI→GRL的桥接是全新方向，理论+实验都支撑良好
- 实验充分度: ⭐⭐⭐⭐ 6个数据集、多种框架、消融、可视化
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法自然，理论分析加分
- 价值: ⭐⭐⭐⭐ 为图对比学习中"如何做好增强"提供了新的思路和工具
