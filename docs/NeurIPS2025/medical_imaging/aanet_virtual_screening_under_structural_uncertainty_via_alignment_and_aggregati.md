# AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation

**会议**: NeurIPS 2025  
**arXiv**: [2506.05768](https://arxiv.org/abs/2506.05768)  
**代码**: [GitHub](https://github.com/Wiley-Z/AANet)  
**领域**: 医学图像 / AI for Science  
**关键词**: 虚拟筛选, 药物发现, 结构不确定性, 对比学习, 蛋白质结合位点  

## 一句话总结
针对现实药物发现中蛋白质 holo 结构不可用的问题，提出 AANet——通过三模态对比学习（配体-holo pocket-检测cavity）对齐表征并用交叉注意力聚合多个候选结合位点，在 apo/predicted 蛋白质结构上的盲筛性能远超 SOTA（DUD-E 上 EF1% 从 11.75 提升至 37.19）。

## 研究背景与动机

1. **领域现状**：结构基础虚拟筛选 (SBVS) 是药物发现的核心步骤，通过评估化合物与蛋白质口袋的匹配度从大型化合物库中识别潜在活性分子。现有方法（Glide docking、DrugCLIP 等）都依赖 holo 蛋白质结构（已知配体结合的结构）。
2. **现有痛点**：(a) 大多数有价值靶点没有 holo 结构，只有 apo（无配体）或 AlphaFold2 预测结构；(b) 在 apo/predicted 结构上现有方法性能断崖式下降；(c) 关键瓶颈不是结构形变，而是 **结合位点定位不准**。
3. **核心矛盾**：DrugCLIP 等深度学习方法对结构噪声鲁棒，但对口袋定位极其敏感。几何检测工具（Fpocket）找到的 cavity 与真实结合位点偏差大。
4. **本文要解决什么？** 如何在不知道配体结合位置的情况下进行准确的虚拟筛选？
5. **切入角度**：几何检测的 cavity 是 holo pocket 的"噪声代理"——学习对齐两者的表征。
6. **核心 idea 一句话**：三模态对齐（配体-holo pocket-cavity）+ 交叉注意力聚合候选位点，使模型在结合位点未知时仍能准确筛选。

## 方法详解

### 整体框架
两阶段：(1) **对齐阶段** — 用三模态对比学习在 PDBBind 有标注数据上预训练，对齐配体、holo pocket 和检测 cavity 的表征；(2) **聚合阶段** — 冻结编码器，训练交叉注意力 adapter，在 AlphaFold2 预测结构 + ChEMBL 活性数据上学习聚合多个候选 cavity。

### 关键设计

1. **三模态对比对齐 (Tri-modal Contrastive Alignment)**:
   - 做什么：学习配体、holo pocket、检测 cavity 三者的对齐表征
   - 核心思路：三种模态两两配对做 pairwise sigmoid loss：$\mathcal{L}_{CL} = \mathcal{L}_{p,l}(P_l, l) + \mathcal{L}_{p,l}(P_c, l) + \mathcal{L}_{p,p}(P_c, P_l)$
   - 设计动机：仅对齐 ligand-pocket 会导致对 holo 口袋位置的过拟合；加入 cavity 模态使模型学习蛋白质结构本身的空间特征
   - 关键细节：cavity 通过 IoU 筛选正样本（IoU > τ），pocket 提取半径从 6Å 扩大到 10Å 以应对 cavity 分裂问题

2. **硬负样本挖掘 (Hard Negative Mining)**:
   - 做什么：从非结合 cavity 中选负样本增强判别能力
   - 核心思路：IoU 低的 cavity 作为负样本，与配体和 holo pocket 做对比推远
   - 设计动机：蛋白质表面有很多几何凹陷但无功能的"假口袋"，模型需要学会区分

3. **交叉注意力聚合 (Cross-Attention Adapter)**:
   - 做什么：对多个候选 cavity 进行动态加权聚合
   - 核心思路：以配体 embedding 为 query，cavity embeddings 为 key/value，单头注意力聚合：$\tilde{e}_c = \sum_{s=1}^S a^{(s)} \cdot \mathcal{F}_s(P_c^{(s)})$
   - 初始化近似恒等映射（高温 softmax ≈ 均匀平均），逐步学习最优权重
   - 设计动机：在无口袋标注的活性数据上训练，让模型自主推断哪个 cavity 最可能是结合位点
   - 注意力监督：有标注数据用 one-hot 标签，无标注数据用预训练 AANet 的 cavity 评分作为软标签，KL 散度监督

### 训练策略
- 对齐阶段：PDBBind 2020 general set（去除与测试集重叠）
- 聚合阶段：ChEMBL35 活性数据 + AlphaFold2 预测结构
- 严格数据去重：所有 DUD-E/LIT-PCBA 相关 UniProt 条目都排除
- 总 loss：$\mathcal{L}_{agg} = \mathcal{L}'_{CL} + \lambda \cdot \mathcal{L}'_{KL}$

## 实验关键数据

### 主实验 (DUD-E, 38 targets)

| 方法 | 结构 | 设置 | BEDROC | EF1% |
|------|------|------|--------|------|
| Glide-SP | holo | oracle | 0.296 | 17.25 |
| DrugCLIP | holo | oracle | 0.516 | 33.70 |
| DrugCLIP | apo-pred | blind | 0.197 | 12.05 |
| **AANet** | holo | oracle | **0.637** | **40.85** |
| **AANet** | apo-pred | blind | **0.623** | **37.19** |

### 对比分析

| 配置 | EF1% (apo-pred blind) | 说明 |
|------|----------------------|------|
| DrugCLIP | 12.05 | 基线 |
| AANet (对齐 only) | ~30+ | 对比对齐已大幅提升 |
| AANet (对齐+聚合) | **37.19** | 聚合进一步提升 |

### 关键发现
- **AANet 在 apo blind 设置上接近 holo 性能**：EF1% 37.19 vs holo 40.85（仅差 3.66），而 DrugCLIP 从 33.70 跌至 12.05
- **性能下降的主因是口袋定位而非结构形变**：DrugCLIP 在 apo-oracle（已知口袋位置）设置下性能几乎不降，但 blind 下暴跌
- **对不同检测工具鲁棒**：在 Fpocket、P2Rank 等不同 cavity 检测器上都有效，说明学习的是结构内在特征
- **在 LIT-PCBA 更难基准上同样有效**：LIT-PCBA 是更真实的筛选场景，AANet 同样大幅领先

## 亮点与洞察
- **问题诊断精准**：系统性分析揭示了 DL-based SBVS 的关键瓶颈是口袋定位而非结构噪声，修正了领域内的模糊认知。这个诊断本身就很有价值。
- **三模态对齐的 proxy idea**：用检测到的 cavity 作为 holo pocket 的噪声代理，然后对齐学习——这个思路可以迁移到任何"标注质量不确定"的场景
- **聚合 adapter 的优雅设计**：初始化为均匀聚合 → 学习为选择性注意力，平滑过渡且允许在无标注数据上训练
- **实际药物发现价值**：使 SBVS 真正可用于"first-in-class"靶点（无 holo 结构的全新靶点），突破了实际应用的关键瓶颈

## 局限性 / 可改进方向
- **依赖 cavity 检测工具**：如果几何检测完全失败（例如全新折叠的蛋白），方法的前提不成立
- **单一蛋白构象**：未考虑蛋白的动态构象集合（ensemble）
- **聚合阶段数据泄漏风险**：虽然做了去重，但 UniProt 级别的去重是否足够严格值得讨论
- **改进方向**：(1) 结合 MD 模拟的构象集合；(2) 端到端训练而非两阶段；(3) 扩展到蛋白质-蛋白质相互作用筛选

## 相关工作与启发
- **vs DrugCLIP**：DrugCLIP 是 AANet 的基础架构（CLIP-style 对比学习），AANet 在此基础上加入了 cavity 对齐和聚合，解决了 DrugCLIP 对口袋定位的敏感性
- **vs Glide/AutoDock**：传统 docking 对结构形变敏感但对口袋定位不那么敏感（因为会在 pocket 内搜索），AANet 和 DrugCLIP 则相反
- **vs TankBind**：TankBind 尝试预测结合位置，但性能较弱；AANet 通过聚合多个候选隐式解决定位问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 三模态对齐+聚合的框架设计针对真实痛点，cavity proxy 思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ DUD-E + LIT-PCBA 双基准，holo/apo-exp/apo-pred × oracle/annot/blind 多维对比
- 写作质量: ⭐⭐⭐⭐ 问题分析深入透彻，实验设计严谨，结构清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了 SBVS 从 holo 到 apo 的关键转化瓶颈，对药物发现有直接实际价值
