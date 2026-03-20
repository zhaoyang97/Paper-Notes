# A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification

**会议**: NeurIPS 2025  
**arXiv**: [2511.04814](https://arxiv.org/abs/2511.04814)  
**作者**: Sebastian Ojeda, Rafael Velasquez, Nicolás Aparicio, Juanita Puentes, Paula Cárdenas, Nicolás Andrade, Gabriel González, Sergio Rincón, Carolina Muñoz-Camargo, Pablo Arbeláez (Universidad de los Andes, Colombia)  
**领域**: others (生物信息学 / 抗菌肽分类)  
**关键词**: Antimicrobial Peptide, Multilabel Classification, Benchmark, Transformer, Cross-Attention, Drug Discovery  

## 一句话总结

提出 **ESCAPE**——首个标准化的多标签抗菌肽分类基准，整合 27 个公开数据库共 80,000+ 肽段，并设计基于双分支 Transformer + 双向交叉注意力的 Baseline 模型，在 mAP 上相对第二名提升 2.56%。

## 研究背景与动机

1. **抗菌素耐药性 (AMR) 危机**：据估计 2025–2050 年间 AMR 感染可导致超 3900 万人死亡，寻找替代分子（如抗菌肽 AMP）迫在眉睫。
2. **AMP 的潜力**：抗菌肽通过不易被病原体规避的机制（如膜破坏、细胞壁合成抑制等）发挥作用，耐药风险较传统抗生素低。
3. **AI 加速 AMP 发现的瓶颈**：现有 AI 方法多将任务简化为二分类（是否抗菌），忽略了 AMP 可同时对多种微生物（细菌、真菌、病毒、寄生虫）有活性的多标签本质。
4. **数据碎片化问题**：各数据库在格式、标注标准、功能类别粒度上差异巨大（如 dbAMP 有 58 类 vs. DRAMP 仅 8 类），模型无法跨数据集公平比较。
5. **缺乏标准基准**：大多数研究使用自定义数据集和划分，无法复现或做方法间的公平对比。
6. **多标签空白**：已有少量多标签方法（如 AMPs-Net、TransImbAMP），但尚无统一的多标签 AMP 基准供社区使用。

## 方法详解

### 整体框架：ESCAPE（数据集 + 基准 + Baseline模型）

ESCAPE 的贡献分三个层级：

- **ESCAPE Dataset**：从 27 个公开 AMP 数据库中编译、清洗并标准化 80,000+ 肽段，统一为 5 类多标签体系（antibacterial / antifungal / antiviral / antiparasitic / antimicrobial），外加 Non-AMP 负样本。
- **ESCAPE Benchmark**：在统一数据集上对 7 种代表性方法进行公平的多标签评估，采用 2-fold 交叉验证 + 测试集、3 个随机种子取均值与标准差。
- **ESCAPE Baseline**：双分支 Transformer 架构，融合序列与结构信息进行多标签分类。

### 关键设计 1：数据编译与清洗流程

- 从 27 个数据库收集实验验证的 AMP 序列，涵盖四种抗微生物活性。
- 负样本（Non-AMP）采用 UniProt 关键词排除法（排除 membrane、toxic、antibiotic 等关键词）+ 从已知非抗菌数据集中纳入。
- 清洗规则：移除含合成残基（O、U、Bal 等）及未定义氨基酸（X）的序列；保留长度 5–250 的肽段；合并跨库重复序列并整合其多标签标注。
- 最终规模：60,950 Non-AMP + 21,409 AMP，按标签分层划分为 2-fold + test set。

### 关键设计 2：双分支 Transformer 编码器

**序列分支**：
- 对氨基酸序列进行 token 化（词表大小 27），padding/截断至固定长度 200。
- 嵌入维度 256，加入 [CLS] token 和位置编码。
- 4 层 Transformer encoder，每层 8 头注意力。

**结构分支**：
- 利用 3D 结构（UniProt/PDB 实验结构或 RosettaFold/AlphaFold3 预测）计算 Cα 原子间距矩阵 $\mathcal{M} \in \mathbb{R}^{N \times N}$。
- 将矩阵 resize 至 224×224，通过 2D 卷积（kernel=16, stride=16）切分为不重叠 patch，每个 patch 投影至 192 维。
- 同样加 [CLS] token + 位置编码，经 4 层 8 头 Transformer 编码。

### 关键设计 3：双向交叉注意力融合

- 序列侧 [CLS] 作为 Query 去 attend 结构侧全部 token（Key/Value），建模"序列信息如何被结构上下文增强"。
- 反向同理，结构侧 Query attend 序列侧 Key/Value。
- 残差连接 + FFN 精炼，最终将两个更新后的 [CLS] 向量拼接后经线性分类头输出 5 维多标签预测。

### 损失函数与训练

- 损失：多标签分类（论文未特别说明，推测为 Binary Cross-Entropy）。
- 优化器：AdamW，学习率 $1 \times 10^{-4}$，batch size 64，训练 100 epoch。
- 评估：2-fold 交叉验证训练两个模型，推理时概率取平均；3 个随机种子（42, 1665, 8914）取均值 ± 标准差。
- 指标：mAP 和 F1-score。

## 实验关键数据

### Table 1 & 2：ESCAPE Benchmark 主结果

| 方法 | mAP (%) | F1 (%) | Antiparasitic AP (%) |
|------|---------|--------|---------------------|
| AMPs-Net | 54.6±0.86 | 57.7±0.70 | 5.3±0.67 |
| TransImbAMP | 64.9±1.11 | 62.0±0.70 | 16.7±0.86 |
| AMP-BERT | 66.9±1.17 | 64.7±0.64 | 21.4±2.61 |
| amPEPpy | 68.5±0.48 | 66.5±0.37 | 23.8±1.61 |
| PEP-Net | 68.4±0.53 | 65.5±0.61 | 16.2±0.84 |
| AVP-IFT | 68.8±0.50 | 66.5±0.59 | 20.0±4.25 |
| AMPlify | 70.3±0.87 | 68.5±0.77 | 27.7±1.33 |
| **ESCAPE Baseline** | **72.1±0.60** | **69.8±0.43** | **37.6±2.87** |

### Table 3：消融实验

| 结构模块 | 序列模块 | 交叉注意力 | mAP (%) | F1 (%) |
|---------|---------|-----------|---------|--------|
| ✓ | - | - | 47.7 | 46.9 |
| - | ✓ | - | 69.4 | 67.6 |
| ✓ | ✓ | ✓ | **72.7** | **69.5** |

### 核心发现

1. **ESCAPE Baseline 全面领先**：mAP 72.1%，相对第二名 AMPlify (70.3%) 提升 2.56%；F1 69.8%，相对提升 1.90%。
2. **稀少类别提升显著**：antiparasitic 类的 AP 从 AMPlify 的 27.7% 跃升至 37.6%，相对提升 **35.7%**。
3. **序列 >> 结构**：仅序列的 mAP 69.4%，远高于仅结构的 47.7%（差 21.7%），说明氨基酸身份信息是分类的决定性因素。
4. **结构作为互补**：加入结构并通过交叉注意力融合后 mAP 从 69.4% → 72.7%，额外提升 3.3 个百分点。
5. **模型大小 ≠ 性能**：排名第二的 amPEPpy 基于随机森林，是计算量最小的方法；BERT 系模型（TransImbAMP、AMP-BERT）反而排名靠后，提示大语言模型在非自然语言领域的迁移存在局限。
6. **预测结构 vs. 实验结构**：仅用预测结构时 mAP 下降 1.5%、F1 下降 1.9%，说明预测结构引入的噪声会削弱模型表现。

## 亮点与洞察

1. **填补领域空白**：首个整合 27 个数据库、涵盖 80,000+ 肽段的标准化多标签 AMP 基准，真正解决了数据碎片化和标注不一致问题。
2. **标签体系设计合理**：将各数据库五花八门的功能类别统一成 4+1 的生物学层级体系，既有区分度又具可解释性。
3. **双向交叉注意力优于简单拼接**：让序列与结构互相 attend，比单纯特征拼接或单模态编码更有效地提取互补信息。
4. **公平评估揭示重要洞察**："更大的模型不一定更好"——随机森林（amPEPpy）可以媲美甚至超越 BERT 系方法，这在 AMP 特定领域是有价值的发现。
5. **对稀少类别的显著改善**：antiparasitic 从 5.3% (AMPs-Net) 到 37.6%，展现了基准+好方法的组合效应。

## 局限性

1. **数据分布代表性**：自然界中的肽段多样性远超数据集覆盖范围，80K 样本不一定反映真实分布。
2. **序列长度偏差**：AMP 天然较短（~30 aa），Non-AMP 较长（~90 aa），长度本身可能成为分类捷径。
3. **结构预测依赖**：部分肽段使用 AlphaFold3/RosettaFold 预测结构，引入额外误差（mAP 下降 1.5%）。
4. **类别严重不平衡**：antiparasitic 仅 417 条（其中 130 条唯一），所有方法在该类上表现仍有较大提升空间。
5. **缺乏生物实验验证**：模型预测在转化为实际药物发现前，需要 wet-lab 验证。
6. **Baseline 模型相对简单**：未探索更先进的预训练蛋白质语言模型（如 ESM-2 作为 backbone）或对比学习等策略。

## 相关工作

- **AMP 数据库**：dbAMP (33K 肽/58 类)、DRAMP (30K/8 类)、LAMP2 (23K/38 类)、SATPdb (19K/10 类) 等，各自覆盖范围和标注粒度差异大。
- **序列方法**：AMPlify (BiLSTM + attention)、AMP-BERT / TransImbAMP (预训练 BERT)、dsAMPGAN (CNN+Attention+BiLSTM)、AMPpred-DLFF (ESM-2 + GAT + CNN)。
- **特征增强方法**：amPEPpy (CTD 特征 + Random Forest)、AMPs-Net (图+理化特征 + GNN)、PEP-Net (one-hot + 理化 + PLM embedding + Transformer)、AVP-IFT (对比学习 + 理化特征)。
- **与本文的区别**：ESCAPE 是首个统一的多标签基准；Baseline 创新在于双向交叉注意力融合序列与 3D 距离矩阵这两种模态。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个大规模标准化多标签 AMP 基准，填补重要空白；Baseline 模型的双向交叉注意力为该领域引入了新的融合范式。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7 种方法公平对比、3 种随机种子、消融实验、预测结构敏感性分析，评估非常全面。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰、动机充分、图表丰富，但部分技术细节（如损失函数选择）可更明确。
- **价值**: ⭐⭐⭐⭐⭐ — 对 AI 驱动的抗菌肽研究具有基础设施性的贡献价值，数据集和基准将推动社区发展。
