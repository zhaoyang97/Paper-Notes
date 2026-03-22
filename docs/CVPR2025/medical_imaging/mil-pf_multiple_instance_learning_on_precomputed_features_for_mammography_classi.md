# MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification

**会议**: CVPR2025  
**arXiv**: [2603.09374](https://arxiv.org/abs/2603.09374)  
**代码**: 已开源  
**领域**: medical_imaging  
**关键词**: Multiple Instance Learning, mammography, foundation model, precomputed features, attention pooling

## 一句话总结

提出 MIL-PF 框架，利用冻结的基础视觉模型预计算特征，配合仅 ~40k 参数的轻量 MIL 聚合头，在乳腺 X 光分类任务上达到 SOTA 性能，大幅降低训练成本。

## 研究背景与动机

- 乳腺癌是女性最常见的恶性肿瘤和癌症死亡的首要原因，乳腺 X 光（mammography）是主要的筛查和诊断手段
- 乳腺 X 光图像具有极高分辨率（可达 4708×5844 像素），缺乏像素级标注，通常只有乳房级别（breast-level）的弱标签
- 端到端微调大模型在该场景下计算开销巨大且不切实际，现有 CLIP 式训练受限于监督信号不足和高分辨率输入的架构限制
- 关键洞察：冻结的通用基础编码器（DINOv2、MedSigLIP）在分布外乳腺 X 光域上已展现出优秀的零样本泛化能力，因此无需微调编码器

## 方法详解

MIL-PF 流水线分为两个阶段：

### 阶段一：特征预计算
- 使用冻结的基础编码器 $\mathcal{F}$ 对每个 bag（同一乳房的所有视图）生成两类嵌入：
  - **全局嵌入** $\mathcal{G}_i = \{\mathcal{F}(I_i^{(n)})\}_n$：对每张完整图像编码，捕获组织结构、乳腺密度等低频全局信号
  - **局部嵌入** $\mathcal{T}_i = \bigcup_n \bigcup_k \{\mathcal{F}(C_i^{(n)(k)})\}$：将图像划分为编码器输入大小的 tile 网格（DINOv2 为 518×518，MedSigLIP 为 448×448），通过启发式 $\mathcal{H}$ 丢弃纯背景块，仅保留含乳腺组织的 tile，编码后捕获稀疏的局部病灶信号
- 每张图像的 tile 数量 $M_i^{(n)}$ 可变，取决于乳房大小和位置
- 构建嵌入数据集 $\mathcal{E} = \{(\mathcal{G}_i, \mathcal{T}_i, y_i)\}_i$，后续训练完全在该固定表示空间进行
- 分类任务不使用重叠 tile；推理时注意力图计算使用 75% 重叠以提升可视化分辨率

### 阶段二：MIL-PF Head 训练
- 采用晚期融合策略，两个流独立聚合后拼接：$\hat{y}_i = h_\theta(\text{concat}(\mathcal{A}^G_\psi(\mathcal{G}_i), \mathcal{A}^T_\omega(\mathcal{T}_i)))$
- **全局聚合器** $\mathcal{A}^G$：
  - 两层 MLP（嵌入维度→16→8，ReLU 激活）将高维嵌入投影到紧凑表示
  - 随后使用 max pooling 聚合多视图全局特征
  - 这一分支做的是任务相关的高层特征处理（因编码器冻结），大部分参数在此
- **局部聚合器** $\mathcal{A}^T$：
  - 同样先过两层 MLP 做任务相关投影
  - 采用 Perceiver 风格的交叉注意力：使用单个可训练 latent 向量 $z$ 作为 query，tile 嵌入投影为 Key 和 Value
  - 计算 $\text{softmax}(zK^T)V$ 得到加权汇总向量，选择性聚焦于与任务相关的稀疏 ROI
  - 相比 mean pooling（信号被大量背景 tile 稀释）和 max pooling（只捕获单个最显著 tile），attention 机制能同时关注多个独立的病灶区域
  - 实验发现单 latent query 已足够，增加更多 latent 无额外收益
- 最终 $h_\theta$ 将拼接的 summary 向量映射为分类预测
- 损失函数为 Binary Cross-Entropy，总可训练参数仅 ~40k

### 多次运行优化
- 由于 head 训练极快（单次约 5-7 分钟，单块 A100 40GB），每个实验运行 36 次独立训练
- 跨运行方差：AUC 最大变动 2%，Spec@Sens=0.9 最大变动 11%
- 选择验证 AUC 最高的模型，该策略稳定地接近测试集最大性能
- 数据划分为 70%/10%/20%（训练/验证/测试），按 BI-RADS 值比例平衡，确保无患者泄漏

## 实验关键数据

- **数据集**：
  - EMBED：~50 万张乳腺 X 光，最大公开数据集之一，高度多样的真实临床场景
  - VinDr：越南乳腺 X 光数据集，含 mass 和 calcification 标注
  - RSNA：乳腺癌筛查竞赛数据集
- **EMBED BI-RADS 恶性分类**（BI-RADS 1 为阴性，BI-RADS 4/5/6 为阳性）：
  - MIL-PF（DINOv2, attn）：AUC=0.916, bAcc=0.850, Spec@Sens=0.9=0.762
  - MIL-PF（MedSigLIP, max）：AUC=0.918, bAcc=0.845, Spec@Sens=0.9=0.735
  - 最强基线 FPN-AbMIL：AUC=0.802, Spec@Sens=0.9=0.367
- **RSNA 癌症检测**：MIL-PF（MedSigLIP, max）AUC=0.925, Spec@Sens=0.9=0.733
- **VinDr 钙化检测**：MIL-PF（DINOv2, attn）AUC=0.967, bAcc=0.930
- **消融实验**：完整 MIL-PF 相比仅用全局视图的单实例学习，AUC 提升最高 5%，Spec@Sens=0.9 提升最高 14%
- **编码器选择**：DINOv2 ViT Giant 和 MedSigLIP 显著优于 MammoCLIP、BiomedCLIP、RADDINO、DINOv3
- **计算效率**：每次前向传播仅 ~2M FLOPS（每个乳房），可训练参数 0.04-0.05M vs 基线 1.76-22.89M

## 亮点

1. **极致轻量**：仅 ~40k 可训练参数即达 SOTA，是最强基线参数量的 1/450
2. **预计算范式**：编码器冻结 + 特征预计算，整个 EMBED 嵌入数据集可放入单个 A100 的一个 batch，5-7 分钟完成训练
3. **Perceiver 式注意力池化**：优雅解决稀疏 ROI 聚合问题，单 latent query 即可有效定位多个病灶区域
4. **乳房级 MIL 建模**：符合临床实际中放射科医生综合多视图判读的流程
5. **可解释性**：注意力图可定位病灶区域，虽基于大 tile 粒度但方向准确

## 局限性

1. **检测分辨率受限**：tile 尺寸较大（448×448 / 518×518），小病灶检测 mAP 极低（mAPs≈0.1-1.2），注意力图定位精度不足
2. **标签噪声**：BI-RADS 标签存在噪声且放射科医生间变异大，评估可靠性受限
3. **晚期融合策略简单**：全局流和局部流未建模复杂交互，可能错过跨尺度关联信息
4. **未利用时序信息**：未考虑患者历史检查或双侧对称性，这些在临床中是重要的诊断线索
5. **依赖编码器质量**：性能上限取决于冻结编码器的表示能力，对新型编码器的泛化性待验证

## 相关工作

| 方法 | 特点 | 与本文对比 |
|------|------|-----------|
| GMIC [Shen et al.] | 全局+局部端到端训练，14.11M 参数 | MIL-PF 用 1/350 参数在 EMBED 上 AUC 高 10% |
| FPN-AbMIL [Mourão et al.] | FPN+注意力 MIL，图像级推理 | MIL-PF 乳房级建模更符合临床，EMBED 上大幅领先 |
| FPN-SetTrans [Mourão et al.] | Set Transformer 聚合 | 复杂度高但性能不如 MIL-PF |
| ES-Attside [Pathak et al.] | 案例级注意力，22.89M 参数 | EMBED 上 AUC 仅 0.836 vs MIL-PF 0.918 |
| MammoCLIP | 乳腺 X 光预训练 CLIP | 在新数据集上泛化不如通用 DINOv2 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 冻结编码器 + 超轻量 MIL 头的范式在乳腺 X 光中首次系统验证
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个大规模数据集、多编码器对比、详细消融、36 次运行统计
- 写作质量: ⭐⭐⭐⭐ — 问题形式化清晰，方法推导连贯
- 价值: ⭐⭐⭐⭐ — 为资源受限研究组提供了高效可行的乳腺 X 光 CAD 方案
