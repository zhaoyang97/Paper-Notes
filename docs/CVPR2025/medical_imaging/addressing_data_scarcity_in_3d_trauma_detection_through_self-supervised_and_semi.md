# Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

**会议**: CVPR2025  
**arXiv**: [2603.12514](https://arxiv.org/abs/2603.12514)  
**代码**: [github.com/shivasmic/3d-trauma-detection-ssl](https://github.com/shivasmic/3d-trauma-detection-ssl)  
**领域**: medical_imaging  
**关键词**: 3D Object Detection, Self-Supervised Learning, Semi-Supervised Learning, Trauma Detection, VDETR

## 一句话总结

提出两阶段标签高效学习框架：先在 1206 例无标注 CT 上用 Masked Image Modeling 自监督预训练 3D U-Net 编码器，再结合 VDETR + Vertex RPE 和 Mean Teacher 半监督学习，仅用 144 例标注数据实现腹部创伤 3D 检测 mAP@0.50 达 45.30%（+115%）。

## 研究背景与动机

1. **腹部创伤检测的临床紧迫性**：急诊中 CT 扫描的人工逐层分析耗时且存在观察者间差异，自动化检测能加速临床决策
2. **标注极度匮乏**：RSNA 腹部创伤数据集 4711 例中仅 206 例（4.4%）有分割标注，传统全监督方法不可行
3. **3D 检测的特殊挑战**：腹腔器官形状不规则，2D 中心点距离不足以描述 3D 包围盒关系；通用 3D 特征提取器迁移效果差
4. **自监督+半监督的互补潜力**：自监督预训练可从无标注数据学习解剖先验，半监督学习可利用大量未标注数据稳定训练
5. **VDETR 的局部性优势**：V-DETR 的 Vertex RPE 通过计算 8 个顶点相对位置编码，提供明确的几何内外关系
6. **现有方法不足**：2D 检测适配 3D 效果差，VoteNet/FCAF3D 依赖大量标注

## 方法详解

### 整体框架

三阶段流程：(1) 3D U-Net 通过 patch-based MIM 在全部 1206 例 CT 上自监督预训练；(2) VDETR 检测器分两阶段训练（冻结→解冻编码器）；(3) Mean Teacher 半监督利用 2000 例未标注数据。

### 核心设计

- **Masked Image Modeling**：将 128³ patch 划分为 8³ 子块，随机遮蔽 75%，U-Net 重建被遮蔽区域的原始强度值
- **3D Vertex RPE**：对每个 query 和体素位置，计算到预测框 8 个顶点的偏移向量，经 MLP 生成注意力偏置：$\mathbf{A} = \text{softmax}(\mathbf{QK}^T + \mathbf{R})$，其中 $\mathbf{R} = \sum_{i=1}^{8}\text{MLP}_i(F(\Delta\mathbf{P}_i))$
- **两阶段训练**：Phase I（0-20 epochs）冻结编码器只训练 VDETR 解码器；Phase II（20-100 epochs）解冻编码器联合微调，编码器学习率为解码器的 1/10（1e-5 vs 1e-4），3 epoch warmup 防止灾难性遗忘
- **Mean Teacher 半监督**：弱增强（高斯噪声 σ=0.01、强度偏移 ±2%）生成教师伪标签，强增强（σ=0.05、偏移 ±10%、弹性变形）生成学生预测，一致性损失包含中心 MSE、尺寸 MSE 和分类 KL 散度
- **分类分支**：冻结编码器 → GAP → 2 层 FC（256→128→7），仅 33,799 可训练参数；加权 BCE 损失中 $w_i^{\text{pos}} = N_i^{\text{neg}} / N_i^{\text{pos}}$（如肠道损伤权重 4.45）处理严重类别不平衡

### 损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda(t) \times (\mathcal{L}_{\text{center}} + \mathcal{L}_{\text{size}} + \mathcal{L}_{\text{cls}})$$

其中 $\lambda(t)$ 从 0 线性增至 0.3（epochs 20-60）。分类任务使用加权 BCE 损失处理类别不平衡。

## 实验关键数据

### 3D 检测验证集消融

| 方法 | Best Epoch | mAP@0.50 | mAP@0.75 |
|------|-----------|----------|----------|
| VDETR (无 SSL) | 5 | 26.36% | 6.82% |
| **VDETR + SSL** | **99** | **56.57%** | **45.12%** |
| 提升 | — | +115% | +562% |

### 3D 检测测试集

| 指标 | 无 SSL | 有 SSL | 提升 |
|------|--------|--------|------|
| mAP@0.50 | 23.03% | 45.30% | +97% |
| mAP@0.75 | 16.67% | 28.72% | +72% |

### 分类任务（冻结编码器 linear probe）

| 类别 | 测试精度 |
|------|---------|
| 肠道损伤 | 97.5% |
| 肝脏高级损伤 | 98.3% |
| 整体平均 | 94.07% |

### 关键发现

- 无半监督时检测在 epoch 5 即达峰值后崩溃（过拟合），加入半监督后训练稳定收敛
- 自监督预训练的特征质量极高：冻结编码器 linear probe 在 epoch 0 即达 94.07%，后续训练无提升
- 半监督意外地对分类任务无帮助（75.4% < 基线 77.7%），可能因伪标签噪声

## 亮点

1. **极端数据效率验证**：仅 144 例标注 + 2000 例未标注即实现可用的 3D 检测
2. **自监督→半监督的协同效应**：预训练提供稳定特征基础，半监督提供隐式正则化，两者结合避免了训练崩溃
3. **Vertex RPE 的 3D 医学适配**：将自然场景 3D 检测的 SOTA 方法迁移到医学领域
4. **完整的系统设计**：数据预处理 → 自监督 → 检测 + 分类全流程开源
5. **自监督质量验证充分**：重建 PSNR 19.39dB + linear probe 76% 精度双重证明预训练特征质量

## 局限性

1. 检测测试集仅 32 例，统计显著性有限
2. 分类 AUC 仅 51.4%（概率校准问题），虽不影响二值预测但反映模型置信度不可靠
3. 仅在 RSNA 腹部创伤数据集验证，未在其他 3D 医学检测任务泛化
4. 相比 RSNA 2023 竞赛冠军（98% AUC，多模型集成），单模型方案差距仍在
5. 计算开销大：全 3D 体素处理需 A100 GPU，batch size 仅为 1-2

## 相关工作

- **3D 检测**：VoteNet、FCAF3D 等全卷积方法依赖大量标注；3DETR/GroupFree 适配 3D 但局部性学习不足
- **V-DETR**：通过 8 顶点 RPE 解决 3D 中局部性问题，本文首次将其应用于医学 3D 检测
- **自监督学习**：MAE 在自然图像有效；Eckstein et al. 验证了预训练对 3D 医学检测的价值
- **半监督检测**：Mean Teacher + Unbiased Teacher 在 2D 检测有效，本文扩展到 3D 医学场景

## 评分

- 新颖性: ⭐⭐⭐ （各组件均为已有方法的组合，但组合方式合理且系统性强）
- 实验充分度: ⭐⭐⭐ （消融实验有说服力，但测试集太小；缺少与其他 3D 检测器的对比）
- 写作质量: ⭐⭐⭐ （结构清晰但篇幅过长，部分细节冗余）
- 价值: ⭐⭐⭐⭐ （标签稀缺的 3D 医学检测是真实痛点，框架有参考价值）
