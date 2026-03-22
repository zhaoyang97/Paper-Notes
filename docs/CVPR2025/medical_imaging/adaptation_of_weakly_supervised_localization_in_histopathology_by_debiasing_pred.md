# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**会议**: CVPR2025  
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)  
**代码**: [anonymous.4open.science/r/SFDA-DeP-1797](https://anonymous.4open.science/r/SFDA-DeP-1797/)  
**领域**: medical_imaging  
**关键词**: Source-Free Domain Adaptation, Weakly Supervised Localization, Histopathology, Machine Unlearning, Prediction Bias

## 一句话总结

提出 SFDA-DeP 方法，受机器遗忘启发，通过识别并纠正源模型在目标域的预测偏差（over-predict 某些类别），解决组织病理学中弱监督定位模型跨器官/跨中心域适应时预测偏差被放大的问题。

## 研究背景与动机

1. **WSOL 的临床意义**：弱监督目标定位（WSOL）仅用图像级标签就能同时进行分类和 ROI 定位，大幅降低病理学标注负担
2. **域偏移是核心挑战**：不同机构的染色协议、扫描仪特性、组织处理流程差异导致跨中心部署时性能严重退化
3. **预测偏差的放大效应**：源模型在强域偏移下对某些类别过度预测，伪标签分布高度偏斜；传统 SFDA 方法（如 SFDA-DE）基于自训练，反而会强化这种偏差
4. **源数据不可获取**：Source-Free DA 更符合临床隐私约束，但缺少源数据使得纠偏更困难
5. **定位任务的特殊性**：分类偏差会进一步传导到空间 CAM，导致定位不一致
6. **跨器官偏移最严重**：从 GlaS（结肠）迁移到 CAMELYON16/17（乳腺）时，预测几乎完全偏向 cancer 类

## 方法详解

### 整体框架

SFDA-DeP 将 Source-Free Domain Adaptation 建模为迭代式的偏差识别-纠正过程，包含 forget/retain 集划分、遗忘损失和定位监督三个核心组件。

### 核心设计

- **偏差检测**：统计目标域上各类预测频率，识别过度预测的主导类 $\mathcal{B}$
- **Forget/Retain 集划分**：在主导类样本中，选取归一化熵最高的 top-$\rho$ 样本作为 forget 集 $\mathbb{B}_f$（不确定样本位于决策边界附近），其余为 retain 集
- **Retain 损失**：标准交叉熵，保持可靠样本的伪标签预测：$\mathcal{L}_{\text{retain}} = \mathbb{E}_{x_i \in \mathbb{B}_r}[-\log(p_i(\hat{y}))]$
- **Forget 损失**：反向交叉熵，使模型"忘记"对不确定样本的主导类预测：$\mathcal{L}_{\text{forget}} = \mathbb{E}_{x_i \in \mathbb{B}_f}[-\log(1 - p_i(\hat{y}))]$
- **定位监督**：轻量像素级分类头 $h$，利用 CAM 提取的前景/背景伪标签进行 pixel-level 二分类：$\mathcal{L}_{\text{loc}} = -(1-Y_p)\log(h(z_p)_0) - Y_p\log(h(z_p)_1)$
- **周期性更新**：每 $m$ 个 epoch 重新构建 forget/retain 集，避免伪标签过拟合

### 损失函数

$$\mathcal{L} = \lambda_{\text{retain}}\mathcal{L}_{\text{retain}} + \lambda_{\text{forget}}\mathcal{L}_{\text{forget}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}}$$

## 实验关键数据

### 数据集

- GlaS（结肠腺体分割）、CAMELYON16（乳腺淋巴结）、CAMELYON17（5 个中心 C17-0~C17-4）

### PixelCAM 在 GlaS → 跨域平均性能

| 方法 | PxAP | CL (分类精度) |
|------|------|---------------|
| Source only | 36.9 | 49.3 |
| SFDA-DE | 28.0 | 54.6 |
| ERL | 25.4 | 59.9 |
| RGV | 34.7 | 52.1 |
| **SFDA-DeP (Ours)** | **44.1** | **67.1** |

### SAT 在 GlaS → 跨域平均性能

| 方法 | PxAP | CL |
|------|------|-----|
| Source only | 21.3 | 52.1 |
| SFDA-DE | 21.6 | 68.7 |
| **SFDA-DeP (Ours)** | **30.3** | **69.2** |

### DeepMIL 在 GlaS → 跨域平均性能

| 方法 | PxAP | CL |
|------|------|-----|
| Source only | 20.9 | 49.8 |
| SFDA-DE | 20.5 | 53.9 |
| CDCL | 27.3 | 55.5 |
| **SFDA-DeP (Ours)** | **40.7** | **73.4** |

### 关键发现

- SFDA-DeP 在所有 WSOL backbone（PixelCAM、SAT、DeepMIL）上均一致优于 SOTA SFDA 基线
- PixelCAM 上相比 SFDA-DE 提升 +16.1 PxAP / +12.5 CL；DeepMIL 上提升 +20.2 PxAP / +19.5 CL
- 传统 SFDA 方法（如 SFDA-DE）在强域偏移下反而放大偏差，分类性能有时比 source-only 更差（如 PixelCAM C17-0 上 PxAP 从 37.2 降至 14.5）
- 动态重采样 forget/retain 集是关键组件，静态划分性能明显下降
- 像素级定位损失对 PxAP 提升贡献显著，分类精度也有辅助增益
- 定位和分类两个任务同时获得显著提升

## 亮点

1. **问题发现有价值**：首次系统揭示 SFDA 在 WSOL 场景下因预测偏差放大而失效的机制
2. **机器遗忘的巧妙借用**：将域适应问题类比为"遗忘旧决策边界、建立新边界"
3. **无需源数据**：完全 source-free，符合临床数据隐私要求
4. **通用性强**：在 CNN（ResNet-50）和 Transformer（DeiT-Tiny）backbone 上均有效

## 局限性

1. 仅在二分类（cancer vs normal）场景验证，未扩展到多类细粒度分类（如癌症亚型）
2. forget 比例 $\rho$ 和损失权重均需在验证集上调参，对超参敏感性分析不够充分
3. CAMELYON17 各中心间性能差异较大（如 C17-1 分类反而下降至 41.3%），跨中心鲁棒性仍有提升空间
4. 像素级定位监督依赖 CAM 质量，若源模型 CAM 本身偏差严重则效果受限
5. 未与基于 prompt 的 foundation model 适应方法（如 SAM）进行对比
6. forget 集中的样本被简单地推离主导类，但可能被推向错误的少数类而非真实标签

## 相关工作

- **WSOL 方法**：DeepMIL、SAT、PixelCAM、NEGEV 等通过 CAM 机制从图像级标签获取定位
- **SFDA 方法**：SFDA-DE、CDCL、ERL、RGV 等基于伪标签/聚类的自训练，但在偏斜预测下效果有限
- **机器遗忘**：传统用于隐私删除，本文创新地将其应用于纠正预测偏差而非删除类别

## 评分

- 新颖性: ⭐⭐⭐⭐ （机器遗忘+SFDA+WSOL 的组合切入角度新颖）
- 实验充分度: ⭐⭐⭐⭐ （3 个 WSOL backbone × 多个目标域 × 多个 SFDA 基线）
- 写作质量: ⭐⭐⭐⭐ （问题分析清晰，Fig.1 直观展示偏差放大现象）
- 价值: ⭐⭐⭐⭐ （解决病理学跨中心部署的实际痛点）
