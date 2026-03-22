# Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**会议**: CVPR2025  
**arXiv**: [2603.11818](https://arxiv.org/abs/2603.11818)  
**代码**: 未公开  
**领域**: medical_imaging  
**关键词**: Ovarian Cancer, CNN, Explainable AI, LIME, SHAP, Integrated Gradients

## 一句话总结

使用 15 种 CNN 变体（LeNet、ResNet、VGG、Inception）在组织病理学图像上检测卵巢癌及亚型，选择 InceptionV3（ReLU）作为最优模型（平均 94.58%），并使用 LIME、SHAP、Integrated Gradients 三种 XAI 方法解释模型预测。

## 研究背景与动机

1. **卵巢癌的高致死性**：全球女性第 7 大常见癌症，因缺乏早期筛查手段通常在晚期才被发现，转移率高
2. **现有检测方法的局限**：经阴道超声和 CA-125 血检准确性不足，确诊需依赖侵入性活检
3. **深度学习在癌症检测中的潜力**：已在乳腺癌、宫颈癌等领域成功应用，但卵巢癌领域研究较少
4. **黑盒问题**：医疗 AI 的决策可解释性对临床接受至关重要
5. **数据集可用性**：Mendeley 提供的 OvarianCancer&Subtypes 组织病理数据集包含 5 个类别
6. **前人工作的不足**：Kasture et al. 使用 VGG16 仅达 84.64% 精度，且依赖更大的增强数据集（24742 张）

## 方法详解

### 整体框架

数据增强 → 张量转换与归一化 → 15 种 CNN 变体训练比较 → 选择 InceptionV3-A → XAI 可解释性分析。

### 核心设计

- **数据增强**：使用 Albumentations 库进行旋转（至 180°）、水平/垂直翻转、亮度/对比度/饱和度/色调调节，从 498 张扩展到 2490 张（5 类各 498 张）
- **模型选择**：测试 3 种 LeNet、4 种 ResNet（34/50/101）、4 种 VGG（16-A/B/C, 19）、4 种 Inception（V1-A/B, V3-A/B）
- **InceptionV3-A 架构**：在 InceptionV1 基础上加入 Batch Normalization，使用 3×3 卷积替代 7×7，修改 Inception 模块 filter 配置，ReLU 激活，Softmax 输出层
- **VGG 使用迁移学习**，而 Inception 从头训练——VGG 虽精度最高但迁移学习的冻结层使 XAI 解释极为困难，因此选择从头训练的 InceptionV3 以便于 XAI 分析
- **XAI 方法**：LIME（局部可解释模型，限定展示 10 个关键特征）、Integrated Gradients（梯度积分归因）、SHAP（Shapley 值局部变体）
- **超参调优**：ResNet 系列采用随机搜索策略，在学习率 [0.0001, 0.1] 和 Dropout [0.0, 0.9] 范围内随机采样 10 组参数

### 损失函数

分类交叉熵损失，5 类 Softmax 输出：$\text{softmax}(z)_i = e^{z_i} / \sum_{j=1}^{N} e^{z_j}$

## 实验关键数据

### 模型性能对比（前 5 名，增强数据集）

| 模型 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| VGG19 | **97.19%** | **97.31%** | **97.19%** | **97.20%** |
| VGG16-A | 96.99% | 96.98% | 96.99% | 96.97% |
| VGG16-B | 96.18% | 96.27% | 96.18% | 96.20% |
| VGG16-C | 96.18% | 96.32% | 96.18% | 96.18% |
| **InceptionV3-A** | **94.58%** | **94.75%** | **94.58%** | **94.62%** |

### 与前人对比

| 模型 | 原始数据集 | 增强数据集 |
|------|-----------|-----------|
| VGG16-O (Kasture et al.) | 50% | 84.64% (20 epoch, 24742 images) |
| VGG16-A (本文) | 77.78% | 96.99% (80 epoch, 2490 images) |
| InceptionV3-A (本文) | 20.20% | 94.58% (80 epoch, 2490 images) |

### 关键发现

- VGG 变体性能最高但因迁移学习不利于 XAI 解释而被放弃
- InceptionV3 从头训练在原始小数据集上仅 20.20%，但增强后性能大幅提升
- 三种 XAI 方法在 Serous 类上显示出一致的关键特征区域，验证了模型决策的合理性
- LIME 和 SHAP/IG 的差异源于 LIME 限制了显示的关键特征数量（仅 10 个），而非模型解释冲突

## 亮点

1. **系统性模型比较**：15 种 CNN 变体的全面横向对比，覆盖经典到现代架构
2. **XAI 多方法交叉验证**：LIME、SHAP、Integrated Gradients 三种互补方法的比较分析提升了可信度
3. **数据效率**：相比前人用 24742 张图像，仅用 2490 张增强数据即超过前人精度
4. **模型选择考虑了可解释性**：没有盲目选最高精度模型，而是考虑 XAI 兼容性

## 局限性

1. **数据集极小**：原始仅 498 张图像（每类约 100 张），增强到 2490 张仍远小于临床需求，结论的泛化能力存疑
2. **方法创新有限**：纯粹的模型选型+调参工作，无新架构或新方法提出
3. **缺少外部验证**：仅在单一 Mendeley 数据集上 80-20 划分，无跨数据集、跨机构泛化测试
4. **未使用非侵入性数据**：宣称目标是非侵入检测但使用的是组织病理图像（需手术/活检获取），与动机矛盾
5. **ResNet 性能异常差**（ResNet-50 仅 34.14%），未深入分析原因，随机搜索仅 10 次迭代且仅训练 3 epoch 选超参，可能严重不足
6. **VGG 迁移学习 vs Inception 从头训练的公平性**：比较前提不一致，迁移学习天然占优
7. **缺少与现代方法对比**：未与 ViT、Swin Transformer、CLIP 等较新方法比较
8. **超参搜索策略粗糙**：ResNet 仅用 10 次随机采样、3 epoch 训练选超参，不足以找到合理配置
8. **XAI 分析较浅**：仅展示了个别样本的可视化，未进行定量的 faithfulness 评估

## 相关工作

- **卵巢癌 AI 检测**：Zhou et al. 综述 AI 在卵巢癌诊断中的应用；Hema et al. 的 FaRe-ConvNN 达 97% 精度
- **CNN 分类架构**：LeNet-5、ResNet、VGGNet、GoogLeNet/Inception 各有特点
- **XAI 方法**：LIME（Ribeiro 2016）、SHAP（Lundberg 2017）、Integrated Gradients（Sundararajan 2017）
- **前人同数据集工作**：Kasture et al. 使用 VGG16 在增强数据上达 84.64%
- **OCT 方向探索**：Schwartz et al. 使用 OCT 记录 + LSTM 检测卵巢癌（AUC 0.81），提供了非侵入性检测的另一路径

## 评分

- 新颖性: ⭐⭐ （标准 CNN 比较实验 + XAI 应用，缺少方法层面创新）
- 实验充分度: ⭐⭐ （数据集太小，无外部验证，ResNet 异常未分析）
- 写作质量: ⭐⭐⭐ （结构完整但叙述冗余）
- 价值: ⭐⭐ （作为应用型工作基本完成目标，但离实际临床部署差距较大）
