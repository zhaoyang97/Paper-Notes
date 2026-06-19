---
title: >-
  [论文解读] SIC: Similarity-Based Interpretable Image Classification with Neural Networks
description: >-
  [ICCV 2025][医学图像][可解释性] 提出 SIC，一个同时提供局部、全局和忠实解释的内在可解释神经网络：通过从训练图像中提取类别代表性的支持向量，基于 B-cos 变换计算输入与支持向量的相似度进行分类，在保持与黑盒模型相当准确率的同时，提供像素级贡献图和基于案例推理的全局解释，在 FunnyBirds 基准上 9 项可解释性指标中 8 项超越 ProtoPNet。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "可解释性"
  - "基于相似度的分类"
  - "B-cos网络"
  - "支持向量"
  - "案例推理"
---

# SIC: Similarity-Based Interpretable Image Classification with Neural Networks

**会议**: ICCV 2025  
**arXiv**: [2501.17328](https://arxiv.org/abs/2501.17328)  
**代码**: [github.com/ai-med/SIC](https://github.com/ai-med/SIC)  
**领域**: 医学图像  
**关键词**: 可解释性, 基于相似度的分类, B-cos网络, 支持向量, 案例推理

## 一句话总结

提出 SIC，一个同时提供局部、全局和忠实解释的内在可解释神经网络：通过从训练图像中提取类别代表性的支持向量，基于 B-cos 变换计算输入与支持向量的相似度进行分类，在保持与黑盒模型相当准确率的同时，提供像素级贡献图和基于案例推理的全局解释，在 FunnyBirds 基准上 9 项可解释性指标中 8 项超越 ProtoPNet。

## 研究背景与动机

### 问题定义

深度学习在医学等高风险领域的部署要求模型的决策过程是透明的。可解释 AI (XAI) 方法分为两类：
- **事后解释**（Post-hoc）：对黑盒模型用近似方法估计特征贡献（如 SHAP、Grad-CAM），但高维输入上存在不可避免的近似误差
- **内在可解释性**（Inherently interpretable）：模型设计本身就提供解释，无需近似

理想的内在可解释模型应同时具备三个属性：**局部解释**（单个预测的依据）、**全局解释**（模型整体行为的概括）、**忠实性**（解释满足完备性、灵敏性等数学公理）。

### 已有方法的不足

| 方法 | 局部 | 全局 | 忠实 | 核心问题 |
|------|-----|------|------|---------|
| NW-Head | ✓ | ✓ | ✗ | 基于 softmax 归一化的欧氏距离，无像素级解释；对抗攻击下仍给出高置信度 |
| ProtoPNet | ✓ | ✓ | ✗ | 将距离图从潜空间上采样到图像空间，但空间对应关系无法保证，解释可能误导 |
| BagNet | ✓ | ✗ | ✓ | 独立分类局部 patch 后求和，缺乏全局解释 |
| B-cos | ✓ | ✗ | (✓) | 每次前向传播总结为线性变换提供局部解释，但权重矩阵是输入依赖的，缺乏全局解释 |

### 核心动机

**关键洞察**：结合 B-cos 网络（提供忠实的像素级解释）和 Nadaraya-Watson 分类头（提供基于案例推理的全局解释），就能同时满足局部、全局和忠实三个要求。关键创新在于引入"证据预测器"将特征映射为非负向量，使得相似度计算天然排除了负贡献，同时通过 k-means 聚类选择固定的类代表性支持向量，将复杂的模型行为简化为"这张测试图像的哪些部分与哪些训练图像的哪些部分相似"的直觉推理。

## 方法详解

### 整体框架

SIC 由三部分组成：
1. **B-cos 特征提取器** $\mathcal{F}_\theta$：将图像映射为潜向量 $f$
2. **证据预测器** $\mathcal{E}$：将 $f$ 转为非负向量 $f^+$，计算与支持向量的相似度
3. **类别 logit**：对每类的支持向量相似度求和得到分类结果

### 关键设计

#### 1. **B-cos 特征提取器**

- **功能**：提取图像的潜空间表示，同时保证每次前向传播可总结为单一线性方程，实现忠实的像素级解释。
- **核心思路**：

  B-cos 变换重新定义了标量积，引入对齐指数 $B$ 增强权重-输入的对齐：

  $$\text{B-cos}(x; w) = \|x\| \cdot |\cos(x, \hat{w})|^B \times \text{sgn}(\cos(x, \hat{w}))$$

  其中 $\hat{w} = w/\|w\|$。每一层的计算都可表示为输入依赖的线性方程 $\text{B-cos}(x, \mathbf{W}) = \tilde{\mathbf{W}}(x) x$，其中：

  $$\tilde{\mathbf{W}}(x) = |\cos(x, \hat{\mathbf{W}})|^{B-1} \odot \hat{\mathbf{W}}$$

  对于 $L$ 层网络，整个前向传播总结为：

  $$\mathcal{F}_\theta(x) = \left(\prod_{j=1}^{L} \tilde{\mathbf{W}_j}(a_j)\right) x = \mathbf{W}_{1 \rightarrow L}(x) \cdot x$$

  像素 $(m,n)$ 的贡献图为：

  $$\phi_j^l(x)_{(m,n)} = \sum_{ch} \left[ [\mathbf{W}_{1 \rightarrow l}]_j^T \odot x \right]_{(ch,m,n)}$$

  输入编码为 6 通道 $[R, G, B, 1-R, 1-G, 1-B]$ 以唯一编码颜色并避免偏向亮区域。

- **设计动机**：B-cos 变换迫使网络聚焦于与权重最对齐的特征，训练中自然学会了只关注最显著的数据特征。线性总结特性保证了解释是前向传播过程的精确描述，而非近似。

#### 2. **证据预测器与支持向量分类**

- **功能**：将实值特征映射为非负向量，提取类代表性的固定支持向量，通过相似度计算进行分类。
- **核心思路**：

  证据预测器包含两部分：
  - **非负映射** $\oplus: \mathbb{R}^d \rightarrow \mathbb{R}_{\geq 0}^d$，将特征 $f$ 转为 $f^+ = \oplus(f)$
  - **相似度度量** $sim(f^+, v_i^c)$，计算输入与每个支持向量的相似度

  类别 logit 定义为：

  $$\mu_c = b + \sum_{v_i^c} \frac{sim(f^+, v_i^c)}{\mathcal{T}}$$

  其中 $b$ 是固定偏置，$\mathcal{T}$ 是温度参数，$v_i^c$ 是类 $c$ 的第 $i$ 个支持向量。

  **支持向量选择**：每个 epoch 后，对每类的所有训练样本特征进行 k-means 聚类，取每个聚类中心最近的真实样本特征作为支持向量：

  $$v_i^c = \arg\min_{f_k^+ | y_k = c} \|f_k^+ - \gamma_i^c\|_2$$

  这确保支持向量是真实训练样本的特征（而非人工构造的原型），便于案例推理解释。

- **设计动机**：
    - 非负映射 $\oplus$ 确保只有正贡献能增加类概率，解决了 NW-Head 中 softmax+欧氏距离的问题（对抗攻击下仍给出高置信度）
    - k-means 选择使支持向量覆盖类内的多样性，同时将需要审查的解释数量限制在 $N_s$ 个
    - 温度参数 $\mathcal{T}$ 控制 logit 大小，避免 B-cos 变换中出现负贡献（过小的温度导致模型通过负贡献来缩放范围）

#### 3. **三层解释机制**

- **功能**：提供全局+局部+像素级的完整解释链。
- **核心思路**：

  **全局解释**（支持向量贡献图）：对每个支持向量 $v_i^c$，用其对应的训练图像计算 B-cos 贡献图。由于 $v_i^c$ 与自身完全对齐（$\cos = 1$），B-cos 变换简化为 $\text{B-cos}(v_i^c, \hat{v}_i^c) = \|v_i^c\|$。贡献图展示了模型从每个支持样本中"学到了什么"——哪些像素区域被编码为该类的代表特征。

  **局部解释**三层结构：
  1. **支持证据**（Support evidence）：$\frac{sim(f^+, v_i^c)}{\mathcal{T}}$ 量化了测试图像与每个支持样本的对齐程度，以及各自对 log-probability 的贡献比例
  2. **测试贡献图**（Test contribution）：利用 B-cos 的线性性质，对类 logit 计算像素级贡献，展示测试图像的哪些部分与支持向量对齐
  3. **支持贡献图**（Support contribution）：支持向量的全局 RGBA 解释，用户可检查支持贡献和测试贡献的交集

  **理论保证**：证明 SIC 的解释满足 Sundararajan 等人提出的 6 个公理——完备性、灵敏性、实现不变性、虚拟性、线性和对称性保持。

- **设计动机**：三层解释构成了完整的推理链——全局解释告诉开发者"模型学了什么"，局部解释告诉用户"这个预测为什么"，像素级贡献图告诉用户"具体看的是图像的哪里"。案例推理（"这张图的这个部分像那张训练图的那个部分"）是人类最直觉的理解方式。

### 损失函数 / 训练策略

- **损失函数**：Binary cross-entropy loss（比标准多类交叉熵产生更强的对齐压力）
- **训练流程**：
  1. 正常训练 B-cos backbone + 证据预测器
  2. 每个 epoch 结束后重新计算全部训练样本特征，k-means 聚类更新支持向量
  3. 训练时随机采样每批的支持向量，测试时使用固定的 k-means 支持向量
- **架构**：支持 DenseNet121 (~8M)、ResNet50 (~26M)、Hybrid ViT (~81M) 三种 backbone
- **支持向量数** $N_s$：每类固定数量

## 实验关键数据

### 主实验

**三个数据集、ResNet50 backbone 的准确率对比**：

| 方法 | Pascal VOC (mAP) | Stanford Dogs (Acc) | RSNA (Bal.Acc) | 可解释性 |
|------|-----------------|--------------------|--------------|---------| 
| 黑盒 ResNet50 | ~83% | ~83% | ~80% | 无 |
| ProtoPNet | ~80% | ~75% | ~76% | 局部+全局（不忠实） |
| BagNet17 | ~78% | ~72% | ~74% | 局部（忠实） |
| NW-Head | ~82% | ~80% | ~78% | 局部+全局（不忠实） |
| B-cos | ~83% | ~82% | ~80% | 局部（忠实） |
| **SIC** | **~83%** | **~79%** | **~79%** | **局部+全局+忠实** |

SIC 在 Pascal VOC 和 RSNA 上与黑盒模型持平（≥+0.29% 和 -0.65%），Stanford Dogs 上稍低（-3.7%），但**优于 14 个对比模型中的 9 个**。

### 消融实验

**FunnyBirds 可解释性基准评估（ResNet50 backbone）**：

| 指标 | ProtoPNet | BagNet | B-cos | **SIC** |
|------|----------|--------|-------|--------|
| Accuracy | ~92% | **~97%** | ~95% | ~96% |
| Background Independence | ~0.85 | ~0.95 | ~0.97 | **~0.99** |
| Completeness (avg) | ~0.90 | ~0.95 | ~0.93 | **~0.97** |
| Correctness | ~0.65 | **~0.76** | ~0.70 | ~0.72 |
| Contrastivity | ~0.85 | **~0.99** | ~0.90 | ~0.95 |
| CSDC | ~0.88 | ~0.94 | ~0.91 | **~0.96** |
| Preservation Check | ~0.92 | ~0.96 | ~0.95 | **~0.98** |
| Deletion Check | ~0.90 | ~0.96 | ~0.93 | **~0.97** |
| Distractability | ~0.88 | ~0.94 | ~0.94 | **~0.96** |

SIC 在 9 项可解释性指标中 **8 项超越** ProtoPNet 和 B-cos，仅在 Correctness 和 Contrastivity 上略低于 BagNet。

### 关键发现

1. **可解释性与准确率的平衡**：SIC 在大多数数据集上仅损失 0-3.7% 准确率即获得完整的解释能力
2. **近乎完美的背景独立性（0.99）**：证明支持向量几乎完全排除了背景特征
3. **高完备性（0.97）**：表明支持向量编码了所有类代表性特征
4. **支持向量的多样性**：k-means 聚类有效捕获了类内的多样性（如不同角度的狗、不同光照条件的 X 光片）
5. **注意力随类别转移**：多标签分类（Pascal VOC）中，模型对不同类预测时聚焦于图像的不同区域，证明潜向量高效编码了多类特征

## 亮点与洞察

1. **首次同时满足三个可解释性要求**：局部、全局、忠实解释的统一是一个重要的理论突破
2. **案例推理的实用价值**：在医学影像诊断中，"这个预测因为 X 光片的这个区域像训练集中某患者的某区域"这种解释方式对医生非常直觉
3. **模型调试应用**：通过分析支持向量的相似度矩阵和 t-SNE 投影，可以发现潜空间中的类混淆问题（如 RSNA 数据中支持向量 0 和 3 虽属不同类但投影接近）
4. **理论贡献**：严格证明了 SIC 解释满足 6 个公理，首次为基于案例推理的可解释模型提供了形式化保证

## 局限与展望

1. **类别数可扩展性**：k-means 聚类的开销随类别数增长，但 120 类的 Stanford Dogs 实验证明中等规模可行
2. **Stanford Dogs 准确率下降最明显（-3.7%）**：细粒度分类任务中 B-cos 的对齐约束限制了特征提取的灵活性
3. **支持向量数量需手动设定**：$N_s$ 的最优值取决于类内多样性，目前没有自适应选择机制
4. **B-cos 训练约束**：B-cos 变换要求所有层使用特殊的标量积，限制了可用的 backbone 架构
5. **温度参数敏感**：过小的温度会导致支持向量贡献图出现负贡献，需要仔细调节

## 相关工作与启发

- 与 ProtoPNet 的本质区别：ProtoPNet 学习类特定的原型"部分"，通过上采样距离图解释。SIC 提取完整图像的支持向量，通过 B-cos 线性总结提供忠实的像素级解释，避免了空间对齐问题
- 与 B-cos 的互补关系：B-cos 提供忠实的局部解释但缺乏全局解释。SIC 通过引入固定支持向量增加了全局解释维度
- 医学应用启发：RSNA 实验展示了如何验证模型确实关注了病灶区域（贡献图与边界框的交集），这种验证方式在临床部署评估中非常有价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ — B-cos + NW-Head 的组合虽基于已有工作，但引入证据预测器和支持向量选择后实现了三重解释的统一，理论证明扎实
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、三种 backbone、FunnyBirds 可解释性基准评估全面，但缺少与更多可解释方法（如 Concept Bottleneck Models）的直接对比
- **写作质量**: ⭐⭐⭐⭐⭐ — 定位清晰（Tab 1 一目了然），解释示例丰富且有洞察力（如 Pascal VOC 的多标签分析）
- **价值**: ⭐⭐⭐⭐ — 在医学等高风险领域的可解释分类上具有实际部署价值，理论保证增强了可信度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](../../NeurIPS2025/medical_imaging/firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)
- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](../../CVPR2025/medical_imaging/interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](../../CVPR2025/medical_imaging/unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)
- [\[NeurIPS 2025\] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray](../../NeurIPS2025/medical_imaging/radzero_similarity-based_cross-attention_for_explainable_vision-language_alignme.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)

</div>

<!-- RELATED:END -->
