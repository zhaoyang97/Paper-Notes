---
title: >-
  [论文解读] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging
description: >-
  [AAAI 2026][医学图像][舌象诊断] 提出MIRNet框架，将自监督掩码自编码器（MAE）预训练与约束感知的图注意力网络（GAT）推理相结合，用于舌象多标签诊断，并发布包含4000张图像22个标签的TongueAtlas-4K基准数据集，Macro Recall提升77.8%、Macro-F1提升33.2%。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "舌象诊断"
  - "图注意力网络"
  - "自监督预训练"
  - "临床约束优化"
  - "多标签分类"
---

# MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging

**会议**: AAAI 2026  
**arXiv**: [2511.10013](https://arxiv.org/abs/2511.10013)  
**代码**: [GitHub](https://github.com/zijie8247/MIRNet)  
**领域**: 医学图像分析 / 舌象诊断  
**关键词**: 舌象诊断, 图注意力网络, 自监督预训练, 临床约束优化, 多标签分类

## 一句话总结

提出MIRNet框架，将自监督掩码自编码器（MAE）预训练与约束感知的图注意力网络（GAT）推理相结合，用于舌象多标签诊断，并发布包含4000张图像22个标签的TongueAtlas-4K基准数据集，Macro Recall提升77.8%、Macro-F1提升33.2%。

## 研究背景与动机

医学图像诊断需要结合精细视觉模式识别与领域知识进行推理，尤其需要理解统计相关的诊断标签和临床先验之间的复杂关系。舌象分析是中医诊断的重要手段，"淡舌"往往与"白苔"同时出现，但这类领域知识在现有方法中仍未被充分利用。

现有舌象诊断方法存在四个相互关联的问题：

**问题一：标注稀缺**。专业医学影像标注成本高昂且耗时，严重制约了监督学习的效果。现有工作如jiang2022deep使用的8676张标注图像数据集并未公开，且仅覆盖7个类别。

**问题二：标签严重不平衡**。舌象诊断中不同症状的发生率差异巨大，例如"白苔"高达78.38%而"暗红舌"仅2.15%，导致模型对罕见病征的检测能力极差。

**问题三：标签相关性建模不足**。诊断标签之间存在显著的统计共现关系（如"薄舌"与"胖大舌"互斥），但现有方法（如独立ResNet + 后期融合、双向RNN建模）缺乏系统性的标签依赖建模。

**问题四：缺乏临床合理性约束**。预测结果可能违反医学常识（如同时预测互斥的诊断），现有模型没有机制来防止这种不合理输出。

**核心Idea**：设计一个端到端框架，用MAE预训练解决标注稀缺、用GAT建模标签依赖关系、用约束感知优化确保临床合理性、用非对称损失处理标签不平衡，四管齐下系统性解决全部挑战。

## 方法详解

### 整体框架

MIRNet采用编码器-解码器架构：MAE预训练的ViT编码器提取图像特征，基于标签共现构建的图通过GAT解码器建模标签间依赖关系，最终通过约束感知的多目标优化函数联合训练。整个流程为：图像 → MAE编码器 → 视觉特征 → 标签图 + GAT → 图精炼特征 → 约束感知优化 → 多标签预测。

### 关键设计

1. **掩码自编码器（MAE）预训练**:

    - 功能：在大规模无标注舌象数据上学习可迁移的视觉表示，解决标注稀缺问题
    - 核心思路：将输入图像分为 $N$ 个不重叠的patch，以75%的高掩码率随机遮挡，ViT编码器处理可见patch生成特征，轻量级解码器重建被遮挡区域。损失函数为被遮挡区域的像素级MSE：$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2$
    - 设计动机：利用15,905张未标注图像进行预训练，使编码器学会舌体的解剖结构特征，为下游任务提供强初始化。ViT-Base-Patch16-224架构，embed_dim=768, depth=12, num_heads=12
    - 预训练编码器输出的视觉特征用于初始化GAT中标签节点的嵌入

2. **基于图注意力网络的标签相关性建模**:

    - 功能：在由诊断标签构成的图上传播信息，捕捉标签间的高阶相关性
    - 核心思路：首先从训练集标注矩阵构建标签共现图。共现矩阵 $\mathbf{M} = \mathbf{Y}^\top\mathbf{Y} - \text{diag}(\mathbf{Y}^\top\mathbf{Y})$，通过动态阈值（非零共现值的第25百分位）构建稀疏邻接矩阵。然后用两层GATv2Conv进行标签传播：在每一层中，通过注意力系数 $\alpha_{ij}$ 聚合邻居信息更新节点表示，最终将初始MAE特征 $\mathbf{v}_k^{(0)}$ 与图精炼特征 $\mathbf{v}_k^{(L)}$ 拼接后通过分类头预测：$\hat{y}_k = \sigma(\mathbf{w}_k^\top [\mathbf{v}_k^{(0)} \| \mathbf{v}_k^{(L)}] + b_k)$
    - 设计动机：保留局部视觉证据的同时注入上下文感知的标签相关性。两项增强机制进一步提升效果：**罕见标签增强**（按逆频率对数重新缩放注意力权重）和**相关性置信度加权**（用归一化共现频率加权注意力边）

3. **约束感知优化**:

    - 功能：将临床知识以可微约束的形式融入训练，确保预测结果符合医学合理性
    - 核心思路：统一优化框架为 $\min_{\theta,\phi} \mathcal{L}_{\text{ASL}} + \lambda_1 \mathcal{L}_{\text{constraint}} + \lambda_2 \mathcal{L}_{\text{prior}}$，包含三部分：
        - **非对称损失（ASL）**：通过频率加权 $\gamma_k = \sqrt{\tau/\mathbb{P}(y_k=1)}$ 和不对称聚焦参数 $\zeta_+ < \zeta_-$ 处理正负样本的不平衡
        - **临床知识约束**：通过Lagrange松弛将互斥约束（$p_a \cdot p_b$）、共现约束（$|p_a - p_b|$）和蕴含约束（$p_a \cdot (1-p_b)$）编码为可微惩罚项，仅在违反时产生损失
        - **统计先验正则化**：用KL散度 $\text{KL}(q(\mathbf{y}|\mathbf{X}) \| p_{\text{data}}(\mathbf{y}))$ 对齐预测的边际分布与经验类先验
    - 设计动机：$\lambda_1=0.1, \lambda_2=0.05$，通过梯度优化自然满足约束，无需硬编码规则

4. **Boosting集成策略**:

    - 功能：进一步改善低频标签的分类性能
    - 核心思路：基础模型在全数据上训练，第二个模型仅在F1<0.5的表现不佳类别上使用增强数据微调。最终预测：5个最差标签使用第二个模型输出，其余保留基础模型输出

### 损失函数 / 训练策略

- 使用AdamW优化器，基础学习率 $1 \times 10^{-3}$，batch size 200，逐层衰减 $\text{layer\_decay}=0.75$，训练200个epoch
- 在NVIDIA A800 GPU上训练
- 数据集按80/10/10比例划分为训练/验证/测试集
- 图像预处理包括严格的颜色校正、DeepLabV3+分割和手动精修

## 实验关键数据

### 主实验

| 模型 | Example-F1 | Micro-F1 | Macro-F1 | Macro Recall | Macro PR-AUC |
|------|-----------|----------|----------|-------------|-------------|
| LGAN | 0.634 | 0.640 | 0.397 | 0.369 | 0.492 |
| Faster R-CNN | 0.651 | 0.662 | 0.381 | 0.339 | 0.493 |
| DenseNet121 | 0.648 | 0.657 | 0.403 | 0.364 | 0.351 |
| C-GMVAE | 0.634 | 0.647 | 0.346 | 0.305 | 0.526 |
| **MIRNet** | **0.680** | **0.683** | **0.525** | **0.599** | 0.527 |
| **MIRNet-Boosting** | 0.675 | 0.678 | **0.537** | **0.655** | **0.543** |

### 消融实验

| 配置 | Example-F1变化 | Macro-F1变化 | Macro Recall变化 | 说明 |
|------|---------------|-------------|-----------------|------|
| MIRNet-C (去约束) | -3.2% | -4.4% | - | 标签一致性受损 |
| MIRNet-G (GAT→MLP) | - | -3.2% | -8.1% | 标签依赖建模不可或缺 |
| MIRNet-P (去预训练) | - | **-23.0%** | **-29.0%** | 影响最严重，预训练至关重要 |

### 关键发现

- MIRNet-Boosting在Macro-F1和Macro Recall上分别比最强基线提升33.2%和77.8%，改善幅度惊人
- 即使不用Boosting，MIRNet也超越所有基线（Macro Recall +62.5%, Macro-F1 +30.4%）
- 在罕见标签上改善尤为显著：暗红舌（2.15%出现率）从所有基线的F1<0.25提升到0.68，灰黑苔从极低提升到0.71
- 维度级漏检大幅减少，舌形维度漏检从基线最高的209例降至MIRNet-Boosting的14例（减少93.3%）
- 消融分析显示三个组件互补：预训练对稀有类影响最大（Macro Recall -29%），约束提供整体最大增益（防止不一致标签），GAT保持精确率/召回率平衡
- MIRNet在四个诊断维度上均表现均衡：舌色0.81、舌形0.77、苔质0.76、苔色0.84，而基线平均分别为0.59、0.43、0.51、0.68

## 亮点与洞察

- 将"学习与推理的集成"理念（类似DRNets/PINNs中物理约束的做法）引入医学图像诊断领域，是一个有价值的方向迁移
- 约束感知优化的设计非常优雅——通过 $\max(0, \phi_j)$ 将硬约束软化为可微损失，既保留了约束的语义又不影响梯度传播
- GAT中罕见标签增强和相关性置信度加权两个小技巧虽简单但有效，值得在其他多标签分类场景中借鉴
- TongueAtlas-4K数据集的发布（4000张图、22个标签、10名专家共识标注）填补了舌象分析缺乏公共基准的空白

## 局限与展望

- 尽管论文声称框架可泛化到其他医学影像任务，但所有实验仅在舌象数据集上进行，缺乏在X-ray、CT等其他模态上的验证
- 标签间的约束规则需要人工定义（互斥、共现、蕴含），在新领域中需要领域专家重新设计
- 22个标签的分类粒度对于实际中医临床可能仍不够细致
- Boosting策略的阈值设定（F1<0.5）和替换标签数量（5个）看起来是经验性选择，鲁棒性未充分验证
- 标签图的构建依赖于训练集统计，当训练集较小或标签分布有偏时，图结构可能不可靠

## 相关工作与启发

- 与DRNets的理念一脉相承——DRNets将热力学先验融入深度学习用于材料发现，MIRNet将中医临床先验融入用于舌象诊断
- 相比LGAN（CNN+双向RNN建模标签相关）和IFRCNet（膨胀卷积+注意力），MIRNet的图推理方式更显式且可解释
- ASL损失函数在多标签分类中处理不平衡的策略值得在其他领域推广
- 预训练对性能的决定性影响（-23% Macro-F1）再次印证了在标注稀缺场景下自监督预训练的重要性

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[CVPR 2026\] Med-CMR: A Fine-Grained Benchmark Integrating Visual Evidence and Clinical Logic for Medical Complex Multimodal Reasoning](../../CVPR2026/medical_imaging/med-cmr_a_fine-grained_benchmark_integrating_visual_evidence_and_clinical_logic_.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](../../CVPR2025/medical_imaging/revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
