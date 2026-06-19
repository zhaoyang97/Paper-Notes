---
title: >-
  [论文解读] GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation
description: >-
  [ECCV 2024][医学图像][多模态学习] 提出基于异构图的全模态生物医学表征学习框架 GTP-4o，通过异构图嵌入显式建模跨模态关系，利用图提示机制补全缺失模态，并设计知识引导的层次化跨模态聚合，在胶质瘤分级和生存预测任务上取得SOTA。 问题背景 生物医学诊断中存在多种临床模态（基因组学、病理图像、细胞图谱、诊断文…
tags:
  - "ECCV 2024"
  - "医学图像"
  - "多模态学习"
  - "异构图"
  - "模态缺失补全"
  - "生物医学表征"
  - "图提示"
---

# GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation

**会议**: ECCV 2024  
**arXiv**: [2407.05540](https://arxiv.org/abs/2407.05540)  
**代码**: 有 ([https://gtp-4-o.github.io/](https://gtp-4-o.github.io/))  
**领域**: 医学图像  
**关键词**: 多模态学习, 异构图, 模态缺失补全, 生物医学表征, 图提示

## 一句话总结

提出基于异构图的全模态生物医学表征学习框架 GTP-4o，通过异构图嵌入显式建模跨模态关系，利用图提示机制补全缺失模态，并设计知识引导的层次化跨模态聚合，在胶质瘤分级和生存预测任务上取得SOTA。

## 研究背景与动机

### 问题背景

生物医学诊断中存在多种临床模态（基因组学、病理图像、细胞图谱、诊断文本），整合多模态数据可以从宏观、微观和分子层面全面认识患者状况，实现更准确的疾病诊断。然而将现有多模态学习方法扩展到多种临床模态时面临两大核心挑战：

### 挑战一：模态间语义异构性巨大

自然图像中"狗"与其声音共享相似的对象语义，但基因组学数据与病理图像之间的语义关系高度模糊。不同模态对之间的关系具有不同的语义属性——例如图像与基因组学的关系可抽象为"express"（表达），图像与文本的关系为"depict"（描述），图像与细胞图谱的关系为"atomize"（细化）。现有方法（最优传输、交叉注意力等）忽略了高阶空间中关系本身的异构性。

### 挑战二：临床实践中模态缺失普遍

由于隐私保护、伦理考量和数据采集技术限制，临床场景中部分模态经常缺失（如TCGA-GBMLGG数据集中约40%患者缺失RNA-Seq数据）。大多数多模态方法假设数据完整性，一旦模态缺失，多模态融合就会失效。

### 动机

上述观察启发作者引入一种统一的非欧几里得表征——异构图，来显式捕获模态特征（节点）和跨模态关系（边）的异构属性，并通过图提示机制自适应补全缺失模态的特征空间。

## 方法详解

### 整体框架

GTP-4o 框架分四步：(1) 数据处理和特征提取获得四种模态的嵌入；(2) 将模态特征转化为异构图嵌入 $\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R}\}$；(3) 模态提示补全 $g_\phi(\mathcal{G})$ 处理缺失模态；(4) 知识引导层次化聚合 $\mathcal{M}$ 融合跨模态信息，最终送入任务头预测。

### 关键设计

#### 1. **异构图嵌入 (Heterogeneous Graph Embedding)**

**功能**：将多模态特征统一映射到异构图空间，显式区分不同模态节点和不同跨模态关系。

**核心思路**：定义异构图 $\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R}\}$，其中节点属性集 $\mathcal{A}=\{G, I, C, T\}$ 对应基因组学、病理图像、细胞图谱和文本四种模态，边关系属性集 $\mathcal{R}=\{\text{express}, \text{depict}, \text{atomize}, \text{intra-modal}\}$ 由领域知识定义。节点映射函数 $\tau(v)=a \in \mathcal{A}$，边映射函数 $\varphi(e)=r \in \mathcal{R}$。

**设计动机**：与将所有模态拉到同一欧氏空间的传统方法不同，异构图可以在节点和边两个层面显式编码语义差异，为后续有选择性的跨模态信息交互奠定基础。

#### 2. **模态提示补全 (Modality-Prompted Completion)**

**功能**：当某种模态缺失时，通过图提示机制生成幻觉节点来补全图嵌入，使缺失表征逼近完整表征。

**核心思路**：包含两级提示：

- **通用提示 (General Prompting)**：对缺失模态 $M_\varnothing$，从非缺失样本的该模态特征中提取分布先验，通过高斯采样初始化 $N_P$ 个提示节点 $\mathcal{V}^P \in \mathbb{R}^{N_P \times d}$。
- **实体依赖提示 (Entity-dependent Prompting)**：引入提示库 $\mathcal{V}^{P_B} \in \mathbb{R}^{N_B \times d}$，每个提示节点 $v^P$ 通过线性映射+softmax产生权重 $w$，加权组合提示库组件：

$$v^P \leftarrow v^P + \sum_{i=1}^{N_B} w_i \cdot v_i^{P_B}$$

补全后更新节点集和边集，恢复被缺失模态破坏的图拓扑和关系。

**设计动机**：同一模态不同样本间共享相似的分布，因此可用已有样本的模态先验指导缺失补全。引入提示库使补全具有上下文感知能力，避免与具体样本无关的静态补全。

#### 3. **知识引导层次化聚合 (Knowledge-guided Hierarchical Aggregation)**

**功能**：在补全后的异构图上进行全局邻域发现和局部多关系特征聚合。

**核心思路**：分为全局和局部两层：

- **全局元路径邻域 (Global Meta-path Neighbouring)**：利用领域知识定义元路径集合 $\Phi$，例如 $G \xrightarrow{\text{express}} I \xrightarrow{\text{atomize}} C$，表示基因通过"表达"关系到图像、再通过"细化"关系到细胞图谱的二跳路径。通过随机游走策略在语义关系空间中搜索最优元路径。
- **局部多关系聚合 (Local Multi-Relation Aggregation)**：对每个目标节点 $v_t$，利用多头注意力（MHA）在其元路径邻域 $\mathcal{N}^{\Phi}_{v_t}$ 上进行信息聚合。注意力分数同时考虑节点特征和边特征：

$$\text{SHA}(e,j) = \frac{v_s^{K,j} \cdot e^K_{v_s \to v_t} \cdot v_t^{Q,j}}{\sqrt{d}}$$

最终通过模态特异池化和均值读出层得到图级别特征。

**设计动机**：元路径将领域知识（基因表达→图像表型→细胞形态）编码为信息传播的通道，实现语义有意义的跨模态交互，而非朴素的全连接聚合。

### 损失函数 / 训练策略

- 使用标准负对数似然 (NLL) 损失：$\min_{\mathcal{M},\mathcal{H},\phi} \mathbb{E}_{\mathcal{G}} \mathcal{L}(\mathcal{H}^{\mathcal{T}} \circ \mathcal{M} \circ \phi(\mathcal{G}), y^{\mathcal{T}})$
- 图聚合和任务头的学习率 $1 \times 10^{-3}$，提示参数的学习率较低为 $2 \times 10^{-4}$
- 图嵌入维度 $d=512$，提示节点数和提示库组件数均为 $N_P=N_B=5$
- 数据增强：随机丢弃边和节点、对节点/边特征加高斯噪声
- Adam优化器，150 epochs，早停策略

## 实验关键数据

### 主实验

在 TCGA-GBMLGG 和 TCGA-KIRC 两个癌症数据集上评估胶质瘤分级（AUC/ACC）和生存预测（C-Index）。

| 方法 | 模态 | GBMLGG分级AUC | GBMLGG分级ACC | GBMLGG生存C-Idx | KIRC生存C-Idx |
|------|------|-------|-------|--------|--------|
| SNN | G | 0.8527 | 0.6583 | 0.7974 | 0.6639 |
| TransMIL | I | 0.9149 | 0.7683 | 0.8017 | 0.6876 |
| HEAT | I | 0.9289 | 0.8057 | 0.8223 | 0.7059 |
| Pathomic | G+I | 0.9172 | 0.7618 | 0.8101 | 0.7152 |
| MCAT | G+I | 0.9288 | 0.7929 | 0.8274 | 0.7235 |
| **GTP-4o** | G+I | **0.9256** | **0.8036** | **0.8296** | **0.7273** |
| TransFusion | G+I+C+T | 0.9245 | 0.7986 | 0.8296 | 0.7289 |
| **GTP-4o** | **G+I+C+T** | **0.9389** | **0.8126** | **0.8351** | **0.7336** |

使用全部四种模态时，GTP-4o 在胶质瘤分级AUC上达到0.9389（比TransFusion高1.44%），生存预测C-Index达到0.8351。

### 消融实验

在 TCGA-GBMLGG 上消融各组件（使用全部四种模态）：

| 配置 | 分级AUC | 分级ACC | 生存C-Idx | 说明 |
|------|---------|---------|-----------|------|
| 无异构图嵌入 | 0.9232 | 0.8030 | 0.8168 | 退化为同构图，性能明显下降 |
| 无异构关系 | 0.9259 | 0.8048 | 0.8201 | 边无异构属性区分 |
| 缺失用零初始化 | 0.9087 | 0.7875 | 0.7946 | 不做补全直接填零，性能最差 |
| 缺失直接丢弃 | 0.9288 | 0.8061 | 0.8233 | 丢弃有缺失模态的患者 |
| 无提示库 | 0.9275 | 0.8081 | 0.8280 | 仅用通用提示 |
| 无知识引导聚合 | 0.9350 | 0.8071 | 0.8342 | 使用随机元路径 |
| **完整GTP-4o** | **0.9389** | **0.8126** | **0.8416** | 全部组件 |

### 关键发现

1. **模态缺失处理差异巨大**：直接零初始化（AUC 0.9087）vs 图提示补全（AUC 0.9389）差距为3.02%，说明缺失处理极其重要
2. **各模态对不同任务贡献不同**：基因组学对生存预测更有价值，病理图像对胶质瘤分级更有效
3. **随模态数增加性能持续提升**：从双模态到四模态呈单调提升趋势
4. **补全质量可视化验证**：补全图的边相似度模式与原始完整图高度一致

## 亮点与洞察

1. **首次将异构图引入多模态生物医学表征**：不同于简单拼接或注意力融合，显式建模模态间语义关系（express/depict/atomize），使跨模态交互更有意义
2. **图提示补全思路新颖**：借鉴prompt learning的思想应用到图补全场景，通过可学习的幻觉节点在训练中自适应弥补缺失信息
3. **元路径嵌入领域知识**：将"基因→图像→细胞"等生物学因果关系编码为图信息传播路径

## 局限与展望

1. **合成文本描述存在噪声**：因数据集无真实临床报告，使用 MiniGPT-4 生成文本描述，可能引入数据噪声
2. **数据集规模较小**：仅在TCGA的两个子集（769和417个患者）上验证，泛化性有待更大规模数据验证
3. **未考虑表格类数据**：如年龄、性别等临床元数据未纳入
4. **计算开销**：异构图构建和元路径搜索增加了计算复杂度，文中未报告效率数据

## 相关工作与启发

- **PathomicFusion**：基因组+病理图像双模态融合基础工作
- **MCAT**：引入交叉注意力的多模态融合，但不建模关系异构性
- **HEAT**：在病理图像内使用异构图，但不涉及跨模态场景
- 启发：图提示补全方法可推广到更多模态缺失场景（如多模态MRI中某序列缺失）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 异构图+图提示补全的组合在生物医学多模态学习中属首创
- **实验充分度**: ⭐⭐⭐⭐ — 消融实验详尽，模态组合分析和缺失率分析全面，但数据集仅两个
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，生物学动机阐述充分，符号系统有时略繁琐
- **价值**: ⭐⭐⭐⭐ — 临床场景下模态缺失是真实痛点，但方法通用性需更多领域验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](../../CVPR2026/medical_imaging/omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)
- [\[ICCV 2025\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](../../ICCV2025/medical_imaging/simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)
- [\[ECCV 2024\] Energy-induced Explicit Quantification for Multi-modality MRI Fusion](energy-induced_explicit_quantification_for_multi-modality_mri_fusion.md)
- [\[ECCV 2024\] RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing](radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)

</div>

<!-- RELATED:END -->
