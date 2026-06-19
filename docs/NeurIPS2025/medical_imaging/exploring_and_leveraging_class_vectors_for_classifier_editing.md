---
title: >-
  [论文解读] Exploring and Leveraging Class Vectors for Classifier Editing
description: >-
  [NeurIPS 2025][医学图像][Class Vector] 提出 Class Vector（类向量），通过计算预训练与微调模型在潜空间中类别质心的差异来捕获类别级适应，利用线性和独立性两个性质，通过简单向量算术实现分类器编辑（遗忘、环境适应、对抗防御），无需重训练即可完成潜空间注入，或用 <1.5K 参数在 1.5 秒内完成权重空间映射。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "Class Vector"
  - "分类器编辑"
  - "潜空间操控"
  - "类遗忘"
  - "对抗防御"
  - "Neural Collapse"
---

# Exploring and Leveraging Class Vectors for Classifier Editing

**会议**: NeurIPS 2025  
**arXiv**: [2510.11268](https://arxiv.org/abs/2510.11268)  
**代码**: 有（基于 CLIP ViT）  
**领域**: 医学图像  
**关键词**: Class Vector, 分类器编辑, 潜空间操控, 类遗忘, 对抗防御, Neural Collapse

## 一句话总结

提出 Class Vector（类向量），通过计算预训练与微调模型在潜空间中类别质心的差异来捕获类别级适应，利用线性和独立性两个性质，通过简单向量算术实现分类器编辑（遗忘、环境适应、对抗防御），无需重训练即可完成潜空间注入，或用 <1.5K 参数在 1.5 秒内完成权重空间映射。

## 研究背景与动机

**分类器编辑需求**：深度分类器经大量训练后行为固化，但用户需要事后修改（如遗忘特定类别、适应新环境、修正预测错误），一刀切的分类器无法满足多样化需求

**现有方法局限**：(a) 计算昂贵——ViT 的重训练成本远高于 CNN；(b) 样本需求大——少样本下知识修改困难且容易引入偏差；(c) 粒度粗——现有方法局限于逐图纠错，缺乏类级别编辑能力

**Task Vector 的局限**：任务向量在权重空间捕获任务级修改，但无法解耦单个类别的适应行为——不适用于细粒度分类器编辑

**核心 idea**：在潜空间中提取类级别的表示偏移向量（Class Vector），利用其线性和正交性实现类级别的精确编辑

## 方法详解

### Class Vector 定义

对于类别 $c$，Class Vector $\kappa_c \in \mathbb{R}^m$ 定义为微调和预训练编码器在最后一层表示的期望差：

$$\kappa_c = \mathbb{E}_{s \in S}[f(s, \theta_{\text{ft}}^e)] - \mathbb{E}_{s \in S}[f(s, \theta_{\text{pre}}^e)]$$

微调模型的类质心可分解为：$z_{\text{ft}}^c = z_{\text{pre}}^c + \kappa_c$

### 理论基础

**定理 3.1（预训练-微调间的 CTL）**：在满足 Cross-Task Linearity 条件下，预训练到微调的路径比两个微调模型之间的路径更线性（CTL 偏差更小）。当 $\|\theta_i - \theta_{\text{pre}}\| < \|\theta_i - \theta_j\|$ 时，有 $\delta_{\text{pre},i} < \delta_{i,j}$。

这意味着：
$$f(x_c; \theta_{\text{pre}} + \alpha\tau) \approx f(x_c; \theta_{\text{pre}}) + \alpha\kappa_c, \quad x_c \in \mathcal{D}_c$$

即在潜空间中缩放 $\kappa_c$ 等价于在权重空间中沿任务向量 $\tau$ 移动——类级别的适应可用简单向量算术完成。

### Class Vector 的关键性质

**线性**：两个类之间插值 $z_{\text{edit}} = -\alpha\kappa_{c_1} + \alpha\kappa_{c_2}$，预测和 logits 线性平滑变化，在中点干净地从 $c_1$ 切换到 $c_2$，无绕道其他类。

**独立性**（基于 Neural Collapse）：

**定理 3.3**：假设 (i) 预训练类嵌入塌缩到共同均值 $\bar{z}^{\text{pre}}$；(ii) 微调后嵌入呈 ETF 形式 $z_c^{\text{ft}} = \mu + u_c$，$\sum_c u_c = 0$；(iii) 全局偏移可忽略。则：
$$\cos(\kappa_c, z_{c'}^{\text{ft}}) \approx 0, \quad c \neq c'$$

即任何类的 Class Vector 与其他类的微调嵌入近似正交——修改一个类不影响其他类。

### 编辑方法

**潜空间注入（免训练）**：
1. 计算表示 $r = f(x, \theta_{\text{ft}}^e)$
2. 门控：$\beta = \mathbf{1}[\text{sim}(r) > \gamma]$，仅当 $r$ 与目标类质心 $z_{\text{ft}}^c$ 的余弦相似度超过阈值时激活
3. 注入编辑：$\hat{y} = g(r + \beta \cdot z_{\text{edit}}, \theta^h)$

**权重空间映射（轻量训练）**：
学习映射 $\phi_{\text{edit}}: \mathbb{R}^m \to \mathbb{R}^{d_e}$，使得：
$$\theta_{\text{edit}}^e = \arg\min_{\theta_{\text{edit}}^e} \|f(x, \theta_{\text{edit}}^e) - (f(x, \theta_{\text{ft}}^e) + z_{\text{edit}})\|^2$$

仅训练编码器最后几层的 LayerNorm，用单个参考样本，<1.5 秒完成。**定理 3.2** 证明对于过参数化编码器（$d_e \gg m$），满足条件的映射有无穷多个。

## 实验关键数据

### 类遗忘（ViT-B/16）

| 方法 | MNIST $\text{ACC}_f$↓ | MNIST $\text{ACC}_r$↑ | EuroSAT $\text{ACC}_f$↓ | EuroSAT $\text{ACC}_r$↑ | GTSRB $\text{ACC}_f$↓ | GTSRB $\text{ACC}_r$↑ |
|------|-------|-------|---------|---------|-------|-------|
| Retrained | 0.1 | 76.4 | 0.0 | 85.7 | 41.8 | 57.5 |
| NegGrad | 0.0 | 43.4 | 0.0 | 11.6 | 0.0 | 15.6 |
| Random Vec | 99.9 | 99.8 | 99.9 | 80.9 | 99.6 | 98.2 |
| **Class Vec** | **0.0** | **99.7** | **0.0** | **99.5** | **0.0** | **98.6** |
| **Class Vec†** | **0.0** | **96.2** | **0.0** | **99.7** | **0.0** | **93.4** |

- Class Vector 在遗忘目标类（$\text{ACC}_f \to 0$）的同时几乎完美保留非目标类性能（$\text{ACC}_r > 93\%$）
- 对比 NegGrad：虽然也能遗忘，但严重损害非目标类（$\text{ACC}_r$ 暴跌）
- 随机向量完全无效，验证 Class Vector 指向有意义方向

### 雪景环境适应

| 方法 | ViT-B/16 | ViT-B/32 | ViT-L/14 |
|------|----------|----------|----------|
| Pretrained | 55.2 | 53.4 | 60.2 |
| Retrained | 55.8 | 55.8 | 75.3 |
| DirMatch | 72.0 | 73.9 | 74.6 |
| **Class Vec** | **69.7** | **71.5** | **74.2** |
| **Class Vec†** | **75.2** | **74.5** | **80.2** |

设 $z_{\text{edit}} = \lambda(\kappa_{\text{snow}+c_1} - \kappa_{c_1})$，$\lambda < 0$ 抑制雪特征。仅需 4 个外部样本即可实现 10-20% 提升，且无需逐图训练（vs DirMatch 需要逐图对齐）。

### 排版攻击防御

Class Vector 通过消除排版文本引起的分类偏移来防御排版攻击——用类级别算术移除攻击向量方向上的表示偏移。

## 亮点

- ⭐⭐⭐⭐⭐ **极高效率**：潜空间注入完全免训练；权重映射仅需 <1.5K 参数 + 单样本 + 1.5 秒
- ⭐⭐⭐⭐ **理论扎实**：CTL + Neural Collapse 双重理论支撑，线性和独立性有严格证明
- ⭐⭐⭐⭐ **高级交互**：非专家用户可通过直觉的向量算术（加减缩放）进行概念级编辑
- ⭐⭐⭐⭐ **广泛应用**：同一框架覆盖遗忘、环境适应、对抗防御、后门攻击优化
- ⭐⭐⭐ **架构通用**：在 MLP、ResNet-18、ViT-B/16/32、ViT-L/14 上均验证有效

## 局限与展望

1. **依赖微调模型**：需要同时访问预训练和微调模型来计算 Class Vector，增加存储需求
2. **类别数限制**：独立性基于 Neural Collapse 假设——类别极多时 ETF 结构可能不完美
3. **细粒度不足**：类级别编辑无法处理类内子群体的差异（如不同品种的狗）
4. **门控阈值 $\gamma$**：潜空间注入需要设定余弦相似度阈值，不同任务可能需要不同值
5. **未验证生成模型**：仅在判别式分类器上验证，扩展到生成模型（如 CLIP 引导扩散）有待探索

## 总体评价 ⭐⭐⭐⭐

将 task vector 的思想从任务级精细化到类级别，理论推导优雅（CTL + NC），实验覆盖全面。最大亮点是极致效率——单样本+1.5秒的编辑能力在实际部署中非常有价值。不足在于医学场景的验证较少，而论文被分到 medical_imaging 赛道的理由可能是其在疾病检测分类器编辑中的潜在应用。

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](../../ICML2026/medical_imaging/factored_classifier-free_guidance.md)
- [\[CVPR 2025\] MoEdit: On Learning Quantity Perception for Multi-Object Image Editing](../../CVPR2025/medical_imaging/moedit_on_learning_quantity_perception_for_multi-object_image_editing.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[ICML 2025\] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models](../../ICML2025/medical_imaging/raptor_scalable_train-free_embeddings_for_3d_medical_volumes_leveraging_pretrain.md)

</div>

<!-- RELATED:END -->
