---
title: >-
  [论文解读] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models
description: >-
  [AAAI2026][医学图像][open-set classification] 提出 Coarse-to-Fine Classification (CFC) 框架，利用 LLM 的零样本推理能力为图节点开放集分类提供语义化 OOD 样本和潜在 OOD 标签空间，实现不仅检测 OOD 还能将其分类到具体未知类别的能力。
tags:
  - "AAAI2026"
  - "医学图像"
  - "open-set classification"
  - "OOD detection"
  - "图神经网络"
  - "large language models"
  - "node classification"
---

# Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models

**会议**: AAAI2026  
**arXiv**: [2512.16244](https://arxiv.org/abs/2512.16244)  
**代码**: [sihuo-design/CFC](https://github.com/sihuo-design/CFC)  
**领域**: 医学图像  
**关键词**: open-set classification, OOD detection, graph neural networks, large language models, node classification

## 一句话总结

提出 Coarse-to-Fine Classification (CFC) 框架，利用 LLM 的零样本推理能力为图节点开放集分类提供语义化 OOD 样本和潜在 OOD 标签空间，实现不仅检测 OOD 还能将其分类到具体未知类别的能力。

## 背景与动机

图神经网络 (GNN) 在闭集场景中表现优秀，但在真实部署中常遇到训练时未见过的类别（即 OOD 样本）。现有的开放集分类方法存在以下关键不足：

1. **依赖合成/辅助 OOD 样本**：需要大量生成样本，计算开销大且无法准确反映真实 OOD 分布
2. **缺乏语义理解**：生成样本或辅助数据不具备真正的语义含义，容易过拟合特定数据集
3. **OOD 子空间过小**：没有语义 OOD 样本导致 OOD 检测的决策边界过于尖锐
4. **无法区分不同未知类**：将所有未知类合并为一个 OOD 类别，在医疗诊断、欺诈检测等高风险场景中严重限制实用性

核心驱动问题：能否在没有 OOD 真实标签的情况下，将 OOD 检测扩展为 OOD 分类？

## 核心问题

本文将传统的开放集分类从 $(C+1)$ 类问题（$C$ 个 ID 类 + 1 个 OOD 类）扩展为 $(C+u)$ 类问题（$C$ 个 ID 类 + $u$ 个 OOD 类），其中 $u$ 在开放集场景中是未知的。需要解决两个关键挑战：

- 如何在没有标注信息的情况下近似 OOD 空间？
- 如何推导出有意义的未知类标签？

## 方法详解

CFC 框架包含三个核心阶段：

### 1. LLM 粗分类器 (Coarse Classifier)

将图数据映射到文本空间，利用 LLM 的专家知识进行 OOD 初检和潜在标签生成。根据 ID 标签空间的覆盖度设计两种检测策略：

- **Easy-Reject**：当 ID 类只覆盖其主类别的小部分时使用（如 Cora, DBLP, WikiCS）。设计置信度感知 prompt，仅在 LLM 高度确信时标记为 OOD，同时生成异常类标签
- **Hard-Reject**：当 ID 类覆盖主类别的大部分时使用（如 Citeseer）。先让 LLM 总结 ID 类的主类别，生成候选 OOD 标签空间，再基于扩展标签空间进行分类

置信度阈值设为 0.7，用于过滤噪声标注。

### 2. GNN 细分类器 (Fine Classifier)

基于粗分类器获得的语义 OOD 样本集 $\mathcal{V}_{\text{ood}}$，构建 $(C+1)$ 类 GNN 分类器：

- **去噪 (Denoising)**：利用标签传播 $\mathbf{Y}^{l(k)} = \mathbf{D}^{-1}\mathbf{A}\mathbf{Y}^{l(k-1)}$ 纠正 LLM 误判的 OOD 样本，每次迭代后重置 ID 训练节点标签，传播 $K$ 轮后丢弃被重新预测为 ID 的 OOD 样本
- **OOD 数据增强**：改进 Manifold Mixup，收集训练集中分类置信度低（靠近决策边界）的 $K$ 个节点，与 OOD 样本中心进行隐层嵌入混合：$\tilde{x}_i = \alpha \boldsymbol{h}_i^k + (1-\alpha)\boldsymbol{h}_c^k$，超参 $\alpha$ 控制生成样本与 OOD 样本的距离
- **联合训练**：在 $\mathcal{V}_{\text{train}} \cup \mathcal{V}_{\text{ood}}^a$ 上用交叉熵损失训练 GCN 分类器

### 3. OOD 分类

对细分类器检测出的 OOD 样本 $\mathcal{V}_{\text{ood}}^f$，利用粗分类阶段生成的潜在 OOD 标签空间：用 TF-IDF 等相似度度量合并相似类别、过滤样本过少的类别，获得后处理 OOD 标签空间；再通过 LLM prompt 对 OOD 样本进行最终分类标注。

### 理论分析

证明了 CFC 通过引入语义 OOD 样本，将 OOD 子空间维度从 $\text{dim}(\mathcal{H}) - (C+1)$ 扩展到 $\text{dim}(\mathcal{H} + \mathcal{H}') - (C+1)$，从而产生更平滑、更平坦的 OOD 检测决策边界。

## 实验关键数据

**数据集**：Cora, Citeseer, WikiCS, DBLP（文本图）；Amazon-Computer, Amazon-Photo（非文本图）。每个数据集设定 $u \geq 2$ 个 OOD 类。

**OOD 检测性能**（两个 OOD 类，overall accuracy）：

| 数据集 | NodeSafe (次优) | CFC | 提升 |
|--------|----------------|-----|------|
| Cora | 85.71% | **90.00%** | +4.3% |
| Citeseer | 72.74% | **77.21%** | +4.5% |
| WikiCS | 79.59% | **80.44%** | +0.9% |
| DBLP | 76.21% | **84.03%** | +7.8% |

**OOD 分类准确率**（使用 GPT-4o + post-OOD label space）：Cora 69.76%、Citeseer 70.30%、WikiCS 57.96%、DBLP 48.45%。

**关键消融**：
- 即使不用去噪和 Mixup (CFC w/o D/M)，仅靠语义 OOD 样本已超越所有基线
- Cora 上 OOD 检测从 GCN_sigmoid 的 0% 提升到 CFC 的 95.74%

## 亮点

1. **问题定义新颖**：首次将图开放集分类从简单的 OOD 检测扩展到 OOD 分类，定义了 $(C+u)$ 类分类问题
2. **语义 OOD 样本**：不靠合成或辅助数据，而是利用 LLM 识别真正语义上属于分布外的样本，可解释性和实用性更强
3. **框架通用性强**：CFC 不仅适用于图数据，还可直接扩展到文本领域
4. **理论支撑充分**：从子空间维度和决策边界平滑性角度证明了语义 OOD 样本的优势
5. **设计精巧**：Easy-Reject/Hard-Reject 的双策略设计考虑了不同 ID 覆盖度场景

## 局限与展望

1. **依赖 LLM 质量**：粗分类阶段严重依赖 GPT-4o 等强 LLM，开源模型（如 Llama2-7b）检测能力明显较弱
2. **文本属性图限制**：非文本图需要额外的特征编码步骤将节点属性转为文本描述，增加了预处理复杂度
3. **OOD 分类准确率有限**：最高约 70%，在高风险场景中可能不够可靠
4. **ID 分类略有下降**：引入 OOD 检测后 ID 准确率有 2-5% 的下降（如 Cora 从 90.64% 降到 87.49%）
5. **计算成本**：需要多次调用 LLM API，实际部署成本不低
6. **OOD 类数未知**：$u$ 的估计完全依赖 LLM 和后处理，缺乏更原则性的自动确定方法

## 与相关工作的对比

| 方法 | 是否需要合成 OOD | 能否 OOD 分类 | 使用 LLM | 图+文本通用 |
|------|:-:|:-:|:-:|:-:|
| G²Pxy | 是（代理未知节点） | 否 | 否 | 否 |
| GNNSafe | 否（能量传播） | 否 | 否 | 否 |
| NodeSafe | 否（能量传播） | 否 | 否 | 否 |
| GOLD | 否（能量传播） | 否 | 否 | 否 |
| **CFC** | **否（语义 OOD）** | **是** | **是** | **是** |

CFC 的核心区别在于利用 LLM 获取语义化的真实 OOD 样本，而非合成样本，且是唯一能够进行多类 OOD 分类的方法。

## 启发与关联

- **LLM 作为开放世界感知器**：LLM 的零样本推理能力可以为传统模型提供分布外的语义信号，这一思路可推广到其他开放世界任务（如开放词汇检测、开放集分割）
- **粗到细策略的通用性**：先用强但有噪声的信号做初筛，再用结构化模型做精细判别的思路，适用于标注预算有限的场景
- **医疗/安全场景的潜力**：在欺诈检测、医疗诊断等需要区分不同未知异常类型的场景中有直接应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次定义图 OOD 分类问题，LLM+GNN 的粗到细框架设计独特
- 实验充分度: ⭐⭐⭐⭐ — 6 个数据集、多种 LLM、完整消融实验和理论分析
- 写作质量: ⭐⭐⭐⭐ — 问题阐述清晰，图示直观，方法描述系统
- 价值: ⭐⭐⭐⭐ — 解决了开放集分类中长期被忽视的 OOD 细分类问题，实用意义强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[CVPR 2026\] LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding](../../CVPR2026/medical_imaging/llada-medv_exploring_large_language_diffusion_models_for_biomedical_image_unders.md)
- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)

</div>

<!-- RELATED:END -->
