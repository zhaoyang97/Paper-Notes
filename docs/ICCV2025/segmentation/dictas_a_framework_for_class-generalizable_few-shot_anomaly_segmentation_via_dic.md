---
title: >-
  [论文解读] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup
description: >-
  [ICCV 2025][语义分割][异常检测] 受人类检查员"查字典"直觉启发，提出 DictAS 框架，将少样本异常分割重新定义为字典查询任务——若查询特征无法从正常样本字典中检索到则判定为异常——通过自监督训练获得类别无关的字典查询能力，在 7 个工业和医学数据集上的 FSAS 性能和推理速度均达到 SOTA。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "异常检测"
  - "少样本异常分割"
  - "字典查询"
  - "CLIP"
  - "自监督学习"
---

# DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup

**会议**: ICCV 2025  
**arXiv**: [2508.13560](https://arxiv.org/abs/2508.13560)  
**代码**: [github.com/xiaozhen228/DictAS](https://github.com/xiaozhen228/DictAS)  
**领域**: 医学图像  
**关键词**: 异常检测, 少样本异常分割, 字典查询, CLIP, 自监督学习

## 一句话总结

受人类检查员"查字典"直觉启发，提出 DictAS 框架，将少样本异常分割重新定义为字典查询任务——若查询特征无法从正常样本字典中检索到则判定为异常——通过自监督训练获得类别无关的字典查询能力，在 7 个工业和医学数据集上的 FSAS 性能和推理速度均达到 SOTA。

## 研究背景与动机

### 问题定义

少样本异常分割（FSAS, Few-Shot Anomaly Segmentation）旨在仅给定少量正常样本的情况下，识别查询图像中的异常区域。这在训练数据稀缺、像素级标注有限的工业缺陷检测和医学图像分析中尤为重要。

FSAS 有两种设置：
- **类别依赖（class-dependent）**：对每个未见类别用其正常样本微调独立模型
- **类别泛化（class-generalizable）**：训练一个统一模型，无需在目标数据上重训练，仅用少量正常样本作为视觉提示即可检测未见类别的异常

本文聚焦更具挑战性的**类别泛化**设置。

### 已有方法的不足

**RegAD**：引入特征配准对齐，但由于需要大量参考图像增强，推理效率低

**FastRecon**：用线性回归做特征重建，但容易出现过度重建（正常和异常特征都被重建好）

**基于 CLIP 的方法**（WinCLIP、APRIL-GAN）：利用 CLIP 的图文对齐能力，通过记忆库提供视觉先验。但这些方法依赖于在辅助训练阶段见过的真实异常样本的"经验知识"，限制了泛化到全新类别的能力

**PromptAD**：虽然效果好，但属于类别依赖方法，需要对每个未见类别微调新模型，不可扩展

### 核心动机

**关键洞察**：即使是新手检查员，也能在只看过几个正常样本的情况下检测未见类别的异常——无需大量先验经验。这一过程类似"查字典"：如果查询区域能在字典中找到对应的正常模式，则为正常；否则为异常。

基于这一直觉，DictAS 将 FSAS 重新定义为字典查询任务，通过自监督学习获得类别无关的查询能力，而不是记忆训练集的正常/异常模式。

## 方法详解

### 整体框架

DictAS 基于 CLIP (ViT-L-14-336) 构建，包含三个核心组件：
1. **Dictionary Construction**：从正常参考图像构建结构化字典
2. **Dictionary Lookup**：通过稀疏查询策略从字典中检索查询区域特征
3. **Query Discrimination Regularization**：增强异常区分能力

### 关键设计

#### 1. **Dictionary Construction（字典构建）**

- **功能**：将正常参考图像的特征组织为结构化字典，包含"索引"（Key）和"内容"（Value），同时生成查询向量（Query）。
- **核心思路**：

  使用三个独立的 AttnBlock（自注意力 Transformer 块）分别生成 Dictionary Query、Key 和 Value：

  $$\mathbf{F}_Q^l = g_Q(\mathbf{F}_q^l) = AttnBlock\_Q(\mathbf{F}_q^l)$$
  $$\mathbf{F}_K^l = g_K(\mathbf{F}_n^l) = AttnBlock\_K(\mathbf{F}_n^l)$$
  $$\mathbf{F}_V^l = g_V(\mathbf{F}_n^l) = \mathbf{F}_n^l + AttnBlock\_V(\mathbf{F}_n^l)$$

  其中 $\mathbf{F}_q^l \in \mathbb{R}^{HW \times C}$ 是查询图像第 $l$ 层特征，$\mathbf{F}_n^l \in \mathbb{R}^{kHW \times C}$ 是 $k$ 张参考图像的拼接特征。Value Generator 加入残差连接以保留细粒度正常特征细节。

  每个 AttnBlock 内部采用多头自注意力 + 两层 MLP：

  $$\mathbf{F}_{out} = TwoLayerMLP(softmax(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C}})\mathbf{V})$$

- **设计动机**：自注意力使每个 patch 能感知全局上下文，提高字典构建的鲁棒性。Key 和 Value 用不同的变换解耦了索引和内容的功能，类似真实字典中"词条"和"释义"的分工。

#### 2. **Dictionary Lookup（字典查询）**

- **功能**：对查询图像的每个 patch 特征，从字典中检索最相关的正常模式。若检索失败（距离大），则判定为异常。
- **核心思路**：

  查询过程分两步——Query-Key 匹配计算相似度 $\mathbf{z} = \mathbf{x}_Q^l \mathbf{F}_K^{lT}$，然后加权融合 Dictionary Value 得到检索结果 $\mathbf{x}_r^l = \hat{\mathbf{w}} \mathbf{F}_V^l$。

  提出三种查询策略：
  - **Maximum Lookup**：取最大相似度的 Value（one-hot）
  - **Dense Lookup**：softmax 加权所有 Value
  - **Sparse Lookup**（默认）：通过 Sparse Probability Module (SPM) 稀疏化权重

  SPM 求解约束优化问题：

  $$\arg\min_{\triangle} \frac{1}{2} \|\mathbf{w} - \mathbf{z}\|^2, \quad \triangle = \{\mathbf{w} | \mathbf{w}_u \geq 0, \sum_{u=1}^{kHW} \mathbf{w}_u = 1\}$$

  解为 $\hat{\mathbf{w}}_u = \max(\mathbf{z}_u - \tau, 0)$，其中 $\tau$ 是动态阈值——通过排序和累积求和自适应确定，自动选择最相关的少数 Value 并抑制冗余。

  **Query Loss**（自监督训练核心）：

  $$\mathcal{L}_q = \sum_l \frac{1}{|\mathcal{N}|} \sum_{j \in \mathcal{N}} d(\mathbf{F}_{q,j}^l, \mathbf{F}_{r,j}^l)$$

  其中 $d$ 为余弦距离，$\mathcal{N}$ 为正常区域的索引集。只对正常区域最小化查询-检索距离，异常区域的距离自然会保持较大。

- **设计动机**：Sparse Lookup 避免了 Dense Lookup 中所有 Value 均匀参与的问题（导致异常区域也能被"平均"出合理检索结果），又比 Maximum Lookup 更灵活。随着参考图像增多，稀疏性帮助抑制冗余，因此 DictAS 在高 shot 下优势更大。

#### 3. **Query Discrimination Regularization（查询判别正则化）**

- **功能**：增强异常区域和正常区域在检索结果上的距离差异，防止模型过度检索导致异常也被"成功检索"。
- **核心思路**：

  **Contrastive Query Constraint (CQC)**：强制异常区域的查询-检索距离大于正常区域：

  $$\mathcal{L}_{CQC} = \sum_l \max(0, \mathbb{E}_\mathcal{N}[d] - \mathbb{E}_\mathcal{A}[d])$$

  **Text Alignment Constraint (TAC)**：利用 CLIP 文本-图像对齐能力，将全局检索结果对齐到"正常"文本嵌入空间：

  $$\mathcal{L}_{TAC} = CE(\tilde{\mathbf{x}}_r \tilde{\mathbf{F}}_{text}^T, 0) + CE(\tilde{\mathbf{x}}_q \tilde{\mathbf{F}}_{text}^T, y_q)$$

  其中文本嵌入采用类似 WinCLIP 的提示模板设计（如 "a photo of a [state] [class]"），$y_q \in \{0,1\}$ 是图像级标签。

- **设计动机**：Query Loss 赋予了模型强大的检索能力，但这是把双刃剑——如果模型在特征提取和检索上都过于强大，可能找到字典中正常特征的组合来匹配任何查询（包括异常）。CQC 在特征空间层面拉开差距，TAC 在语义层面约束检索结果必须是"正常的"。

### 损失函数 / 训练策略

- **总损失**：$\mathcal{L} = \mathcal{L}_q + \lambda_1 \mathcal{L}_{CQC} + \lambda_2 \mathcal{L}_{TAC}$，$\lambda_1 = \lambda_2 = 0.1$
- **自监督训练**：无需像素级标注；使用 DRÆM 的异常合成算法生成查询图像，几何变换生成参考图像
- **辅助数据**：VisA 全部正常图像作为训练集（测试 VisA 时改用 MVTecAD）
- **训练配置**：Adam 优化器，lr=0.0001，batch=24，训练 30 epochs，单卡 RTX 3090
- **训练时参考图像数**：$k=1$（效率考虑），推理时 $k \geq 1$
- **CLIP backbone 冻结**：提取第 6/12/18/24 层特征，backbone 不更新

## 实验关键数据

### 主实验

**4-shot 设置下 7 个数据集像素级指标 (AUROC, PRO, AP)**：

| 数据集 | WinCLIP | APRIL-GAN | PromptAD† | **DictAS** |
|--------|---------|-----------|----------|----------|
| MVTecAD | (92.4, 83.8, 39.2) | (92.2, 86.6, 46.6) | (96.0, 92.4, 57.5) | **(98.6, 95.1, 66.8)** |
| VisA | (96.0, 86.5, 25.7) | (96.2, 86.6, 30.6) | (97.9, 89.5, 37.5) | **(98.8, 91.9, 41.8)** |
| RESC (医学) | (93.1, 75.7, 38.4) | (93.7, 77.6, 57.3) | (96.8, 86.8, 71.3) | **(97.5, 89.7, 74.9)** |
| BraTS (医学) | (93.3, 64.0, 33.4) | (91.3, 63.0, 40.0) | (96.6, 77.0, 54.4) | **(97.3, 77.2, 59.3)** |
| 工业平均 | (94.5, 82.7, 29.3) | (94.7, 84.8, 38.5) | (97.1, 89.6, 47.0) | **(98.4, 92.2, 52.5)** |
| 医学平均 | (93.2, 69.8, 35.9) | (92.5, 70.3, 48.7) | (96.7, 82.2, 62.9) | **(97.4, 83.4, 67.1)** |

†注：PromptAD 是类别依赖方法（每个类别需微调），DictAS 是类别泛化方法（统一模型）。

### 消融实验

**组件消融（MVTecAD, 4-shot, %）**：

| 配置 | AUROC | PRO | AP | 说明 |
|------|-------|-----|-----|------|
| w/o Query Generator | 97.5 | 94.2 | 63.5 | 少了查询变换 |
| w/o Key Generator | 97.9 | 94.5 | 63.8 | 少了索引变换 |
| w/o Value Generator | 98.0 | 94.6 | 64.2 | 少了内容变换 |
| w/o $\mathcal{L}_{CQC}$ | 97.4 | 94.1 | 64.6 | 少了对比约束 |
| w/o $\mathcal{L}_{TAC}$ | 98.0 | 94.6 | 65.0 | 少了文本对齐约束 |
| w/o 两个正则化 | 97.1 | 93.5 | 63.7 | 少了所有正则化 |
| **Full DictAS** | **98.6** | **95.1** | **66.8** | 完整模型 |

**查询策略消融（MVTecAD, AP%）**：

| 策略 | 1-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|---------|
| Maximum Lookup | 52.2 | 59.1 | 59.7 | 60.6 |
| Dense Lookup | 60.2 | 63.7 | 63.6 | 63.8 |
| **Sparse Lookup** | **61.1** | **66.8** | **67.0** | **68.5** |

### 关键发现

1. **Sparse Lookup 在高 shot 时优势更大**：Dense Lookup 从 4-shot 到 16-shot 几乎不提升（63.7→63.8），而 Sparse Lookup 持续提升（66.8→68.5），证实稀疏策略有效抑制了参考图像增多带来的冗余
2. **DictAS 超越类别依赖方法**：统一模型的 DictAS 在所有指标上超过需要逐类微调的 PromptAD
3. **推理速度最快**：73.5ms/图，快于所有对比方法（WinCLIP 8227.5ms, AnomalyGPT 1555.2ms）
4. **稳定性高**：AP 标准差仅 0.4%，低于所有竞争方法
5. **TAC 对细粒度判别更重要**：$\mathcal{L}_{TAC}$ 的消融影响（+2.2% AP）大于 $\mathcal{L}_{CQC}$（+1.8% AP）

## 亮点与洞察

1. **直觉驱动的问题建模**："查字典"的类比非常直觉且自然——将异常检测转化为检索问题，比"重建"或"分类"的视角更本质
2. **自监督训练范式**：不需要任何真实异常样本和像素标注，仅用正常图像+异常合成即可训练，大幅降低了数据需求
3. **稀疏查询的精妙设计**：SPM 通过凸优化自适应确定阈值，在保持检索准确性的同时天然抑制冗余
4. **双重正则化的必要性**：t-SNE 可视化清晰展示了正则化前后残差特征的可分性差异

## 局限与展望

1. **依赖 CLIP backbone**：对 CLIP 预训练质量有隐式依赖，尽管消融显示对 backbone 选择不太敏感
2. **异常合成算法的影响**：使用了 DRÆM 的异常合成策略，合成异常的多样性和真实性可能限制了正则化的效果
3. **仅用 FSAS 场景评估**：未探索在有充足数据时是否仍有优势（如全 shot 设置）
4. **文本提示设计较简单**：TAC 使用的文本模板依赖类别名称，在工业场景中可能需要更精细的提示工程

## 相关工作与启发

- 与 WinCLIP 的区别：WinCLIP 利用 CLIP 的图文对齐和记忆库做异常检测，但依赖预定义的文本提示和静态视觉先验。DictAS 则通过学习动态自适应的检索权重实现灵活的字典查询
- 与 AnomalyGPT 的区别：后者引入 LLM 实现多轮对话交互，但推理开销巨大（1555ms vs 73.5ms）；DictAS 专注于高效的端到端检测
- 启发：字典查询范式可推广到其他检测任务——任何"正常模式有限、异常模式无限"的场景都可以用这种思路

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 字典查询的类比虽直觉但设计精巧，Sparse Lookup + 双重正则化有效
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7 个数据集（工业+医学）、5 种 shot 设置、详尽的消融（组件/策略/backbone/分辨率）、t-SNE 可视化
- **写作质量**: ⭐⭐⭐⭐ — 字典类比贯穿全文，逻辑清晰
- **价值**: ⭐⭐⭐⭐ — 类别泛化 FSAS 的实用解决方案，推理速度和性能的优秀平衡

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](object-level_correlation_for_few-shot_segmentation.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
