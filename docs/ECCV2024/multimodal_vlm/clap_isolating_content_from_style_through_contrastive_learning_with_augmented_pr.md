---
title: >-
  [论文解读] CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts
description: >-
  [ECCV2024][多模态VLM][CLIP] 从因果生成模型视角出发，提出 CLAP（Contrastive Learning with Augmented Prompts），通过文本 prompt 增强 + 对比学习训练一个轻量解耦网络，将 CLIP 预训练特征中的 content 与 style 分离，仅用文本训练即可同时提升图像和文本两侧的表征质量，在 zero-shot、few-shot 分类和对抗鲁棒性上均取得一致提升。
tags:
  - "ECCV2024"
  - "多模态VLM"
  - "CLIP"
  - "内容-风格解耦"
  - "对比学习"
  - "文本增强"
  - "因果表征学习"
---

# CLAP: Isolating Content from Style through Contrastive Learning with Augmented Prompts

**会议**: ECCV2024  
**arXiv**: [2311.16445](https://arxiv.org/abs/2311.16445)  
**代码**: [YichaoCai1/CLAP](https://github.com/YichaoCai1/CLAP)  
**领域**: 多模态VLM  
**关键词**: CLIP, 内容-风格解耦, 对比学习, 文本增强, 因果表征学习

## 一句话总结

从因果生成模型视角出发，提出 CLAP（Contrastive Learning with Augmented Prompts），通过文本 prompt 增强 + 对比学习训练一个轻量解耦网络，将 CLIP 预训练特征中的 content 与 style 分离，仅用文本训练即可同时提升图像和文本两侧的表征质量，在 zero-shot、few-shot 分类和对抗鲁棒性上均取得一致提升。

## 研究背景与动机

**现状**：CLIP 等对比视觉-语言模型通过大规模图文对比预训练获得了强大的泛化能力，广泛用于 zero-shot 分类、prompt learning 等下游任务。

**痛点**：CLIP 学到的特征将 content（内容，如类别语义）和 style（风格，如图像纹理、光照、prompt 措辞）混合在一起。这导致三个关键问题：

**对 prompt 敏感**：不同文本 prompt（如 "a photo of a [class]" vs "[class]"）产生的 zero-shot 性能差异显著

**few-shot 性能受限**：少样本场景下分布偏移使风格信息干扰分类

**对抗脆弱**：对抗攻击本质上改变 style 信息，混合特征容易被攻击误导

**核心矛盾**：理论上（von Kügelgen et al., NeurIPS 2021），通过对所有 style 变量施加 soft intervention 并用对比学习即可实现 content-style 解耦，但实际中图像增强难以充分改变所有 style 因子——例如把照片变成素描在图像空间极难实现，文本空间却只需改几个词。

**切入角度**：视觉和语言数据共享同一隐空间（content c + style s），因此可以在文本模态上做增强来训练解耦网络，然后直接迁移到视觉模态使用。文本天然具有高语义性和逻辑结构，对 style 属性的操控比图像精确得多。

**核心 idea**：用 template-based prompt 增强（删属性、换顺序、插噪声）构造 style 变化的文本对，通过 InfoNCE 对比学习训练一个轻量 MLP 解耦网络，嫁接在冻结的 CLIP 编码器之上，提取纯 content 特征。

## 方法详解

### 整体框架

CLAP 的流程分为三步：

1. **训练阶段**：在冻结的 CLIP 文本编码器之上，用 template-based prompt 和对应增强 prompt 对训练一个解耦网络 $f_c$
2. **迁移阶段**：由于视觉和语言共享隐空间，训练好的 $f_c$ 直接迁移使用
3. **推理阶段**：对图像 $x$，特征为 $f_c^* \circ f_x^*(x)$；对文本 $t$，特征为 $f_c^* \circ f_t^*(t)$，用 cosine similarity 做 zero-shot 分类

### 关键设计 1：因果生成模型（理论基础）

- **功能**：建立视觉-语言数据的因果生成模型，将隐空间分为 content $c$ 和 style $s$
- **核心思路**：$s := g_s(c)$, $x := g_x(c,s)$, $t := g_t(c,s)$, $y := g_y(c)$。标签 $y$ 仅由 content 决定，图像和文本由共享的 $(c,s)$ 通过不同生成过程产生
- **设计动机**：提供理论保证——当所有 style 变量被 soft intervention 改变时，对比学习可以 block-identify content 变量。这为用数据增强解耦 content 提供了正当性

### 关键设计 2：文本 Prompt 增强策略

- **功能**：设计针对 template prompt 的增强方法，在不改变 content 的前提下最大化 style 变化
- **核心思路**：基于结构化 prompt "a [art style] [image type] of a [object size] [object color] [class]"，设计五种增强操作：
    - **OSD** (Object Size Deletion)：删除尺寸描述
    - **OCD** (Object Color Deletion)：删除颜色描述
    - **ITD** (Image Type Deletion)：删除图像类型
    - **ASD** (Art Style Deletion)：删除艺术风格
    - **SPO** (Swapping Prompt Order)：交换 prompt 各部分顺序
    - **IGN** (Inserting Gaussian Noise)：在 tokenized prompt 中插入高斯噪声（均值 0，标准差 0.02）
- **设计动机**：与图像 random masking 类似，但文本增强能精确删除某个 style 属性而不损害 content。例如删除 "realistic" 只改变风格，不影响类别信息 "car"。而图像 masking 可能同时破坏 content 和 style

### 关键设计 3：解耦网络结构（Residual MLP + Zero Init）

- **功能**：设计一个轻量网络附加在 CLIP 编码器之上，从混合特征中提取 content
- **核心思路**：采用 residual MLP 结构，主分支包括一个正态初始化的 Linear → SiLU → 一个 **零初始化**（无 bias）的 Linear。shortcut 分支保留原始输入特征。推理时引入权重系数 $\alpha$ 控制主分支输出与输入特征的融合比例
- **设计动机**：零初始化借鉴自 ControlNet 的 zero-conv 思想，确保训练初始阶段网络输出等于输入特征（从 CLIP 预训练空间开始优化），避免从随机起点开始破坏已有表征

### 损失函数 / 训练策略

CLAP 的训练目标为双项 InfoNCE 损失：

$$f_c^* = \arg\min_{f_c} \mathbb{E}_{\{t_i\} \in \mathcal{D}_t} \left[ \mathcal{L}(f_c \circ f_t^*; \{t_i, \tilde{t}_i\}, \tau) + \lambda \mathcal{L}(f_c \circ f_t^*; \{t_i^c, \tilde{t}_i\}, 1) \right]$$

- **第一项**：原始 prompt $t_i$ 与增强 prompt $\tilde{t}_i$ 构成正对，其他样本的增强 prompt 为负对，温度 $\tau$
- **第二项**：用 class name $t_i^c$ 和增强 prompt $\tilde{t}_i$ 构成正对，增大 prompt pair 之间的变化幅度，权重 $\lambda$，温度固定为 1
- 训练数据完全是合成的 template prompt（每类 480 个，由 10 颜色 × 3 尺寸 × 8 图像类型 × 2 风格组合），无需真实图像
- 使用 Adam 优化器，lr=0.0001，最多 8000 步，early stopping（5 个 checkpoint 无改善）

## 实验关键数据

### 主实验

评估在 PACS、VLCS、OfficeHome、DomainNet 四个多域数据集上进行，使用 ViT-B/16 CLIP。

**Zero-shot 结果（平均 top-1 acc %）**：

| Prompt | Method | PACS | VLCS | OfficeHome | DomainNet | Overall |
|--------|--------|------|------|------------|-----------|---------|
| ZS(C) "[class]" | CLIP | 95.7 | 76.4 | 79.8 | 57.8 | 77.4 |
| | Im.Aug | 96.5 | 79.5 | 77.0 | 51.5 | 76.1 |
| | **CLAP** | **97.2** | **82.6** | **81.0** | **58.7** | **79.9** |
| ZS(PC) "a photo of a [class]" | CLIP | 96.1 | 82.4 | 82.5 | 57.7 | 79.7 |
| | **CLAP** | **97.2** | **83.4** | **83.0** | **59.0** | **80.6** |
| ZS(NC) "[noise][class]" | CLIP | 90.8 | 68.3 | 71.5 | 51.0 | 70.4 |
| | **CLAP** | **97.2** | **81.0** | **73.5** | **52.6** | **76.1** |

CLAP 在所有 prompt 形式和所有数据集上均超越 CLIP 和 Im.Aug。特别是对噪声 prompt ZS(NC)，CLAP 的 overall 提升达 +5.7%（70.4→76.1）。

**Few-shot 结果**：1-shot 场景下 CLAP 分别超过 CLIP 线性探测 +10%（PACS）、+3.5%（VLCS）、+2.5%（OfficeHome）、+1.5%（DomainNet）。

**对抗鲁棒性（avg top-1 acc % under attacks）**：

| Setting | Method | FGSM Avg | PGD-20 Avg | CW-20 Avg | Overall |
|---------|--------|----------|------------|-----------|---------|
| ZS(C) | CLIP | 58.2 | 13.0 | 11.0 | 29.2 |
| | Im.Aug | 62.7 | 13.2 | 11.0 | 31.1 |
| | **CLAP** | **65.8** | **14.0** | **12.1** | **32.7** |
| 1-shot | CLIP | 42.2 | 16.9 | 6.8 | 23.7 |
| | **CLAP** | **50.7** | **28.6** | **9.2** | **31.9** |

### 消融实验

**Prompt 增强组合消融（VLCS 数据集）**：

| 增强组合 | ZS(Avg.) ↑ | R ↓ | δ ↓ | Δ(NC) ↓ |
|---------|-----------|-----|-----|---------|
| CLIP baseline | 77.3 | 6.1 | 2.8 | 8.1 |
| EDA | 81.6 | 1.9 | 0.9 | 2.3 |
| OSD+OCD+ITD+ASD+SPO | 82.0 | 1.2 | 0.6 | 1.7 |
| OSD+OCD+ITD+ASD | 80.1 | 2.5 | 1.2 | 3.0 |
| **OSD+OCD+SPO+IGN** | **82.6** | **0.8** | **0.4** | **1.6** |

最优组合为 OSD+OCD+SPO+IGN，达到最高平均准确率和最低方差。

**Prompt 来源消融（VLCS 数据集）**：

| 来源 | ZS(Avg.) ↑ | R ↓ | δ ↓ |
|------|-----------|-----|-----|
| CLIP baseline | 77.3 | 6.1 | 2.8 |
| LLM (ChatGPT-3.5) | 78.2 | 3.2 | 1.5 |
| Random | 81.6 | 0.7 | 0.3 |
| PromptStyler | 81.2 | 2.7 | 1.2 |
| Template (ours) | 81.6 | 1.9 | 0.9 |

结论：更简单的 prompt 形式（Random、Template）反而表现更好。

### 关键发现

1. **文本增强远优于图像增强**：CLAP 在所有指标上一致超过 Im.Aug，且 Im.Aug 在部分数据集（如 DomainNet）上反而低于 CLIP baseline，说明图像增强无法充分改变所有 style 因子
2. **Prompt 鲁棒性显著提升**：CLAP 将 prompt 间的性能波动（R）从 2.7% 降至 1.1%，噪声 prompt 掉点（Δ(NC)）从 7.0% 降至 3.8%
3. **训练效率高**：CLAP 仅需文本数据训练，在 PACS/VLCS 上约 11 分钟完成，DomainNet 约 47 分钟（Im.Aug 需 3.3 小时）
4. **跨模型泛化**：在 ViT-L/14 和 ResNet50x16 上重复实验，CLAP 同样一致提升 zero-shot 性能并减少方差
5. **t-SNE 可视化**：CLAP 的表征展现出更清晰的类间分离和更紧凑的类内聚类

## 亮点与洞察

1. **因果视角连接理论与实践**：将 von Kügelgen et al. 的因果解耦理论首次实际应用到 CLIP-like 模型上，建立了从 SCM 到工程方法的完整链路
2. **跨模态迁移的巧妙利用**：在文本模态训练解耦网络，直接迁移到视觉模态使用——利用了 CLIP 对齐空间的本质特性，极大降低了训练成本（无需真实图像）
3. **文本增强 > 图像增强的深刻洞察**：文本数据的逻辑结构性使其比高维图像更适合做 property-wise 的 style 干预，这一观察具有广泛启发意义
4. **Zero-init residual 设计思想**：从 ControlNet 的 zero-conv 借鉴到 MLP 场景，保证微调从预训练空间开始，避免灾难性遗忘
5. **方法极其轻量**：解耦网络仅为两层 MLP，不增加推理延迟，不需修改 CLIP 编码器权重

## 局限与展望

1. **依赖 template prompt 结构**：增强策略（OSD、OCD 等）是针对特定 prompt 模板设计的，泛化到自由文本描述可能需要新的增强策略
2. **合成数据的类别覆盖**：训练 prompt 需要预先知道类别名称列表，对开放词汇场景的适用性有待验证
3. **Style 变量的完全改变假设**：理论要求所有 style 因子都改变才能保证 block identifiability，但实际增强未必能完全满足
4. **仅评估了分类任务**：未在 retrieval、detection、segmentation 等更广泛的 vision-language 任务上验证
5. **超参数需按数据集调节**：$\alpha$、$\tau$、$\lambda$、latent dim 等超参对不同数据集需要不同设定

## 相关工作与启发

- **因果表征学习方向**：von Kügelgen et al. (NeurIPS 2021) 的 content-style 理论是本文直接基础；后续可结合 identifiable causal model (Liu et al., ICLR 2024) 做更强理论保证
- **Prompt Learning**：CoOp、CoCoOp、MaPLe 等方法通过学习 prompt 来适配下游任务，但不改变 CLIP 表征本身。CLAP 提供了一个正交的改进方向——先把特征解耦，再做 prompt learning
- **PromptStyler**：同样利用 style 多样性增强，但目标是 source-free domain generalization，CLAP 则聚焦于通用表征改善
- **ControlNet 的 zero-init**：zero initialization 的设计思想可以推广到更多基于预训练模型微调的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 从因果理论出发提出文本增强替代图像增强的思路新颖，跨模态解耦网络迁移的 idea 有创意
- **技术质量**: ⭐⭐⭐⭐ — 理论动机清晰，方法设计合理，实验对比充分（4 数据集 × 多种评估设定）
- **实验充分度**: ⭐⭐⭐⭐ — 消融全面（增强组合、prompt 来源、超参、模型规模），但缺少更多任务类型的验证
- **实用价值**: ⭐⭐⭐⭐ — 轻量即插即用，训练仅需文本、不需图像，极具工程友好性
- **写作质量**: ⭐⭐⭐⭐ — 因果模型→理论动机→方法→实验的叙事清晰，图表质量高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](x-former_unifying_contrastive_and_reconstruction_learning_for_mllms.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[ECCV 2024\] ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling](artvlm_attribute_recognition_through_vision-based_prefix_language_modeling.md)

</div>

<!-- RELATED:END -->
