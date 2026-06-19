---
title: >-
  [论文解读] Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation
description: >-
  [ECCV 2024][语义分割][weakly-supervised] 首次提出完全使用网络图像（而非精心设计的数据集图像）进行弱监督增量语义分割，通过傅里叶域判别器筛选网络图像 + caption 驱动的 rehearsal 策略保持旧类知识，在 PASCAL VOC 15-5 设定下达到 73.4% mIoU。
tags:
  - "ECCV 2024"
  - "语义分割"
  - "weakly-supervised"
  - "incremental learning"
  - "web images"
  - "视觉语言"
  - "catastrophic forgetting"
---

# Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.13363](https://arxiv.org/abs/2407.13363)  
**代码**: [https://github.com/dota-109/Web-WILSS](https://github.com/dota-109/Web-WILSS)  
**领域**: 语义分割 / 增量学习  
**关键词**: weakly-supervised, incremental learning, web images, vision-language model, catastrophic forgetting

## 一句话总结
首次提出完全使用网络图像（而非精心设计的数据集图像）进行弱监督增量语义分割，通过傅里叶域判别器筛选网络图像 + caption 驱动的 rehearsal 策略保持旧类知识，在 PASCAL VOC 15-5 设定下达到 73.4% mIoU。

## 研究背景与动机
**领域现状**：语义分割的类增量学习（CILSS）允许模型逐步学习新类别，但传统方法依赖昂贵的像素级标注。弱监督增量学习（WILSS）将增量步骤的标注降低为图像级标签，但仍要求使用精心策划的数据集图像。

**现有痛点**：(1) WILSON、FMWISS 等 WILSS 方法仍需要从目标数据集中获取训练图像；(2) 现有方法在多步单类增量（如 15-1）设定下性能严重退化；(3) 不支持仅使用单类新数据的增量步骤，需要负样本。

**核心矛盾**：实际场景中，预训练模型需要适应新类时，目标域数据可能极度有限（如隐私限制），此时从网络获取补充数据是必然选择。但网络数据面临两个关键挑战：(1) 分布与训练数据不同；(2) 图像级标签噪声严重（用类名搜索的图片可能包含多个类或不含目标类）。

**本文目标**：在增量步骤中完全使用网络图像（而非数据集图像）学习新类，同时也用网络图像做旧类 rehearsal 防止灾难性遗忘。

**切入角度**：(1) 用傅里叶域特征做域判别器筛选与训练数据分布接近的网络图像；(2) 用 caption 模型替代简单的类名标签提供多类监督；(3) 保存 caption 而非图像来查询旧类 rehearsal 图像，解决隐私和存储问题。

**核心 idea**：傅里叶域判别器 + caption 驱动的网络图像选择和 rehearsal ，实现无需原始数据集的弱监督增量分割。

## 方法详解

### 整体框架
基于 WILSON 框架（共享编码器 $E^t$ + 分割解码器 $D^t$ + 定位器 $L^t$），本文在增量步骤中将数据源从数据集图像替换为网络图像。两条并行的数据管线：(1) 新类学习管线：用类名从网络搜索图像 → 傅里叶域判别器筛选 → caption 模型提供多标签监督；(2) 旧类保持管线：用之前保存的 caption 从网络搜索图像 → caption 重生成+语义过滤 → 伪标签训练。

### 关键设计
1. **傅里叶域判别器 (Fourier Domain-based Discriminator)**

    - 功能：筛选与原始训练数据分布相似的网络图像
    - 核心思路：在初始步骤（$t=0$）训练一个 EfficientNet-B0 判别器 $M_D$，输入为图像的傅里叶变换幅度谱 $(p_{ds}, p_{web}) = M_D(|\mathcal{F}(\mathbf{x})|)$，其中 $p_{ds}$ 为属于原始数据集的概率。仅当 $p_{ds}/p_{web} > 1$ 时保留该网络图像
    - 设计动机：傅里叶域的幅度谱在不同类别间的统计特性更一致（主要反映风格/纹理而非语义），因此在初始步骤训练的判别器可以在后续步骤（新类未见）中仍然有效。相比像素域判别，傅里叶域对类别变化更鲁棒

2. **Caption 标注 (Caption Labeling)**

    - 功能：用视觉语言模型为网络图像提供多类图像级标签，替代简单的搜索关键词标签
    - 核心思路：用 OpenFlamingo 模型生成图像描述 $w = M_{CAP}(\mathbf{x})$，然后将描述中的名词与预定义的类名词汇表 $\mathcal{W}^c$（含同义词、复数等）匹配：若 $\exists w_i \in w : w_i \in \mathcal{W}^c$ 则 $y^c = 1$
    - 设计动机：用类名搜索的网络图像可能包含多个类（如搜索"boat"得到"一个人站在船上"的图片），简单的单标签监督会导致错误；caption 可以同时识别 person 和 boat，提供正确的多标签。还能识别并丢弃不含目标类的图片

3. **Caption 驱动的 Rehearsal 查询 (Caption-based Querying)**

    - 功能：存储旧图像的 caption 而非图像本身，用 caption 从网络搜索相似图像做 rehearsal
    - 核心思路：初始步骤中为所有训练图像生成 caption 并保存。增量步骤中用这些 caption 作为搜索查询从网络下载图像：$\mathcal{X}_r^{web} = \{\mathbf{x} = \mathcal{D}^{web}(q') | q' = M_{CAP}(\mathbf{x}) : \mathbf{x} \in \mathcal{X}\}$
    - 设计动机：(1) 存储 caption 比存储图像节省存储且避免隐私问题；(2) 用 caption 搜索的图像包含更丰富的语义内容（多个类共现），比仅用类名搜索的图像更接近原始分布

4. **Caption 驱动的 Rehearsal 过滤 (Caption-based Filtering)**

    - 功能：验证下载的 rehearsal 图像是否包含原始图像的核心语义内容
    - 核心思路：对下载图像重新生成 caption $q''$，提取两个 caption 的前两个名词 $(n_1', n_2')$ 和 $(n_1'', n_2'')$（用 Penn TreeBank 句法分析），利用 WordNet 提取各名词的上位词构建向量描述符 $v$，计算余弦相似度。若任一名词对的相似度超过阈值 $T=0.6$ 则保留
    - 设计动机：基于 caption 搜索不保证内容匹配，需要二次验证。使用 WordNet 语义层级而非精确匹配，允许同义/上下位词的灵活匹配（如 dog/animal）

### 损失函数 / 训练策略
- **损失函数**：$\mathcal{L} = \mathcal{L}_{SEG} + \mathcal{L}_{CLS} + \mathcal{L}_{KDE} + \mathcal{L}_{KDL}$
    - $\mathcal{L}_{SEG}$：像素级分割损失（伪标签监督）
    - $\mathcal{L}_{CLS}$：图像级分类损失（多标签软间隔损失）
    - $\mathcal{L}_{KDE}$：编码器特征蒸馏损失（$E^t$ 与 $E^{t-1}$ 的 MSE）
    - $\mathcal{L}_{KDL}$：定位器与旧模型的一致性损失
- **伪标签生成**：合并定位器预测、旧模型预测，新类用定位器、旧类用旧模型
- **网络设置**：DeeplabV3 + ResNet-101（VOC）/ Wide-ResNet-38（COCO）；SGD，初始步骤 30 epochs，增量步骤 40 epochs
- **网络数据**：每类下载 10K 候选 → 筛选 500 用于训练；rehearsal 每个 caption 下载 20 张，保留 100 张

## 实验关键数据

### 主实验（PASCAL VOC 单步多类设定）

| 方法 | 训练数据 | Rehearsal | 15-5 Disjoint All | 15-5 Overlap All | 10-10 Disjoint All | 10-10 Overlap All |
|------|---------|-----------|-------------------|------------------|--------------------|--------------------|
| WILSON | VOC | - | 67.3 | 67.2 | 60.8 | 65.0 |
| RaSP | VOC | - | - | 70.0 | - | 65.9 |
| FMWISS | VOC | VOC(50) | 70.7 | 73.3 | 64.6 | 69.1 |
| **Ours** | **VOC** | **WEB(100)** | **71.1** | **73.3** | **61.7** | **65.7** |
| **Ours** | **VOC** | **WEB(500)** | **72.0** | **73.4** | **61.0** | **65.3** |
| WILSON | WEB | - | 68.9 | 67.8 | 58.6 | 62.1 |
| **Ours** | **WEB** | **WEB(100)** | **70.5** | **71.7** | **60.4** | **65.3** |

### 消融实验

| 配置 | 15-5 Overlap All | 说明 |
|------|------------------|------|
| Baseline WILSON (WEB) | 67.8 | 网络图像无筛选无 caption |
| + 傅里叶域判别器 | 68.4 | +0.6，域筛选有效 |
| + Caption 标注 | 69.5 | +1.7，多标签监督关键 |
| + Caption Rehearsal | **71.7** | +3.9，旧类保持效果显著 |

### 关键发现
- 完全使用网络图像（训练+rehearsal 都来自网络）可以达到接近使用原始数据集的性能（71.7 vs 73.4 在 15-5 overlap），证明了网络数据的可行性
- Caption 标注的贡献最大（+1.7），说明网络图像的多类共现问题是核心挑战
- 傅里叶域判别器在仅初始步骤训练后可以推广到新类的域筛选
- 更多 rehearsal 图像（500 vs 100）在使用原始数据集时有帮助，但在纯网络数据设定下反而略差——网络数据质量不稳定，多不如精
- Caption 查询的 rehearsal 图像比类名查询包含更丰富的语义上下文

## 亮点与洞察
- **首个完全基于网络的 WILSS 框架**：将增量学习的数据需求从精心策划的数据集降低到仅需类名，极大拓展了实际应用范围（如隐私敏感场景、新域适配）。Web+Web 设定下仍能达到合理性能是实用价值的关键证明。
- **Caption 作为轻量级记忆载体**：用 caption 替代图像存储做 rehearsal 是巧妙的设计——存储成本几乎为零、无隐私风险，通过搜索+过滤重建出包含正确语义上下文的多样化图像。
- **傅里叶域的跨类泛化**：利用幅度谱的风格/纹理统计特性（不依赖语义内容）实现跨类别的域判别，初始步骤训练一次即可终身使用。

## 局限与展望
- 网络图像的质量和多样性受搜索引擎限制，不同引擎/不同语言查询可能差异很大
- Caption 模型（OpenFlamingo）的描述质量直接影响标注和查询效果，错误 caption 会引入噪声
- 傅里叶域判别器对极端分布偏移（如从自然场景到医学图像）可能失效
- 多步单类增量（15-1 设定）的性能仍有较大下降，长步骤序列下累积误差问题待解决
- 目前仅验证了 PASCAL VOC 和 COCO，更多样的数据集评估尚缺

## 相关工作与启发
- **vs WILSON**: WILSON 是本文的基础框架，但需要数据集图像；本文用网络图像替代后在 15-5 设定下从 67.3 提升到 71.1-72.0（有 VOC）、从 68.9 提升到 70.5（纯 WEB）
- **vs FMWISS**: FMWISS 依赖 DINO+MaskCLIP 提供像素级伪监督且需要 VOC 图像做 rehearsal；本文仅用图像级标签和网络图像，在 15-5 disjoint 上超越（72.0 vs 70.7）
- **vs RECALL**: RECALL 是首个在 CILSS 中使用网络数据的方法，但仅用于旧类 rehearsal 且使用像素级伪标签；本文同时用于新类学习和旧类保持，且纯弱监督

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次提出完全基于网络的WILSS设定，傅里叶域判别器和caption rehearsal都有新意
- 实验充分度: ⭐⭐⭐⭐ VOC和COCO双数据集，多种增量设定（15-5/10-10/15-1），消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法流程图直观
- 价值: ⭐⭐⭐⭐ 降低增量分割的数据门槛有实际意义，网络数据利用范式值得关注

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [\[ECCV 2024\] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)
- [\[ECCV 2024\] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](../../CVPR2026/segmentation/frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] From Softmax to Dirichlet: Evidential Learning for Semi-supervised Semantic Segmentation](../../CVPR2026/segmentation/from_softmax_to_dirichlet_evidential_learning_for_semi-supervised_semantic_segme.md)

</div>

<!-- RELATED:END -->
