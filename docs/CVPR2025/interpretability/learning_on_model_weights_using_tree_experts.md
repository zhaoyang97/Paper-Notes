---
title: >-
  [论文解读] Learning on Model Weights using Tree Experts
description: >-
  [CVPR 2025][可解释性][权重空间学习] 发现公开模型大多属于少数 Model Tree（从共同祖先微调而来），在同一 Tree 内学习权重远比跨 Tree 简单；提出 ProbeX——首个针对单隐藏层权重的轻量 probing 方法，通过 Tucker 张量分解实现参数量 30 倍压缩，并首次实现了将模型权重与文本表示对齐的零样本模型分类（89.8% 准确率）。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "权重空间学习"
  - "Model Tree"
  - "Probing Expert"
  - "零样本模型分类"
  - "模型仓库搜索"
---

# Learning on Model Weights using Tree Experts

**会议**: CVPR 2025  
**arXiv**: [2410.13569](https://arxiv.org/abs/2410.13569)  
**代码**: [https://horwitz.ai/probex/](https://horwitz.ai/probex/)  
**领域**: 可解释性  
**关键词**: 权重空间学习, Model Tree, Probing Expert, 零样本模型分类, 模型仓库搜索

## 一句话总结
发现公开模型大多属于少数 Model Tree（从共同祖先微调而来），在同一 Tree 内学习权重远比跨 Tree 简单；提出 ProbeX——首个针对单隐藏层权重的轻量 probing 方法，通过 Tucker 张量分解实现参数量 30 倍压缩，并首次实现了将模型权重与文本表示对齐的零样本模型分类（89.8% 准确率）。

## 研究背景与动机

1. **领域现状**：Hugging Face 上公开模型已超百万个，但大多数缺乏充分文档，用户难以判断哪个模型适合自己的任务。权重空间学习（metanetwork）试图从模型权重直接推断模型功能。
2. **现有痛点**：模型权重受优化过程中大量 nuisance 因素影响（如神经元排列、初始化权重），使得跨模型学习非常困难。现有方法（如 Neural Graphs）要么不能扩展到大模型，要么准确率接近随机。
3. **核心矛盾**：不同初始化的模型权重差异巨大，即使训练数据相同，权重空间中的语义信息也被噪声淹没。但实际上大多数公开模型并非随机初始化——它们通常从少数基础模型（如 Llama3、DINO）微调而来，形成 Model Tree 结构。
4. **本文目标** 如何利用 Model Tree 结构简化权重空间学习，并扩展到大模型？
5. **切入角度**：同一 Model Tree 内的模型共享初始化权重，nuisance 变异大幅减少。因此可以为每个 Tree 训练独立的轻量专家（MoE 架构），而非一个通用 metanetwork。
6. **核心 idea**：用 Model Tree 分组减少 nuisance + Tucker 分解设计轻量 ProbeX 架构 = 首次实现大规模模型权重到语言的对齐。

## 方法详解

### 整体框架
输入是一个神经网络的单层权重矩阵 $X \in \mathbb{R}^{d_W \times d_H}$，输出是任务预测（训练类别预测或与文本表示对齐的 embedding）。系统包含两个阶段：(1) 通过层级聚类将模型路由到对应的 Model Tree；(2) 在每个 Tree 内使用独立的 ProbeX 专家进行预测。多个 Tree 通过 MoE（Mixture of Experts）组合。

### 关键设计

1. **Model Tree 感知的 MoE 路由**:

    - 功能：自动将输入模型归类到正确的 Model Tree，激活对应专家。
    - 核心思路：使用层级聚类对训练集中的模型权重聚类，计算每个聚类中心 $\hat{X}_k$。推理时通过最近邻分配：$R(X) = \arg\min_k \|X - \hat{X}_k\|_2$。实验中路由准确率为 100%。
    - 设计动机：动机实验证明跨 Tree 学习存在负迁移——增加其他 Tree 的数据反而降低单个 Tree 上的性能。MoE 解耦各 Tree 避免干扰。

2. **ProbeX 单层 Probing 架构**:

    - 功能：用极少参数从单隐藏层权重矩阵提取有意义的表示。
    - 核心思路：设计可学习的 probe 向量 $u_1, ..., u_{r_U}$，将其传过权重矩阵 $X$ 得到响应 $z_l = X^T u_l$。然后通过共享降维矩阵 $V$ 和每个 probe 独有的编码矩阵 $M_l$ 编码：$e_l = M_l \sigma(V^T z_l)$。汇总所有 probe 编码 $e = \sum_l e_l$，最后通过预测头 $T$ 映射到输出 $y = Te$。理论上证明线性 ProbeX 与 dense expert 在 Tucker 分解假设下表达力等价（Proposition 2），但参数量减少约 30 倍。
    - 设计动机：Dense expert 需要 $d_H \times d_W \times d_Y$ 参数（可达数十亿），不可行。ProbeX 通过 probe 探测 + 矩阵分解将参数量从数亿降到 230 万，训练时间从数小时降到 10 分钟。

3. **权重-语言表示对齐**:

    - 功能：将模型权重映射到与 CLIP 文本 embedding 共享的空间，实现零样本模型分类。
    - 核心思路：以 Stable Diffusion 微调模型为例，训练 ProbeX 将模型编码 $e$ 与其训练数据类别的 CLIP 文本 embedding 对齐，使用类似 CLIP 的对比损失——正确配对的余弦相似度最高。推理时，对该模型计算 ProbeX 编码，与所有候选类别的文本 embedding 做余弦距离，选最近的。
    - 设计动机：这是权重空间学习领域首次实现零样本能力。SD 模型的 cross-attention 层天然包含文本相关信息，有利于对齐。

### 损失函数 / 训练策略
- 分类任务：100 个二元分类头的交叉熵损失（预测 CIFAR100 中哪 50 个类被用于训练）。
- 对齐任务：类似 CLIP 的对比损失，最大化正确配对的余弦相似度。
- 所有参数（$V, u_l, M_l, T$）端到端训练。单层 ProbeX 在单 GPU（10GB VRAM）上训练不到 10 分钟。

## 实验关键数据

### 主实验

**训练数据集类别预测（判别式模型，CIFAR100 50/100 类）：**

| Model Tree | Dense Expert Acc | Dense #Params | ProbeX Acc | ProbeX #Params |
|------------|-----------------|---------------|------------|----------------|
| ResNet | 0.713 | 105M (×45) | **0.842** | 2.3M |
| DINO | 0.614 | 59M (×25) | **0.705** | 2.3M |
| MAE | 0.666 | 59M (×25) | **0.765** | 2.3M |
| Sup. ViT | 0.663 | 59M (×25) | **0.885** | 2.3M |

**零样本模型分类（SD 生成式模型）：**

| 数据集 | Dense In-dist | Dense Zero-shot | ProbeX In-dist | ProbeX Zero-shot |
|--------|--------------|-----------------|----------------|------------------|
| SD_200 | 0.801 | 0.706 | **0.973** | **0.898** |
| SD_1k | 0.382 | 0.343 | **0.296** | **0.505** |

### 消融实验

| 配置 | In-dist Acc | Zero-shot Acc | 说明 |
|------|------------|---------------|------|
| ProbeX (无 ReLU) | 0.953 | 0.564 | 线性版本分类尚可但泛化差 |
| ProbeX (有 ReLU) | **0.973** | **0.898** | 非线性显著提升零样本能力 |
| CLIP text encoder | **0.898** | - | 最优文本编码器 |
| OpenCLIP | 0.860 | - | 稍低 |
| BLIP2 | 0.564 | - | 差距明显 |

### 关键发现
- **Model Tree 内学习 vs 跨 Tree**：同 Tree 内线性分类器可达 0.844，跨 Tree 仅 0.502（接近随机），差距巨大。
- **负迁移现象**：添加其他 Tree 的数据会降低当前 Tree 的性能，证明 MoE 设计的必要性。
- **非线性对零样本至关重要**：ReLU 使零样本准确率从 56.4% 跃升至 89.8%，但对 in-distribution 只有 2% 的提升。
- **Hugging Face 上 20 个 Model Tree 覆盖 50% 模型**，说明 Tree-aware 学习方案实际可行。

## 亮点与洞察
- **Model Tree 视角的 insight 非常深刻**：发现"大多数公开模型属于少数 Model Tree"这一事实，将看似不可能的跨架构权重学习问题转化为 Tree 内的简单线性问题。这个观察本身就很有价值。
- **ProbeX 的 Tucker 分解设计**：用理论（Proposition 1-2）证明 probing 与 dense expert 等价，再通过矩阵分解压缩参数，兼具理论基础和实用性。
- **零样本模型分类**：首次将模型权重嵌入到 CLIP 的语义空间中，开创了"用文本搜索模型"的新范式，对 Hugging Face 等模型仓库的搜索功能有直接应用价值。

## 局限与展望
- **无法泛化到未见 Tree**：新出现的 Model Tree 需要重新训练专家，虽然只需 10 分钟但仍非零成本。
- **判别式模型的零样本对齐效果差**：初步实验中判别式模型（ViT 分类器）不能与文本好好对齐。作者猜测 SD 的 cross-attention 层是对齐成功的关键，这限制了方法的通用性。
- **依赖 Tree 已知或可聚类**：如果模型来自未知的、不在训练集中的 Tree，路由可能失败。
- **更深 ProbeX 编码器反而过拟合**：多层编码器性能下降，暗示需要更好的正则化策略。

## 相关工作与启发
- **vs Neural Graphs [Kofinas et al.]**: Neural Graphs 试图用图神经网络处理模型权重的排列不变性，但无法扩展到 ViT 等大模型。ProbeX 通过 probing 绕过排列问题，且计算量极低。
- **vs StatNN [Unterthiner et al.]**: StatNN 提取权重的简单统计量（均值、方差、分位数），丢失了大量结构信息。ProbeX 通过可学习的 probe 向量主动探索权重空间的结构。
- **对模型仓库管理的启发**：ProbeX 可直接用于 Hugging Face 等平台，自动为缺少文档的模型生成"训练内容标签"，甚至实现类似 CLIP 的"文本搜索模型"功能。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Model Tree insight + 首个单层 probing + 首个零样本模型分类，多个"首次"
- 实验充分度: ⭐⭐⭐⭐ 14000 模型的大规模实验，判别式+生成式两类模型，但缺少更多真实模型仓库的验证
- 写作质量: ⭐⭐⭐⭐ 动机实验层层递进，理论与实验结合紧密
- 价值: ⭐⭐⭐⭐⭐ 开创了权重空间学习的新范式，对模型管理和搜索有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](../../ACL2026/interpretability/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](../../CVPR2026/interpretability/draft_and_refine_with_visual_experts.md)
- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)

</div>

<!-- RELATED:END -->
