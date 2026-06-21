---
title: >-
  [论文解读] Exploring Cross-Modal Flows for Few-Shot Learning
description: >-
  [ICLR 2026][多模态VLM][Few-Shot Learning] 把图像特征向文本特征对齐这件事，从所有 PEFT 方法的"一步到位"重构成 Flow Matching 的"多步迭代修正"，用一个即插即用的速度场（velocity field）把难数据集上纠缠的跨模态分布逐步拉齐，从而显著提升少样本分类。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "Few-Shot Learning"
  - "CLIP"
  - "PEFT"
  - "Flow Matching"
  - "跨模态"
  - "Velocity Field"
---

# Exploring Cross-Modal Flows for Few-Shot Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ks6Gg8nd0y](https://openreview.net/forum?id=ks6Gg8nd0y)  
**代码**: [HKUST-LongGroup/FMA](https://github.com/HKUST-LongGroup/FMA)  
**领域**: 多模态视觉语言模型 / 少样本学习  
**关键词**: Few-Shot Learning, CLIP, PEFT, Flow Matching, Cross-Modal Alignment, Velocity Field  

## 一句话总结
把图像特征向文本特征对齐这件事，从所有 PEFT 方法的"一步到位"重构成 Flow Matching 的"多步迭代修正"，用一个即插即用的速度场（velocity field）把难数据集上纠缠的跨模态分布逐步拉齐，从而显著提升少样本分类。

## 研究背景与动机
**领域现状**：CLIP、ALIGN 这类预训练 VLM 通过对比学习把图像和文本编码到共享空间，做到了不错的零样本对齐。但内在的模态复杂性使它们无法在所有场景下完美对齐，因此少样本下游任务通常要再微调。由于全量微调代价高，社区发展出三大类参数高效微调（PEFT）方法：Prompt Tuning（CoOp/CoCoOp，移动文本特征）、Adapter-based（CLIP-Adapter，在图像编码器后接 MLP 移动图像特征）、LoRA-based（CLIP-LoRA，在双编码器内插低秩矩阵同时移动图文特征）。

**现有痛点**：作者首次指出，无论哪一类 PEFT，本质上都是**一步调整（one-step adjustment）**——推理时只用一次前向传播把输入特征直接映射到目标位置。文中用线性探针（linear probing）作参照基线剥离"数据带来的增益"，并按 CLIP RN50 零样本性能定义"数据集难度"，发现 PEFT 相对线性探针的优势在简单集（OxfordPets）明显，但在难集（FGVCAircraft）几乎消失。

**核心矛盾**：难数据集上图文分布高度纠缠，需要复杂的非线性变换才能对齐，而一步映射根本建模不了这种复杂变换。

**本文目标**：让跨模态对齐变成可多步迭代的过程，使每一步只需预测一个局部更新，从而把难分布逐步"修正"到位。

**核心 idea**：**[多步对齐]** 借鉴主要用于图像生成的 Flow Matching（FM）理论——它学一个速度场，把源分布沿多步轨迹搬运到目标分布。作者把**所有图像特征当源分布、所有类名文本特征当目标分布**，训练一个速度场实现"图像特征 → 对应文本特征"的多步搬运，即完成分类。但直接套用 FM 有两个坑：(1) 训练出的速度场不保证类别对应，可能把某类图像搬到别类文本；(2) 生成式 FM 的推理目标是"抵达目标分布"，而分类只需"离正确类比离错误类更近"，目标不一致。FMA 用三个设计逐一化解。

## 方法详解

### 整体框架
FMA（Flow Matching Alignment）分三步：先用预训练 VLM（如 CLIP 零样本，或 CoOp/CLIP-LoRA 等任意 PEFT）把图像编码成源特征 $x_0$、把"a photo of {class}"模板编码成目标文本特征 $z$；再在共享空间里训练一个速度场 $u_\theta^t$，学会把图像特征沿直线轨迹搬向其对应文本特征；推理时用一个"早停求解器"迭代搬运测试图像特征，在它对分类已足够判别时就停下，用中间特征算与各类文本的余弦相似度做预测。整个速度场只吃共享空间里的两组特征、与具体特征提取方法无关，因此是**即插即用**的多步修正模块。

```mermaid
flowchart LR
    A[图像 I] -->|图像编码器| X0[源特征 x0]
    B["文本 a photo of {class}"] -->|文本编码器| Z[目标文本特征 z]
    X0 --> C[Coupling Enforcement<br/>只配对真实类文本]
    Z --> C
    C --> D[Noise Augmentation<br/>注入时变高斯噪声]
    D --> E["训练速度场 u_theta(x_t)<br/>L_FM 最小二乘回归"]
    E --> F[Early Stopping Solver<br/>迭代 M 步后停]
    F --> G[中间特征 x_T̂ 算余弦相似度→分类]
```

### 关键设计

**1. Coupling Enforcement（耦合强制）：让速度场学会"分类方向"而非"平均方向"。** 标准 FM 训练随机配对图像特征 $x_0$ 和文本特征 $z$，沿直线插值 $x_t = tz + (1-t)x_0$，回归条件速度 $v_t(x_t|z) = z - x_0$，损失为 $L_{FM}(\theta) = \mathbb{E}_{x_t,t}[\|u_\theta^t(x_t) - (z-x_0)\|^2]$。问题在于学到的边际速度 $v_t(x_t)$ 是对所有可能文本 $z$ 的期望积分，会把图像特征推向**所有文本的平均**，导致错分。作者的做法是：给定图像特征 $x_0$，**只采样它真实类别对应的文本特征 $z_c$ 来配对**。由于小数据集在高维流形上稀疏、轨迹近似互不相交，对给定 $x_t$ 只存在唯一的 $z$，于是边际速度恰好退化为条件速度 $v_t(x_t) = v_t(x_t|z_c)$（论文 Proposition 1）。这等于偷偷把每个图像特征沿"指向正确类文本"的方向搬运，理论上保证 $x_1 = z_c$（Proposition 2），把对齐变成了天然的分类器。

**2. Noise Augmentation（噪声增强）：用 Schrödinger 桥式扰动救活被耦合掏空的训练数据。** 耦合强制虽优雅，却带来数据稀缺——每张图只配一个目标文本，可用训练对从随机配对的 $N^2$ 量级骤降到 $\frac{1}{N}$，速度场定义域大片采样不到，给不出可靠指引。为此作者向中间特征 $x_t$ 注入**时变高斯噪声**，得到增强特征 $\hat{x}_t \sim \mathcal{N}(\hat{x}_t \mid x_t,\, t(1-t)\sigma^2(x_t))$，其中 $\sigma^2(x_t)$ 是 $x_t$ 各维的标准差。这个 $t(1-t)$ 形状的方差在轨迹两端为零、中间最大，灵感来自 Schrödinger 桥——给条件概率路径加非零方差能填满轨迹邻域，避免分布塌缩到低维流形（参考 Score-based 思路），从而学到更准、更鲁棒的速度估计。增强后的真值方向相应改为 $v_t(\hat{x}_t|z_c) = \frac{z_c - \hat{x}_t}{1-t}$。

**3. Early Stopping Solver（早停求解器）：堵住"为对齐而对齐"在分类上的反噬。** 推理时香草 FM 用 ODE 求解器（如欧拉法）迭代 $x_{t+h} = x_t + h\cdot u_\theta^t(x_t)$ 走完整个 $[0,1]$ 得到 $x_1$ 再分类。但作者观测到一个关键不一致：随 $t\to 1$，中间特征到目标文本的距离确实单调变小（越来越"像"文本），可分类准确率却**先升后降**——因为速度场不可能训得完美，后期反而会把部分特征推向错误类的文本（Figure 5）。既然分类只需特征"足够判别"而非"抵达目标分布"，作者就固定步长 $h$、只走常数步 $M$，在 $\hat{T} = h\cdot M$ 处取中间特征 $x_{\hat{T}}$ 做分类。$M$ 通过验证集挑性能最高的值确定。这一停既省推理时间，又规避了后期搬错方向的风险。文中也指出"为每个样本找自适应的最优 $t$"是更优但留给未来的方向。

## 实验关键数据

### 主实验（11 数据集，基于 CLIP-LoRA + ViT-B/16）
FMA 在 CLIP-LoRA 提取的特征上再训一个速度场（默认 6 个 ResNet block、步长 $h=0.1$），不引入额外数据。在 1/4/16-shot 上对比 CoOp、CoCoOp、CLIP-Adapter、Tip-Adapter、PLOT++、KgCoOp、ProGrad、CLIP-LoRA 等 8 个 SOTA，FMA 在多数数据集取得最佳，且**难数据集上增益远大于简单数据集**，印证"跨模态分布越复杂越需要多步修正"的核心论点。

### 即插即用泛化（5 种 backbone × 难/易集平均，部分摘录）

| Backbone | D(Adapt) | +FMA | E(Adapt) | +FMA | D(Harmonic) | +FMA |
|---|---|---|---|---|---|---|
| CLIP | 48.9 | **68.9** | 78.8 | **87.6** | 48.9 | **57.5** |
| CoOp | 71.4 | **74.0** | 87.1 | **87.9** | 54.8 | **55.6** |
| CoCoOp | 64.1 | **68.5** | 85.0 | **87.3** | 55.9 | **58.2** |
| CLIP-Adapter | 62.4 | **67.9** | 86.6 | **87.4** | 53.6 | **56.0** |
| CLIP-LoRA | 76.1 | **77.8** | 88.2 | **88.8** | 57.1 | **58.4** |

挂在任意 backbone 上都涨，零样本 CLIP 上难集 adaptation 从 48.9→68.9（+20.0）最夸张；同时 generalization（跨数据集迁移）列基本不掉，说明 FMA 没牺牲泛化能力。

### 关键发现
- **难数据集是主战场**：难集（Aircraft/EuroSAT/DTD/SUN/Cars）的相对增益普遍大于易集，多步修正对纠缠分布的价值被实证。
- **早停是必需而非可选**：准确率随推理步数先升后降的曲线（Figure 5）直接证明，跑满 FM 全程会把特征搬向错误文本，ESS 的早停同时换来速度和精度。
- **方法无关性**：从零样本 CLIP 到各类 PEFT 都能叠加增益，验证了"只需共享空间两组特征"的即插即用设计。

## 亮点与洞察
- **一针见血的统一视角**：把 Prompt/Adapter/LoRA 三类看似不同的 PEFT 统一抽象为"一步特征移动"，并据此定位它们在难数据集上的失效根因，这个 framing 本身就很有解释力。
- **跨界迁移用得巧**：Flow Matching 几乎只在生成领域用，作者把"多步比一步容易学"的核心直觉迁到判别式分类，并诚实地揪出生成与分类目标不一致这个隐患，再用早停补上。
- **理论与策略咬合**：Coupling Enforcement 不只是工程技巧，而是通过定义 Dirac 形式的条件概率路径让边际速度=条件速度，从数学上保证了"搬到正确类"，两条 Proposition 把直觉落到了实处。

## 局限与展望
- **早停步数 $M$ 靠验证集网格搜索**，是全局常数而非样本自适应；作者也承认"为每个样本找最优 $t$"才是更优解，但留作未来工作。
- **耦合强制的非交叉假设依赖小数据**：论证"轨迹互不相交→边际=条件"建立在高维稀疏、数据量小的前提上，数据量增大或类别极多时这一假设是否仍成立存疑。
- **额外训练成本**：FMA 是在已训好的 PEFT 特征之上再训一个速度网络（6 个 ResNet block），相比纯一步 PEFT 多了一段训练与多步推理开销，论文虽用早停缓解但仍非零成本。

## 相关工作与启发
- **少样本 VLM 适配**：与 CoOp/CoCoOp（Prompt Tuning）、CLIP-Adapter/Tip-Adapter（Adapter）、CLIP-LoRA（LoRA）并列，FMA 不替代它们而是叠加其上做"多步精修"。
- **Flow Matching / Diffusion**：承接 Rectified Flow、Lipman 等 FM 理论与 Score-SDE，把 noise-to-data 的生成范式拓展到 feature-to-feature 的判别范式；噪声增强借鉴 Schrödinger 桥。
- **启发**：本文提示一条通用思路——凡是"把 A 表征对齐到 B 表征"的任务（检索、跨域适配、表征对齐），当一步映射不够时，都可考虑用可学速度场做多步迭代修正，并针对下游目标（而非生成目标）设计专门的早停/截断策略。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次把 PEFT 统一为"一步方法"并首次将 Flow Matching 引入少样本判别任务，视角与方法都新。
- **实验充分度**: ⭐⭐⭐⭐ 11 数据集 × 多 shot × 5 backbone 的即插即用验证扎实，难/易分组对比有说服力；若有更多推理开销与自适应早停的分析会更完整。
- **写作质量**: ⭐⭐⭐⭐ 动机递进清晰（一步→失效→多步→两坑→三设计），图示与两条 Proposition 配合到位。
- **价值**: ⭐⭐⭐⭐ 即插即用、方法无关，对难数据集增益明显，对跨模态对齐的范式有启发意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](../../CVPR2026/multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[ICLR 2026\] Preserve and Sculpt: Manifold-Aligned Fine-tuning of Vision-Language Models for Few-Shot Learning](preserve_and_sculpt_manifold-aligned_fine-tuning_of_vision-language_models_for_f.md)
- [\[ICLR 2026\] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f.md)
- [\[ICLR 2026\] FlowBind: Efficient Any-to-Any Generation with Bidirectional Flows](flowbind_efficient_any-to-any_generation_with_bidirectional_flows.md)

</div>

<!-- RELATED:END -->
