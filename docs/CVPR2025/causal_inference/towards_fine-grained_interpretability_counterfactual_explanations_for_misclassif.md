---
title: >-
  [论文解读] FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition
description: >-
  [CVPR 2025][因果推理][反事实解释] 提出 FG-VCE（Fine-Grained Visual Contrastive Explanation）框架，通过 Shapley 值计算特征点贡献度、显著性分区模块隔离局部特征、以及迭代反事实生成策略，首次实现了对象级和部件级的细粒度反事实解释，揭示模型误分类的具体原因——"哪些细粒度特征导致了错误"以及"哪些局部区域主导了预测改变"。
tags:
  - "CVPR 2025"
  - "因果推理"
  - "反事实解释"
  - "细粒度分类"
  - "Shapley值"
  - "显著性分区"
  - "误分类分析"
---

# FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition

**会议**: CVPR 2025  
**arXiv**: [2511.07974](https://arxiv.org/abs/2511.07974)  
**代码**: 无  
**领域**: 因果推理 / 可解释性  
**关键词**: 反事实解释, 细粒度分类, Shapley值, 显著性分区, 误分类分析

## 一句话总结
提出 FG-VCE（Fine-Grained Visual Contrastive Explanation）框架，通过 Shapley 值计算特征点贡献度、显著性分区模块隔离局部特征、以及迭代反事实生成策略，首次实现了对象级和部件级的细粒度反事实解释，揭示模型误分类的具体原因——"哪些细粒度特征导致了错误"以及"哪些局部区域主导了预测改变"。

## 研究背景与动机

**领域现状**：归因解释方法（如 GradCAM、Saliency Map）通过标注图像中对预测最重要的区域来提供模型解释。这些方法在粗粒度任务（区分猫狗）中足够，但在细粒度任务（区分鸟类 species）中价值有限——因为它们通常只高亮整个物体而不区分有意义的局部区域。

**现有痛点**：(1) 传统归因方法无论模型预测正确还是错误，高亮的区域往往看起来一样——如 Fig 1 所示，正确预测"信天翁"和错误预测"军舰鸟"时高亮区域几乎无差别；(2) 现有反事实解释方法（如 CCE）通过不可感知的对抗扰动（PGD attack）改变预测，但生成的解释缺乏直觉意义——扰动在像素级不可见，归因图几乎不变；(3) 缺乏部件级解释——不知道是"头部"、"翅膀"还是"腿部"的特征导致了误分类。

**核心矛盾**：细粒度分类中，相似类别之间的区分依赖极其细微的局部特征差异（如鸟喙形状、羽毛纹理），但归因方法的分辨率不足以捕获这些差异。

**本文目标**：生成细粒度的反事实解释，回答两个问题：(1) 哪些细粒度特征导致了模型误分类？(2) 哪些局部特征主导了反事实调整？

**切入角度**：利用正确分类样本和误分类样本之间的对比——找到误分类样本中的哪些局部特征如果替换为正确分类样本的对应特征，就能改变预测。

**核心 idea**：非生成式反事实方法——通过在误分类样本和正确分类参考样本之间进行特征级匹配和迭代替换，生成语义一致的反事实。用近似 Shapley 值量化每个特征点的贡献，引入 Saliency Partition 将特征图分解为局部独立区域。

## 方法详解

### 整体框架
FG-VCE 包含三个阶段：(1) 特征提取：对误分类样本和正确分类参考样本提取深层特征表示；(2) 显著性分区：使用近似 Shapley 值计算特征点贡献度，通过空间局部化核将特征图分解为具有区域特异性相关性的独立区域；(3) 反事实生成与解释：迭代选择误分类样本中贡献度最高的特征区域，在参考样本中找到语义最相似的区域进行替换，直到预测改变。

### 关键设计

1. **近似 Shapley 值贡献度量**:

    - 功能：量化每个特征点对模型预测的边际贡献
    - 核心思路：对特征图中的每个空间位置 $(i,j)$，计算其 Shapley 值——即该位置存在/不存在对预测概率的边际影响的加权平均。由于精确 Shapley 值计算量指数级，使用 Monte Carlo 采样近似：随机选择特征子集 $S$，计算 $v(S \cup \{(i,j)\}) - v(S)$，对多次采样取平均
    - 设计动机：传统归因方法（梯度、CAM）反映的是"特征重要性"而非"特征贡献"，Shapley 值作为博弈论中唯一满足公平性公理的分配方案，能更准确地量化每个区域的独立贡献

2. **显著性分区模块（Saliency Partition）**:

    - 功能：将全局特征图分解为局部独立的显著性区域
    - 核心思路：在 Shapley 值计算中引入空间局部化核——对每个位置 $(i,j)$，只考虑其空间邻域内的特征交互，而忽略远距离特征的影响。具体地，使用固定大小的空间核（如 $7 \times 7$）约束 Shapley 值的计算范围，使得每个位置的贡献值只反映局部上下文信息
    - 设计动机：标准特征图中相邻位置的表示高度耦合（因卷积感受野重叠），直接用 Shapley 值会将耦合区域的贡献混为一体。局部核打破了这种耦合，产生更细粒度的区域分割

3. **迭代反事实生成**:

    - 功能：逐步构造语义一致的反事实样本
    - 核心思路：维护一组"高贡献特征候选集"——从正确分类的参考样本中提取的关键特征区域。每次迭代中：(i) 选择误分类样本中 Shapley 值最高的特征区域；(ii) 在候选集中找到语义余弦相似度最高的匹配区域；(iii) 替换该区域的特征；(iv) 检查预测是否改变。直到预测改变为止，被替换的所有区域就是"导致误分类的关键特征"
    - 设计动机：一次性替换所有特征太粗糙且不可解释；迭代替换能精确定位"最少修改哪些区域就能纠正预测"，提供最小化的反事实解释

### 损失函数 / 训练策略
FG-VCE 本身不需要训练——它是一个后验解释方法（post-hoc explanation），基于已训练好的分类模型生成解释。唯一需要的是分类模型的特征提取器和预测头。

## 实验关键数据

### 主实验：CUB-200-2011 和 Stanford Dogs

| 方法 | CUB-200 Del-Ins ↑ | CUB-200 SCIC ↑ | Dogs Del-Ins ↑ | Dogs SCIC ↑ | 语义一致性 ↑ |
|---|---|---|---|---|---|
| GradCAM | 0.312 | 0.425 | 0.289 | 0.401 | 0.621 |
| CCE (PGD-based) | 0.358 | 0.478 | 0.321 | 0.443 | 0.539 |
| ACE (Autoencoder) | 0.401 | 0.512 | 0.367 | 0.489 | 0.684 |
| SCOUT | 0.423 | 0.534 | 0.385 | 0.507 | 0.702 |
| **FG-VCE (Ours)** | **0.487** | **0.602** | **0.441** | **0.571** | **0.758** |

### 细粒度区域定位质量

| 方法 | Part IoU ↑ | 部件覆盖率 ↑ | 误导区域 ↓ |
|---|---|---|---|
| GradCAM | 0.184 | 42.3% | 38.7% |
| CCE | 0.213 | 48.1% | 35.2% |
| **FG-VCE** | **0.342** | **71.8%** | **18.3%** |

### 消融实验

| 配置 | Del-Ins ↑ | SCIC ↑ | 说明 |
|---|---|---|---|
| FG-VCE (full) | 0.487 | 0.602 | 完整方法 |
| w/o Saliency Partition | 0.421 | 0.538 | 退化到全局 Shapley，失去局部精度 |
| w/o Shapley (用梯度) | 0.378 | 0.501 | 梯度归因不够准确 |
| w/o 迭代替换 (一步) | 0.398 | 0.489 | 一次性替换过于粗糙 |
| 替换为随机区域 | 0.312 | 0.412 | 验证语义匹配的必要性 |

### 关键发现
- **FG-VCE 在所有指标上全面领先**：CUB-200 上 Del-Ins 从 0.423 提升到 0.487（+15.1%），SCIC 从 0.534 到 0.602（+12.7%）
- **部件级定位质量大幅提升**：Part IoU 从 0.213 提升到 0.342（+60.6%），说明 FG-VCE 确实能定位细粒度部件（头、翅膀等）
- **Saliency Partition 贡献最大**：去掉后 Del-Ins 从 0.487 降到 0.421，证明局部核分解是关键创新
- **语义一致性最高（0.758）**：反事实不是通过不可感知的对抗扰动而是通过语义匹配替换生成的，保持了视觉可理解性

## 亮点与洞察
- **首次系统化细粒度反事实解释**：开创了"哪个部件导致误分类"这一研究方向。之前工作要么只做对象级解释，要么使用对抗扰动（不可解释）
- **非生成式方法的优势**：不依赖 GAN/Diffusion 等生成模型来创造反事实，而是在真实样本之间做特征移植，避免了生成模型引入的 artifact
- **Shapley 值 + 空间局部化**：博弈论的公平分配方案与空间先验的结合非常巧妙——Shapley 保证了贡献量化的理论合理性，空间核保证了区域独立性
- **可迁移性强**：框架不依赖特定分类器架构，可直接用于任何 CNN 或 ViT 模型的后验解释

## 局限与展望
- Shapley 值的 Monte Carlo 近似需要多次前向推理，计算开销较大（每次解释约数十次推理）
- 空间局部化核尺寸固定，可能无法适应不同大小的视觉部件（翅膀 vs 眼睛）
- 参考样本集的质量影响反事实质量——如果参考集中没有正确类别的高质量样本，反事实可能不理想
- 仅在二维图像分类上验证，目标检测、分割等任务的细粒度解释未探索

## 相关工作与启发
- **vs GradCAM**：只提供热力图级解释，无法区分正确/错误预测的不同区域；FG-VCE 提供部件级反事实
- **vs CCE**：使用 PGD 对抗攻击生成反事实，扰动不可感知也不可解释；FG-VCE 的特征替换可视化可解释
- **vs ACE/SCOUT**：基于概念/原型的解释，但缺乏细粒度部件定位；FG-VCE 通过 Saliency Partition 精确到部件

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次细粒度反事实解释，Shapley+空间核组合新颖
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、完整消融，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，框架图直观
- 价值: ⭐⭐⭐⭐ 开辟了细粒度可解释性新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](../../ICLR2026/causal_inference/counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](../../NeurIPS2025/causal_inference/performative_validity_of_recourse_explanations.md)
- [\[ICLR 2026\] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](../../ICLR2026/causal_inference/synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)

</div>

<!-- RELATED:END -->
